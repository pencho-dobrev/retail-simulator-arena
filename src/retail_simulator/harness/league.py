"""Phase 1.5 self-play tooling: frozen-policy pool primitives + opponent distributions.

* the **pool data structure** (Slice A T-A1): the :class:`PoolMember` value type,
  the :class:`FrozenPolicyPool` ordered collection (with atomic on-disk manifest
  persistence, F4-A), :func:`make_seeded_pool` (the F5-A scripted-archetype seed
  builder), and the named-league registry the CLI (Slice B T-A4) routes
  ``--league=<name>`` through;
* the **opponent-distribution layer** (Slice A T-A2): the
  :class:`OpponentDistribution` Protocol + two concrete shapes
  (:class:`UniformOpponentDistribution` — F2-A RECOMMENDED — and
  :class:`LatestNOpponentDistribution` — F2-B alternative), mirroring
  :mod:`harness.curriculum`'s ``ScenarioDistribution`` / ``UniformScenarioDistribution``
  pattern.

**F3-A composition contract (recorded for T-A3).** The league wraps OVER the 1.4
:class:`CurriculumEnvWrapper` (``LeagueEnvWrapper(CurriculumEnvWrapper(env))``): the
outer wrapper draws an opponent, the inner draws a scenario, the two draws are
INDEPENDENT (each uses its own harness ``np.random.Generator``). The pool layer
shipped here is composition-agnostic — it stores members; the wrapper handles
sampling + seating.

**Lazy SB3 imports.** ``stable_baselines3`` is heavy (~1s import). It is imported
LAZILY inside :meth:`FrozenPolicyPool.materialize`'s checkpoint branch only — never
at module load. ``import retail_simulator.harness.league`` does NOT pull SB3.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np

from retail_simulator.harness.geo_archetypes import (
    GEO_ARCHETYPE_LABELS,
    geo_archetypes as _geo_archetypes,
)
from retail_simulator.harness.spike_archetypes import (
    AGGRESSOR,
    LEAN,
    PATIENT,
    calibrated_archetypes,
)

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.core.config import CoreConfig, RegionConfig
    from retail_simulator.harness.curriculum import ScenarioDistribution

logger = logging.getLogger(__name__)

# Manifest schema version: bump only on a breaking on-disk format change.
# H4 (the "archetype" PoolMember kind) deliberately did NOT bump this: `Frozen
# PolicyPool.load` gates on EXACT equality against this constant, so a bump
# would reject every pre-existing on-disk manifest (version=1, no "archetype"
# members) outright — a breaking change H4 does not need. H4 only WIDENED an
# already-validated enum (`PoolMember.kind`); the manifest SHAPE (the set of
# JSON keys) is unchanged. An OLDER reader loading a manifest that contains an
# "archetype" member still fails loud — `PoolMember.from_json_dict`'s ``kind
# not in (...)`` check rejects the unrecognized string with a clear
# ``ValueError`` — so there is no silent misinterpretation risk either way.
MANIFEST_VERSION: int = 1

# RLB-8a follow-up (senior-review M1+M5): the version a manifest carrying real
# arena-seating provenance (`home_regions`/`arena_region_table_hash`/
# `arena_lock_knobs`, all non-`None`) is stamped with — see
# `FrozenPolicyPool.save`/`.load`. Earlier RLB-8a shipped this pair
# UNCONDITIONALLY (`null` when unset, `MANIFEST_VERSION` never bumped),
# reasoning the same way H4 did above — but unlike H4's widened `kind` enum,
# that shape is NOT a no-op for a pool with no seating: it grew EVERY default
# pool's on-disk manifest for two keys no reader benefits from (measured:
# 633 -> 692 bytes), breaking byte-identity with every pre-RLB-8a pool. The
# fix: the three keys are emitted ONLY when there is a real seating to record
# (see `save`), so a pool with none keeps `MANIFEST_VERSION` and stays
# byte-identical. Bumping the version exactly when the keys are present means
# pre-RLB-8a code loading a manifest that DOES carry seating now hits
# `load`'s version check and fails LOUD, instead of silently discarding the
# seating it doesn't know to look for.
MANIFEST_VERSION_ARENA_SEATING: int = 2

# H5 senior-review MAJOR fix round: the version a manifest carrying a real,
# notable reward mode (`reward_mode`, non-`None` -- see `FrozenPolicyPool.
# record_reward_mode`/`.save`/`.load`) is stamped with. Same reasoning as
# `MANIFEST_VERSION_ARENA_SEATING` above: the key is emitted ONLY when there
# is something notable to record (`train_league` decides "profit" -- every
# calibrated/arena/arena-balanced config's own default -- is NOT notable, so
# a pool that never touches `--reward` keeps `MANIFEST_VERSION`/
# `MANIFEST_VERSION_ARENA_SEATING` and stays byte-identical). Bumping the
# version exactly when the key is present means pre-H5 code loading a
# manifest that DOES carry a recorded reward mode fails LOUD at `load`'s
# version check, instead of silently discarding the one thing that lets
# `--resume` continue scoring a weighted-trained pool as "weighted" without
# retyping `--reward` (the senior-review MAJOR this fixes: before this, a
# resumed run derived `training.reward_mode` from `--reward`'s own CLI
# default rather than from what the pool actually trained under).
MANIFEST_VERSION_REWARD_MODE: int = 3

# Win-rate matrix on-disk format version (Slice C T-C1-WIN). The on-disk
# ``<pool_dir>/win_rate_matrix.json`` carries this value so a future format
# change can be detected at :meth:`WinRateMatrix.load` time and surfaced as a
# clear ``ValueError`` rather than a silent shape mismatch. Mirrors the
# ``MANIFEST_VERSION`` discipline for the pool manifest.
WIN_RATE_MATRIX_VERSION: int = 1

# The Phase 0.5 scripted archetype names (the F5-A pool seed roster). A pool
# member with ``kind == "scripted"`` must carry one of these names — they map
# onto :func:`retail_simulator.core.world._make_npc_policy`'s allowlist
# {"discounter", "premium", "balanced"} (the league does NOT seat "wandering",
# which is a non-strategic exploration substrate, not a league opponent).
DEFAULT_POOL_SEED_ARCHETYPES: tuple[str, ...] = ("discounter", "premium", "balanced")

POOL_SPIKE_ARCHETYPE_LABELS: tuple[str, ...] = (LEAN, AGGRESSOR, PATIENT)

# RLB-8b: the public allowlist widens from the spike-only three to the UNION of
# the spike family above and the four closed-loop geo archetypes
# (:data:`~retail_simulator.harness.geo_archetypes.GEO_ARCHETYPE_LABELS`,
# imported rather than re-listed so the two can't drift). Every PRE-EXISTING
# caller that means "the spike three specifically" (e.g. ``tests/harness/
# test_league_archetype_members.py``, entirely H4-scoped) must import
# :data:`POOL_SPIKE_ARCHETYPE_LABELS` instead of this union.
POOL_ARCHETYPE_LABELS: tuple[str, ...] = POOL_SPIKE_ARCHETYPE_LABELS + GEO_ARCHETYPE_LABELS

# The Phase 1.5 basic league name (mirrors PHASE_1_4_BASIC_CURRICULUM_NAME). The
# CLI's ``--league=<name>`` flag (Slice B T-A4) routes this string onto
# :func:`build_league`.
DEFAULT_LEAGUE_NAME: str = "phase_1_5_basic_league"

# Sentinel used by scripted pool members to mark "this member is not a learner
# checkpoint" (learner checkpoints carry round_index >= 0). Kept as a module
# constant so a regression that adds a "0th round scripted" entry would have to
# explicitly opt out of the sentinel.
_SCRIPTED_ROUND_INDEX: int = -1

# The PoolMember.kind allowlist. "archetype" (H4) is a FIXED calibrated Phase 4
# spike posture (see POOL_ARCHETYPE_LABELS) — distinct from "scripted", which
# names a CORE NPC archetype (the world's npc_archetypes seam:
# "discounter"/"premium"/"balanced"). Both are non-learner, fixed opponents;
# only "checkpoint" is ever a learner snapshot.
PoolMemberKind = Literal["scripted", "checkpoint", "archetype"]


@dataclass(frozen=True)
class PoolMember:
    """ """

    member_id: str
    kind: PoolMemberKind
    archetype_name: str
    checkpoint_path: Path | None
    round_index: int
    policy_kind: Literal["mlp", "lstm"] = "mlp"

    def __post_init__(self) -> None:
        if not self.member_id:
            raise ValueError("PoolMember.member_id must be non-empty")
        # Phase 1.6 Slice A T-A3: validate policy_kind at the boundary. Both
        # "mlp" (default — byte-identical to Phase 0–1.5 under F-COMPAT=A) and
        # "lstm" (the new RecurrentPPO opt-in) are legal; anything else is a
        # programming error.
        if self.policy_kind not in ("mlp", "lstm"):
            raise ValueError(
                f"PoolMember {self.member_id!r} policy_kind must be 'mlp' or 'lstm'; "
                f"got {self.policy_kind!r}"
            )
        if self.kind == "scripted":
            if self.checkpoint_path is not None:
                raise ValueError(
                    f"scripted PoolMember {self.member_id!r} must have checkpoint_path=None; "
                    f"got {self.checkpoint_path!r}"
                )
            if self.archetype_name not in DEFAULT_POOL_SEED_ARCHETYPES:
                raise ValueError(
                    f"scripted PoolMember {self.member_id!r} archetype_name "
                    f"{self.archetype_name!r} not in scripted allowlist "
                    f"{DEFAULT_POOL_SEED_ARCHETYPES}"
                )
            if self.round_index != _SCRIPTED_ROUND_INDEX:
                raise ValueError(
                    f"scripted PoolMember {self.member_id!r} must use the "
                    f"round_index={_SCRIPTED_ROUND_INDEX} sentinel; got {self.round_index}"
                )
        elif self.kind == "checkpoint":
            if self.checkpoint_path is None:
                raise ValueError(
                    f"checkpoint PoolMember {self.member_id!r} must carry a checkpoint_path"
                )
            if not self.archetype_name.startswith("frozen_policy:"):
                raise ValueError(
                    f"checkpoint PoolMember {self.member_id!r} archetype_name must start "
                    f"with 'frozen_policy:'; got {self.archetype_name!r}"
                )
            if self.round_index < 0:
                raise ValueError(
                    f"checkpoint PoolMember {self.member_id!r} round_index must be >= 0; "
                    f"got {self.round_index}"
                )
        elif self.kind == "archetype":
            if self.checkpoint_path is not None:
                raise ValueError(
                    f"archetype PoolMember {self.member_id!r} must have checkpoint_path=None; "
                    f"got {self.checkpoint_path!r}"
                )
            if self.archetype_name not in POOL_ARCHETYPE_LABELS:
                raise ValueError(
                    f"archetype PoolMember {self.member_id!r} archetype_name "
                    f"{self.archetype_name!r} not in archetype allowlist "
                    f"{POOL_ARCHETYPE_LABELS}"
                )
            if self.round_index != _SCRIPTED_ROUND_INDEX:
                raise ValueError(
                    f"archetype PoolMember {self.member_id!r} must use the "
                    f"round_index={_SCRIPTED_ROUND_INDEX} sentinel; got {self.round_index}"
                )
        else:
            raise ValueError(
                f"PoolMember {self.member_id!r} kind must be 'scripted', 'checkpoint', "
                f"or 'archetype'; got {self.kind!r}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for inclusion in the pool manifest.

        Paths serialize as POSIX strings so the manifest is portable across OSes
        (the league's on-disk artifact is the same shape whether produced on Linux
        in CI or a contributor's Windows checkout).
        """
        return {
            "member_id": self.member_id,
            "kind": self.kind,
            "archetype_name": self.archetype_name,
            "checkpoint_path": (
                self.checkpoint_path.as_posix() if self.checkpoint_path is not None else None
            ),
            "round_index": self.round_index,
            # Phase 1.6 Slice A T-A3: policy_kind is the LAST emitted key so a
            # manifest diff against a Phase 0–1.5-shaped fixture shows the
            # added field at the tail rather than reshuffling existing rows.
            "policy_kind": self.policy_kind,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> PoolMember:
        """Deserialize from a manifest dict, validating shape at the boundary.

        Unknown / missing required keys surface as a clear ``ValueError`` rather
        than a quiet ``KeyError`` from the dataclass constructor — this is the
        boundary between on-disk data and in-memory invariants.
        """
        try:
            member_id = data["member_id"]
            kind = data["kind"]
            archetype_name = data["archetype_name"]
            raw_path = data["checkpoint_path"]
            round_index = data["round_index"]
        except KeyError as exc:
            raise ValueError(
                f"PoolMember manifest entry missing required key {exc.args[0]!r}; got {data!r}"
            ) from exc
        checkpoint_path = Path(raw_path) if raw_path is not None else None
        if kind not in ("scripted", "checkpoint", "archetype"):
            raise ValueError(
                f"PoolMember kind must be 'scripted', 'checkpoint', or 'archetype'; got {kind!r}"
            )
        # Phase 1.6 Slice A T-A3 (F-COMPAT=A): policy_kind is OPTIONAL on read
        # so any pool manifest written before Phase 1.6 — which lacks this
        # field entirely — loads as the default "mlp". The constructor's
        # ``__post_init__`` validates the value; an invalid string on disk
        # surfaces there with a clear error.
        policy_kind_raw = data.get("policy_kind", "mlp")
        return cls(
            member_id=str(member_id),
            kind=kind,
            archetype_name=str(archetype_name),
            checkpoint_path=checkpoint_path,
            round_index=int(round_index),
            policy_kind=policy_kind_raw,
        )


class FrozenPolicyPool:
    """Ordered collection of :class:`PoolMember` values — the league's opponent set.

    The pool is the on-disk artifact the multi-round training loop (Slice B) grows:
    each round saves a fresh learner checkpoint AND appends a new ``PoolMember``.
    The manifest is written atomically (tmp file + ``os.replace``) so a crash
    between the SB3 ``model.save`` and the manifest update never leaves the pool in
    a partially-consistent state.

    The order of ``members()`` is the order of ``add()`` calls (insertion order);
    the manifest preserves it. ``round_index`` is independent of ordering — a
    future PFSP distribution (post-1.5 F1-C) might reorder by win-rate; the
    ordering here is just "what came first."

    ``home_regions``/``arena_region_table_hash``/``arena_lock_knobs`` (RLB-8a)
    record the seating this pool's checkpoints were actually TRAINED under —
    ``None`` (the default) for every pre-RLB-8a pool and every pool trained
    without explicit home-region seating. Set via :meth:`record_arena_seating`,
    never at construction by a caller building a fresh pool (:func:`make_seeded_pool`
    does not know the training config); read back by :meth:`load` so a ``--resume``
    run — which never calls :func:`run_league_training` again — still sees the
    seating its checkpoints were trained under.

    ``reward_mode`` (the H5 fix round) records the reward mode this
    pool's checkpoints were actually TRAINED under — ``None`` for every pre-H5 pool
    AND every pool trained under the "profit" default (only the notable "weighted"
    case is ever recorded — see :meth:`record_reward_mode`). Set via
    :meth:`record_reward_mode`, mirroring ``home_regions`` above exactly: never at
    construction, read back by :meth:`load` so ``--resume`` can recover it too.

    ``region_table`` (RLB-8b) is the IN-MEMORY-ONLY region table :meth:`materialize`
    needs to build a geo archetype member's closure (:func:`~retail_simulator.
    harness.geo_archetypes.geo_archetypes` is config-aware, unlike
    :func:`~retail_simulator.harness.spike_archetypes.calibrated_archetypes`).
    Deliberately NOT part of the persisted manifest (see :meth:`save`) — a region
    table is a large, config-derived value the manifest already has a compact
    fingerprint for (``arena_region_table_hash``), so serializing it again would
    be pure duplication. A fresh pool gets it from :func:`make_seeded_pool`
    (constructor path, below); a pool restored via :meth:`load` starts with
    ``None`` (the table cannot be recovered from disk) and a resuming caller that
    needs to materialize a geo member must reattach one via
    :meth:`attach_region_table`, which verifies it against the recorded
    ``arena_region_table_hash`` rather than trusting it blindly.
    """

    # Predict-closure cache for ``materialize``, keyed by member_id (populated
    # lazily on first access). Holds THREE different kinds of value in the SAME
    # id space: a loaded SB3 model for the checkpoint branch, an H4
    # ``calibrated_archetypes()`` closure for the spike archetype branch, and
    # (RLB-8b) a ``geo_archetypes()`` closure for the geo archetype branch —
    # all cached for the same reason (a pool reused across many resets must not
    # reload/rebuild the same predict callable N times).
    _materialize_cache: dict[str, Any]

    # The FULL {label: closure} roster from ONE ``geo_archetypes()`` call, cached
    # separately from ``_materialize_cache`` (which is keyed per-member, not
    # per-family) — populated lazily on the FIRST geo member materialized and
    # reused for every geo member after it, mirroring ``ladder.py``'s own
    # ``_materialize_predicts`` contract ("built EXACTLY ONCE per invocation,
    # never per archetype participant"). Without this, materializing every geo
    # member in a 4-posture pool would call ``geo_archetypes()`` 4 SEPARATE
    # times, each rebuilding (and discarding) the other 3 postures' closures.
    _geo_archetype_roster: dict[str, Any] | None

    def __init__(
        self,
        pool_dir: Path,
        *,
        members: tuple[PoolMember, ...] = (),
        home_regions: tuple[int, ...] | None = None,
        arena_region_table_hash: str | None = None,
        arena_lock_knobs: dict[str, float] | None = None,
        region_table: tuple[RegionConfig, ...] | None = None,
        reward_mode: str | None = None,
    ) -> None:
        """Construct an in-memory pool view rooted at ``pool_dir``.

        ``pool_dir`` is recorded but NOT created here — :func:`make_seeded_pool`
        creates the directory at the start of a fresh pool; :meth:`load` reads
        from an existing one. Constructing a bare pool with no members is the
        canonical empty-pool shape and is rejected only at the distribution layer
        (T-A2) — the data structure permits it.

        ``home_regions``/``arena_region_table_hash``/``arena_lock_knobs`` (RLB-8a)
        default to ``None`` — a caller building a fresh pool does not yet know
        what it will be trained under; :func:`run_league_training` sets them via
        :meth:`record_arena_seating` once it knows.

        ``region_table`` (RLB-8b, default ``None``) is set by :func:`make_seeded_pool`
        when it seeds one or more geo archetype members; every other caller leaves
        it ``None`` (byte-identical — it is never persisted, see the class
        docstring).

        ``reward_mode`` (the H5 fix round, default ``None``) mirrors
        ``home_regions``'s own contract: a caller building a fresh pool does not yet
        know what it will be trained under; :func:`run_league_training` sets it via
        :meth:`record_reward_mode` once it knows. ``None`` means "nothing notable to
        record" (the byte-identical default every pool had before H5) — see
        :meth:`record_reward_mode` for why "notable" is a caller decision, not this
        class's.
        """
        self._pool_dir: Path = Path(pool_dir)
        # Validate uniqueness of member_id BEFORE storing — duplicates are a
        # programming error (the pool's uniqueness invariant), not a runtime
        # condition to recover from.
        seen: set[str] = set()
        for member in members:
            if member.member_id in seen:
                raise ValueError(
                    f"FrozenPolicyPool members carry duplicate member_id "
                    f"{member.member_id!r}; member_ids must be unique within a pool"
                )
            seen.add(member.member_id)
        self._members: list[PoolMember] = list(members)
        self._materialize_cache = {}
        self._geo_archetype_roster = None
        self._home_regions: tuple[int, ...] | None = home_regions
        self._arena_region_table_hash: str | None = arena_region_table_hash
        self._arena_lock_knobs: dict[str, float] | None = arena_lock_knobs
        self._region_table: tuple[RegionConfig, ...] | None = region_table
        self._reward_mode: str | None = reward_mode

    @property
    def pool_dir(self) -> Path:
        """The on-disk root for this pool's manifest + checkpoint zips."""
        return self._pool_dir

    @property
    def home_regions(self) -> tuple[int, ...] | None:
        """The home-region seating this pool's checkpoints were trained under.

        ``None`` (the default) for every pre-RLB-8a pool and every pool trained
        without explicit seating (:func:`run_league_training`'s own default).
        Set via :meth:`record_arena_seating`.
        """
        return self._home_regions

    @property
    def arena_region_table_hash(self) -> str | None:
        """The region-table fairness hash (:func:`~retail_simulator.harness.arena.
        arena_region_table_hash`) of the config this pool's checkpoints were
        trained under — ``None`` exactly when :attr:`home_regions` is ``None``
        (set together by :meth:`record_arena_seating`)."""
        return self._arena_region_table_hash

    @property
    def arena_lock_knobs(self) -> dict[str, float] | None:
        """ """
        return self._arena_lock_knobs

    def record_arena_seating(
        self,
        *,
        home_regions: tuple[int, ...] | None,
        arena_region_table_hash: str | None,
        arena_lock_knobs: dict[str, float] | None = None,
    ) -> None:
        """Attach RLB-8a seating provenance to this pool and persist it immediately.

        :func:`run_league_training` calls this ONCE per run, right after resolving
        its effective ``home_regions``/config — even when every argument is
        ``None`` (the byte-identical default path) — so ``manifest.json`` always
        reflects the CURRENT seating (see :meth:`save`/:meth:`load`: the three
        keys are omitted entirely for a pool with no seating, present together
        for one that has it). Persisted via :meth:`save` (not just held in
        memory) so a manifest read back under ``--resume`` — which never calls
        :func:`run_league_training` again — still carries the seating the
        checkpoints were originally trained under, exactly like H4's
        ``seed_archetypes`` is recovered from the manifest under ``--resume``.
        ``arena_lock_knobs`` defaults to ``None`` so every pre-M3 caller of this
        method keeps working unchanged.
        """
        self._home_regions = home_regions
        self._arena_region_table_hash = arena_region_table_hash
        self._arena_lock_knobs = arena_lock_knobs
        self.save()

    @property
    def reward_mode(self) -> str | None:
        """The reward mode this pool's checkpoints were actually TRAINED under
        (the H5 fix round).

        ``None`` for every pre-H5 pool AND every pool trained under the
        "profit" default (:meth:`record_reward_mode`'s caller,
        :func:`train_league`, only ever records the NOTABLE "weighted" case —
        see that method's own docstring). Set via :meth:`record_reward_mode`;
        read back by :meth:`load` so a ``--resume`` run — which never calls
        :func:`train_league`/:func:`run_league_training` again — can still
        recover the objective its checkpoints were trained under, exactly
        like :attr:`home_regions` above.
        """
        return self._reward_mode

    def record_reward_mode(self, reward_mode: str | None) -> None:
        """Attach the H5 reward-mode provenance to this pool and persist it immediately.

        :func:`run_league_training` calls this ONCE per run, right after
        :meth:`record_arena_seating` — even when ``reward_mode`` is ``None`` (the
        byte-identical default path), for the SAME reason :meth:`record_arena_seating`
        always writes: a reused pool object must never keep a stale value from an
        earlier call.

        ``None`` means "nothing notable to record". Deciding what counts as notable is
        the CALLER's job (:func:`train_league` passes ``None`` when the resolved
        config's ``reward.mode == "profit"`` — every ``calibrated_spike_config``/
        ``arena_lock_config``/``arena_balanced_config`` default — and the concrete mode
        otherwise), NOT this method's or :meth:`save`'s: :func:`run_league_training`'s
        OWN standalone default (:func:`~retail_simulator.harness.parallel_gate.
        competition_config`, ``mode="weighted"``) does not share "profit" as its
        baseline, so a value-based check IN this method would wrongly stamp every bare
        (no explicit ``config``) caller's manifest with a "notable" reward_mode. This
        method and :meth:`save` stay value-agnostic instead — exactly like
        :meth:`record_arena_seating`/``home_regions`` already are.

        Senior-review MAJOR (H5 fix round): before this method existed, a resumed
        pool's ``training.reward_mode`` (``scripts/phase4_rl_confirm.py``) was derived
        from ``--reward``'s own CLI default ("profit") rather than from what the pool
        actually trained under — resuming a weighted-trained pool without retyping the
        flag silently re-scored its checkpoints on the profit objective, with no
        warning. Persisting the value here closes that gap the same way
        :meth:`record_arena_seating` already closed it for ``seed_archetypes``/
        ``home_regions``.
        """
        self._reward_mode = reward_mode
        self.save()

    @property
    def region_table(self) -> tuple[RegionConfig, ...] | None:
        """The in-memory region table (RLB-8b) available to :meth:`materialize`
        for a geo archetype member — ``None`` until set by :func:`make_seeded_pool`
        or :meth:`attach_region_table`. Never persisted (see the class docstring)."""
        return self._region_table

    def attach_region_table(self, region_table: tuple[RegionConfig, ...]) -> None:
        """Attach an in-memory region table so a RESTORED pool can materialize geo members.

        :meth:`load` cannot recover ``region_table`` itself (it is not part of the
        manifest — see the class docstring); this is the method a ``--resume``-style
        caller that knows its effective ``--env``/config WOULD call here after
        loading, to re-attach one. NOT YET WIRED to any production caller as of
        RLB-8b: neither ``scripts/phase4_rl_confirm.py``'s ``--resume`` path nor
        ``cli.py``'s ``league evaluate`` command materializes an archetype member off
        a loaded pool today, so neither calls this. Kept anyway — it is the only
        place a verify-against-the-recorded-hash check on re-attachment can live, and
        a future caller that DOES need to materialize a geo member after ``load()``
        should reach for this rather than setting ``region_table`` unchecked. NOT
        persisted — this is purely an in-memory handle for THIS process's
        :meth:`materialize` calls, mirroring :attr:`region_table`'s own contract.

        When this pool has a recorded :attr:`arena_region_table_hash` (a real
        seating was trained under), ``region_table`` is hashed the SAME way
        (:func:`~retail_simulator.harness.arena.arena_region_table_hash`, probed
        via a throwaway :class:`~retail_simulator.core.config.CoreConfig` whose
        ``demand.regions`` is ``region_table`` — that function reads ONLY that
        field) and must match, or this raises ``ValueError``: never silently
        materialize a geo archetype against a DIFFERENT region table than the one
        this pool's members were actually seeded/trained under. A pool with no
        recorded hash (never trained under an explicit seating) skips the check —
        there is nothing to verify against.
        """
        if self._arena_region_table_hash is not None:
            from dataclasses import replace  # noqa: PLC0415

            from retail_simulator.core.config import CoreConfig  # noqa: PLC0415
            from retail_simulator.harness.arena import (  # noqa: PLC0415
                arena_region_table_hash as _arena_region_table_hash,
            )

            base = CoreConfig.default()
            probe_config = replace(base, demand=replace(base.demand, regions=region_table))
            actual_hash = _arena_region_table_hash(probe_config)
            if actual_hash != self._arena_region_table_hash:
                raise ValueError(
                    "FrozenPolicyPool.attach_region_table: the given region_table "
                    f"hashes to {actual_hash!r}, but this pool's manifest recorded "
                    f"{self._arena_region_table_hash!r} — refusing to materialize "
                    "geo archetypes against a different region table than the one "
                    "this pool's members were seeded/trained under"
                )
        self._region_table = region_table

    def members(self) -> tuple[PoolMember, ...]:
        """Immutable snapshot of the current member list (insertion order)."""
        return tuple(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[PoolMember]:
        return iter(self._members)

    def add(self, member: PoolMember) -> None:
        """Append ``member`` to the pool; reject duplicate ``member_id``.

        The uniqueness check is the same invariant ``__init__`` enforces — adding
        a member that collides with an existing one is a programming error.
        """
        for existing in self._members:
            if existing.member_id == member.member_id:
                raise ValueError(
                    f"FrozenPolicyPool already contains a member with member_id "
                    f"{member.member_id!r}; member_ids must be unique within a pool"
                )
        self._members.append(member)

    def add_checkpoint(
        self,
        model: Any,
        round_index: int,
        *,
        seed: int,
        policy_kind: Literal["mlp", "lstm"] = "mlp",
    ) -> PoolMember:
        """Save ``model`` to disk, append a checkpoint ``PoolMember``, persist atomically.

        Used by the multi-round training loop (Slice B): after each round's PPO
        training completes, the trained model is frozen into the pool. The on-disk
        artifacts are:

        * ``{pool_dir}/round_{round_index}.zip`` — the SB3 zip (written by
          ``model.save``).
        * ``{pool_dir}/manifest.json`` — re-written atomically to include the new
          member.

        The ``seed`` parameter participates ONLY in the member_id naming convention
        (``"checkpoint:round_<n>_seed_<s>"``) — it is a provenance handle for the
        operator, NOT an RNG draw. The order is checkpoint-save THEN manifest-save
        so a crash between leaves an orphan zip on disk (recoverable: drop the
        zip OR re-add it) rather than a manifest pointing at a missing zip.

        Phase 1.6 Slice A T-A4 — the ``policy_kind`` kwarg is threaded onto the
        constructed :class:`PoolMember` (default ``"mlp"`` preserves byte-identity
        with every Phase 0–1.5 manifest under F-COMPAT=A). The pool itself does
        NOT introspect the model's policy class here — the caller is the
        authority (and is the only seam that knows whether ``model`` was built by
        ``_make_shared_ppo(..., policy_kind="mlp")`` or ``"lstm"``). The field is
        consumed at :meth:`materialize` time to select the MLP vs LSTM closure
        shape (F-MATERIALIZE-API=A).
        """
        relative_path = Path(f"round_{round_index}.zip")
        # SB3's PPO.save accepts both str and PathLike; pass the absolute path.
        absolute_path = self._pool_dir / relative_path
        model.save(absolute_path)
        member = PoolMember(
            member_id=f"checkpoint:round_{round_index}_seed_{seed}",
            kind="checkpoint",
            archetype_name=f"frozen_policy:round_{round_index}_seed_{seed}",
            checkpoint_path=relative_path,
            round_index=round_index,
            policy_kind=policy_kind,
        )
        self.add(member)
        self.save()
        return member

    def save(self, pool_dir: Path | None = None) -> None:
        """Write ``manifest.json`` atomically (tmp file + ``os.replace``).

        Writes to a tmp file in the SAME directory as the target (so ``os.replace``
        is a pure-rename, not a cross-filesystem copy), then atomically replaces
        the target. On POSIX ``os.replace`` is atomic; on Windows it is atomic
        within a volume. A crash between ``tmp.write`` and ``os.replace`` leaves
        the OLD manifest intact (the load path always sees a complete manifest).

        Default writes to ``self._pool_dir / "manifest.json"``; an explicit
        ``pool_dir`` lets the caller persist a snapshot to a different location
        (used by tests).

        RLB-8a follow-up (senior-review M1+M5): ``home_regions``/
        ``arena_region_table_hash``/``arena_lock_knobs`` are emitted ONLY when
        there is a real seating to record (:meth:`record_arena_seating` was
        called with a non-``None`` ``home_regions``) — a pool with no seating
        omits all three keys entirely, so its manifest stays byte-identical to
        every pre-RLB-8a pool (the previous "always emit, null when unset" shape
        grew every default pool's manifest for no reader benefit). ``version``
        is stamped :data:`MANIFEST_VERSION_ARENA_SEATING` exactly when the keys
        are present, so pre-RLB-8a code loading a manifest that DOES carry
        seating fails loud at :meth:`load` instead of silently discarding it —
        see the ``MANIFEST_VERSION_ARENA_SEATING`` docstring note.

        The H5 fix round: ``reward_mode`` is emitted the SAME
        way — only when :meth:`record_reward_mode` was called with a non-``None``
        value — independently of whether arena seating is also present (a
        calibrated pool trained with ``--reward weighted`` has a recorded
        ``reward_mode`` but no seating at all). ``version`` is bumped to
        :data:`MANIFEST_VERSION_REWARD_MODE` whenever ``reward_mode`` IS present,
        taking precedence over :data:`MANIFEST_VERSION_ARENA_SEATING` when both
        are (3 > 2 — see that constant's own docstring for why "notable" is
        decided by the caller, not here).
        """
        target_dir = Path(pool_dir) if pool_dir is not None else self._pool_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "manifest.json"
        payload: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "members": [m.to_json_dict() for m in self._members],
        }
        if self._home_regions is not None:
            # `list(...)` mirrors PoolMember.to_json_dict's own tuple -> list
            # convention (JSON has no tuple type). Note: `json.dump` below uses
            # `sort_keys=True`, so these keys do NOT land "at the tail" of the
            # written file regardless of insertion order here (alphabetical:
            # arena_lock_knobs, arena_region_table_hash, home_regions, members,
            # reward_mode, version) — see `load`'s own key-order-agnostic
            # `.get(...)` reads.
            payload["version"] = MANIFEST_VERSION_ARENA_SEATING
            payload["home_regions"] = list(self._home_regions)
            payload["arena_region_table_hash"] = self._arena_region_table_hash
            payload["arena_lock_knobs"] = self._arena_lock_knobs
        if self._reward_mode is not None:
            payload["version"] = MANIFEST_VERSION_REWARD_MODE
            payload["reward_mode"] = self._reward_mode
        # NamedTemporaryFile with delete=False so we keep the file handle until
        # close, then os.replace it onto the target. delete=True races on Windows.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target_dir),
            prefix=".manifest-",
            suffix=".json.tmp",
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, target)

    @classmethod
    def load(cls, pool_dir: Path) -> FrozenPolicyPool:
        """Restore a pool from ``pool_dir / "manifest.json"`` (deterministic).

        Raises ``FileNotFoundError`` if the manifest is missing (a clear error so
        a missing-pool typo doesn't silently produce an empty pool); raises
        ``ValueError`` if the manifest's ``version`` is unknown (forward-compat
        guard — a newer pool written by a future release will surface as a clear
        error, not a silent shape mismatch).

        RLB-8a follow-up (senior-review M1+M5): accepts both
        :data:`MANIFEST_VERSION` (no arena seating recorded — the three keys are
        entirely absent) and :data:`MANIFEST_VERSION_ARENA_SEATING` (a real
        seating IS recorded — the three keys are present); an unrecognised
        version still fails loud. ``home_regions``/``arena_region_table_hash``/
        ``arena_lock_knobs`` are read back via ``.get(..., None)`` regardless, so
        a manifest at either version loads correctly.

        The H5 fix round: also accepts
        :data:`MANIFEST_VERSION_REWARD_MODE` (a recorded ``reward_mode`` IS
        present); ``reward_mode`` is likewise read back via ``.get(..., None)``
        regardless of version, so a manifest at any of the three versions —
        including one carrying BOTH arena seating and a reward mode — loads
        correctly.
        """
        path = Path(pool_dir) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"FrozenPolicyPool.load: no manifest at {path}; "
                f"call FrozenPolicyPool({pool_dir!r}) or make_seeded_pool({pool_dir!r}) first"
            )
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        version = payload.get("version")
        if version not in (
            MANIFEST_VERSION,
            MANIFEST_VERSION_ARENA_SEATING,
            MANIFEST_VERSION_REWARD_MODE,
        ):
            raise ValueError(
                f"FrozenPolicyPool.load: manifest at {path} has unknown version {version!r}; "
                f"this code supports version {MANIFEST_VERSION}, "
                f"{MANIFEST_VERSION_ARENA_SEATING}, or {MANIFEST_VERSION_REWARD_MODE}"
            )
        members = tuple(PoolMember.from_json_dict(item) for item in payload.get("members", []))
        raw_home_regions = payload.get("home_regions")
        home_regions = tuple(raw_home_regions) if raw_home_regions is not None else None
        return cls(
            Path(pool_dir),
            members=members,
            home_regions=home_regions,
            arena_region_table_hash=payload.get("arena_region_table_hash"),
            arena_lock_knobs=payload.get("arena_lock_knobs"),
            reward_mode=payload.get("reward_mode"),
        )

    def materialize(self, member: PoolMember, env: Any) -> Callable[..., Any]:
        """ """
        del env  # forward-compat; the current implementation needs no env reference

        if member.kind == "scripted":

            def _scripted_placeholder(_obs: Any, **_kwargs: Any) -> Any:
                # ``**_kwargs`` absorbs the T-A4 ``episode_start`` kwarg so a
                # caller that uniformly passes it doesn't blow up on the
                # scripted branch's NotImplementedError path (the placeholder
                # raises on call regardless of arguments — the kwarg-acceptance
                # is just signature-uniformity with the checkpoint branches).
                raise NotImplementedError(
                    f"scripted PoolMember {member.member_id!r} is seated via the env's "
                    f"npc_archetypes (the core's _make_npc_policy seam), NOT via this "
                    f"callable; the league wrapper (Slice A T-A3) handles the seating"
                )

            return _scripted_placeholder

        if member.kind == "archetype":
            # H4/RLB-8b — no SB3, no disk zip: the closure comes straight from
            # the calibrated spike roster or (a geo name) the geo roster, and is
            # cached like a checkpoint model so a pool reused across many resets
            # does not rebuild it N times.
            if member.member_id not in self._materialize_cache:
                if member.archetype_name in GEO_ARCHETYPE_LABELS:
                    if self._region_table is None:
                        raise ValueError(
                            f"archetype PoolMember {member.member_id!r} is a geo "
                            "archetype but this pool has no region_table — pass "
                            "one to make_seeded_pool(..., region_table=...) when "
                            "seeding it, or call attach_region_table(...) after "
                            "FrozenPolicyPool.load(...) (a restored pool never "
                            "recovers its region table from the manifest)"
                        )
                    # Build the FULL 4-posture roster ONCE per pool instance (on the
                    # first geo member materialized), never once per member — mirrors
                    # ladder.py's own _materialize_predicts contract (see
                    # _geo_archetype_roster's class-level comment).
                    if self._geo_archetype_roster is None:
                        self._geo_archetype_roster = _geo_archetypes(
                            len(self._region_table), self._region_table
                        )
                    self._materialize_cache[member.member_id] = self._geo_archetype_roster[
                        member.archetype_name
                    ]
                else:
                    self._materialize_cache[member.member_id] = calibrated_archetypes()[
                        member.archetype_name
                    ]
            return self._materialize_cache[member.member_id]

        # checkpoint branch — lazy SB3 imports so module load stays cheap.
        if member.member_id not in self._materialize_cache:
            if member.checkpoint_path is None:
                # Defensive: PoolMember.__post_init__ guarantees non-None for
                # kind=="checkpoint", but typing infers Path | None. Raising here
                # surfaces a future invariant break loudly instead of silently
                # passing None to PPO.load.
                raise RuntimeError(
                    f"checkpoint PoolMember {member.member_id!r} has no checkpoint_path; "
                    f"the dataclass invariant was bypassed"
                )
            if member.policy_kind == "lstm":
                # Phase 1.6 Slice A T-A4 — lazy sb3-contrib import (F-LAZY=A).
                # Mirrors the lazy-SB3 contract: importing this module under the
                # base [train] extra (no [recurrent]) STILL succeeds; the
                # ImportError surfaces only when an LSTM checkpoint is actually
                # materialized. The architect's helpful-message convention names
                # the [recurrent] extra explicitly.
                try:
                    from sb3_contrib import RecurrentPPO  # noqa: PLC0415 - lazy by design
                except ImportError as exc:  # pragma: no cover - exercised when extra missing
                    raise ImportError(
                        f"materializing PoolMember {member.member_id!r} with "
                        f"policy_kind='lstm' requires the [recurrent] extra; "
                        f"install with: pip install 'retail-simulator[recurrent]' "
                        f"(or: pip install sb3-contrib>=2.3,<3.0)"
                    ) from exc
                self._materialize_cache[member.member_id] = RecurrentPPO.load(
                    self._pool_dir / member.checkpoint_path
                )
            else:
                from stable_baselines3 import PPO  # noqa: PLC0415 - lazy by design

                self._materialize_cache[member.member_id] = PPO.load(
                    self._pool_dir / member.checkpoint_path
                )
        loaded_model = self._materialize_cache[member.member_id]

        if member.policy_kind == "mlp":

            def _checkpoint_predict_mlp(obs: Any, **_kwargs: Any) -> Any:
                # ``**_kwargs`` absorbs the T-A4 ``episode_start`` kwarg —
                # MLP has no recurrent state to reset, so the kwarg is
                # accepted-and-ignored (the F-MATERIALIZE-API=A backward-compat
                # path: every Slice A/B/C caller that does ``predict(obs)``
                # without the kwarg continues to behave identically).
                action, _state = loaded_model.predict(obs, deterministic=True)
                return action

            return _checkpoint_predict_mlp

        # LSTM branch (F-MATERIALIZE-API=A) — the closure carries lstm_states
        # across consecutive calls and resets them when ``episode_start=True``.
        # The state is closure-local (a 1-element mutable list — the standard
        # Python idiom for a mutable cell captured by an inner function). The
        # SB3 ``RecurrentPPO.predict`` API:
        #
        #     predict(obs, state=<tuple of ndarrays | None>,
        #             episode_start=<ndarray of bool, shape (n_envs,)>,
        #             deterministic=True) -> (action, new_state)
        #
        # We pass ``n_envs=1`` shape ``(1,)`` for ``episode_start`` (the league
        # materialize seam evaluates one env at a time — the head-to-head
        # primitive's contract).
        state_cell: list[Any] = [None]

        def _checkpoint_predict_lstm(obs: Any, *, episode_start: bool = False) -> Any:
            if episode_start:
                # SB3 contract: state=None + episode_starts=True means "fresh
                # hidden state". Resetting here BEFORE the predict call keeps
                # the two paths consistent (the alternative — letting the
                # ``episode_starts`` flag alone drive the reset inside SB3 —
                # would leave the carried cell holding stale state if SB3 were
                # to change the reset semantics on a future release).
                state_cell[0] = None
            episode_starts_arr = np.array([episode_start], dtype=bool)
            action, new_state = loaded_model.predict(
                obs,
                state=state_cell[0],
                episode_start=episode_starts_arr,
                deterministic=True,
            )
            state_cell[0] = new_state
            return action

        return _checkpoint_predict_lstm


@dataclass(frozen=True)
class WinRateEntry:
    """One cell of the win-rate matrix — head-to-head wins/total for a (learner, opponent) pair.

    Frozen so the entry is hashable + a value-type; the surrounding
    :class:`WinRateMatrix` rebuilds entries (rather than mutating them) when
    :meth:`WinRateMatrix.record` accumulates a new outcome. The invariants
    (``wins >= 0``, ``total >= 0``, ``wins <= total``) are checked at
    construction so a corrupted on-disk record surfaces as a clear
    :class:`ValueError` at :meth:`WinRateMatrix.load` time rather than as a
    silent division-by-zero later.
    """

    wins: int
    total: int

    def __post_init__(self) -> None:
        if self.wins < 0:
            raise ValueError(f"WinRateEntry: wins must be >= 0, got {self.wins}")
        if self.total < 0:
            raise ValueError(f"WinRateEntry: total must be >= 0, got {self.total}")
        if self.wins > self.total:
            raise ValueError(f"WinRateEntry: wins ({self.wins}) cannot exceed total ({self.total})")

    @property
    def rate(self) -> float:
        """Empirical win-rate; raises if ``total == 0`` (caller should check via ``total`` first)."""
        if self.total == 0:
            raise ValueError("WinRateEntry.rate: total == 0; check `total` before accessing rate")
        return self.wins / self.total


@dataclass
class WinRateMatrix:
    """Per-(learner, opponent) head-to-head win-rate tracker — PFSP's input signal.

    Sparse dict keyed by ``(learner_id, opponent_id)``. The ``learner_id`` is
    the pool member ID of the round-N learner; the ``opponent_id`` is the pool
    member being evaluated against. Recording a win adds to the (learner,
    opponent) entry only — NO symmetric/reverse-record (the opponent's
    perspective is not auto-derived; if the caller needs both directions it
    records each explicitly). This keeps the matrix's growth rule unambiguous
    and lets PFSP's weighting function read a single learner's row.

    Persisted atomically as JSON at ``<pool_dir>/win_rate_matrix.json``
    (mirrors the :meth:`FrozenPolicyPool.save` tmp+os.replace+fsync pattern).
    """

    # NOTE: NOT frozen — :meth:`record` mutates the internal dict. The mutable
    # container is internal; the public API (record/win_rate/save/load) is the
    # surface contract callers depend on.
    _entries: dict[tuple[str, str], WinRateEntry] = field(default_factory=dict)

    def record(self, learner_id: str, opponent_id: str, learner_won: bool) -> None:
        """Record one head-to-head outcome.

        Repeated calls with the same ``(learner_id, opponent_id)`` accumulate
        the new outcome on top of any existing entry (the matrix tracks the
        cumulative head-to-head over the league's life — round-end evals
        keep adding without resetting).
        """
        key = (learner_id, opponent_id)
        prev = self._entries.get(key, WinRateEntry(wins=0, total=0))
        self._entries[key] = WinRateEntry(
            wins=prev.wins + (1 if learner_won else 0),
            total=prev.total + 1,
        )

    def win_rate(self, learner_id: str, opponent_id: str) -> float | None:
        """Empirical win-rate for ``(learner_id, opponent_id)``, or ``None`` if no record.

        The PFSP distribution distinguishes "no head-to-head yet" (``None`` —
        cold-start handled via ``bias_for_unseen``) from "head-to-head exists
        with rate X" (a float in ``[0, 1]``).
        """
        key = (learner_id, opponent_id)
        entry = self._entries.get(key)
        if entry is None or entry.total == 0:
            return None
        return entry.rate

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[tuple[tuple[str, str], WinRateEntry], ...]:
        """Sorted immutable snapshot of all entries (for testing / persistence)."""
        return tuple(sorted(self._entries.items()))

    def save(self, path: Path) -> None:
        """Atomic JSON write — tmp file + ``fsync`` + ``os.replace``.

        Mirrors :meth:`FrozenPolicyPool.save`'s discipline (write to a tmp file
        in the same directory, fsync, then ``os.replace`` onto the target). A
        crash between the tmp write and the replace leaves the OLD matrix
        intact (the load path always sees a complete file). JSON has no
        native tuple keys, so entries serialize as a list of records sorted
        by ``(learner_id, opponent_id)`` for diff-ability.
        """
        payload = {
            "version": WIN_RATE_MATRIX_VERSION,
            "entries": [
                {
                    "learner_id": learner_id,
                    "opponent_id": opponent_id,
                    "wins": entry.wins,
                    "total": entry.total,
                }
                for (learner_id, opponent_id), entry in self.entries()
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> WinRateMatrix:
        """Load from atomic JSON; raises :class:`FileNotFoundError` if missing.

        Raises :class:`ValueError` on an unknown version or a malformed
        record shape — the boundary between on-disk data and in-memory
        invariants. A missing file surfaces as :class:`FileNotFoundError`
        so callers can distinguish "fresh pool" (caller may fall back to
        an empty matrix) from "operator typo" (caller may surface the
        error).
        """
        if not path.is_file():
            raise FileNotFoundError(f"WinRateMatrix.load: file not found at {path}")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError(
                f"WinRateMatrix.load: expected JSON object, got {type(payload).__name__}"
            )
        version = payload.get("version")
        if version != WIN_RATE_MATRIX_VERSION:
            raise ValueError(
                f"WinRateMatrix.load: unknown version {version!r} at {path}; "
                f"expected {WIN_RATE_MATRIX_VERSION}"
            )
        entries_raw = payload.get("entries", [])
        if not isinstance(entries_raw, list):
            raise ValueError("WinRateMatrix.load: 'entries' must be a JSON array")
        matrix = cls()
        for record in entries_raw:
            if not isinstance(record, dict):
                raise ValueError(
                    f"WinRateMatrix.load: entry must be an object, got {type(record).__name__}"
                )
            try:
                learner_id = str(record["learner_id"])
                opponent_id = str(record["opponent_id"])
                wins = int(record["wins"])
                total = int(record["total"])
            except KeyError as exc:
                raise ValueError(
                    f"WinRateMatrix.load: missing required key {exc.args[0]!r}"
                ) from exc
            matrix._entries[(learner_id, opponent_id)] = WinRateEntry(wins=wins, total=total)
        return matrix


def make_seeded_pool(
    pool_dir: Path,
    *,
    archetypes: tuple[str, ...] = DEFAULT_POOL_SEED_ARCHETYPES,
    spike_archetypes: tuple[str, ...] = (),
    geo_archetypes: tuple[str, ...] = (),
    region_table: tuple[RegionConfig, ...] | None = None,
) -> FrozenPolicyPool:
    """ """
    for name in archetypes:
        if name not in DEFAULT_POOL_SEED_ARCHETYPES:
            raise ValueError(
                f"make_seeded_pool: archetype {name!r} not in scripted allowlist "
                f"{DEFAULT_POOL_SEED_ARCHETYPES}"
            )
    for name in spike_archetypes:
        if name not in POOL_SPIKE_ARCHETYPE_LABELS:
            raise ValueError(
                f"make_seeded_pool: spike archetype {name!r} not in allowlist "
                f"{POOL_SPIKE_ARCHETYPE_LABELS}"
            )
    for name in geo_archetypes:
        if name not in GEO_ARCHETYPE_LABELS:
            raise ValueError(
                f"make_seeded_pool: geo archetype {name!r} not in allowlist "
                f"{GEO_ARCHETYPE_LABELS}"
            )
    if geo_archetypes and region_table is None:
        raise ValueError(
            f"make_seeded_pool: geo archetypes {geo_archetypes} were given but no "
            "region_table — a geo posture's expansion rule needs to know the world "
            "it plays in (geo_archetypes.geo_archetypes' own region_table argument); "
            "pass the effective config's demand.regions"
        )
    directory = Path(pool_dir)
    directory.mkdir(parents=True, exist_ok=True)
    scripted_members = tuple(
        PoolMember(
            member_id=f"scripted:{name}",
            kind="scripted",
            archetype_name=name,
            checkpoint_path=None,
            round_index=_SCRIPTED_ROUND_INDEX,
        )
        for name in archetypes
    )
    spike_archetype_members = tuple(
        PoolMember(
            member_id=f"archetype:{name}",
            kind="archetype",
            archetype_name=name,
            checkpoint_path=None,
            round_index=_SCRIPTED_ROUND_INDEX,
        )
        for name in spike_archetypes
    )
    geo_archetype_members = tuple(
        PoolMember(
            member_id=f"archetype:{name}",
            kind="archetype",
            archetype_name=name,
            checkpoint_path=None,
            round_index=_SCRIPTED_ROUND_INDEX,
        )
        for name in geo_archetypes
    )
    pool = FrozenPolicyPool(
        directory,
        members=scripted_members + spike_archetype_members + geo_archetype_members,
        region_table=region_table,
    )
    pool.save()
    return pool


# --- OpponentDistribution protocol + concrete shapes (Slice A T-A2) ----------


@runtime_checkable
class OpponentDistribution(Protocol):
    """ """

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:  # pragma: no cover - Protocol
        ...


@dataclass(frozen=True)
class UniformOpponentDistribution:
    """Uniform sampler over the pool's current members (F2-A RECOMMENDED).

    The simplest concrete :class:`OpponentDistribution`: each
    ``sample(rng, pool)`` returns ONE of ``pool.members()`` chosen uniformly
    via ``rng.integers(len(pool))``. Mirrors
    :class:`harness.curriculum.UniformScenarioDistribution`: determinism is
    exactly the rng's (identical ``rng`` state ⇒ identical draw sequence); a
    future :class:`PFSPOpponentDistribution` (F2-C, post-1.5) is a strict
    drop-in.

    The empty-pool case raises :class:`ValueError` at sample-time — the
    distribution layer guards the boundary; the data structure permits the
    empty-pool state (a freshly-created pool dir with no members yet).

    The ``win_rate_matrix`` + ``learner_id`` keyword arguments (Slice C
    T-C1-PFSP Protocol extension) are accepted and IGNORED — uniform
    sampling is independent of head-to-head outcomes.
    """

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        """Draw ONE :class:`PoolMember` uniformly at random from ``pool``."""
        # win_rate_matrix + learner_id are accepted for Protocol compatibility
        # with :class:`PFSPOpponentDistribution`; uniform sampling ignores them.
        del win_rate_matrix, learner_id
        if len(pool) == 0:
            raise ValueError("UniformOpponentDistribution: pool is empty; sample() is ill-defined")
        index = int(rng.integers(0, len(pool)))
        return pool.members()[index]


@dataclass(frozen=True)
class LatestNOpponentDistribution:
    """Uniform over the latest N learner-checkpoint members (F2-B alternative).

    Eligibility:

    * The "latest N checkpoints" set is ``pool.members()`` filtered to
      ``kind == "checkpoint"``, sorted by ``round_index`` DESCENDING, then
      truncated to the first ``n`` entries (the newest N). Ties on
      ``round_index`` resolve via the pool's stable insertion order from
      :meth:`FrozenPolicyPool.members` (so the same pool yields the same
      eligible set across processes).
    * If ``include_scripted=True``: the eligible set is the union of the
      latest-N checkpoints with EVERY FIXED-opponent member — ``kind in
      ("scripted", "archetype")`` (H4: the calibrated Phase 4 spike
      archetypes are fixed opponents exactly like the core scripted NPCs,
      so they share the same diversity-floor treatment). The final eligible
      tuple preserves :meth:`FrozenPolicyPool.members` insertion order and
      de-duplicates by ``member_id``. The field name ``include_scripted`` is
      UNCHANGED (on-disk/API back-compat) even though it now also gates
      archetype members.
    * If ``include_scripted=False``: the eligible set is the latest-N
      checkpoints ONLY (no diversity floor; the policy always plays a
      recent learner). An eligible set that ends up empty raises
      :class:`ValueError` at sample-time (the same boundary contract as
      :class:`UniformOpponentDistribution`).

    Sampling is then uniform over the eligible tuple via
    ``rng.integers(len(eligible))`` — same determinism contract as the
    uniform distribution.
    """

    n: int
    include_scripted: bool = True

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(
                f"LatestNOpponentDistribution: n must be >= 1, got {self.n}; "
                "the trailing window must include at least one checkpoint"
            )

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        """Draw ONE :class:`PoolMember` uniformly from the eligible set.

        The ``win_rate_matrix`` + ``learner_id`` keyword arguments (Slice C
        T-C1-PFSP Protocol extension) are accepted and IGNORED — the
        latest-N window is a recency policy, not a head-to-head one.
        """
        # win_rate_matrix + learner_id are accepted for Protocol compatibility
        # with :class:`PFSPOpponentDistribution`; latest-N sampling ignores them.
        del win_rate_matrix, learner_id
        members = pool.members()
        # The latest-N checkpoint set: stable insertion order is the tie-break
        # for equal round_index — sorting by (-round_index, insertion_index)
        # keeps the documented contract explicit.
        checkpoint_indexed = [
            (insertion_index, member)
            for insertion_index, member in enumerate(members)
            if member.kind == "checkpoint"
        ]
        checkpoint_indexed.sort(key=lambda pair: (-pair[1].round_index, pair[0]))
        latest_checkpoint_ids: set[str] = {
            member.member_id for _idx, member in checkpoint_indexed[: self.n]
        }

        eligible_ids: set[str] = set(latest_checkpoint_ids)
        if self.include_scripted:
            for member in members:
                if member.kind in ("scripted", "archetype"):
                    eligible_ids.add(member.member_id)

        # Preserve the pool's insertion order in the eligible tuple — the
        # documented stable-ordering contract (so determinism survives the
        # de-duplication step).
        eligible = tuple(member for member in members if member.member_id in eligible_ids)
        if not eligible:
            raise ValueError(
                "LatestNOpponentDistribution: no eligible members; pool has no "
                "checkpoint members and include_scripted=False"
            )
        index = int(rng.integers(0, len(eligible)))
        return eligible[index]


@dataclass(frozen=True)
class PFSPOpponentDistribution:
    """ """

    alpha: float = 2.0
    bias_for_unseen: float = 1.0
    mix_uniform: float = 0.0

    def __post_init__(self) -> None:
        # alpha < 0 inverts the curve (winners get high weight instead of
        # losers); reject at construction time so the operator catches the
        # typo immediately rather than at sample-time.
        if self.alpha < 0:
            raise ValueError(f"PFSPOpponentDistribution: alpha must be >= 0, got {self.alpha}")
        # bias_for_unseen <= 0 would zero out (or invert) the cold-start
        # weighting — at exactly 0 an all-unseen pool yields total-weight=0
        # and the sampler is ill-defined; reject the whole non-positive range
        # so the contract is unambiguous.
        if self.bias_for_unseen <= 0:
            raise ValueError(
                f"PFSPOpponentDistribution: bias_for_unseen must be > 0, got {self.bias_for_unseen}"
            )
        # mix_uniform is a mixing FRACTION — outside [0, 1] the formula below
        # either subtracts probability mass back out (negative) or overshoots
        # past a valid distribution (>1); reject at construction time like the
        # two checks above rather than letting sample() silently misbehave.
        if not 0.0 <= self.mix_uniform <= 1.0:
            raise ValueError(
                f"PFSPOpponentDistribution: mix_uniform must be in [0.0, 1.0], "
                f"got {self.mix_uniform}"
            )

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        """Sample opponent via PFSP weighting; excludes the learner from its own opponent set."""
        members = pool.members()
        if not members:
            raise ValueError("PFSPOpponentDistribution: pool is empty")

        # Exclude the learner from its own opponent set — a learner does not
        # play itself in PFSP. When ``learner_id`` is None (no learner
        # identified yet), the entire pool is eligible.
        eligible = [m for m in members if m.member_id != learner_id]
        if not eligible:
            raise ValueError(
                f"PFSPOpponentDistribution: no eligible opponents in pool of "
                f"{len(members)} (learner_id={learner_id!r} excluded; pool may "
                f"be size 1)"
            )

        # Compute per-member weights. The matrix is consulted only when BOTH
        # the matrix and the learner_id are provided; otherwise every member
        # falls back to ``bias_for_unseen`` (cold-start uniform-with-bias).
        weights = np.empty(len(eligible), dtype=np.float64)
        for i, m in enumerate(eligible):
            wr: float | None = None
            if win_rate_matrix is not None and learner_id is not None:
                wr = win_rate_matrix.win_rate(learner_id, m.member_id)
            if wr is None:
                # No-history fallback (F-WIN-INIT=A).
                weights[i] = self.bias_for_unseen
            else:
                # AlphaStar-canonical (1 - x)^alpha (F-PFSP-FN=A). A
                # perfectly-beaten opponent (win_rate=1.0) gets weight=0,
                # which is exactly the "stop drilling already-beaten
                # opponents" semantics the algorithm prescribes.
                weights[i] = (1.0 - wr) ** self.alpha

        total = float(weights.sum())
        if total <= 0.0:
            weights = np.ones(len(eligible), dtype=np.float64)
            total = float(len(eligible))

        probs = weights / total
        if self.mix_uniform > 0.0:
            probs = (1.0 - self.mix_uniform) * probs + self.mix_uniform / len(eligible)
        idx = int(rng.choice(len(eligible), p=probs))
        return eligible[idx]


# --- Named-league registry ---------------------------------------------------

# A small registry of name → builder. The CLI's ``--league=<name>`` flag (Slice B
# T-A4) looks up the builder, calls it with the operator-supplied pool_dir, and
# threads the returned pool into :func:`run_league_training` (Slice B). The
# default builder is :func:`make_seeded_pool` — the F5-A pool seed; future named
# leagues (a "no-scripted" league, a "diverse" league, …) register here.
KNOWN_LEAGUES: dict[str, Callable[[Path], FrozenPolicyPool]] = {
    DEFAULT_LEAGUE_NAME: lambda pool_dir: make_seeded_pool(pool_dir),
}


def known_leagues() -> tuple[str, ...]:
    """The names the CLI's ``--league=<name>`` flag accepts (sorted for stability)."""
    return tuple(sorted(KNOWN_LEAGUES))


def build_league(name: str, pool_dir: Path) -> FrozenPolicyPool:
    """Look up a named league builder and call it with ``pool_dir``.

    Unknown names raise :class:`ValueError` listing the known names — boundary
    validation at the CLI surface, never a silent fallback.
    """
    if name not in KNOWN_LEAGUES:
        raise ValueError(f"unknown league {name!r}; known leagues: {list(known_leagues())}")
    builder = KNOWN_LEAGUES[name]
    return builder(Path(pool_dir))


# --- LeagueEnvWrapper (Slice A T-A3) -----------------------------------------

# Lazy / module-top import: pettingzoo's ``ParallelEnv`` is the base class
# SuperSuit's ``pettingzoo_env_to_vec_env_v1`` requires via an ``isinstance``
# check (see ``supersuit/vector/vector_constructors.py`` — the bridge does
# NOT duck-type). Inheriting it here makes the league wrapper a first-class
# PettingZoo Parallel env so the SuperSuit bridge / Slice B
# ``build_parallel_vec_env`` integration paths work end-to-end without a
# downstream adapter shim. PettingZoo is already a non-lazy dependency of
# this module (transitively via ``retail_simulator``'s package init); the
# explicit import here only makes the dependency loud.
from pettingzoo import ParallelEnv  # noqa: E402 - kept near the wrapper for locality


class LeagueEnvWrapper(ParallelEnv):  # type: ignore[misc, unused-ignore]
    """ """

    metadata: dict[str, Any] = {"name": "retail_sim_league_parallel_v0", "render_modes": []}

    def __init__(
        self,
        wrapped_env_factory: Callable[[tuple[str, ...]], Any],
        pool: FrozenPolicyPool,
        distribution: OpponentDistribution,
        *,
        league_seed: int = 0,
        league_mode: Literal["self_play", "single_learner"] = "self_play",
        single_learner_env_factory: Callable[[int, tuple[str, ...]], Any] | None = None,
        max_episode_steps: int | None = None,
        episode_seed_base: int | None = None,
    ) -> None:
        """ """
        if league_mode == "single_learner" and single_learner_env_factory is None:
            raise ValueError(
                "single_learner mode requires single_learner_env_factory: "
                "Callable[[int, tuple[str, ...]], Any]; got None"
            )
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be None or an int >= 1, got {max_episode_steps}"
            )

        self._wrapped_env_factory = wrapped_env_factory
        self._single_learner_env_factory = single_learner_env_factory
        self._pool = pool
        self._distribution = distribution
        self._league_seed = league_seed
        self._league_mode: Literal["self_play", "single_learner"] = league_mode
        self._max_episode_steps = max_episode_steps
        self._ticks_since_reset: int = 0
        self._episode_seed_seq: np.random.SeedSequence | None = (
            np.random.SeedSequence(episode_seed_base) if episode_seed_base is not None else None
        )
        # The CARRIED league Generator: built once in __init__ from league_seed,
        # never re-seeded across reset() calls. The carried-Generator pattern
        # (the three-RNG isolation contract — recorded at module top, enforced
        # here): only reset() consumes; step() and __init__'s snapshot draw do
        # NOT consume.
        self._league_rng: np.random.Generator = np.random.default_rng(league_seed)

        # Build an INITIAL env from the first opponent so ``observation_space`` /
        # ``action_space`` are queryable BEFORE the first reset (the PettingZoo
        # API convention; same pattern as CurriculumEnvWrapper). The initial draw
        # is from a SNAPSHOT of league_rng state so the carried state is
        # untouched — the first reset() draws fresh.
        snapshot = self._league_rng.bit_generator.state
        initial_member = self._distribution.sample(self._league_rng, self._pool)
        self._league_rng.bit_generator.state = snapshot

        # State carried across step() calls under single_learner + checkpoint
        # opponent: the frozen opponent's deterministic predict closure and the
        # hidden seat-1 observation captured at the most recent reset/step. Both
        # are None when the wrapper is in self_play mode OR when the current
        # opponent is scripted (the scripted branch is seated by the env's
        # NPC mechanism — no in-process action injection needed). The closure
        # follows the F-MATERIALIZE-API=A contract (Phase 1.6 Slice A T-A4):
        # ``Callable[[obs, *, episode_start: bool = False], action]``.
        self._frozen_predict: Callable[..., Any] | None = None
        self._frozen_seat_index: int | None = None
        self._last_obs_seat_1: Any = None
        self._first_step_after_reset: bool = True
        # The hidden seat's stable PettingZoo id under single_learner +
        # checkpoint — the inner env builds with n_learning_agents=2 and names
        # seats ``retailer_0`` / ``retailer_1`` (parallel_env._agent_id).
        self._hidden_agent_id: str = "retailer_1"

        if self._league_mode == "self_play":
            if initial_member.kind != "scripted":
                raise NotImplementedError(
                    "self_play LeagueEnvWrapper only seats scripted-archetype pool members; "
                    "checkpoint seating requires league_mode='single_learner'. "
                    f"Got member kind={initial_member.kind!r}."
                )
            self._env: Any = self._wrapped_env_factory((initial_member.archetype_name,))
            # The PettingZoo agent interface delegates straight to the wrapped env
            # under self_play. Mirrors CurriculumEnvWrapper exactly so SuperSuit's
            # MarkovVectorEnv finds the same surface either way.
            self.possible_agents: list[str] = list(self._env.possible_agents)
            self.agents: list[str] = list(self._env.agents)
        else:
            # single_learner: build the FIRST inner env from the sampled member
            # using the single_learner factory. For scripted, n_learning_agents=1
            # + the scripted archetype as an NPC; for checkpoint, n_learning_agents=2
            # + no NPC (the wrapper hides seat 1 and injects its action at step
            # time from materialize()). Either way the surface exposed to the
            # caller is the SINGLE PettingZoo agent ``retailer_0``.
            assert self._single_learner_env_factory is not None  # checked above
            self._env = self._build_single_learner_env(initial_member)
            self.possible_agents = ["retailer_0"]
            self.agents = ["retailer_0"]
            if initial_member.kind in ("checkpoint", "archetype"):
                self._frozen_predict = self._pool.materialize(initial_member, self._env)
                self._frozen_seat_index = 1

        # SuperSuit's MarkovVectorEnv reads ``par_env.unwrapped.render_mode``
        # (the 0.5 / 1.4 documented gap). Mirror the curriculum wrapper: expose
        # a None render_mode here so the bridge sees a well-defined attribute
        # regardless of how it inspects the wrapper.
        self.render_mode: Any = getattr(self._env, "render_mode", None)
        # Track the last sampled member (a DX hook for ``info`` / diagnostics).
        self._last_member: PoolMember = initial_member

    def _build_single_learner_env(self, member: PoolMember) -> Any:
        """Construct the inner env for ``single_learner`` mode for ``member``.

        Scripted members ⇒ ``factory(1, (archetype_name,))`` — the inner env has
        one PettingZoo learner + one NPC seat, mirroring the 0.4 byte-identity
        shape. Every NON-scripted member (``kind in ("checkpoint", "archetype")``)
        ⇒ ``factory(2, ())`` — the inner env has two learners; the wrapper hides
        seat 1 from PettingZoo and injects seat 1's action at step time from
        :meth:`FrozenPolicyPool.materialize` (H4: an archetype member's
        open-loop closure is injected through this SAME path a checkpoint's
        SB3 predict closure uses). Both branches go through the SAME factory
        seam so the caller controls config / seed / wrappers uniformly.
        """
        assert self._single_learner_env_factory is not None
        if member.kind == "scripted":
            return self._single_learner_env_factory(1, (member.archetype_name,))
        return self._single_learner_env_factory(2, ())

    @property
    def unwrapped(self) -> "LeagueEnvWrapper":
        """The "innermost" env — we are the outermost wrapper, so return self.

        SuperSuit's ``MarkovVectorEnv`` accesses ``par_env.unwrapped.render_mode``;
        a wrapper that returned the underlying env here would leak the inner
        reference (which is rebuilt every ``reset()``), so we return ``self``
        and expose ``render_mode`` on the wrapper.
        """
        return self

    @property
    def league_rng(self) -> np.random.Generator:
        """ """
        return self._league_rng

    @property
    def last_member(self) -> PoolMember:
        """The most recently sampled :class:`PoolMember` (set by the last ``reset()``)."""
        return self._last_member

    def observation_space(self, agent: str) -> Any:
        """The homogeneous observation Box (schema v10 — UNCHANGED across opponents).

        Under ``single_learner`` mode the wrapper exposes ONLY ``retailer_0``'s
        space (the hidden seat is a wrapper internal, not a PZ agent); the
        inner env's space for ``retailer_0`` is identical to ``retailer_1``'s
        (homogeneous), so this is a delegation that hides the second key.
        """
        return self._env.observation_space(agent)

    def action_space(self, agent: str) -> Any:
        """The homogeneous action Box (schema v10 — UNCHANGED across opponents)."""
        return self._env.action_space(agent)

    @property
    def observation_spaces(self) -> dict[str, Any]:
        """The per-agent observation Box dict (PettingZoo / SuperSuit attribute surface).

        Under ``single_learner`` mode returns ONLY ``retailer_0``'s entry — the
        hidden seat is NOT a PettingZoo agent.
        """
        inner_spaces = cast(dict[str, Any], self._env.observation_spaces)
        if self._league_mode == "single_learner":
            return {"retailer_0": inner_spaces["retailer_0"]}
        return inner_spaces

    @property
    def action_spaces(self) -> dict[str, Any]:
        """The per-agent action Box dict (PettingZoo / SuperSuit attribute surface).

        Under ``single_learner`` mode returns ONLY ``retailer_0``'s entry — the
        hidden seat is NOT a PettingZoo agent.
        """
        inner_spaces = cast(dict[str, Any], self._env.action_spaces)
        if self._league_mode == "single_learner":
            return {"retailer_0": inner_spaces["retailer_0"]}
        return inner_spaces

    def _resolve_episode_seed(self, seed: int | None) -> int | None:
        """ """
        if seed is not None or self._episode_seed_seq is None or self._max_episode_steps is None:
            return seed
        child = self._episode_seed_seq.spawn(1)[0]
        return int(child.generate_state(1, dtype=np.uint32)[0])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Draw the next opponent, rebuild the env, and return its ``(observations, infos)``.

        The league draw happens HERE (NOT inside ``World.step`` — F-LOCUS-A): the
        next :class:`PoolMember` is sampled from ``distribution`` via the
        CARRIED ``league_rng`` (the only step that consumes its state outside
        ``__init__``'s snapshot draw); a fresh inner env is constructed from the
        factory with the sampled archetype name; the new env's
        ``reset(seed=seed, options=options)`` is called and its output returned.
        The ``seed`` argument seeds the new env's CORE PCG64 (the per-tick draw
        stream) — the league_rng is NOT advanced by reset's seed argument. This
        is the isolation contract.

        When ``episode_seed_base`` was given to ``__init__`` and ``max_episode_steps``
        is set, an incoming ``seed=None`` (exactly what the bridge's auto-reset
        always passes) is first replaced by a fresh per-episode seed derived via
        :meth:`_resolve_episode_seed` BEFORE it reaches the inner env's own
        ``reset(seed=...)`` — see that method's docstring for why (without it,
        every episode within a round would replay the SAME world). An explicit
        seed a caller passes is never touched.

        Under ``single_learner`` mode: sample the member, rebuild via the
        single-learner factory, cache the materialized predict closure +
        capture the hidden seat's initial observation (used for checkpoint OR
        archetype injection at step time), and return ONLY ``retailer_0``'s
        slice of the inner env's reset output.
        """
        member = self._distribution.sample(self._league_rng, self._pool)
        self._last_member = member
        # Close the prior env (frees its World resources); rebuild from the
        # factory. PettingZoo parallel envs are cheap to construct.
        self._env.close()
        self._first_step_after_reset = True
        self._ticks_since_reset = 0
        effective_seed = self._resolve_episode_seed(seed)

        if self._league_mode == "self_play":
            if member.kind != "scripted":
                raise NotImplementedError(
                    "self_play LeagueEnvWrapper only seats scripted-archetype pool members; "
                    "checkpoint seating requires league_mode='single_learner'. "
                    f"Got member kind={member.kind!r}."
                )
            self._env = self._wrapped_env_factory((member.archetype_name,))
            self.possible_agents = list(self._env.possible_agents)
            self.agents = list(self._env.agents)
            return cast(
                tuple[dict[str, Any], dict[str, Any]],
                self._env.reset(seed=effective_seed, options=options),
            )

        # single_learner mode: build the inner env appropriate for member.kind,
        # then cache the frozen predict closure (checkpoint OR archetype
        # branch) so step() can inject seat-1's action deterministically.
        self._env = self._build_single_learner_env(member)
        if member.kind in ("checkpoint", "archetype"):
            self._frozen_predict = self._pool.materialize(member, self._env)
            self._frozen_seat_index = 1
        else:
            self._frozen_predict = None
            self._frozen_seat_index = None
        self.possible_agents = ["retailer_0"]
        self.agents = ["retailer_0"]

        inner_obs, inner_infos = cast(
            tuple[dict[str, Any], dict[str, Any]],
            self._env.reset(seed=effective_seed, options=options),
        )
        # Capture the hidden seat's initial observation so the next step() can
        # feed it into the frozen predict closure. The scripted branch leaves
        # this None (the inner env seats the NPC itself — no observation needed).
        self._last_obs_seat_1 = inner_obs.get(self._hidden_agent_id)
        observations = {"retailer_0": inner_obs["retailer_0"]}
        infos = {"retailer_0": inner_infos.get("retailer_0", {})}
        return observations, infos

    def _apply_episode_truncation(self, trunc: dict[str, Any], infos: dict[str, Any]) -> None:
        """ """
        if self._max_episode_steps is None:
            return
        if self._ticks_since_reset < self._max_episode_steps:
            return
        for agent in trunc:
            trunc[agent] = True
            infos.setdefault(agent, {})["TimeLimit.truncated"] = True

    def step(self, actions: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
        """ """
        self._ticks_since_reset += 1
        if self._league_mode == "self_play":
            observations, rewards, term, trunc, infos = self._env.step(actions)
            self.agents = list(self._env.agents)
            self._apply_episode_truncation(trunc, infos)
            return observations, rewards, term, trunc, infos

        # single_learner mode.
        if self._frozen_predict is None:
            # Scripted opponent: the inner env was built with the scripted
            # archetype as an NPC seat, so the joint action passed to the inner
            # env is just retailer_0's action — the NPC fills its seat itself.
            joint = {"retailer_0": actions["retailer_0"]}
        else:
            seat_1_action = self._frozen_predict(
                self._last_obs_seat_1, episode_start=self._first_step_after_reset
            )
            joint = {
                "retailer_0": actions["retailer_0"],
                self._hidden_agent_id: seat_1_action,
            }

        inner_obs, inner_rewards, inner_term, inner_trunc, inner_infos = self._env.step(joint)
        # T-A5: any successful step() clears the first-step flag so the next
        # step() (within the same episode) gets ``episode_start=False``. The
        # flag is re-armed by ``reset()``. We flip AFTER the inner env's step()
        # so a step()-that-raises does not silently turn off the LSTM reset for
        # the next step() in the same episode (mid-episode failure semantics
        # are caller-owned; this just preserves the contract until the next
        # reset()).
        self._first_step_after_reset = False
        # Update the cached hidden-seat observation for the NEXT step's
        # checkpoint injection. In the scripted branch the inner env may or may
        # not return a retailer_1 key (it does NOT — n_learning_agents=1), so
        # the .get(...) defends without re-running any seam logic.
        self._last_obs_seat_1 = inner_obs.get(self._hidden_agent_id)
        # The wrapper hides retailer_1 — agents stays the PZ surface (single
        # learner). Use the inner env's agents intersected with our exposed
        # surface so a future agent-death contract is respected.
        self.agents = ["retailer_0"] if "retailer_0" in self._env.agents else []
        observations = {"retailer_0": inner_obs["retailer_0"]}
        rewards = {"retailer_0": inner_rewards["retailer_0"]}
        term = {"retailer_0": inner_term["retailer_0"]}
        trunc = {"retailer_0": inner_trunc["retailer_0"]}
        infos = {"retailer_0": inner_infos.get("retailer_0", {})}
        self._apply_episode_truncation(trunc, infos)
        return observations, rewards, term, trunc, infos

    def close(self) -> None:
        """Close the wrapped env (the league carries no resources of its own)."""
        self._env.close()


# --- Multi-round league training loop (Slice B T-B2) -------------------------


@dataclass(frozen=True)
class LeagueTrainingResult:
    """ """

    pool_dir: Path
    rounds_completed: int
    checkpoints: tuple[PoolMember, ...]
    learning_curves: tuple[tuple[tuple[int, float], ...], ...]
    seeds: tuple[int, int, int]
    train_steps_per_round: int
    win_rate_matrix_snapshot: WinRateMatrix | None = None
    max_episode_steps: int | None = None


def _single_learner_wrapped_factory_guard(_archetypes: tuple[str, ...]) -> Any:
    """Module-level guard the league wrapper falls back on under single_learner mode.

    The wrapper's ``__init__`` signature requires a ``wrapped_env_factory``
    even when ``league_mode="single_learner"`` is requested (the single-learner
    factory replaces it as the env-construction seam). Passing a function that
    raises on call surfaces a refactor that accidentally routes a self_play
    call path under single_learner; passing a stub that silently builds an env
    would mask such a bug. Lives at module scope so mypy --strict has a
    well-typed reference (the inline lambda alternative produces a
    Callable[[tuple[str, ...]], Any] mypy can't infer without an annotation).
    """
    raise AssertionError(
        "wrapped_env_factory must not be invoked under league_mode='single_learner'; "
        "single_learner_env_factory is the construction seam in that mode"
    )


@dataclass(frozen=True)
class _MatrixAwareDistribution:
    """Per-round proxy that injects the win-rate matrix + learner_id into ``sample``.

    Slice C.1 T-C1-RUNLOOP private helper. The :class:`LeagueEnvWrapper`'s
    ``self._distribution.sample(rng, pool)`` call site does NOT pass the
    Slice C.1 :class:`WinRateMatrix` + ``learner_id`` kwargs (the Slice B
    wrapper contract is preserved unchanged). The training loop wraps the
    caller's distribution with one of these proxies at the top of each round
    so the matrix + learner_id are forwarded transparently — uniform /
    latest-n distributions ignore them via the T-C1-PFSP Protocol extension;
    PFSP consumes them.

    Not exported; the proxy is a runtime-only adapter that satisfies the
    :class:`OpponentDistribution` Protocol structurally.
    """

    inner: OpponentDistribution
    win_rate_matrix: WinRateMatrix
    learner_id: str | None

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        # The Protocol extension accepts callers' explicit kwargs but the
        # round-loop is the authoritative source — we override with the
        # captured pair so a wrapping caller can't accidentally elide them.
        del win_rate_matrix, learner_id
        return self.inner.sample(
            rng,
            pool,
            win_rate_matrix=self.win_rate_matrix,
            learner_id=self.learner_id,
        )


def _load_warm_start_parameters(
    model: Any, *, source_checkpoint: Path, policy_kind: Literal["mlp", "lstm"]
) -> None:
    """ """
    if policy_kind == "lstm":
        # Mirrors FrozenPolicyPool.materialize's own LSTM branch: by the time
        # warm_start reaches this function, `model` itself was already built with
        # policy_kind="lstm" (i.e. _make_shared_ppo already required sb3-contrib
        # to construct THIS round's model), so the import below is guaranteed to
        # succeed — no ImportError-with-a-helpful-message wrapping needed here.
        from sb3_contrib import RecurrentPPO  # noqa: PLC0415 - lazy by design

        source_model: Any = RecurrentPPO.load(source_checkpoint, device=model.device)
    else:
        from stable_baselines3 import PPO  # noqa: PLC0415 - lazy by design

        source_model = PPO.load(source_checkpoint, device=model.device)
    model.policy.load_state_dict(source_model.policy.state_dict())


def run_league_training(
    initial_pool: FrozenPolicyPool,
    n_rounds: int,
    distribution_factory: Callable[[FrozenPolicyPool], OpponentDistribution],
    train_steps_per_round: int,
    seeds: tuple[int, int, int],
    *,
    scenario_distribution: ScenarioDistribution | None = None,
    config: CoreConfig | None = None,
    n_learning_agents: int = 1,
    pfsp_alpha: float = 2.0,
    round_end_eval_episodes: int = 5,
    distribution: OpponentDistribution | None = None,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
    home_regions: tuple[int, ...] | None = None,
    mask_expansion: bool = False,
    reward_mode: str | None = None,
    max_episode_steps: int | None = None,
    warm_start: bool = False,
) -> LeagueTrainingResult:
    """ """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")

    # Lazy heavy imports — keeps module import cheap and aligned with the
    # rest of the harness's bridge code (parallel_gate, league_bridge_smoke).
    import supersuit as ss

    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.harness.curriculum import CurriculumEnvWrapper
    from retail_simulator.harness.parallel_gate import (
        _make_curve_callback,
        _make_shared_ppo,
        competition_config,
    )

    if scenario_distribution is not None and home_regions is not None:
        raise ValueError(
            "run_league_training: scenario_distribution and home_regions cannot "
            "both be given — CurriculumEnvWrapper has no seating seam of its own"
        )
    if scenario_distribution is not None and mask_expansion:
        raise ValueError(
            "run_league_training: scenario_distribution and mask_expansion cannot "
            "both be given — CurriculumEnvWrapper has no masking seam of its own"
        )

    league_seed, curriculum_seed, core_seed = seeds
    # RLB-8a: computed ONCE and reused everywhere this function needs "the
    # config this run actually trains on" — the single-learner env factory
    # below AND the provenance hash recorded on the pool just after. Two
    # independent `config if config is not None else competition_config()`
    # derivations would risk silently diverging if either one's fallback ever
    # changed without the other noticing.
    effective_config = config if config is not None else competition_config()

    master_seq = np.random.SeedSequence(league_seed)
    round_seqs = master_seq.spawn(n_rounds)
    per_round_league_seeds: tuple[int, ...] = tuple(
        int(round_seqs[k].generate_state(1, dtype=np.uint32)[0]) for k in range(n_rounds)
    )

    # Slice C.1 T-C1-RUNLOOP — load (or initialize) the persistent win-rate
    # matrix. The fresh-pool case (no matrix on disk) yields an empty matrix;
    # the resume case (a prior run wrote the file) restores the cumulative
    # head-to-head record. F-WIN-PERSIST=A: ``<pool_dir>/win_rate_matrix.json``.
    win_rate_matrix_path = initial_pool.pool_dir / "win_rate_matrix.json"
    try:
        win_rate_matrix = WinRateMatrix.load(win_rate_matrix_path)
    except FileNotFoundError:
        win_rate_matrix = WinRateMatrix()

    # RLB-8a: record the seating provenance on the pool ONCE, before round 0,
    # regardless of n_rounds — see FrozenPolicyPool.record_arena_seating's own
    # docstring for why this always writes (even when there is nothing to
    # record) rather than only writing when home_regions is given. The hash +
    # knobs are computed only when there is a seating to record (matches the
    # summary-JSON "null for calibrated" convention scripts/phase4_rl_confirm.py's
    # own training block follows).
    #
    # Senior-review M3: arena_region_table_hash reads ONLY config.demand.regions
    # — it cannot discriminate two arena configs sharing a region table but
    # differing in the six economics knobs below: the three the Phase-6 lock
    # fixes under Amendment 2 (signed 2026-09-14) — a FUTURE amendment could
    # still move them
    # (overhead_reference_stores/overhead_exponent/provided_capacity_per_store),
    # plus (senior-review fix round, credit-line-and-rebalance) credit_limit/
    # overdraft_rate — the two knobs that actually distinguish "arena" from
    # "arena-balanced" today (see phase4_rl_confirm.py's --resume
    # disambiguation, which reads credit_limit specifically because it is a
    # knob Amendment 2 never touched, unlike
    # provided_capacity_per_store) — plus (arm-employee-satisfaction, then
    # reverted by writeoff-revert-0.05) writeoff_frac_max, which happens to
    # equal the lock's own value again today but is still recorded for the
    # same future-proofing reason as credit_limit/overdraft_rate: a future
    # amendment could move it independently, exactly like
    # :attr:`arena_lock_knobs` (below) explains at more length.
    # Reading them straight off effective_config.wage/.warehouse records
    # reality regardless of which arena config variant actually produced them.
    if home_regions is not None:
        from retail_simulator.harness.arena import (
            arena_region_table_hash as _arena_region_table_hash,
        )

        region_table_hash = _arena_region_table_hash(effective_config)
        arena_lock_knobs: dict[str, float] | None = {
            "overhead_reference_stores": effective_config.wage.overhead_reference_stores,
            "overhead_exponent": effective_config.wage.overhead_exponent,
            "provided_capacity_per_store": effective_config.warehouse.provided_capacity_per_store,
            "credit_limit": effective_config.wage.credit_limit,
            "overdraft_rate": effective_config.wage.overdraft_rate,
            "writeoff_frac_max": effective_config.wage.writeoff_frac_max,
        }
    else:
        region_table_hash = None
        arena_lock_knobs = None
    initial_pool.record_arena_seating(
        home_regions=home_regions,
        arena_region_table_hash=region_table_hash,
        arena_lock_knobs=arena_lock_knobs,
    )
    # The H5 fix round: recorded UNCONDITIONALLY (even when
    # `reward_mode` is None), for the same reason record_arena_seating above
    # always writes -- see FrozenPolicyPool.record_reward_mode's own docstring
    # for why "None means nothing notable" is decided by the CALLER, not here.
    initial_pool.record_reward_mode(reward_mode)

    def _single_learner_factory(n_learn: int, archetypes: tuple[str, ...]) -> Any:
        """ """
        if scenario_distribution is not None:
            return CurriculumEnvWrapper(
                scenario_distribution,
                n_learning_agents=n_learn,
                npc_archetypes=archetypes,
                curriculum_seed=curriculum_seed,
                core_seed=core_seed,
            )
        return RetailParallelEnv(
            effective_config,
            n_learning_agents=n_learn,
            npc_archetypes=archetypes,
            seed=core_seed,
            home_regions=home_regions,
            # S8: seat 0 is ALWAYS the trainee under single_learner mode (never the
            # frozen opponent, scripted or hidden) — see the docstring above.
            mask_expansion_seats=(0,) if mask_expansion else (),
        )

    def _build_bridged_vec(wrapper: LeagueEnvWrapper) -> Any:
        """Run the SuperSuit bridge over an EXISTING wrapper (mirrors build_parallel_vec_env).

        The 0.5 bridge wraps a freshly-built env; here we must bridge the
        already-constructed :class:`LeagueEnvWrapper` (so the league_rng's
        state + the round-specific distribution are preserved). The call
        shape is verbatim from :func:`build_parallel_vec_env`'s body —
        ``pettingzoo_env_to_vec_env_v1`` then ``concat_vec_envs_v1``. The
        ``render_mode`` guard mirrors the parallel-gate path (SuperSuit's
        MarkovVectorEnv reads ``unwrapped.render_mode``).
        """
        if not hasattr(wrapper, "render_mode") or wrapper.render_mode is None:
            wrapper.render_mode = None
        vec_pz = ss.pettingzoo_env_to_vec_env_v1(wrapper)
        return ss.concat_vec_envs_v1(
            vec_pz, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
        )

    learning_curves: list[tuple[tuple[int, float], ...]] = []

    # ---- Round 0: build the random-init snapshot WITHOUT calling .learn ----
    #
    # The throw-away wrapper exists ONLY to satisfy PPO's vec-env requirement;
    # we close it immediately after PPO is constructed. Sampling the first
    # scripted member directly (rather than reading distribution.sample())
    # keeps round-0 deterministic and independent of which distribution the
    # caller's factory will produce on round-1.
    initial_members = initial_pool.members()
    if not initial_members:
        raise ValueError(
            "initial_pool must contain at least one member to host the round-0 "
            "PPO constructor; use make_seeded_pool(...) to seed the pool"
        )
    first_scripted: PoolMember | None = next(
        (m for m in initial_members if m.kind == "scripted"), None
    )
    if first_scripted is None:
        raise ValueError(
            "initial_pool must contain at least one scripted member to host the "
            "round-0 PPO constructor; the F5-A seeded pool provides this"
        )

    # A throw-away distribution that always returns the first scripted member
    # — we never advance the league_rng more than the wrapper's __init__
    # snapshot draw needs, and the wrapper's reset() inside SB3 never fires
    # (no .learn() in round-0).
    @dataclass(frozen=True)
    class _FixedMemberDistribution:
        member: PoolMember

        def sample(
            self,
            _rng: np.random.Generator,
            _pool: FrozenPolicyPool,
            *,
            win_rate_matrix: WinRateMatrix | None = None,
            learner_id: str | None = None,
        ) -> PoolMember:
            # T-C1-PFSP Protocol extension kwargs accepted for compatibility;
            # the fixed-member shim ignores them (round-0 has no win-rate data).
            del win_rate_matrix, learner_id
            return self.member

    round_zero_distribution: OpponentDistribution = _FixedMemberDistribution(first_scripted)
    round_zero_wrapper = LeagueEnvWrapper(
        wrapped_env_factory=_single_learner_wrapped_factory_guard,
        pool=initial_pool,
        distribution=round_zero_distribution,
        # Slice C.1 T-C1-RUNLOOP — per-round spawn-derived seed (round 0).
        league_seed=per_round_league_seeds[0],
        league_mode="single_learner",
        single_learner_env_factory=_single_learner_factory,
        max_episode_steps=max_episode_steps,
        episode_seed_base=core_seed,
    )
    try:
        vec_throw = _build_bridged_vec(round_zero_wrapper)
        try:
            model_zero = _make_shared_ppo(
                vec_throw,
                seed=core_seed,
                policy_kind=policy_kind,
                ent_coef=ent_coef,
                gae_lambda=gae_lambda,
            )
            initial_pool.add_checkpoint(
                model_zero,
                round_index=0,
                seed=core_seed,
                policy_kind=policy_kind,
            )
            learning_curves.append(())
        finally:
            vec_throw.close()
    finally:
        round_zero_wrapper.close()

    # ---- Rounds 1..n_rounds-1: train then snapshot ------------------------
    # ``pfsp_alpha`` is informational for callers that pass ``distribution``
    # directly (the PFSP instance carries its own alpha); reserved for the
    # future CLI-driven path that auto-builds a PFSP distribution.
    del pfsp_alpha
    for round_index in range(1, n_rounds):
        # Slice C.1 T-C1-RUNLOOP — distribution selection. When the caller
        # supplied an explicit ``distribution`` (the PFSP opt-in path) use
        # it directly; otherwise rebuild via ``distribution_factory`` at the
        # TOP of each round so the pool GROWTH from the prior round's
        # add_checkpoint is visible to this round's opponent draws — the
        # AlphaStar pool-grows-each-round semantics (preserves Slice B).
        base_distribution: OpponentDistribution = (
            distribution if distribution is not None else distribution_factory(initial_pool)
        )
        # Slice C.1 T-C1-RUNLOOP — thread the persistent win-rate matrix +
        # the prior-round learner's id into the distribution's ``sample(...)``
        # call. ``LeagueEnvWrapper.sample(rng, pool)`` does NOT pass these
        # kwargs today (Slice B contract preserved); we wrap with a small
        # per-round proxy that forwards them. Uniform / Latest-N ignore the
        # kwargs (T-C1-PFSP Protocol extension); PFSP consumes them.
        #
        # The "learner_id" the matrix is keyed against is the MOST RECENTLY
        # APPENDED checkpoint at the top of round K — i.e., round-(K-1)'s
        # snapshot. (Round-K's own checkpoint doesn't exist yet at this
        # point in the loop.) This is the matrix row PFSP weights against
        # to pick round-K's opponent — "what do we know about how the most-
        # recent learner fares against each pool member?".
        prior_learner_members = [m for m in initial_pool.members() if m.kind == "checkpoint"]
        prior_learner_id = prior_learner_members[-1].member_id if prior_learner_members else None
        round_distribution: OpponentDistribution = _MatrixAwareDistribution(
            inner=base_distribution,
            win_rate_matrix=win_rate_matrix,
            learner_id=prior_learner_id,
        )
        wrapper = LeagueEnvWrapper(
            wrapped_env_factory=_single_learner_wrapped_factory_guard,
            pool=initial_pool,
            distribution=round_distribution,
            # Slice C.1 T-C1-RUNLOOP — per-round spawn-derived seed (M2).
            league_seed=per_round_league_seeds[round_index],
            league_mode="single_learner",
            single_learner_env_factory=_single_learner_factory,
            max_episode_steps=max_episode_steps,
            episode_seed_base=core_seed,
        )
        try:
            vec = _build_bridged_vec(wrapper)
            try:
                curve: list[tuple[int, float]] = []
                callback = _make_curve_callback(curve)
                model = _make_shared_ppo(
                    vec,
                    seed=core_seed,
                    policy_kind=policy_kind,
                    ent_coef=ent_coef,
                    gae_lambda=gae_lambda,
                )
                if warm_start and round_index >= 2:
                    warm_start_source = prior_learner_members[-1]
                    warm_start_checkpoint_path = warm_start_source.checkpoint_path
                    if warm_start_checkpoint_path is None:
                        # Defensive: PoolMember.__post_init__ guarantees non-None
                        # for kind=="checkpoint", but typing infers Path | None --
                        # mirrors FrozenPolicyPool.materialize's own defensive
                        # check for the identical invariant.
                        raise RuntimeError(
                            f"warm start: checkpoint PoolMember "
                            f"{warm_start_source.member_id!r} has no checkpoint_path; "
                            f"the dataclass invariant was bypassed"
                        )
                    _load_warm_start_parameters(
                        model,
                        source_checkpoint=initial_pool.pool_dir / warm_start_checkpoint_path,
                        policy_kind=policy_kind,
                    )
                    logger.info(
                        "warm start: round %d policy initialized from %s",
                        round_index,
                        warm_start_source.member_id,
                    )
                model.learn(
                    total_timesteps=train_steps_per_round,
                    callback=callback,
                    progress_bar=False,
                )
                round_k_member = initial_pool.add_checkpoint(
                    model,
                    round_index=round_index,
                    seed=core_seed,
                    policy_kind=policy_kind,
                )
                learning_curves.append(tuple(curve))
            finally:
                vec.close()
        finally:
            wrapper.close()

        # Slice C.1 T-C1-RUNLOOP — round-end head-to-head evaluation. Runs
        # for ALL distributions (uniform/latest-n ignore the matrix; PFSP
        # consumes it on the next round's draw). The matrix is persisted
        # atomically after each fold so a resumed run sees the cumulative
        # head-to-head record. Per-round seed offset (core_seed + round_index)
        # keeps the eval reproducible AND distinct from the per-round-pair
        # offsets `_run_round_end_evaluation` derives internally.
        round_end_results = _run_round_end_evaluation(
            initial_pool,
            round_k_member,
            n_episodes=round_end_eval_episodes,
            config=config,
            seed=core_seed + round_index,
            home_regions=home_regions,
        )
        for opp_id, entry in round_end_results.items():
            # `record(...)` adds one outcome per call; iterate over the
            # bucketed wins/losses to fold the per-pair result into the
            # cumulative matrix. The slightly-awkward shape (wins/total
            # vs per-outcome calls) is the documented bridge between the
            # round-end evaluator's bucketed return and the matrix's
            # incremental API; a `record_outcomes(...)` bulk helper would
            # be a future ergonomics win but is out of scope here.
            for _ in range(entry.wins):
                win_rate_matrix.record(
                    learner_id=round_k_member.member_id,
                    opponent_id=opp_id,
                    learner_won=True,
                )
            for _ in range(entry.total - entry.wins):
                win_rate_matrix.record(
                    learner_id=round_k_member.member_id,
                    opponent_id=opp_id,
                    learner_won=False,
                )
        win_rate_matrix.save(win_rate_matrix_path)

    checkpoints = tuple(m for m in initial_pool.members() if m.kind == "checkpoint")
    # n_learning_agents is part of the public signature for forward-compat
    # (an AlphaStar variant might run a 2-learner shared-policy round), but
    # the current loop pins to the single-learner shape per the AlphaStar
    # canonical claim — the parameter is reserved, not yet consumed.
    del n_learning_agents
    return LeagueTrainingResult(
        pool_dir=initial_pool.pool_dir,
        rounds_completed=n_rounds,
        checkpoints=checkpoints,
        learning_curves=tuple(learning_curves),
        seeds=seeds,
        train_steps_per_round=train_steps_per_round,
        win_rate_matrix_snapshot=win_rate_matrix,
        max_episode_steps=max_episode_steps,
    )


@dataclass(frozen=True)
class PoolSnapshotImprovementReport:
    """ """

    round_n_mean: float
    round_1_mean: float
    round_n_std: float
    round_1_std: float
    margin: float
    threshold: float
    beats_round_1: bool
    n_episodes: int
    per_round_means: tuple[tuple[int, float], ...]


# RLB-3 (battle scoreboard): the four canonical head-to-head-scored components —
# mirrors RewardComponents'/weighted_objective's field order
# (retail_simulator.core.reward). The battle-scoreboard sink-filling code below
# iterates this fixed tuple.
HEAD_TO_HEAD_COMPONENTS: tuple[str, ...] = ("profit", "revenue", "market_share", "loyalty")

# RLB-3 senior-review follow-up (m5): the subset of HEAD_TO_HEAD_COMPONENTS whose
# per-episode reduction in _position_controlled_head_to_head is a PER-TICK MEAN
# (divide by n_ticks) rather than a raw SUM — a share/EMA is a per-tick quantity,
# summing it across ticks would not be meaningful. Named as its own constant (not
# a literal tuple inlined at the reduction site) so adding a future component to
# HEAD_TO_HEAD_COMPONENTS can't silently inherit the wrong reduction by accident.
PER_TICK_MEAN_COMPONENTS: tuple[str, ...] = ("market_share", "loyalty")


def _head_to_head_episode_scores(
    env: Any,
    cfg: Any,
    *,
    seed: int,
    predict_for_agent: dict[str, Callable[..., Any]],
    component_sink: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """ """
    from retail_simulator.harness.learnability import DEFAULT_MAX_EPISODE_STEPS
    from retail_simulator.harness.parallel_gate import (
        _components_from_info,
        weighted_objective,
    )

    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    if component_sink is not None:
        for aid in env.possible_agents:
            component_sink[aid] = {
                "profit": 0.0,
                "revenue": 0.0,
                "market_share": 0.0,
                "loyalty": 0.0,
            }
    is_first_step = True
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions = {
            aid: predict_for_agent[aid](observations[aid], episode_start=is_first_step)
            for aid in env.agents
        }
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
            if component_sink is not None:
                # RLB-3: a second, deliberately SEPARATE reconstruction from the
                # SAME info — kept apart from the score-computation line above so
                # that line stays byte-for-byte untouched.
                components = _components_from_info(infos[aid])
                component_sink[aid]["profit"] += components.profit
                component_sink[aid]["revenue"] += components.revenue
                component_sink[aid]["market_share"] += components.market_share
                component_sink[aid]["loyalty"] += components.loyalty
        is_first_step = False
    if component_sink is not None:
        for aid in env.possible_agents:
            component_sink[aid]["n_ticks"] = float(DEFAULT_MAX_EPISODE_STEPS)
    return totals


def _position_controlled_head_to_head(
    env: Any,
    cfg: Any,
    *,
    predict_a: Callable[..., Any],
    predict_b: Callable[..., Any],
    n_episodes: int,
    seed: int,
    env_factory: Callable[[], Any] | None = None,
    component_sink: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[float], list[float]]:
    """ """
    scores_a: list[float] = []
    scores_b: list[float] = []
    # When env_factory is provided we step on freshly-built envs each iteration;
    # we still need a stable agent-id list to drive the seat permutation. The
    # agent IDs are stable across env constructions (seats 0..N-1 named via
    # _agent_id), so reading from the passed env (when present) is equivalent
    # to reading from a freshly-built one. When env is None (the
    # reuse_env=False caller in evaluate_pool_snapshot_improvement), peek at a
    # freshly-built env once and close it just to learn the agent set.
    if env is not None:
        possible_agents = list(env.possible_agents)
    else:
        if env_factory is None:  # pragma: no cover - defensive; not reachable from callers
            raise ValueError(
                "_position_controlled_head_to_head: either env or env_factory must be provided"
            )
        peek_env = env_factory()
        try:
            possible_agents = list(peek_env.possible_agents)
        finally:
            peek_env.close()
    for episode in range(n_episodes):
        episode_seed = seed + episode
        a_seat_vals: list[float] = []
        b_seat_vals: list[float] = []
        # RLB-3: one (a_seat, inner component sink) pair per seat rotation this
        # episode, collected in step with a_seat_vals/b_seat_vals so the
        # per-component means below mirror scores_a/scores_b exactly.
        rotation_component_sinks: list[tuple[str, dict[str, dict[str, float]]]] = []
        for a_seat in possible_agents:
            predict_for_agent: dict[str, Callable[..., Any]] = {
                aid: (predict_a if aid == a_seat else predict_b) for aid in possible_agents
            }
            inner_component_sink: dict[str, dict[str, float]] | None = (
                {} if component_sink is not None else None
            )
            if env_factory is None:
                totals = _head_to_head_episode_scores(
                    env,
                    cfg,
                    seed=episode_seed,
                    predict_for_agent=predict_for_agent,
                    component_sink=inner_component_sink,
                )
            else:
                # Concern-B path: build a fresh env, reset+step, then close.
                episode_env = env_factory()
                try:
                    totals = _head_to_head_episode_scores(
                        episode_env,
                        cfg,
                        seed=episode_seed,
                        predict_for_agent=predict_for_agent,
                        component_sink=inner_component_sink,
                    )
                finally:
                    episode_env.close()
            a_seat_vals.append(totals[a_seat])
            b_seat_vals.extend(v for aid, v in totals.items() if aid != a_seat)
            if inner_component_sink is not None:
                rotation_component_sinks.append((a_seat, inner_component_sink))
        scores_a.append(float(np.mean(a_seat_vals)))
        scores_b.append(float(np.mean(b_seat_vals)))
        if component_sink is not None:
            for name in HEAD_TO_HEAD_COMPONENTS:
                component_a_vals: list[float] = []
                component_b_vals: list[float] = []
                for rotation_a_seat, inner_sink in rotation_component_sinks:
                    for aid, agent_totals in inner_sink.items():
                        value = (
                            agent_totals[name] / agent_totals["n_ticks"]
                            if name in PER_TICK_MEAN_COMPONENTS
                            else agent_totals[name]
                        )
                        if aid == rotation_a_seat:
                            component_a_vals.append(value)
                        else:
                            component_b_vals.append(value)
                a_side_mean = float(np.mean(component_a_vals))
                b_side_mean = float(np.mean(component_b_vals))
                component_sink.setdefault(name, []).append((a_side_mean, b_side_mean))
    return scores_a, scores_b


_PER_ROUND_CURVE_EPISODES: int = 3


def evaluate_pool_snapshot_improvement(
    pool: FrozenPolicyPool,
    training_result: LeagueTrainingResult,
    *,
    n_eval_episodes: int = 20,
    config: CoreConfig | None = None,
    seed: int = 0,
    reuse_env: bool = True,
    home_regions: tuple[int, ...] | None = None,
) -> PoolSnapshotImprovementReport:
    """ """
    # Lazy import — competition_config / RetailParallelEnv pull pettingzoo, which
    # the league module otherwise avoids until call time.
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.harness.learnability import GATE_SIGMA_MULTIPLE
    from retail_simulator.harness.parallel_gate import competition_config

    checkpoints = training_result.checkpoints
    if len(checkpoints) < 2:
        raise ValueError(
            "evaluate_pool_snapshot_improvement requires at least 2 checkpoints "
            "(round-1 and round-N) to compare; "
            f"got {len(checkpoints)} checkpoint(s) in training_result"
        )

    # Identify round-1 (the FIRST trained checkpoint, the one trained against
    # the round-0 random-init pool) and round-N (max round_index — the latest
    # learner snapshot). Round-0 is the random-init snapshot; round-1 is the
    # FIRST .learn-ed checkpoint per run_league_training's contract.
    by_round = {member.round_index: member for member in checkpoints}
    if 1 not in by_round:
        raise ValueError(
            "evaluate_pool_snapshot_improvement requires a checkpoint at round_index=1 "
            f"(the first trained snapshot); got round_indexes {sorted(by_round)}"
        )
    round_1_member = by_round[1]
    round_n_member = checkpoints[-1]  # checkpoints are appended in round order

    effective_config = config if config is not None else competition_config()
    cfg = effective_config.reward

    predict_round_n = pool.materialize(round_n_member, env=None)
    predict_round_1 = pool.materialize(round_1_member, env=None)

    # Either build the eval env ONCE and reuse it across all episodes (Slice B
    # default — the same pattern evaluate_head_to_head follows; per-seat
    # info['components'] is read directly from the raw parallel env, NOT
    # through the SuperSuit bridge) OR build a fresh env per inner episode
    # (Slice C T-C1-REUSE / Concern-B regression path). The byte-identity
    # contract: the two paths produce field-equal reports.
    def _build_eval_env() -> Any:
        return RetailParallelEnv(
            effective_config, n_learning_agents=2, seed=seed, home_regions=home_regions
        )

    if reuse_env:
        shared_env: Any | None = _build_eval_env()
        primitive_env_factory: Callable[[], Any] | None = None
    else:
        shared_env = None
        primitive_env_factory = _build_eval_env
    try:
        round_n_scores, round_1_scores = _position_controlled_head_to_head(
            shared_env,
            cfg,
            predict_a=predict_round_n,
            predict_b=predict_round_1,
            n_episodes=n_eval_episodes,
            seed=seed,
            env_factory=primitive_env_factory,
        )

        # Per-round means curve (REPORTED) — for each checkpoint k (in round-index
        # order), run a cheap mini-eval of round-k vs round-1, or of round-1 vs
        # round-0 when k == 1 (no peer-1 to compare with). Each entry is the
        # round-k side's mean score across _PER_ROUND_CURVE_EPISODES.
        # NOTE: we re-materialize per loop iteration; FrozenPolicyPool's load
        # cache means each checkpoint zip is read AT MOST ONCE per process.
        per_round: list[tuple[int, float]] = []
        for member in checkpoints:
            predict_k = pool.materialize(member, env=None)
            if member.round_index == 1:
                # round-1's peer is round-0 (the random-init snapshot — also a
                # checkpoint in the pool, persisted by run_league_training).
                round_0 = by_round.get(0)
                if round_0 is None:
                    # Defensive: a 1-round result might lack round-0 (the loop
                    # writes it; a hand-built pool may not). Compare round-1
                    # against itself in that case so the curve still reports
                    # a well-defined value.
                    peer_predict = predict_k
                else:
                    peer_predict = pool.materialize(round_0, env=None)
            else:
                peer_predict = predict_round_1
            k_scores, _peer_scores = _position_controlled_head_to_head(
                shared_env,
                cfg,
                predict_a=predict_k,
                predict_b=peer_predict,
                n_episodes=_PER_ROUND_CURVE_EPISODES,
                seed=seed + 10_000 + member.round_index,
                env_factory=primitive_env_factory,
            )
            per_round.append((member.round_index, float(np.mean(k_scores))))
    finally:
        if shared_env is not None:
            shared_env.close()

    round_n_mean = float(np.mean(round_n_scores))
    round_1_mean = float(np.mean(round_1_scores))
    round_n_std = float(np.std(round_n_scores))
    round_1_std = float(np.std(round_1_scores))
    margin = round_n_mean - round_1_mean
    threshold = float(GATE_SIGMA_MULTIPLE) * round_1_std
    beats_round_1 = margin >= threshold

    return PoolSnapshotImprovementReport(
        round_n_mean=round_n_mean,
        round_1_mean=round_1_mean,
        round_n_std=round_n_std,
        round_1_std=round_1_std,
        margin=margin,
        threshold=threshold,
        beats_round_1=beats_round_1,
        n_episodes=n_eval_episodes,
        per_round_means=tuple(per_round),
    )


# --- Round-end head-to-head evaluation (Slice C.1 T-C1-ROUNDEND) -------------


def _run_round_end_evaluation(
    pool: FrozenPolicyPool,
    learner_member: PoolMember,
    *,
    n_episodes: int = 5,
    config: "CoreConfig | None" = None,
    seed: int = 0,
    home_regions: tuple[int, ...] | None = None,
) -> dict[str, WinRateEntry]:
    """ """
    # Lazy imports — RetailParallelEnv pulls pettingzoo, kept out of module load.
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.harness.learnability import DEFAULT_MAX_EPISODE_STEPS
    from retail_simulator.harness.parallel_gate import (
        _components_from_info,
        competition_config,
        weighted_objective,
    )

    if learner_member.kind != "checkpoint":
        raise ValueError(
            f"_run_round_end_evaluation: learner_member must be a checkpoint, "
            f"got kind={learner_member.kind!r} (the learner is the current round's "
            f"trained policy; scripted members can only be opponents)"
        )

    opponents = tuple(m for m in pool.members() if m.member_id != learner_member.member_id)
    if not opponents:
        return {}

    effective_config = config if config is not None else competition_config()
    reward_cfg = effective_config.reward

    learner_predict = pool.materialize(learner_member, env=None)
    rng = np.random.default_rng(seed)
    results: dict[str, WinRateEntry] = {}

    for opp_idx, opp in enumerate(opponents):
        wins = 0
        if opp.kind in ("checkpoint", "archetype"):
            # Checkpoint-vs-checkpoint (or vs an H4 archetype): position-
            # controlled head-to-head — materialize works for both kinds.
            # Reuse the Slice B primitive so the scoring substrate
            # (weighted_objective on info['components'], per-seat
            # alternation) is byte-identical to evaluate_pool_snapshot_improvement.
            opp_predict = pool.materialize(opp, env=None)
            env = RetailParallelEnv(
                effective_config, n_learning_agents=2, seed=seed, home_regions=home_regions
            )
            try:
                # Per-opponent base seed: the same offsetting scheme
                # evaluate_pool_snapshot_improvement uses for its per-round curves
                # so concurrent eval loops don't collide on the same episode_seed.
                pair_seed = seed + opp_idx * 10_000
                learner_scores, opp_scores = _position_controlled_head_to_head(
                    env,
                    reward_cfg,
                    predict_a=learner_predict,
                    predict_b=opp_predict,
                    n_episodes=n_episodes,
                    seed=pair_seed,
                )
            finally:
                env.close()
            # One episode → one win-or-loss outcome (mean per-seat scores
            # compared; ties broken via the seeded rng-coin-flip).
            for learner_score, opp_score in zip(learner_scores, opp_scores, strict=True):
                if learner_score > opp_score:
                    wins += 1
                elif learner_score == opp_score:
                    if rng.random() < 0.5:
                        wins += 1
        else:
            # Scripted opponent: single-learner-mode env per episode.
            # A FRESH env per episode is required because each episode's
            # seed differs and the env is bound to its constructor seed
            # (mirrors the Concern-B reuse_env=False path's discipline:
            # any cross-episode env reuse must go through reset(seed=...),
            # which we already get from constructing fresh).
            for ep_idx in range(n_episodes):
                ep_seed = seed + opp_idx * 10_000 + ep_idx * 1_000
                env_sl = RetailParallelEnv(
                    effective_config,
                    n_learning_agents=1,
                    npc_archetypes=(opp.archetype_name,),
                    seed=ep_seed,
                    home_regions=home_regions,
                )
                try:
                    observations, _infos = env_sl.reset(seed=ep_seed)
                    learner_total = 0.0
                    is_first_step = True
                    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
                        learner_obs = observations["retailer_0"]
                        learner_action = learner_predict(learner_obs, episode_start=is_first_step)
                        joint = {"retailer_0": learner_action}
                        observations, _rewards, _term, _trunc, infos = env_sl.step(joint)
                        learner_total += weighted_objective(
                            _components_from_info(infos["retailer_0"]), reward_cfg
                        )
                        is_first_step = False
                finally:
                    env_sl.close()
                # Fallback win signal (documented in the docstring above):
                # the env does NOT surface the scripted NPC's per-seat
                # objective through the step dict (rewards/infos are keyed
                # only by possible_agents = ["retailer_0"]). "Learner wins
                # iff learner_score > 0" treats the scripted NPC as a
                # constant zero-objective baseline — non-degenerate enough
                # for PFSP's weight curve to differentiate strong vs weak
                # learners against the same scripted opponent. Ties at
                # exactly zero resolve via the seeded rng-coin-flip.
                if learner_total > 0.0:
                    wins += 1
                elif learner_total == 0.0:
                    if rng.random() < 0.5:
                        wins += 1
        results[opp.member_id] = WinRateEntry(wins=wins, total=n_episodes)

    return results


@dataclass(frozen=True)
class RealPpoGeneralizationDiagnostic:
    """ """

    mean_curriculum_score: float
    mean_fixed_score: float
    per_scenario_curriculum_means: tuple[float, ...]
    per_scenario_fixed_means: tuple[float, ...]
    margin: float
    n_scenarios: int
    n_episodes_per_scenario: int
    train_steps: int
    per_scenario_curriculum_std_rewards: tuple[float, ...] = ()
    per_scenario_fixed_std_rewards: tuple[float, ...] = ()


def _default_holdout_scenarios() -> tuple[CoreConfig, ...]:
    """ """
    from retail_simulator.harness.curriculum import (
        _VARIANT_B_REGION_0_MIX,
        _VARIANT_B_REGION_1_MIX,
        _VARIANT_C_REGION_0_MIX,
        _VARIANT_C_REGION_1_MIX,
        phase_1_4_six_segment_config,
    )

    # Mirror variant-B (region-1 mix in region 0; region-0 mix in region 1).
    mirror_b = phase_1_4_six_segment_config(
        region_0_mix=_VARIANT_B_REGION_1_MIX,
        region_1_mix=_VARIANT_B_REGION_0_MIX,
    )
    # Mirror variant-C (the symmetric mirroring of the bargain-leaning regime).
    mirror_c = phase_1_4_six_segment_config(
        region_0_mix=_VARIANT_C_REGION_1_MIX,
        region_1_mix=_VARIANT_C_REGION_0_MIX,
    )
    return (mirror_b, mirror_c)


# Holdout-scenario tuple the CLI's ``league generalization-diagnostic`` command
# defaults to. Built lazily on first access via :func:`_default_holdout_scenarios`.
# Kept as a module-level cached property pattern (a single tuple object the
# orchestrator can identity-compare for "did the caller override?") would also
# work; the function is the leaner shape and keeps imports cheap.
_DEFAULT_HOLDOUT_SCENARIOS: tuple[CoreConfig, ...] | None = None


def default_holdout_scenarios() -> tuple[CoreConfig, ...]:
    """Cached accessor for the seed held-out scenarios (see :func:`_default_holdout_scenarios`).

    The cache is populated on first call so the 1.4 helpers (and pettingzoo
    transitively) are not pulled at module import time. Subsequent calls return
    the SAME tuple instance for identity-stable comparisons.
    """
    global _DEFAULT_HOLDOUT_SCENARIOS
    if _DEFAULT_HOLDOUT_SCENARIOS is None:
        _DEFAULT_HOLDOUT_SCENARIOS = _default_holdout_scenarios()
    return _DEFAULT_HOLDOUT_SCENARIOS


def run_real_ppo_generalization_diagnostic(
    curriculum: ScenarioDistribution,
    holdout_scenarios: tuple[CoreConfig, ...],
    train_steps: int,
    seeds: tuple[int, int],
    *,
    n_eval_episodes_per_scenario: int = 5,
    fixed_config: CoreConfig | None = None,
    n_learning_agents: int = 2,
) -> RealPpoGeneralizationDiagnostic:
    """ """
    if not holdout_scenarios:
        raise ValueError(
            "run_real_ppo_generalization_diagnostic requires at least one held-out "
            "scenario; the diagnostic measures per-scenario means"
        )
    if train_steps < 1:
        raise ValueError(
            f"train_steps must be >= 1, got {train_steps}; "
            "at least one timestep is needed to train either model"
        )
    if n_eval_episodes_per_scenario < 1:
        raise ValueError(
            f"n_eval_episodes_per_scenario must be >= 1, got {n_eval_episodes_per_scenario}; "
            "at least one evaluation episode per scenario is needed for a mean"
        )

    # Lazy heavy imports — keeps module import cheap and aligned with the rest of
    # the harness's bridge code (parallel_gate, league_bridge_smoke).
    from retail_simulator.harness.curriculum import evaluate_generalization
    from retail_simulator.harness.parallel_gate import (
        _make_shared_ppo,
        build_parallel_vec_env,
        competition_config,
    )

    curriculum_seed, core_seed = seeds

    # --- Train the curriculum-PPO -------------------------------------------
    # The build_parallel_vec_env path applies the F5-A CurriculumEnvWrapper
    # around the parallel env (curriculum draws are training-time, harness-only;
    # the schema is UNCHANGED so the bridged Box is identical to the fixed run).
    vec_curric = build_parallel_vec_env(
        n_learning_agents=n_learning_agents,
        seed=core_seed,
        curriculum=curriculum,
        curriculum_seed=curriculum_seed,
    )
    try:
        curriculum_model = _make_shared_ppo(vec_curric, seed=core_seed)
        curriculum_model.learn(total_timesteps=train_steps, progress_bar=False)
    finally:
        vec_curric.close()

    # --- Train the fixed-PPO -------------------------------------------------
    fixed_config_effective = fixed_config if fixed_config is not None else competition_config()
    vec_fixed = build_parallel_vec_env(
        config=fixed_config_effective,
        n_learning_agents=n_learning_agents,
        seed=core_seed,
    )
    try:
        fixed_model = _make_shared_ppo(vec_fixed, seed=core_seed)
        fixed_model.learn(total_timesteps=train_steps, progress_bar=False)
    finally:
        vec_fixed.close()

    holdout_tuple = tuple(holdout_scenarios)
    report = evaluate_generalization(
        curriculum_model,
        fixed_model,
        holdout_tuple,
        episodes_per_scenario=n_eval_episodes_per_scenario,
        n_learning_agents=n_learning_agents,
        seed=core_seed,
    )

    return RealPpoGeneralizationDiagnostic(
        mean_curriculum_score=report.mean_curriculum_score,
        mean_fixed_score=report.mean_fixed_score,
        per_scenario_curriculum_means=report.per_scenario_curriculum_means,
        per_scenario_fixed_means=report.per_scenario_fixed_means,
        margin=report.margin,
        n_scenarios=report.n_scenarios,
        n_episodes_per_scenario=report.n_episodes_per_scenario,
        train_steps=int(train_steps),
    )


__all__ = [
    "DEFAULT_LEAGUE_NAME",
    "DEFAULT_POOL_SEED_ARCHETYPES",
    "HEAD_TO_HEAD_COMPONENTS",
    "KNOWN_LEAGUES",
    "MANIFEST_VERSION",
    "PER_TICK_MEAN_COMPONENTS",
    "WIN_RATE_MATRIX_VERSION",
    "FrozenPolicyPool",
    "LatestNOpponentDistribution",
    "LeagueEnvWrapper",
    "LeagueTrainingResult",
    "OpponentDistribution",
    "PFSPOpponentDistribution",
    "PoolMember",
    "PoolSnapshotImprovementReport",
    "RealPpoGeneralizationDiagnostic",
    "UniformOpponentDistribution",
    "WinRateEntry",
    "WinRateMatrix",
    "build_league",
    "default_holdout_scenarios",
    "evaluate_pool_snapshot_improvement",
    "known_leagues",
    "make_seeded_pool",
    "run_league_training",
    "run_real_ppo_generalization_diagnostic",
]
