"""Margin-based competition ladder — the rating core (Phase 2.2 Slice 1).

What this module computes:
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np

from retail_simulator.core.config import CoreConfig
from retail_simulator.harness.geo_archetypes import GEO_ARCHETYPE_LABELS, geo_archetypes
from retail_simulator.harness.spike_archetypes import (
    AGGRESSOR,
    LEAN,
    PATIENT,
    ArchetypePredict,
    calibrated_archetypes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARCHETYPE_SPEC_PREFIX",
    "CALIBRATED_ENV_LABEL",
    "DEFAULT_ENV_LABEL",
    "LADDER_ARCHETYPE_LABELS",
    "LADDER_SPIKE_ARCHETYPE_LABELS",
    "LadderEntry",
    "LadderResult",
    "PairwiseMargin",
    "ParticipantKind",
    "ParticipantSpec",
    "RANDOM_SPEC_PREFIX",
    "TournamentSummary",
    "parse_participant_spec",
    "run_ladder",
]

DEFAULT_ENV_LABEL = "default (competition)"
CALIBRATED_ENV_LABEL = "redesign (phase-4 calibrated)"

# Participant-spec grammar (RLB-2 mixed-field ladder): a bare string is still a
# checkpoint zip path; these two prefixes select a scripted or random participant
# instead. See :func:`parse_participant_spec`.
ARCHETYPE_SPEC_PREFIX = "archetype:"
RANDOM_SPEC_PREFIX = "random:"

# The calibrated three-archetype (spike, open-loop) roster's own labels. Imported
# from spike_archetypes (core-only: encoding/schema reads, no envs/RL) so this
# stays a module-TOP import without pulling sb3. Kept as its OWN name (RLB-8b) —
# distinct from the union below — so a caller that specifically needs "the
# world-agnostic open-loop family" (e.g. a fixed-expansion_choice legality check)
# never has to re-derive it by subtracting :data:`GEO_ARCHETYPE_LABELS` back out.
LADDER_SPIKE_ARCHETYPE_LABELS = (LEAN, AGGRESSOR, PATIENT)

# The archetype names ``archetype:<name>`` accepts — RLB-8b widens this from the
# spike-only three to the UNION of the spike family above and the four closed-loop
# geo archetypes (:data:`~retail_simulator.harness.geo_archetypes.
# GEO_ARCHETYPE_LABELS`, imported rather than re-listed so the two can't drift).
# Every PRE-EXISTING caller that means "the spike three specifically" (e.g.
# ``scripts/gen_board_artifacts.py``'s default-env board, which has no region
# table to seat a geo archetype on) must import :data:`LADDER_SPIKE_ARCHETYPE_LABELS`
# instead of this union — see that module's own comment.
LADDER_ARCHETYPE_LABELS = LADDER_SPIKE_ARCHETYPE_LABELS + GEO_ARCHETYPE_LABELS

ParticipantKind = Literal["checkpoint", "archetype", "random"]


@dataclass(frozen=True)
class PairwiseMargin:
    """ """

    label_a: str
    label_b: str
    per_seed_margins: tuple[float, ...]
    mean_margin: float
    margin_se: float
    ci_low: float | None = None
    ci_high: float | None = None
    component_mean_margins: dict[str, float] | None = None
    component_ci: dict[str, tuple[float, float]] | None = None
    component_per_seed_margins: dict[str, tuple[float, ...]] | None = None


@dataclass(frozen=True)
class LadderEntry:
    """One ranked participant: its field-margin, bootstrap CI, and dense rank.

    ``kind`` (RLB-2) records the participant's spec kind — ``"checkpoint"``,
    ``"archetype"``, or ``"random"`` (:data:`ParticipantKind`). For a non-checkpoint
    participant, ``checkpoint_path`` carries its spec string verbatim (e.g.
    ``"archetype:lean"``), not a real zip path. Defaults to ``"checkpoint"`` so a
    pre-RLB-2 caller's serialized entries stay byte-identical.

    ``component_field_margins`` (RLB-3, battle scoreboard) is this participant's
    mean ORIENTED per-component margin vs the rest of the field (mirrors
    ``field_margin`` but per :data:`~retail_simulator.harness.league.
    HEAD_TO_HEAD_COMPONENTS` name) — ``None`` when ANY pair in the ladder lacks
    component data (all-or-nothing: a single rollout path either supplies
    components for every pair or none). Defaults to ``None`` so a pre-RLB-3
    caller's entries stay byte-identical.
    """

    label: str
    checkpoint_path: str
    field_margin: float
    ci_low: float
    ci_high: float
    rank: int
    kind: ParticipantKind = "checkpoint"
    component_field_margins: dict[str, float] | None = None


@dataclass(frozen=True)
class TournamentSummary:
    """The dominance/cycle/Condorcet read of the field's OWN score margins (RLB-3).

    Built from THIS ladder's pairwise score margins + CIs (reuses
    :func:`retail_simulator.harness.spike._measure_tournament` — no
    re-implementation; mirrors ``scripts/phase4_rl_confirm.py``'s ``measure()``).

    * ``dominance`` — ``{label: (labels it beats, i.e. mean_margin > 0)}``.
    * ``n_cyclic_triples`` — count of 3-participant subsets whose sub-tournament is
      a cycle (not transitive); ``total_triples`` is ``C(n, 3)`` (``0`` below 3
      participants) — the denominator for reading the count.
    * ``has_condorcet_winner``/``condorcet_winner`` — whether some participant beats
      EVERY other, and who.
    * ``decisive_cycle`` — True iff the realized cycle's every edge has a bootstrap
      CI excluding 0 (a statistically real win, not noise); False when the field is
      transitive (nothing to be decisive about). **First-triple-only (mirrors
      ``harness.spike._measure_tournament`` exactly):** for a field with MORE THAN
      ONE cyclic triple, this reflects ONLY the first cyclic triple encountered in
      ``itertools.combinations(labels, 3)`` order — never an aggregate (AND/OR) over
      every cyclic triple, and never the LAST one. In a mixed field
      (``aggressor, lean, patient, random_0, rl_r1, ...``) that first triple is
      always the three scripted archetypes, so this flag alone carries NO
      information about whether any OTHER cyclic triple — one touching a learned
      checkpoint — is itself decisive. Use ``decisive_triples`` for the complete,
      order-independent picture.
    * ``decisive_triples`` (RLB-7) — EVERY cyclic 3-subset of the field (by
      point-estimate margins, the SAME A-over-B orientation :class:`PairwiseMargin`
      already defines) whose three edges' bootstrap CIs all exclude 0 in the
      cycle's own direction — computed from the SAME pairwise margins/CIs as
      ``decisive_cycle`` (no bootstrap re-run). Each entry is listed
      ``(x, y, z)`` in its cycle order (``x`` beats ``y`` beats ``z`` beats ``x``)
      with the lexicographically smallest label listed first. Defaults to ``()``
      so a pre-RLB-7 caller's artifact stays byte-identical.
    * ``checkpoint_inclusive_decisive_triples`` (RLB-7) — how many entries of
      ``decisive_triples`` contain at least one participant whose
      :attr:`LadderEntry.kind` is ``"checkpoint"``. This is the piece that makes
      the runbook's C3 criterion ("no dominant learned policy": no Condorcet
      winner over the field AND at least one decisive cycle touching a learned
      checkpoint) decidable straight off the ladder artifact instead of by hand.
      Defaults to ``0`` so a pre-RLB-7 caller's artifact stays byte-identical.
    """

    dominance: dict[str, tuple[str, ...]]
    n_cyclic_triples: int
    total_triples: int
    has_condorcet_winner: bool
    condorcet_winner: str | None
    decisive_cycle: bool
    decisive_triples: tuple[tuple[str, str, str], ...] = ()
    checkpoint_inclusive_decisive_triples: int = 0


@dataclass(frozen=True)
class LadderResult:
    """ """

    ladder_id: str
    n_seeds: int
    seed_set: tuple[int, ...]
    n_episodes: int
    entries: tuple[LadderEntry, ...]
    pairwise: tuple[PairwiseMargin, ...]
    kendall_tau_split_half: float | None
    bootstrap_resamples: int
    bootstrap_seed: int
    config_label: str = DEFAULT_ENV_LABEL
    tournament: TournamentSummary | None = None
    home_regions: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ParticipantSpec:
    """A parsed participant spec: its kind, NORMALIZED source string, and default label.

    ``source`` is the CANONICAL spec (:func:`_materialize_predicts` re-parses it
    from ``source`` rather than the caller threading the kind separately) — for a
    checkpoint this is the path AS GIVEN, but for a random participant it is
    RE-SERIALIZED as ``f"random:{seed}"`` so that ``"random:03"`` / ``"random:+3"``
    / ``"random: 3"`` (all parse to the same integer seed) mint the SAME
    ``source`` and therefore the SAME content-hash ``ladder_id`` — one owner
    (this function) for the random-seed grammar, rather than every consumer
    re-deriving its own idea of "the" spelling. ``seed`` carries the parsed
    integer seed for ``kind == "random"`` (``None`` otherwise) so a caller
    doesn't need to re-slice ``source`` itself. ``default_label`` is what a
    caller (the CLI, or a direct library call) uses when it doesn't supply an
    explicit label.
    """

    kind: ParticipantKind
    source: str
    default_label: str
    seed: int | None = None


def parse_participant_spec(spec: str) -> ParticipantSpec:
    """Parse a participant spec into its kind + default label (no filesystem checks).

    Three grammars:

    * ``"archetype:<name>"`` — a scripted archetype participant, either a
      calibrated OPEN-LOOP spike posture (:data:`LADDER_SPIKE_ARCHETYPE_LABELS`)
      or a CLOSED-LOOP geo posture (:data:`~retail_simulator.harness.
      geo_archetypes.GEO_ARCHETYPE_LABELS`, RLB-8b). ``<name>`` must be one of
      :data:`LADDER_ARCHETYPE_LABELS` (the union of the two); the default label
      IS the name. Which FAMILY a name belongs to only matters at materialize
      time (:func:`_materialize_predicts`) — a geo name needs a region table,
      which this parse step has no way to check (no filesystem/config reads
      here at all).
    * ``"random:<seed>"`` — a random-baseline participant sampling its own copy of
      the env's action space, re-seeded from ``<seed>`` every episode
      (:func:`_make_random_predict`). ``<seed>`` must be a non-negative integer
      (``int()`` handles a leading ``+`` / surrounding whitespace / leading
      zeros); the RETURNED ``source`` is the canonical ``f"random:{seed}"`` —
      NOT necessarily byte-identical to ``spec`` — and the default label is
      ``f"random_{seed}"``.
    * anything else — a checkpoint zip path. The default label is
      ``f"{path.parent.name}/{path.stem}"`` when the path has a named parent
      directory (disambiguates the common case where every canonical checkpoint
      shares the stem ``round_2``), else the bare stem. ``source`` here is
      ``spec`` VERBATIM (no path normalization — that stays a CLI boundary
      concern; see ``retail_simulator.cli.league_ladder``'s checkpoint-only
      ``Path`` round-trip).

    Raises ``ValueError`` on an unknown archetype name or a malformed/negative
    random seed. Does NO filesystem checks (existence / ``.zip`` suffix) — those
    are a CLI boundary concern (``retail_simulator.cli.league_ladder``), not a
    library one.
    """
    if spec.startswith(ARCHETYPE_SPEC_PREFIX):
        name = spec[len(ARCHETYPE_SPEC_PREFIX) :]
        if name not in LADDER_ARCHETYPE_LABELS:
            raise ValueError(
                f"unknown archetype {name!r} in {spec!r}; must be one of "
                f"{LADDER_ARCHETYPE_LABELS}"
            )
        return ParticipantSpec(kind="archetype", source=spec, default_label=name)
    if spec.startswith(RANDOM_SPEC_PREFIX):
        seed_str = spec[len(RANDOM_SPEC_PREFIX) :]
        try:
            seed = int(seed_str)
        except ValueError as exc:
            raise ValueError(
                f"random participant seed must be an integer; got {seed_str!r} in {spec!r}"
            ) from exc
        if seed < 0:
            raise ValueError(f"random participant seed must be >= 0; got {seed} in {spec!r}")
        return ParticipantSpec(
            kind="random",
            source=f"{RANDOM_SPEC_PREFIX}{seed}",
            default_label=f"random_{seed}",
            seed=seed,
        )
    path = Path(spec)
    parent_name = path.parent.name
    default_label = f"{parent_name}/{path.stem}" if parent_name else path.stem
    return ParticipantSpec(kind="checkpoint", source=spec, default_label=default_label)


def _kendall_tau(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    """ """
    if len(rank_a) != len(rank_b):
        raise ValueError(
            f"_kendall_tau requires equal-length rankings; "
            f"got len(rank_a)={len(rank_a)}, len(rank_b)={len(rank_b)}"
        )
    n = len(rank_a)
    if n < 2:
        # No pairs to compare — degenerate but consistent (both orders trivially
        # agree). Returning 1.0 keeps the diagnostic well-defined at the edge.
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = rank_a[i] - rank_a[j]
            b_order = rank_b[i] - rank_b[j]
            product = a_order * b_order
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            # product == 0 ⇒ a tie in one ranking; not expected for total orders,
            # and counted as neither concordant nor discordant (τ-a convention).
    total_pairs = n * (n - 1) // 2
    return float(concordant - discordant) / float(total_pairs)


def _field_margin_from_pair_means(
    label: str,
    others: Sequence[str],
    pair_mean: dict[tuple[str, str], float],
) -> float:
    """Mean of ``label``'s margin versus every OTHER participant.

    ``pair_mean`` is keyed by the sorted ``(label_a, label_b)`` pair and stores
    A-over-B; when ``label`` is the SECOND member of a stored pair the margin is
    negated (B-over-A). A participant's self-margin is 0 (position control) and is
    excluded — ``others`` carries exactly the field, never ``label`` itself.
    """
    margins: list[float] = []
    for other in others:
        if label < other:
            margins.append(pair_mean[(label, other)])
        else:
            margins.append(-pair_mean[(other, label)])
    return float(np.mean(margins))


def _bootstrap_margin_ci(
    label: str,
    others: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    *,
    n_seeds: int,
    resampled_seed_indices: np.ndarray,
) -> tuple[float, float]:
    """Bootstrap 95% CI on ``label``'s field-margin by resampling the SEED axis.

    ``resampled_seed_indices`` has shape ``(n_resamples, n_seeds)`` — each row is a
    seed-INDEX resample (with replacement) drawn ONCE and applied CONSISTENTLY to
    every pair, so each bootstrap iteration produces a COHERENT field-margin (the
    same surrogate world across all of ``label``'s pairings). For each iteration we
    recompute ``label``'s field-margin from the per-seed pair margins over the
    resampled indices; the CI is the 2.5th / 97.5th percentiles.

    Resampling the seed index (not the per-pair margins independently) is what
    makes the field coherent: a participant's edge over A and over B in the same
    surrogate draw is computed on the SAME resampled seeds.
    """
    # Stack each pairing's per-seed margins (oriented as label-over-other) into a
    # (n_others, n_seeds) matrix once, then index by the resampled seed columns.
    oriented_rows: list[np.ndarray] = []
    for other in others:
        if label < other:
            oriented_rows.append(pair_per_seed[(label, other)])
        else:
            oriented_rows.append(-pair_per_seed[(other, label)])
    # Shape (n_others, n_seeds). Mean over the OTHER axis (axis 0) is the field
    # margin per seed; we then resample the seed axis.
    oriented = np.asarray(oriented_rows, dtype=np.float64)
    # field_margin_per_seed[s] = mean over others of label-over-other on seed s.
    field_per_seed = oriented.mean(axis=0)  # shape (n_seeds,)
    # For each bootstrap row, average the field-per-seed over the resampled seed
    # indices. resampled_seed_indices is (n_resamples, n_seeds).
    boot_field = field_per_seed[resampled_seed_indices].mean(axis=1)  # (n_resamples,)
    ci_low = float(np.percentile(boot_field, 2.5))
    ci_high = float(np.percentile(boot_field, 97.5))
    return ci_low, ci_high


def _detect_policy_kind(checkpoint_path: str | Path) -> Literal["mlp", "lstm"]:
    """ """
    try:
        with zipfile.ZipFile(checkpoint_path) as zf:
            with zf.open("data") as data_member:
                data = json.load(data_member)
        serialized = data["policy_class"][":serialized:"]
        decoded = base64.b64decode(serialized)
    except (
        zipfile.BadZipFile,
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        # The PEEK failed (not a readable SB3 zip / missing field / bad base64).
        # Default to "mlp"; a real model-body load failure surfaces in materialize.
        # WARN so a future SB3 zip-format change can't SILENTLY demote a genuine
        # LSTM to MLP (the silent-misrating failure mode this whole slice fixes) —
        # an operator/CI can see the peek degraded rather than discovering it as a
        # mysteriously-bad rating.
        logger.warning(
            "policy-kind auto-detect peek failed for %s (%s: %s); defaulting to 'mlp'",
            checkpoint_path,
            type(exc).__name__,
            exc,
        )
        return "mlp"
    if b"recurrent" in decoded or b"sb3_contrib" in decoded:
        return "lstm"
    return "mlp"


def _make_random_predict(seed: int, action_space: Any) -> Callable[..., Any]:
    """ """
    space = copy.deepcopy(action_space)
    episode_index = 0
    space.seed(seed)

    def predict(_obs: object, *, episode_start: bool = False) -> Any:
        nonlocal episode_index
        if episode_start:
            space.seed(seed + 1_000_003 * episode_index)
            episode_index += 1
        return np.asarray(space.sample(), dtype=np.float32)

    return predict


def _materialize_predicts(
    spec_by_label: dict[str, str],
    *,
    action_space: Any | None = None,
    config: CoreConfig | None = None,
) -> dict[str, Callable[..., Any]]:
    """Build each participant's predict closure ONCE, routed by its spec's kind.

    Three routes (:func:`parse_participant_spec`):

    * ``checkpoint`` — the existing one-member :class:`FrozenPolicyPool` path: wraps
      the zip and materializes it via the EXISTING :meth:`FrozenPolicyPool.materialize`
      path (which lazily loads the SB3 zip — the lazy-SB3 contract holds: importing
      this module does NOT pull sb3).
    * ``archetype`` — a closure from EITHER :func:`~retail_simulator.harness.
      spike_archetypes.calibrated_archetypes` (a :data:`LADDER_SPIKE_ARCHETYPE_LABELS`
      name) or :func:`~retail_simulator.harness.geo_archetypes.geo_archetypes` (a
      :data:`~retail_simulator.harness.geo_archetypes.GEO_ARCHETYPE_LABELS` name,
      RLB-8b) — whichever family a label belongs to is built EXACTLY ONCE per
      :func:`_materialize_predicts` invocation (never per archetype participant)
      and shared across every archetype label of THAT family in ``spec_by_label``.
      A geo label with no ``config`` supplied raises ``ValueError`` (mirrors the
      ``random`` branch's ``action_space`` requirement just below). ``config`` is
      read LAZILY — only ``.demand.regions`` on the FIRST geo label encountered,
      never for a field with no geo participant at all. This is a PRIVATE
      helper (leading underscore): it trusts its internal callers rather than
      defensively re-validating ``config`` itself — the real boundary is
      :func:`run_ladder`, its only caller, which already validates
      ``config``/``config_label`` coherence up front (M2's inseparability
      check) before any rollout starts.
    * ``random`` — :func:`_make_random_predict` over ``action_space``; raises
      ``ValueError`` if no ``action_space`` was supplied.

    The returned ``{label: predict_closure}`` map is reused across EVERY pair AND
    EVERY seed in :func:`run_ladder`, so a checkpoint's ``PPO.load`` runs once per
    participant — NOT once per (pair, seed); an archetype/random closure is
    likewise built once and reused. ``cold load × C(n,2) × n_seeds × 2`` would
    otherwise dominate the wall-clock at the default ``n_seeds=30``.
    """
    # Lazy import: pull the league internals only at call time so importing this
    # module stays cheap and sb3-free.
    from retail_simulator.harness.league import (  # noqa: PLC0415
        FrozenPolicyPool,
        PoolMember,
    )

    predicts: dict[str, Callable[..., Any]] = {}
    spike_archetype_map: dict[str, ArchetypePredict] | None = None
    geo_archetype_map: dict[str, Callable[..., Any]] | None = None
    for label, spec in spec_by_label.items():
        parsed = parse_participant_spec(spec)
        if parsed.kind == "archetype":
            if parsed.default_label in GEO_ARCHETYPE_LABELS:
                if geo_archetype_map is None:
                    if config is None:
                        raise ValueError(
                            f"participant {label!r} ({spec!r}) is a geo archetype but "
                            "no config was provided to materialize it (no region table "
                            "to build geo_archetypes() from)"
                        )
                    # ONE geo_archetypes() call per invocation, shared across every
                    # geo participant — mirrors calibrated_archetypes()'s own
                    # "materialize once" contract just below. `config.demand.regions`
                    # is read HERE, not at the top of the function, so a caller with
                    # zero geo participants never has to satisfy this attribute.
                    region_table = config.demand.regions
                    geo_archetype_map = geo_archetypes(len(region_table), region_table)
                predicts[label] = geo_archetype_map[parsed.default_label]
            else:
                if spike_archetype_map is None:
                    # ONE calibrated_archetypes() call per invocation, shared across
                    # every archetype participant — mirrors the checkpoint path's
                    # "materialize once" contract (never re-built per pair/seed).
                    spike_archetype_map = calibrated_archetypes()
                predicts[label] = spike_archetype_map[parsed.default_label]
        elif parsed.kind == "random":
            if action_space is None:
                raise ValueError(
                    f"participant {label!r} ({spec!r}) is a random baseline but no "
                    "action_space was provided to materialize it"
                )
            # parsed.seed is guaranteed non-None for kind == "random" (the ONLY
            # branch parse_participant_spec sets it on) — this is the one owner of
            # the random-seed grammar; nothing here re-slices RANDOM_SPEC_PREFIX.
            assert parsed.seed is not None
            predicts[label] = _make_random_predict(parsed.seed, action_space)
        elif parsed.kind == "checkpoint":
            # checkpoint_path is resolved against pool_dir as ``pool_dir / path``
            # inside materialize, so seat pool_dir at the zip's parent and carry
            # just the filename. A one-member pool reuses the existing, tested
            # load+predict closure rather than calling PPO.load directly.
            path = Path(spec)
            policy_kind = _detect_policy_kind(path)
            member = PoolMember(
                member_id=f"checkpoint:ladder_{label}",
                kind="checkpoint",
                archetype_name=f"frozen_policy:ladder_{label}",
                checkpoint_path=Path(path.name),
                round_index=0,
                policy_kind=policy_kind,
            )
            pool = FrozenPolicyPool(path.parent, members=(member,))
            predicts[label] = pool.materialize(member, env=None)
        else:
            # Unreachable while ParticipantKind is exactly {checkpoint, archetype,
            # random} — an explicit raise (not a bare `else`) so a FUTURE fourth
            # kind can never silently fall through into the checkpoint Path(spec)
            # branch above.
            raise ValueError(f"participant {label!r} has unhandled kind {parsed.kind!r}")
    return predicts


def _default_pairwise_per_seed(
    sorted_labels: Sequence[str],
    spec_by_label: dict[str, str],
    *,
    seed_set: Sequence[int],
    n_episodes: int,
    config: CoreConfig | None = None,
    home_regions: tuple[int, ...] | None = None,
) -> tuple[
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], dict[str, tuple[float, ...]] | None],
]:
    """The REAL default rollouts: build the env, materialize ONCE, then loop pairs × seeds.

    ``config`` (default ``None``) selects the env: ``None`` falls back to
    :func:`~retail_simulator.harness.parallel_gate.competition_config` (the
    pre-RLB-2 default); any other :class:`CoreConfig` (e.g.
    :func:`~retail_simulator.harness.spike.calibrated_spike_config`) runs the
    ladder on THAT env instead.

    The env is built FIRST, and materializing runs INSIDE the same ``try`` right
    after — a random participant's predict closure needs the env's action space
    (:func:`_materialize_predicts`'s ``action_space`` kwarg), and building the env
    first means a materialize failure still reaches the ``finally`` and closes it.
    :func:`_position_controlled_head_to_head` re-seeds the world per call
    (``env.reset(seed=...)``), so ONE shared env reused across every pair and seed
    is correct and deterministic. Returns a ``(per_seed_by_pair,
    component_per_seed_by_pair)`` pair. ``per_seed_by_pair`` is ``{(label_a,
    label_b): per_seed_margins}`` for every unordered pair in ``label_a <
    label_b`` order, where each per-seed entry is ``mean(scores_a) -
    mean(scores_b)`` on that seed (UNCHANGED from pre-RLB-3).

    ``component_per_seed_by_pair`` (RLB-3, battle scoreboard) is ``{(label_a,
    label_b): {component_name: per_seed_margins} | None}`` — the per
    :data:`~retail_simulator.harness.league.HEAD_TO_HEAD_COMPONENTS` per-seed
    margin (``a_mean - b_mean`` averaged over episodes, read off
    :func:`~retail_simulator.harness.league._position_controlled_head_to_head`'s
    ``component_sink``), or ``None`` for a pair whose rollout primitive never
    filled the sink (e.g. a test double that ignores the ``component_sink``
    kwarg) — decided fresh per pair, all-or-nothing across that pair's seeds.

    This path runs ONLY when ``run_ladder`` was not handed an injected
    ``head_to_head`` — the test seam keeps its own per-(pair, seed) call shape.

    ``home_regions`` (RLB-8a, default ``None``) forwards straight to this shared
    env's construction (:class:`~retail_simulator.envs.parallel_env.
    RetailParallelEnv`'s own ``home_regions`` kwarg, S3); ``None`` reproduces
    every pre-RLB-8a call byte-for-byte.
    """
    # Lazy imports: pull pettingzoo (RetailParallelEnv) + the league primitive only
    # at call time so importing this module stays cheap and sb3-free.
    from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: PLC0415
    from retail_simulator.harness.league import (  # noqa: PLC0415
        _position_controlled_head_to_head,
    )
    from retail_simulator.harness.parallel_gate import competition_config  # noqa: PLC0415

    effective_config = config if config is not None else competition_config()

    per_seed_by_pair: dict[tuple[str, str], list[float]] = {}
    component_per_seed_by_pair: dict[tuple[str, str], dict[str, tuple[float, ...]] | None] = {}
    env = RetailParallelEnv(
        effective_config, n_learning_agents=2, seed=seed_set[0], home_regions=home_regions
    )
    try:
        predicts = _materialize_predicts(
            spec_by_label,
            action_space=env.action_space(env.possible_agents[0]),
            # RLB-8b: always forward the effective config — `_materialize_predicts`
            # only reads `.demand.regions` off it when an `archetype:<geo name>`
            # participant is actually present, so this is harmless for every
            # checkpoint/random/spike-only field (a test double standing in for
            # `effective_config` need not even implement `.demand`). Lets a geo
            # participant materialize a REAL closure under whatever env
            # `run_ladder` resolved (`default`/`calibrated`'s 2-region table
            # included — degenerate there, same as a spike posture, but not an
            # error; `arena`'s 25-region table is where it matters).
            config=effective_config,
        )
        for i, label_a in enumerate(sorted_labels):
            for label_b in sorted_labels[i + 1 :]:
                per_seed: list[float] = []
                # RLB-3: per-component per-seed margins for THIS pair; flips to
                # None the moment a seed's sink comes back empty (a rollout
                # primitive that doesn't support component_sink) — all-or-nothing
                # for the pair, not per-seed.
                component_per_seed: dict[str, list[float]] | None = {}
                for seed in seed_set:
                    component_sink: dict[str, list[tuple[float, float]]] = {}
                    scores_a, scores_b = _position_controlled_head_to_head(
                        env,
                        effective_config.reward,
                        predict_a=predicts[label_a],
                        predict_b=predicts[label_b],
                        n_episodes=n_episodes,
                        seed=seed,
                        component_sink=component_sink,
                    )
                    per_seed.append(float(np.mean(scores_a)) - float(np.mean(scores_b)))
                    if component_per_seed is not None:
                        if component_sink:
                            for name, per_episode in component_sink.items():
                                margin = float(np.mean([a - b for a, b in per_episode]))
                                component_per_seed.setdefault(name, []).append(margin)
                        else:
                            component_per_seed = None
                per_seed_by_pair[(label_a, label_b)] = per_seed
                component_per_seed_by_pair[(label_a, label_b)] = (
                    {name: tuple(values) for name, values in component_per_seed.items()}
                    if component_per_seed is not None
                    else None
                )
    finally:
        env.close()
    return per_seed_by_pair, component_per_seed_by_pair


def _content_hash_ladder_id(
    participants: Sequence[tuple[str, str]],
    *,
    n_seeds: int,
    n_episodes: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    config_label: str = DEFAULT_ENV_LABEL,
    home_regions: tuple[int, ...] | None = None,
) -> str:
    """ """
    fingerprint: dict[str, Any] = {
        "participants": [list(p) for p in participants],
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
    }
    if config_label != DEFAULT_ENV_LABEL:
        fingerprint["config_label"] = config_label
    if home_regions is not None:
        fingerprint["home_regions"] = list(home_regions)
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    return f"ladder-{digest[:12]}"


class _ScoreMarginRow(NamedTuple):
    """One pair's score margin + bootstrap CI — the raw ingredients for a
    :class:`~retail_simulator.harness.spike.PairMargin` (RLB-3 battle scoreboard).

    Kept separate from :class:`PairwiseMargin` (whose ``ci_low``/``ci_high`` are
    ``float | None`` for pre-RLB-3 byte-compat) so :func:`_build_tournament_summary`
    never has to narrow an ``Optional`` that, by the time ``run_ladder`` calls it,
    is always concrete.
    """

    label_a: str
    label_b: str
    per_seed_margins: tuple[float, ...]
    mean_margin: float
    margin_se: float
    ci_low: float
    ci_high: float


def _build_tournament_summary(
    score_margin_rows: Sequence[_ScoreMarginRow],
    labels: Sequence[str],
    kind_by_label: Mapping[str, ParticipantKind],
) -> TournamentSummary:
    """Build the dominance/cycle/Condorcet summary from THIS ladder's own score margins.

    Reuses :func:`retail_simulator.harness.spike._measure_tournament` (dominance
    graph + 3-cycle count + Condorcet detection + per-cycle decisiveness) over
    :class:`~retail_simulator.harness.spike.PairMargin` records built from
    ``score_margin_rows`` — no re-implementation, mirrors
    ``scripts/phase4_rl_confirm.py``'s ``measure()``. ``total_triples`` is
    ``C(len(labels), 3)`` (:func:`itertools.combinations` naturally yields none
    below 3 labels, so no special-casing is needed for a 2-participant field).
    Lazy import keeps this module sb3-free at load time (``spike`` pulls the same
    rollout stack as ``league``).

    RLB-7: also reuses :func:`retail_simulator.harness.spike._triple_cycles` (the
    SAME per-triple cycle-direction detector ``_measure_tournament`` uses
    internally, applied here to EVERY 3-subset rather than stopping at the first
    cyclic one) to build ``decisive_triples`` — from the SAME ``beats``/CI-exclusion
    facts, derived fresh from ``pair_margins`` (no bootstrap re-run, no second
    rollout). ``kind_by_label`` is consulted only to count how many decisive
    triples touch a ``"checkpoint"`` participant.
    """
    from retail_simulator.harness.spike import (  # noqa: PLC0415
        PairMargin,
        _measure_tournament,
        _triple_cycles,
    )

    pair_margins = tuple(
        PairMargin(
            label_a=row.label_a,
            label_b=row.label_b,
            per_seed_margins=row.per_seed_margins,
            mean_margin=row.mean_margin,
            margin_se=row.margin_se,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
        )
        for row in score_margin_rows
    )
    (
        dominance,
        n_cyclic_triples,
        has_condorcet_winner,
        condorcet_winner,
        _cycle_direction,
        decisive_cycle,
    ) = _measure_tournament(pair_margins, tuple(labels))
    triples = list(combinations(labels, 3))
    total_triples = len(triples)

    # RLB-7: the SAME per-pair facts _measure_tournament derives internally
    # (beats-by-sign, CI-excludes-zero) — recomputed here, not re-bootstrapped,
    # so EVERY cyclic triple's decisiveness can be read, not just the first one
    # `decisive_cycle` mirrors (see the TournamentSummary docstring).
    beats: dict[tuple[str, str], bool] = {}
    ci_excludes_zero: dict[frozenset[str], bool] = {}
    for pm in pair_margins:
        beats[(pm.label_a, pm.label_b)] = pm.mean_margin > 0.0
        beats[(pm.label_b, pm.label_a)] = pm.mean_margin < 0.0
        ci_excludes_zero[frozenset((pm.label_a, pm.label_b))] = (
            pm.ci_low > 0.0 if pm.mean_margin > 0.0 else pm.ci_high < 0.0
        )

    decisive_triples: list[tuple[str, str, str]] = []
    for triple in triples:
        order = _triple_cycles(triple, beats)
        if order is None:
            continue
        # Smallest-label-first is INHERITED, not a re-sort: `triple` comes from
        # `combinations(labels, 3)` over the already-sorted `labels` this function
        # requires of its caller, so `triple[0]` is already the lexicographically
        # smallest of the three; `_triple_cycles` always starts its walk at
        # `triple[0]` (spike.py), so `order[0]` carries that same property forward.
        assert order[0] == triple[0]
        edges = (
            frozenset((order[0], order[1])),
            frozenset((order[1], order[2])),
            frozenset((order[2], order[0])),
        )
        if all(ci_excludes_zero[edge] for edge in edges):
            decisive_triples.append(order)

    checkpoint_inclusive_decisive_triples = sum(
        1
        for triple in decisive_triples
        if any(kind_by_label[label] == "checkpoint" for label in triple)
    )

    return TournamentSummary(
        dominance=dominance,
        n_cyclic_triples=n_cyclic_triples,
        total_triples=total_triples,
        has_condorcet_winner=has_condorcet_winner,
        condorcet_winner=condorcet_winner,
        decisive_cycle=decisive_cycle,
        decisive_triples=tuple(decisive_triples),
        checkpoint_inclusive_decisive_triples=checkpoint_inclusive_decisive_triples,
    )


def run_ladder(
    participants: Sequence[tuple[str, str]],
    *,
    n_seeds: int = 30,
    n_episodes: int = 1,
    ladder_id: str | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 0,
    head_to_head: Callable[..., tuple[list[float], list[float]]] | None = None,
    config: CoreConfig | None = None,
    config_label: str = DEFAULT_ENV_LABEL,
    home_regions: tuple[int, ...] | None = None,
) -> LadderResult:
    """ """
    if len(participants) < 2:
        raise ValueError(f"run_ladder requires at least 2 participants; got {len(participants)}")
    labels = [label for _spec, label in participants]
    if len(set(labels)) != len(labels):
        raise ValueError(f"run_ladder participant labels must be unique; got {labels}")
    if n_seeds < 1:
        raise ValueError(f"run_ladder requires n_seeds >= 1; got {n_seeds}")
    if n_episodes < 1:
        raise ValueError(f"run_ladder requires n_episodes >= 1; got {n_episodes}")
    # M2: config and config_label are kept INSEPARABLE — the id/artifact path
    # follow the LABEL, so letting them disagree would silently mint the wrong
    # id (or overwrite the default env's artifact with a non-default run).
    if config is not None and config_label == DEFAULT_ENV_LABEL:
        raise ValueError(
            "run_ladder: a non-default config requires a distinct config_label — "
            "the ladder_id and artifact path follow the LABEL, so a non-default "
            "config left at the default label would silently mint the DEFAULT "
            "env's id and overwrite its artifact"
        )
    if config is None and config_label != DEFAULT_ENV_LABEL:
        raise ValueError(
            f"run_ladder: config_label={config_label!r} was given but config is "
            "None (falls back to competition_config()) — the ladder_id/artifact "
            "path would then claim an env that never actually ran; pass the "
            "matching config"
        )
    if config is not None:
        # M2 (senior review): the two checks above only distinguish DEFAULT
        # from non-default — an arena config labelled CALIBRATED_ENV_LABEL (or
        # a calibrated config labelled anything else, e.g. ARENA_ENV_LABEL)
        # passed both unchecked and minted an artifact that lied about which
        # world ran (verified in both directions). Compare against a REAL
        # property of `config`: calibrated_spike_config()'s region table is
        # the one this codebase treats as "the" calibrated shape (it never
        # touches demand.regions, so it stays at CoreConfig's own 2-region
        # default) — every OTHER non-default env (today: the arena and
        # arena-balanced envs, both at ARENA_N_REGIONS=25 regions) has a
        # DIFFERENT region table, so comparing full equality (not just length)
        # also catches a same-length coincidence. One check catches BOTH mismatch
        # directions: labelled calibrated but not calibrated-shaped, or
        # calibrated-shaped but labelled something else. Lazy import:
        # harness.spike pulls the same heavy rollout stack harness.league
        # does (module docstring), so this stays out of module-load time.
        from retail_simulator.harness.spike import calibrated_spike_config  # noqa: PLC0415

        is_calibrated_shaped = config.demand.regions == calibrated_spike_config().demand.regions
        labeled_calibrated = config_label == CALIBRATED_ENV_LABEL
        if is_calibrated_shaped != labeled_calibrated:
            raise ValueError(
                f"run_ladder: config_label={config_label!r} does not match its "
                f"config's region table (config is calibrated-shaped: "
                f"{is_calibrated_shaped}, label claims calibrated: "
                f"{labeled_calibrated}) — a (config, config_label) pair naming "
                "the wrong env would mint an artifact that lies about which "
                "world ran"
            )
    if home_regions is not None:
        # RLB-8a: fail fast, before any rollout — mirrors the config/config_label
        # inseparability check above. Lazy import: competition_config pulls
        # pettingzoo transitively via envs.parallel_env's own imports elsewhere
        # in this module, so keep this out of module-load time too.
        from retail_simulator.harness.parallel_gate import competition_config  # noqa: PLC0415

        effective_n_regions = len(
            (config if config is not None else competition_config()).demand.regions
        )
        # m1 (senior review): mirror envs/parallel_env.py's own
        # `_build_seat_plan` validation (RetailParallelEnv's home_regions
        # boundary check) exactly — the ladder's shared env always seats 2
        # learning agents and 0 NPCs (_default_pairwise_per_seed), so
        # anything but exactly 2 entries is accepted here today, folded into
        # the content-hash ladder_id, and recorded on the artifact even
        # though it can never reach a real seat plan. Same for a non-integer
        # / bool index: it would satisfy the range check below yet seat a
        # store NOWHERE (or, for bool, seat a technically in-range but
        # never-intended region 0/1).
        if len(home_regions) != 2:
            raise ValueError(
                f"run_ladder: home_regions must have exactly 2 entries (the "
                "ladder's shared env always seats 2 learning agents, no "
                f"NPCs); got {len(home_regions)}"
            )
        for seat_index, home in enumerate(home_regions):
            if (
                not isinstance(home, (int, np.integer))
                or isinstance(home, bool)
                or not (0 <= home < effective_n_regions)
            ):
                raise ValueError(
                    f"run_ladder: home_regions[{seat_index}]={home!r} (type "
                    f"{type(home).__name__}) is invalid; must be an integer "
                    f"region index satisfying 0 <= r < {effective_n_regions}"
                )

    # Parse every spec UP FRONT: a malformed spec (bad archetype name / seed) must
    # raise before the seam or any rollout runs, not partway through a pairing
    # loop. This also NORMALIZES each spec (e.g. "random:03" -> "random:3") so the
    # content-hash ladder_id and every downstream consumer see ONE canonical
    # spelling, not whatever the caller happened to type (m2/m3).
    parsed_by_label: dict[str, ParticipantSpec] = {
        label: parse_participant_spec(spec) for spec, label in participants
    }
    kind_by_label: dict[str, ParticipantKind] = {
        label: parsed.kind for label, parsed in parsed_by_label.items()
    }
    normalized_participants = [(parsed_by_label[label].source, label) for label in labels]

    resolved_ladder_id = (
        ladder_id
        if ladder_id is not None
        else _content_hash_ladder_id(
            normalized_participants,
            n_seeds=n_seeds,
            n_episodes=n_episodes,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            config_label=config_label,
            home_regions=home_regions,
        )
    )
    seed_set = tuple(range(n_seeds))
    spec_by_label = {label: parsed.source for label, parsed in parsed_by_label.items()}

    # Every unordered pair in (label_a < label_b) order — a STABLE, deterministic
    # iteration so the result is reproducible regardless of input ordering.
    sorted_labels = sorted(labels)

    # Gather each pair's per-seed margins. The DEFAULT path materializes every
    # participant ONCE and reuses the predicts across all pairs/seeds; the injected
    # ``head_to_head`` seam (unit tests) keeps its per-(pair, seed) call shape.
    # RLB-3 (battle scoreboard): the default path ALSO returns, per pair, the
    # per-component per-seed margins when the underlying primitive supplies them
    # (None otherwise); the injected seam predates component support and never
    # supplies them — every pair gets None on that path.
    component_per_seed_by_pair: dict[tuple[str, str], dict[str, tuple[float, ...]] | None]
    if head_to_head is None:
        per_seed_by_pair, component_per_seed_by_pair = _default_pairwise_per_seed(
            sorted_labels,
            spec_by_label,
            seed_set=seed_set,
            n_episodes=n_episodes,
            config=config,
            home_regions=home_regions,
        )
    else:
        per_seed_by_pair = {}
        component_per_seed_by_pair = {}
        for i, label_a in enumerate(sorted_labels):
            for label_b in sorted_labels[i + 1 :]:
                per_seed_list: list[float] = []
                for seed in seed_set:
                    scores_a, scores_b = head_to_head(
                        spec_by_label[label_a],
                        spec_by_label[label_b],
                        n_episodes=n_episodes,
                        seed=seed,
                    )
                    per_seed_list.append(float(np.mean(scores_a)) - float(np.mean(scores_b)))
                per_seed_by_pair[(label_a, label_b)] = per_seed_list
                component_per_seed_by_pair[(label_a, label_b)] = None

    pair_mean: dict[tuple[str, str], float] = {}
    pair_margin_se: dict[tuple[str, str], float] = {}
    pair_per_seed: dict[tuple[str, str], np.ndarray] = {}
    for i, label_a in enumerate(sorted_labels):
        for label_b in sorted_labels[i + 1 :]:
            per_seed = per_seed_by_pair[(label_a, label_b)]
            per_seed_arr = np.asarray(per_seed, dtype=np.float64)
            mean_margin = float(per_seed_arr.mean())
            # Population sd (ddof=0) over sqrt(n_seeds): the standard error of the
            # seed mean under the fixed-seed-set fairness contract.
            margin_se = float(np.std(per_seed_arr, ddof=0)) / sqrt(n_seeds)
            pair_mean[(label_a, label_b)] = mean_margin
            pair_margin_se[(label_a, label_b)] = margin_se
            pair_per_seed[(label_a, label_b)] = per_seed_arr

    # Field-margin per participant (mean margin vs the rest; self excluded).
    field_margin_by_label: dict[str, float] = {}
    for label in sorted_labels:
        others = [other for other in sorted_labels if other != label]
        field_margin_by_label[label] = _field_margin_from_pair_means(label, others, pair_mean)

    # Bootstrap: draw the seed-INDEX resamples ONCE (an independent RNG, NOT tied
    # to rollout seeds) and apply them CONSISTENTLY to every participant's field —
    # a single coherent surrogate seed set per bootstrap iteration. RLB-3 (battle
    # scoreboard) reuses this SAME draw below for every per-pair score/component
    # CI, so the whole artifact is one coherent bootstrap.
    rng = np.random.default_rng(bootstrap_seed)
    resampled_seed_indices = rng.integers(0, n_seeds, size=(bootstrap_resamples, n_seeds))
    ci_by_label: dict[str, tuple[float, float]] = {}
    for label in sorted_labels:
        others = [other for other in sorted_labels if other != label]
        ci_by_label[label] = _bootstrap_margin_ci(
            label,
            others,
            pair_per_seed,
            n_seeds=n_seeds,
            resampled_seed_indices=resampled_seed_indices,
        )

    # RLB-3 (battle scoreboard): per-pair score CI (the SAME draw as above) + the
    # per-component mean/CI/per-seed breakdown, plus the raw rows
    # _build_tournament_summary needs. component_pair_mean mirrors pair_mean but
    # per component name, feeding each entry's component_field_margins below via
    # the SAME _field_margin_from_pair_means helper the score field-margin uses.
    pairwise: list[PairwiseMargin] = []
    component_pair_mean: dict[str, dict[tuple[str, str], float]] = {}
    all_pairs_have_components = True
    score_margin_rows: list[_ScoreMarginRow] = []
    for i, label_a in enumerate(sorted_labels):
        for label_b in sorted_labels[i + 1 :]:
            pair_key = (label_a, label_b)
            mean_margin = pair_mean[pair_key]
            margin_se = pair_margin_se[pair_key]
            per_seed_tuple = tuple(float(v) for v in pair_per_seed[pair_key])
            ci_low, ci_high = _bootstrap_margin_ci(
                label_a,
                [label_b],
                pair_per_seed,
                n_seeds=n_seeds,
                resampled_seed_indices=resampled_seed_indices,
            )
            score_margin_rows.append(
                _ScoreMarginRow(
                    label_a=label_a,
                    label_b=label_b,
                    per_seed_margins=per_seed_tuple,
                    mean_margin=mean_margin,
                    margin_se=margin_se,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )

            components = component_per_seed_by_pair[pair_key]
            component_mean_margins: dict[str, float] | None = None
            component_ci: dict[str, tuple[float, float]] | None = None
            component_per_seed_margins: dict[str, tuple[float, ...]] | None = None
            if components is None:
                all_pairs_have_components = False
            else:
                component_mean_margins = {}
                component_ci = {}
                # n3: a defensive copy — component_per_seed_by_pair[pair_key] is this
                # loop's own local, but PairwiseMargin is frozen/"public" and callers
                # must not be able to mutate run_ladder's internal per-pair dict by
                # mutating the returned entry's component_per_seed_margins.
                component_per_seed_margins = dict(components)
                for name, values in components.items():
                    arr = np.asarray(values, dtype=np.float64)
                    component_mean_margins[name] = float(arr.mean())
                    component_pair_mean.setdefault(name, {})[pair_key] = float(arr.mean())
                    c_ci_low, c_ci_high = _bootstrap_margin_ci(
                        label_a,
                        [label_b],
                        {pair_key: arr},
                        n_seeds=n_seeds,
                        resampled_seed_indices=resampled_seed_indices,
                    )
                    component_ci[name] = (c_ci_low, c_ci_high)

            pairwise.append(
                PairwiseMargin(
                    label_a=label_a,
                    label_b=label_b,
                    per_seed_margins=per_seed_tuple,
                    mean_margin=mean_margin,
                    margin_se=margin_se,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    component_mean_margins=component_mean_margins,
                    component_ci=component_ci,
                    component_per_seed_margins=component_per_seed_margins,
                )
            )

    # Rank field-margin DESC; tie-break by label for a deterministic total order.
    ranked_labels = sorted(sorted_labels, key=lambda lbl: (-field_margin_by_label[lbl], lbl))
    entries: list[LadderEntry] = []
    for rank_index, label in enumerate(ranked_labels, start=1):
        ci_low, ci_high = ci_by_label[label]
        component_field_margins: dict[str, float] | None = None
        if all_pairs_have_components:
            others = [other for other in sorted_labels if other != label]
            component_field_margins = {
                name: _field_margin_from_pair_means(label, others, component_pair_mean[name])
                for name in component_pair_mean
            }
        entries.append(
            LadderEntry(
                label=label,
                checkpoint_path=spec_by_label[label],
                field_margin=field_margin_by_label[label],
                ci_low=ci_low,
                ci_high=ci_high,
                rank=rank_index,
                kind=kind_by_label[label],
                component_field_margins=component_field_margins,
            )
        )

    kendall_tau_split_half = _split_half_kendall_tau(sorted_labels, pair_per_seed, n_seeds=n_seeds)
    # RLB-3: the tournament summary is ALWAYS computed (default and injected-seam
    # paths alike) from THIS ladder's own score margins — never gated on
    # component availability.
    tournament = _build_tournament_summary(score_margin_rows, sorted_labels, kind_by_label)

    return LadderResult(
        ladder_id=resolved_ladder_id,
        n_seeds=n_seeds,
        seed_set=seed_set,
        n_episodes=n_episodes,
        entries=tuple(entries),
        pairwise=tuple(pairwise),
        kendall_tau_split_half=kendall_tau_split_half,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        config_label=config_label,
        tournament=tournament,
        home_regions=home_regions,
    )


def _rank_labels_on_seed_subset(
    labels: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    seed_indices: np.ndarray,
) -> list[str]:
    """Rank ``labels`` by field-margin computed over ONLY ``seed_indices``.

    Used by the split-half diagnostic: recompute each participant's field-margin
    restricted to a seed subset (evens / odds) and return the labels best-first.
    Mean over the subset of seeds of the per-seed field margin.
    """
    field_by_label: dict[str, float] = {}
    for label in labels:
        others = [other for other in labels if other != label]
        oriented_rows: list[np.ndarray] = []
        for other in others:
            if label < other:
                oriented_rows.append(pair_per_seed[(label, other)][seed_indices])
            else:
                oriented_rows.append(-pair_per_seed[(other, label)][seed_indices])
        oriented = np.asarray(oriented_rows, dtype=np.float64)
        field_by_label[label] = float(oriented.mean())
    return sorted(labels, key=lambda lbl: (-field_by_label[lbl], lbl))


def _split_half_kendall_tau(
    labels: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    *,
    n_seeds: int,
) -> float | None:
    """ """
    if n_seeds < 4:
        return None
    even_indices = np.arange(0, n_seeds, 2)
    odd_indices = np.arange(1, n_seeds, 2)
    even_order = _rank_labels_on_seed_subset(labels, pair_per_seed, even_indices)
    odd_order = _rank_labels_on_seed_subset(labels, pair_per_seed, odd_indices)
    # Map each label to its rank position in each half, then τ over those ranks in
    # a FIXED label order so the index pairs line up.
    even_rank = {label: pos for pos, label in enumerate(even_order)}
    odd_rank = {label: pos for pos, label in enumerate(odd_order)}
    fixed = sorted(labels)
    rank_a = [even_rank[label] for label in fixed]
    rank_b = [odd_rank[label] for label in fixed]
    return _kendall_tau(rank_a, rank_b)
