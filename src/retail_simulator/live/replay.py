"""Phase 2 Slice A T-A3 — the offline replay validator (AC-4 substrate).

Slice A scoping (NOT-YET): the manifest's ``config_summary`` field is a
forward-compat dict reserved for richer scenarios; the Slice A reconstruction
path pins ``CoreConfig.default()`` and seats only the manifest's
``n_agents`` + ``npc_archetypes``. Slice B+ may extend the reconstruction
to honor a full scenario snapshot — the dataclass already carries the
``config_summary`` field for that extension point.

Pure-Python — no asyncio, no ``websockets``. Importable without the ``[serve]``
extra (the lazy-import contract from the Slice A package docstring); only
stdlib + numpy + ``core`` + ``envs.parallel_env``. The replay tool is the
offline complement to Slice B's online server.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.parallel_env import RetailParallelEnv


@dataclass(frozen=True)
class ReplayManifest:
    """Per-game manifest written alongside the action-log.

    Carries every input the determinism contract needs so :func:`replay` can
    rebuild the SAME :class:`RetailParallelEnv` from disk: the seam's seed,
    the seat plan (``n_agents`` + ``npc_archetypes``), the schema version
    (catches a live game replayed under a re-numbered schema), and the
    package version (catches a live game replayed under different economics
    code). ``config_summary`` is a Slice B+ extension hook — Slice A pins
    ``CoreConfig.default()`` and ignores this field on read.
    """

    game_id: str
    seed: int
    n_agents: int
    npc_archetypes: tuple[str, ...]
    n_ticks: int
    started_at_iso: str
    schema_version: int
    software_version: str
    # Reserved for richer scenario reconstruction (Slice B+); empty dict on Slice A.
    config_summary: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize as a single JSON object string. ``npc_archetypes`` flattens to a list."""
        payload = asdict(self)
        # Tuples are not JSON-native; ``asdict`` flattens to list, and
        # :meth:`from_json` rebuilds the tuple so the dataclass stays hashable.
        return json.dumps(payload, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> ReplayManifest:
        """Parse a manifest JSON string. Raises ``ValueError`` on missing keys.

        Boundary validation: a missing required key fails loudly with the key
        name in the error message (operator-greppable). Unknown extra keys are
        IGNORED to keep forward-compat: a Slice B+ manifest can carry extra
        fields and this loader still parses it.
        """
        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ReplayManifest.from_json: malformed JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"ReplayManifest.from_json: expected JSON object, got {type(data).__name__}"
            )
        try:
            return cls(
                game_id=str(data["game_id"]),
                seed=int(data["seed"]),
                n_agents=int(data["n_agents"]),
                npc_archetypes=tuple(str(a) for a in data["npc_archetypes"]),
                n_ticks=int(data["n_ticks"]),
                started_at_iso=str(data["started_at_iso"]),
                schema_version=int(data["schema_version"]),
                software_version=str(data["software_version"]),
                config_summary=dict(data.get("config_summary", {})),
            )
        except KeyError as exc:
            raise ValueError(
                f"ReplayManifest.from_json: missing required key {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True)
class ActionLogRecord:
    """One JSONL record per tick (``runs/<game-id>/action_log.jsonl``).

    Carries BOTH halves of a tick (the joint action applied AND the per-seat
    observations + rewards the seam returned) so :func:`replay` reduces to
    "re-step and diff" — no separate observations file. ``missing_seats``
    records which seats fell back to last-valid-or-default (AC-2 audit
    trail); ``deadline_ns`` is the absolute wall-clock deadline the live
    server collected this tick against (operator audit).

    Fields use Python lists (not numpy arrays) so the dataclass is JSON-safe
    via :func:`json.dumps`; :meth:`to_json` is a one-call ``dumps``.
    """

    tick: int
    joint_action: dict[str, list[float]]
    observations: dict[str, list[float]]
    rewards: dict[str, float]
    missing_seats: list[int]
    deadline_ns: int

    def to_json(self) -> str:
        """Serialize as a single-line JSON object (the JSONL contract)."""
        # No indent: JSONL is one record per line. ``json.dumps`` defaults to
        # no trailing newline; the writer adds ``"\n"`` per record.
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> ActionLogRecord:
        """Parse one JSONL line into a record. Raises ``ValueError`` on malformed input.

        The boundary parser: validates required keys are present and reasonably
        shaped (dicts of lists / dicts of floats / list of ints). Per-element
        type coercion is permissive (int accepted as float, per JSON semantics).
        """
        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ActionLogRecord.from_json: malformed JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"ActionLogRecord.from_json: expected JSON object, got {type(data).__name__}"
            )
        try:
            tick = int(data["tick"])
            joint_raw = data["joint_action"]
            observations_raw = data["observations"]
            rewards_raw = data["rewards"]
            missing_raw = data["missing_seats"]
            deadline_ns = int(data["deadline_ns"])
        except KeyError as exc:
            raise ValueError(
                f"ActionLogRecord.from_json: missing required key {exc.args[0]!r}"
            ) from exc
        if not isinstance(joint_raw, dict):
            raise ValueError("ActionLogRecord.from_json: 'joint_action' must be a JSON object")
        if not isinstance(observations_raw, dict):
            raise ValueError("ActionLogRecord.from_json: 'observations' must be a JSON object")
        if not isinstance(rewards_raw, dict):
            raise ValueError("ActionLogRecord.from_json: 'rewards' must be a JSON object")
        if not isinstance(missing_raw, list):
            raise ValueError("ActionLogRecord.from_json: 'missing_seats' must be a JSON array")
        return cls(
            tick=tick,
            joint_action={str(k): [float(x) for x in v] for k, v in joint_raw.items()},
            observations={str(k): [float(x) for x in v] for k, v in observations_raw.items()},
            rewards={str(k): float(v) for k, v in rewards_raw.items()},
            missing_seats=[int(s) for s in missing_raw],
            deadline_ns=deadline_ns,
        )


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a replay run.

    Boolean ``ok`` is the AC-4 verdict: ``True`` iff every per-seat
    ``(observation, reward)`` matched bit-for-bit against the recorded log.
    On failure ``first_divergence_tick`` localizes the divergence and
    ``divergence_summary`` carries an operator-readable explanation.
    ``n_ticks_replayed`` is the count of records successfully processed
    (1-based; on a tick-0 divergence the value is 1).
    """

    ok: bool
    n_ticks_replayed: int
    first_divergence_tick: int | None
    divergence_summary: str | None


def read_manifest(manifest_path: Path) -> ReplayManifest:
    """Read + parse a manifest JSON file.

    ``FileNotFoundError`` (with the offending path in the message) is the
    explicit signal a missing path produces — same discipline as
    :meth:`FrozenPolicyPool.load`. Parser errors propagate as ``ValueError``
    from :meth:`ReplayManifest.from_json`.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"read_manifest: no manifest at {manifest_path}; "
            f"the file is written by the live server at game start"
        )
    return ReplayManifest.from_json(manifest_path.read_text(encoding="utf-8"))


def read_action_log(log_path: Path) -> Iterator[ActionLogRecord]:
    """Lazily yield one :class:`ActionLogRecord` per JSONL line.

    The lazy iterator lets ``replay`` early-exit on first divergence without
    materializing the whole log — important for long games (104 ticks at
    schema v10 is ~2 KB but Phase 2.1+ may grow).

    A trailing blank line (the operator's editor adding a newline) is
    silently tolerated. A genuinely truncated last line (a crash mid-write)
    surfaces as :class:`ValueError` from :meth:`ActionLogRecord.from_json` —
    the reader does NOT swallow it; the AC-4 invariant prefers loud failure.
    """
    if not log_path.exists():
        raise FileNotFoundError(
            f"read_action_log: no action log at {log_path}; "
            f"the file is written by the live server per tick"
        )
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                # Blank line: a trailing editor newline / whitespace; not a record.
                continue
            yield ActionLogRecord.from_json(stripped)


def replay(manifest_path: Path, log_path: Path) -> ReplayResult:
    """ """
    manifest = read_manifest(manifest_path)

    # The Slice A reconstruction (CoreConfig.default()); Slice B+ may switch
    # on manifest.config_summary to rehydrate a richer scenario.
    env = RetailParallelEnv(
        config=CoreConfig.default(),
        n_learning_agents=manifest.n_agents,
        npc_archetypes=manifest.npc_archetypes,
        seed=manifest.seed,
    )
    env.reset(seed=manifest.seed)

    n_processed = 0
    for record in read_action_log(log_path):
        # Build the joint action dict for the env in the same agent-id keying
        # the live server used. The live server's action log is keyed by
        # ``retailer_<seat>`` (the parallel env's agent_id); we cast each
        # list[float] back to a fresh float32 ndarray.
        joint_action = {
            agent_id: np.asarray(action, dtype=np.float32)
            for agent_id, action in record.joint_action.items()
        }
        obs, rewards, _terms, _truncs, _infos = env.step(joint_action)

        # Per-seat verification: compare each logged ``(obs, reward)`` against
        # the just-stepped env. Bit-exact via ``np.array_equal`` for obs +
        # exact ``==`` for the scalar reward (both must round-trip bit-for-bit
        # under JSON per the protocol's float-precision invariant).
        for agent_id, logged_obs_list in record.observations.items():
            logged_obs = np.asarray(logged_obs_list, dtype=np.float32)
            live_obs = obs[agent_id]
            if not np.array_equal(logged_obs, live_obs):
                return ReplayResult(
                    ok=False,
                    n_ticks_replayed=n_processed + 1,
                    first_divergence_tick=record.tick,
                    divergence_summary=(
                        f"observation divergence at tick {record.tick} seat {agent_id}: "
                        f"first delta at dim "
                        f"{int(np.argmax(logged_obs != live_obs))} "
                        f"logged={logged_obs[np.argmax(logged_obs != live_obs)]!r} "
                        f"live={live_obs[np.argmax(logged_obs != live_obs)]!r}"
                    ),
                )

            logged_reward = record.rewards[agent_id]
            live_reward = rewards[agent_id]
            if logged_reward != live_reward:
                return ReplayResult(
                    ok=False,
                    n_ticks_replayed=n_processed + 1,
                    first_divergence_tick=record.tick,
                    divergence_summary=(
                        f"reward divergence at tick {record.tick} seat {agent_id}: "
                        f"logged={logged_reward!r} live={live_reward!r}"
                    ),
                )

        n_processed += 1

    return ReplayResult(
        ok=True,
        n_ticks_replayed=n_processed,
        first_divergence_tick=None,
        divergence_summary=None,
    )


def replay_game_dir(game_dir: Path, *, strict: bool = True) -> ReplayResult:
    """ """
    # Boundary validation: the game_dir argument is the operator's
    # entry point — a typo'd path should surface with a greppable error
    # naming what's missing (same discipline as :func:`read_manifest`).
    if not game_dir.is_dir():
        raise FileNotFoundError(f"game_dir not found: {game_dir}")

    manifest_path = game_dir / "game_manifest.json"
    log_path = game_dir / "action_log.jsonl"

    # Surface each missing canonical file SEPARATELY so the operator knows
    # which half of the per-game artifact pair to investigate (the live
    # server writes both; missing manifest ≠ missing log in terms of root
    # cause — manifest is written at game start, log per-tick).
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest at {manifest_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"missing action log at {log_path}")

    result = replay(manifest_path, log_path)

    if not strict:
        # ``strict=False`` is the crashed-server-audit relaxation: accept
        # whatever ticks were captured; the partial trajectory is the
        # operator's evidence and the manifest's n_ticks is advisory.
        return result

    # ``strict=True``: an OK kernel result with fewer ticks than the
    # manifest promised is a truncation — upgrade to a divergence so the
    # CLI surfaces a non-zero exit. Genuine kernel divergences are left
    # untouched (the kernel's first-divergence diagnostic is the right
    # one to surface; we only fill in the truncation case the kernel
    # itself does not detect).
    if result.ok:
        manifest = read_manifest(manifest_path)
        if result.n_ticks_replayed < manifest.n_ticks:
            return ReplayResult(
                ok=False,
                n_ticks_replayed=result.n_ticks_replayed,
                first_divergence_tick=result.n_ticks_replayed,
                divergence_summary=(
                    f"action log truncated at tick {result.n_ticks_replayed} — "
                    f"manifest expected {manifest.n_ticks}"
                ),
            )
    return result


__all__ = [
    "ActionLogRecord",
    "ReplayManifest",
    "ReplayResult",
    "read_action_log",
    "read_manifest",
    "replay",
    "replay_game_dir",
]
