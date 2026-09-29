"""Determinism: a single injected PCG64 generator and serializable RNG state.

The sim uses **one** ``numpy.default_rng(seed)`` (PCG64), injected explicitly and
threaded through every stochastic step. There is no module-level randomness here
and core/ never touches ``random`` / ``os.urandom`` / module-level
``numpy.random`` — same seed + same actions must reproduce the same trajectory
bit-for-bit (Linux, pinned NumPy major).

Per-tick DRAW ORDER contract (the *only* source of stochasticity ordering;
``world.py`` must honor it exactly):
    1. Demand resolution noise.
    2. NPC actions, in agent-index order (``WorldState.retailers`` order).
    3. Accounting noise (none in Phase 0.0).
    4. Perception noise for the perceived competitor view — ONE constant-size
       vectorized draw, drawn LAST (after all economic draws / the next-state build).
       Phase 1.0. Its RNG consumption is FIXED per tick (shape = n_observers ×
       n_noised_competitor_scalars), INDEPENDENT of research fidelity and of which
       fields are noised: fidelity scales the noise MAGNITUDE only, never the draw
       COUNT. Because it is last and constant-size, draws #1–#3 (the entire economic
       trajectory) consume the identical RNG stream they did in 0.5 — so the economics
       are byte-identical to 0.5 at the research default AND across research levels.

Phase 1.1 — the SCM lever ADDS NO NEW RNG DRAW. The per-region fill-rate cap is a
closed-form function of the decoded ``service_level`` (``fill_rate = floor + (1 -
floor) * s``); the per-region ``service_score`` recurrence is a deterministic EMA. The
draw order #1–#4 + the per-tick draw count are UNCHANGED from 1.0; the 1.0 constant-
size draw-#4 guarantee is preserved unconditionally.

Phase 1.2 (F3 = PERMANENT) — the automation lever ADDS NO NEW RNG DRAW. The
upgrade gate is a deterministic ledger (monotonicity + differential-affordability
check on PRE-capex cash); the seated ``RetailerState.automation_tier`` advances
deterministically to ``target`` on a valid upgrade or stays at ``prev`` on a no-op
(no EMA, no decay); the lagged COGS multiplier is a closed-form deterministic
indexing ``savings_per_tier[automation_tier_prev]``. The draw order #1–#4 + the
per-tick draw count are UNCHANGED from 1.0/1.1; the 1.0 constant-size draw-#4
guarantee is preserved unconditionally.

``snapshot`` / ``restore`` round-trip ``rng.bit_generator.state`` so that
``(WorldState, snapshot(rng))`` fully serializes the sim for replay and Phase-2
checkpointing.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Construct the canonical PCG64 generator.

    ``seed=None`` draws fresh OS entropy (non-reproducible) — callers that need
    determinism must pass an explicit integer seed.
    """
    return np.random.default_rng(seed)


def snapshot(rng: np.random.Generator) -> dict[str, Any]:
    """Capture the generator's bit-generator state as plain serializable data.

    Returns a deep copy so later draws on ``rng`` cannot mutate the snapshot.
    """
    import copy

    return copy.deepcopy(dict(rng.bit_generator.state))


def restore(state: dict[str, Any]) -> np.random.Generator:
    """Rebuild a generator positioned exactly where ``snapshot`` was taken.

    Subsequent draws reproduce the snapshotted stream bit-for-bit. We start from
    a PCG64 generator and overwrite its state, so the restored stream is
    independent of whatever seed seeds the placeholder.
    """
    import copy

    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state)
    return rng
