"""The NPC calibrate loop — measure each archetype's competence vs random (Phase 0.5).

Plus :func:`reward_variance_check` — a reward variance-balance / boundedness signal over
a random rollout (reuses :func:`~retail_simulator.core.reward.reward_variance_balance`):
the four un-normalized components' variances within ~3× and no degenerate (zero-variance)
component. REPORTED, NEVER gates (like the diversity report).

Layer note: ``harness -> {envs, core}``; this module imports ONLY ``core`` (the fast seam
+ the scripted policies + the reward health check) — no RL framework, no envs. Numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from retail_simulator.core.config import CoreConfig, SeatSpec
from retail_simulator.core.npc import Balanced, Discounter, NPCPolicy, Premium
from retail_simulator.core.reward import reward_variance_balance
from retail_simulator.core.schema import ActionSchema, default_action_schema
from retail_simulator.core.world import World

# The scoring horizon (~2 simulated years) — the same window the gates score over.
DEFAULT_HORIZON_TICKS: int = 104
# Seeds averaged per archetype to tame the multiplicative demand noise (~0.05).
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
COMPETENCE_BAND_LOW: float = 0.20
COMPETENCE_BAND_HIGH: float = 0.65

# The archetype-under-test seat + its random opponent seat in the 2-seat measurement.
_ARCHETYPE_SEAT: int = 0
_OPPONENT_SEAT: int = 1

PRIMARY_METRIC: dict[str, str] = {
    "discounter": "market_share",
    "premium": "profit",
    "balanced": "market_share",
}


def _make_archetype(name: str, schema: ActionSchema) -> NPCPolicy:
    """Instantiate the scripted policy for an archetype name (the calibrate roster).

    Mirrors ``world._make_npc_policy`` but local to the harness (the calibrate loop is a
    measurement tool, not the seam) so it can drive an archetype's ``act`` directly into
    a LEARNING seat's joint action. Each policy is built with the world's schema so its
    encoded action matches the contract the seam decodes.
    """
    if name == "discounter":
        return Discounter(schema=schema)
    if name == "premium":
        return Premium(schema=schema)
    if name == "balanced":
        return Balanced(schema=schema)
    raise ValueError(f"unknown NPC archetype {name!r}")


def _two_learning_seat_config(config: CoreConfig | None) -> CoreConfig:
    """ """
    base = config if config is not None else CoreConfig.default()
    region_cfgs = base.demand.regions
    n_regions = len(region_cfgs) if region_cfgs else 1
    seats = (
        SeatSpec(is_npc=False, presence=(1,) * n_regions),
        SeatSpec(is_npc=False),
    )
    return replace(base, seats=seats)


def _score_seat0_metric(
    config: CoreConfig,
    *,
    seat0_policy: NPCPolicy | None,
    seed: int,
    horizon: int,
    schema: ActionSchema,
) -> dict[str, float]:
    """Run one rollout; return seat 0's mean per-tick reward components.

    Seat 0 is driven by ``seat0_policy`` (an archetype's ``act``) when given, else by a
    RANDOM agent (the baseline). Seat 1 is ALWAYS a RANDOM opponent (a fixed-seed random
    field, so the archetype and its baseline face the SAME opponent distribution). The
    rollout reads seat 0's UN-normalized ``info['components']`` each tick and returns the
    per-component MEAN over the horizon — the basis for the primary-metric margin.
    """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    # A seeded action sampler for the random seats (seat 1 always; seat 0 in baseline).
    action_space_rng = np.random.default_rng(seed + 10_000)
    action_low, action_high = _action_bounds(schema)

    sums = {"profit": 0.0, "revenue": 0.0, "market_share": 0.0, "loyalty": 0.0}
    for _ in range(horizon):
        if seat0_policy is not None:
            a0 = seat0_policy.act(state, world.rng, _ARCHETYPE_SEAT)
        else:
            a0 = _sample_action(action_space_rng, action_low, action_high)
        a1 = _sample_action(action_space_rng, action_low, action_high)
        result = world.step(state, {_ARCHETYPE_SEAT: a0, _OPPONENT_SEAT: a1})
        components = result.infos[_ARCHETYPE_SEAT]["components"]
        assert isinstance(components, dict)
        for key in sums:
            sums[key] += float(components[key])
        state = result.next_state
    return {key: value / horizon for key, value in sums.items()}


def _action_bounds(schema: ActionSchema) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension (low, high) of the flat action layout, for a random sampler.

    Mirrors the env's action Box bounds WITHOUT importing ``envs`` (the calibrate loop
    stays core-only): a continuous lever spans ``[low, high]``; the trailing discrete
    expansion block is sampled in ``[0, 1]`` (its logits, argmax-decoded). The random
    seat's action is decoded + clipped by the seam anyway, so this only needs to span the
    representative range.
    """
    lows: list[float] = []
    highs: list[float] = []
    for lever in schema.levers:
        if lever.low is not None and lever.high is not None:
            lows.append(float(lever.low))
            highs.append(float(lever.high))
    # The trailing dims past the continuous levers are the discrete expansion logit
    # block (argmax-decoded); sample each logit in [0, 1]. total_dim is the SoT for the
    # flat action width, so the block width is total_dim - (continuous lever count).
    block = schema.total_dim - len(lows)
    lows.extend(0.0 for _ in range(block))
    highs.extend(1.0 for _ in range(block))
    return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)


def _sample_action(rng: np.random.Generator, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Sample one flat action uniformly in ``[low, high]`` (the random baseline action)."""
    return rng.uniform(low, high).astype(np.float32)


@dataclass(frozen=True)
class ArchetypeCalibration:
    """ """

    archetype: str
    primary_metric: str
    archetype_score: float
    random_score: float
    margin: float
    beats_random: bool
    in_band: bool


def calibrate_archetype(
    archetype: str,
    *,
    config: CoreConfig | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    horizon: int = DEFAULT_HORIZON_TICKS,
) -> ArchetypeCalibration:
    """Measure one archetype's beat-random margin on its primary metric (the fast seam).

    Runs the archetype in seat 0 vs a random opponent, and a random agent in seat 0 vs
    the SAME opponent, over ``seeds`` rollouts; the margin is the fractional improvement
    of the archetype's mean primary metric over the random baseline's. NO PPO — the pure
    :class:`World` seam only, so it is fast. Returns an :class:`ArchetypeCalibration`;
    the orchestrator tunes the archetype's (module-constant) lever values until the
    margin lands in the 20–65% band. Raises on an unknown archetype name.
    """
    if archetype not in PRIMARY_METRIC:
        raise ValueError(f"unknown archetype {archetype!r}; known: {sorted(PRIMARY_METRIC)}")
    schema = default_action_schema()
    cfg = _two_learning_seat_config(config)
    policy = _make_archetype(archetype, schema)
    metric = PRIMARY_METRIC[archetype]

    archetype_values: list[float] = []
    random_values: list[float] = []
    for seed in seeds:
        arch = _score_seat0_metric(
            cfg, seat0_policy=policy, seed=seed, horizon=horizon, schema=schema
        )
        rand = _score_seat0_metric(
            cfg, seat0_policy=None, seed=seed, horizon=horizon, schema=schema
        )
        archetype_values.append(arch[metric])
        random_values.append(rand[metric])

    archetype_score = float(np.mean(archetype_values))
    random_score = float(np.mean(random_values))
    margin = _fractional_margin(archetype_score, random_score)
    in_band = COMPETENCE_BAND_LOW <= margin <= COMPETENCE_BAND_HIGH
    return ArchetypeCalibration(
        archetype=archetype,
        primary_metric=metric,
        archetype_score=archetype_score,
        random_score=random_score,
        margin=margin,
        beats_random=margin > 0.0,
        in_band=in_band,
    )


def _fractional_margin(archetype_score: float, random_score: float) -> float:
    """``(archetype − random) / |random|`` — the fractional beat-random margin.

    ``inf`` when the random baseline is ~0 and the archetype scores positive (an
    unbounded relative beat); 0.0 when both are ~0 (no signal either way).
    """
    if abs(random_score) <= 1e-12:
        return float("inf") if archetype_score > 0.0 else 0.0
    return (archetype_score - random_score) / abs(random_score)


def calibrate_all(
    *,
    config: CoreConfig | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    horizon: int = DEFAULT_HORIZON_TICKS,
) -> dict[str, ArchetypeCalibration]:
    """ """
    return {
        archetype: calibrate_archetype(archetype, config=config, seeds=seeds, horizon=horizon)
        for archetype in PRIMARY_METRIC
    }


def reward_variance_check(
    *,
    config: CoreConfig | None = None,
    seed: int = 0,
    horizon: int = DEFAULT_HORIZON_TICKS,
) -> dict[str, float]:
    """ """
    schema = default_action_schema()
    cfg = _two_learning_seat_config(config)
    world = World(config=cfg, seed=seed)
    state = world.reset(seed)
    action_rng = np.random.default_rng(seed + 10_000)
    low, high = _action_bounds(schema)

    trajectories: dict[str, list[float]] = {
        "profit": [],
        "revenue": [],
        "market_share": [],
        "loyalty": [],
    }
    for _ in range(horizon):
        a0 = _sample_action(action_rng, low, high)
        a1 = _sample_action(action_rng, low, high)
        result = world.step(state, {_ARCHETYPE_SEAT: a0, _OPPONENT_SEAT: a1})
        # The seam already exposes seat 0's UN-normalized components in info (the SoT);
        # collect them straight off the StepResult — no recomputation.
        components = result.infos[_ARCHETYPE_SEAT]["components"]
        assert isinstance(components, dict)
        for key in trajectories:
            trajectories[key].append(float(components[key]))
        state = result.next_state
    return reward_variance_balance(trajectories)
