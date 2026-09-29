"""Reward model: profit-only primitive + the Phase 0.5 weighted/z-normalized reward.

The PROFIT-ONLY primitive is unchanged: :func:`compute_reward` returns the agent's raw
profit (the 0.0–0.4 reward, no weights / no normalization). :func:`reward_components`
returns the four un-normalized components ``{profit, revenue, market_share, loyalty}``
for ``info['components']`` (always un-normalized — no schema bump); as of
F-reward-post-penalty (2026-09-14) its optional ``writeoff``/``penalty`` kwargs let a
caller fold ``World.step``'s two post-accounting P&L cost lines into ``profit`` from
ONE built object, before either the reward path or ``info['components']`` reads it
(see :func:`compute_reward_for_seat`).

Phase 0.5 adds, ON TOP, the configurable weighted reward (selected by
:class:`retail_simulator.core.config.RewardConfig`):

* :class:`RewardNormalizer` — one :class:`RunningStats` per component, advanced ONCE
  per tick inside ``World.step`` AFTER accounting (a deterministic post-accounting
  fold; consumes NO RNG). It z-normalizes each component against ITS running stats
  (update-then-read, σ-floored, warmup-/clamp-guarded) then applies the locked
  weights. It lives in ``WorldState.reward_stats`` (one per LEARNING seat), so it
  serializes with ``(WorldState, rng_state)`` and replays bit-for-bit. It serializes
  to/from plain numeric triples (``count, mean, m2`` per component).
* :func:`compute_reward_for_seat` — the mode-aware seam helper over a CALLER-BUILT
  :class:`RewardComponents` (profit ⇒ ``components.profit`` directly; weighted ⇒ the
  normalizer's z-sum over ``components``).
* :func:`weighted_objective` — the pure, STATIONARY un-normalized weighted combo the
  GATE scores (NOT the moving normalized reward).
* :func:`reward_variance_balance` — the soft reward-health check (per-component
  variance over a rollout + the max/min ratio; flags a zero-variance component).

Pure and framework-free: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retail_simulator.core.accounting import AccountingResult
from retail_simulator.core.config import RewardConfig
from retail_simulator.core.demand import DemandResult

_Z_CLAMP: float = 10.0
# Standard-deviation floor: below this σ is treated as "not yet meaningful" and the
# component's z-score is 0.0 (warmup / constant-component guard; no div-by-zero).
_STD_EPS: float = 1e-8

# Canonical component order — the keys the normalizer/objective iterate, matching
# RewardComponents and the RewardConfig weights.
_COMPONENTS: tuple[str, ...] = ("profit", "revenue", "market_share", "loyalty")


@dataclass(frozen=True)
class RewardComponents:
    """Un-normalized reward components for one retailer (for ``info``).

    Phase 0.0 reward equals ``profit`` alone; the other three are carried for
    inspection and for the Phase 0.5 weighted/normalized reward.
    """

    profit: float
    revenue: float
    market_share: float
    loyalty: float

    def as_dict(self) -> dict[str, float]:
        """The four components keyed by name (the ``info['components']`` shape)."""
        return {
            "profit": self.profit,
            "revenue": self.revenue,
            "market_share": self.market_share,
            "loyalty": self.loyalty,
        }


def reward_components(
    result: AccountingResult,
    demand: DemandResult,
    agent_index: int,
    *,
    writeoff: float = 0.0,
    penalty: float = 0.0,
) -> RewardComponents:
    """ """
    units = np.asarray(demand.units, dtype=np.float64)  # [R, K]
    total_demand = np.asarray(demand.total_demand, dtype=np.float64)  # [R]
    loyalty = np.asarray(result.new_loyalty, dtype=np.float64)  # [R, K]

    total_units = float(units.sum())
    agent_units = float(units[:, agent_index].sum())
    # Demand-weighted overall share = agent units / total units across regions. Guard
    # the no-demand tick (every region's pie 0 — e.g. priced out): report 0.0 share.
    overall_share = agent_units / total_units if total_units > 0.0 else 0.0

    region_pie = float(total_demand.sum())
    if region_pie > 0.0:
        # Weight each region's loyalty by its realized pie -> a value in [0, 1].
        overall_loyalty = float((total_demand @ loyalty[:, agent_index]) / region_pie)
    else:
        # No realized demand anywhere: fall back to a plain mean across regions (still
        # in [0, 1] since each per-region loyalty is in [0, 1]).
        overall_loyalty = float(loyalty[:, agent_index].mean())

    profit = float(result.profit[agent_index])
    if writeoff:
        profit -= writeoff
    if penalty:
        profit -= penalty

    return RewardComponents(
        profit=profit,
        revenue=float(result.revenue[agent_index]),
        market_share=overall_share,
        loyalty=overall_loyalty,
    )


def compute_reward(result: AccountingResult, agent_index: int) -> float:
    """Phase 0.0 reward for one retailer: its raw profit this tick.

    No weighting or normalization (see module docstring). Returns a plain float.
    """
    return float(result.profit[agent_index])


@dataclass
class RunningStats:
    """Welford online mean/variance — SCAFFOLD for Phase 0.5 z-normalization.

    Numerically stable single-pass mean/variance. NOT used by the Phase 0.0
    reward; provided so the 0.5 per-component z-norm has a deterministic,
    serializable accumulator (its fields can be stored in ``WorldState``).

    ``variance``/``std`` use the population (biased) estimator, which is the
    conventional choice for running normalization.
    """

    count: int = 0
    mean: float = 0.0
    # Sum of squared deviations from the running mean (Welford's M2).
    m2: float = 0.0

    def update(self, value: float) -> None:
        """Fold one observation into the running statistics."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Population variance (0.0 until at least one value is seen)."""
        if self.count == 0:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return self.variance**0.5

    def as_triple(self) -> tuple[int, float, float]:
        """Plain numeric ``(count, mean, m2)`` for serialization (round-trips)."""
        return (self.count, self.mean, self.m2)

    @classmethod
    def from_triple(cls, triple: tuple[int, float, float]) -> RunningStats:
        """Rebuild from a ``(count, mean, m2)`` triple (the inverse of :meth:`as_triple`)."""
        count, mean, m2 = triple
        return cls(count=int(count), mean=float(mean), m2=float(m2))


def _zscore(value: float, stats: RunningStats, warmup: int) -> float:
    """ """
    if stats.count <= warmup:
        return 0.0
    std = stats.std
    if std <= _STD_EPS:
        return 0.0
    z = (value - stats.mean) / std
    return float(min(max(z, -_Z_CLAMP), _Z_CLAMP))


@dataclass
class RewardNormalizer:
    """ """

    profit: RunningStats = field(default_factory=RunningStats)
    revenue: RunningStats = field(default_factory=RunningStats)
    market_share: RunningStats = field(default_factory=RunningStats)
    loyalty: RunningStats = field(default_factory=RunningStats)

    def update_and_normalize(self, components: RewardComponents, cfg: RewardConfig) -> float:
        """ """
        warmup = cfg.normalizer_warmup
        weights = {
            "profit": cfg.weight_profit,
            "revenue": cfg.weight_revenue,
            "market_share": cfg.weight_market_share,
            "loyalty": cfg.weight_loyalty,
        }
        total = 0.0
        for name in _COMPONENTS:
            stats: RunningStats = getattr(self, name)
            value = float(getattr(components, name))
            stats.update(value)  # update-then-read: fold this tick first
            total += weights[name] * _zscore(value, stats, warmup)
        return float(total)

    def to_triples(self) -> dict[str, tuple[int, float, float]]:
        """Serialize to plain ``{component: (count, mean, m2)}`` (round-trips)."""
        return {name: getattr(self, name).as_triple() for name in _COMPONENTS}

    @classmethod
    def from_triples(cls, triples: dict[str, tuple[int, float, float]]) -> RewardNormalizer:
        """Rebuild from the :meth:`to_triples` mapping (the inverse)."""
        return cls(**{name: RunningStats.from_triple(triples[name]) for name in _COMPONENTS})


def weighted_objective(components: RewardComponents, cfg: RewardConfig) -> float:
    """ """
    if cfg.mode == "profit":
        return float(components.profit)
    return float(
        cfg.weight_profit * components.profit
        + cfg.weight_revenue * components.revenue
        + cfg.weight_market_share * components.market_share
        + cfg.weight_loyalty * components.loyalty
    )


def compute_reward_for_seat(
    components: RewardComponents,
    cfg: RewardConfig,
    normalizer: RewardNormalizer | None,
    *,
    seat: int | None = None,
) -> float:
    """ """
    if cfg.mode == "profit":
        return float(components.profit)
    if normalizer is None:
        suffix = "" if seat is None else f" for seat {seat}"
        raise ValueError(f"weighted reward requires a RewardNormalizer{suffix}")
    return normalizer.update_and_normalize(components, cfg)


def reward_variance_balance(
    component_trajectories: dict[str, list[float]],
) -> dict[str, float]:
    """ """
    if not component_trajectories:
        raise ValueError("component_trajectories must not be empty")
    variances: dict[str, float] = {}
    for name, trajectory in component_trajectories.items():
        if not trajectory:
            raise ValueError(f"component trajectory for {name!r} must not be empty")
        variances[name] = float(np.var(np.asarray(trajectory, dtype=np.float64)))

    values = list(variances.values())
    max_var = max(values)
    min_var = min(values)
    any_zero = any(v <= _STD_EPS for v in values)

    if min_var <= _STD_EPS:
        # A degenerate (zero-variance) component: ratio is unbounded unless EVERY
        # component is degenerate (then the ratio is undefined → report 0.0/1.0).
        ratio = float("inf") if max_var > _STD_EPS else 0.0
    else:
        ratio = max_var / min_var

    result: dict[str, float] = dict(variances)
    result["max_min_ratio"] = ratio
    result["zero_variance"] = 1.0 if any_zero else 0.0
    return result
