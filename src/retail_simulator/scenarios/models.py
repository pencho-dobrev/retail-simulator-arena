"""``ScenarioConfig`` is the YAML-backed, boundary-validated surface the Operator edits
(REQUIREMENTS: edit scenario YAML → validate → calibrate → gate → competition). It
maps a small, YAGNI-minimal set of operator knobs onto the framework-free
:class:`~retail_simulator.core.config.CoreConfig` + a seat plan
(:class:`~retail_simulator.core.config.SeatSpec` tuple), reusing ``core``'s validation
(``validate_expansion_capacity``, ``_validate_segment_mix``, ``STARTING_CASH``,
``NPC_ARCHETYPES``) — it re-implements NO world rules.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from retail_simulator.core.config import (
    NPC_ARCHETYPES,
    CoreConfig,
    CostConfig,
    DemandConfig,
    ExpansionConfig,
    RegionConfig,
    RewardConfig,
    SeatSpec,
    _validate_segment_mix,
)
from retail_simulator.core.schema import N_REGIONS_MAX, validate_expansion_capacity
from retail_simulator.core.state import (
    BRAND_LOYAL,
    CONVENIENCE,
    PRICE_SENSITIVE,
    QUALITY_SEEKING,
    SegmentParams,
    default_segments,
)
from retail_simulator.core.world import STARTING_CASH

_REGION_0_DEMAND: float = 1000.0
_REGION_1_DEMAND: float = 600.0
_REGION_0_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.4,
    QUALITY_SEEKING: 0.25,
    CONVENIENCE: 0.2,
    BRAND_LOYAL: 0.15,
}
_REGION_1_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.25,
    QUALITY_SEEKING: 0.15,
    CONVENIENCE: 0.15,
    BRAND_LOYAL: 0.45,
}
_DEFAULT_PRICE_ELASTICITY: float = -1.5
_DEFAULT_COGS_FRACTION: float = 0.4
_DEFAULT_FIXED_OPEX_PER_STORE: float = 50.0
_DEFAULT_EXPANSION_CAPEX: float = 5_000.0
_COMPETITION_BETA_REFERENCE: float = 1.5


class RegionSpec(BaseModel):
    """One region's operator-overridable demand knobs (maps to ``RegionConfig``).

    ``base_regional_demand`` is the region's pie; ``segment_mix`` splits it by segment
    and MUST sum to 1.0 (the same invariant ``core``'s ``_validate_segment_mix``
    enforces — reused here so a malformed mix is rejected at the boundary, not silently
    absorbed into broken share invariants).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_regional_demand: float = Field(gt=0.0)
    segment_mix: dict[str, float]

    @field_validator("segment_mix")
    @classmethod
    def _mix_sums_to_one(cls, mix: dict[str, float]) -> dict[str, float]:
        # Reuse core's segment-mix semantics (non-empty, non-negative, sums to 1.0)
        # so the scenario surface and the world model validate IDENTICALLY.
        _validate_segment_mix(mix)
        return mix

    def to_region_config(self) -> RegionConfig:
        """Build the framework-free ``RegionConfig`` (a fresh dict; never aliased)."""
        return RegionConfig(
            base_regional_demand=self.base_regional_demand,
            segment_mix=dict(self.segment_mix),
        )


class RewardWeights(BaseModel):
    """The four DESIGN-locked reward weights ({.5/.2/.2/.1}); sum-to-1.0 if weighted.

    Defaults reproduce ``RewardConfig``'s locked weights. The sum-to-1.0 check is
    cross-field with ``reward_mode`` (a profit-mode scenario never reads them), so it is
    enforced in ``ScenarioConfig``'s collect-all validator, not here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profit: float = 0.5
    revenue: float = 0.2
    market_share: float = 0.2
    loyalty: float = 0.1

    def total(self) -> float:
        """Sum of the four weights (the sum-to-1.0 invariant operates on this)."""
        return self.profit + self.revenue + self.market_share + self.loyalty


class ScenarioConfig(BaseModel):
    """ """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    seed: int = 42
    n_regions: int = Field(default=2, ge=1, le=N_REGIONS_MAX)
    n_learning_agents: int = Field(default=2, ge=1)
    npc_archetypes: tuple[str, ...] = ("discounter",)
    reward_mode: Literal["profit", "weighted"] = "weighted"
    reward_weights: RewardWeights = RewardWeights()
    # Per-region demand knobs (operator-overridable). Default = the 0.4 two-region
    # plan (region 0 full size 4-segment, region 1 asymmetric brand-loyal-heavy).
    regions: tuple[RegionSpec, ...] = (
        RegionSpec(base_regional_demand=_REGION_0_DEMAND, segment_mix=_REGION_0_MIX),
        RegionSpec(base_regional_demand=_REGION_1_DEMAND, segment_mix=_REGION_1_MIX),
    )
    # Cost / elasticity / expansion knobs (operator-overridable; defaults = 0.4).
    price_elasticity: float = Field(default=_DEFAULT_PRICE_ELASTICITY, lt=0.0)
    cogs_fraction: float = Field(default=_DEFAULT_COGS_FRACTION, ge=0.0, le=1.0)
    fixed_opex_per_store: float = Field(default=_DEFAULT_FIXED_OPEX_PER_STORE, ge=0.0)
    expansion_capex: float = Field(default=_DEFAULT_EXPANSION_CAPEX, ge=0.0)

    @field_validator("npc_archetypes")
    @classmethod
    def _archetypes_known(cls, archetypes: tuple[str, ...]) -> tuple[str, ...]:
        # Every NPC seat must name a known archetype (the same set core's SeatSpec
        # validates against). Reported per-offending-name so an operator sees exactly
        # which entry is wrong.
        unknown = [a for a in archetypes if a not in NPC_ARCHETYPES]
        if unknown:
            raise ValueError(
                f"unknown NPC archetype(s) {unknown}; known archetypes: {list(NPC_ARCHETYPES)}"
            )
        return archetypes

    @model_validator(mode="after")
    def _validate_boundaries(self) -> ScenarioConfig:
        """ """
        errors: list[str] = []

        # n_regions must fit the expansion lever's fixed capacity (schema v12: a
        # ceiling, not an equality — pydantic's own Field(le=N_REGIONS_MAX) above
        # already rejects > N_REGIONS_MAX, this call is the defensive analog of
        # validate_automation_width's construction-time check) AND must equal the
        # number of region specs supplied.
        try:
            validate_expansion_capacity(self.n_regions)
        except ValueError as exc:
            errors.append(str(exc))
        if len(self.regions) != self.n_regions:
            errors.append(
                f"n_regions={self.n_regions} must equal the number of region specs "
                f"({len(self.regions)})"
            )

        # Weighted reward weights must sum to 1.0 (a profit-mode scenario never reads
        # them, so only enforce when weighted).
        if self.reward_mode == "weighted":
            total = self.reward_weights.total()
            if abs(total - 1.0) > 1e-9:
                errors.append(
                    f"reward_weights must sum to 1.0 when reward_mode='weighted', got {total}"
                )

        if self.n_learning_agents < 1:
            errors.append("n_learning_agents must be >= 1 (at least one learning seat)")

        # Starting cash must cover the expansion capex, or the expansion lever is dead
        # (the same invariant World construction enforces).
        if STARTING_CASH < self.expansion_capex:
            errors.append(
                f"STARTING_CASH ({STARTING_CASH}) must be >= expansion_capex "
                f"({self.expansion_capex}) or the expansion lever is dead"
            )

        if errors:
            joined = "; ".join(errors)
            raise ValueError(f"invalid scenario {self.name!r}: {joined}")
        return self

    def seat_plan(self) -> tuple[SeatSpec, ...]:
        """Build the seat plan: ``n_learning_agents`` learners then the NPC seats.

        Seat 0 is always a learning seat (``AGENT_INDEX = 0`` for the Gym view).
        Presence is left ``None`` so ``world.reset``'s seat-kind defaults apply (learner
        in region 0, NPC in every region). Phase 6.0: a region no seat is ever present
        in (e.g. an all-learning scenario with ``n_regions`` > 1 and no NPC) is now a
        legal, if uncontested, empty region rather than a rejected boundary case.
        """
        learners = tuple(SeatSpec(is_npc=False) for _ in range(self.n_learning_agents))
        npcs = tuple(SeatSpec(is_npc=True, archetype=name) for name in self.npc_archetypes)
        return learners + npcs

    def competition_segments(self) -> dict[str, SegmentParams]:
        """ """
        segments = default_segments()
        price_sensitive = segments[PRICE_SENSITIVE]
        segments[PRICE_SENSITIVE] = replace(
            price_sensitive, beta_reference=_COMPETITION_BETA_REFERENCE
        )
        return segments

    def to_core_config(self) -> CoreConfig:
        """ """
        demand = DemandConfig(
            base_regional_demand=self.regions[0].base_regional_demand,
            price_elasticity=self.price_elasticity,
            segment_mix=dict(self.regions[0].segment_mix),
            regions=tuple(region.to_region_config() for region in self.regions),
        )
        cost = CostConfig(
            cogs_fraction=self.cogs_fraction,
            fixed_opex_per_store=self.fixed_opex_per_store,
        )
        reward = RewardConfig(
            mode=self.reward_mode,
            weight_profit=self.reward_weights.profit,
            weight_revenue=self.reward_weights.revenue,
            weight_market_share=self.reward_weights.market_share,
            weight_loyalty=self.reward_weights.loyalty,
        )
        expansion = ExpansionConfig(expansion_capex=self.expansion_capex)
        return CoreConfig(
            demand=demand,
            cost=cost,
            reward=reward,
            expansion=expansion,
            seats=self.seat_plan(),
            segments=self.competition_segments(),
        )

    @classmethod
    def default(cls) -> ScenarioConfig:
        """The competition default: weighted reward, 2 learning agents, 0.4 economics.

        All other fields take their class defaults (a single discounter NPC, the
        two-region 0.4 demand plan, the 0.4 cost/expansion knobs), so
        ``default().to_core_config()`` reproduces the accepted 0.4 economics with the
        0.5 weighted reward selected.
        """
        return cls(name="competition-default")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioConfig:
        """ """
        text = Path(path).read_text()
        data = yaml.safe_load(text)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(
                f"scenario YAML at {path} must be a mapping of fields, got {type(data).__name__}"
            )
        return cls(**data)


__all__ = ["RegionSpec", "RewardWeights", "ScenarioConfig", "ValidationError"]
