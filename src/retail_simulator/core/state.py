"""World state data structures — plain data, no behavior.

These dataclasses hold the *entire* mutable world (plus the RNG state, threaded
separately via ``rng.py``) so that ``(WorldState, rng.bit_generator.state)``
fully serializes the sim for replay and Phase-2 checkpointing.

Strictly data: no stepping, no demand math, no RNG draws. ``world.py`` (Wave 4)
is the only place that advances a ``WorldState``. Frozen where a value is fixed
for the world's lifetime (params); mutable where the simulation updates it each
tick (retailer/world running state).

numpy is permitted here, but Phase 0.0 needs only scalars; we keep fields as
plain Python floats/ints so ``WorldState`` is trivially serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported for typing only — keeps state.py runtime-dependency-free (avoids the
    # reward -> config -> state import cycle). The annotation is a string under
    # `from __future__ import annotations`, so it is never evaluated at import time;
    # the field's default_factory (dict) needs no RewardNormalizer reference.
    from retail_simulator.core.reward import RewardNormalizer

PRICE_SENSITIVE = "price_sensitive"
QUALITY_SEEKING = "quality_seeking"
CONVENIENCE = "convenience"
BRAND_LOYAL = "brand_loyal"
PREMIUM_HUNTER = "premium_hunter"
BARGAIN_SEEKER = "bargain_seeker"


SEGMENT_KEYS: frozenset[str] = frozenset(
    {
        PRICE_SENSITIVE,
        QUALITY_SEEKING,
        CONVENIENCE,
        BRAND_LOYAL,
        PREMIUM_HUNTER,
        BARGAIN_SEEKER,
    }
)


@dataclass(frozen=True)
class SegmentParams:
    """Attractiveness sensitivity weights for one customer segment.

    Phase 0.0 only exercised ``beta_price``; Phase 0.1 populated ``beta_marketing``
    for the quality-seeking segment; Phase 0.2 populates ``beta_assortment`` per
    segment (the contemporaneous breadth lever); Phase 0.3 adds ``beta_promotion``
    (the contemporaneous promo-lift weight, the analog of ``beta_assortment``). The
    loyalty weight remains declared (defaulted to 0.0) for a later phase. Frozen:
    segment definitions are fixed for a world's lifetime; no structural change in
    0.3 (``beta_promotion`` is data, not a new segment).
    """

    beta_price: float = 2.0
    beta_assortment: float = 0.0
    beta_marketing: float = 0.0
    # Contemporaneous promo-lift weight (Phase 0.3): how strongly a segment is drawn
    # by this tick's decoded promotion (the analog of beta_assortment). Default 0.0
    # so single-segment fallbacks are unaffected.
    beta_promotion: float = 0.0
    beta_loyalty: float = 0.0
    beta_reference: float = 0.0
    beta_under_reference_aversion: float = 0.0
    beta_service: float = 0.0
    beta_program: float = 0.0


def default_segments() -> dict[str, SegmentParams]:
    """ """
    return {
        PRICE_SENSITIVE: SegmentParams(
            beta_price=2.0,
            beta_assortment=0.4,
            beta_marketing=0.0,
            beta_promotion=1.5,
            beta_loyalty=0.0,
            beta_program=0.0,
            beta_under_reference_aversion=0.0,
        ),
        QUALITY_SEEKING: SegmentParams(
            beta_price=0.5,
            beta_assortment=2.0,
            beta_marketing=1.0,
            beta_promotion=0.3,
            beta_loyalty=0.0,
            beta_program=0.0,
            beta_under_reference_aversion=0.0,
        ),
        CONVENIENCE: SegmentParams(
            beta_price=1.0,
            beta_assortment=1.0,
            beta_marketing=0.5,
            beta_promotion=0.8,
            beta_loyalty=0.0,
            beta_program=0.0,
            beta_under_reference_aversion=0.0,
        ),
        BRAND_LOYAL: SegmentParams(
            beta_price=0.6,
            beta_assortment=0.5,
            beta_marketing=0.6,
            beta_promotion=0.2,
            # Slice B: the moat is ON. The brand-loyal segment is the ONLY segment
            # with a non-zero loyalty weight (a strong 2.5 moat); paired with its LOW
            # beta_price (0.6, the lowest) this makes it a price-TOLERANCE moat — it
            # sticks with the retailer it is loyal to even under a price undercut.
            beta_loyalty=2.5,
            beta_program=0.0,
            beta_under_reference_aversion=0.0,
        ),
    }


@dataclass(frozen=True)
class RegionState:
    """ """

    population: float = 1000.0
    base_regional_demand: float = 1000.0
    segment_mix: dict[str, float] = field(default_factory=lambda: {PRICE_SENSITIVE: 1.0})
    realized_regional_demand: float = 0.0
    # Seated consumer-stockpile / satiation stock (Phase 0.3). The single source of
    # truth for forward-buying between ticks: built from PRIOR-tick promos
    # (share-weighted across retailers), READ by demand to SUPPRESS the total pie
    # this tick, and RESEATED by ``world.step`` from this tick's promo for t+1.
    # Starts at 0.0 (no promo history) and is part of the serialized
    # (WorldState, rng_state). A DEBT (opposite sign to awareness's asset).
    stockpile: float = 0.0


@dataclass
class RetailerState:
    """ """

    name: str
    is_npc: bool
    cash: float
    price_index: float = 1.0
    # Per-region arrays, indexed by region (length n_regions). A retailer not
    # present in region r has stores_per_region[r] == 0 and does not compete there.
    stores_per_region: tuple[int, ...] = (1,)
    store_age_per_region: tuple[int, ...] = (0,)
    # Seated awareness stock per region (Phase 0.1 asset, now per-region). The SoT
    # for awareness between ticks: built from PRIOR-tick spend WHERE PRESENT, READ
    # by demand at tick t, RESEATED by ``world.step`` for t+1. Starts at 0.0 arrays
    # (no history); part of the serialized (WorldState, rng_state).
    awareness_per_region: tuple[float, ...] = (0.0,)
    # Seated loyalty stock per region (Phase 0.4 moat, the EMA of per-region share).
    # READ by demand (multiplied by beta_loyalty — 0.0 in Slice A), RESEATED by
    # ``world.step`` from this tick's per-region share. The awareness template,
    # per-region. Starts at 0.0 arrays; serializes with WorldState.
    loyalty_per_region: tuple[float, ...] = (0.0,)
    last_profit: float = 0.0
    last_revenue: float = 0.0
    # Aggregate (region-0) realized share last tick, kept for the legacy obs slot.
    last_market_share: float = 0.0
    # Per-region realized share last tick (reporting-only): the per-region analog of
    # ``last_market_share``, seated by ``world.step`` from ``accounting.market_share``
    # so the obs reports per-region share without recomputing.
    last_market_share_per_region: tuple[float, ...] = (0.0,)
    # Reporting-only record of the assortment breadth acted LAST tick (Phase 0.2).
    # Written by ``world.step`` from the decoded action; read by ``encode_observation``
    # — NEVER read by ``resolve_demand`` (assortment stays memoryless). Serializes
    # with WorldState (deterministic, no RNG).
    last_assortment: float = 0.0
    # Reporting-only record of the promotion intensity acted LAST tick (Phase 0.3).
    # Written by ``world.step``; read by ``encode_observation``. NOT the seated
    # promotion dynamic (that is ``RegionState.stockpile``).
    last_promotion: float = 0.0
    # Reporting-only record of the expansion choice decoded LAST tick (Phase 0.4):
    # 0 = no-op, r = opened region r. Surfaced in obs so the agent sees what it just
    # did (the ``last_*`` reporting pattern). Serializes with WorldState.
    last_expansion: float = 0.0
    service_score_per_region: tuple[float, ...] = (1.0,)
    automation_tier: int = 0
    # Reporting-only record of the TARGET automation tier decoded LAST tick (Phase
    # 1.2): the decoded TARGET in ``{0, 1, 2}`` BEFORE the mask resolves
    # monotonicity / differential-affordability. Written by ``world.step``; surfaced
    # in info / metrics.jsonl for the gate's uptake measurement (analog of
    # expansion's ``last_expansion``). The SoT for the seated tier is
    # ``automation_tier``; this field is reporting-only. Defaults to 0.
    last_automation_tier_action: int = 0
    #
    # ``wage_level`` — the retailer's PAID wage (the lever/spend the spike archetypes
    # set; seated to ``cfg.wage.wage_base`` at reset, 0.0 at default config). Read as
    # "paid_wage" by the wage→happiness coupling AND contributes to the present-others'
    # reference-wage read for rivals (the ``_reference_prices`` construction).
    wage_level: float = 0.0
    # ``employee_happiness`` in [0, 1] — the EMA driven DOWN when paid_wage <
    # required_wage; starts at PERFECT 1.0 (the high-happiness fixed point — the
    # service_score template). At default config it stays 1.0 forever (required == paid
    # == 0). Higher unhappiness ⇒ worse write-off severity.
    employee_happiness: float = 1.0
    # ``accumulated_overhead_basis`` — reporting-only record of the growth-overhead cost
    # charged LAST tick (MECHANIC 3). 0.0 at reset and whenever the overhead coeff is 0.
    accumulated_overhead_basis: float = 0.0
    # ``last_turnover`` — this-tick revenue, seated on the NEXT state so the write-off
    # (a %-of-TURNOVER penalty) reads the PRIOR-tick turnover (the lagged read — the
    # awareness template). 0.0 at reset.
    last_turnover: float = 0.0
    warehouse_capacity: float = float("inf")
    #
    # ``last_warehouse_utilization`` — this retailer's served-demand / seated-
    # capacity ratio from the tick just resolved (0.0 when capacity is
    # unmetered/inf, or at reset). Default 0.0.
    last_warehouse_utilization: float = 0.0
    # ``last_stockout_rate`` — this retailer's demand-weighted lost-sales fraction
    # (``Σ lost / Σ demanded`` across supplier + warehouse losses) from the tick
    # just resolved. Default 0.0 (no losses / at reset).
    last_stockout_rate: float = 0.0
    #
    # ``insolvent_ticks`` — CONSECUTIVE ticks this seat's cash has stayed below
    # ``-(credit_limit + restructure_debt_threshold)``; reset to 0 the instant cash
    # recovers above that line. Reaching ``restructure_after_ticks`` triggers a
    # restructuring (which also resets this back to 0).
    insolvent_ticks: int = 0
    # ``restructurings`` — reporting-only count of how many times this seat has been
    # restructured this episode (the ``last_automation_tier_action`` reporting-counter
    # template — never read by any economic array).
    restructurings: int = 0
    # NFD-1 — the NPC seat's PENDING flat action: the flat vector (44-wide as of
    # schema v12; 14-wide pre-v12) this seat's
    # scripted policy decided at draw #2 LAST tick, consumed by
    # ``demand.py::_decode_levers`` THIS tick as if it were this seat's
    # ``joint_action`` entry. That is the SINGLE mechanism by which a scripted seat
    # reaches the economy — decode → capex gates → clamp → demand → accounting → seats,
    # the same path a learning seat takes — so EVERY lever it plays carries the same
    # one-tick lag its ``price_index`` always did, with no ``is_npc`` branch downstream.
    #
    # ``None`` for every LEARNING seat (its action arrives in ``joint_action``) and for
    # an NPC that has not acted yet (reset / tick 0). That tick-0 ``None`` is the
    # byte-identity anchor: with no pending action ``_decode_levers`` falls back to the
    # per-lever registry defaults exactly as it always has, so an NPC "holds course" on
    # its first tick — which is what the price seat already did.
    #
    # Stored as PLAIN FLOATS (not the float32 array the policy returned) so it
    # serializes with ``(WorldState, rng_state)`` and replays bit-for-bit; the widening
    # is lossless and ``decode_action`` reads it through ``np.asarray(..., float64)``
    # either way, so decoding the tuple and decoding the array are bit-identical.
    pending_action: tuple[float, ...] | None = None

    @property
    def stores(self) -> int:
        """Legacy accessor: total open-store count across regions.

        Kept so 0.3 readers (envs/harness reporting, tests) that referenced the
        scalar ``stores`` still work; the per-region SoT is ``stores_per_region``.
        """
        return sum(self.stores_per_region)

    @property
    def awareness(self) -> float:
        """Legacy accessor: region-0 awareness (the unchanged legacy obs slot)."""
        return self.awareness_per_region[0]

    @property
    def loyalty_stock(self) -> float:
        """Legacy accessor: region-0 loyalty (the unchanged legacy obs slot)."""
        return self.loyalty_per_region[0]

    @property
    def service_score(self) -> float:
        """Legacy accessor: region-0 service_score (Phase 1.1, paralleling awareness).

        Returns the region-0 seated service_score for any reporting that wants a single
        scalar; the per-region SoT is ``service_score_per_region``.
        """
        return self.service_score_per_region[0]


@dataclass(frozen=True)
class PerceivedCompetitor:
    """ """

    competitor_price: float = 0.0
    competitor_market_share: float = 0.0
    competitor_awareness: float = 0.0
    competitor_assortment: float = 0.0
    competitor_promotion: float = 0.0
    competitor_loyalty_region_0: float = 0.0
    competitor_awareness_region_1: float = 0.0
    competitor_loyalty_region_1: float = 0.0
    competitor_market_share_region_1: float = 0.0
    competitor_mean_price_region_0: float = 0.0
    competitor_mean_price_region_1: float = 0.0
    extra_regions: tuple[tuple[float, float, float, float], ...] = ()


@dataclass
class WorldState:
    """The complete, serializable world (sans RNG, which is threaded separately).

    ``retailers`` is ordered: index 0 is the learning agent, subsequent indices
    are NPCs in agent-index order (the order NPC actions are drawn in — draw #2
    of the per-tick contract). Mutable: ``world.py`` advances ``tick`` and the
    retailer running state.
    """

    tick: int
    regions: list[RegionState]
    retailers: list[RetailerState]
    segments: dict[str, SegmentParams] = field(
        default_factory=lambda: {PRICE_SENSITIVE: SegmentParams()}
    )
    # Phase 0.5 weighted/normalized reward state: one RewardNormalizer per LEARNING
    # seat, keyed by seat index (NPC seats get NONE — their reward is never used).
    # ``world.reset`` seats one per learning seat; ``world.step`` advances it ONCE per
    # tick AFTER accounting (a deterministic post-accounting fold; no RNG draw) and
    # seats the advanced normalizer in the next state. Plain data (each normalizer is
    # four (count, mean, m2) triples) so it serializes with (WorldState, rng_state)
    # and replays bit-for-bit. Empty for mode="profit" (the 0.0–0.4 path never touches
    # it — the profit reward stays byte-identical).
    reward_stats: dict[int, RewardNormalizer] = field(default_factory=dict)
    perceived_competitors: dict[int, PerceivedCompetitor] = field(default_factory=dict)
    research_fidelity: dict[int, float] = field(default_factory=dict)
