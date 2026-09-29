"""Versioned, typed action/observation contracts for the sim core.

This module is the single source of truth for *what* an action and an
observation contain. ``encoding.py`` owns *how* they are laid out in a flat
vector; framework Spaces (``envs/spaces.py``, later waves) are built from the
specs defined here so that agents, the encoder, and the future network codec
all agree.

Phase 0.0 is degenerate: only the ``price_index`` lever is active and the
observation describes a single agent against a single discounter NPC in one
region. The registry is structured so that marketing/assortment/promotion/
expansion levers — and extra observation fields — can be appended in later
phases without renumbering or breaking existing agents (new dimensions are
appended; existing offsets never move). Schema v12 is the one deliberate
exception: it WIDENS the existing ``expansion`` block in place (a fixed
capacity, not an append), so offsets AFTER it shift — see the
``ACTION_SCHEMA_VERSION`` v11 -> v12 comment below for why.

Framework-free: standard library + light enums only (no numpy needed here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

# Bump when the action layout changes in a way agents must be aware of. New
# levers appended at the tail keep existing offsets stable but still bump this
# so agents can branch on the version they were trained against.
#
# Phase 1.0 bumps 5 -> 6: the ``research`` continuous lever is appended at the
# tail (flat offset 6) AND the COMPETITOR observation fields change MEANING (exact
# -> perceived/noisy) at the SAME offsets, with one self field (``research_fidelity``)
# appended (obs 30 -> 31). Existing action/obs offsets never move; the version bump
# signals the semantic change.
#
#
#
#
#
#
ACTION_SCHEMA_VERSION: int = 12

N_REGIONS: int = 2

N_REGIONS_MAX: int = 32

# Phase 1.2 ships exactly THREE automation tiers (the 3-wide masked-discrete
# automation lever): tier 0 = no automation (the registry default + byte-identity
# anchor), tier 1 = base, tier 2 = advanced. The automation lever's discrete width
# MUST equal this constant; ``validate_automation_width`` enforces it at construction
# (the analog of ``validate_expansion_width``). F3 = PERMANENT: the mask is
# MONOTONICITY + DIFFERENTIAL-AFFORDABILITY (mirrors expansion's "current-or-higher"
# pattern — see ``compute_automation_mask`` in core/world.py). The seated tier on
# ``RetailerState`` is MONOTONE NON-DECREASING; a target below the current seat is a
# no-op, an unaffordable upgrade is a no-op.
N_AUTOMATION_TIERS: int = 3


class LeverKind(Enum):
    """Whether a lever is a bounded scalar or a finite choice."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass(frozen=True)
class LeverSpec:
    """One controllable lever in the action space.

    ``low``/``high`` apply to :data:`LeverKind.CONTINUOUS` levers; ``n`` applies
    to :data:`LeverKind.DISCRETE` levers. ``enabled`` gates whether the lever
    contributes dimensions to the current flat layout — disabled future levers
    are declared (for documentation and forward-compat) but excluded from the
    encoding until their phase lands.
    """

    name: str
    kind: LeverKind
    default: float
    introduced_phase: str
    enabled: bool
    low: float | None = None
    high: float | None = None
    n: int | None = None

    @property
    def width(self) -> int:
        """Number of flat dimensions this lever occupies when enabled."""
        if self.kind is LeverKind.CONTINUOUS:
            return 1
        if self.n is None:
            raise ValueError(f"discrete lever {self.name!r} must define n")
        # Discrete levers are encoded as a masked logits block (argmax-decoded).
        return self.n


# Ordered registry. Order is the canonical flat-layout order; enabled levers are
# concatenated in this order by ``encoding.flatten_action_space_layout``. Append
# new levers at the tail in later phases — never reorder existing entries.
LEVER_REGISTRY: tuple[LeverSpec, ...] = (
    LeverSpec(
        name="price_index",
        kind=LeverKind.CONTINUOUS,
        low=0.5,
        high=2.0,
        default=1.0,
        introduced_phase="0.0",
        enabled=True,
    ),
    # --- Phase 0.1: the marketing lever (spend_fraction, lagged investment).
    # Appended at the tail so price stays at offset 0 and existing agents see a
    # stable layout; the version bump signals the contract change. The value is
    # the spend_fraction in [0, 1]. ---
    LeverSpec(
        name="marketing",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="0.1",
        enabled=True,
    ),
    # --- Phase 0.2: the assortment lever (breadth fraction, contemporaneous
    # setting — unlike the lagged marketing investment). Appended at the tail so
    # price@0/marketing@1 stay stable and assortment lands at offset 2; the version
    # bump signals the contract change. The value is the breadth fraction in
    # [0, 1] (0.0 = narrowest). ---
    LeverSpec(
        name="assortment",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="0.2",
        enabled=True,
    ),
    # --- Phase 0.3: the promotion lever (intensity fraction; a DUAL lever — a
    # contemporaneous demand lift + per-unit margin cost, PLUS a seated per-region
    # consumer stockpile that suppresses future demand). Appended at the tail so
    # price@0/marketing@1/assortment@2 stay stable and promotion lands at offset 3;
    # the version bump signals the contract change. The value is the promo intensity
    # in [0, 1] (0.0 = no promotion). ---
    LeverSpec(
        name="promotion",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="0.3",
        enabled=True,
    ),
    # --- Phase 0.4: the expansion lever (DISCRETE, OPEN-ONLY/IRREVERSIBLE,
    # CASH-GATED, MASKED — the project's first masked discrete action). A
    # ``LeverKind.DISCRETE`` lever of width n is a masked logits block argmax-decoded
    # by ``encoding.decode_action`` (the dormant 0.0 discrete path, now activated).
    # n == N_REGIONS: index 0 = no-op, index r = open region r. Appended at the tail
    # so price@0/marketing@1/assortment@2/promotion@3 stay stable; the logit block
    # lands at dims 4..4+n-1 (4–5 at n_regions=2). The version bumps to 5.
    #
    LeverSpec(
        name="expansion",
        kind=LeverKind.DISCRETE,
        n=N_REGIONS_MAX,
        default=0.0,
        introduced_phase="0.4",
        enabled=True,
    ),
    LeverSpec(
        name="research",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="1.0",
        enabled=True,
    ),
    #
    LeverSpec(
        name="service_level",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=1.0,
        introduced_phase="1.1",
        enabled=True,
    ),
    # --- Phase 1.2: the automation lever (DISCRETE, 3-wide, MONOTONE-IRREVERSIBLE,
    # CASH-GATED, MASKED — the project's second masked discrete action after
    # expansion). A ``LeverKind.DISCRETE`` lever of width ``N_AUTOMATION_TIERS = 3``
    # is a masked logits block argmax-decoded by ``encoding.decode_action`` (the same
    # dormant discrete-lever path expansion exercises). Decoded TARGET tier ∈ {0, 1,
    # 2}: tier 0 = no automation (the registry default + byte-identity anchor); tier
    # 1 = base; tier 2 = advanced. Each tier carries a CUMULATIVE capex level
    # (``AutomationConfig.capex_per_tier[t]`` — the "price tag of owning that tier")
    # and a multiplicative COGS-savings rate (``savings_per_tier[t]`` — the per-unit
    # variable-COGS reduction at the seated tier).
    #
    #
    #
    # Appended at the tail so price@0/marketing@1/assortment@2/promotion@3/expansion
    # @4–5/research@6/service_level@7 stay stable; the automation logit block lands at
    # dims 8..10 (8–10 at n_tiers=3). Action flat dim 8 -> **11**; obs flat dim 35 ->
    # **36** (one SELF ``automation_tier`` field appended at the obs tail). The
    # version bumps to 9. ---
    LeverSpec(
        name="automation",
        kind=LeverKind.DISCRETE,
        n=N_AUTOMATION_TIERS,
        default=0.0,
        introduced_phase="1.2",
        enabled=True,
    ),
    #
    LeverSpec(
        name="loyalty_spend",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="1.3",
        enabled=True,
    ),
    # --- Phase 5.0: two Phase-4 SEATED postures (``wage_level``, ``warehouse_capacity``)
    # are promoted to per-tick LEVERS. Appended at the tail so price@0/marketing@1/
    # assortment@2/promotion@3/expansion@4-5/research@6/service_level@7/automation@8-10/
    # loyalty_spend@11 stay stable; ``wage_spend`` lands at flat offset 12,
    # ``warehouse_invest`` at flat offset 13. The version bumps to 11.
    #
    # ``wage_spend`` ("pay staff vs risk a happiness write-off") is a CONTINUOUS
    # per-tick fraction in [0, 1]: once the seam consumes it (B-2), the paid wage is
    # ``wage_base + wage_spend · WageConfig.wage_spend_scale`` on top of the existing
    # ``wage_base`` — a divisible per-tick spend, the marketing/price template, not a
    # one-shot capex.
    #
    # ``warehouse_invest`` ("provision throughput capacity ahead of growth") is a
    # CONTINUOUS per-tick fraction in [0, 1]: once consumed it adds
    # ``warehouse_invest · WarehouseConfig.capacity_per_unit_invest`` units to the
    # retailer's warehouse throughput capacity (a non-depreciating stock — "spend
    # now, stock from next tick"), expensed at
    # ``warehouse_invest · WarehouseConfig.cost_per_unit_invest`` per tick.
    #
    LeverSpec(
        name="wage_spend",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="5.0",
        enabled=True,
    ),
    LeverSpec(
        name="warehouse_invest",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="5.0",
        enabled=True,
    ),
)


def enabled_levers() -> tuple[LeverSpec, ...]:
    """Levers active in the current phase, in canonical flat-layout order."""
    return tuple(lever for lever in LEVER_REGISTRY if lever.enabled)


def validate_expansion_capacity(n_regions: int) -> None:
    """ """
    expansion = next(lever for lever in LEVER_REGISTRY if lever.name == "expansion")
    assert expansion.n is not None  # discrete levers always define n (LeverSpec.width)
    if not (1 <= n_regions <= expansion.n):
        raise ValueError(
            f"n_regions={n_regions} must be in [1, {expansion.n}] (the expansion "
            f"lever's fixed capacity, N_REGIONS_MAX={expansion.n})"
        )


def validate_automation_width(n_tiers: int) -> None:
    """ """
    automation = next(lever for lever in LEVER_REGISTRY if lever.name == "automation")
    if automation.n != n_tiers:
        raise ValueError(
            f"automation lever n={automation.n} must equal n_tiers={n_tiers} "
            "(one discrete choice per tier: tier_0 + ... + tier_n-1)"
        )


class ObservationGroup(Enum):
    """Which conceptual block an observation field belongs to."""

    SELF = "self"
    MARKET = "market"
    COMPETITOR = "competitor"


@dataclass(frozen=True)
class ObservationField:
    """One named scalar dimension of the flat observation vector."""

    name: str
    group: ObservationGroup
    description: str


# Ordered observation layout. ``encoding.encode_observation`` fills these in this
# exact order. Phase 0.0 is full-information: the single competitor block is the
# discounter NPC. Later phases append fields; appending keeps earlier offsets stable.
#
OBSERVATION_SCHEMA: tuple[ObservationField, ...] = (
    # --- Self block ---
    ObservationField("cash", ObservationGroup.SELF, "agent cash on hand"),
    ObservationField("last_profit", ObservationGroup.SELF, "profit from previous tick"),
    ObservationField("last_revenue", ObservationGroup.SELF, "revenue from previous tick"),
    ObservationField("market_share", ObservationGroup.SELF, "agent regional market share"),
    ObservationField("price_level", ObservationGroup.SELF, "agent current price_index"),
    ObservationField("loyalty", ObservationGroup.SELF, "agent loyalty stock"),
    ObservationField("stores", ObservationGroup.SELF, "agent store count"),
    # --- Market aggregates ---
    ObservationField(
        "total_regional_demand", ObservationGroup.MARKET, "total units demanded in the region"
    ),
    ObservationField(
        "mean_competitor_price", ObservationGroup.MARKET, "mean price_index across competitors"
    ),
    # --- Competitor block (single discounter NPC in 0.0; PERCEIVED/noisy in 1.0) ---
    ObservationField(
        "competitor_price",
        ObservationGroup.COMPETITOR,
        "perceived discounter price_index (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "competitor_market_share",
        ObservationGroup.COMPETITOR,
        "perceived discounter market share (noised; sharpened by research fidelity)",
    ),
    # --- Phase 0.1: awareness stocks appended at the tail (existing offsets never
    # move). The reported value is the SEATED ``state.awareness`` the agent's
    # demand reads THIS tick (built from prior-tick spend), consistent with the
    # lagged model — not a post-spend value. ---
    ObservationField(
        "awareness",
        ObservationGroup.SELF,
        "agent seated awareness stock (read by demand this tick)",
    ),
    ObservationField(
        "competitor_awareness",
        ObservationGroup.COMPETITOR,
        "perceived discounter seated awareness stock (noised; sharpened by research fidelity)",
    ),
    # --- Phase 0.2: assortment breadth appended at the tail (existing offsets never
    # move; assortment lands at index 13, competitor_assortment at 14). These are
    # REPORTING-ONLY: the breadth the retailer acted LAST tick (read from the
    # reporting-only ``RetailerState.last_assortment``), surfaced so the agent can
    # see what it / the competitor just did. They are NOT a demand input — demand
    # reads the decoded action this tick, never these fields. ---
    ObservationField(
        "assortment",
        ObservationGroup.SELF,
        "agent assortment breadth acted last tick (contemporaneous setting)",
    ),
    ObservationField(
        "competitor_assortment",
        ObservationGroup.COMPETITOR,
        "perceived discounter assortment breadth last tick (noised; sharpened by research fidelity)",
    ),
    # --- Phase 0.3: promotion fields appended at the tail (existing offsets never
    # move; promotion lands at index 15, competitor_promotion at 16, stockpile at
    # 17). ``promotion``/``competitor_promotion`` are REPORTING-ONLY (the promo acted
    # LAST tick, from the reporting-only ``RetailerState.last_promotion``), surfaced
    # so the agent can see what it / the competitor just did — NOT a demand input.
    # ``stockpile`` is the REAL seated ``RegionState.stockpile`` the agent's demand
    # reads THIS tick to suppress the pie (the analog of reporting ``awareness``):
    # it IS genuine seated state, so the agent can observe the debt currently
    # suppressing demand and learn to let it decay before promoting again. ---
    ObservationField(
        "promotion",
        ObservationGroup.SELF,
        "agent promotion intensity acted last tick (contemporaneous lift)",
    ),
    ObservationField(
        "competitor_promotion",
        ObservationGroup.COMPETITOR,
        "perceived discounter promotion intensity last tick (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "stockpile",
        ObservationGroup.MARKET,
        "seated regional consumer stockpile (suppresses demand this tick)",
    ),
    ObservationField(
        "expansion",
        ObservationGroup.SELF,
        "agent expansion choice acted last tick (0 = no-op, r = opened region r)",
    ),
    # (2) The region-0 loyalty pair (0.3 omitted loyalty for self/competitor in the
    # region-0 slots; the existing index-5 ``loyalty`` is the aggregate/region-0
    # value, this makes region 0's per-region loyalty explicit alongside region 1's).
    ObservationField(
        "loyalty_region_0",
        ObservationGroup.SELF,
        "agent seated loyalty stock in region 0 (per-region moat, read by demand)",
    ),
    ObservationField(
        "competitor_loyalty_region_0",
        ObservationGroup.COMPETITOR,
        "perceived discounter seated loyalty stock in region 0 (noised; sharpened by research fidelity)",
    ),
    # (3) Region-1 SELF block: per-region awareness/loyalty/presence/share.
    ObservationField(
        "awareness_region_1",
        ObservationGroup.SELF,
        "agent seated awareness stock in region 1 (read by demand this tick)",
    ),
    ObservationField(
        "loyalty_region_1",
        ObservationGroup.SELF,
        "agent seated loyalty stock in region 1 (per-region moat)",
    ),
    ObservationField(
        "stores_region_1",
        ObservationGroup.SELF,
        "agent presence in region 1 (0 = absent, 1 = present)",
    ),
    ObservationField(
        "market_share_region_1",
        ObservationGroup.SELF,
        "agent realized market share in region 1 last tick",
    ),
    # (4) Region-1 MARKET block: per-region realized demand + stockpile.
    ObservationField(
        "total_regional_demand_region_1",
        ObservationGroup.MARKET,
        "total units demanded in region 1 last tick (realized pie)",
    ),
    ObservationField(
        "stockpile_region_1",
        ObservationGroup.MARKET,
        "seated region-1 consumer stockpile (suppresses region-1 demand this tick)",
    ),
    # (5) Region-1 COMPETITOR block: the incumbent's per-region awareness/loyalty/share.
    ObservationField(
        "competitor_awareness_region_1",
        ObservationGroup.COMPETITOR,
        "perceived discounter seated awareness stock in region 1 (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "competitor_loyalty_region_1",
        ObservationGroup.COMPETITOR,
        "perceived discounter seated loyalty stock in region 1 (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "competitor_market_share_region_1",
        ObservationGroup.COMPETITOR,
        "perceived discounter realized market share in region 1 last tick (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "research_fidelity",
        ObservationGroup.SELF,
        "agent current perception fidelity in [0, 1) (how much to trust the perceived competitor block)",
    ),
    ObservationField(
        "competitor_mean_price_region_0",
        ObservationGroup.COMPETITOR,
        "perceived (noised) per-region mean price of ALL other present retailers "
        "= the demand reference (sharpened by research)",
    ),
    ObservationField(
        "competitor_mean_price_region_1",
        ObservationGroup.COMPETITOR,
        "perceived (noised) per-region mean price of ALL other present retailers "
        "= the demand reference (sharpened by research)",
    ),
    ObservationField(
        "service_score_region_0",
        ObservationGroup.SELF,
        "agent's seated per-region service_score in [0,1] (per-region liability — "
        "decays back to 1.0; lower after stockouts)",
    ),
    ObservationField(
        "service_score_region_1",
        ObservationGroup.SELF,
        "agent's seated per-region service_score in [0,1] (per-region liability — "
        "decays back to 1.0; lower after stockouts)",
    ),
    # --- Phase 1.2 tail: the agent's OWN seated automation TIER (SELF, EXACT — own
    # state, not noised). A per-retailer MONOTONE NON-DECREASING integer in
    # ``[0, N_AUTOMATION_TIERS - 1]``, surfaced as a float for the Box dtype contract
    # (the structural analog of expansion's per-region ``stores_per_region`` but
    # rolled to a per-retailer scalar). The COGS-savings READ uses this PRE-update
    # seated tier (the one-tick lag — read/charge/seat discipline, expansion
    # template). Existing offsets 0–34 NEVER renumber; the field lands at obs offset
    # 35 (obs 35 -> 36). F3 = PERMANENT: there is no EMA — the seat IS the integer
    # tier itself; once upgraded the savings persist for the rest of the episode. ---
    ObservationField(
        "automation_tier",
        ObservationGroup.SELF,
        "agent's seated monotone non-decreasing automation tier (integer in "
        "[0, n_tiers-1] as float) — drives the COGS-savings multiplier (lagged)",
    ),
    ObservationField(
        "employee_happiness",
        ObservationGroup.SELF,
        "agent's seated employee happiness in [0,1] (write-off risk gauge; EMA "
        "toward the wage-vs-required target)",
    ),
    ObservationField(
        "wage_level",
        ObservationGroup.SELF,
        "agent's wage paid last tick (wage_base, or wage_base + wage_spend · "
        "wage_spend_scale once the seam consumes the lever)",
    ),
    ObservationField(
        "warehouse_utilization",
        ObservationGroup.SELF,
        "agent's warehouse utilization last tick in [0,1] (served / capacity; "
        "0.0 when capacity is unmetered/inf)",
    ),
    ObservationField(
        "last_stockout_rate",
        ObservationGroup.SELF,
        "agent's demand-weighted lost-sales fraction last tick in [0,1] "
        "(supplier + warehouse losses)",
    ),
    ObservationField(
        "others_present_region_0",
        ObservationGroup.MARKET,
        "count of OTHER present retailers in region 0 (exact; a pure read, no RNG)",
    ),
    ObservationField(
        "others_present_region_1",
        ObservationGroup.MARKET,
        "count of OTHER present retailers in region 1 (exact; a pure read, no RNG)",
    ),
)


def _region_observation_block(r: int) -> tuple[ObservationField, ...]:
    """ """
    return (
        # --- SELF (5) ---
        ObservationField(
            f"awareness_region_{r}",
            ObservationGroup.SELF,
            f"agent seated awareness stock in region {r} (read by demand this tick)",
        ),
        ObservationField(
            f"loyalty_region_{r}",
            ObservationGroup.SELF,
            f"agent seated loyalty stock in region {r} (per-region moat)",
        ),
        ObservationField(
            f"stores_region_{r}",
            ObservationGroup.SELF,
            f"agent presence in region {r} (0 = absent, 1 = present)",
        ),
        ObservationField(
            f"market_share_region_{r}",
            ObservationGroup.SELF,
            f"agent realized market share in region {r} last tick",
        ),
        ObservationField(
            f"service_score_region_{r}",
            ObservationGroup.SELF,
            f"agent's seated per-region service_score in [0,1] in region {r} "
            "(per-region liability — decays back to 1.0; lower after stockouts)",
        ),
        # --- MARKET (3) ---
        ObservationField(
            f"total_regional_demand_region_{r}",
            ObservationGroup.MARKET,
            f"total units demanded in region {r} last tick (realized pie)",
        ),
        ObservationField(
            f"stockpile_region_{r}",
            ObservationGroup.MARKET,
            f"seated region-{r} consumer stockpile (suppresses region-{r} demand this tick)",
        ),
        ObservationField(
            f"others_present_region_{r}",
            ObservationGroup.MARKET,
            f"count of OTHER present retailers in region {r} (exact; a pure read, no RNG)",
        ),
        # --- COMPETITOR (4, noised) ---
        ObservationField(
            f"competitor_awareness_region_{r}",
            ObservationGroup.COMPETITOR,
            f"perceived discounter seated awareness stock in region {r} "
            "(noised; sharpened by research fidelity)",
        ),
        ObservationField(
            f"competitor_loyalty_region_{r}",
            ObservationGroup.COMPETITOR,
            f"perceived discounter seated loyalty stock in region {r} "
            "(noised; sharpened by research fidelity)",
        ),
        ObservationField(
            f"competitor_market_share_region_{r}",
            ObservationGroup.COMPETITOR,
            f"perceived discounter realized market share in region {r} last tick "
            "(noised; sharpened by research fidelity)",
        ),
        ObservationField(
            f"competitor_mean_price_region_{r}",
            ObservationGroup.COMPETITOR,
            f"perceived (noised) per-region mean price of ALL other present retailers "
            f"in region {r} = the demand reference (sharpened by research)",
        ),
    )


@lru_cache(maxsize=None)
def observation_schema(n_regions: int) -> tuple[ObservationField, ...]:
    """ """
    if n_regions <= 2:
        return OBSERVATION_SCHEMA
    extra = tuple(field for r in range(2, n_regions) for field in _region_observation_block(r))
    return OBSERVATION_SCHEMA + extra


@dataclass(frozen=True)
class ActionSchema:
    """Immutable snapshot of the action contract handed to the encoder/decoder.

    Carries the schema version and the enabled levers so ``decode_action`` can
    validate/clip against the exact contract an agent was given.
    """

    version: int = ACTION_SCHEMA_VERSION
    levers: tuple[LeverSpec, ...] = field(default_factory=enabled_levers)

    @property
    def total_dim(self) -> int:
        """Total flat action dimensions across all enabled levers."""
        return sum(lever.width for lever in self.levers)


@lru_cache(maxsize=1)
def default_action_schema() -> ActionSchema:
    """The action schema for the current phase (cached singleton — immutable)."""
    return ActionSchema()
