"""Resolved domain parameters for the sim core (framework-free dataclasses).

RL-wrapper concerns (``gamma``, ``max_episode_steps``, truncation) deliberately
live **outside** core — see :data:`MAX_EPISODE_STEPS_DEFAULT` for the documented
hint constant that envs/ should consume.

No numpy, no behavior — pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retail_simulator.core.state import (
    BRAND_LOYAL,
    CONVENIENCE,
    PRICE_SENSITIVE,
    QUALITY_SEEKING,
    SEGMENT_KEYS,
    SegmentParams,
)

# Scoring window / training horizon hint (~2 simulated years). This is an RL
# wrapper concern (TimeLimit truncation) and MUST NOT be baked into core stepping
# logic; exposed here only so envs/ has a single documented default to read.
MAX_EPISODE_STEPS_DEFAULT: int = 104


def _validate_segment_mix(segment_mix: dict[str, float]) -> None:
    """ """
    if not segment_mix:
        raise ValueError("segment_mix must not be empty")
    unknown = set(segment_mix) - SEGMENT_KEYS
    if unknown:
        raise ValueError(
            f"segment_mix contains unknown segment keys: {sorted(unknown)}; "
            f"legal keys are {sorted(SEGMENT_KEYS)}"
        )
    if any(share < 0.0 for share in segment_mix.values()):
        raise ValueError(f"segment_mix shares must be non-negative: {segment_mix}")
    total = sum(segment_mix.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"segment_mix must sum to 1.0, got {total} for {segment_mix}")


@dataclass(frozen=True)
class RegionConfig:
    """ """

    base_regional_demand: float = 1000.0
    segment_mix: dict[str, float] = field(
        default_factory=lambda: {
            PRICE_SENSITIVE: 0.4,
            QUALITY_SEEKING: 0.25,
            CONVENIENCE: 0.2,
            BRAND_LOYAL: 0.15,
        }
    )

    def __post_init__(self) -> None:
        _validate_segment_mix(self.segment_mix)


@dataclass(frozen=True)
class DemandConfig:
    """ """

    base_regional_demand: float = 1000.0
    # Overall price elasticity of total regional demand (negative: higher price
    # shrinks the pie). Per-segment attractiveness weights live in SegmentParams.
    price_elasticity: float = -1.5
    noise_sigma: float = 0.05
    # Population shares by segment key; must sum to 1.0. Kept as the region-0
    # default / single-region fallback. The Phase 0.4 default adds the 4th
    # brand-loyal segment so the loyalty moat has a population (loyalty stays
    # inert in demand until Slice B turns beta_loyalty on).
    segment_mix: dict[str, float] = field(
        default_factory=lambda: {
            PRICE_SENSITIVE: 0.4,
            QUALITY_SEEKING: 0.25,
            CONVENIENCE: 0.2,
            BRAND_LOYAL: 0.15,
        }
    )
    # NEW (Phase 0.4): per-region overrides. Empty => single region from the
    # scalar fields above (0.3 behavior). The default is TWO regions: region 0 at
    # full size with the 4-segment mix, region 1 ASYMMETRIC (smaller +
    # brand-loyal-heavy). Exact numbers DELEGATED to the sweep tuning loop.
    regions: tuple[RegionConfig, ...] = field(
        default_factory=lambda: (
            RegionConfig(
                base_regional_demand=1000.0,
                segment_mix={
                    PRICE_SENSITIVE: 0.4,
                    QUALITY_SEEKING: 0.25,
                    CONVENIENCE: 0.2,
                    BRAND_LOYAL: 0.15,
                },
            ),
            RegionConfig(
                base_regional_demand=600.0,
                segment_mix={
                    PRICE_SENSITIVE: 0.25,
                    QUALITY_SEEKING: 0.15,
                    CONVENIENCE: 0.15,
                    BRAND_LOYAL: 0.45,
                },
            ),
        )
    )
    # MECHANIC 5 (regional share ceiling — see the class docstring above): the
    # maximum per-region aggregated choice share any one present retailer may
    # hold. Default 1.0 ⇒ OFF (the byte-identity anchor — ``resolve_demand``
    # does not even call the ceiling helper at this default).
    max_share_per_region: float = 1.0

    def __post_init__(self) -> None:
        _validate_segment_mix(self.segment_mix)
        for region in self.regions:
            _validate_segment_mix(region.segment_mix)
        if not (0.0 < self.max_share_per_region <= 1.0):
            raise ValueError(
                f"max_share_per_region must be in (0.0, 1.0], got {self.max_share_per_region}"
            )


@dataclass(frozen=True)
class MarketingConfig:
    """ """

    cost_per_unit_spend: float = 200.0
    decay: float = 0.3
    # MECHANIC 6 candidate (convex marketing cost — see the class docstring above):
    # convexity of marketing cost in spend_fraction. Default 1.0 ⇒ LINEAR / OFF (the
    # byte-identity anchor — ``core/demand.py`` executes the exact pre-existing
    # linear expression at this default, never ``spend_fraction ** 1.0``, which is
    # not guaranteed bit-identical). A value > 1.0 makes a high spend_fraction cost
    # disproportionately more at the margin (cost = spend_fraction**exponent *
    # cost_per_unit_spend), turning "max marketing every round" from a dominant
    # fixed choice into an interior, opponent/state-dependent optimum.
    marketing_cost_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.marketing_cost_exponent < 1.0:
            raise ValueError(
                f"marketing_cost_exponent must be >= 1, got {self.marketing_cost_exponent}"
            )


@dataclass(frozen=True)
class AssortmentConfig:
    """Parameters of the contemporaneous assortment-breadth lever (Phase 0.2).

    Assortment is a per-tick SETTING (like price), not an investment (like
    marketing): the decoded ``assortment`` in [0, 1] enters THIS tick's utility
    (+beta_assortment_s * assortment) and costs THIS tick. There is no seated
    state and no recurrence.
    """

    assortment_cost_per_unit: float = 120.0


@dataclass(frozen=True)
class PromotionConfig:
    """ """

    # CONTEMPORANEOUS margin compression. Money charged per unit MOVED per unit of
    # promotion: promo_cost_per_unit * promotion_i * units_i is the opex add-on this
    # tick. Funded per unit (a discount you pay on what you sell), so a promo that
    # moves more units costs more — this is what makes constant-max promo a real
    # margin trade-off, not a free demand boost.
    # CALIBRATED (2026-05-24) against the 4-lever sweep: at the original seeds
    # (0.15 / 1.0 / 0.5) a sustained promo never paid (promotion-inert) — the
    # per-unit cost + stockpile debt over-dominated the lift. Weakening both the
    # cost and the stockpile build, and slowing decay, yields a non-degenerate
    # interior optimum (~0.6) where a sustained promo pays its way.
    # RE-CALIBRATED (2026-05-25, Phase 0.4): the 4th (brand-loyal) segment + the
    # asymmetric loyal-heavy region 1 dropped the share-weighted promo lift
    # (blended beta_promotion ~1.0 -> ~0.77) and loyalty now captures the
    # brand-loyal segment, so the 0.3 friction left promotion inert again.
    # Lowering the cost + stockpile build restores an interior promo optimum.
    promo_cost_per_unit: float = 0.009
    stockpile_build: float = 0.17
    stockpile_decay: float = 0.7
    # MECHANIC-6b candidate (promotion contest — see the class docstring above):
    # weight netting each retailer's promotion against its present rivals' mean
    # promotion (self-excluded, per region) before the utility reads it. Default
    # 0.0 ⇒ ABSOLUTE / OFF (the byte-identity anchor — core/demand.py executes the
    # exact pre-existing ``bpr * promotion`` expression at this default, never a
    # contest-form subtraction, which is not guaranteed bit-identical even though
    # it nets to the same value at weight 0). A value > 0.0 makes the lift RELATIVE:
    # out-promoting present rivals lifts utility, matching their promotion cancels
    # it, netting BELOW their mean pushes it negative — turning promotion from a
    # dead absolute lever into a reactive, tit-for-tat burst weapon.
    promotion_contest_weight: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.promotion_contest_weight <= 1.0):
            raise ValueError(
                "promotion_contest_weight must be in [0.0, 1.0], got "
                f"{self.promotion_contest_weight}"
            )


@dataclass(frozen=True)
class PricingConfig:
    """Bounds and baseline for the pricing lever (mirrors schema price_index)."""

    min_price_index: float = 0.5
    max_price_index: float = 2.0
    baseline_price_index: float = 1.0


@dataclass(frozen=True)
class CostConfig:
    """Cost structure used by the accounting engine (Wave 2)."""

    # Cost of goods as a fraction of the base price (price_index == 1.0).
    cogs_fraction: float = 0.4
    # Fixed operating expense charged per store per tick. Placeholder default;
    # Wave 2 / calibration tunes this against the interior-optimum requirement.
    fixed_opex_per_store: float = 50.0


@dataclass(frozen=True)
class LoyaltyConfig:
    """Loyalty stock dynamics: L = alpha * L + (1 - alpha) * market_share.

    Accrued per-region every tick (Phase 0.4); enters the brand-loyal segment's
    utility via ``beta_loyalty`` once loyalty-in-demand lands (Slice B). In Slice A
    loyalty accrues but is inert in demand (``beta_loyalty = 0.0`` for all
    segments).
    """

    alpha: float = 0.85


@dataclass(frozen=True)
class LoyaltyProgramConfig:
    """ """

    cost_per_unit_spend: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_per_unit_spend < 0.0:
            raise ValueError(f"cost_per_unit_spend must be >= 0, got {self.cost_per_unit_spend}")


@dataclass(frozen=True)
class ExpansionConfig:
    """Parameters of the expansion lever (Phase 0.4).

    Expansion is DISCRETE, OPEN-ONLY/IRREVERSIBLE, CASH-GATED, and MASKED:
    opening a store in a region the retailer is not yet present in costs a one-time
    ``expansion_capex`` (charged once, on a valid open) and adds an ongoing
    ``fixed_opex_per_store`` per tick (CostConfig, charged on every open store every
    tick). The mask (cash >= capex AND not-already-present) is advisory; an invalid
    decoded open is a deterministic no-op in ``world.step`` (charges nothing).
    """

    expansion_capex: float = 5_000.0


@dataclass(frozen=True)
class ResearchConfig:
    """ """

    cost_per_fidelity: float = 15.0
    noise_sigma_base: float = 0.3
    # Saturating fidelity mapping denominator: fidelity = spend / (spend + k). k > 0 so
    # fidelity is interior (in [0, 1), never exactly 1 even at spend = 1).
    fidelity_k: float = 0.3

    def __post_init__(self) -> None:
        if self.cost_per_fidelity < 0.0:
            raise ValueError(f"cost_per_fidelity must be >= 0, got {self.cost_per_fidelity}")
        if self.noise_sigma_base < 0.0:
            raise ValueError(f"noise_sigma_base must be >= 0, got {self.noise_sigma_base}")
        if self.fidelity_k <= 0.0:
            raise ValueError(f"fidelity_k must be > 0, got {self.fidelity_k}")

    def fidelity(self, research_spend: float) -> float:
        """ """
        return research_spend / (research_spend + self.fidelity_k)


@dataclass(frozen=True)
class SCMConfig:
    """ """

    cogs_premium: float = 0.0
    # CALIBRATED (2026-05-26 — moved 0.7→0.5): at s=0 the agent fulfils 50% of demand
    # (not 70%), so the cheap-supplier end is genuinely punishing — the tradeoff vs the
    # reliable end carries enough magnitude for the constant-service curve to show a
    # clearly interior optimum (the FREE-baseline non-degeneracy bar). No byte-identity
    # impact: at the registry default service_level=1.0, fill_rate(1.0)=1.0 regardless
    # of the floor (the floor only takes effect at s<1, which CoreConfig.default never
    # produces from a free learner because cogs_premium=0 makes any service equivalent).
    fill_rate_floor: float = 0.5
    fill_rate_curvature: float = 1.0
    lost_sales_share_penalty: float = 0.0
    service_score_decay: float = 0.1

    def __post_init__(self) -> None:
        if self.cogs_premium < 0.0:
            raise ValueError(f"cogs_premium must be >= 0, got {self.cogs_premium}")
        if not (0.0 <= self.fill_rate_floor <= 1.0):
            raise ValueError(f"fill_rate_floor must be in [0, 1], got {self.fill_rate_floor}")
        if self.lost_sales_share_penalty < 0.0:
            raise ValueError(
                f"lost_sales_share_penalty must be >= 0, got {self.lost_sales_share_penalty}"
            )
        if not (0.0 < self.service_score_decay <= 1.0):
            raise ValueError(
                f"service_score_decay must be in (0, 1], got {self.service_score_decay}"
            )

    def fill_rate(self, service_level: float) -> float:
        """ """
        return self.fill_rate_floor + (1.0 - self.fill_rate_floor) * service_level


@dataclass(frozen=True)
class AutomationConfig:
    """ """

    capex_per_tier: tuple[float, float, float] = (0.0, 0.0, 0.0)
    savings_per_tier: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        from retail_simulator.core.schema import N_AUTOMATION_TIERS

        if len(self.capex_per_tier) != N_AUTOMATION_TIERS:
            raise ValueError(
                f"capex_per_tier must have N_AUTOMATION_TIERS={N_AUTOMATION_TIERS} "
                f"entries, got {len(self.capex_per_tier)}"
            )
        if len(self.savings_per_tier) != N_AUTOMATION_TIERS:
            raise ValueError(
                f"savings_per_tier must have N_AUTOMATION_TIERS="
                f"{N_AUTOMATION_TIERS} entries, got {len(self.savings_per_tier)}"
            )
        if any(c < 0.0 for c in self.capex_per_tier):
            raise ValueError(f"capex_per_tier must be non-negative, got {self.capex_per_tier}")
        if self.capex_per_tier[0] != 0.0:
            raise ValueError(
                f"capex_per_tier[0] must be 0.0 (the no-automation anchor), "
                f"got {self.capex_per_tier[0]}"
            )
        for i in range(len(self.capex_per_tier) - 1):
            if self.capex_per_tier[i + 1] < self.capex_per_tier[i]:
                raise ValueError(
                    f"capex_per_tier must be monotone non-decreasing, got {self.capex_per_tier}"
                )
        if any(not (0.0 <= t < 1.0) for t in self.savings_per_tier):
            raise ValueError(f"savings_per_tier must be in [0, 1), got {self.savings_per_tier}")
        if self.savings_per_tier[0] != 0.0:
            raise ValueError(
                f"savings_per_tier[0] must be 0.0 (the no-automation anchor; "
                f"a tier-0 retailer earns zero savings), got "
                f"{self.savings_per_tier[0]}"
            )
        for i in range(len(self.savings_per_tier) - 1):
            if self.savings_per_tier[i + 1] < self.savings_per_tier[i]:
                raise ValueError(
                    f"savings_per_tier must be monotone non-decreasing, got {self.savings_per_tier}"
                )


@dataclass(frozen=True)
class WageConfig:
    """ """

    # MECHANIC 1: affordability-clamp gate. Default False ⇒ no clamp (byte-identical).
    cash_budget_enabled: bool = False
    overdraft_rate: float = 0.0
    # MECHANIC 1 (working-capital credit line): widens the discretionary-spend clamp's
    # basis from ``max(cash − committed_capex, 0.0)`` to
    # ``max(cash + credit_limit − committed_capex, 0.0)``
    # (``core/affordability.py::compute_spend_clamp``) — a retailer can trade THROUGH a
    # negative-cash dip on working capital instead of being permanently zombied (zero
    # discretionary spend ⇒ zero demand effect ⇒ no way to earn back the shortfall,
    # since bankruptcy is deferred). Shipped INERT at 0.0: the clamp's default branch
    # executes the SAME ``max(cash − committed_capex, 0.0)`` expression, not merely a
    # numerically-equal one (the ``overdraft_rate != 0.0`` / ``overhead_reference_stores
    # == 1.0`` skip idiom above). Funds OPERATIONS ONLY, never expansion/automation
    # capex — ``gated_expansion_capex``/``gated_automation_capex`` deliberately keep
    # gating on raw ``cash`` (see their own docstrings in ``core/affordability.py``);
    # widening what a retailer may discretionarily SPEND must never widen what it may
    # OPEN, or credit would make growth easier, undoing the growth brake the clamp
    # exists to create. Must be >= 0 (a limit, not a target). MAY be armed alone — with
    # ``overdraft_rate`` at its default 0.0 the wider budget costs nothing every tick (a
    # free credit line); only the REVERSE pairing (an armed ``overdraft_rate`` with no
    # line to bound it) is invalid, see ``overdraft_rate`` above.
    credit_limit: float = 0.0
    restructure_after_ticks: int = 0
    # MECHANIC 4: how many of the seat's best CLOSEABLE regions (by seated
    # ``last_market_share_per_region``) survive a restructuring; every other present
    # closeable region's store is closed. Region 0 (the action-schema's stranded
    # no-op slot — see ``core/world.py::_restructuring_survivor_regions``) is
    # ALWAYS exempt: it never counts against this budget and is never closed. Must
    # be >= 1 — a restructuring is a recovery path, not an exit, so it always keeps
    # at least one closeable region. Default 1.
    restructure_keep_stores: int = 1
    # MECHANIC 4: widens the restructuring trigger's debt line beyond ``credit_limit``
    # alone. Default 0.0 ⇒ the line sits exactly at ``-credit_limit`` (the affordability
    # clamp's own zombie threshold). Must be >= 0 (a widening, never a tightening, of
    # the line beyond the clamp's own).
    restructure_debt_threshold: float = 0.0
    # MECHANIC 2: the rival-coupled reference-wage weight. Default 0.0 ⇒ the write-off
    # gate is OFF on this leg ⇒ NO draw #5 (see ``writeoff_severity_scale``).
    wage_coupling_coeff: float = 0.0
    # MECHANIC 2: the own-share term weight (the non-monotone/own-size input). Default 0.
    wage_own_share_coeff: float = 0.0
    # MECHANIC 2: the required-wage intercept. Also the reset wage_level anchor.
    wage_base: float = 0.0
    # MECHANIC 2: the write-off severity scale. Default 0.0 ⇒ the write-off gate is OFF
    # on this leg ⇒ NO draw #5 (paired with wage_coupling_coeff above — the draw fires
    # only when AT LEAST ONE of the two is non-zero).
    writeoff_severity_scale: float = 0.0
    # MECHANIC 2: the bounded write-off draw range (a fraction of turnover). Default
    # (0.0, 0.0) ⇒ the draw, when taken, is identically 0.0.
    writeoff_frac_min: float = 0.0
    writeoff_frac_max: float = 0.0
    # MECHANIC 2: the leader-weighting exponent on retailer size. Default 1.0 ⇒ plain
    # %-of-turnover (size ** 0 == 1, no extra factor); > 1.0 taxes the biggest more.
    leader_weight_exponent: float = 1.0
    # MECHANIC 2: the employee-happiness EMA rate in (0, 1]. Default 0.1 (slow). Inert
    # at default config because happiness only DROPS when required_wage > paid_wage, and
    # at default config (all-zero wage coeffs/base) required == paid == 0 ⇒ happiness
    # stays 1.0; the write-off draw is not taken regardless.
    happiness_decay: float = 0.1
    # MECHANIC 3: the store-count threshold above which super-linear overhead bites.
    overhead_threshold_stores: float = 0.0
    # MECHANIC 3: the overhead coefficient. Default 0.0 ⇒ the overhead line is 0
    # (byte-identical).
    overhead_superlinear_coeff: float = 0.0
    # MECHANIC 3: the overhead exponent (super-linear when > 1). Default 1.0.
    overhead_exponent: float = 1.0
    overhead_reference_stores: float = 1.0
    # PHASE 5.0: the per-tick ``wage_spend`` lever's money scale. Once the seam
    # (B-2) is live, ``paid_wage = wage_base + wage_spend · wage_spend_scale``.
    # Default 0.0 ⇒ the seam is inert (``paid_wage`` stays ``wage_base``
    # regardless of the decoded ``wage_spend`` value) — the byte-identity anchor.
    wage_spend_scale: float = 0.0
    # PHASE 5.0: divides the paid-vs-required wage gap before it drives the
    # happiness EMA target (a smoothing knob so a small shortfall doesn't
    # instantly saturate the target to 0). Must be > 0 (a divisor). Default 1.0
    # ⇒ the gap is used unscaled — today's formula, byte-identical.
    wage_shortfall_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.overdraft_rate < 0.0:
            raise ValueError(f"overdraft_rate must be >= 0, got {self.overdraft_rate}")
        if self.credit_limit < 0.0:
            raise ValueError(f"credit_limit must be >= 0, got {self.credit_limit}")
        if self.overdraft_rate != 0.0 and self.credit_limit == 0.0:
            raise ValueError(
                f"""overdraft_rate ({self.overdraft_rate}) requires a positive credit_limit (got 0.0) - with no credit line to bound the charge, overdraft_rate would price the whole deficit every tick and, with bankruptcy deferred, compound geometrically forever (section 6 decision 3)"""
            )
        if self.restructure_after_ticks < 0:
            raise ValueError(
                f"restructure_after_ticks must be >= 0, got {self.restructure_after_ticks}"
            )
        if self.restructure_keep_stores < 1:
            raise ValueError(
                f"restructure_keep_stores must be >= 1, got {self.restructure_keep_stores}"
            )
        if self.restructure_debt_threshold < 0.0:
            raise ValueError(
                f"restructure_debt_threshold must be >= 0, got {self.restructure_debt_threshold}"
            )
        if self.wage_coupling_coeff < 0.0:
            raise ValueError(f"wage_coupling_coeff must be >= 0, got {self.wage_coupling_coeff}")
        if self.wage_own_share_coeff < 0.0:
            raise ValueError(f"wage_own_share_coeff must be >= 0, got {self.wage_own_share_coeff}")
        if self.wage_base < 0.0:
            raise ValueError(f"wage_base must be >= 0, got {self.wage_base}")
        if self.writeoff_severity_scale < 0.0:
            raise ValueError(
                f"writeoff_severity_scale must be >= 0, got {self.writeoff_severity_scale}"
            )
        if self.writeoff_frac_min < 0.0:
            raise ValueError(f"writeoff_frac_min must be >= 0, got {self.writeoff_frac_min}")
        if self.writeoff_frac_max < self.writeoff_frac_min:
            raise ValueError(
                f"writeoff_frac_max ({self.writeoff_frac_max}) must be >= "
                f"writeoff_frac_min ({self.writeoff_frac_min})"
            )
        if not (0.0 < self.happiness_decay <= 1.0):
            raise ValueError(f"happiness_decay must be in (0, 1], got {self.happiness_decay}")
        if self.wage_spend_scale < 0.0:
            raise ValueError(f"wage_spend_scale must be >= 0, got {self.wage_spend_scale}")
        if self.wage_shortfall_scale <= 0.0:
            raise ValueError(f"wage_shortfall_scale must be > 0, got {self.wage_shortfall_scale}")
        if self.overhead_threshold_stores < 0.0:
            raise ValueError(
                f"overhead_threshold_stores must be >= 0, got {self.overhead_threshold_stores}"
            )
        if self.overhead_superlinear_coeff < 0.0:
            raise ValueError(
                f"overhead_superlinear_coeff must be >= 0, got {self.overhead_superlinear_coeff}"
            )
        if self.overhead_exponent < 1.0:
            raise ValueError(f"overhead_exponent must be >= 1, got {self.overhead_exponent}")
        if self.overhead_reference_stores <= 0.0:
            raise ValueError(
                f"overhead_reference_stores must be > 0, got {self.overhead_reference_stores}"
            )

    @property
    def writeoff_gate_active(self) -> bool:
        """ """
        return self.wage_coupling_coeff != 0.0 or self.writeoff_severity_scale != 0.0

    @property
    def restructuring_gate_active(self) -> bool:
        """ """
        return self.restructure_after_ticks > 0


@dataclass(frozen=True)
class SupplierConfig:
    """ """

    # MASTER GATE. Default False ⇒ no contention (byte-identical).
    supplier_contention_enabled: bool = False
    # Per-region supplier capacity (total fulfillable units across present retailers).
    # Default inf ⇒ the regional demand total NEVER exceeds it ⇒ contention never binds
    # (byte-identical) even if the gate were flipped on. The scenario sets a finite value.
    supplier_capacity: float = float("inf")
    # The priority exponent on ``service_level``: priority weight = service_level **
    # preference_sharpness. Default 1.0 (linear in service). 0.0 ⇒ equal shares
    # (no preference); > 1.0 sharpens the favor toward high-service retailers.
    preference_sharpness: float = 1.0

    def __post_init__(self) -> None:
        if not (self.supplier_capacity > 0.0):
            raise ValueError(f"supplier_capacity must be > 0, got {self.supplier_capacity}")
        if self.preference_sharpness < 0.0:
            raise ValueError(f"preference_sharpness must be >= 0, got {self.preference_sharpness}")


@dataclass(frozen=True)
class WarehouseConfig:
    """ """

    # SUB-MECHANIC A: master gate for the throughput cap + spill. Default False ⇒ the
    # ``units`` array flows untouched in ``resolve_demand`` (byte-identical).
    throughput_enabled: bool = False
    # SUB-MECHANIC A: the PROVIDED warehouse throughput capacity each retailer is seated
    # with at reset (the independent provisioning posture). Default ``inf`` ⇒ the
    # throughput cap never binds ⇒ byte-identical. A finite value makes the cap bind for
    # any retailer whose served demand exceeds it — so a high-footprint AGGRESSOR
    # out-grows a uniform capacity (footprint-differentiated) without a per-seat hook.
    default_warehouse_capacity: float = float("inf")
    # SUB-MECHANIC A (24-region arena support, see the class docstring above for the
    # full rationale): footprint-SCALED top-up to the seated capacity above.
    # ``effective_capacity_k = warehouse_capacity_k + provided_capacity_per_store ·
    # stores_total_k`` (``core/demand.py``'s ``effective_warehouse_capacity`` — the
    # ONE place this arithmetic happens). Default ``0.0`` ⇒ the helper's ``!= 0.0``
    # guard returns ``warehouse_capacity`` UNCHANGED (byte-identical — not a "+ 0.0"
    # copy). See ``capacity_per_store`` below for how the two knobs interact, and
    # ``ramp_ticks`` below for how a NEWLY-opened store's contribution to THIS
    # top-up phases in over time instead of counting in full the tick it opens.
    provided_capacity_per_store: float = 0.0
    # SUB-MECHANIC B: required capacity per store. required_capacity_k =
    # capacity_per_store · stores_total_k. Default 0.0 ⇒ required == 0 ⇒ no shortfall.
    # A DIFFERENT knob from ``provided_capacity_per_store`` above (required vs
    # provided) — CONFIGURED independently, but NOT computed independently: the
    # shortfall below is computed against that field's EFFECTIVE (boosted) capacity,
    # not the raw seated value, so a non-degenerate penalty requires
    # ``capacity_per_store > provided_capacity_per_store``.
    capacity_per_store: float = 0.0
    # SUB-MECHANIC B: the penalty severity scale. Default 0.0 ⇒ the penalty gate is OFF
    # ⇒ NO draw #6 (the byte-identity / draw-order anchor).
    penalty_severity_scale: float = 0.0
    # SUB-MECHANIC B: the bounded penalty draw range (a fraction of prior turnover).
    # Default (0.0, 0.0) ⇒ the draw, when taken, is identically 0.0.
    penalty_frac_min: float = 0.0
    penalty_frac_max: float = 0.0
    # SUB-MECHANIC B: the shortfall half-saturation constant. severity saturates as
    # shortfall/(shortfall + halfsat) ∈ [0, 1). Must be > 0 (a divisor). Default 1.0.
    penalty_halfsat: float = 1.0
    # PHASE 5.0: the per-tick ``warehouse_invest`` lever's capacity yield. Once the
    # seam (B-2) is live, ``capacity_{t+1} = capacity_t + warehouse_invest ·
    # capacity_per_unit_invest`` (a non-depreciating stock — spend now, stock from
    # next tick). Default 0.0 ⇒ the carry stays today's (unchanged capacity)
    # regardless of the decoded ``warehouse_invest`` value — the byte-identity
    # anchor.
    capacity_per_unit_invest: float = 0.0
    # PHASE 5.0: the per-tick ``warehouse_invest`` lever's money cost. Once the seam
    # (B-2) is live, ``spend = warehouse_invest · cost_per_unit_invest`` — expensed
    # opex, not capex. Default 0.0 ⇒ no new opex line regardless of the decoded
    # ``warehouse_invest`` value — the byte-identity anchor.
    cost_per_unit_invest: float = 0.0
    #
    ramp_ticks: int = 0

    def __post_init__(self) -> None:
        if self.provided_capacity_per_store < 0.0:
            raise ValueError(
                f"provided_capacity_per_store must be >= 0, got "
                f"{self.provided_capacity_per_store}"
            )
        if self.capacity_per_store < 0.0:
            raise ValueError(f"capacity_per_store must be >= 0, got {self.capacity_per_store}")
        if self.penalty_severity_scale < 0.0:
            raise ValueError(
                f"penalty_severity_scale must be >= 0, got {self.penalty_severity_scale}"
            )
        if self.penalty_frac_min < 0.0:
            raise ValueError(f"penalty_frac_min must be >= 0, got {self.penalty_frac_min}")
        if self.penalty_frac_max < self.penalty_frac_min:
            raise ValueError(
                f"penalty_frac_max ({self.penalty_frac_max}) must be >= "
                f"penalty_frac_min ({self.penalty_frac_min})"
            )
        if not (self.penalty_halfsat > 0.0):
            raise ValueError(f"penalty_halfsat must be > 0, got {self.penalty_halfsat}")
        if self.capacity_per_unit_invest < 0.0:
            raise ValueError(
                f"capacity_per_unit_invest must be >= 0, got {self.capacity_per_unit_invest}"
            )
        if self.cost_per_unit_invest < 0.0:
            raise ValueError(f"cost_per_unit_invest must be >= 0, got {self.cost_per_unit_invest}")
        if self.ramp_ticks < 0:
            raise ValueError(f"ramp_ticks must be >= 0, got {self.ramp_ticks}")

    @property
    def penalty_gate_active(self) -> bool:
        """ """
        return self.penalty_severity_scale != 0.0


@dataclass(frozen=True)
class RewardConfig:
    """ """

    mode: str = "profit"  # "profit" | "weighted"
    weight_profit: float = 0.5
    weight_revenue: float = 0.2
    weight_market_share: float = 0.2
    weight_loyalty: float = 0.1
    normalizer_warmup: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("profit", "weighted"):
            raise ValueError(f"reward mode must be 'profit' or 'weighted', got {self.mode!r}")
        if self.normalizer_warmup < 0:
            raise ValueError(f"normalizer_warmup must be >= 0, got {self.normalizer_warmup}")
        total = (
            self.weight_profit
            + self.weight_revenue
            + self.weight_market_share
            + self.weight_loyalty
        )
        # Only enforce sum-to-1 for the weighted mode: a profit-mode config never
        # reads the weights, so a caller need not curate them (the default still
        # sums to 1.0 either way).
        if self.mode == "weighted" and abs(total - 1.0) > 1e-9:
            raise ValueError(f"weighted reward weights must sum to 1.0, got {total}")


NPC_ARCHETYPES: tuple[str, ...] = ("discounter", "premium", "balanced", "wandering")


@dataclass(frozen=True)
class SeatSpec:
    """ """

    is_npc: bool
    archetype: str | None = None
    presence: tuple[int, ...] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.is_npc:
            if self.archetype is None:
                raise ValueError("an NPC seat must name an archetype")
            if self.archetype not in NPC_ARCHETYPES:
                raise ValueError(
                    f"unknown NPC archetype {self.archetype!r}; known: {NPC_ARCHETYPES}"
                )
        elif self.archetype is not None:
            raise ValueError("a learning seat must not name an archetype")
        if self.presence is not None and any(s < 0 for s in self.presence):
            raise ValueError(f"seat presence store counts must be non-negative: {self.presence}")


@dataclass(frozen=True)
class CoreConfig:
    """Top-level resolved configuration for the pure world model."""

    demand: DemandConfig = field(default_factory=DemandConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    loyalty: LoyaltyConfig = field(default_factory=LoyaltyConfig)
    marketing: MarketingConfig = field(default_factory=MarketingConfig)
    assortment: AssortmentConfig = field(default_factory=AssortmentConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    # Research lever / perception layer (Phase 1.0). The default ResearchConfig +
    # research lever default 0.0 keep the economics BYTE-IDENTICAL to 0.5 (the opex
    # term is cost·0 = 0; perception is drawn LAST as draw #4 so it never perturbs the
    # #1–#3 economic stream). Calibration of the noise/cost/k seeds is delegated to the
    # sweep (Slice B + the orchestrator).
    research: ResearchConfig = field(default_factory=ResearchConfig)
    scm: SCMConfig = field(default_factory=SCMConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    loyalty_program: LoyaltyProgramConfig = field(default_factory=LoyaltyProgramConfig)
    wage: WageConfig = field(default_factory=WageConfig)
    supplier: SupplierConfig = field(default_factory=SupplierConfig)
    warehouse: WarehouseConfig = field(default_factory=WarehouseConfig)
    # Reward shape (Phase 0.5). Default mode="profit" keeps the 0.0–0.4 reward
    # byte-identical (regression-safe); the scenario/competition config selects
    # mode="weighted".
    reward: RewardConfig = field(default_factory=RewardConfig)
    seats: tuple[SeatSpec, ...] | None = None
    segments: dict[str, SegmentParams] | None = None

    @classmethod
    def default(cls) -> CoreConfig:
        """The Phase 0.0 default configuration (reward mode="profit", no seat plan)."""
        return cls()
