"""Analytical action-space sweep (degeneracy detection) for the Phase 0.3 gate.

Cost: ``grid_points^2 + 2*grid_points`` cells (the 2-D grid + the assortment axis
+ the promotion axis) — far cheaper than ``grid_points^4``.

It runs against the pure :class:`~retail_simulator.core.world.World` seam (NOT an
RL env): a no-RL, no-learning sweep, fast, exercising exactly the same transition
the trained agent would. Averaging over several seeds tames the multiplicative
demand noise (``noise_sigma ~= 0.05``).

Degeneracy verdicts — the 2-D price x marketing surface AND the assortment axis AND
the promotion axis must each be non-trivially worth setting (the sweep PASS == not
degenerate on price AND marketing AND assortment AND promotion):
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import (
    AutomationConfig,
    CoreConfig,
    LoyaltyProgramConfig,
    SCMConfig,
    SeatSpec,
)
from retail_simulator.core.state import (
    BRAND_LOYAL,
    CONVENIENCE,
    PRICE_SENSITIVE,
    SegmentParams,
    WorldState,
    default_segments,
)
from retail_simulator.core.world import AGENT_INDEX, World

# Obs index of the PERCEIVED competitor price (Phase 1.0): the reactive research
# heuristic best-responds to THIS noised read of the varying opponent — a sharper
# read (higher fidelity) means a less-noisy competitor_price ⇒ a better best-response.
# Read from the schema below (never hardcoded) so a layout change cannot silently
# point the heuristic at the wrong slot.
_COMPETITOR_PRICE_OBS_NAME: str = "competitor_price"

# Default sweep resolution / horizon. A modest grid in each lever keeps the 2-D
# cell count (grid_points^2) affordable on the seam; ticks_per_point ticks per
# (cell, seed) average out the demand noise, with a few seeds so the per-point
# standard error is small relative to the profit curvature.
DEFAULT_GRID_POINTS: int = 11
DEFAULT_TICKS_PER_POINT: int = 1000
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

# Marketing (spend_fraction) sweep range — the action bounds for the lever.
SPEND_LOW: float = 0.0
SPEND_HIGH: float = 1.0

# Assortment (breadth fraction) sweep range — the action bounds for the lever.
ASSORT_LOW: float = 0.0
ASSORT_HIGH: float = 1.0

# Promotion (intensity fraction) sweep range — the action bounds for the lever.
PROMO_LOW: float = 0.0
PROMO_HIGH: float = 1.0

PINNED_ASSORTMENT: float = 0.5

PINNED_PROMOTION: float = 0.3

# A clear optimum must beat the comparison cell by more than this multiple of the
# per-point standard error to count as a real (non-noise) win. ~3 sigma.
MARGIN_SIGMA_MULTIPLE: float = 3.0

# Floating-point tolerance for classifying a grid point as a boundary value.
_BOUNDARY_ATOL: float = 1e-9

SCORING_WINDOW_TICKS: int = 104

DEFAULT_EXPANSION_TIMINGS: tuple[int, ...] = (0, 10, 26, 52)

# Sentinel ``T`` for the never-expand baseline row (a policy that never opens).
NEVER_EXPAND_T: int = -1

# A timing curve counts as NON-flat (timing matters) only if the best T beats the
# WORST expand-at-T by > this multiple of their combined std-error — i.e. some
# T is meaningfully worse than the best T. Reuses the shared sigma multiple.
TIMING_FLAT_SIGMA_MULTIPLE: float = MARGIN_SIGMA_MULTIPLE

# Research (fidelity) sweep range — the research lever's action bounds. The axis maps
# a research SPEND in [0, 1] (the lever) through the config's saturating fidelity
# mapping to a per-tick perception FIDELITY in [0, 1); the verdict reasons about the
# SPEND axis (the decision variable the agent controls), reporting the resulting
# fidelity for provenance.
RESEARCH_LOW: float = 0.0
RESEARCH_HIGH: float = 1.0

DEFAULT_RESEARCH_SPENDS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

RESEARCH_OPPONENT_ARCHETYPE: str = "wandering"

RESEARCH_SECONDARY_OPPONENT_ARCHETYPE: str = "balanced"

RESEARCH_GATE_BETA_REFERENCE: float = 1.5

SERVICE_LOW: float = 0.0
SERVICE_HIGH: float = 1.0

# Default SCM-axis grid: spends in [0, 1] at modest density (the endpoints are
# load-bearing for the verdict — the constant-baseline best must come from the same
# grid that contains 0.0 and 1.0 so the "pinned at extreme" classification is well-
# defined). 11 points matches DEFAULT_GRID_POINTS for symmetry with the other axes.
DEFAULT_SERVICE_LEVELS: tuple[float, ...] = tuple(round(0.1 * i, 2) for i in range(11))

SCM_GATE_COGS_PREMIUM: float = 0.10
SCM_GATE_LOST_SALES_SHARE_PENALTY: float = 1.0
SCM_GATE_CONVENIENCE_BETA_SERVICE: float = 5.0


AUTOMATION_GATE_CAPEX_PER_TIER: tuple[float, float, float] = (0.0, 1500.0, 5000.0)
AUTOMATION_GATE_SAVINGS_PER_TIER: tuple[float, float, float] = (0.0, 0.10, 0.16)

DEFAULT_AUTOMATION_TIMINGS: tuple[int, ...] = (0, 5, 10, 20)

# Sentinel ``T`` for the never-upgrade baseline row (a policy that never upgrades).
NEVER_UPGRADE_T: int = -1

# Sentinel ``T`` for the ramp_0_to_1_to_2 row (encoded so the row carries a single
# integer policy field). Reserved value below NEVER_UPGRADE_T so it never collides.
RAMP_POLICY_T: int = -2

LOYALTY_LOW: float = 0.0
LOYALTY_HIGH: float = 1.0

# Default loyalty-spend grid: 11 points spanning [0, 1] at modest density (matches
# DEFAULT_GRID_POINTS / DEFAULT_SERVICE_LEVELS for symmetry). The endpoints are
# load-bearing for the verdict — the "pinned at extreme" classification requires
# both 0.0 (the no-program baseline) and 1.0 (the max-spend cell).
DEFAULT_LOYALTY_SPENDS: tuple[float, ...] = tuple(round(0.1 * i, 2) for i in range(11))

LOYALTY_GATE_COST_PER_UNIT_SPEND: float = 25.0
LOYALTY_GATE_BRAND_LOYAL_BETA_PROGRAM: float = 6.0


@dataclass(frozen=True)
class SweepRow2D:
    """One grid cell: a fixed (price, spend_fraction) held for the whole horizon.

    * ``price_index`` — the agent's fixed price for this cell.
    * ``spend_fraction`` — the agent's fixed marketing spend_fraction for this cell.
    * ``mean_profit`` — mean per-tick agent profit across post-burn-in ticks and
      seeds.
    * ``std_error`` — standard error of that mean (sample std / sqrt(n_samples)),
      so the verdict can ask whether a win exceeds the noise floor.
    * ``mean_market_share`` — mean per-tick agent market share (diagnostic).
    * ``n_samples`` — number of per-tick profit samples behind the mean.
    """

    price_index: float
    spend_fraction: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class SweepRow1D:
    """One assortment-axis cell: a fixed assortment at the pinned ``(p*, m*)``.

    Carries the pinned price/spend for provenance so the row is self-describing
    (the assortment axis is meaningful only at the joint optimum it was run at).

    * ``assortment`` — the agent's fixed assortment breadth for this cell.
    * ``fixed_price`` / ``fixed_spend`` — the joint (price, marketing) optimum the
      axis was swept at (provenance).
    * ``mean_profit`` / ``std_error`` / ``mean_market_share`` / ``n_samples`` — as
      in :class:`SweepRow2D`.
    """

    assortment: float
    fixed_price: float
    fixed_spend: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class SweepRow1DPromotion:
    """One promotion-axis cell: a fixed promotion at the pinned ``(p*, m*, a*)``.

    Carries the pinned price/spend/assortment for provenance so the row is
    self-describing (the promotion axis is meaningful only at the optimum it was
    run at, AND only when measured at steady state — the burn-in discards the
    transient so the cell reflects the sustained value of the promo level).

    * ``promotion`` — the agent's fixed promotion intensity for this cell.
    * ``fixed_price`` / ``fixed_spend`` / ``fixed_assortment`` — the (price,
      marketing, assortment) optimum the axis was swept at (provenance).
    * ``mean_profit`` / ``std_error`` / ``mean_market_share`` / ``n_samples`` — as
      in :class:`SweepRow2D`.
    """

    promotion: float
    fixed_price: float
    fixed_spend: float
    fixed_assortment: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class ExpansionTimingRow:
    """ """

    open_tick: int
    mean_profit: float
    std_error: float
    n_seeds: int

    @property
    def is_never_expand(self) -> bool:
        """True for the never-expand baseline row (open_tick == NEVER_EXPAND_T)."""
        return self.open_tick == NEVER_EXPAND_T


@dataclass(frozen=True)
class ResearchFidelityRow:
    """ """

    research_spend: float
    fidelity: float
    mean_score: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class SweepRow1DService:
    """ """

    service_level: float
    fixed_price: float
    fixed_spend: float
    fixed_assortment: float
    fixed_promotion: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class SweepRow1DAutomation:
    """ """

    policy_name: str
    target_tier: int
    upgrade_tick: int
    ramp_tick: int
    fixed_price: float
    fixed_spend: float
    fixed_assortment: float
    fixed_promotion: float
    mean_profit: float
    std_error: float
    n_seeds: int

    @property
    def is_never_upgrade(self) -> bool:
        """True for the never-upgrade baseline row (``upgrade_tick == NEVER_UPGRADE_T``)."""
        return self.upgrade_tick == NEVER_UPGRADE_T


@dataclass(frozen=True)
class SweepRow1DLoyalty:
    """ """

    loyalty_spend: float
    fixed_price: float
    fixed_spend: float
    fixed_assortment: float
    fixed_promotion: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class LoyaltyVsPromotionOrthogonality:
    """ """

    base_loyalty: float
    base_promotion: float
    base_profit: float
    swap_profits: tuple[tuple[float, float, float, float], ...]
    max_swap_delta: float
    threshold: float
    passed: bool
    reason: str


@dataclass(frozen=True)
class SweepResult:
    """ """

    rows: tuple[SweepRow2D, ...]
    assortment_rows: tuple[SweepRow1D, ...]
    promotion_rows: tuple[SweepRow1DPromotion, ...]
    profit_max_cell: tuple[float, float]
    profit_max_value: float
    assortment_max_value: float
    assortment_max_cell: float
    promotion_max_value: float
    is_degenerate: bool
    reason: str
    assortment_reason: str
    promotion_reason: str
    pinned_assortment: float
    pinned_promotion: float
    ticks_per_point: int
    burn_in_ticks: int
    seeds: tuple[int, ...]
    expansion_timing_rows: tuple[ExpansionTimingRow, ...] = ()
    expansion_timing_degenerate: bool | None = None
    expansion_timing_reason: str = ""
    start_present_both: bool = False
    research_rows: tuple[ResearchFidelityRow, ...] = ()
    research_degenerate: bool | None = None
    research_reason: str = ""
    service_rows: tuple[SweepRow1DService, ...] = ()
    service_variable_mean: float | None = None
    service_variable_std_error: float | None = None
    service_degenerate: bool | None = None
    service_reason: str = ""
    automation_rows: tuple[SweepRow1DAutomation, ...] = ()
    automation_degenerate: bool | None = None
    automation_reason: str = ""
    loyalty_rows: tuple[SweepRow1DLoyalty, ...] = ()
    loyalty_degenerate: bool | None = None
    loyalty_reason: str = ""
    loyalty_orthogonality: LoyaltyVsPromotionOrthogonality | None = None

    @property
    def passed(self) -> bool:
        """ """
        return not self.is_degenerate


def _is_boundary(value: float, low: float, high: float) -> bool:
    """True if ``value`` coincides with the range low or high boundary."""
    return math.isclose(value, low, abs_tol=_BOUNDARY_ATOL) or math.isclose(
        value, high, abs_tol=_BOUNDARY_ATOL
    )


def _beats_profit(
    best_profit: float,
    best_se: float,
    other_profit: float,
    other_se: float,
    margin_sigma_multiple: float,
) -> bool:
    """True if ``best`` beats ``other`` by > margin_sigma_multiple * combined SE.

    The std-error used is the combined error of the two cells (added in
    quadrature), so a noisy comparison cell cannot fake (or hide) a real win.
    """
    combined_se = math.hypot(best_se, other_se)
    margin = best_profit - other_profit
    return margin > margin_sigma_multiple * combined_se


def _beats(best: SweepRow2D, other: SweepRow2D, margin_sigma_multiple: float) -> bool:
    """True if 2-D cell ``best`` beats ``other`` by > margin * combined SE."""
    return _beats_profit(
        best.mean_profit, best.std_error, other.mean_profit, other.std_error, margin_sigma_multiple
    )


def verdict_for_sweep(
    rows: tuple[SweepRow2D, ...],
    *,
    price_low: float,
    price_high: float,
    spend_low: float = SPEND_LOW,
    spend_high: float = SPEND_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, tuple[float, float], float]:
    """ """
    if not rows:
        raise ValueError("cannot judge an empty sweep")

    best = max(rows, key=lambda r: r.mean_profit)
    profit_max_cell = (best.price_index, best.spend_fraction)
    profit_max_value = best.mean_profit

    # --- Price leg: compare along the price axis AT the best spend. ---
    same_spend = [
        r
        for r in rows
        if math.isclose(r.spend_fraction, best.spend_fraction, abs_tol=_BOUNDARY_ATOL)
    ]
    price_boundaries = [r for r in same_spend if _is_boundary(r.price_index, price_low, price_high)]
    if not price_boundaries:
        raise ValueError(
            "sweep grid has no price-boundary cells at the best spend; "
            "include the pricing-range endpoints"
        )

    if _is_boundary(best.price_index, price_low, price_high):
        return (
            True,
            (
                f"price-boundary: profit-max price {best.price_index:.4f} is at a boundary "
                f"([{price_low:.2f}, {price_high:.2f}]) at spend {best.spend_fraction:.3f} "
                f"-> dominant boundary pricing strategy"
            ),
            profit_max_cell,
            profit_max_value,
        )
    for boundary in price_boundaries:
        if not _beats(best, boundary, margin_sigma_multiple):
            combined_se = math.hypot(best.std_error, boundary.std_error)
            return (
                True,
                (
                    f"price-boundary: interior price {best.price_index:.4f} "
                    f"(profit {best.mean_profit:.2f}) does not beat price boundary "
                    f"{boundary.price_index:.2f} (profit {boundary.mean_profit:.2f}) at "
                    f"spend {best.spend_fraction:.3f} by >{margin_sigma_multiple:g}x std-error: "
                    f"margin {best.mean_profit - boundary.mean_profit:.2f} "
                    f"<= threshold {margin_sigma_multiple * combined_se:.2f}"
                ),
                profit_max_cell,
                profit_max_value,
            )

    # --- Marketing leg: the best spend must be non-trivial AND pay off vs 0. ---
    if math.isclose(best.spend_fraction, spend_low, abs_tol=_BOUNDARY_ATOL):
        # best spend == spend_low (== 0): marketing is inert.
        return (
            True,
            (
                f"marketing-inert: profit-max spend {best.spend_fraction:.3f} is the floor "
                f"({spend_low:.2f}) -> marketing never worth using (lever is inert)"
            ),
            profit_max_cell,
            profit_max_value,
        )
    if math.isclose(best.spend_fraction, spend_high, abs_tol=_BOUNDARY_ATOL):
        # best spend pinned at max: trivial "spend everything".
        return (
            True,
            (
                f"marketing-saturated: profit-max spend {best.spend_fraction:.3f} is pinned at "
                f"the max ({spend_high:.2f}) -> trivial spend-everything strategy"
            ),
            profit_max_cell,
            profit_max_value,
        )

    # ROI: the best spend must beat spend=0 at the SAME price by margin.
    same_price_zero_spend = [
        r
        for r in rows
        if math.isclose(r.price_index, best.price_index, abs_tol=_BOUNDARY_ATOL)
        and math.isclose(r.spend_fraction, spend_low, abs_tol=_BOUNDARY_ATOL)
    ]
    if not same_price_zero_spend:
        raise ValueError(
            "sweep grid has no spend=0 cell at the best price; include the spend floor"
        )
    zero_spend = same_price_zero_spend[0]
    if not _beats(best, zero_spend, margin_sigma_multiple):
        combined_se = math.hypot(best.std_error, zero_spend.std_error)
        return (
            True,
            (
                f"marketing-no-payoff: best spend {best.spend_fraction:.3f} at price "
                f"{best.price_index:.4f} (profit {best.mean_profit:.2f}) does not beat spend=0 "
                f"(profit {zero_spend.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - zero_spend.mean_profit:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f}"
            ),
            profit_max_cell,
            profit_max_value,
        )

    return (
        False,
        (
            f"clear interior optimum at price {best.price_index:.4f}, spend "
            f"{best.spend_fraction:.3f} (profit {best.mean_profit:.2f}): beats both price "
            f"boundaries and spend=0 by >{margin_sigma_multiple:g}x std-error (both levers pay)"
        ),
        profit_max_cell,
        profit_max_value,
    )


def assortment_verdict_for_axis(
    rows: tuple[SweepRow1D, ...],
    *,
    assort_low: float = ASSORT_LOW,
    assort_high: float = ASSORT_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """ """
    if not rows:
        raise ValueError("cannot judge an empty assortment axis")

    best = max(rows, key=lambda r: r.mean_profit)
    best_assortment = best.assortment
    best_profit = best.mean_profit

    zero_cells = [r for r in rows if math.isclose(r.assortment, assort_low, abs_tol=_BOUNDARY_ATOL)]
    if not zero_cells:
        raise ValueError("assortment axis has no assortment=0 cell; include the assortment floor")
    zero_assort = zero_cells[0]

    if math.isclose(best.assortment, assort_low, abs_tol=_BOUNDARY_ATOL):
        # best assortment == floor (== 0): breadth never helps.
        return (
            True,
            (
                f"assortment-inert: profit-max assortment {best.assortment:.3f} is the floor "
                f"({assort_low:.2f}) -> breadth never worth setting (lever is inert)"
            ),
            best_assortment,
            best_profit,
        )
    if math.isclose(best.assortment, assort_high, abs_tol=_BOUNDARY_ATOL):
        # best assortment pinned at max: trivial "always broadest".
        return (
            True,
            (
                f"assortment-saturated: profit-max assortment {best.assortment:.3f} is pinned at "
                f"the max ({assort_high:.2f}) -> trivial always-broadest strategy"
            ),
            best_assortment,
            best_profit,
        )

    # ROI: the best (interior) assortment must beat assortment=0 by margin.
    if not _beats_profit(
        best.mean_profit,
        best.std_error,
        zero_assort.mean_profit,
        zero_assort.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, zero_assort.std_error)
        return (
            True,
            (
                f"assortment-no-payoff: best assortment {best.assortment:.3f} "
                f"(profit {best.mean_profit:.2f}) does not beat assortment=0 "
                f"(profit {zero_assort.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - zero_assort.mean_profit:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f}"
            ),
            best_assortment,
            best_profit,
        )

    return (
        False,
        (
            f"clear interior assortment optimum at {best.assortment:.3f} "
            f"(profit {best.mean_profit:.2f}) at price {best.fixed_price:.4f}, spend "
            f"{best.fixed_spend:.3f}: beats assortment=0 by >{margin_sigma_multiple:g}x "
            f"std-error (breadth pays its opex)"
        ),
        best_assortment,
        best_profit,
    )


def promotion_verdict_for_axis(
    rows: tuple[SweepRow1DPromotion, ...],
    *,
    promo_low: float = PROMO_LOW,
    promo_high: float = PROMO_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """ """
    if not rows:
        raise ValueError("cannot judge an empty promotion axis")

    best = max(rows, key=lambda r: r.mean_profit)
    best_promotion = best.promotion
    best_profit = best.mean_profit

    zero_cells = [r for r in rows if math.isclose(r.promotion, promo_low, abs_tol=_BOUNDARY_ATOL)]
    if not zero_cells:
        raise ValueError("promotion axis has no promotion=0 cell; include the promotion floor")
    zero_promo = zero_cells[0]

    if math.isclose(best.promotion, promo_low, abs_tol=_BOUNDARY_ATOL):
        # best promotion == floor (== 0): promo never helps once costed + debted.
        return (
            True,
            (
                f"promotion-inert: profit-max promotion {best.promotion:.3f} is the floor "
                f"({promo_low:.2f}) -> promotion never worth using once the per-unit cost + "
                f"stockpile debt are paid (lever is inert)"
            ),
            best_promotion,
            best_profit,
        )
    if math.isclose(best.promotion, promo_high, abs_tol=_BOUNDARY_ATOL):
        # best promotion pinned at max: trivial "always-max promo" -> the
        # anti-arbitrage is not biting at steady state (stockpile/cost too weak).
        return (
            True,
            (
                f"promotion-saturated: profit-max promotion {best.promotion:.3f} is pinned at "
                f"the max ({promo_high:.2f}) -> trivial always-max-promo strategy "
                f"(the stockpile/cost is too weak; the anti-arbitrage is not biting)"
            ),
            best_promotion,
            best_profit,
        )

    # ROI: the best (interior) promotion must beat promotion=0 by margin.
    if not _beats_profit(
        best.mean_profit,
        best.std_error,
        zero_promo.mean_profit,
        zero_promo.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, zero_promo.std_error)
        return (
            True,
            (
                f"promotion-no-payoff: best promotion {best.promotion:.3f} "
                f"(profit {best.mean_profit:.2f}) does not beat promotion=0 "
                f"(profit {zero_promo.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - zero_promo.mean_profit:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f}"
            ),
            best_promotion,
            best_profit,
        )

    return (
        False,
        (
            f"clear interior promotion optimum at {best.promotion:.3f} "
            f"(profit {best.mean_profit:.2f}) at price {best.fixed_price:.4f}, spend "
            f"{best.fixed_spend:.3f}, assortment {best.fixed_assortment:.3f}: beats "
            f"promotion=0 by >{margin_sigma_multiple:g}x std-error (a sustained, "
            f"margin-costed promo pays for itself)"
        ),
        best_promotion,
        best_profit,
    )


def expansion_verdict_for_timing(
    rows: tuple[ExpansionTimingRow, ...],
    *,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
    flat_sigma_multiple: float = TIMING_FLAT_SIGMA_MULTIPLE,
) -> tuple[bool, str, int, float]:
    """ """
    if not rows:
        raise ValueError("cannot judge an empty expansion-timing leg")

    never_cells = [r for r in rows if r.is_never_expand]
    if not never_cells:
        raise ValueError(
            "expansion-timing leg has no never_expand baseline row "
            f"(open_tick == {NEVER_EXPAND_T}); include the never-expand baseline"
        )
    never = never_cells[0]

    expand_rows = [r for r in rows if not r.is_never_expand]
    if not expand_rows:
        raise ValueError(
            "expansion-timing leg has no expand_at_T policy rows; "
            "include at least one expand_at_T policy"
        )

    best = max(expand_rows, key=lambda r: r.mean_profit)
    best_open_tick = best.open_tick
    best_profit = best.mean_profit

    # --- Leg (a): the best expand_at_T must beat never_expand by margin. ---
    if not _beats_profit(
        best.mean_profit,
        best.std_error,
        never.mean_profit,
        never.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, never.std_error)
        return (
            True,
            (
                f"expansion-never-pays: best expand_at_T (T={best.open_tick}, profit "
                f"{best.mean_profit:.2f}) does not beat never_expand (profit "
                f"{never.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - never.mean_profit:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f} "
                f"(capex/opex too high or region-1 too small/unloyal)"
            ),
            best_open_tick,
            best_profit,
        )

    # --- Leg (b): the timing curve must be NON-flat (some T meaningfully worse). ---
    worst = min(expand_rows, key=lambda r: r.mean_profit)
    timing_matters = _beats_profit(
        best.mean_profit,
        best.std_error,
        worst.mean_profit,
        worst.std_error,
        flat_sigma_multiple,
    )
    if not timing_matters:
        # The curve is flat. If T==0 is the best (immediate dominates flat) it is the
        # free-lunch verdict; otherwise it is still flat-degenerate (timing inert).
        combined_se = math.hypot(best.std_error, worst.std_error)
        immediate_best = best.open_tick == 0
        label = "expansion-free-lunch" if immediate_best else "expansion-timing-flat"
        detail = (
            "expand_immediately (T=0) dominates and the timing curve is flat -> "
            "expansion is a no-brainer always-open-at-0 (capex/opex too low or "
            "region-1 too rich)"
            if immediate_best
            else (
                "the best expand_at_T beats never_expand but the timing curve is flat "
                "-> WHEN to open does not matter (timing is not a real decision)"
            )
        )
        return (
            True,
            (
                f"{label}: timing curve is flat (best T={best.open_tick} profit "
                f"{best.mean_profit:.2f} does not beat worst T={worst.open_tick} profit "
                f"{worst.mean_profit:.2f} by >{flat_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - worst.mean_profit:.2f} "
                f"<= threshold {flat_sigma_multiple * combined_se:.2f}); {detail}"
            ),
            best_open_tick,
            best_profit,
        )

    return (
        False,
        (
            f"non-degenerate: interior timing optimum at T={best.open_tick} "
            f"(profit {best.mean_profit:.2f}) beats never_expand (profit "
            f"{never.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error AND the "
            f"timing curve is non-flat (worst T={worst.open_tick} profit "
            f"{worst.mean_profit:.2f} is meaningfully worse) -> expansion pays AND "
            f"WHEN to open is a real decision (timing + cash-gating matter)"
        ),
        best_open_tick,
        best_profit,
    )


def research_verdict_for_fidelity(
    rows: tuple[ResearchFidelityRow, ...],
    *,
    research_low: float = RESEARCH_LOW,
    research_high: float = RESEARCH_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """ """
    if not rows:
        raise ValueError("cannot judge an empty research axis")

    zero_cells = [
        r for r in rows if math.isclose(r.research_spend, research_low, abs_tol=_BOUNDARY_ATOL)
    ]
    if not zero_cells:
        raise ValueError(
            "research axis has no research=0 baseline cell (the noisiest-read floor); "
            "include the research-spend floor"
        )
    baseline = zero_cells[0]

    best = max(rows, key=lambda r: r.mean_score)
    best_research_spend = best.research_spend
    best_score = best.mean_score

    # --- Leg (a): a sharper read must beat the research=0 (noisiest-read) baseline. ---
    if not _beats_profit(
        best.mean_score,
        best.std_error,
        baseline.mean_score,
        baseline.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, baseline.std_error)
        return (
            True,
            (
                f"research-inert: best research spend {best.research_spend:.3f} "
                f"(score {best.mean_score:.2f}) does not beat research=0 "
                f"(score {baseline.mean_score:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_score - baseline.mean_score:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f} "
                f"(information worthless against this opponent — noise too weak or "
                f"opponent too constant)"
            ),
            best_research_spend,
            best_score,
        )

    # --- Leg (b): the optimal research spend must be INTERIOR (not pinned at max). ---
    # research=0 best is impossible here (leg (a) requires best to BEAT the research=0
    # baseline by margin, so best != the baseline cell), so the only degenerate corner
    # left is "pinned at the max" (free information ⇒ always buy max).
    if math.isclose(best.research_spend, research_high, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"research-free: optimal research spend {best.research_spend:.3f} is pinned at "
                f"the max ({research_high:.2f}) -> a sharper read pays but research never costs "
                f"enough to hold back (free information, always-buy-max is trivially best); "
                f"the cost (cost_per_fidelity) is too low to make the optimum interior"
            ),
            best_research_spend,
            best_score,
        )

    return (
        False,
        (
            f"non-degenerate: interior research optimum at spend {best.research_spend:.3f} "
            f"(fidelity {best.fidelity:.3f}, score {best.mean_score:.2f}) beats research=0 "
            f"(score {baseline.mean_score:.2f}) by >{margin_sigma_multiple:g}x std-error AND "
            f"is interior (not pinned at the max) -> a sharper read is worth more than the "
            f"noise AND the research cost makes the optimal fidelity a real spend-vs-value "
            f"tradeoff (value of information against a varying opponent)"
        ),
        best_research_spend,
        best_score,
    )


def scm_verdict_for_service(
    rows: tuple[SweepRow1DService, ...],
    variable_mean_profit: float,
    variable_std_error: float,
    *,
    service_low: float = SERVICE_LOW,
    service_high: float = SERVICE_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """ """
    if not rows:
        raise ValueError("cannot judge an empty SCM service-level axis")

    low_cells = [
        r for r in rows if math.isclose(r.service_level, service_low, abs_tol=_BOUNDARY_ATOL)
    ]
    high_cells = [
        r for r in rows if math.isclose(r.service_level, service_high, abs_tol=_BOUNDARY_ATOL)
    ]
    if not low_cells or not high_cells:
        raise ValueError(
            "SCM service-level axis must include both the service_low (0.0) and "
            "service_high (1.0) endpoint cells so the constant-baseline 'pinned at "
            "extreme' classification is well-defined"
        )

    best = max(rows, key=lambda r: r.mean_profit)
    best_constant_service = best.service_level
    best_constant_profit = best.mean_profit

    # --- Leg (a): best constant must be INTERIOR (not pinned at either extreme). ---
    # If a single fixed service_level at the boundary dominates, a free PPO learner
    # converges to that fixed value and does not USE the lever (the 1.0 collapse
    # antidote). The non-degenerate verdict requires a real tradeoff at the constant
    # baseline before measuring the variable comparator.
    if math.isclose(best.service_level, service_low, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"scm-pinned-at-extreme: best constant-service {best.service_level:.3f} is "
                f"pinned at the floor ({service_low:.2f}) -> always-cheapest-supplier is a "
                f"free cost cut (a free PPO learner converges to s=0 and does not use the "
                f"lever); cogs_premium too high or lost_sales_share_penalty too low"
            ),
            best_constant_service,
            best_constant_profit,
        )
    if math.isclose(best.service_level, service_high, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"scm-pinned-at-extreme: best constant-service {best.service_level:.3f} is "
                f"pinned at the max ({service_high:.2f}) -> always-perfect-service is a "
                f"trivial dominant strategy (a free PPO learner converges to s=1 and does "
                f"not use the lever); cogs_premium too low or lost_sales_share_penalty too high"
            ),
            best_constant_service,
            best_constant_profit,
        )

    # --- Leg (b): the variable comparator must beat the best constant by margin. ---
    # The Phase-1.0 lesson, enforced. The free alternative for a PPO learner is the
    # best constant-service (what a learner that ignores the lever's per-tick
    # modulation collapses to). For the lever to be non-degenerate, conditioning on
    # observable own state must EXTRACT value beyond that — operationalized via the
    # per-region-optimal heuristic (the upper bound on a per-region-conditional
    # policy). If the variable comparator does NOT beat the best constant by margin,
    # the per-region asymmetry isn't worth conditioning on => degenerate.
    if not _beats_profit(
        variable_mean_profit,
        variable_std_error,
        best_constant_profit,
        best.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(variable_std_error, best.std_error)
        return (
            True,
            (
                f"scm-inert: per-region-optimal variable-service score "
                f"{variable_mean_profit:.2f} does not beat best constant-service "
                f"{best_constant_service:.3f} (profit {best_constant_profit:.2f}) by "
                f">{margin_sigma_multiple:g}x std-error: margin "
                f"{variable_mean_profit - best_constant_profit:.2f} <= threshold "
                f"{margin_sigma_multiple * combined_se:.2f} (the per-region asymmetry is not "
                f"worth conditioning on -> a free PPO learner that picks the best fixed "
                f"value matches the per-region-conditional alternative)"
            ),
            best_constant_service,
            best_constant_profit,
        )

    return (
        False,
        (
            f"non-degenerate: best constant-service {best_constant_service:.3f} (profit "
            f"{best_constant_profit:.2f}) is INTERIOR (not pinned at either extreme) AND "
            f"per-region-optimal variable-service score {variable_mean_profit:.2f} beats "
            f"the best constant by >{margin_sigma_multiple:g}x std-error -> a free PPO "
            f"learner that conditions on observable own state can extract value beyond the "
            f"best fixed setting (the SCM lever's per-region tradeoff is genuinely worth "
            f"conditioning on)"
        ),
        best_constant_service,
        best_constant_profit,
    )


def automation_verdict_for_timing(
    rows: tuple[SweepRow1DAutomation, ...],
    *,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
    flat_sigma_multiple: float = TIMING_FLAT_SIGMA_MULTIPLE,
) -> tuple[bool, str, str, float]:
    """ """
    if not rows:
        raise ValueError("cannot judge an empty automation upgrade-timing leg")

    never_cells = [r for r in rows if r.is_never_upgrade]
    if not never_cells:
        raise ValueError(
            "automation upgrade-timing leg has no never_upgrade baseline row "
            f"(upgrade_tick == {NEVER_UPGRADE_T}); include the never-upgrade baseline"
        )
    never = never_cells[0]

    upgrade_rows = [r for r in rows if not r.is_never_upgrade]
    if not upgrade_rows:
        raise ValueError(
            "automation upgrade-timing leg has no upgrade-policy rows; "
            "include at least one upgrade_to_K_at_t=T policy"
        )

    best = max(upgrade_rows, key=lambda r: r.mean_profit)
    best_policy_name = best.policy_name
    best_profit = best.mean_profit

    # --- Leg (a): the best upgrade policy must beat never_upgrade by margin. ---
    if not _beats_profit(
        best.mean_profit,
        best.std_error,
        never.mean_profit,
        never.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, never.std_error)
        return (
            True,
            (
                f"automation-never-pays: best upgrade policy {best.policy_name} "
                f"(profit {best.mean_profit:.2f}) does not beat never_upgrade "
                f"(profit {never.mean_profit:.2f}) by "
                f">{margin_sigma_multiple:g}x std-error: margin "
                f"{best.mean_profit - never.mean_profit:.2f} <= threshold "
                f"{margin_sigma_multiple * combined_se:.2f} (capex_per_tier too "
                f"high OR savings_per_tier too low — no upgrade timing recoups "
                f"the one-shot capex over the remaining window)"
            ),
            best_policy_name,
            best_profit,
        )

    # --- Leg (b): the timing curve must be NON-flat (some interior policy is
    # meaningfully worse than the best). Plus: ``upgrade_to_2_at_t=0`` must NOT
    # be a free dominator. ---
    worst = min(upgrade_rows, key=lambda r: r.mean_profit)
    timing_matters = _beats_profit(
        best.mean_profit,
        best.std_error,
        worst.mean_profit,
        worst.std_error,
        flat_sigma_multiple,
    )
    # Identify the upgrade_to_2_at_t=0 row (if present) so the verdict can detect
    # the free-lunch failure: t=0 dominates AND the curve is flat.
    immediate_t2 = next(
        (r for r in upgrade_rows if r.target_tier == 2 and r.upgrade_tick == 0),
        None,
    )
    immediate_is_best = immediate_t2 is not None and immediate_t2.policy_name == best.policy_name

    if not timing_matters:
        combined_se = math.hypot(best.std_error, worst.std_error)
        label = "automation-free-lunch" if immediate_is_best else "automation-timing-flat"
        detail = (
            "upgrade_to_2_at_t=0 dominates and the timing curve is flat -> "
            "automation is a no-brainer always-upgrade-at-0 (capex_per_tier "
            "too low OR savings_per_tier too high OR STARTING_CASH too high — "
            "the cash-gating is not biting)"
            if immediate_is_best
            else (
                "the best upgrade policy beats never_upgrade but the timing "
                "curve is flat -> WHEN to upgrade does not matter (timing is "
                "not a real decision)"
            )
        )
        return (
            True,
            (
                f"{label}: timing curve is flat (best {best.policy_name} profit "
                f"{best.mean_profit:.2f} does not beat worst {worst.policy_name} "
                f"profit {worst.mean_profit:.2f} by >{flat_sigma_multiple:g}x "
                f"std-error: margin {best.mean_profit - worst.mean_profit:.2f} "
                f"<= threshold {flat_sigma_multiple * combined_se:.2f}); {detail}"
            ),
            best_policy_name,
            best_profit,
        )

    return (
        False,
        (
            f"non-degenerate: interior upgrade-timing optimum at "
            f"{best.policy_name} (profit {best.mean_profit:.2f}) beats "
            f"never_upgrade (profit {never.mean_profit:.2f}) by "
            f">{margin_sigma_multiple:g}x std-error AND the timing curve is "
            f"non-flat (worst {worst.policy_name} profit {worst.mean_profit:.2f} "
            f"is meaningfully worse) -> automation pays AND WHEN to upgrade is "
            f"a real decision (timing + cash-gating matter)"
        ),
        best_policy_name,
        best_profit,
    )


def loyalty_verdict_for_spend(
    rows: tuple[SweepRow1DLoyalty, ...],
    *,
    loyalty_low: float = LOYALTY_LOW,
    loyalty_high: float = LOYALTY_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """ """
    if not rows:
        raise ValueError("cannot judge an empty loyalty-spend axis")

    low_cells = [
        r for r in rows if math.isclose(r.loyalty_spend, loyalty_low, abs_tol=_BOUNDARY_ATOL)
    ]
    high_cells = [
        r for r in rows if math.isclose(r.loyalty_spend, loyalty_high, abs_tol=_BOUNDARY_ATOL)
    ]
    if not low_cells or not high_cells:
        raise ValueError(
            "loyalty-spend axis must include both the loyalty_low (0.0) and "
            "loyalty_high (1.0) endpoint cells so the constant-baseline 'pinned at "
            "extreme' classification is well-defined"
        )

    best = max(rows, key=lambda r: r.mean_profit)
    best_loyalty_spend = best.loyalty_spend
    best_profit = best.mean_profit

    low_cell = low_cells[0]  # the no-program baseline (the FREE alternative anchor)
    high_cell = high_cells[0]  # the max-spend cell — the "always-spend-max" candidate

    # --- Leg (a): pinned-at-zero -- "no program" dominates ----------------------
    # If the best constant-loyalty is at the floor (0.0), a free PPO learner converges
    # to spend=0 and does not USE the lever. Calibration miss — beta_program too low
    # OR cost_per_unit_spend too high. The lever is degenerate before measuring any
    # interior margin.
    if math.isclose(best.loyalty_spend, loyalty_low, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"loyalty-pinned-at-zero: best constant loyalty_spend "
                f"{best_loyalty_spend:.3f} is pinned at the floor "
                f"({loyalty_low:.2f}) -> the no-program baseline dominates (a free "
                f"PPO learner converges to spend=0 and does not USE the lever); "
                f"beta_program too low OR cost_per_unit_spend too high"
            ),
            best_loyalty_spend,
            best_profit,
        )

    # --- Leg (b): loyalty-free -- "always spend max" dominates ------------------
    # If the best constant is at the max (1.0), the lever has no real tradeoff at
    # the constant baseline — always-spend-max is a trivial dominant strategy. A
    # free PPO learner converges to spend=1 and the gate trivially passes without
    # the lever rewarding any conditioning.
    if math.isclose(best.loyalty_spend, loyalty_high, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"loyalty-free: best constant loyalty_spend "
                f"{best_loyalty_spend:.3f} is pinned at the max "
                f"({loyalty_high:.2f}) -> always-spend-max is the trivial dominant "
                f"strategy (a free PPO learner converges to spend=1 and the gate "
                f"trivially passes); cost_per_unit_spend too low OR beta_program "
                f"too high"
            ),
            best_loyalty_spend,
            best_profit,
        )

    # --- Leg (c): loyalty-inert -- the lever does not move profit by margin ----
    # The interior best must beat BOTH endpoints by > margin x combined SE. If
    # neither endpoint is beaten, the spread is within the noise floor and the
    # lever is degenerate — calibration miss (beta_program too low OR
    # cost_per_unit_spend too high) ⇒ a free PPO learner finds no gradient.
    beats_low = _beats_profit(
        best.mean_profit,
        best.std_error,
        low_cell.mean_profit,
        low_cell.std_error,
        margin_sigma_multiple,
    )
    beats_high = _beats_profit(
        best.mean_profit,
        best.std_error,
        high_cell.mean_profit,
        high_cell.std_error,
        margin_sigma_multiple,
    )
    if not (beats_low and beats_high):
        # Use the LOSER of the two endpoint-vs-best comparisons in the reason so
        # the operator sees which side the optimum failed to clear.
        low_combined = math.hypot(best.std_error, low_cell.std_error)
        high_combined = math.hypot(best.std_error, high_cell.std_error)
        return (
            True,
            (
                f"loyalty-inert: best constant loyalty_spend "
                f"{best_loyalty_spend:.3f} (profit {best_profit:.2f}) does not beat "
                f"BOTH endpoints by >{margin_sigma_multiple:g}x std-error: "
                f"vs floor margin {best.mean_profit - low_cell.mean_profit:.2f} "
                f"(threshold {margin_sigma_multiple * low_combined:.2f}) AND "
                f"vs max margin {best.mean_profit - high_cell.mean_profit:.2f} "
                f"(threshold {margin_sigma_multiple * high_combined:.2f}); "
                f"calibration miss - beta_program too low OR cost_per_unit_spend "
                f"too high (a free PPO learner finds no gradient on the lever)"
            ),
            best_loyalty_spend,
            best_profit,
        )

    return (
        False,
        (
            f"non-degenerate: best constant loyalty_spend {best_loyalty_spend:.3f} "
            f"(profit {best_profit:.2f}) is INTERIOR AND beats BOTH endpoints by "
            f">{margin_sigma_multiple:g}x std-error (vs no-program "
            f"{low_cell.mean_profit:.2f}, vs max-spend {high_cell.mean_profit:.2f}) "
            f"-> a real cost-vs-boost tradeoff; a free PPO learner that conditions "
            f"on observable own state (per-region loyalty stock + brand-loyal mix + "
            f"own margin) can extract value beyond the best fixed setting"
        ),
        best_loyalty_spend,
        best_profit,
    )


def loyalty_vs_promotion_orthogonality_verdict(
    *,
    base_loyalty: float,
    base_promotion: float,
    base_profit: float,
    base_std_error: float,
    swap_profits: tuple[tuple[float, float, float, float], ...],
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> LoyaltyVsPromotionOrthogonality:
    """ """
    if not swap_profits:
        return LoyaltyVsPromotionOrthogonality(
            base_loyalty=float(base_loyalty),
            base_promotion=float(base_promotion),
            base_profit=float(base_profit),
            swap_profits=(),
            max_swap_delta=0.0,
            threshold=0.0,
            passed=False,
            reason=(
                "no swaps tested: the orthogonality-vs-promotion diagnostic ran "
                "with zero substitution cells (degenerate FAIL — the diagnostic "
                "cannot conclude levers are orthogonal without at least one swap)"
            ),
        )

    # The substitution cell that moves profit FURTHEST from the base is the
    # discriminator — if even the most-different swap doesn't move profit by
    # margin, every swap is within noise and the levers are degenerate.
    best_swap = max(swap_profits, key=lambda s: abs(s[2] - float(base_profit)))
    swap_loyalty, swap_promotion, swap_profit, swap_se = best_swap
    max_delta = abs(swap_profit - float(base_profit))
    combined_se = math.hypot(float(base_std_error), float(swap_se))
    threshold = float(margin_sigma_multiple) * combined_se

    if max_delta > threshold:
        return LoyaltyVsPromotionOrthogonality(
            base_loyalty=float(base_loyalty),
            base_promotion=float(base_promotion),
            base_profit=float(base_profit),
            swap_profits=tuple(swap_profits),
            max_swap_delta=float(max_delta),
            threshold=float(threshold),
            passed=True,
            reason=(
                f"""orthogonal: strict substitution (loyalty {swap_loyalty:.3f}, promotion {swap_promotion:.3f}) moved profit by {max_delta:.2f} (>{margin_sigma_multiple:g}x combined std-error = {threshold:.2f}) vs the base (loyalty {base_loyalty:.3f}, promotion {base_promotion:.3f}, profit {base_profit:.2f}); the levers are NOT economically substitutable (the boost is moat-anchored and the lift is flat — structurally distinct, F6)"""
            ),
        )

    return LoyaltyVsPromotionOrthogonality(
        base_loyalty=float(base_loyalty),
        base_promotion=float(base_promotion),
        base_profit=float(base_profit),
        swap_profits=tuple(swap_profits),
        max_swap_delta=float(max_delta),
        threshold=float(threshold),
        passed=False,
        reason=(
            f"degenerate: every strict promotion<->loyalty substitution preserved "
            f"profit within {margin_sigma_multiple:g}x combined std-error "
            f"(max |delta| {max_delta:.2f} <= threshold {threshold:.2f}); the "
            f"two levers are economically isomorphic at the joint optima — "
            f"loyalty has collapsed into 'promotion with extra steps' (F6 "
            f"orthogonality violation; the boost is no longer moat-anchored)"
        ),
    )


def _mean_and_std_error(profits: list[float]) -> tuple[float, float, int]:
    """Mean, standard error of the mean, and sample count for a profit list."""
    arr = np.asarray(profits, dtype=np.float64)
    n = arr.size
    # Sample std (ddof=1) / sqrt(n) is the standard error of the mean; guard the
    # n == 1 case where the sample std is undefined.
    std_error = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return float(arr.mean()), std_error, int(n)


def _seat_present_both(state: WorldState) -> None:
    """ """
    agent = state.retailers[AGENT_INDEX]
    mature_age = agent.store_age_per_region[0] if agent.store_age_per_region else 0
    agent.stores_per_region = tuple(1 for _ in agent.stores_per_region)
    agent.store_age_per_region = tuple(mature_age for _ in agent.store_age_per_region)


def _profit_samples_for_cell(
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    *,
    config: CoreConfig,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
    start_present_both: bool = False,
) -> tuple[list[float], list[float]]:
    """ """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    if start_present_both:
        _seat_present_both(state)
    fixed_action = np.array(
        [price, spend, assortment, promotion, 0.0, 0.0]
        + [0.0] * 30
        + [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    profits: list[float] = []
    market_shares: list[float] = []
    for tick in range(ticks):
        result = world.step(state, {AGENT_INDEX: fixed_action})
        state = result.next_state
        if tick < burn_in_ticks:
            continue  # discard the awareness + stockpile transient before measuring
        agent = state.retailers[AGENT_INDEX]
        profits.append(float(agent.last_profit))
        market_shares.append(float(agent.last_market_share))
    return profits, market_shares


def _default_burn_in_ticks(config: CoreConfig) -> int:
    """"""
    awareness_burn_in = math.ceil(3.0 / config.marketing.decay)
    stockpile_burn_in = math.ceil(3.0 / config.promotion.stockpile_decay)
    return max(awareness_burn_in, stockpile_burn_in)


def _sweep_2d(
    config: CoreConfig,
    *,
    grid_points: int,
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    pinned_assortment: float,
    pinned_promotion: float,
    start_present_both: bool,
) -> list[SweepRow2D]:
    """Run the 2-D price x marketing grid at a pinned assortment + promotion."""
    price_low = config.pricing.min_price_index
    price_high = config.pricing.max_price_index
    price_grid = np.linspace(price_low, price_high, grid_points)
    spend_grid = np.linspace(SPEND_LOW, SPEND_HIGH, grid_points)

    rows: list[SweepRow2D] = []
    for price in price_grid:
        price_f = float(price)
        for spend in spend_grid:
            spend_f = float(spend)
            all_profits: list[float] = []
            all_shares: list[float] = []
            for seed in seeds:
                profits, shares = _profit_samples_for_cell(
                    price_f,
                    spend_f,
                    pinned_assortment,
                    pinned_promotion,
                    config=config,
                    ticks=ticks_per_point,
                    burn_in_ticks=burn_in_ticks,
                    seed=seed,
                    start_present_both=start_present_both,
                )
                all_profits.extend(profits)
                all_shares.extend(shares)

            mean_profit, std_error, n = _mean_and_std_error(all_profits)
            rows.append(
                SweepRow2D(
                    price_index=price_f,
                    spend_fraction=spend_f,
                    mean_profit=mean_profit,
                    std_error=std_error,
                    mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                    n_samples=n,
                )
            )
    return rows


def _sweep_assortment_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    pinned_promotion: float,
    grid_points: int,
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1D]:
    """Run the 1-D assortment axis at the (price, marketing) optimum, promo pinned."""
    assort_grid = np.linspace(ASSORT_LOW, ASSORT_HIGH, grid_points)

    rows: list[SweepRow1D] = []
    for assortment in assort_grid:
        assortment_f = float(assortment)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _profit_samples_for_cell(
                fixed_price,
                fixed_spend,
                assortment_f,
                pinned_promotion,
                config=config,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_profit, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            SweepRow1D(
                assortment=assortment_f,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                mean_profit=mean_profit,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows


def _sweep_promotion_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    grid_points: int,
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1DPromotion]:
    """ """
    promo_grid = np.linspace(PROMO_LOW, PROMO_HIGH, grid_points)

    rows: list[SweepRow1DPromotion] = []
    for promotion in promo_grid:
        promotion_f = float(promotion)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _profit_samples_for_cell(
                fixed_price,
                fixed_spend,
                fixed_assortment,
                promotion_f,
                config=config,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_profit, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            SweepRow1DPromotion(
                promotion=promotion_f,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                mean_profit=mean_profit,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows


def _expansion_action(
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    *,
    open_region_1: bool,
) -> npt.NDArray[np.float32]:
    """ """
    open_logit = 1.0 if open_region_1 else 0.0
    return np.array(
        [
            price,
            spend,
            assortment,
            promotion,
            0.0,
            open_logit,
            *([0.0] * 30),
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _expansion_timing_window_total(
    *,
    open_tick: int,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    config: CoreConfig,
    window_ticks: int,
    seed: int,
) -> float:
    """Total agent profit over the scoring window for ONE expansion-timing policy.

    Runs a full trajectory on the :class:`World` seam (no RL): the agent holds the
    continuous optimum every tick and emits the open-region-1 action ONLY at
    ``open_tick`` (a never-expand policy passes ``open_tick == NEVER_EXPAND_T``, which
    never matches a tick index, so it never opens). The seam's cash/presence gate
    decides whether the open at ``open_tick`` actually fires. The score is the SUM of
    per-tick agent profit over the window (the full-trajectory total, not a per-tick
    mean) — that is what makes WHEN to open (paying capex earlier, then carrying the
    per-store opex longer vs building the loyal-segment moat sooner) a real tradeoff.
    """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    total_profit = 0.0
    for tick in range(window_ticks):
        action = _expansion_action(
            fixed_price,
            fixed_spend,
            fixed_assortment,
            fixed_promotion,
            open_region_1=(tick == open_tick),
        )
        result = world.step(state, {AGENT_INDEX: action})
        state = result.next_state
        total_profit += float(state.retailers[AGENT_INDEX].last_profit)
    return total_profit


def _sweep_expansion_timing(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    timings: tuple[int, ...],
    window_ticks: int,
    seeds: tuple[int, ...],
) -> list[ExpansionTimingRow]:
    """ """
    rows: list[ExpansionTimingRow] = []
    for open_tick in (NEVER_EXPAND_T, *timings):
        per_seed_totals = [
            _expansion_timing_window_total(
                open_tick=open_tick,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                config=config,
                window_ticks=window_ticks,
                seed=seed,
            )
            for seed in seeds
        ]
        mean_profit, std_error, n = _mean_and_std_error(per_seed_totals)
        rows.append(
            ExpansionTimingRow(
                open_tick=open_tick,
                mean_profit=mean_profit,
                std_error=std_error,
                n_seeds=n,
            )
        )
    return rows


def _competitor_price_obs_index(config: CoreConfig) -> int:
    """The flat-obs index of the perceived ``competitor_price`` field (from the schema).

    Read from ``OBSERVATION_SCHEMA`` (the layout authority) rather than hardcoded, so a
    future obs-layout change moves the reactive heuristic's read with it. ``config`` is
    accepted for symmetry with the rest of the sweep (the schema is global today).
    """
    from retail_simulator.core.schema import OBSERVATION_SCHEMA

    for idx, field in enumerate(OBSERVATION_SCHEMA):
        if field.name == _COMPETITOR_PRICE_OBS_NAME:
            return idx
    raise ValueError(f"no {_COMPETITOR_PRICE_OBS_NAME!r} field in OBSERVATION_SCHEMA")


def research_gate_segments() -> dict[str, SegmentParams]:
    """ """
    segments = default_segments()
    segments[PRICE_SENSITIVE] = replace(
        segments[PRICE_SENSITIVE], beta_reference=RESEARCH_GATE_BETA_REFERENCE
    )
    return segments


def research_opponent_config(
    base_config: CoreConfig,
    *,
    opponent_archetype: str = RESEARCH_OPPONENT_ARCHETYPE,
) -> CoreConfig:
    """ """
    region_cfgs = base_config.demand.regions
    n_regions = len(region_cfgs) if region_cfgs else 1
    seat_plan = (
        SeatSpec(is_npc=False, name="agent", presence=(1,) * n_regions),
        SeatSpec(is_npc=True, archetype=opponent_archetype, name=opponent_archetype),
    )
    return replace(base_config, seats=seat_plan, segments=research_gate_segments())


def _research_samples_for_cell(
    research_spend: float,
    *,
    config: CoreConfig,
    competitor_price_idx: int,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    """ """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    price_low = config.pricing.min_price_index
    price_high = config.pricing.max_price_index
    # The first read is the reset perceived view (seated to TRUE values at reset). Seed
    # the heuristic's price from the agent's own initial price slot so tick 0 has a
    # sensible action before any perceived competitor read exists.
    reset_obs, _ = world.agent_observation(state, AGENT_INDEX)
    perceived_competitor_price = float(reset_obs[competitor_price_idx])

    profits: list[float] = []
    market_shares: list[float] = []
    for tick in range(ticks):
        # Best-response: undercut the perceived competitor price a notch, clamped into
        # the pricing band. If the perceived price reads as the 0.0 absent-competitor
        # sentinel (should not happen here — the opponent is always seated), fall back to
        # a mid price. A noisier perceived price ⇒ a worse-aimed undercut.
        if perceived_competitor_price > price_low:
            target = perceived_competitor_price * 0.97
        else:
            target = 0.5 * (price_low + price_high)
        price = float(min(max(target, price_low), price_high))
        action = np.array(
            [price, 0.4, 0.5, 0.0, 0.0, 0.0]
            + [0.0] * 30
            + [research_spend, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        result = world.step(state, {AGENT_INDEX: action})
        state = result.next_state
        # Read THIS tick's seated perceived competitor price for next tick's response.
        perceived_competitor_price = float(result.observations[AGENT_INDEX][competitor_price_idx])
        if tick < burn_in_ticks:
            continue
        agent = state.retailers[AGENT_INDEX]
        profits.append(float(agent.last_profit))
        market_shares.append(float(agent.last_market_share))
    return profits, market_shares


def _sweep_research_axis(
    config: CoreConfig,
    *,
    research_spends: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
) -> list[ResearchFidelityRow]:
    """ """
    opponent_config = research_opponent_config(config)
    competitor_price_idx = _competitor_price_obs_index(opponent_config)

    rows: list[ResearchFidelityRow] = []
    for research_spend in research_spends:
        spend_f = float(research_spend)
        fidelity = opponent_config.research.fidelity(spend_f)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _research_samples_for_cell(
                spend_f,
                config=opponent_config,
                competitor_price_idx=competitor_price_idx,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_score, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            ResearchFidelityRow(
                research_spend=spend_f,
                fidelity=float(fidelity),
                mean_score=mean_score,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows


def scm_gate_segments(
    base_segments: dict[str, SegmentParams] | None = None,
) -> dict[str, SegmentParams]:
    """ """
    segments = (
        {key: value for key, value in base_segments.items()}
        if base_segments is not None
        else default_segments()
    )
    if CONVENIENCE not in segments:
        raise ValueError(
            "scm_gate_segments requires the CONVENIENCE segment in base_segments (the SCM lever concentrates beta_service in CONVENIENCE — /-F4 of); got keys"
            + repr(sorted(segments))
        )
    segments[CONVENIENCE] = replace(
        segments[CONVENIENCE], beta_service=SCM_GATE_CONVENIENCE_BETA_SERVICE
    )
    return segments


def scm_gate_config(
    base_config: CoreConfig | None = None,
    *,
    base_segments: dict[str, SegmentParams] | None = None,
) -> CoreConfig:
    """ """
    base = base_config if base_config is not None else CoreConfig.default()
    return replace(
        base,
        scm=SCMConfig(
            cogs_premium=SCM_GATE_COGS_PREMIUM,
            fill_rate_floor=base.scm.fill_rate_floor,
            fill_rate_curvature=base.scm.fill_rate_curvature,
            lost_sales_share_penalty=SCM_GATE_LOST_SALES_SHARE_PENALTY,
            service_score_decay=base.scm.service_score_decay,
        ),
        segments=scm_gate_segments(base_segments=base_segments),
    )


def _scm_action(
    service_level: float,
    *,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
) -> npt.NDArray[np.float32]:
    """ """
    return np.array(
        [
            price,
            spend,
            assortment,
            promotion,
            0.0,
            0.0,
            *([0.0] * 30),
            0.0,
            service_level,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _scm_samples_for_cell(
    service_level: float,
    *,
    config: CoreConfig,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
    start_present_both: bool = False,
) -> tuple[list[float], list[float]]:
    """ """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    if start_present_both:
        _seat_present_both(state)
    fixed_action = _scm_action(
        service_level, price=price, spend=spend, assortment=assortment, promotion=promotion
    )

    profits: list[float] = []
    market_shares: list[float] = []
    for tick in range(ticks):
        result = world.step(state, {AGENT_INDEX: fixed_action})
        state = result.next_state
        if tick < burn_in_ticks:
            continue
        agent = state.retailers[AGENT_INDEX]
        profits.append(float(agent.last_profit))
        market_shares.append(float(agent.last_market_share))
    return profits, market_shares


def _sweep_scm_service_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    service_levels: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1DService]:
    """ """
    rows: list[SweepRow1DService] = []
    for service_level in service_levels:
        s_f = float(service_level)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _scm_samples_for_cell(
                s_f,
                config=config,
                price=fixed_price,
                spend=fixed_spend,
                assortment=fixed_assortment,
                promotion=fixed_promotion,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_profit, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            SweepRow1DService(
                service_level=s_f,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                mean_profit=mean_profit,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows


def _per_region_optimal_score_for_scm(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    service_levels: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
) -> tuple[float, float]:
    """ """
    region_cfgs = config.demand.regions
    n_regions = len(region_cfgs) if region_cfgs else 1

    # For each region r, run the SCM axis with the agent present in ONLY that region.
    # The cleanest way to seat "present in region r only" is the SAME mechanism the
    # continuous chain uses (``_seat_present_both`` flips stores everywhere); here we
    # flip the agent's per-region presence to a single-region mask after reset. Each
    # region's best is the per-region oracle; their sum is the variable score.
    total_mean = 0.0
    summed_variance = 0.0
    for r in range(n_regions):
        per_region_rows: list[SweepRow1DService] = []
        for service_level in service_levels:
            s_f = float(service_level)
            all_profits: list[float] = []
            for seed in seeds:
                profits, _shares = _scm_samples_for_cell_region(
                    s_f,
                    region_index=r,
                    config=config,
                    price=fixed_price,
                    spend=fixed_spend,
                    assortment=fixed_assortment,
                    promotion=fixed_promotion,
                    ticks=ticks_per_point,
                    burn_in_ticks=burn_in_ticks,
                    seed=seed,
                )
                all_profits.extend(profits)

            mean_profit, std_error, n = _mean_and_std_error(all_profits)
            per_region_rows.append(
                SweepRow1DService(
                    service_level=s_f,
                    fixed_price=fixed_price,
                    fixed_spend=fixed_spend,
                    fixed_assortment=fixed_assortment,
                    fixed_promotion=fixed_promotion,
                    mean_profit=mean_profit,
                    std_error=std_error,
                    mean_market_share=0.0,
                    n_samples=n,
                )
            )

        # Per-region best: the highest-profit row in that region's sweep.
        per_region_best = max(per_region_rows, key=lambda r_: r_.mean_profit)
        total_mean += per_region_best.mean_profit
        # Independent per-region rollouts ⇒ summed variance.
        summed_variance += per_region_best.std_error**2

    variable_std_error = math.sqrt(summed_variance)
    return total_mean, variable_std_error


def _seat_present_in_region_only(state: WorldState, region_index: int) -> None:
    """Seat the agent PRESENT IN ONLY ``region_index`` at reset (per-region heuristic).

    For the per-region-optimal heuristic (:func:`_per_region_optimal_score_for_scm`).
    Flips the agent's ``stores_per_region`` to a one-hot mask at ``region_index`` —
    the sweep-only regime that isolates per-region SCM economics. The seam reads
    presence to decide which regions the retailer contributes utility to, so an
    agent absent from a region pays no opex / earns no demand there.

    Stage 3 (store-ramp-up, section 4.11; senior-review m2, 2026-09-16): mirrors
    ``_seat_present_both``'s age fix -- seats ``region_index``'s age as MATURE
    (broadcasting ``store_age_per_region[0]``, still correct here since only
    PRESENCE changes below, not the ages world.reset already seeded) and every
    other region's age back to its never-opened default (0), matching their
    forced-absent presence.
    """
    agent = state.retailers[AGENT_INDEX]
    mature_age = agent.store_age_per_region[0] if agent.store_age_per_region else 0
    agent.stores_per_region = tuple(
        1 if r == region_index else 0 for r in range(len(agent.stores_per_region))
    )
    agent.store_age_per_region = tuple(
        mature_age if r == region_index else 0 for r in range(len(agent.store_age_per_region))
    )


def _scm_samples_for_cell_region(
    service_level: float,
    *,
    region_index: int,
    config: CoreConfig,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Single-region variant of :func:`_scm_samples_for_cell` for the oracle heuristic.

    Same as :func:`_scm_samples_for_cell` but seats the agent PRESENT IN ONLY
    ``region_index`` at reset (the per-region single-region world the
    per-region-optimal oracle runs in).
    """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    _seat_present_in_region_only(state, region_index)
    fixed_action = _scm_action(
        service_level, price=price, spend=spend, assortment=assortment, promotion=promotion
    )

    profits: list[float] = []
    market_shares: list[float] = []
    for tick in range(ticks):
        result = world.step(state, {AGENT_INDEX: fixed_action})
        state = result.next_state
        if tick < burn_in_ticks:
            continue
        agent = state.retailers[AGENT_INDEX]
        profits.append(float(agent.last_profit))
        market_shares.append(float(agent.last_market_share))
    return profits, market_shares


def automation_gate_segments() -> dict[str, SegmentParams]:
    """ """
    return default_segments()


def automation_gate_config(base_config: CoreConfig | None = None) -> CoreConfig:
    """ """
    base = base_config if base_config is not None else CoreConfig.default()
    return replace(
        base,
        automation=AutomationConfig(
            capex_per_tier=AUTOMATION_GATE_CAPEX_PER_TIER,
            savings_per_tier=AUTOMATION_GATE_SAVINGS_PER_TIER,
        ),
    )


def _automation_action(
    automation_tier: int,
    *,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
) -> npt.NDArray[np.float32]:
    """ """
    auto_logits = [0.0, 0.0, 0.0]
    auto_logits[automation_tier] = 1.0
    return np.array(
        [
            price,
            spend,
            assortment,
            promotion,
            0.0,
            0.0,
            *([0.0] * 30),
            0.0,
            1.0,
            *auto_logits,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _automation_policy_name(*, target_tier: int, upgrade_tick: int, ramp_tick: int = 0) -> str:
    """Human-readable label for an automation upgrade-timing policy (CSV / format)."""
    if upgrade_tick == NEVER_UPGRADE_T:
        return "never_upgrade"
    if upgrade_tick == RAMP_POLICY_T:
        return f"ramp_0_to_1_to_2_at_t={ramp_tick}"
    return f"upgrade_to_{target_tier}_at_t={upgrade_tick}"


def _automation_target_for_tick(
    tick: int, *, target_tier: int, upgrade_tick: int, ramp_tick: int
) -> int:
    """The TARGET tier the agent picks at ``tick`` under the given upgrade-timing policy.

    Under F3 = PERMANENT, the agent's lever is the DECODED TARGET; the seam's gate
    resolves monotonicity + differential-affordability. To realize a one-shot
    upgrade-at-T policy, the agent picks target 0 before T and ``target_tier`` from
    T onward (the same-tier no-op at every later tick is free; the seam never
    downgrades). For the ramp policy, target 1 at t=0..ramp_tick-1 and target 2
    from ramp_tick onward.
    """
    if upgrade_tick == NEVER_UPGRADE_T:
        return 0
    if upgrade_tick == RAMP_POLICY_T:
        return 1 if tick < ramp_tick else 2
    return target_tier if tick >= upgrade_tick else 0


def _automation_timing_window_total(
    *,
    target_tier: int,
    upgrade_tick: int,
    ramp_tick: int,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    config: CoreConfig,
    window_ticks: int,
    seed: int,
    start_present_both: bool,
) -> float:
    """Total agent profit over the scoring window for ONE automation upgrade-timing policy.

    Runs a full trajectory on the :class:`World` seam (no RL): the agent holds the
    continuous optimum every tick and emits the TARGET automation tier per
    :func:`_automation_target_for_tick`. The seam's upgrade gate decides whether
    each tick's target actually fires (a same-tier pick is a free no-op; an
    unaffordable differential at the upgrade tick no-ops, itself a finding). The
    score is the SUM of per-tick agent profit over the window (the full-trajectory
    total — what makes WHEN to upgrade a real tradeoff: an early upgrade pays the
    capex sooner but harvests savings for more ticks; a late upgrade waits for
    cash but loses tail-window savings).
    """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    if start_present_both:
        _seat_present_both(state)
    total_profit = 0.0
    for tick in range(window_ticks):
        tier = _automation_target_for_tick(
            tick,
            target_tier=target_tier,
            upgrade_tick=upgrade_tick,
            ramp_tick=ramp_tick,
        )
        action = _automation_action(
            tier,
            price=fixed_price,
            spend=fixed_spend,
            assortment=fixed_assortment,
            promotion=fixed_promotion,
        )
        result = world.step(state, {AGENT_INDEX: action})
        state = result.next_state
        total_profit += float(state.retailers[AGENT_INDEX].last_profit)
    return total_profit


def _sweep_automation_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    timings: tuple[int, ...],
    window_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1DAutomation]:
    """ """
    # Pick a ramp_tick from the available timings (skip t=0 — the ramp must split
    # the upgrade into two separate ticks). Fall back to a small interior value if
    # all timings collapse to 0.
    interior_timings = [t for t in timings if t > 0]
    ramp_tick = interior_timings[0] if interior_timings else 5

    policy_specs: list[tuple[int, int, int]] = [(0, NEVER_UPGRADE_T, 0)]
    # upgrade_to_1_at_t=0 — the cheap-immediate upgrade (mostly to confirm the
    # gate fires; tier 1 dominated by tier 2 in any reasonable calibration).
    policy_specs.append((1, 0, 0))
    # upgrade_to_2_at_t=T for each T in timings (T=0 is the "immediate" free-lunch
    # check; interior T values are the load-bearing non-degeneracy test).
    for t in timings:
        policy_specs.append((2, int(t), 0))
    # ramp_0_to_1_to_2 — gradual path; path-independent total capex.
    policy_specs.append((2, RAMP_POLICY_T, ramp_tick))

    rows: list[SweepRow1DAutomation] = []
    for target_tier, upgrade_tick, ramp_t in policy_specs:
        per_seed_totals = [
            _automation_timing_window_total(
                target_tier=target_tier,
                upgrade_tick=upgrade_tick,
                ramp_tick=ramp_t,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                config=config,
                window_ticks=window_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            for seed in seeds
        ]
        mean_profit, std_error, n = _mean_and_std_error(per_seed_totals)
        rows.append(
            SweepRow1DAutomation(
                policy_name=_automation_policy_name(
                    target_tier=target_tier,
                    upgrade_tick=upgrade_tick,
                    ramp_tick=ramp_t,
                ),
                target_tier=target_tier,
                upgrade_tick=upgrade_tick,
                ramp_tick=ramp_t,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                mean_profit=mean_profit,
                std_error=std_error,
                n_seeds=n,
            )
        )
    return rows


def loyalty_gate_segments(
    *, beta_program_loyal: float = LOYALTY_GATE_BRAND_LOYAL_BETA_PROGRAM
) -> dict[str, SegmentParams]:
    """ """
    segments = default_segments()
    segments[BRAND_LOYAL] = replace(segments[BRAND_LOYAL], beta_program=beta_program_loyal)
    return segments


def loyalty_gate_config(
    base_config: CoreConfig | None = None,
    *,
    beta_program_loyal: float = LOYALTY_GATE_BRAND_LOYAL_BETA_PROGRAM,
    cost_per_unit_spend: float = LOYALTY_GATE_COST_PER_UNIT_SPEND,
) -> CoreConfig:
    """ """
    base = base_config if base_config is not None else CoreConfig.default()
    return replace(
        base,
        loyalty_program=LoyaltyProgramConfig(cost_per_unit_spend=cost_per_unit_spend),
        segments=loyalty_gate_segments(beta_program_loyal=beta_program_loyal),
    )


def _loyalty_action(
    loyalty_spend: float,
    *,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
) -> npt.NDArray[np.float32]:
    """ """
    return np.array(
        [
            price,
            spend,
            assortment,
            promotion,
            0.0,
            0.0,
            *([0.0] * 30),
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            loyalty_spend,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _loyalty_samples_for_cell(
    loyalty_spend: float,
    *,
    config: CoreConfig,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
    start_present_both: bool = False,
) -> tuple[list[float], list[float]]:
    """ """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    if start_present_both:
        _seat_present_both(state)
    fixed_action = _loyalty_action(
        loyalty_spend, price=price, spend=spend, assortment=assortment, promotion=promotion
    )

    profits: list[float] = []
    market_shares: list[float] = []
    for tick in range(ticks):
        result = world.step(state, {AGENT_INDEX: fixed_action})
        state = result.next_state
        if tick < burn_in_ticks:
            continue
        agent = state.retailers[AGENT_INDEX]
        profits.append(float(agent.last_profit))
        market_shares.append(float(agent.last_market_share))
    return profits, market_shares


def _sweep_loyalty_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    loyalty_spends: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1DLoyalty]:
    """ """
    rows: list[SweepRow1DLoyalty] = []
    for loyalty_spend in loyalty_spends:
        l_f = float(loyalty_spend)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _loyalty_samples_for_cell(
                l_f,
                config=config,
                price=fixed_price,
                spend=fixed_spend,
                assortment=fixed_assortment,
                promotion=fixed_promotion,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_profit, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            SweepRow1DLoyalty(
                loyalty_spend=l_f,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                mean_profit=mean_profit,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows


def _loyalty_promotion_swap_action(
    loyalty_spend: float,
    promotion: float,
    *,
    price: float,
    spend: float,
    assortment: float,
) -> npt.NDArray[np.float32]:
    """ """
    return np.array(
        [
            price,
            spend,
            assortment,
            promotion,
            0.0,
            0.0,
            *([0.0] * 30),
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            loyalty_spend,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _sweep_loyalty_orthogonality_probe(
    config: CoreConfig,
    *,
    base_loyalty: float,
    base_promotion: float,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    swap_deltas: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> tuple[float, float, tuple[tuple[float, float, float, float], ...]]:
    """ """

    def _run(loyalty: float, promotion: float) -> tuple[float, float]:
        all_profits: list[float] = []
        for seed in seeds:
            world = World(config=config, seed=seed)
            state = world.reset(seed)
            if start_present_both:
                _seat_present_both(state)
            action = _loyalty_promotion_swap_action(
                loyalty,
                promotion,
                price=fixed_price,
                spend=fixed_spend,
                assortment=fixed_assortment,
            )
            for tick in range(ticks_per_point):
                result = world.step(state, {AGENT_INDEX: action})
                state = result.next_state
                if tick < burn_in_ticks:
                    continue
                all_profits.append(float(state.retailers[AGENT_INDEX].last_profit))
        mean, se, _n = _mean_and_std_error(all_profits)
        return mean, se

    base_mean, base_se = _run(base_loyalty, base_promotion)

    swap_cells: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float]] = set()
    for delta in swap_deltas:
        d = float(delta)
        if d <= 0.0:
            continue
        # Two symmetric substitutions per delta: (a) raise loyalty by delta, lower
        # promotion by delta; (b) raise promotion by delta, lower loyalty by delta.
        candidates = (
            (
                min(max(base_loyalty + d, LOYALTY_LOW), LOYALTY_HIGH),
                min(max(base_promotion - d, PROMO_LOW), PROMO_HIGH),
            ),
            (
                min(max(base_loyalty - d, LOYALTY_LOW), LOYALTY_HIGH),
                min(max(base_promotion + d, PROMO_LOW), PROMO_HIGH),
            ),
        )
        for new_loyalty, new_promotion in candidates:
            key = (round(new_loyalty, 6), round(new_promotion, 6))
            # Skip the no-op (clipping collapsed the swap to the base) AND duplicates.
            if (
                math.isclose(new_loyalty, base_loyalty, abs_tol=_BOUNDARY_ATOL)
                and math.isclose(new_promotion, base_promotion, abs_tol=_BOUNDARY_ATOL)
            ) or key in seen:
                continue
            seen.add(key)
            mean, se = _run(new_loyalty, new_promotion)
            swap_cells.append((float(new_loyalty), float(new_promotion), float(mean), float(se)))

    return float(base_mean), float(base_se), tuple(swap_cells)


# The default substitution deltas the orthogonality probe sweeps. A small (0.1)
# AND a larger (0.3) shift cover both fine and coarse substitutions; the verdict
# uses the max-delta cell so the bar is "does AT LEAST ONE substitution move
# profit by margin" (the orthogonal-by-construction reading).
DEFAULT_LOYALTY_ORTHO_DELTAS: tuple[float, ...] = (0.1, 0.3)


def run_sweep(
    config: CoreConfig | None = None,
    *,
    grid_points: int = DEFAULT_GRID_POINTS,
    ticks_per_point: int = DEFAULT_TICKS_PER_POINT,
    burn_in_ticks: int | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    pinned_assortment: float = PINNED_ASSORTMENT,
    pinned_promotion: float = PINNED_PROMOTION,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
    start_present_both: bool = False,
    run_expansion_timing: bool = True,
    expansion_timings: tuple[int, ...] = DEFAULT_EXPANSION_TIMINGS,
    expansion_window_ticks: int = SCORING_WINDOW_TICKS,
    run_research: bool = True,
    research_spends: tuple[float, ...] = DEFAULT_RESEARCH_SPENDS,
    run_scm: bool = True,
    scm_service_levels: tuple[float, ...] = DEFAULT_SERVICE_LEVELS,
    run_automation: bool = True,
    automation_timings: tuple[int, ...] = DEFAULT_AUTOMATION_TIMINGS,
    automation_window_ticks: int = SCORING_WINDOW_TICKS,
    run_loyalty: bool = True,
    loyalty_spends: tuple[float, ...] = DEFAULT_LOYALTY_SPENDS,
    loyalty_ortho_deltas: tuple[float, ...] = DEFAULT_LOYALTY_ORTHO_DELTAS,
) -> SweepResult:
    """ """
    if config is None:
        config = CoreConfig.default()
    if grid_points < 2:
        raise ValueError("grid_points must be >= 2 (need both range endpoints)")
    if burn_in_ticks is None:
        burn_in_ticks = _default_burn_in_ticks(config)
    if burn_in_ticks < 0:
        raise ValueError("burn_in_ticks must be >= 0")
    if ticks_per_point <= burn_in_ticks:
        raise ValueError(
            f"ticks_per_point ({ticks_per_point}) must exceed burn_in_ticks "
            f"({burn_in_ticks}) so a post-burn-in sample remains"
        )
    if not seeds:
        raise ValueError("at least one seed is required")

    price_low = config.pricing.min_price_index
    price_high = config.pricing.max_price_index

    # --- Pass 1: the 2-D price x marketing surface at pinned assortment + promo. -
    rows = _sweep_2d(
        config,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        burn_in_ticks=burn_in_ticks,
        seeds=seeds,
        pinned_assortment=pinned_assortment,
        pinned_promotion=pinned_promotion,
        start_present_both=start_present_both,
    )
    row_tuple = tuple(rows)
    is_degenerate_2d, reason_2d, profit_max_cell, profit_max_value = verdict_for_sweep(
        row_tuple,
        price_low=price_low,
        price_high=price_high,
        margin_sigma_multiple=margin_sigma_multiple,
    )

    # --- Pass 2: the 1-D assortment axis at the joint (price*, marketing*). ---
    best_price, best_spend = profit_max_cell
    assortment_rows = _sweep_assortment_axis(
        config,
        fixed_price=best_price,
        fixed_spend=best_spend,
        pinned_promotion=pinned_promotion,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        burn_in_ticks=burn_in_ticks,
        seeds=seeds,
        start_present_both=start_present_both,
    )
    assortment_tuple = tuple(assortment_rows)
    (
        is_degenerate_assort,
        reason_assort,
        best_assortment,
        assortment_max_value,
    ) = assortment_verdict_for_axis(
        assortment_tuple,
        margin_sigma_multiple=margin_sigma_multiple,
    )

    # --- Pass 3: the 1-D promotion axis at (price*, marketing*, assortment*),
    # measured at steady state (the burn-in covers awareness AND the stockpile). ---
    promotion_rows = _sweep_promotion_axis(
        config,
        fixed_price=best_price,
        fixed_spend=best_spend,
        fixed_assortment=best_assortment,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        burn_in_ticks=burn_in_ticks,
        seeds=seeds,
        start_present_both=start_present_both,
    )
    promotion_tuple = tuple(promotion_rows)
    (
        is_degenerate_promo,
        reason_promo,
        best_promotion,
        promotion_max_value,
    ) = promotion_verdict_for_axis(
        promotion_tuple,
        margin_sigma_multiple=margin_sigma_multiple,
    )

    continuous_degenerate = is_degenerate_2d or is_degenerate_assort or is_degenerate_promo

    region_cfgs = config.demand.regions
    n_regions = len(region_cfgs) if region_cfgs else 1
    expansion_timing_tuple: tuple[ExpansionTimingRow, ...] = ()
    expansion_timing_degenerate: bool | None = None
    expansion_timing_reason = ""
    if run_expansion_timing and n_regions >= 2:
        expansion_timing_rows = _sweep_expansion_timing(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            timings=expansion_timings,
            window_ticks=expansion_window_ticks,
            seeds=seeds,
        )
        expansion_timing_tuple = tuple(expansion_timing_rows)
        (
            expansion_timing_degenerate,
            expansion_timing_reason,
            _best_open_tick,
            _best_timing_profit,
        ) = expansion_verdict_for_timing(
            expansion_timing_tuple,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    research_tuple: tuple[ResearchFidelityRow, ...] = ()
    research_degenerate: bool | None = None
    research_reason = ""
    if run_research:
        research_rows = _sweep_research_axis(
            config,
            research_spends=research_spends,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
        )
        research_tuple = tuple(research_rows)
        (
            research_degenerate,
            research_reason,
            _best_research_spend,
            _best_research_score,
        ) = research_verdict_for_fidelity(
            research_tuple,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    service_tuple: tuple[SweepRow1DService, ...] = ()
    service_variable_mean: float | None = None
    service_variable_std_error: float | None = None
    service_degenerate: bool | None = None
    service_reason = ""
    if run_scm and n_regions >= 2:
        scm_rows = _sweep_scm_service_axis(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            service_levels=scm_service_levels,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
            start_present_both=start_present_both,
        )
        service_tuple = tuple(scm_rows)
        # The per-region-optimal heuristic — the FREE variable alternative (the
        # 1.0-lesson antidote). Compares the per-region oracle's score against the
        # best-of-grid constant-service score inside ``scm_verdict_for_service``.
        service_variable_mean, service_variable_std_error = _per_region_optimal_score_for_scm(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            service_levels=scm_service_levels,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
        )
        (
            service_degenerate,
            service_reason,
            _best_constant_service,
            _best_constant_profit,
        ) = scm_verdict_for_service(
            service_tuple,
            service_variable_mean,
            service_variable_std_error,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    automation_tuple: tuple[SweepRow1DAutomation, ...] = ()
    automation_degenerate: bool | None = None
    automation_reason = ""
    if run_automation:
        automation_rows = _sweep_automation_axis(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            timings=automation_timings,
            window_ticks=automation_window_ticks,
            seeds=seeds,
            start_present_both=start_present_both,
        )
        automation_tuple = tuple(automation_rows)
        (
            automation_degenerate,
            automation_reason,
            _best_policy_name,
            _best_profit,
        ) = automation_verdict_for_timing(
            automation_tuple,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    loyalty_tuple: tuple[SweepRow1DLoyalty, ...] = ()
    loyalty_degenerate: bool | None = None
    loyalty_reason = ""
    loyalty_orthogonality: LoyaltyVsPromotionOrthogonality | None = None
    if run_loyalty:
        loyalty_rows = _sweep_loyalty_axis(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            loyalty_spends=loyalty_spends,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
            start_present_both=start_present_both,
        )
        loyalty_tuple = tuple(loyalty_rows)
        (
            loyalty_degenerate,
            loyalty_reason,
            best_loyalty_spend,
            _best_loyalty_profit,
        ) = loyalty_verdict_for_spend(
            loyalty_tuple,
            margin_sigma_multiple=margin_sigma_multiple,
        )

        # F6 orthogonality-vs-promotion diagnostic at the joint optima
        # (loyalty*, promotion*). Anchor at the loyalty-axis best AND the
        # promotion-axis best (the chained continuous optimum, already found
        # above). At least one strict substitution must move profit by > 3σ to
        # PASS — otherwise the levers are economically indistinguishable.
        base_mean, base_se, swap_cells = _sweep_loyalty_orthogonality_probe(
            config,
            base_loyalty=best_loyalty_spend,
            base_promotion=best_promotion,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            swap_deltas=loyalty_ortho_deltas,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
            start_present_both=start_present_both,
        )
        loyalty_orthogonality = loyalty_vs_promotion_orthogonality_verdict(
            base_loyalty=best_loyalty_spend,
            base_promotion=best_promotion,
            base_profit=base_mean,
            base_std_error=base_se,
            swap_profits=swap_cells,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    timing_fails = bool(expansion_timing_degenerate)
    research_fails = bool(research_degenerate)
    scm_fails = bool(service_degenerate)
    automation_fails = bool(automation_degenerate)
    loyalty_fails = bool(loyalty_degenerate) or (
        loyalty_orthogonality is not None and not loyalty_orthogonality.passed
    )
    is_degenerate = (
        continuous_degenerate
        or timing_fails
        or research_fails
        or scm_fails
        or automation_fails
        or loyalty_fails
    )

    return SweepResult(
        rows=row_tuple,
        assortment_rows=assortment_tuple,
        promotion_rows=promotion_tuple,
        profit_max_cell=profit_max_cell,
        profit_max_value=profit_max_value,
        assortment_max_value=assortment_max_value,
        assortment_max_cell=best_assortment,
        promotion_max_value=promotion_max_value,
        # PASS requires non-degenerate on the continuous chain (price AND marketing AND
        # assortment AND promotion) AND, when run, the expansion-timing leg AND, when
        # run, the research (value-of-information) leg AND, when run, the SCM leg AND,
        # when run, the Phase-1.2 automation leg — the SEVEN-leg AND truth table.
        is_degenerate=is_degenerate,
        reason=reason_2d,
        assortment_reason=reason_assort,
        promotion_reason=reason_promo,
        pinned_assortment=pinned_assortment,
        pinned_promotion=pinned_promotion,
        ticks_per_point=ticks_per_point,
        burn_in_ticks=burn_in_ticks,
        seeds=tuple(seeds),
        expansion_timing_rows=expansion_timing_tuple,
        expansion_timing_degenerate=expansion_timing_degenerate,
        expansion_timing_reason=expansion_timing_reason,
        start_present_both=start_present_both,
        research_rows=research_tuple,
        research_degenerate=research_degenerate,
        research_reason=research_reason,
        service_rows=service_tuple,
        service_variable_mean=service_variable_mean,
        service_variable_std_error=service_variable_std_error,
        service_degenerate=service_degenerate,
        service_reason=service_reason,
        automation_rows=automation_tuple,
        automation_degenerate=automation_degenerate,
        automation_reason=automation_reason,
        loyalty_rows=loyalty_tuple,
        loyalty_degenerate=loyalty_degenerate,
        loyalty_reason=loyalty_reason,
        loyalty_orthogonality=loyalty_orthogonality,
    )


def format_sweep_table(result: SweepResult) -> str:
    """ """
    header = f"{'price':>8} {'spend':>8} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
    sep = "-" * len(header)
    lines = [
        f"== 2-D price x marketing sweep (assortment pinned at {result.pinned_assortment:.3f}, "
        f"promotion pinned at {result.pinned_promotion:.3f}) ==",
        header,
        sep,
    ]
    best_price, best_spend = result.profit_max_cell
    for row in result.rows:
        is_best = math.isclose(row.price_index, best_price) and math.isclose(
            row.spend_fraction, best_spend
        )
        flag = " *" if is_best else "  "
        lines.append(
            f"{row.price_index:>8.4f} {row.spend_fraction:>8.3f} {row.mean_profit:>14.2f} "
            f"{row.std_error:>10.3f} {row.mean_market_share:>10.4f}{flag}"
        )
    lines.append(sep)
    lines.append(
        f"profit-max cell: price {best_price:.4f}, spend {best_spend:.3f}  "
        f"(profit {result.profit_max_value:.2f})"
    )
    lines.append(f"reason: {result.reason}")

    a_header = f"{'assort':>8} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
    a_sep = "-" * len(a_header)
    lines.append("")
    lines.append(f"== 1-D assortment axis at price {best_price:.4f}, spend {best_spend:.3f} ==")
    lines.append(a_header)
    lines.append(a_sep)
    best_assort_row = max(result.assortment_rows, key=lambda r: r.mean_profit, default=None)
    best_assort = best_assort_row.assortment if best_assort_row is not None else float("nan")
    for arow in result.assortment_rows:
        is_best = math.isclose(arow.assortment, best_assort)
        flag = " *" if is_best else "  "
        lines.append(
            f"{arow.assortment:>8.3f} {arow.mean_profit:>14.2f} "
            f"{arow.std_error:>10.3f} {arow.mean_market_share:>10.4f}{flag}"
        )
    lines.append(a_sep)
    lines.append(
        f"profit-max assortment: {best_assort:.3f}  (profit {result.assortment_max_value:.2f})"
    )
    lines.append(f"reason: {result.assortment_reason}")

    p_header = f"{'promo':>8} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
    p_sep = "-" * len(p_header)
    lines.append("")
    lines.append(
        f"== 1-D promotion axis at price {best_price:.4f}, spend {best_spend:.3f}, "
        f"assortment {result.assortment_max_cell:.3f} (steady state) =="
    )
    lines.append(p_header)
    lines.append(p_sep)
    best_promo_row = max(result.promotion_rows, key=lambda r: r.mean_profit, default=None)
    best_promo = best_promo_row.promotion if best_promo_row is not None else float("nan")
    for prow in result.promotion_rows:
        is_best = math.isclose(prow.promotion, best_promo)
        flag = " *" if is_best else "  "
        lines.append(
            f"{prow.promotion:>8.3f} {prow.mean_profit:>14.2f} "
            f"{prow.std_error:>10.3f} {prow.mean_market_share:>10.4f}{flag}"
        )
    lines.append(p_sep)
    lines.append(
        f"profit-max promotion: {best_promo:.3f}  (profit {result.promotion_max_value:.2f})"
    )
    lines.append(f"reason: {result.promotion_reason}")

    if result.expansion_timing_rows:
        t_header = f"{'open_tick':>10} {'total_profit':>14} {'std_err':>10}"
        t_sep = "-" * len(t_header)
        lines.append("")
        lines.append(
            f"== expansion-timing leg at price {best_price:.4f}, spend {best_spend:.3f}, "
            f"assortment {result.assortment_max_cell:.3f}, promotion {best_promo:.3f} "
            f"(total profit over {SCORING_WINDOW_TICKS}-tick window) =="
        )
        lines.append(t_header)
        lines.append(t_sep)
        expand_rows = [r for r in result.expansion_timing_rows if not r.is_never_expand]
        best_timing = max(expand_rows, key=lambda r: r.mean_profit, default=None)
        best_tick = best_timing.open_tick if best_timing is not None else None
        for trow in result.expansion_timing_rows:
            label = "never" if trow.is_never_expand else f"{trow.open_tick}"
            is_best = (not trow.is_never_expand) and trow.open_tick == best_tick
            flag = " *" if is_best else "  "
            lines.append(f"{label:>10} {trow.mean_profit:>14.2f} {trow.std_error:>10.3f}{flag}")
        lines.append(t_sep)
        lines.append(f"reason: {result.expansion_timing_reason}")

    if result.research_rows:
        r_header = (
            f"{'research':>10} {'fidelity':>10} {'mean_score':>14} "
            f"{'std_err':>10} {'mkt_share':>10}"
        )
        r_sep = "-" * len(r_header)
        lines.append("")
        lines.append(
            "== research (value-of-information) axis vs the wandering NPC "
            "(exogenous, non-inferable varying opponent; reactive best-response heuristic) =="
        )
        lines.append(r_header)
        lines.append(r_sep)
        best_research = max(result.research_rows, key=lambda r: r.mean_score, default=None)
        best_spend = best_research.research_spend if best_research is not None else float("nan")
        for rrow in result.research_rows:
            is_best = math.isclose(rrow.research_spend, best_spend)
            flag = " *" if is_best else "  "
            lines.append(
                f"{rrow.research_spend:>10.3f} {rrow.fidelity:>10.3f} {rrow.mean_score:>14.2f} "
                f"{rrow.std_error:>10.3f} {rrow.mean_market_share:>10.4f}{flag}"
            )
        lines.append(r_sep)
        lines.append(f"reason: {result.research_reason}")

    if result.service_rows:
        s_header = f"{'service':>10} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
        s_sep = "-" * len(s_header)
        lines.append("")
        lines.append(
            "== SCM (value-of-service) axis vs the default Discounter (SCM scenario ON; "
            "the constant-service grid is the FREE no-SCM baseline a free PPO learner "
            "collapses to — the Phase-1.0 lesson) =="
        )
        lines.append(s_header)
        lines.append(s_sep)
        best_scm_row = max(result.service_rows, key=lambda r: r.mean_profit, default=None)
        best_scm = best_scm_row.service_level if best_scm_row is not None else float("nan")
        for srow in result.service_rows:
            is_best = math.isclose(srow.service_level, best_scm)
            flag = " *" if is_best else "  "
            lines.append(
                f"{srow.service_level:>10.3f} {srow.mean_profit:>14.2f} "
                f"{srow.std_error:>10.3f} {srow.mean_market_share:>10.4f}{flag}"
            )
        lines.append(s_sep)
        if result.service_variable_mean is not None:
            v_mean = result.service_variable_mean
            v_se = result.service_variable_std_error if result.service_variable_std_error else 0.0
            lines.append(
                f"per-region-optimal (variable) score: {v_mean:.2f} +/- {v_se:.3f} "
                f"(the FREE-variable alternative — beats best constant by margin ⇒ "
                f"non-degenerate)"
            )
        lines.append(f"reason: {result.service_reason}")

    if result.automation_rows:
        a_header = f"{'policy':>24} {'total_profit':>14} {'std_err':>10}"
        a_sep = "-" * len(a_header)
        lines.append("")
        lines.append(
            "== automation upgrade-timing leg vs the default Discounter "
            "(automation scenario ON; never_upgrade is the FREE no-automation "
            "baseline — the Phase-1.0 lesson) =="
        )
        lines.append(a_header)
        lines.append(a_sep)
        upgrade_rows = [r for r in result.automation_rows if not r.is_never_upgrade]
        best_auto_row = max(upgrade_rows, key=lambda r: r.mean_profit, default=None)
        best_policy_name = best_auto_row.policy_name if best_auto_row is not None else ""
        for auto_row in result.automation_rows:
            is_best = (not auto_row.is_never_upgrade) and auto_row.policy_name == best_policy_name
            flag = " *" if is_best else "  "
            lines.append(
                f"{auto_row.policy_name:>24} {auto_row.mean_profit:>14.2f} "
                f"{auto_row.std_error:>10.3f}{flag}"
            )
        lines.append(a_sep)
        lines.append(f"reason: {result.automation_reason}")

    if result.loyalty_rows:
        l_header = f"{'loyalty':>10} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
        l_sep = "-" * len(l_header)
        lines.append("")
        lines.append(
            "== loyalty-program (value-of-active-loyalty) axis vs the default "
            "Discounter (loyalty scenario ON; the constant-loyalty grid is the "
            "FREE no-program baseline a free PPO learner collapses to — the "
            "Phase-1.0 lesson) =="
        )
        lines.append(l_header)
        lines.append(l_sep)
        best_loy_row = max(result.loyalty_rows, key=lambda r: r.mean_profit, default=None)
        best_loy = best_loy_row.loyalty_spend if best_loy_row is not None else float("nan")
        for lrow in result.loyalty_rows:
            is_best = math.isclose(lrow.loyalty_spend, best_loy)
            flag = " *" if is_best else "  "
            lines.append(
                f"{lrow.loyalty_spend:>10.3f} {lrow.mean_profit:>14.2f} "
                f"{lrow.std_error:>10.3f} {lrow.mean_market_share:>10.4f}{flag}"
            )
        lines.append(l_sep)
        lines.append(f"reason: {result.loyalty_reason}")
        if result.loyalty_orthogonality is not None:
            ortho = result.loyalty_orthogonality
            lines.append(
                f"orthogonality-vs-promotion: max-swap |delta| "
                f"{ortho.max_swap_delta:.2f} vs threshold {ortho.threshold:.2f} "
                f"(passed={ortho.passed}); {ortho.reason}"
            )

    lines.append("")
    verdict = "DEGENERATE" if result.is_degenerate else "NON-DEGENERATE"
    legs = "price AND marketing AND assortment AND promotion"
    if result.expansion_timing_rows:
        legs += " AND expansion-timing"
    if result.research_rows:
        legs += " AND research"
    if result.service_rows:
        legs += " AND SCM-service"
    if result.automation_rows:
        legs += " AND automation"
    if result.loyalty_rows:
        legs += " AND loyalty-program"
    lines.append(f"overall verdict: {verdict} (non-degenerate on {legs})")
    return "\n".join(lines)


def sweep_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the 2-D sweep cells as flat dicts for CSV writing (pandas-readable)."""
    return [
        {
            "price_index": row.price_index,
            "spend_fraction": row.spend_fraction,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.rows
    ]


def assortment_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the assortment-axis cells as flat dicts for CSV writing."""
    return [
        {
            "assortment": row.assortment,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.assortment_rows
    ]


def promotion_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the promotion-axis cells as flat dicts for CSV writing."""
    return [
        {
            "promotion": row.promotion,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "fixed_assortment": row.fixed_assortment,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.promotion_rows
    ]


def expansion_timing_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the expansion-timing policies as flat dicts for CSV writing.

    One row per full-trajectory policy: the never-expand baseline
    (``open_tick == NEVER_EXPAND_T``) plus one per ``expand_at_T``. ``mean_profit``
    is the mean (across seeds) of the TOTAL profit over the scoring window. Empty
    when the timing leg was not run (single region / disabled).
    """
    return [
        {
            "open_tick": row.open_tick,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "n_seeds": row.n_seeds,
        }
        for row in result.expansion_timing_rows
    ]


def research_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the research (value-of-information) axis cells as flat dicts for CSV.

    One row per research-spend cell (the reactive heuristic vs the Balanced NPC), each
    carrying the spend, the resulting fidelity, the mean score (profit netting research
    opex), its std-error, the mean market share and the sample count. Empty when the
    research leg was not run (disabled).
    """
    return [
        {
            "research_spend": row.research_spend,
            "fidelity": row.fidelity,
            "mean_score": row.mean_score,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.research_rows
    ]


def scm_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the SCM (value-of-service) axis cells as flat dicts for CSV (Phase 1.1).

    One row per service_level cell (the constant-service sweep against the default
    Discounter with the SCM scenario ON), each carrying the service_level, the pinned
    continuous optima, the mean profit (already netted of variable COGS by the per-
    retailer ``cogs_fraction`` SoT inside ``DemandResult``), its std-error, the mean
    market share and the sample count. Empty when the SCM leg was not run (disabled or
    single-region).
    """
    return [
        {
            "service_level": row.service_level,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "fixed_assortment": row.fixed_assortment,
            "fixed_promotion": row.fixed_promotion,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.service_rows
    ]


def automation_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int | str]]:
    """Return the automation upgrade-timing axis cells as flat dicts for CSV (Phase 1.2).

    One row per upgrade-timing policy (the whole-trajectory comparison against the
    default Discounter with the automation scenario ON), each carrying the policy
    label, the target tier, the upgrade tick (sentinels for never_upgrade and the
    ramp policy), the optional ramp tick, the pinned continuous optima, the mean
    total profit, its std-error, and the seed count. Mirrors the expansion-timing
    CSV (F3 = PERMANENT — upgrade is one-shot, expansion template). Empty when the
    automation leg was not run.
    """
    return [
        {
            "policy_name": row.policy_name,
            "target_tier": row.target_tier,
            "upgrade_tick": row.upgrade_tick,
            "ramp_tick": row.ramp_tick,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "fixed_assortment": row.fixed_assortment,
            "fixed_promotion": row.fixed_promotion,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "n_seeds": row.n_seeds,
        }
        for row in result.automation_rows
    ]


def loyalty_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the loyalty-program axis cells as flat dicts for CSV (Phase 1.3).

    One row per ``loyalty_spend`` cell (the constant-spend sweep against the
    default Discounter with the loyalty scenario ON), each carrying the
    ``loyalty_spend``, the pinned continuous optima, the mean profit (already
    netted of the new opex line ``cost_per_unit_spend × loyalty_spend`` inside
    ``apply_accounting``), its std-error, the mean market share and the sample
    count. Empty when the loyalty leg was not run. Mirrors
    ``scm_axis_csv_rows``.
    """
    return [
        {
            "loyalty_spend": row.loyalty_spend,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "fixed_assortment": row.fixed_assortment,
            "fixed_promotion": row.fixed_promotion,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.loyalty_rows
    ]
