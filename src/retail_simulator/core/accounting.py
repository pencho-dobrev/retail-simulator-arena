"""Accounting engine: revenue / COGS / opex / profit / cash (HOT PATH).

Turns the units sold by :func:`demand.resolve_demand` into per-retailer financial
outcomes and the next cash position, plus the loyalty-stock update. This is
**draw #3** of the per-tick order (no accounting noise in Phase 0.0, so no RNG is
drawn here yet — the RNG is intentionally not a parameter).

Economic conventions (Phase 0.0, documented so Wave 3+ can rely on them):

**This module is a PURE LEDGER — it never decides affordability (CLAMP-T2).** It
charges EXACTLY the spend it is handed. Under ``cash_budget_enabled`` the
affordability decision is made one layer up, in ``World.step``, BEFORE demand
resolves: ``core/affordability.py`` computes one clamp per retailer from the
start-of-tick cash and scales the LEVERS, so the value demand reads and the value
billed here are the same number. Two consequences are load-bearing:

* **profit is non-increasing in the start-of-tick cash.** Nothing here reads ``cash``
  to size a cost (only the ``overdraft_rate`` line, which is *larger* the more
  negative cash is, and the final cash identity). The pre-CLAMP-T2 in-accounting
  clamp DID: ``clamp = cash / discretionary`` had no lower bound, so a negative start
  cash flipped the whole discretionary block into INCOME and a broke retailer booked
  last tick's shortfall as this tick's profit.
* **capex is never scaled.** It is charged in full, as gated: the expansion/automation
  masks already checked affordability against the start-of-tick cash, and a one-shot
  purchase is not a quantity you can buy a fraction of ("the seam applies the GATE,
  accounting does the ARITHMETIC").

Loyalty stock updates PER REGION as ``L[R,K] = alpha * loyalty_per_region +
(1 - alpha) * shares`` (broadcast over regions) — a retailer absent from a region
has 0 share there, so its loyalty in that region decays toward 0 (a moat you do not
defend erodes). It is state/obs only in Slice A (loyalty inert in demand).

Pure and framework-free: numpy only. Does not mutate ``state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.demand import DemandResult, per_region_stock
from retail_simulator.core.state import WorldState

# Reference unit price. ``price_index`` is a multiplier on this; defined as a
# module constant (not config) because it is the unit in which the whole money
# system is denominated, not a tunable knob.
BASE_PRICE: float = 1.0


@dataclass(frozen=True)
class AccountingResult:
    """Per-retailer financial outcome of one tick (arrays in agent-index order).

    ``revenue``/``cogs``/``opex``/``marketing``/``capex``/``profit``/``new_cash``
    are ``[K]`` (per retailer, regions summed). ``market_share`` mirrors
    ``DemandResult.shares`` (``[R, K]`` per-region realized share). ``new_loyalty``
    is the post-update PER-REGION loyalty stock (``[R, K]``). ``new_cash`` already
    includes this tick's profit minus capex.
    """

    revenue: npt.NDArray[np.float64]  # [K]
    cogs: npt.NDArray[np.float64]  # [K]
    opex: npt.NDArray[np.float64]  # [K]
    marketing: npt.NDArray[np.float64]  # [K]
    capex: npt.NDArray[np.float64]  # [K]
    profit: npt.NDArray[np.float64]  # [K]
    new_cash: npt.NDArray[np.float64]  # [K]
    market_share: npt.NDArray[np.float64]  # [R, K]
    new_loyalty: npt.NDArray[np.float64]  # [R, K]
    overhead: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cash_negative: npt.NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))


def apply_accounting(
    state: WorldState,
    demand_result: DemandResult,
    config: CoreConfig,
    capex: npt.NDArray[np.float64] | None = None,
    automation_capex: npt.NDArray[np.float64] | None = None,
    wage_bill: npt.NDArray[np.float64] | None = None,
    overhead: npt.NDArray[np.float64] | None = None,
    warehouse_spend: npt.NDArray[np.float64] | None = None,
) -> AccountingResult:
    """ """
    cost_cfg = config.cost
    alpha = config.loyalty.alpha

    units = np.asarray(demand_result.units, dtype=np.float64)  # [R, K]
    price_indices = demand_result.price_indices  # [K]
    cash = np.array([r.cash for r in state.retailers], dtype=np.float64)
    # Per-store opex scales with the number of OPEN stores summed across regions
    # (Phase 0.4): a 2-region agent pays fixed_opex_per_store * 2 per tick.
    stores_total = np.array([sum(r.stores_per_region) for r in state.retailers], dtype=np.float64)
    # Seated per-region loyalty [R, K] (the awareness template, per-region). S6:
    # reuse demand.py's (now vectorized) padded-array builder instead of a bespoke
    # K×R nested comprehension — this was a pre-existing duplicate of the exact same
    # ragged-tuple-to-[R,K] shape ``per_region_stock`` already builds for
    # ``resolve_demand``'s own awareness/loyalty/service_score reads.
    n_regions = len(state.regions)
    loyalty_per_region = per_region_stock(state, "loyalty_per_region", n_regions)  # [R, K]

    # Per-region revenue/COGS/promo summed to a per-retailer [K] total. price_indices
    # broadcasts over the region axis (one price per retailer applied everywhere).
    revenue = (units * price_indices[None, :] * BASE_PRICE).sum(axis=0)  # [K]
    # Phase 1.1: COGS uses the per-retailer ``DemandResult.cogs_fraction`` (NOT
    # ``cost_cfg.cogs_fraction`` directly). The fraction broadcasts over the region axis
    # (one fraction per retailer; the SCM lever is scalar per retailer). At
    # ``cogs_premium = 0`` (default) it equals ``cost_cfg.cogs_fraction`` for every
    # retailer ⇒ the line collapses to the 1.0 form, byte-identical.
    cogs_fraction = np.asarray(demand_result.cogs_fraction, dtype=np.float64)  # [K]
    cogs = (cogs_fraction[None, :] * units * BASE_PRICE).sum(axis=0)  # [K]

    # opex = per-store opex (× stores_total) + assortment cost (ONCE per retailer, a
    # scalar lever applied everywhere — charging it per region would double-count a
    # single breadth decision) + the per-unit-moved promo cost (per region, summed,
    # because units are per-region). All read from the SoTs, never re-decoded.
    assortment = np.asarray(demand_result.assortment, dtype=np.float64)  # [K]
    promotion = np.asarray(demand_result.promotion, dtype=np.float64)  # [K]
    promo_cost = (config.promotion.promo_cost_per_unit * promotion[None, :] * units).sum(
        axis=0
    )  # [K]
    # Phase 1.0 research opex: cost_per_fidelity · research_spend, charged ONCE per
    # retailer (a scalar spend, like the assortment add-on). Every shipped NPC archetype
    # emits research 0 (they are not observers) so a scripted seat contributes 0 — but it
    # is charged like anyone else if one ever does (NFD-1); at the lever default research
    # 0 the whole line is 0 ⇒ opex byte-identical to 0.5.
    # The ONLY economic effect of the research lever — deterministic, no RNG.
    research = np.asarray(demand_result.research, dtype=np.float64)  # [K]
    research_cost = config.research.cost_per_fidelity * research  # [K]
    loyalty_spend_arr = np.asarray(demand_result.loyalty_spend, dtype=np.float64)  # [K]
    loyalty_program_cost = config.loyalty_program.cost_per_unit_spend * loyalty_spend_arr  # [K]
    opex = (
        cost_cfg.fixed_opex_per_store * stores_total
        + config.assortment.assortment_cost_per_unit * assortment
        + promo_cost
        + research_cost
        + loyalty_program_cost
    )
    marketing = np.asarray(demand_result.marketing_spend, dtype=np.float64)  # [K]

    # Phase 1.2: total capex = expansion + automation (both cash items; both default
    # to zeros for backward-compat with all pre-1.2 callers AND for the byte-identity
    # gate at default config — automation_capex is zero because the default
    # ``AutomationConfig.capex_per_tier = (0,0,0)``). Stored on AccountingResult.capex
    # as the SUM (the cash-identity contributor); the seam threads them in already
    # gated by the affordability check (no double-application).
    expansion_capex_arr = (
        np.zeros_like(revenue) if capex is None else np.asarray(capex, dtype=np.float64)
    )
    automation_capex_arr = (
        np.zeros_like(revenue)
        if automation_capex is None
        else np.asarray(automation_capex, dtype=np.float64)
    )
    capex_arr = expansion_capex_arr + automation_capex_arr

    wage_bill_arr = (
        np.zeros_like(revenue) if wage_bill is None else np.asarray(wage_bill, dtype=np.float64)
    )
    overhead_arr = (
        np.zeros_like(revenue) if overhead is None else np.asarray(overhead, dtype=np.float64)
    )
    warehouse_spend_arr = (
        np.zeros_like(revenue)
        if warehouse_spend is None
        else np.asarray(warehouse_spend, dtype=np.float64)
    )
    cash_budget_enabled = config.wage.cash_budget_enabled
    # ``overdraft_rate`` is one more trigger (senior review m1): arming ONLY that knob —
    # no wage bill, no overhead, no warehouse spend, budget gate off — must still reach
    # the Phase-4 branch, or the line silently charges nothing. Default 0.0 keeps the
    # inert path exactly as it was.
    phase4_active = (
        wage_bill is not None
        or overhead is not None
        or cash_budget_enabled
        or warehouse_spend is not None
        or config.wage.overdraft_rate != 0.0
    )

    # M1: the per-retailer negative-cash signal (populated only on the Phase-4 path —
    # an EMPTY array otherwise so the inert path is byte-identical and carries no flag).
    cash_negative = np.zeros(0, dtype=bool)
    if not phase4_active:
        # Pre-Phase-4 path — UNCHANGED float-op order (the byte-identity anchor).
        profit = revenue - cogs - opex - marketing
        new_cash = cash + profit - capex_arr
    else:
        # Phase-4/5 ledger path. NOTHING here is scaled: the affordability clamp
        # (MECHANIC 1) already ran in ``World.step`` BEFORE demand, so ``marketing`` /
        # ``assortment`` / ``research`` / ``loyalty_spend`` / ``wage_bill`` /
        # ``warehouse_spend`` arrive at the value the retailer could actually fund AND
        # the value demand resolved against (CLAMP-T2/T3 — see the module docstring).
        # ``promo_cost`` is a variable cost of SALES (funded per unit sold, hence never
        # clamped, design note Q1) and capex is charged IN FULL, as gated.
        assortment_cost = config.assortment.assortment_cost_per_unit * assortment
        non_clamped_opex = research_cost + loyalty_program_cost
        opex = (
            cost_cfg.fixed_opex_per_store * stores_total
            + assortment_cost
            + promo_cost
            + non_clamped_opex
            + wage_bill_arr
            + warehouse_spend_arr
            + overhead_arr
        )
        #
        # Bound the charged deficit to ``credit_limit`` — the credit the retailer can
        # actually draw on (``core/affordability.py::compute_spend_clamp``) — instead
        # of the whole unbounded ``-cash``. With bankruptcy deferred there is no
        # terminal state, so an unbounded charge would compound geometrically forever
        # once cash goes negative (``cash_{t+1} ~= (1+overdraft_rate)*cash_t``,
        # policy-independent, unbounded) instead of the affine-at-worst dynamic a REAL
        # credit line implies: you are only ever charged interest on what you actually
        # drew, and what you can draw is capped by the limit. This also makes
        # ``core/config.py``'s own ``credit_limit`` docstring true — it already
        # promises the line is "priced every tick it is drawn on", and drawn credit
        # can never exceed ``credit_limit``.
        #
        overdraft_rate = config.wage.overdraft_rate
        if overdraft_rate != 0.0:
            drawn = np.minimum(np.maximum(0.0, -cash), config.wage.credit_limit)
            opex = opex + overdraft_rate * drawn
        profit = revenue - cogs - opex - marketing
        new_cash = cash + profit - capex_arr
        cash_negative = new_cash < 0.0

    market_share = demand_result.shares  # [R, K]
    # Per-region loyalty accrual: alpha * seated + (1-alpha) * this-tick share.
    new_loyalty = alpha * loyalty_per_region + (1.0 - alpha) * market_share  # [R, K]

    return AccountingResult(
        revenue=revenue,
        cogs=cogs,
        opex=opex,
        marketing=marketing,
        capex=capex_arr,
        profit=profit,
        new_cash=new_cash,
        market_share=market_share,
        new_loyalty=new_loyalty,
        overhead=overhead_arr,
        cash_negative=cash_negative,
    )
