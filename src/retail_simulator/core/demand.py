"""Vectorized Huff/MNL demand resolution (HOT PATH).

Phase 0.4: one category, **multiple regions with PRESENCE**, **four customer
segments** (price-sensitive + quality-seeking + convenience + brand-loyal),
pricing, **lagged marketing**, the **contemporaneous assortment** lever, the
**promotion** lever (a DUAL lever: a contemporaneous lift PLUS an intertemporal
consumer-stockpiling debt), and the new **expansion** lever (decoded here as the
SoT; the cash/presence gate lives in ``world.step``). Demand is now a
``[segment, region, retailer]`` tensor: each region is resolved INDEPENDENTLY (its
own pie, its own per-segment softmax over the retailers PRESENT there, its own
seated awareness/stockpile/loyalty/noise). This module answers, per region:

The per-segment attractiveness utility, for segment ``s``, region ``r``,
retailer ``k``::

    u[s, r, k] = -beta_price_s * log(price_k / baseline)
                 + beta_marketing_s * awareness_k_r       # seated per-region (lagged)
                 + beta_assortment_s * assortment_k        # contemporaneous (scalar lever)
                 + beta_promotion_s * promotion_k          # contemporaneous (scalar lever)
                 + beta_loyalty_s * loyalty_k_r            # seated per-region (lagged)
    u[s, r, k] = -inf   where presence[r, k] is False      # absent retailer excluded

Awareness and loyalty are per-retailer-PER-REGION seated stocks (the lagged
template): demand READS the already-seated values (built from PRIOR-tick spend /
share); it does NOT use this tick's spend/share, and it does NOT advance them
(those recurrences live in ``world.step``). **Slice A keeps ``beta_loyalty = 0.0``
for all segments**, so the loyalty term is wired but numerically inert (Slice B
flips the brand-loyal weight on). Assortment AND the promotion lift are
**contemporaneous**: demand READS the DECODED scalar levers of THIS tick (applied
in every region the retailer is present in), NOT state fields.

The promotion lever's INTERTEMPORAL half is the seated per-region consumer
``stockpile`` (the **debt**). Demand READS each region's seated ``stockpile`` to
SUPPRESS that region's TOTAL pie via ``pie *= 1 / (1 + stockpile)`` (bounded in
(0, 1]). It reads ONLY the seated value, NOT this tick's promotion; the stockpile
recurrence (the seat) lives in ``world.step``, so a promo at t suppresses demand
from t+1 — the lag is deliberate, per region.

Phase 1.0 adds ``DemandResult.research`` ``[K]`` — the decoded research-budget SoT.
It is decoded here (so the single-decode hot-path discipline holds) but is
PERCEPTION-ONLY: it does NOT enter the utility tensor, the pie, the shares, or the
noise draw. Demand outputs are byte-identical to 0.5 for ANY research value at the
same seed/other-actions.

Phase 1.2 adds ``DemandResult.automation_tier_action`` ``[K]`` — the decoded TARGET
automation tier per retailer (in ``{0, 1, 2}``). It is decoded here (single-decode
hot-path discipline) but the choice itself does NOT enter the utility tensor, the
pie, the shares, or the noise draw — only the seam consumes it (the monotonicity +
differential-affordability gate in ``world.step``). The COGS-savings READ uses the
PRE-update SEATED ``RetailerState.automation_tier`` (the one-tick lag —
read/charge/seat discipline, awareness/expansion template), indexed into
``savings_per_tier`` as a multiplicative ``cogs_fraction · (1 −
savings_per_tier[automation_tier_prev])`` AFTER the SCM ``cogs_premium``. A
``np.any(automation_tier_prev != 0)``-guarded skip-the-term preserves bit-for-bit
byte-identity at the default config (every retailer is seated at tier 0 at reset,
and the default savings_per_tier=(0,0,0) keeps the multiplier inert anyway —
F3 = PERMANENT).

Pure and framework-free: numpy only. Inputs in, ``DemandResult`` out; no mutation
of ``state``, no hidden RNG, no module-level randomness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from retail_simulator.core.affordability import DecodedLevers
from retail_simulator.core.config import CoreConfig, WarehouseConfig
from retail_simulator.core.encoding import decode_action
from retail_simulator.core.schema import ActionSchema
from retail_simulator.core.state import WorldState


@dataclass(frozen=True)
class DemandResult:
    """ """

    units: npt.NDArray[np.float64]  # [R, K]
    shares: npt.NDArray[np.float64]  # [R, K]
    total_demand: npt.NDArray[np.float64]  # [R]
    price_indices: npt.NDArray[np.float64]  # [K]
    marketing_spend: npt.NDArray[np.float64]  # [K]
    spend_fractions: npt.NDArray[np.float64]  # [K]
    assortment: npt.NDArray[np.float64]  # [K]
    promotion: npt.NDArray[np.float64]  # [K]
    # NEW (Phase 0.4): the decoded expansion choice per retailer this tick (the SoT
    # the seam gates). NPCs/absent retailers → 0 (no-op).
    expansion: npt.NDArray[np.float64]  # [K]
    # NEW (Phase 0.4): the presence mask [R, K] (stores_per_region[r] > 0), used to
    # exclude absent retailers (-inf pre-softmax) and zero their per-region units.
    presence: npt.NDArray[np.bool_]  # [R, K]
    # NEW (Phase 1.0): the decoded research spend per retailer this tick (the SoT for
    # the research opex line + the seam's perception fidelity). PERCEPTION-ONLY — NEVER
    # read by any demand math (utility/pie/shares/noise). NPCs/absent retailers → 0.
    research: npt.NDArray[np.float64]  # [K]
    # NEW (Phase 1.1): the decoded SCM service_level per retailer this tick (the SoT;
    # NPCs/absent retailers → 1.0 — the registry default). Drives the per-region
    # fill-rate cap (units = units_sold post-cap) AND the variable COGS line.
    service_level: npt.NDArray[np.float64]  # [K]
    # NEW (Phase 1.1): per-retailer effective COGS fraction this tick (the SoT
    # accounting reads — NOT ``cost.cogs_fraction`` directly). At ``cogs_premium = 0``
    # equals the base for ANY service_level ⇒ byte-identical to 1.0.
    cogs_fraction: npt.NDArray[np.float64]  # [K]
    # NEW (Phase 1.1): per-region per-retailer stockout rate this tick in [0, 1] —
    # ``max(0, demand_units - units_sold) / max(1, demand_units)`` — the SoT the seater
    # reads to advance the per-region service_score EMA on next state. At
    # ``service_level = 1.0`` it is 0 everywhere (fill-rate cap is a no-op).
    stockout_rate: npt.NDArray[np.float64]  # [R, K]
    # NEW (Phase 1.2): the decoded TARGET automation tier per retailer this tick (the
    # SoT for the seam's monotonicity + differential-affordability gate). One decoded
    # integer per retailer in ``{0, 1, 2}``; NPCs/absent retailers → 0 (the registry
    # default — NPCs are not automation decision-makers in 1.2). The seam reads this,
    # checks (target >= prev) AND (PRE-capex cash >= capex_per_tier[target] −
    # capex_per_tier[prev]); on a valid upgrade it seats the new tier + debits the
    # differential capex, on an invalid target it no-ops (seat unchanged, capex 0).
    automation_tier_action: npt.NDArray[np.int64]  # [K]
    loyalty_spend: npt.NDArray[np.float64]  # [K]
    # NEW (Phase 5.0, B-1): the decoded ``wage_spend`` per retailer this tick (the SoT,
    # in [0, 1]; NPCs/absent retailers → 0.0 — the registry default). The seam
    # (``core/world.py``) consumes this as ``s_t`` in the per-tick paid-wage formula
    # when ``WageConfig.wage_spend_scale`` is armed (B-2); at the default scale (0.0)
    # the seam's read of ``wage_level`` is unaffected — required, not defaulted (a
    # ``field(default_factory=...)`` of the wrong shape here fails deep inside
    # ``world.step``'s per-seat array ops rather than at construction).
    wage_spend: npt.NDArray[np.float64]  # [K]
    # NEW (Phase 5.0, B-1): the decoded ``warehouse_invest`` per retailer this tick (the
    # SoT, in [0, 1]; NPCs/absent retailers → 0.0 — the registry default). The seam
    # consumes this as ``v_t`` in the per-tick capacity-carry recurrence when
    # ``WarehouseConfig.capacity_per_unit_invest``/``cost_per_unit_invest`` are armed
    # (B-2); required, not defaulted — see ``wage_spend`` above.
    warehouse_invest: npt.NDArray[np.float64]  # [K]
    # NEW (Phase 5.0, B-1): per-retailer AGGREGATE lost-sales fraction across regions —
    # the demand-weighted ``Σ_r lost / Σ_r demanded`` (NOT a per-region average of
    # ``stockout_rate``, which would over-weight a low-demand region). 0.0 where a
    # retailer demanded nothing anywhere (the same ``max(.., 1)`` divisor guard as
    # ``stockout_rate``). Computed from the SAME ``demand_units``/``units`` arrays as
    # ``stockout_rate`` — a PURE REPORTING aggregate that feeds NO economic array (only
    # the seam's ``last_stockout_rate`` reporting seat, B-2). Required, not defaulted
    # — see ``wage_spend`` above.
    lost_sales_fraction: npt.NDArray[np.float64]  # [K]
    # NEW (CLAMP-T3): the per-retailer AFFORDABILITY CLAMP in [0, 1] that produced the
    # lever values in this struct — 1.0 = fully funded, 0.0 = could fund nothing. ALL
    # ONES whenever ``config.wage.cash_budget_enabled`` is False (the default), so the
    # field is always well-defined and never gates anything by its absence. The clamp
    # is decided in ``World.step`` BEFORE this function runs (``core/affordability.py``)
    # and the levers above ARRIVE already scaled by it — so this is a REPORTING record
    # of the decision (surfaced as ``info['spend_clamp']``, the DX signal an agent needs
    # to learn that it over-committed), never an instruction to scale anything again.
    # Required, not defaulted — see ``wage_spend`` above.
    spend_clamp: npt.NDArray[np.float64]  # [K]


def _retailer_price_indices(state: WorldState) -> npt.NDArray[np.float64]:
    """Price index per retailer, in agent-index order."""
    return np.array([r.price_index for r in state.retailers], dtype=np.float64)


def _decode_levers(
    state: WorldState,
    joint_action: dict[int, npt.NDArray[np.float32]] | None,
    schema: ActionSchema | None,
) -> DecodedLevers:
    """ """
    prices = _retailer_price_indices(state)
    n = len(state.retailers)
    spend = np.zeros(n, dtype=np.float64)
    assortment = np.zeros(n, dtype=np.float64)
    promotion = np.zeros(n, dtype=np.float64)
    expansion = np.zeros(n, dtype=np.float64)
    research = np.zeros(n, dtype=np.float64)
    # Phase 1.1: service_level defaults to 1.0 (the registry default — the high-service
    # baseline). A retailer that played nothing this tick therefore behaves like a
    # baseline-COGS, perfect-fill-rate supplier, preserving the 0.5/1.0 economics
    # byte-for-byte.
    service_level = np.ones(n, dtype=np.float64)
    automation = np.zeros(n, dtype=np.int64)
    loyalty_spend = np.zeros(n, dtype=np.float64)
    # Phase 5.0: wage_spend/warehouse_invest default to 0.0 (the registry defaults —
    # the byte-identity anchors). A retailer that played nothing therefore pays no wage
    # premium and invests nothing, matching the pre-Phase-5.0 economics for ANY config.
    wage_spend = np.zeros(n, dtype=np.float64)
    warehouse_invest = np.zeros(n, dtype=np.float64)
    # NFD-1: ONE per-retailer resolution replaces the old "early-return when there is no
    # joint action, else loop the joint action" pair, because a seat's action can now
    # arrive from either side. The precedence is documented above; a seat matching
    # neither source keeps the registry defaults seeded right here (``continue``), which
    # is bit-for-bit the value the old early return produced.
    for idx, retailer in enumerate(state.retailers):
        flat: npt.NDArray[np.float32] | tuple[float, ...] | None = (
            joint_action.get(idx) if joint_action else None
        )
        if flat is None:
            # A LEARNING seat's pending action is always None; only a scripted seat
            # carries one, and only from the tick after it first acted.
            if not (retailer.is_npc and retailer.pending_action is not None):
                continue
            flat = retailer.pending_action
        decoded = decode_action(flat, schema)
        prices[idx] = decoded.levers["price_index"]
        # ``marketing``/``assortment``/``promotion``/``expansion``/``research``/
        # ``service_level``/``automation``/``loyalty_spend`` may be absent if the
        # lever is disabled in the schema; fall back to 0.0 (or 1.0 for service_level
        # — the byte-identity anchor) so a narrower schema still resolves cleanly.
        spend[idx] = decoded.levers.get("marketing", 0.0)
        assortment[idx] = decoded.levers.get("assortment", 0.0)
        promotion[idx] = decoded.levers.get("promotion", 0.0)
        expansion[idx] = decoded.levers.get("expansion", 0.0)
        research[idx] = decoded.levers.get("research", 0.0)
        service_level[idx] = decoded.levers.get("service_level", 1.0)
        # Phase 1.2: automation default is 0 (the registry default — no investment).
        automation[idx] = int(decoded.levers.get("automation", 0.0))
        # Phase 1.3: loyalty_spend default is 0.0 (the registry default — no program;
        # the byte-identity anchor).
        loyalty_spend[idx] = decoded.levers.get("loyalty_spend", 0.0)
        # Phase 5.0: wage_spend/warehouse_invest default to 0.0 (the registry defaults
        # — the byte-identity anchors).
        wage_spend[idx] = decoded.levers.get("wage_spend", 0.0)
        warehouse_invest[idx] = decoded.levers.get("warehouse_invest", 0.0)
    return DecodedLevers(
        price_indices=prices,
        spend_fraction=spend,
        assortment=assortment,
        promotion=promotion,
        expansion=expansion,
        research=research,
        service_level=service_level,
        automation_tier=automation,
        loyalty_spend=loyalty_spend,
        wage_spend=wage_spend,
        warehouse_invest=warehouse_invest,
    )


def _present_others_mean(
    values: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """ """
    pres = presence.astype(np.float64)  # [R, K]
    masked = values[None, :] * pres  # [R, K], absent -> 0
    sum_present = masked.sum(axis=1, keepdims=True)  # [R, 1]
    count_present = pres.sum(axis=1, keepdims=True)  # [R, 1]
    # sum-minus-self / count-minus-self: arithmetically drop the retailer itself from
    # both the numerator and the denominator (so it is never its own reference).
    others_sum = sum_present - masked  # [R, K]
    others_count = count_present - pres  # [R, K]
    # present-with-rivals -> the others' mean; sole-present (or absent) -> own value.
    # ``maximum(..., 1)`` only guards the divide; ``where`` selects the fallback so a
    # 0-count divide is never realized.
    return np.where(
        others_count > 0.0,
        others_sum / np.maximum(others_count, 1.0),
        values[None, :],
    )


def _reference_prices(
    price_indices: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """ """
    return _present_others_mean(price_indices, presence)


def _reference_promotion(
    promotion: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """ """
    return _present_others_mean(promotion, presence)


def _segment_region_shares(
    price_indices: npt.NDArray[np.float64],
    awareness_per_region: npt.NDArray[np.float64],
    loyalty_per_region: npt.NDArray[np.float64],
    assortment: npt.NDArray[np.float64],
    promotion: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
    beta_prices: npt.NDArray[np.float64],
    beta_marketings: npt.NDArray[np.float64],
    beta_assortments: npt.NDArray[np.float64],
    beta_promotions: npt.NDArray[np.float64],
    beta_loyalties: npt.NDArray[np.float64],
    baseline: float,
    beta_references: npt.NDArray[np.float64] | None = None,
    service_score_per_region: npt.NDArray[np.float64] | None = None,
    beta_services: npt.NDArray[np.float64] | None = None,
    beta_programs: npt.NDArray[np.float64] | None = None,
    loyalty_spend: npt.NDArray[np.float64] | None = None,
    beta_under_reference_aversions: npt.NDArray[np.float64] | None = None,
    promotion_contest_weight: float = 0.0,
) -> npt.NDArray[np.float64]:
    """ """
    log_price = np.log(price_indices / baseline)  # [K]
    # Build [S, R, K] by broadcasting: betas on [S, 1, 1]; per-region seated stocks
    # on [1, R, K]; per-retailer scalar levers on [1, 1, K]; log-price on [1, 1, K].
    bp = beta_prices[:, None, None]
    bm = beta_marketings[:, None, None]
    ba = beta_assortments[:, None, None]
    bpr = beta_promotions[:, None, None]
    bl = beta_loyalties[:, None, None]
    if promotion_contest_weight == 0.0:
        promo_term = promotion[None, None, :]
    else:
        promo_ref = _reference_promotion(promotion, presence)  # [R, K]
        promo_effective = promotion[None, :] - promotion_contest_weight * promo_ref  # [R, K]
        promo_term = promo_effective[None, :, :]
    utility = (
        -bp * log_price[None, None, :]
        + bm * awareness_per_region[None, :, :]
        + ba * assortment[None, None, :]
        + bpr * promo_term
        + bl * loyalty_per_region[None, :, :]
    )  # [S, R, K]
    if beta_references is not None and bool(np.any(beta_references != 0.0)):
        bref = beta_references[:, None, None]  # [S, 1, 1]
        ref = _reference_prices(price_indices, presence)  # [R, K]
        penalty = np.maximum(0.0, np.log(price_indices[None, None, :] / ref[None, :, :]))
        utility = utility - bref * penalty  # asymmetric: only over-pricing is punished
    if beta_under_reference_aversions is not None and bool(
        np.any(beta_under_reference_aversions != 0.0)
    ):
        bunder = beta_under_reference_aversions[:, None, None]  # [S, 1, 1]
        ref_under = _reference_prices(price_indices, presence)  # [R, K]
        under_penalty = np.maximum(
            0.0, np.log(ref_under[None, :, :] / price_indices[None, None, :])
        )
        utility = utility - bunder * under_penalty  # symmetric: only under-pricing punished
    if (
        beta_services is not None
        and service_score_per_region is not None
        and bool(np.any(beta_services != 0.0))
    ):
        bsvc = beta_services[:, None, None]  # [S, 1, 1]
        utility = utility + bsvc * service_score_per_region[None, :, :]  # [S, R, K]
    if (
        beta_programs is not None
        and loyalty_spend is not None
        and bool(np.any(beta_programs != 0.0))
    ):
        bprog = beta_programs[:, None, None]  # [S, 1, 1]
        # same-tick boost: amplifies EXISTING per-region loyalty stock for the
        # brand-loyal segment in proportion to the per-tick spend. READS the
        # PRE-update seated loyalty_per_region (the one-tick lag; awareness
        # template). recipe (iv): the gradient is THIS tick's softmax → THIS
        # tick's share → THIS tick's profit. EMA accrual remains a side effect.
        utility = utility + bprog * (loyalty_spend[None, None, :] * loyalty_per_region[None, :, :])
    utility = np.where(presence[None, :, :], utility, -np.inf)
    shifted = utility - np.max(utility, axis=2, keepdims=True)
    weights = np.exp(shifted)  # absent -> exp(-inf - max) = 0
    return weights / np.sum(weights, axis=2, keepdims=True)


def _presence_matrix(state: WorldState, n_regions: int) -> npt.NDArray[np.bool_]:
    """ """
    n = len(state.retailers)
    presence = np.zeros((n_regions, n), dtype=bool)
    for k, retailer in enumerate(state.retailers):
        spr = retailer.stores_per_region
        length = min(len(spr), n_regions)
        if length:
            presence[:length, k] = np.asarray(spr[:length]) > 0
    return presence


def per_region_stock(state: WorldState, attr: str, n_regions: int) -> npt.NDArray[np.float64]:
    """Build a ``[R, K]`` seated-stock matrix from a per-region retailer attribute.

    ``attr`` names any ``tuple[..., ...]``-shaped per-region ``RetailerState`` field
    — ``"awareness_per_region"``, ``"loyalty_per_region"``, or
    ``"service_score_per_region"`` (all ``tuple[float, ...]``); also reused, since
    S6, for the int-valued ``"stores_per_region"`` by ``resolve_demand``'s warehouse
    block and by ``core/accounting.py``'s per-region loyalty read (exact under the
    output array's float64 cast for realistic store counts). A retailer's per-region
    tuple shorter than ``n_regions`` reads 0.0 for missing regions.

    Senior-review m1 (2026-09-05): public (no leading underscore) specifically
    because ``core/accounting.py`` calls it across the module boundary — a private
    helper reused by another module should be promoted, not reached into.

    S6 (24-region hot path): one slice ASSIGNMENT per retailer replaces the old
    inner ``for r in range(n_regions)`` Python loop (the K×R double loop). Pure data
    movement (no arithmetic), so bit-identical to the old per-element copy.
    """
    n = len(state.retailers)
    out = np.zeros((n_regions, n), dtype=np.float64)
    for k, retailer in enumerate(state.retailers):
        per_region: tuple[float, ...] | tuple[int, ...] = getattr(retailer, attr)
        length = min(len(per_region), n_regions)
        if length:
            out[:length, k] = per_region[:length]
    return out


def _footprint_totals(state: WorldState, n_regions: int) -> npt.NDArray[np.float64]:
    """ """
    n = len(state.retailers)
    out = np.zeros(n, dtype=np.float64)
    for k, retailer in enumerate(state.retailers):
        spr = retailer.stores_per_region
        out[k] = sum(spr[r] for r in range(n_regions) if r < len(spr))
    return out


def _mature_store_equivalents(
    state: WorldState, n_regions: int, ramp_ticks: int
) -> npt.NDArray[np.float64]:
    """ """
    n = len(state.retailers)
    out = np.zeros(n, dtype=np.float64)
    for k, retailer in enumerate(state.retailers):
        spr = retailer.stores_per_region
        ages = retailer.store_age_per_region
        total = 0.0
        for r in range(n_regions):
            if r >= len(spr) or spr[r] <= 0:
                continue
            age_r = ages[r] if r < len(ages) else 0
            total += spr[r] * min(1.0, age_r / ramp_ticks)
        out[k] = total
    return out


def _capacity_footprint(
    state: WorldState, n_regions: int, warehouse_cfg: WarehouseConfig
) -> npt.NDArray[np.float64]:
    """ """
    if warehouse_cfg.ramp_ticks > 0:
        return _mature_store_equivalents(state, n_regions, warehouse_cfg.ramp_ticks)
    return _footprint_totals(state, n_regions)


def effective_warehouse_capacity(
    config: CoreConfig,
    warehouse_capacity: npt.NDArray[np.float64],
    stores_total: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """ """
    provided = config.warehouse.provided_capacity_per_store
    if provided == 0.0:
        return warehouse_capacity
    return warehouse_capacity + provided * stores_total


def _apply_share_ceiling(
    shares: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
    cap: float,
) -> npt.NDArray[np.float64]:
    """ """
    if cap >= 1.0:
        return shares
    n_retailers = shares.shape[1]
    s = shares.copy()
    # Present-and-never-yet-capped mask. Absent retailers start False here (never
    # eligible to be capped OR to receive redistributed excess); once a retailer
    # is capped this round it is cleared here PERMANENTLY for every later round.
    uncapped = presence.copy()
    for _ in range(n_retailers):
        over = uncapped & (s > cap)
        if not bool(np.any(over)):
            break  # every region has converged — no present uncapped retailer exceeds cap
        excess = np.where(over, s - cap, 0.0).sum(axis=1, keepdims=True)  # [R, 1]
        s = np.where(over, cap, s)
        uncapped = uncapped & ~over  # capping is permanent
        # Redistribute each region's freed excess over its STILL-uncapped present
        # retailers, weighted by their current share. A region with nothing left
        # to redistribute to (``recipient_sum == 0`` — every present retailer is
        # now capped) leaves that round's excess UNSERVED: ``add`` is 0 everywhere
        # in that region, so ``s`` there stays at the just-clamped (sum < 1.0) values.
        recipient_sum = np.where(uncapped, s, 0.0).sum(axis=1, keepdims=True)  # [R, 1]
        safe_recipient_sum = np.where(recipient_sum > 0.0, recipient_sum, 1.0)
        add = np.where(
            uncapped & (recipient_sum > 0.0),
            excess * s / safe_recipient_sum,
            0.0,
        )
        s = s + add
    return s


def resolve_demand(
    state: WorldState,
    joint_action: dict[int, npt.NDArray[np.float32]] | None,
    rng: np.random.Generator,
    config: CoreConfig,
    schema: ActionSchema | None = None,
    levers: DecodedLevers | None = None,
    spend_clamp: npt.NDArray[np.float64] | None = None,
) -> DemandResult:
    """ """
    demand_cfg = config.demand
    pricing_cfg = config.pricing
    marketing_cfg = config.marketing
    promotion_cfg = config.promotion
    cost_cfg = config.cost
    scm_cfg = config.scm
    baseline = pricing_cfg.baseline_price_index

    # CLAMP-T1: _decode_levers now returns a DecodedLevers dataclass instead of a bare
    # 11-tuple (core/affordability.py); unpack into the SAME local names used below so
    # the rest of this function's arithmetic is untouched (a pure access-mechanism
    # change, not a decode change). CLAMP-T3: the caller may hand in the levers it
    # already decoded (and possibly clamp-scaled) — the decode is pure and RNG-free, so
    # who runs it is invisible to the trajectory.
    decoded_levers = _decode_levers(state, joint_action, schema) if levers is None else levers
    price_indices = decoded_levers.price_indices
    spend_fraction = decoded_levers.spend_fraction
    assortment = decoded_levers.assortment
    promotion = decoded_levers.promotion
    expansion = decoded_levers.expansion
    research = decoded_levers.research
    service_level = decoded_levers.service_level
    automation_tier = decoded_levers.automation_tier
    loyalty_spend = decoded_levers.loyalty_spend
    wage_spend = decoded_levers.wage_spend
    warehouse_invest = decoded_levers.warehouse_invest

    regions = state.regions
    n_regions = len(regions)

    presence = _presence_matrix(state, n_regions)
    region_has_presence = presence.any(axis=1)  # [R]

    # READ the seated per-region awareness + loyalty (lagged; no RNG draw).
    awareness_per_region = per_region_stock(state, "awareness_per_region", n_regions)
    loyalty_per_region = per_region_stock(state, "loyalty_per_region", n_regions)
    # Phase 1.1: READ the seated per-region service_score (the new liability; the
    # awareness template with reset value 1.0). The seater lives in ``world.step``; demand
    # only READS it pre-update (the read/charge/seat discipline).
    service_score_per_region = per_region_stock(state, "service_score_per_region", n_regions)

    shares = np.zeros((n_regions, len(state.retailers)), dtype=np.float64)
    suppressed_pie = np.zeros(n_regions, dtype=np.float64)

    region_keys_union: set[str] = set()
    for region in regions:
        region_keys_union.update(region.segment_mix.keys())
    segment_order = {key: position for position, key in enumerate(state.segments)}
    segment_keys: list[str] = sorted(region_keys_union, key=lambda key: segment_order[key])
    beta_prices = np.array(
        [state.segments[key].beta_price for key in segment_keys], dtype=np.float64
    )
    beta_marketings = np.array(
        [state.segments[key].beta_marketing for key in segment_keys], dtype=np.float64
    )
    beta_assortments = np.array(
        [state.segments[key].beta_assortment for key in segment_keys], dtype=np.float64
    )
    beta_promotions = np.array(
        [state.segments[key].beta_promotion for key in segment_keys], dtype=np.float64
    )
    beta_loyalties = np.array(
        [state.segments[key].beta_loyalty for key in segment_keys], dtype=np.float64
    )
    beta_references = np.array(
        [state.segments[key].beta_reference for key in segment_keys], dtype=np.float64
    )
    beta_services = np.array(
        [state.segments[key].beta_service for key in segment_keys], dtype=np.float64
    )
    beta_programs = np.array(
        [state.segments[key].beta_program for key in segment_keys], dtype=np.float64
    )
    beta_under_reference_aversions = np.array(
        [state.segments[key].beta_under_reference_aversion for key in segment_keys],
        dtype=np.float64,
    )
    # Maps a segment key to its row position in ``segment_keys`` / the batched
    # ``_segment_region_shares`` call's segment axis below. Used only by the
    # per-region aggregation loop (not a tensor op; O(S) dict, S <= 6).
    canonical_index = {key: position for position, key in enumerate(segment_keys)}

    covered = np.flatnonzero(region_has_presence)
    uncovered = np.flatnonzero(~region_has_presence)

    for region_idx in uncovered:
        region = regions[int(region_idx)]
        avg_price = baseline
        base_demand = (
            region.base_regional_demand * (avg_price / baseline) ** demand_cfg.price_elasticity
        )
        suppressed_pie[region_idx] = base_demand * (1.0 / (1.0 + region.stockpile))

    if covered.size:
        # S6: ONE batched [S, R_covered, K] call replaces what used to be up to R
        # separate [S, 1, K] calls (one per covered region). This is safe because
        # _segment_region_shares has NO reduction across the region axis anywhere in
        # its body — every reduction (the softmax max/sum) is over the RETAILER axis,
        # per (segment, region) independently — so the value at [:, i, :] of the
        # batched call is bit-identical to calling the function on region
        # covered[i] alone with a 1-row slice. Verified on every ``.npz`` pin plus a
        # direct equality test against the pre-S6 reference loop
        # (tests/core/test_demand_vectorization.py).
        seg_region_shares_all = _segment_region_shares(
            price_indices,
            awareness_per_region[covered, :],
            loyalty_per_region[covered, :],
            assortment,
            promotion,
            presence[covered, :],
            beta_prices,
            beta_marketings,
            beta_assortments,
            beta_promotions,
            beta_loyalties,
            baseline,
            beta_references,
            service_score_per_region[covered, :],
            beta_services,
            beta_programs=beta_programs,
            loyalty_spend=loyalty_spend,
            beta_under_reference_aversions=beta_under_reference_aversions,
            promotion_contest_weight=promotion_cfg.promotion_contest_weight,
        )  # [S, R_covered, K]

        for i, region_idx in enumerate(covered):
            region = regions[int(region_idx)]
            # Aggregate per-segment shares by this region's population weights ->
            # [K]. Kept as an EXPLICIT per-region matvec (not a batched einsum): the
            # design note flags this segment-axis reduction as the one op whose
            # result could depend on summation order, and an explicit loop of small
            # per-region BLAS matvecs is the form proven bit-identical to the pre-S6
            # per-region loop (see the direct equality test cited above).
            #
            # Senior-review M1 fix (2026-09-05): summed in THIS REGION's OWN
            # segment_mix key order, not the shared canonical order — floating-point
            # addition is not associative, so a dot product's bit-exact result
            # depends on the order its terms are summed in, and the reference loop
            # sums each region in ITS OWN native order (it never hoists a shared
            # order). ``native_rows`` gathers this region's segment rows out of the
            # canonical-order ``seg_region_shares_all`` in NATIVE order first, so the
            # dot product below sums the exact same terms in the exact same
            # sequence the reference loop would. For a region whose native order
            # already MATCHES the canonical order (every shipped config) ``native_
            # rows`` is the identity permutation, so this is byte-identical to
            # indexing the canonical order directly. A region need not carry every
            # segment in ``segment_keys``: ``native_weight`` only ever contains the
            # keys THIS region actually has (unlike the deleted ``seg_weight_matrix``,
            # there is no ``.get(key, 0.0)`` here — a segment this region omits is
            # simply never a term in ITS sum, exactly as the reference loop would
            # never build a beta/weight entry for a key ``region.segment_mix`` lacks).
            native_keys = list(region.segment_mix.keys())
            native_weight = np.array(
                [region.segment_mix[key] for key in native_keys], dtype=np.float64
            )
            native_rows = [canonical_index[key] for key in native_keys]
            shares[region_idx, :] = native_weight @ seg_region_shares_all[native_rows, i, :]

            # Pie size: constant-elasticity response to the PRESENT-only average price
            # level in this region (a region the agent has not entered sees only the
            # NPC's price). Higher prices shrink total demand (elasticity negative).
            present_prices = price_indices[presence[region_idx, :]]
            avg_price = float(np.mean(present_prices))
            base_demand = (
                region.base_regional_demand * (avg_price / baseline) ** demand_cfg.price_elasticity
            )
            suppressed_pie[region_idx] = base_demand * (1.0 / (1.0 + region.stockpile))

    if demand_cfg.max_share_per_region < 1.0:
        shares = _apply_share_ceiling(shares, presence, demand_cfg.max_share_per_region)

    noise = rng.lognormal(mean=0.0, sigma=demand_cfg.noise_sigma, size=n_regions)
    realized_total = suppressed_pie * noise  # [R]

    # units[r] = realized_total[r] * shares[r] * presence[r] (absent retailers have
    # 0 share already, so the presence multiply is belt-and-suspenders + dtype).
    demand_units = realized_total[:, None] * shares * presence.astype(np.float64)  # [R, K]

    fill_rates = np.array(
        [scm_cfg.fill_rate(float(s)) for s in service_level], dtype=np.float64
    )  # [K]
    available_units = realized_total[:, None] * fill_rates[None, :]  # [R, K]
    units = np.minimum(demand_units, available_units)  # [R, K] — the realized sales

    supplier_cfg = config.supplier
    if supplier_cfg.supplier_contention_enabled and np.isfinite(supplier_cfg.supplier_capacity):
        capacity = supplier_cfg.supplier_capacity
        pres_f = presence.astype(np.float64)  # [R, K]
        # Relationship priority weight per retailer: service_level ** sharpness. Masked to
        # the retailers PRESENT in each region (absent retailers contend for nothing).
        priority = service_level**supplier_cfg.preference_sharpness  # [K]
        weights = priority[None, :] * pres_f  # [R, K] — absent → 0
        weight_sum = weights.sum(axis=1, keepdims=True)  # [R, 1]
        present_count = pres_f.sum(axis=1, keepdims=True)  # [R, 1]
        no_preference = (weight_sum <= 0.0) & (present_count > 0.0)  # [R, 1]
        weights = np.where(no_preference, pres_f, weights)
        weight_sum = np.where(no_preference, present_count, weight_sum)
        # Each present retailer's slice of the scarce capacity (normalized over present
        # rivals). ``weight_sum`` is a FLOAT SUM (Σ service_level**sharpness over the
        # present retailers, or ``present_count`` in the no-preference case above) that
        # can lie in ``(0.0, 1.0)`` — e.g. a single present retailer at service_level
        # 0.5 has weight_sum == 0.5 — so a ``np.maximum(sum, 1.0)`` floor would silently
        # SCALE DOWN every such quotient instead of merely guarding the 0/0 case.
        # ``weight_sum == 0`` now arises ONLY from case (1) above (no present
        # retailer) — case (2) is resolved above, before this guard ever sees a zero.
        # ``safe_priority_weight_sum`` substitutes a placeholder 1.0 ONLY where the
        # true sum is 0 (``where`` selects 0 share there anyway), leaving every genuine
        # positive sum — including one below 1 — exact. The cap ``share * capacity`` is
        # the supplier-allocation ceiling.
        safe_priority_weight_sum = np.where(weight_sum > 0.0, weight_sum, 1.0)
        capacity_share = np.where(
            weight_sum > 0.0, weights / safe_priority_weight_sum, 0.0
        )  # [R, K]
        supplier_cap = capacity_share * capacity  # [R, K]
        # Contention BITES only in regions where total DEMANDED units exceed capacity;
        # elsewhere the per-region cap is left wide open (no cut). ``where`` keeps the
        # uncontended regions on the EXACT fill-rate-capped ``units`` (no spurious min).
        regional_demand = demand_units.sum(axis=1, keepdims=True)  # [R, 1]
        contended = regional_demand > capacity  # [R, 1]
        units = np.where(contended, np.minimum(units, supplier_cap), units)  # [R, K]

    #
    #
    warehouse_cfg = config.warehouse
    warehouse_capacity = np.array(
        [retailer.warehouse_capacity for retailer in state.retailers], dtype=np.float64
    )  # [K]
    if warehouse_cfg.throughput_enabled and bool(np.any(np.isfinite(warehouse_capacity))):
        units_pre_wh = units
        # S6: reuse the (now vectorized) per_region_stock padded-array builder
        # instead of a bespoke K×R nested comprehension — identical semantics (a
        # retailer's tuple shorter than n_regions reads 0 for the missing regions),
        # one fewer duplicate implementation of the same ragged-tuple-to-[R,K] shape.
        stores = per_region_stock(state, "stores_per_region", n_regions)  # [R, K]
        # Footprint share: stores[r,k] / stores_total_k. GUARD stores_total == 0 → 0 (the
        # _reference_prices divide-guard idiom — a retailer with no stores anywhere gets a
        # 0 region share, never realized since it is absent / serves 0 units anyway). The
        # per-retailer capacity is split across the regions it operates in by store count.
        stores_total = stores.sum(axis=0, keepdims=True)  # [1, K]
        region_share = np.where(
            stores_total > 0.0, stores / np.maximum(stores_total, 1.0), 0.0
        )  # [R, K]
        stores_total_effective = _capacity_footprint(state, n_regions, warehouse_cfg)
        effective_capacity = effective_warehouse_capacity(
            config, warehouse_capacity, stores_total_effective
        )  # [K]
        # The cap is ``capacity_k * region_share[r,k]``. GUARD against the ``inf * 0``
        # NaN (an inf-capacity retailer ABSENT from region r has region_share 0; numpy's
        # eager ``inf * 0`` is NaN, which would then poison ``minimum``). Compute the
        # product ONLY where the share is positive (the retailer operates in r); a 0
        # region_share gets a 0 cap (the retailer serves 0 units there anyway — absent or
        # no-stores). This keeps the array NaN-free for finite AND inf capacities.
        with np.errstate(invalid="ignore"):
            # ``inf * 0`` (an inf-capacity retailer with share 0) is NaN with an
            # invalid-value warning; it is masked to 0.0 by the ``where`` (share == 0 ⇒
            # False leg), so the NaN is never realized — silence the provably-masked warn.
            warehouse_cap = np.where(
                region_share > 0.0, effective_capacity[None, :] * region_share, 0.0
            )  # [R, K]
        units_wh = np.minimum(units_pre_wh, warehouse_cap)  # [R, K] — capped served demand
        # SPILL (deterministic, single-pass, rival-coupled). Per region: the units a
        # retailer LOST to its own warehouse cap are redistributed to PRESENT rivals in
        # proportion to their SPARE warehouse capacity. ``lost`` is the warehouse-stage
        # residual ONLY (R1). ``total_lost[r]`` is the region's spillable pool;
        # ``spare[r,k]`` is each retailer's room under its own cap. SINGLE-PASS: no
        # cascade (a recipient's received spill does NOT itself re-spill if it overflows
        # — the residual when total spare < total lost is simply lost).
        lost = np.maximum(0.0, units_pre_wh - units_wh)  # [R, K], >= 0
        total_lost = lost.sum(axis=1, keepdims=True)  # [R, 1]
        spare = np.maximum(0.0, warehouse_cap - units_wh)  # [R, K], >= 0
        weight_spare = np.where(np.isfinite(spare), spare, total_lost)  # [R, K], finite
        weight_sum = weight_spare.sum(axis=1, keepdims=True)  # [R, 1]
        safe_spare_weight_sum = np.where(weight_sum > 0.0, weight_sum, 1.0)
        spill_in_raw = total_lost * np.where(
            weight_sum > 0.0, weight_spare / safe_spare_weight_sum, 0.0
        )  # [R, K]
        # Clamp each recipient's spill to its OWN (true) spare so no recipient's units
        # exceed its warehouse cap (the R1 invariant). When ``total_lost`` exceeds the
        # total spare the clamp caps each recipient and the un-absorbed residual is LOST to
        # market (the single-pass, no-cascade semantics — a recipient that overflows does
        # NOT re-spill). ``min(spill, inf) = spill`` for an inf-spare recipient.
        spill_in = np.minimum(spill_in_raw, spare)  # [R, K]
        units = units_wh + spill_in  # [R, K] — the warehouse-final served demand

    stockout_rate = np.maximum(0.0, demand_units - units) / np.maximum(demand_units, 1.0)

    # Phase 5.0, B-1: per-retailer AGGREGATE lost-sales fraction across regions — the
    # demand-weighted ``Σ_r lost / Σ_r demanded`` (NOT a per-region average of
    # ``stockout_rate`` above, which would over-weight a low-demand region). Built
    # from the SAME ``demand_units``/``units`` arrays as ``stockout_rate`` — a pure
    # reporting aggregate that feeds NO economic array (only the seam's
    # ``last_stockout_rate`` reporting seat, B-2). 0.0 where a retailer demanded
    # nothing anywhere (the same ``max(.., 1)`` divisor guard).
    lost_units_per_retailer = np.maximum(0.0, demand_units - units).sum(axis=0)  # [K]
    demanded_per_retailer = demand_units.sum(axis=0)  # [K]
    lost_sales_fraction = lost_units_per_retailer / np.maximum(demanded_per_retailer, 1.0)  # [K]

    cogs_fraction = cost_cfg.cogs_fraction * (1.0 + scm_cfg.cogs_premium * service_level)

    automation_cfg = config.automation
    savings_per_tier_arr = np.asarray(
        automation_cfg.savings_per_tier, dtype=np.float64
    )  # [n_tiers]
    automation_tier_prev = np.array(
        [retailer.automation_tier for retailer in state.retailers], dtype=np.int64
    )  # [K]
    if bool(np.any(automation_tier_prev != 0)):
        cogs_fraction = cogs_fraction * (1.0 - savings_per_tier_arr[automation_tier_prev])

    marketing_exponent = marketing_cfg.marketing_cost_exponent
    if marketing_exponent == 1.0:
        marketing_spend = spend_fraction * marketing_cfg.cost_per_unit_spend
    else:
        marketing_spend = spend_fraction**marketing_exponent * marketing_cfg.cost_per_unit_spend
    return DemandResult(
        units=units,
        shares=shares,
        total_demand=realized_total,
        price_indices=price_indices,
        marketing_spend=marketing_spend,
        spend_fractions=spend_fraction,
        assortment=assortment,
        promotion=promotion,
        expansion=expansion,
        presence=presence,
        # Phase 1.0: the decoded research SoT, carried verbatim. It never touched any
        # demand math above (utility/pie/shares/noise) — only accounting (opex) + the
        # seam (perception fidelity) consume it.
        research=research,
        # Phase 1.1: the decoded service_level SoT + the per-retailer effective COGS
        # fraction (accounting reads it) + the per-region stockout rate (the seater
        # reads it). At the default config (cogs_premium=0, beta_service=0) every output
        # in this struct that depends on service_level is byte-identical to 1.0.
        service_level=service_level,
        cogs_fraction=cogs_fraction,
        stockout_rate=stockout_rate,
        # Phase 1.2 (F3 = PERMANENT): the decoded TARGET automation tier SoT, carried
        # verbatim. It never touched any demand math (utility/pie/shares/noise — the
        # choice surface is UNAFFECTED, only the COGS multiplier was) — only the seam
        # (the monotonicity + differential-affordability gate, the seat advance) and
        # the seater's reporting consume it.
        automation_tier_action=automation_tier,
        # Phase 1.3: the decoded ``loyalty_spend`` SoT, carried verbatim. The single
        # source of truth for the brand-loyal same-tick boost (applied above in
        # ``_segment_region_shares``, gated by ``beta_program``) AND the per-tick opex
        # line (applied downstream in ``apply_accounting``, gated by
        # ``cost_per_unit_spend``). At default config (``beta_program = 0`` everywhere
        # AND ``cost_per_unit_spend = 0``) BOTH consumers are inert ⇒ the byte-identity
        # contract holds across ALL decoded values.
        loyalty_spend=loyalty_spend,
        wage_spend=wage_spend,
        warehouse_invest=warehouse_invest,
        lost_sales_fraction=lost_sales_fraction,
        # CLAMP-T3: the affordability clamp that produced the lever values above (all
        # ones when the caller ran no clamp — the default and every pre-T3 caller).
        # Reporting only; the scaling already happened before this function was called.
        spend_clamp=(
            np.ones(len(state.retailers), dtype=np.float64) if spend_clamp is None else spend_clamp
        ),
    )
