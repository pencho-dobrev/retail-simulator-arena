"""Shared DX info-enrichment for the RL adapters (Gym + PettingZoo Parallel).

Adapter layer: numpy + core only (no gymnasium / pettingzoo needed here).
"""

from __future__ import annotations

from typing import Any

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.state import WorldState


def _stockout_rate_from_service_score_change(
    prior: float,
    next_score: float,
    *,
    decay: float,
    lost_penalty: float,
) -> float:
    """ """
    if lost_penalty <= 0.0:
        # Default-config invariant: no stockouts are possible (the lever's economic
        # baseline is service_level=1.0 + cogs_premium=0). Report 0.0 — the truth at
        # the default. A scenario that wants stockout signal sets penalty > 0.
        return 0.0
    raw = next_score - (1.0 - decay) * prior
    # raw = decay * (1 - lost_penalty * sr); solve for sr.
    sr = (1.0 - raw / decay) / lost_penalty
    # Clamp into [0, 1]: the inversion may exceed bounds if the EMA hit the [0, 1]
    # clip in world.step (an upper bound on the truth — the EMA saturated).
    return float(min(max(sr, 0.0), 1.0))


def enrich_agent_info(
    base_info: dict[str, Any],
    state: WorldState,
    seat_index: int,
    episode_step: int,
    *,
    marketing_spend: float,
    assortment: float,
    promotion: float,
    expansion_action: float,
    research_action: float,
    service_level_action: float = 1.0,
    prior_service_score_per_region: tuple[float, ...] | None = None,
    config: CoreConfig | None = None,
    automation_choice: float = 0.0,
    prior_automation_tier: int | None = None,
    loyalty_spend_action: float = 0.0,
    wage_spend_action: float = 0.0,
    warehouse_invest_action: float = 0.0,
) -> dict[str, Any]:
    """ """
    me = state.retailers[seat_index]
    stockpile = sum(region.stockpile for region in state.regions)
    # The seated fidelity is the seam's authority (draw #4 seats it per observer; the
    # reset placeholder is 0.0). Read it; default 0.0 only guards a seat with no seated
    # view (NPC seats are not observers — they never appear here).
    research_fidelity = state.research_fidelity.get(seat_index, 0.0)
    # Phase 1.1: the agent's OWN seated per-region service_score is the seam's authority
    # (own state, EXACT — never noised). Read it; padding with the 1.0 fixed point if
    # a state carries fewer per-region entries than the schema's two slots.
    service_score = me.service_score_per_region
    service_score_region_0 = float(service_score[0]) if len(service_score) > 0 else 1.0
    service_score_region_1 = float(service_score[1]) if len(service_score) > 1 else 1.0

    if prior_service_score_per_region is None or config is None:
        stockout_rate_region_0 = 0.0
        stockout_rate_region_1 = 0.0
    else:
        decay = config.scm.service_score_decay
        lost_penalty = config.scm.lost_sales_share_penalty
        prior_0 = (
            prior_service_score_per_region[0] if len(prior_service_score_per_region) > 0 else 1.0
        )
        prior_1 = (
            prior_service_score_per_region[1] if len(prior_service_score_per_region) > 1 else 1.0
        )
        stockout_rate_region_0 = _stockout_rate_from_service_score_change(
            prior_0, service_score_region_0, decay=decay, lost_penalty=lost_penalty
        )
        stockout_rate_region_1 = _stockout_rate_from_service_score_change(
            prior_1, service_score_region_1, decay=decay, lost_penalty=lost_penalty
        )

    # Phase 1.2 (F3 = PERMANENT): the SEATED tier the seam wrote (the SoT for "the
    # tier that the next tick's demand will read"). Monotone non-decreasing integer;
    # on a no-op upgrade it equals the prior tier, on a valid upgrade it equals the
    # decoded target. At reset / on a state with no step run, this is 0 (the
    # RetailerState default — the no-automation baseline).
    automation_tier_seated = int(me.automation_tier)
    # Phase 1.2: the DECODED TARGET tier the seam wrote AFTER the upgrade gate read
    # it (the reporting-only target, BEFORE the mask resolves monotonicity /
    # affordability — the analog of expansion's ``last_expansion``). At reset this
    # is 0 (the RetailerState default).
    last_automation_tier_action = int(me.last_automation_tier_action)
    # Phase 1.2: the cash debited THIS tick by the upgrade gate — the DIFFERENTIAL
    # ``capex_per_tier[automation_tier] − capex_per_tier[prior_automation_tier]``
    # (the SAME quantity ``apply_accounting`` charged; single-sourced from the SoT
    # — no recompute). When the caller omits ``config`` (e.g. at reset before a
    # step) OR ``prior_automation_tier`` (no prior available), default to 0.0 (the
    # truth — no tick has been billed). Defensive clamps mirror the seam's gate
    # (an out-of-range tier folds to capex 0).
    if config is None or prior_automation_tier is None:
        automation_capex = 0.0
    else:
        capex_per_tier = config.automation.capex_per_tier
        seated_safe = (
            automation_tier_seated if 0 <= automation_tier_seated < len(capex_per_tier) else 0
        )
        prior_safe = (
            prior_automation_tier if 0 <= prior_automation_tier < len(capex_per_tier) else 0
        )
        differential = float(capex_per_tier[seated_safe]) - float(capex_per_tier[prior_safe])
        automation_capex = max(differential, 0.0)

    info = dict(base_info)
    info.update(
        {
            "cash": float(me.cash),
            "revenue": float(me.last_revenue),
            "profit": float(me.last_profit),
            "market_share": float(me.last_market_share),
            "loyalty": float(me.loyalty_stock),
            # Multi-region open-store count across regions (Phase 0.4).
            "stores": int(sum(me.stores_per_region)),
            "tick": int(state.tick),
            "episode_step": int(episode_step),
            "awareness": float(me.awareness),
            "marketing_spend": float(marketing_spend),
            "assortment": float(assortment),
            "promotion": float(promotion),
            "stockpile": float(stockpile),
            "expansion_action": float(expansion_action),
            # Phase 1.0: the decoded research spend this tick (the action taken) +
            # the seated per-observer perception fidelity (the seam's authority).
            "research_action": float(research_action),
            "research_fidelity": float(research_fidelity),
            # Phase 1.1: the decoded SCM action this tick + the per-region seated
            # service_score (the agent's OWN state, EXACT) + the per-region stockout
            # rate recovered from the deterministic EMA recurrence (a pure read).
            "service_level_action": float(service_level_action),
            "service_score_region_0": service_score_region_0,
            "service_score_region_1": service_score_region_1,
            "stockout_rate_region_0": stockout_rate_region_0,
            "stockout_rate_region_1": stockout_rate_region_1,
            "automation_tier_action": float(automation_choice),
            "automation_tier": automation_tier_seated,
            "last_automation_tier_action": last_automation_tier_action,
            "automation_capex": automation_capex,
            "loyalty_spend_action": float(loyalty_spend_action),
            # Phase 5.0: the decoded wage_spend/warehouse_invest fractions this
            # tick (the action taken; 0.0 at reset or when an older caller omits
            # them) + the seated employee_happiness/warehouse_utilization/
            # last_stockout_rate (the agent's OWN state, EXACT — a pure read off
            # ``me``, no inversion needed). The seam consumes both decoded levers
            # when their config knobs are armed; at ``CoreConfig.default()`` (both
            # knobs inert) the three seated reads stay at their byte-identity
            # defaults (1.0 / 0.0 / 0.0) regardless of the decoded action.
            "wage_spend_action": float(wage_spend_action),
            "warehouse_invest_action": float(warehouse_invest_action),
            "employee_happiness": float(me.employee_happiness),
            "warehouse_utilization": float(me.last_warehouse_utilization),
            "last_stockout_rate": float(me.last_stockout_rate),
        }
    )
    # action_mask is part of the public contract; the seam computes the REAL mask
    # (compute_expansion_mask) and carries it in base_info, so it passes through
    # unchanged. setdefault only guards a future seam revision that drops it — it never
    # overwrites the seam's authoritative mask.
    info.setdefault("action_mask", None)
    return info
