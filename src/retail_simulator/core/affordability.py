"""Affordability: the single pure owner of "can this retailer afford this spend."

**THE ORDER INSIDE ONE TICK (CLAMP-T3 — the contract this module exists to make
possible).** ``World.step``, under ``config.wage.cash_budget_enabled``, runs::

    1. decode every seat's action ONCE       -> DecodedLevers   (no RNG)
       (a learning seat's from ``joint_action``; an NPC seat's from the
       ``pending_action`` its policy decided last tick — NFD-1. Every seat is
       clamped by the same rule below, whichever way its action arrived.)
    2. capex gates, FIRST and IN FULL        -> gated_expansion_capex
                                                gated_automation_capex
    3. intended_discretionary_spend(levers, wage_bill, config)     (the money)
    4. compute_spend_clamp(cash, intent, committed_capex, credit_limit)  -> clamp[K] in [0, 1]
    5. scale_discretionary_levers(levers, clamp)           -> the SCALED levers
    6. resolve_demand(..., levers=scaled, spend_clamp=clamp)       (draw #1)
    7. apply_accounting(...) charges exactly the scaled money      (pure ledger)

Steps 3-5 are the whole fix. Because the LEVERS are scaled — not just the money —
one number is simultaneously the cost, the demand effect, and what every
downstream seat (awareness, research fidelity, wage level, warehouse capacity)
reads. Capex comes first and is never scaled: the two gates already checked it
against the same start-of-tick cash, and a store you cannot fully pay for does
not half-open (design note Q2). Promotion is deliberately NOT scaled — promo cost
is a VARIABLE COST OF SALES, funded per unit actually sold (design note Q1), so
the consumer stockpile still builds from the unclamped promotion.

When the gate is off (``CoreConfig.default()``) ``World.step`` runs its
pre-existing sequence untouched and every clamp is 1.0 — bit-for-bit identical.

No RNG, no config-gating: every function here is pure arithmetic over its
arguments. Callers decide WHEN to call these — this module never reads a gate
flag itself, so it stays trivially unit-testable and has no opinion on
byte-identity paths.

Pure and framework-free: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.state import WorldState


@dataclass(frozen=True)
class DecodedLevers:
    """The per-retailer levers decoded from a tick's ``joint_action`` (``[K]`` each).

    Field-for-field the same arrays ``demand.py::_decode_levers`` used to return as
    a bare 11-tuple (same names as the local variables ``resolve_demand`` unpacked
    it into, same dtypes, same shapes) — this dataclass is purely a naming/typing
    upgrade over that tuple, not a new decode. See ``demand.py::_decode_levers``
    for the decode semantics (boundary validation via ``decode_action``, the
    per-lever NPC/absent-retailer registry defaults, which levers are
    contemporaneous vs lagged-investment vs perception-only).
    """

    price_indices: npt.NDArray[np.float64]  # [K]
    spend_fraction: npt.NDArray[np.float64]  # [K]
    assortment: npt.NDArray[np.float64]  # [K]
    promotion: npt.NDArray[np.float64]  # [K]
    expansion: npt.NDArray[np.float64]  # [K]
    research: npt.NDArray[np.float64]  # [K]
    service_level: npt.NDArray[np.float64]  # [K]
    automation_tier: npt.NDArray[np.int64]  # [K]
    loyalty_spend: npt.NDArray[np.float64]  # [K]
    wage_spend: npt.NDArray[np.float64]  # [K]
    warehouse_invest: npt.NDArray[np.float64]  # [K]


def gated_expansion_capex(
    state: WorldState,
    choices: npt.NDArray[np.float64],
    config: CoreConfig,
) -> tuple[npt.NDArray[np.float64], list[tuple[int, ...]]]:
    """ """
    n_regions = len(state.regions)
    capex_cfg = config.expansion.expansion_capex
    capex = np.zeros(len(state.retailers), dtype=np.float64)
    next_stores_per_region: list[tuple[int, ...]] = []
    for idx, retailer in enumerate(state.retailers):
        choice = int(choices[idx])
        stores = list(retailer.stores_per_region)
        # Pad to n_regions defensively (1-region fallback states).
        while len(stores) < n_regions:
            stores.append(0)
        valid_open = (
            1 <= choice < n_regions
            and stores[choice] == 0
            and retailer.cash >= capex_cfg  # PRE-capex (start-of-tick) cash
        )
        if valid_open:
            capex[idx] = capex_cfg
            stores[choice] = 1  # open-only here; MECHANIC 4 restructuring is the
            # only path that ever closes a store (never this gate)
        next_stores_per_region.append(tuple(stores))
    return capex, next_stores_per_region


def gated_automation_capex(
    state: WorldState,
    targets: npt.NDArray[np.int64],
    config: CoreConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """ """
    capex_per_tier = config.automation.capex_per_tier
    n = len(state.retailers)
    automation_capex = np.zeros(n, dtype=np.float64)
    new_tier = np.zeros(n, dtype=np.int64)
    reported_target = np.zeros(n, dtype=np.int64)
    for idx, retailer in enumerate(state.retailers):
        target = int(targets[idx])
        prev = retailer.automation_tier
        # Clamp pathological targets defensively (decode_action argmaxes a 3-wide
        # block so the value is in [0, n_tiers - 1] under normal flow, but a
        # NPC/absent retailer default has a 0 path that never trips this).
        if not 0 <= target < len(capex_per_tier):
            target = prev
        differential = capex_per_tier[target] - capex_per_tier[prev]
        monotone = target >= prev
        affordable = retailer.cash >= differential
        if monotone and affordable:
            new_tier[idx] = target
            automation_capex[idx] = differential
        else:
            new_tier[idx] = prev
            automation_capex[idx] = 0.0
        reported_target[idx] = target
    return automation_capex, new_tier, reported_target


def intended_discretionary_spend(
    levers: DecodedLevers,
    wage_bill: npt.NDArray[np.float64],
    config: CoreConfig,
) -> npt.NDArray[np.float64]:
    """ """
    marketing_exponent = config.marketing.marketing_cost_exponent
    if marketing_exponent == 1.0:
        marketing_term = levers.spend_fraction * config.marketing.cost_per_unit_spend
    else:
        marketing_term = (
            levers.spend_fraction**marketing_exponent * config.marketing.cost_per_unit_spend
        )
    return (
        marketing_term
        + config.assortment.assortment_cost_per_unit * levers.assortment
        + config.research.cost_per_fidelity * levers.research
        + config.loyalty_program.cost_per_unit_spend * levers.loyalty_spend
        + wage_bill
        + levers.warehouse_invest * config.warehouse.cost_per_unit_invest
    )


def scale_discretionary_levers(
    levers: DecodedLevers,
    clamp: npt.NDArray[np.float64],
) -> DecodedLevers:
    """``levers`` with every DISCRETIONARY lever multiplied by the per-retailer ``clamp``.

    Scaled: ``spend_fraction``, ``assortment``, ``research``, ``loyalty_spend``,
    ``wage_spend``, ``warehouse_invest`` — the six levers whose value IS a purchase.
    A retailer that can fund 40 % of its intent buys 40 % of the breadth, 40 % of the
    awareness build, 40 % of the fidelity: the scaled object is what
    ``resolve_demand`` reads, so cost and effect cannot come apart.

    Carried through UNSCALED: ``price_indices`` (a decision, not a purchase),
    ``promotion`` (a variable cost of sales — the consumer stockpile must keep
    building from the full promoted intensity, design note Q1), ``expansion`` and
    ``automation_tier`` (one-shot choices the capex gates already resolved against
    the same cash — scaling them would half-open a store).

    Note the WAGE BILL is not derivable from the scaled ``wage_spend`` alone
    (``clamp · (base + s·scale) ≠ base + (clamp·s)·scale``): the seam scales the paid
    wage itself, and the scaled ``wage_spend`` here is the reported SoT of what was
    actually bought. Pure — a new frozen instance; the input is never mutated.
    """
    return DecodedLevers(
        price_indices=levers.price_indices,
        spend_fraction=levers.spend_fraction * clamp,
        assortment=levers.assortment * clamp,
        promotion=levers.promotion,
        expansion=levers.expansion,
        research=levers.research * clamp,
        service_level=levers.service_level,
        automation_tier=levers.automation_tier,
        loyalty_spend=levers.loyalty_spend * clamp,
        wage_spend=levers.wage_spend * clamp,
        warehouse_invest=levers.warehouse_invest * clamp,
    )


def compute_spend_clamp(
    cash: npt.NDArray[np.float64],
    intended_discretionary: npt.NDArray[np.float64],
    committed_capex: npt.NDArray[np.float64],
    credit_limit: float = 0.0,
) -> npt.NDArray[np.float64]:
    """ """
    if credit_limit == 0.0:
        # Byte-identity anchor: the SAME code the pre-credit-line clamp always ran,
        # not merely a numerically-equal ``cash + 0.0 - committed_capex``.
        available = np.maximum(cash - committed_capex, 0.0)
    else:
        available = np.maximum(cash + credit_limit - committed_capex, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(intended_discretionary > 0.0, available / intended_discretionary, 1.0)
    return np.minimum(ratio, 1.0)
