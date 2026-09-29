"""NPC policies: the ``NPCPolicy`` protocol and the scripted archetypes.

NPCs fill the non-learning seats. Their actions are **draw #2** of the per-tick
order, evaluated in agent-index (seat) order (see ``core/rng.py``). Every policy
takes the injected RNG so a stochastic NPC stays replayable; the scripted
archetypes shipped here (Discounter, Premium, Balanced) are all DETERMINISTIC and
consume NO RNG draw — they accept ``rng`` and leave it untouched.

The :class:`NPCPolicy` protocol returns a **flat action vector** in the same
layout as the agent's action (``encoding.flatten_action_space_layout`` /
``encode_action``), so a frozen *learned* policy — which also emits a flat action
— drops into the same seat with no adapter. ``world.py`` decodes/validates the
returned flat action through ``decode_action`` exactly as it does the agent's.

**Every lever a policy emits is played (NFD-1).** ``World.step`` seats the whole
returned vector as the seat's ``RetailerState.pending_action`` and
``demand.py::_decode_levers`` consumes it on the NEXT tick as that seat's
joint-action entry — so an archetype's assortment, promotion, service_level,
automation tier, loyalty spend, expansion vote, research, wage and warehouse
investment move the economy (and cost it money) exactly as a learning seat's
would, one tick after the policy decided them. Before NFD-1 only ``price_index``
and (under an armed cash budget) ``marketing`` had any effect and the rest of the
posture was reporting-only, which made the values below descriptive fiction; they
are now the real thing, so changing one changes the game. The one-tick lag is
structural: ``act`` runs at draw #2, AFTER demand has already resolved (draw #1),
so a policy's decision cannot reach the same tick's economy without reordering the
determinism contract.

**``research`` is the one lever that is COST-ONLY for a scripted seat.** Its opex
line is charged like any other, but the thing the money buys — a sharpened
perceived view of rivals — is drawn (draw #4) and seated only for OBSERVERS, and
NPC seats are not observers. A scripted policy that spends on research therefore
pays for nothing it can use. That asymmetry is deliberate, not an oversight: the
ledger must be blind to seat kind (a posture costs the same money wherever it
sits — the parity property in ``tests/core/test_npc_full_decode.py``), and
special-casing the charge would reintroduce exactly the is-this-seat-an-NPC branch
NFD-1 removed. Every shipped archetype emits ``research = 0.0``, so nothing pays
it today; a future policy that sets it non-zero is buying nothing, and should be
given an observer seat rather than a refund.

Phase 0.5 (F5) adds the acting ``seat_index`` to ``act`` so a REACTIVE archetype
(:class:`Balanced`) can EXCLUDE ITSELF when reading the other present retailers'
prices. ``Discounter``/``Premium`` accept-and-ignore it (their action is fixed).
The seam passes the seat index when it calls each NPC seat in seat order.

Pure and framework-free: numpy only.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import StructuredAction, encode_action
from retail_simulator.core.schema import ActionSchema, default_action_schema
from retail_simulator.core.state import WorldState

# Default discounter price_index: a fixed aggressive discount (~0.7) per the PO's
# NPC spec. Lives as a module default; a per-instance value can override it.
DISCOUNTER_PRICE_INDEX: float = 0.7
DISCOUNTER_PROMOTION: float = 0.0

PREMIUM_PRICE_INDEX: float = 1.4
PREMIUM_MARKETING: float = 0.7
PREMIUM_ASSORTMENT: float = 0.9
PREMIUM_PROMOTION: float = 0.1

BALANCED_REACT_FACTOR: float = 0.95
BALANCED_MARKETING: float = 0.6
BALANCED_ASSORTMENT: float = 0.7
BALANCED_PROMOTION: float = 0.7
# Fallback price the Balanced NPC matches when it is the only present retailer (no
# "other present" prices to read): the neutral pricing baseline.
BALANCED_BASELINE_PRICE_INDEX: float = 1.0

WANDERING_BASELINE_PRICE_INDEX: float = 1.0
WANDERING_AMPLITUDE: float = 0.3
WANDERING_PERIOD: int = 26
WANDERING_MARKETING: float = 0.4
WANDERING_ASSORTMENT: float = 0.5
WANDERING_PROMOTION: float = 0.3


class NPCPolicy(Protocol):
    """Interface every NPC (and frozen learned policy) seat satisfies.

    ``act`` is called once per tick for the NPC's seat, in seat-index order. It
    receives the current ``WorldState`` (full info in 0.0), the injected RNG, and
    the acting ``seat_index`` (Phase 0.5 F5 — so a reactive archetype can exclude
    itself when reading the other retailers' state), and returns a flat action
    vector (price_index continuous at offset 0). Stochastic policies MUST draw only
    from the passed ``rng`` so behavior is replayable; deterministic policies ignore
    it. ``seat_index`` is the policy's own index into ``state.retailers``.
    """

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int
    ) -> npt.NDArray[np.float32]: ...


class Discounter:
    """ """

    def __init__(
        self,
        price_index: float = DISCOUNTER_PRICE_INDEX,
        schema: ActionSchema | None = None,
    ) -> None:
        self._price_index = price_index
        self._schema = schema

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int = 0
    ) -> npt.NDArray[np.float32]:
        """Return the fixed-discount, no-marketing, narrow, no-promo, no-expand action.

        Deterministic (no ``rng`` draw): a narrow discounter prices at the fixed
        discount, never markets, runs a narrow range (``assortment=0.0``), runs no
        promotions (``promotion=0.0``), and never expands (``expansion=0.0`` — a
        no-op one-hot in the discrete block). Its presence in both regions comes from
        the reset map, not this action. ``seat_index`` is accepted for the Phase 0.5
        protocol (a static archetype ignores it — its action does not depend on which
        seat it occupies).
        """
        structured = StructuredAction(
            levers={
                "price_index": self._price_index,
                "marketing": 0.0,
                "assortment": 0.0,
                "promotion": DISCOUNTER_PROMOTION,
                "expansion": 0.0,
                "research": 0.0,
                "service_level": 1.0,
                # Phase 1.2 (F3 = PERMANENT): NPCs are not automation decision-makers
                # — they emit target tier 0 every tick (the registry default — no
                # upgrade ever fires), so their seated ``automation_tier`` stays at 0
                # forever (monotonicity + same-tier no-op) and they contribute 0
                # differential capex. The byte-identity baseline (Discounter/Premium/
                # Balanced/Wandering all keep the 1.1 economics byte-for-byte).
                "automation": 0.0,
                # Phase 1.3: NPCs are not loyalty-program decision-makers — they emit
                # ``loyalty_spend = 0.0`` every tick (the registry default — no
                # program), so they contribute 0 same-tick boost AND 0 loyalty-program
                # opex. Preserves the 1.2 byte-identity baseline.
                "loyalty_spend": 0.0,
                # Phase 5.0: NPCs are not wage/warehouse decision-makers — they emit
                # ``wage_spend = 0.0`` and ``warehouse_invest = 0.0`` every tick (the
                # registry defaults — no premium pay, no capacity investment),
                # preserving the 1.3 byte-identity baseline.
                "wage_spend": 0.0,
                "warehouse_invest": 0.0,
            }
        )
        return encode_action(structured, self._schema)


class Premium:
    """ """

    def __init__(
        self,
        price_index: float = PREMIUM_PRICE_INDEX,
        marketing: float = PREMIUM_MARKETING,
        assortment: float = PREMIUM_ASSORTMENT,
        promotion: float = PREMIUM_PROMOTION,
        schema: ActionSchema | None = None,
    ) -> None:
        self._price_index = price_index
        self._marketing = marketing
        self._assortment = assortment
        self._promotion = promotion
        self._schema = schema

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int = 0
    ) -> npt.NDArray[np.float32]:
        """Return the fixed premium (high-price, broad, high-marketing) 7-D action.

        Deterministic (no ``rng`` draw); ``seat_index`` is accepted-and-ignored (a
        static archetype's action does not depend on its seat). Expansion is a no-op
        (choice 0) — Premium does not expand; its presence comes from the reset plan.
        """
        structured = StructuredAction(
            levers={
                "price_index": self._price_index,
                "marketing": self._marketing,
                "assortment": self._assortment,
                "promotion": self._promotion,
                "expansion": 0.0,
                "research": 0.0,
                "service_level": 1.0,
                "automation": 0.0,
                # Phase 1.3: NPCs are not loyalty-program decision-makers — emit
                # ``loyalty_spend = 0.0`` (the registry default; no boost, no opex).
                "loyalty_spend": 0.0,
                # Phase 5.0: NPCs are not wage/warehouse decision-makers — emit
                # ``wage_spend = 0.0`` and ``warehouse_invest = 0.0`` (the registry
                # defaults; no premium pay, no capacity investment).
                "wage_spend": 0.0,
                "warehouse_invest": 0.0,
            }
        )
        return encode_action(structured, self._schema)


class Balanced:
    """ """

    def __init__(
        self,
        react_factor: float = BALANCED_REACT_FACTOR,
        marketing: float = BALANCED_MARKETING,
        assortment: float = BALANCED_ASSORTMENT,
        promotion: float = BALANCED_PROMOTION,
        min_price_index: float | None = None,
        max_price_index: float | None = None,
        schema: ActionSchema | None = None,
    ) -> None:
        self._react_factor = react_factor
        self._marketing = marketing
        self._assortment = assortment
        self._promotion = promotion
        self._min_price_index = min_price_index
        self._max_price_index = max_price_index
        self._schema = schema

    def _price_bounds(self) -> tuple[float, float]:
        """Resolve (min, max) price bounds: the configured values, else the schema's.

        Falls back to the ``price_index`` lever's ``low``/``high`` from the (default
        or injected) schema so the bounds always match the action contract the seam
        clips against — the decoded price is re-clipped at ``decode_action`` anyway,
        so this only keeps the reactive price in-range before encoding.
        """
        if self._min_price_index is not None and self._max_price_index is not None:
            return self._min_price_index, self._max_price_index
        schema = self._schema if self._schema is not None else default_action_schema()
        price_lever = next(lever for lever in schema.levers if lever.name == "price_index")
        assert price_lever.low is not None and price_lever.high is not None
        return price_lever.low, price_lever.high

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int
    ) -> npt.NDArray[np.float32]:
        """ """
        other_prices = [
            r.price_index
            for idx, r in enumerate(state.retailers)
            if idx != seat_index and r.stores > 0
        ]
        if other_prices:
            observed = sum(other_prices) / len(other_prices)
        else:
            observed = BALANCED_BASELINE_PRICE_INDEX
        min_price, max_price = self._price_bounds()
        price = min(max(observed * self._react_factor, min_price), max_price)
        structured = StructuredAction(
            levers={
                "price_index": price,
                "marketing": self._marketing,
                "assortment": self._assortment,
                "promotion": self._promotion,
                "expansion": 0.0,
                "research": 0.0,
                "service_level": 1.0,
                "automation": 0.0,
                # Phase 1.3: NPCs are not loyalty-program decision-makers — emit
                # ``loyalty_spend = 0.0`` (the registry default; no boost, no opex).
                "loyalty_spend": 0.0,
                # Phase 5.0: NPCs are not wage/warehouse decision-makers — emit
                # ``wage_spend = 0.0`` and ``warehouse_invest = 0.0`` (the registry
                # defaults; no premium pay, no capacity investment).
                "wage_spend": 0.0,
                "warehouse_invest": 0.0,
            }
        )
        return encode_action(structured, self._schema)


class Wandering:
    """ """

    def __init__(
        self,
        baseline: float = WANDERING_BASELINE_PRICE_INDEX,
        amplitude: float = WANDERING_AMPLITUDE,
        period: int = WANDERING_PERIOD,
        marketing: float = WANDERING_MARKETING,
        assortment: float = WANDERING_ASSORTMENT,
        promotion: float = WANDERING_PROMOTION,
        min_price_index: float | None = None,
        max_price_index: float | None = None,
        schema: ActionSchema | None = None,
    ) -> None:
        if period <= 0:
            raise ValueError(f"Wandering period must be > 0, got {period}")
        self._baseline = baseline
        self._amplitude = amplitude
        self._period = period
        self._marketing = marketing
        self._assortment = assortment
        self._promotion = promotion
        self._min_price_index = min_price_index
        self._max_price_index = max_price_index
        self._schema = schema

    def _price_bounds(self) -> tuple[float, float]:
        """Resolve (min, max) price bounds: the configured values, else the schema's."""
        if self._min_price_index is not None and self._max_price_index is not None:
            return self._min_price_index, self._max_price_index
        schema = self._schema if self._schema is not None else default_action_schema()
        price_lever = next(lever for lever in schema.levers if lever.name == "price_index")
        assert price_lever.low is not None and price_lever.high is not None
        return price_lever.low, price_lever.high

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int = 0
    ) -> npt.NDArray[np.float32]:
        """ """
        del seat_index  # the schedule is seat-independent (exogenous, tick-driven)
        raw = self._baseline + self._amplitude * math.sin(2.0 * math.pi * state.tick / self._period)
        min_price, max_price = self._price_bounds()
        price = min(max(raw, min_price), max_price)
        structured = StructuredAction(
            levers={
                "price_index": price,
                "marketing": self._marketing,
                "assortment": self._assortment,
                "promotion": self._promotion,
                "expansion": 0.0,
                "research": 0.0,
                "service_level": 1.0,
                "automation": 0.0,
                # Phase 1.3: NPCs are not loyalty-program decision-makers — emit
                # ``loyalty_spend = 0.0`` (the registry default; no boost, no opex).
                "loyalty_spend": 0.0,
                # Phase 5.0: NPCs are not wage/warehouse decision-makers — emit
                # ``wage_spend = 0.0`` and ``warehouse_invest = 0.0`` (the registry
                # defaults; no premium pay, no capacity investment).
                "wage_spend": 0.0,
                "warehouse_invest": 0.0,
            }
        )
        return encode_action(structured, self._schema)
