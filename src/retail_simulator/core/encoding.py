"""Single authority for flat <-> structured action/observation layout.

Everything that needs to know *where* a value sits in a flat vector — the RL
adapters' Spaces, the action wrapper, and the future network codec — derives it
from here, so there is exactly one place that defines offsets and ordering.

Pure and framework-free: numpy only. No clamping/normalization policy beyond the
boundary validation documented below (observations are normalized in a wrapper,
never in core/).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from retail_simulator.core.schema import (
    ActionSchema,
    LeverKind,
    LeverSpec,
    ObservationField,
    default_action_schema,
    observation_schema,
)
from retail_simulator.core.state import PerceivedCompetitor, RetailerState, WorldState

OBS_DTYPE = np.float32


@dataclass(frozen=True)
class LeverLayout:
    """Where one lever sits in the flat action vector."""

    spec: LeverSpec
    offset: int
    width: int


@dataclass(frozen=True)
class ActionLayout:
    """Resolved flat action layout: ordered lever slots + total dimension."""

    levers: tuple[LeverLayout, ...]
    total_dim: int

    def slot(self, name: str) -> LeverLayout:
        """Look up the layout slot for a lever by name."""
        for lever in self.levers:
            if lever.spec.name == name:
                return lever
        raise KeyError(f"lever {name!r} not in flat layout")


_DEFAULT_LAYOUT: ActionLayout | None = None


def _compute_action_layout(schema: ActionSchema) -> ActionLayout:
    layouts: list[LeverLayout] = []
    offset = 0
    for lever in schema.levers:
        width = lever.width
        layouts.append(LeverLayout(spec=lever, offset=offset, width=width))
        offset += width
    return ActionLayout(levers=tuple(layouts), total_dim=offset)


def flatten_action_space_layout(schema: ActionSchema | None = None) -> ActionLayout:
    """Compute the flat layout (offsets/order) for the enabled levers.

    Levers are concatenated in the schema's registry order. Continuous levers
    occupy one slot; discrete levers occupy ``n`` slots (masked logits block).

    The layout is a pure function of the (immutable) schema and is needed on every
    decode/encode (the hot path). For the default schema — a cached singleton —
    the result is memoized and returned by identity check, avoiding both the
    recompute and the cost of hashing the nested schema. The returned
    ``ActionLayout`` is immutable; callers must not mutate it.
    """
    global _DEFAULT_LAYOUT
    if schema is None or schema is default_action_schema():
        if _DEFAULT_LAYOUT is None:
            _DEFAULT_LAYOUT = _compute_action_layout(default_action_schema())
        return _DEFAULT_LAYOUT
    return _compute_action_layout(schema)


def _competitors(state: WorldState) -> list[RetailerState]:
    """NPC retailers (the discounter in 0.0), in agent-index order."""
    return [r for r in state.retailers if r.is_npc]


def true_competitor_view(state: WorldState, agent_index: int) -> PerceivedCompetitor:
    """ """
    del agent_index  # the single competitor block is observer-independent in 1.0
    competitors = _competitors(state)
    first_competitor = competitors[0] if competitors else None
    return _true_competitor_view(first_competitor, len(state.regions))


def _true_competitor_view(
    first_competitor: RetailerState | None, n_regions: int
) -> PerceivedCompetitor:
    """ """
    n_extra = max(0, n_regions - 2)
    if first_competitor is None:
        return PerceivedCompetitor(
            extra_regions=tuple((0.0, 0.0, 0.0, 0.0) for _ in range(n_extra))
        )
    extra_regions = tuple(
        (
            _competitor_region_value(first_competitor.awareness_per_region, r),
            _competitor_region_value(first_competitor.loyalty_per_region, r),
            _competitor_region_value(first_competitor.last_market_share_per_region, r),
            0.0,
        )
        for r in range(2, n_regions)
    )
    return PerceivedCompetitor(
        competitor_price=first_competitor.price_index,
        competitor_market_share=first_competitor.last_market_share,
        competitor_awareness=first_competitor.awareness,
        competitor_assortment=first_competitor.last_assortment,
        competitor_promotion=first_competitor.last_promotion,
        competitor_loyalty_region_0=_competitor_region_value(
            first_competitor.loyalty_per_region, 0
        ),
        competitor_awareness_region_1=_competitor_region_value(
            first_competitor.awareness_per_region, 1
        ),
        competitor_loyalty_region_1=_competitor_region_value(
            first_competitor.loyalty_per_region, 1
        ),
        competitor_market_share_region_1=_competitor_region_value(
            first_competitor.last_market_share_per_region, 1
        ),
        extra_regions=extra_regions,
    )


def encode_observation(state: WorldState, agent_index: int) -> npt.NDArray[np.float32]:
    """ """
    me = state.retailers[agent_index]
    competitors = _competitors(state)
    n_regions = len(state.regions)

    total_regional_demand = sum(region.realized_regional_demand for region in state.regions)
    if competitors:
        mean_competitor_price = sum(c.price_index for c in competitors) / len(competitors)
    else:
        mean_competitor_price = me.price_index

    # The single competitor block is the first NPC; absent NPCs encode as zeros so
    # the vector length is stable regardless of seating. The TRUE values are the
    # fallback when no perceived view is seated for this observer.
    first_competitor = competitors[0] if competitors else None
    true_competitor = _true_competitor_view(first_competitor, n_regions)
    # Phase 1.0: read the observer's SEATED perceived (noised) competitor block; fall
    # back to the TRUE values when none is seated (a bare WorldState / non-observer).
    perceived = state.perceived_competitors.get(agent_index, true_competitor)
    competitor_price = perceived.competitor_price
    competitor_market_share = perceived.competitor_market_share
    competitor_awareness = perceived.competitor_awareness
    competitor_assortment = perceived.competitor_assortment
    competitor_promotion = perceived.competitor_promotion

    stockpile = sum(region.stockpile for region in state.regions)

    values: dict[str, float] = {
        "cash": me.cash,
        "last_profit": me.last_profit,
        "last_revenue": me.last_revenue,
        # Aggregate (region-0) reporting share for the legacy offset-3 slot.
        "market_share": me.last_market_share,
        "price_level": me.price_index,
        # Legacy offset-5: region-0 loyalty (the per-region SoT index 0).
        "loyalty": me.loyalty_stock,
        # Legacy offset-6: total open-store count across regions (presence count).
        "stores": float(me.stores),
        "total_regional_demand": total_regional_demand,
        "mean_competitor_price": mean_competitor_price,
        "competitor_price": competitor_price,
        "competitor_market_share": competitor_market_share,
        # Legacy offset-11: region-0 seated awareness (the per-region SoT index 0).
        "awareness": me.awareness,
        "competitor_awareness": competitor_awareness,
        "assortment": me.last_assortment,
        "competitor_assortment": competitor_assortment,
        "promotion": me.last_promotion,
        "competitor_promotion": competitor_promotion,
        # Legacy offset-17: aggregate seated stockpile (region 0 in the 1-region path).
        "stockpile": stockpile,
        "expansion": me.last_expansion,
        "loyalty_region_0": _region_value(me.loyalty_per_region, 0),
        "competitor_loyalty_region_0": perceived.competitor_loyalty_region_0,
        "awareness_region_1": _region_value(me.awareness_per_region, 1),
        "loyalty_region_1": _region_value(me.loyalty_per_region, 1),
        # Presence is 0/1 in 0.4 (one store per opened region). SELF — EXACT.
        "stores_region_1": float(_region_int(me.stores_per_region, 1) > 0),
        "market_share_region_1": _region_value(me.last_market_share_per_region, 1),
        "total_regional_demand_region_1": _region_realized_demand(state, 1),
        "stockpile_region_1": _region_stockpile(state, 1),
        "competitor_awareness_region_1": perceived.competitor_awareness_region_1,
        "competitor_loyalty_region_1": perceived.competitor_loyalty_region_1,
        "competitor_market_share_region_1": perceived.competitor_market_share_region_1,
        # --- Phase 1.0 tail: the agent's OWN current perception fidelity (SELF, EXACT).
        # Default 0.0 when no fidelity is seated (a bare WorldState / pre-rollout). ---
        "research_fidelity": state.research_fidelity.get(agent_index, 0.0),
        "competitor_mean_price_region_0": perceived.competitor_mean_price_region_0,
        "competitor_mean_price_region_1": perceived.competitor_mean_price_region_1,
        # --- Phase 1.1 tail: the agent's OWN seated per-region service_score (SELF,
        # EXACT — own state is always known; not noised). Falls back to 1.0 (the reset
        # fixed point) when the per-region array is shorter than n_regions (a
        # 1-region-fallback state); the offsets are the obs tail at 33/34, never moved. ---
        "service_score_region_0": _service_score_region(me, 0),
        "service_score_region_1": _service_score_region(me, 1),
        # --- Phase 1.2 tail (F3 = PERMANENT): the agent's OWN seated automation TIER
        # (SELF, EXACT — own state, not noised; not subject to perception). A per-
        # retailer MONOTONE NON-DECREASING integer in ``[0, N_AUTOMATION_TIERS-1]``,
        # cast to float for the Box dtype contract. The SoT the next-tick demand
        # reads (PRE-update / lagged) to apply the COGS multiplier via
        # ``savings_per_tier[automation_tier_prev]``. PURE READ off
        # ``RetailerState.automation_tier`` (no RNG). Offset 35 — appended at the obs
        # tail; existing offsets 0–34 NEVER move. ---
        "automation_tier": float(me.automation_tier),
        # --- Phase 5.0 tail: four SELF fields for the two new Phase-5 levers (own
        # state, EXACT — not noised). PURE READS off ``RetailerState`` (no RNG). All
        # four are the existing Phase-4 seats or the NEW B-1 reporting seats — B-1
        # writes 0.0 to the two reporting seats at reset and never consumes the new
        # levers, so these read the same values a pre-B-1 world would carry. Offsets
        # 36–39 — appended at the obs tail; existing offsets 0–35 NEVER move. ---
        "employee_happiness": me.employee_happiness,
        "wage_level": me.wage_level,
        "warehouse_utilization": me.last_warehouse_utilization,
        "last_stockout_rate": me.last_stockout_rate,
        "others_present_region_0": _others_present(state, agent_index, 0),
        "others_present_region_1": _others_present(state, agent_index, 1),
    }

    for r in range(2, n_regions):
        values.update(_extra_region_values(state, me, agent_index, perceived, r))

    fields: tuple[ObservationField, ...] = observation_schema(n_regions)
    vector = np.array([values[f.name] for f in fields], dtype=OBS_DTYPE)
    return vector


def _others_present(state: WorldState, agent_index: int, r: int) -> float:
    """ """
    return float(
        sum(
            1
            for k, retailer in enumerate(state.retailers)
            if k != agent_index and _region_int(retailer.stores_per_region, r) > 0
        )
    )


def _extra_region_values(
    state: WorldState,
    me: RetailerState,
    agent_index: int,
    perceived: PerceivedCompetitor,
    r: int,
) -> dict[str, float]:
    """ """
    extra_idx = r - 2
    if 0 <= extra_idx < len(perceived.extra_regions):
        comp_awareness, comp_loyalty, comp_share, comp_mean_price = perceived.extra_regions[
            extra_idx
        ]
    else:
        comp_awareness = comp_loyalty = comp_share = comp_mean_price = 0.0
    return {
        f"awareness_region_{r}": _region_value(me.awareness_per_region, r),
        f"loyalty_region_{r}": _region_value(me.loyalty_per_region, r),
        f"stores_region_{r}": float(_region_int(me.stores_per_region, r) > 0),
        f"market_share_region_{r}": _region_value(me.last_market_share_per_region, r),
        f"service_score_region_{r}": _service_score_region(me, r),
        f"total_regional_demand_region_{r}": _region_realized_demand(state, r),
        f"stockpile_region_{r}": _region_stockpile(state, r),
        f"others_present_region_{r}": _others_present(state, agent_index, r),
        f"competitor_awareness_region_{r}": comp_awareness,
        f"competitor_loyalty_region_{r}": comp_loyalty,
        f"competitor_market_share_region_{r}": comp_share,
        f"competitor_mean_price_region_{r}": comp_mean_price,
    }


def _region_value(per_region: tuple[float, ...], r: int) -> float:
    """Read a per-region float array at region ``r``, 0.0 if the region is absent.

    A single-region (0.3-fallback) state has length-1 tuples; a region-1 read then
    falls back to 0.0 so the vector length stays stable regardless of n_regions.
    """
    return per_region[r] if r < len(per_region) else 0.0


def _region_int(per_region: tuple[int, ...], r: int) -> int:
    """Read a per-region int array at region ``r``, 0 if the region is absent."""
    return per_region[r] if r < len(per_region) else 0


def _competitor_region_value(per_region: tuple[float, ...] | None, r: int) -> float:
    """Read a competitor's per-region float at ``r`` (0.0 if no competitor/region)."""
    if per_region is None:
        return 0.0
    return per_region[r] if r < len(per_region) else 0.0


def _region_realized_demand(state: WorldState, r: int) -> float:
    """Realized regional demand for region ``r`` (0.0 if the region is absent)."""
    return state.regions[r].realized_regional_demand if r < len(state.regions) else 0.0


def _region_stockpile(state: WorldState, r: int) -> float:
    """Seated stockpile for region ``r`` (0.0 if the region is absent)."""
    return state.regions[r].stockpile if r < len(state.regions) else 0.0


def _service_score_region(me: RetailerState, r: int) -> float:
    """Read the agent's per-region seated service_score, defaulting to 1.0 (Phase 1.1).

    The per-region array may be shorter than ``n_regions`` (a 1-region-fallback state
    that pre-dates Phase 1.1's per-region SoT); a missing region reads as 1.0 (the
    reset/fixed-point value of the score — perfect service, the byte-identity anchor),
    NOT 0.0 (which would be a stocked-out reputation — the opposite extreme).
    """
    spr = me.service_score_per_region
    return spr[r] if r < len(spr) else 1.0


@dataclass(frozen=True)
class StructuredAction:
    """Decoded, validated action keyed by lever name.

    Continuous levers map to a clipped float; discrete levers map to the
    argmax-decoded integer choice.
    """

    levers: dict[str, float]


def decode_action(
    flat_action: npt.NDArray[np.float32] | list[float] | tuple[float, ...],
    schema: ActionSchema | None = None,
) -> StructuredAction:
    """Decode a flat action into validated structured levers.

    Boundary validation lives here (trust internal callers afterwards):
    continuous levers are clipped to ``[low, high]``; discrete levers are decoded
    by argmax over their logits block. Raises ``ValueError`` on wrong length.

    The plain-float tuple form is what ``RetailerState.pending_action`` stores
    (NFD-1): widening float32 to float64 is lossless and ``np.asarray`` below does it
    for the array form too, so decoding the tuple is bit-identical to decoding the
    float32 array the policy returned.
    """
    if schema is None:
        schema = default_action_schema()
    layout = flatten_action_space_layout(schema)

    flat = np.asarray(flat_action, dtype=np.float64).ravel()
    if flat.shape[0] != layout.total_dim:
        raise ValueError(f"flat_action has length {flat.shape[0]}, expected {layout.total_dim}")

    decoded: dict[str, float] = {}
    for slot in layout.levers:
        block = flat[slot.offset : slot.offset + slot.width]
        if slot.spec.kind is LeverKind.CONTINUOUS:
            assert slot.spec.low is not None and slot.spec.high is not None
            # Python clamp is bit-identical to np.clip for finite scalars and far
            # cheaper on the hot path (decode runs several times per tick).
            value = float(block[0])
            decoded[slot.spec.name] = min(max(value, slot.spec.low), slot.spec.high)
        else:
            decoded[slot.spec.name] = float(int(np.argmax(block)))
    return StructuredAction(levers=decoded)


def encode_action(
    structured: StructuredAction, schema: ActionSchema | None = None
) -> npt.NDArray[np.float32]:
    """Encode structured lever values back into a flat vector.

    Inverse of ``decode_action`` for round-tripping/tests. Continuous levers
    write their (already in-range) value; discrete levers write a one-hot block.
    """
    if schema is None:
        schema = default_action_schema()
    layout = flatten_action_space_layout(schema)

    flat = np.zeros(layout.total_dim, dtype=OBS_DTYPE)
    for slot in layout.levers:
        value = structured.levers[slot.spec.name]
        if slot.spec.kind is LeverKind.CONTINUOUS:
            flat[slot.offset] = value
        else:
            flat[slot.offset + int(value)] = 1.0
    return flat
