"""Build Gymnasium Spaces FROM the core schema (single source of truth).

The flat action/observation layout is owned by ``core.encoding`` and described by
``core.schema``; this module turns those contracts into concrete Gymnasium
``Box`` spaces so the RL adapter never hardcodes a dimension or an offset. If a
lever or observation field is added to the schema in a later phase, the spaces
grow automatically.

Layer note: this is the adapter layer, so importing ``gymnasium`` here is allowed
(CI forbids it only under ``core/``). All world/layout knowledge still comes from
``core``.
"""

from __future__ import annotations

import re

import numpy as np
from gymnasium import spaces

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.encoding import OBS_DTYPE, flatten_action_space_layout
from retail_simulator.core.schema import (
    N_REGIONS_MAX,
    ActionSchema,
    LeverKind,
    ObservationField,
    ObservationGroup,
    default_action_schema,
    observation_schema,
)

# Generous finite magnitude for unbounded monetary fields (cash/profit/revenue).
# These accumulate over a continuing episode, so the bound must comfortably
# exceed any realized value while staying finite (gymnasium's env_checker and SB3
# both prefer finite Box bounds over +/-inf). Chosen large enough that observed
# rollout maxima (tens of thousands of cash) sit ~3 orders of magnitude below it.
_MONETARY_BOUND: float = 1e9

# Per-field bounds derived from the field's conceptual group / known semantics.
# Keyed by ObservationField.name; falls through to a group-based default. Kept as
# (low, high). Shares/loyalty live in [0, 1]; the agent price level mirrors the
# pricing lever bounds; competitor price mirrors it too (the discounter sits
# inside [0.5, 2.0]); monetary and demand magnitudes use the generous bound.
_SHARE_BOUNDS: tuple[float, float] = (0.0, 1.0)

_REGION_SUFFIX_RE: re.Pattern[str] = re.compile(r"^(.+)_region_(\d+)$")

# One bound RULE per per-region-block role, keyed by the BARE role name the regex
# above extracts (see the field roles in ``schema._region_observation_block`` / the
# hand-written region-0/1 block in ``schema.OBSERVATION_SCHEMA``) — exhaustive over
# all twelve per-region roles.
_REGION_ROLE_SHARE: frozenset[str] = frozenset(
    {
        "loyalty",
        "stores",
        "market_share",
        "service_score",
        "competitor_loyalty",
        "competitor_market_share",
    }
)
_REGION_ROLE_AWARENESS: frozenset[str] = frozenset({"awareness", "competitor_awareness"})
_REGION_ROLE_MONETARY: frozenset[str] = frozenset({"total_regional_demand", "others_present"})
_REGION_ROLE_STOCKPILE: str = "stockpile"
_REGION_ROLE_PRICE: str = "competitor_mean_price"


def _region_role_bounds(
    role: str, price_high: float, awareness_cap: float, stockpile_cap: float
) -> tuple[float, float] | None:
    """Bound for a per-region-block field ROLE (the name with its region digits
    stripped), or ``None`` if ``role`` names no known per-region role.
    """
    if role in _REGION_ROLE_SHARE:
        return _SHARE_BOUNDS
    if role in _REGION_ROLE_AWARENESS:
        return (0.0, awareness_cap)
    if role in _REGION_ROLE_MONETARY:
        # A retailer/units COUNT with no natural tight cap at this layer (the SAME
        # generous non-negative magnitude bound the aggregate ``stores``/
        # ``total_regional_demand`` fields use below).
        return (0.0, _MONETARY_BOUND)
    if role == _REGION_ROLE_STOCKPILE:
        return (0.0, stockpile_cap)
    if role == _REGION_ROLE_PRICE:
        # Low = 0.0 (not price_low): a bare/unseeded perceived view defaults this
        # field to 0.0 (``PerceivedCompetitor``'s documented bare-state fallback), so
        # the bound must contain both that sentinel and the clamped-seeded value.
        return (0.0, price_high)
    return None


def _observation_field_bounds(
    field: ObservationField, config: CoreConfig, n_regions: int
) -> tuple[float, float]:
    """Resolve (low, high) for one observation field from its semantics.

    Price-like fields are bounded by the pricing config; shares/loyalty are
    bounded to [0, 1]; monetary and demand magnitudes use a generous finite
    bound. Profit may be negative (opex can exceed margin), so its low is the
    negative of the monetary bound. ``n_regions`` is needed ONLY for the
    aggregate ``"stockpile"`` field (see below) — every per-region-block field
    (incl. ``stockpile_region_r``) resolves its bound from ``config`` alone via
    ``_region_role_bounds``.
    """
    price_low = config.pricing.min_price_index
    price_high = config.pricing.max_price_index

    match = _REGION_SUFFIX_RE.match(field.name)
    if match is not None:
        role = match.group(1)
        awareness_cap = 1.0 / config.marketing.decay
        stockpile_cap = 1.0 / config.promotion.stockpile_decay
        region_bounds = _region_role_bounds(role, price_high, awareness_cap, stockpile_cap)
        if region_bounds is not None:
            return region_bounds

    # Fields whose range is a known closed interval, regardless of group.
    if field.name in ("market_share", "loyalty", "competitor_market_share"):
        return _SHARE_BOUNDS
    if field.name == "competitor_price":
        # The single competitor block reads the first NPC's price (in the pricing
        # bounds); when NO competitor is seated (e.g. a learning-only PettingZoo world
        # with no NPC seat) ``encode_observation`` encodes it as the 0.0 sentinel
        # ("absent NPCs encode as zeros so the vector length is stable"), so the LOW
        # bound is 0.0, not min_price_index. The single-agent Gym view always seats the
        # discounter, so its competitor_price stays within [min_price, max_price] ⊂ this.
        return (0.0, price_high)
    if field.name in ("price_level", "mean_competitor_price"):
        # The agent's own price + the mean competitor price (which falls back to the
        # agent's own price when there is no competitor) both sit within the pricing
        # bounds — neither takes the 0.0 absent-competitor sentinel.
        return (price_low, price_high)
    if field.name in ("awareness", "competitor_awareness"):
        # Seated awareness stock (Phase 0.1, lagged model). The recurrence
        # clip((1-decay)*awareness + spend, 0, 1/decay) caps awareness at
        # 1/decay, so the tight bound is [0, 1/decay] — NOT the [0, 1] share
        # bounds (awareness can exceed 1.0) and NOT the loose 1e9 monetary
        # fallback (env_checker/SB3 prefer tight finite bounds for stability).
        return (0.0, 1.0 / config.marketing.decay)
    if field.name in (
        "assortment",
        "competitor_assortment",
        "promotion",
        "competitor_promotion",
    ):
        # Reporting-only breadth/promotion acted last tick (assortment 0.2,
        # promotion 0.3). The decoded value is a clean fraction in [0, 1] (the
        # lever's action bounds), so these obs fields are bounded [0, 1] like a
        # share — NOT the awareness [0, 1/decay] cap and NOT the 1e9 monetary
        # fallback.
        return _SHARE_BOUNDS
    if field.name == "research_fidelity":
        # Phase 1.0: the agent's OWN current perception fidelity — the saturating
        # mapping fidelity = spend / (spend + k) yields a value in [0, 1) (interior,
        # never exactly 1), so the closed [0, 1] share bound contains it tightly. NOT
        # the 1e9 monetary fallback (SB3/env_checker prefer tight bounds). Own state
        # is EXACT, so no clamp is needed — the value is already in range.
        return _SHARE_BOUNDS
    # NOTE: "service_score_region_0"/"_1" (Phase 1.1 — the per-region liability; the
    # EMA recurrence is clipped to [0, 1] in ``world.step``) are resolved by the
    # generalized "<role>_region_<r>" regex path above (role "service_score", in
    # ``_REGION_ROLE_SHARE``) — no literal entry needed here.
    if field.name == "automation_tier":
        # Phase 1.2 (F3 = PERMANENT): the agent's OWN seated automation TIER — a
        # per-retailer monotone non-decreasing integer in ``[0, N_AUTOMATION_TIERS −
        # 1]`` carried as a float for the Box dtype contract (the FIFTH seated
        # dynamic; structural analog of expansion's per-region ``stores_per_region``
        # but rolled to a per-retailer scalar). The seat advances ONLY in
        # ``world.step``'s monotonicity + differential-affordability gate; it never
        # decreases. The tight finite range is derived from
        # ``N_AUTOMATION_TIERS``, NOT the 1e9 monetary fallback (SB3/env_checker
        # prefer tight bounds). Own state is EXACT — no clamp is needed at encode.
        from retail_simulator.core.schema import N_AUTOMATION_TIERS

        return (0.0, float(N_AUTOMATION_TIERS - 1))
    if field.name in (
        "employee_happiness",
        "warehouse_utilization",
        "last_stockout_rate",
    ):
        return _SHARE_BOUNDS
    if field.name == "stockpile":
        return (0.0, n_regions / config.promotion.stockpile_decay)
    if field.name == "expansion":
        return (0.0, float(N_REGIONS_MAX - 1))
    if field.name == "last_profit":
        # Profit can be negative when opex outweighs margin at a given price.
        return (-_MONETARY_BOUND, _MONETARY_BOUND)
    if field.name in (
        "cash",
        "last_revenue",
        "total_regional_demand",
        "stores",
        # Phase 5.0: the wage actually PAID last tick (``wage_base`` at reset /
        # default; once the seam (B-2) is live, ``wage_base + wage_spend ·
        # wage_spend_scale``) — an unbounded-above money magnitude like cash/
        # revenue, so it shares the same generous non-negative bound.
        "wage_level",
    ):
        # Every per-region realized-demand / other-present-count field
        # (``total_regional_demand_region_r`` / ``others_present_region_r``, incl.
        # region 0/1) shares this identical generous bound via the generalized
        # region-role path above (roles in ``_REGION_ROLE_MONETARY``).
        return (0.0, _MONETARY_BOUND)

    # Defensive group-based fallback so newly-added fields still get a sane box:
    # SELF/MARKET magnitudes default to the generous non-negative bound; an
    # unknown field is never left unbounded.
    if field.group in (ObservationGroup.SELF, ObservationGroup.MARKET):
        return (0.0, _MONETARY_BOUND)
    return (0.0, _MONETARY_BOUND)


def build_observation_space(config: CoreConfig | None = None) -> spaces.Box:
    """ """
    if config is None:
        config = CoreConfig.default()

    n_regions = len(config.demand.regions) if config.demand.regions else 1
    fields = observation_schema(n_regions)

    lows = np.array(
        [_observation_field_bounds(f, config, n_regions)[0] for f in fields],
        dtype=OBS_DTYPE,
    )
    highs = np.array(
        [_observation_field_bounds(f, config, n_regions)[1] for f in fields],
        dtype=OBS_DTYPE,
    )
    return spaces.Box(low=lows, high=highs, shape=(len(fields),), dtype=OBS_DTYPE)


def build_action_space(schema: ActionSchema | None = None) -> spaces.Box:
    """Build the flat action ``Box`` from the enabled levers.

    Uses ``flatten_action_space_layout`` for offsets/width and each ``LeverSpec``
    for the per-dim bounds. In Phase 0.3 this is a 4-D Box ``[[0.5, 0.0, 0.0,
    0.0], [2.0, 1.0, 1.0, 1.0]]`` (price_index@0 on [0.5, 2.0], marketing@1 on
    [0.0, 1.0], assortment@2 on [0.0, 1.0], promotion@3 on [0.0, 1.0]); when
    further levers are enabled, continuous levers contribute their [low, high] and
    discrete levers contribute an unbounded logits block (argmax-decoded by
    ``action_wrapper``), all without changing this function.
    """
    if schema is None:
        schema = default_action_schema()
    layout = flatten_action_space_layout(schema)

    lows = np.empty(layout.total_dim, dtype=OBS_DTYPE)
    highs = np.empty(layout.total_dim, dtype=OBS_DTYPE)
    for slot in layout.levers:
        if slot.spec.kind is LeverKind.CONTINUOUS:
            assert slot.spec.low is not None and slot.spec.high is not None
            lows[slot.offset] = slot.spec.low
            highs[slot.offset] = slot.spec.high
        else:
            # Discrete lever: a masked logits block. Logits are unbounded reals
            # (decoded by argmax), so the Box spans the full float range here.
            lows[slot.offset : slot.offset + slot.width] = -_MONETARY_BOUND
            highs[slot.offset : slot.offset + slot.width] = _MONETARY_BOUND
    return spaces.Box(low=lows, high=highs, shape=(layout.total_dim,), dtype=OBS_DTYPE)
