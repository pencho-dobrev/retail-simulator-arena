"""The World facade — the ONLY place the world advances (THE SEAM).

``World`` ties the Wave 1 foundation (state/config/rng/encoding/schema) to the
Wave 2 economics (demand/accounting/reward/npc) into one canonical transition.
The Gymnasium single-agent view (Wave 4) and the future PettingZoo Parallel env
are thin adapters over :meth:`World.step`; keeping the transition here is what
makes training == live behavior identical.

Determinism contract (see ``core/rng.py``): a single injected PCG64
``numpy.Generator`` threaded explicitly, with the fixed per-tick draw order
honored exactly inside :meth:`World.step`:

    1. Demand resolution noise            (draw #1, in ``resolve_demand``)
    2. NPC actions, in agent-index order  (draw #2, in each ``NPCPolicy.act``)
    3. Accounting noise                   (draw #3, none in Phase 0.0)
    4. Perception noise                   (draw #4, Phase 1.0; one
       constant-size vectorized draw for the per-observer perceived competitor view,
       drawn after the next-state build. Its COUNT is fixed per tick (independent of
       research fidelity / which fields are noised) so draws #1–#3 — the whole economic
       trajectory — consume the identical stream they did in 0.5; the economics are
       byte-identical to 0.5 at the research default AND across research levels.)
    5. Coupled-wage write-off             (draw #5, LAST — Phase 4; ONE bounded
       ``rng.uniform(frac_min, frac_max, size=K)`` call of FIXED shape [K]. GATED:
       taken ONLY when ``WageConfig.writeoff_gate_active`` (the write-off mechanic is
       armed). When OFF (the default config) NO draw #5 occurs ⇒ draws #1–#4 — the whole
       existing trajectory — are byte-identical to pre-Phase-4. The affordability clamp
       + growth overhead are deterministic, no RNG.)

The Phase-4 AFFORDABILITY CLAMP (MECHANIC 1) is decided BEFORE draw #1 — the action is
decoded, the two capex gates run, and the discretionary levers are scaled to what the
retailer can fund, so demand resolves against the spend that was actually paid for
(CLAMP-T3; see ``core/affordability.py`` and :meth:`World.step`). It adds NO draw and
moves none: decode, gates and clamp are pure arithmetic.

``(WorldState, rng.snapshot())`` fully serializes the sim: a fresh ``World`` with
the restored RNG, fed the same actions, reproduces the trajectory bit-for-bit.

Purity: ``World`` keeps NO per-tick history and never mutates the ``WorldState``
passed to :meth:`step` — it returns a brand-new ``WorldState``. All randomness
flows through the injected ``Generator`` only. Episode boundaries
(``max_episode_steps``/truncation) and the discount factor are RL-wrapper
concerns and live OUTSIDE core: ``terminated``/``truncated`` are always ``False``
here.

Pure and framework-free: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from retail_simulator.core.accounting import apply_accounting
from retail_simulator.core.affordability import (
    compute_spend_clamp,
    gated_automation_capex,
    gated_expansion_capex,
    intended_discretionary_spend,
    scale_discretionary_levers,
)
from retail_simulator.core.config import CoreConfig, SeatSpec
from retail_simulator.core.demand import (
    _capacity_footprint,
    _decode_levers,
    _presence_matrix,
    _reference_prices,
    _retailer_price_indices,
    effective_warehouse_capacity,
    resolve_demand,
)
from retail_simulator.core.encoding import (
    decode_action,
    encode_observation,
    true_competitor_view,
)
from retail_simulator.core.npc import Balanced, Discounter, NPCPolicy, Premium, Wandering
from retail_simulator.core.reward import (
    RewardNormalizer,
    compute_reward_for_seat,
    reward_components,
)
from retail_simulator.core.rng import make_rng, snapshot
from retail_simulator.core.schema import (
    N_AUTOMATION_TIERS,
    N_REGIONS_MAX,
    ActionSchema,
    default_action_schema,
    validate_automation_width,
    validate_expansion_capacity,
)
from retail_simulator.core.state import (
    PerceivedCompetitor,
    RegionState,
    RetailerState,
    WorldState,
    default_segments,
)

# Phase 0.0 seating: index 0 is the learning agent, index 1 the discounter NPC.
AGENT_INDEX: int = 0


def default_seat_plan() -> tuple[SeatSpec, ...]:
    """The 0.0–0.4 two-seat plan: learning agent[0] + discounter NPC[1].

    ``world.reset`` uses this when ``CoreConfig.seats`` is absent so the default
    world stays BYTE-IDENTICAL to 0.4: a learning seat (present in region 0 only) and
    a discounter NPC (present in every region — the home-market rival + region
    incumbent). Presence is left ``None`` so the seat-kind defaults apply (the same
    region-0-only / all-regions presence the 0.4 reset hardcoded).
    """
    return (
        SeatSpec(is_npc=False, name="agent"),
        SeatSpec(is_npc=True, archetype="discounter", name="discounter"),
    )


def _make_npc_policy(archetype: str, schema: ActionSchema) -> NPCPolicy:
    """ """
    if archetype == "discounter":
        return Discounter(schema=schema)
    if archetype == "premium":
        return Premium(schema=schema)
    if archetype == "balanced":
        return Balanced(schema=schema)
    if archetype == "wandering":
        return Wandering(schema=schema)
    raise ValueError(f"unknown NPC archetype {archetype!r}")


def _default_seat_name(spec: SeatSpec, seat_index: int) -> str:
    """Derive a display name for a seat that did not supply one.

    A learning seat becomes ``f"retailer_{i}"`` (mirroring the PettingZoo agent ids);
    an NPC seat becomes ``f"{archetype}_{i}"``. Used only when ``SeatSpec.name`` is
    absent (the default plan names its seats explicitly, so the default world's names
    stay "agent"/"discounter").
    """
    if spec.is_npc:
        return f"{spec.archetype}_{seat_index}"
    return f"retailer_{seat_index}"


STARTING_CASH: float = 15_000.0
INITIAL_LOYALTY: float = 0.0
INITIAL_AWARENESS: float = 0.0
INITIAL_STOCKPILE: float = 0.0
INITIAL_SERVICE_SCORE: float = 1.0
INITIAL_AUTOMATION_TIER: int = 0


def compute_expansion_mask(
    state: WorldState, agent_index: int, config: CoreConfig
) -> npt.NDArray[np.bool_]:
    """ """
    me = state.retailers[agent_index]
    capex = config.expansion.expansion_capex
    n_regions = len(state.regions)
    mask = np.zeros(N_REGIONS_MAX, dtype=bool)
    mask[0] = True
    for r in range(1, n_regions):
        already_present = r < len(me.stores_per_region) and me.stores_per_region[r] > 0
        mask[r] = (me.cash >= capex) and (not already_present)
    return mask


def compute_automation_mask(
    state: WorldState, agent_index: int, config: CoreConfig
) -> npt.NDArray[np.bool_]:
    """ """
    me = state.retailers[agent_index]
    capex_per_tier = config.automation.capex_per_tier
    prev = me.automation_tier
    mask = np.zeros(N_AUTOMATION_TIERS, dtype=bool)
    for t in range(N_AUTOMATION_TIERS):
        monotone = t >= prev
        differential = capex_per_tier[t] - capex_per_tier[prev]
        affordable = me.cash >= differential
        mask[t] = monotone and affordable
    return mask


def _restructuring_survivor_regions(
    stores_per_region: tuple[int, ...],
    last_market_share_per_region: tuple[float, ...],
    n_regions: int,
    keep_stores: int,
) -> set[int]:
    """ """
    present = [r for r in range(1, n_regions) if stores_per_region[r] > 0]
    ranked = sorted(present, key=lambda r: (-last_market_share_per_region[r], r))
    return set(ranked[:keep_stores])


#
#
_PERCEIVED_FIELDS: tuple[tuple[str, str], ...] = (
    ("competitor_price", "price"),
    ("competitor_market_share", "unit"),
    ("competitor_awareness", "awareness"),
    ("competitor_assortment", "unit"),
    ("competitor_promotion", "unit"),
    ("competitor_loyalty_region_0", "unit"),
    ("competitor_awareness_region_1", "awareness"),
    ("competitor_loyalty_region_1", "unit"),
    ("competitor_market_share_region_1", "unit"),
    ("competitor_mean_price_region_0", "price"),
    ("competitor_mean_price_region_1", "price"),
)


def _bounds_for_kind(
    kind: str, price_bounds: tuple[float, float], awareness_cap: float
) -> tuple[float, float]:
    """ """
    if kind == "price":
        return price_bounds
    if kind == "awareness":
        return (0.0, awareness_cap)
    return (0.0, 1.0)  # "unit": shares / loyalty / assortment / promotion


def _perceived_field_bounds(config: CoreConfig) -> list[tuple[float, float]]:
    """ """
    price_bounds = (config.pricing.min_price_index, config.pricing.max_price_index)
    awareness_cap = 1.0 / config.marketing.decay
    return [
        _bounds_for_kind(kind, price_bounds, awareness_cap) for _name, kind in _PERCEIVED_FIELDS
    ]


_EXTRA_REGION_FIELD_KINDS: tuple[str, str, str, str] = ("awareness", "unit", "unit", "price")


def _extra_region_bounds(config: CoreConfig) -> list[tuple[float, float]]:
    """Per-column (low, high) bounds for ONE extra-region noised competitor block.

    Reused for every region r >= 2 (:func:`_seat_perceived_views` concatenates one
    copy per extra region) since the four kinds are region-index-independent.
    """
    price_bounds = (config.pricing.min_price_index, config.pricing.max_price_index)
    awareness_cap = 1.0 / config.marketing.decay
    return [
        _bounds_for_kind(kind, price_bounds, awareness_cap) for kind in _EXTRA_REGION_FIELD_KINDS
    ]


def _learning_seat_indices(state: WorldState) -> list[int]:
    """"""
    return [idx for idx, r in enumerate(state.retailers) if not r.is_npc]


def _true_reference_means(state: WorldState) -> npt.NDArray[np.float64]:
    """ """
    n_regions = len(state.regions)
    prices = _retailer_price_indices(state)  # [K], TRUE
    presence = _presence_matrix(state, n_regions)  # [R, K]
    return _reference_prices(prices, presence)  # [R, K]


def _reference_wages(state: WorldState) -> npt.NDArray[np.float64]:
    """The present-OTHERS' mean wage each retailer faces per region (Phase 4 MECHANIC 2).

    The wage analog of ``demand._reference_prices`` (sum-minus-self, self-excluded, pure
    read, no RNG): ``ref[r, k]`` is the mean ``wage_level`` of the OTHER retailers PRESENT
    in region ``r``. A sole-present (or absent) retailer falls back to its OWN wage so the
    rival-coupling term is penalty-free for a dominate (the same construction the
    reference-price helper uses). Reuses the EXACT ``_reference_prices`` arithmetic on the
    wage vector + presence, so the rival-coupling is the established cross-retailer read
    (the structural precedent the spec mandates). Returns a ``[R, K]`` tensor.
    """
    n_regions = len(state.regions)
    wages = np.array([r.wage_level for r in state.retailers], dtype=np.float64)  # [K]
    presence = _presence_matrix(state, n_regions)  # [R, K]
    return _reference_prices(wages, presence)  # [R, K]


def _observer_reference_overrides(
    ref: npt.NDArray[np.float64], obs_idx: int, n_regions: int
) -> dict[str, float]:
    """ """
    region_0 = float(ref[0, obs_idx])
    region_1 = float(ref[1, obs_idx]) if n_regions > 1 else region_0
    overrides = {
        "competitor_mean_price_region_0": region_0,
        "competitor_mean_price_region_1": region_1,
    }
    for r in range(2, n_regions):
        overrides[f"competitor_mean_price_region_{r}"] = float(ref[r, obs_idx])
    return overrides


def _seat_perceived_views(
    next_state: WorldState,
    research_by_seat: dict[int, float],
    config: CoreConfig,
    rng: np.random.Generator,
) -> None:
    """ """
    observers = _learning_seat_indices(next_state)
    n_obs = len(observers)
    n_regions = len(next_state.regions)
    n_extra_regions = max(0, n_regions - 2)
    n_fields = len(_PERCEIVED_FIELDS) + 4 * n_extra_regions
    noise = rng.standard_normal(size=(n_obs, n_fields))
    sigma_base = config.research.noise_sigma_base
    bounds = _perceived_field_bounds(config) + _extra_region_bounds(config) * n_extra_regions
    ref = _true_reference_means(next_state)  # [R, K]
    for row, obs_idx in enumerate(observers):
        fidelity = config.research.fidelity(research_by_seat.get(obs_idx, 0.0))
        scale = sigma_base * (1.0 - fidelity)
        true_view = true_competitor_view(next_state, obs_idx)
        # n2 (S2 senior-review follow-up): ``n_extra_regions`` (above, from THIS
        # function's own n_regions) and ``len(true_view.extra_regions)`` (from
        # ``_true_competitor_view``'s OWN, separately-computed n_regions arithmetic)
        # must never drift apart — both derive ``max(0, n_regions - 2)`` from the same
        # ``n_regions``, but via two independent call paths.
        assert len(true_view.extra_regions) == n_extra_regions, (
            f"true_competitor_view seated {len(true_view.extra_regions)} extra regions "
            f"but this seam expects {n_extra_regions} (n_regions={n_regions}) — the two "
            "derivations have drifted apart"
        )
        # The ref fields read the per-observer (self-excluded) demand reference, NOT
        # the 0.0 default on true_view; everything else reads the competitor scalar.
        ref_overrides = _observer_reference_overrides(ref, obs_idx, n_regions)
        noised: dict[str, float] = {}
        for col, (name, _kind) in enumerate(_PERCEIVED_FIELDS):
            true_value = ref_overrides.get(name, float(getattr(true_view, name)))
            low, high = bounds[col]
            value = true_value + float(noise[row, col]) * scale
            noised[name] = min(max(value, low), high)
        extra_regions: list[tuple[float, float, float, float]] = []
        base_col = len(_PERCEIVED_FIELDS)
        for i, (true_awareness, true_loyalty, true_share, _placeholder) in enumerate(
            true_view.extra_regions
        ):
            r = i + 2
            true_region_values = (
                true_awareness,
                true_loyalty,
                true_share,
                ref_overrides[f"competitor_mean_price_region_{r}"],
            )
            region_out = []
            for j in range(4):
                col = base_col + i * 4 + j
                low, high = bounds[col]
                value = true_region_values[j] + float(noise[row, col]) * scale
                region_out.append(min(max(value, low), high))
            extra_regions.append((region_out[0], region_out[1], region_out[2], region_out[3]))
        next_state.perceived_competitors[obs_idx] = PerceivedCompetitor(
            **noised, extra_regions=tuple(extra_regions)
        )
        next_state.research_fidelity[obs_idx] = fidelity


def _seat_true_perceived_views(state: WorldState, config: CoreConfig) -> None:
    """Reset placeholder: seat each observer's perceived view to the TRUE values (no draw).

    At reset no tick has run, so there is no perceived view yet (the analog of
    realized-demand being 0.0 at reset). Seat the perceived block to the TRUE competitor
    values (zero-noise placeholder) and ``research_fidelity`` to the documented reset
    placeholder 0.0 (no research has been spent). The reset path draws NO perception
    noise — preserving reset byte-identity. Mutates ``state`` in place (the fresh reset
    state). ``config`` is accepted for signature symmetry with the step seater (the true
    view needs no config).
    """
    del config
    n_regions = len(state.regions)
    ref = _true_reference_means(state)  # [R, K]
    n_extra_regions = max(0, n_regions - 2)
    for obs_idx in _learning_seat_indices(state):
        true_view = true_competitor_view(state, obs_idx)
        # n2 (S2 senior-review follow-up): the reset-path analog of the same guard in
        # ``_seat_perceived_views`` — ``true_competitor_view``'s own n_regions
        # arithmetic must never drift from this function's.
        assert len(true_view.extra_regions) == n_extra_regions, (
            f"true_competitor_view seated {len(true_view.extra_regions)} extra regions "
            f"but this seam expects {n_extra_regions} (n_regions={n_regions}) — the two "
            "derivations have drifted apart"
        )
        ref_overrides = _observer_reference_overrides(ref, obs_idx, n_regions)
        extra_regions = tuple(
            (awareness, loyalty, share, ref_overrides[f"competitor_mean_price_region_{i + 2}"])
            for i, (awareness, loyalty, share, _placeholder) in enumerate(true_view.extra_regions)
        )
        state.perceived_competitors[obs_idx] = replace(
            true_view,
            competitor_mean_price_region_0=ref_overrides["competitor_mean_price_region_0"],
            competitor_mean_price_region_1=ref_overrides["competitor_mean_price_region_1"],
            extra_regions=extra_regions,
        )
        state.research_fidelity[obs_idx] = 0.0


@dataclass(frozen=True)
class StepResult:
    """Outcome of one :meth:`World.step`, keyed by retailer (agent-index) order.

    Wave 4's ``RetailEnv`` wraps this: it reads ``observations[AGENT_INDEX]`` and
    ``rewards[AGENT_INDEX]`` for the single-agent Gym view and merges
    ``infos[AGENT_INDEX]`` into the env ``info`` dict. PettingZoo (0.5) consumes
    the full per-retailer dicts directly.

    * ``next_state`` — the new ``WorldState`` (a fresh object; the input is never
      mutated). Pair with ``World.rng_state`` to fully serialize the sim.
    * ``observations`` — per-retailer full-information observation vectors
      (``encode_observation``), float32.
    * ``rewards`` — per-retailer reward (raw profit in 0.0).
    * ``infos`` — per-retailer info: the un-normalized reward ``components`` and
      an ``action_mask`` placeholder (``None`` in 0.0: the action is pure
      continuous pricing, so there is nothing to mask).
    * ``terminated`` / ``truncated`` — per-retailer flags, ALWAYS ``False``: the
      world is continuing and truncation is a wrapper concern (do not implement
      ``TimeLimit`` here).
    """

    next_state: WorldState
    observations: dict[int, npt.NDArray[np.float32]]
    rewards: dict[int, float]
    infos: dict[int, dict[str, object]]
    terminated: dict[int, bool] = field(default_factory=dict)
    truncated: dict[int, bool] = field(default_factory=dict)


class World:
    """ """

    def __init__(self, config: CoreConfig | None = None, seed: int | None = None) -> None:
        self._config = config if config is not None else CoreConfig.default()
        self._schema: ActionSchema = default_action_schema()
        regions = self._config.demand.regions
        n_regions = len(regions) if regions else 1
        validate_expansion_capacity(n_regions)
        # Phase 1.2: the automation lever's discrete width MUST equal the configured
        # ``n_tiers``. Asserted at construction (a mismatch is a config error;
        # ``AutomationConfig.__post_init__`` also size-checks the per-tier arrays).
        validate_automation_width(len(self._config.automation.capex_per_tier))
        if STARTING_CASH < self._config.expansion.expansion_capex:
            raise ValueError(
                f"STARTING_CASH ({STARTING_CASH}) must be >= expansion_capex "
                f"({self._config.expansion.expansion_capex}) or the expansion lever is dead"
            )
        self._rng: np.random.Generator = make_rng(seed)
        # Resolve the seat plan (absent => the 0.4 two-seat plan, byte-identical) and
        # build one NPC policy per NPC seat, keyed by seat index. Seat 0 must be a
        # learning seat (AGENT_INDEX = 0 for the Gym view). Each scripted policy is
        # deterministic (consumes no RNG) but is invoked in seat order in step() so the
        # draw-order contract holds when a stochastic NPC arrives later.
        self._seat_plan: tuple[SeatSpec, ...] = (
            self._config.seats if self._config.seats is not None else default_seat_plan()
        )
        if not self._seat_plan:
            raise ValueError("seat plan must have at least one seat")
        if self._seat_plan[0].is_npc:
            raise ValueError("seat 0 must be a learning seat (AGENT_INDEX = 0)")
        self._npc_policies: dict[int, NPCPolicy] = {
            idx: _make_npc_policy(spec.archetype, self._schema)
            for idx, spec in enumerate(self._seat_plan)
            if spec.is_npc and spec.archetype is not None
        }

    @property
    def config(self) -> CoreConfig:
        """The resolved configuration backing this world."""
        return self._config

    @property
    def rng(self) -> np.random.Generator:
        """The injected generator (exposed so wrappers can serialize it)."""
        return self._rng

    def rng_state(self) -> dict[str, object]:
        """Serializable RNG state; pair with a ``WorldState`` to checkpoint."""
        return snapshot(self._rng)

    def reset(self, seed: int | None = None) -> WorldState:
        """Build the initial ``WorldState``; re-seed the RNG when ``seed`` given.

        Returns the ``WorldState`` (not an ``(obs, info)`` tuple) so the core
        stays transport-agnostic; Wave 4's Gym ``reset`` calls
        :meth:`agent_observation` on the result to produce Gymnasium's
        ``(obs, info)``. Re-seeding here makes ``World(seed=s).reset()`` and
        ``World().reset(seed=s)`` start identical trajectories.
        """
        if seed is not None:
            self._rng = make_rng(seed)

        # Build the regions from config (Phase 0.4): n regions from
        # demand.regions when non-empty (region 0 full, region 1 asymmetric), else a
        # single region from the scalar fields (the 0.3 fallback path). Realized
        # demand is 0.0 and stockpile empty until a tick runs.
        region_cfgs = self._config.demand.regions
        regions: list[RegionState]
        if region_cfgs:
            regions = [
                RegionState(
                    population=rc.base_regional_demand,
                    base_regional_demand=rc.base_regional_demand,
                    segment_mix=dict(rc.segment_mix),
                    realized_regional_demand=0.0,
                    stockpile=INITIAL_STOCKPILE,
                )
                for rc in region_cfgs
            ]
        else:
            regions = [
                RegionState(
                    population=self._config.demand.base_regional_demand,
                    base_regional_demand=self._config.demand.base_regional_demand,
                    segment_mix=dict(self._config.demand.segment_mix),
                    realized_regional_demand=0.0,
                    stockpile=INITIAL_STOCKPILE,
                )
            ]
        n_regions = len(regions)
        zeros = (0.0,) * n_regions

        retailers: list[RetailerState] = []
        initial_service_score = tuple(INITIAL_SERVICE_SCORE for _ in range(n_regions))
        for idx, spec in enumerate(self._seat_plan):
            if spec.presence is not None:
                stores = tuple(
                    spec.presence[r] if r < len(spec.presence) else 0 for r in range(n_regions)
                )
            elif spec.is_npc:
                stores = tuple(1 for _ in range(n_regions))
            else:
                stores = tuple(1 if r == 0 else 0 for r in range(n_regions))
            ramp_ticks = self._config.warehouse.ramp_ticks
            initial_store_age = tuple(ramp_ticks if stores[r] > 0 else 0 for r in range(n_regions))
            retailers.append(
                RetailerState(
                    name=spec.name if spec.name is not None else _default_seat_name(spec, idx),
                    is_npc=spec.is_npc,
                    cash=STARTING_CASH,
                    price_index=self._config.pricing.baseline_price_index,
                    stores_per_region=stores,
                    store_age_per_region=initial_store_age,
                    awareness_per_region=zeros,
                    loyalty_per_region=zeros,
                    last_market_share_per_region=zeros,
                    service_score_per_region=initial_service_score,
                    wage_level=self._config.wage.wage_base,
                    employee_happiness=1.0,
                    accumulated_overhead_basis=0.0,
                    last_turnover=0.0,
                    warehouse_capacity=self._config.warehouse.default_warehouse_capacity,
                    # Phase 4 MECHANIC 4 (RESTRUCTURING): no seat starts insolvent or
                    # restructured — both counters seat at 0 regardless of whether the
                    # gate is armed (the wage_level/happiness/overhead/turnover block's
                    # own "seat the no-history anchor explicitly" convention above).
                    insolvent_ticks=0,
                    restructurings=0,
                    # NFD-1: no seat holds a pending action at reset — no policy has run
                    # yet (draw #2 lives inside ``step``, never here). So every NPC HOLDS
                    # COURSE on tick 0, decoding the registry defaults exactly as it
                    # always did. That is the byte-identity anchor for the default plan.
                    pending_action=None,
                )
            )

        reward_stats: dict[int, RewardNormalizer] = {}
        if self._config.reward.mode == "weighted":
            reward_stats = {
                idx: RewardNormalizer() for idx, r in enumerate(retailers) if not r.is_npc
            }

        segments = (
            dict(self._config.segments) if self._config.segments is not None else default_segments()
        )
        state = WorldState(
            tick=0,
            regions=regions,
            retailers=retailers,
            segments=segments,
            reward_stats=reward_stats,
        )
        # Phase 1.0: seat each observer's PERCEIVED competitor view to the TRUE values
        # (zero-noise placeholder) + research_fidelity 0.0 — NO perception draw at reset
        # (no tick has run; the analog of realized-demand being 0.0 at reset). This keeps
        # reset byte-identity and gives encode_observation a valid seated block to read.
        _seat_true_perceived_views(state, self._config)
        return state

    def agent_observation(
        self, state: WorldState, agent_index: int = AGENT_INDEX
    ) -> tuple[npt.NDArray[np.float32], dict[str, object]]:
        """ """
        obs = encode_observation(state, agent_index)
        n_regions = len(state.regions)
        warehouse_cfg = self._config.warehouse
        reset_footprint = _capacity_footprint(state, n_regions, warehouse_cfg)
        info: dict[str, object] = {
            "action_mask": {
                "expansion": compute_expansion_mask(state, agent_index, self._config),
                "automation": compute_automation_mask(state, agent_index, self._config),
            },
            # CLAMP-T3: the same DX key ``step`` emits, so the contract does not have a
            # hole at reset. 1.0 is the truth here — no tick has run, so no spend has
            # been cut short.
            "spend_clamp": 1.0,
            # C-wage-probe-decomposition: the same DX key ``step`` emits, so the
            # contract does not have a hole at reset here either. 0.0 is the truth —
            # no tick has run, so no write-off was taken.
            "writeoff": 0.0,
            "mature_store_equivalents": float(reset_footprint[agent_index]),
            # Phase 4 MECHANIC 4 (RESTRUCTURING): the same two DX keys ``step`` emits,
            # so the contract does not have a hole at reset either. No tick has run,
            # so no seat has ever been restructured and its seated insolvency counter
            # starts at 0 (``RetailerState``'s own reset value) regardless of whether
            # the gate is armed.
            "restructured": False,
            "insolvent_ticks": int(state.retailers[agent_index].insolvent_ticks),
        }
        return obs, info

    def step(
        self, state: WorldState, joint_action: dict[int, npt.NDArray[np.float32]]
    ) -> StepResult:
        """ """
        # 1. DECODE THE JOINT ACTION — ONCE per tick, before anything else (CLAMP-T3).
        #    ``decode_action``'s boundary validation + clip live inside; a malformed
        #    action raises here. Absent retailers fall back to their seated state price
        #    and the per-lever registry defaults. Decoding is pure and consumes NO RNG,
        #    so hoisting it out of ``resolve_demand`` (where it used to run) leaves the
        #    draw order and the trajectory untouched — it only makes the decoded levers
        #    available to the capex gates and the affordability clamp below, which must
        #    run BEFORE demand.
        decoded_levers = _decode_levers(state, joint_action, self._schema)

        #
        #    ``gated_expansion_capex`` returns the per-retailer capex + the presence to
        #    seat; ``gated_automation_capex`` returns the capex, the tier to seat, and
        #    the ``reported_target`` (the decoded target AFTER its defensive
        #    out-of-range clamp but BEFORE the affordability mask — "what did you ask
        #    for", as opposed to "what did you get"). ``last_expansion`` records the
        #    CHOICE (even a no-op) for reporting.
        n_regions = len(state.regions)
        capex, next_stores_per_region = gated_expansion_capex(
            state, decoded_levers.expansion, self._config
        )
        last_expansion: list[float] = [
            float(decoded_levers.expansion[idx]) for idx in range(len(state.retailers))
        ]
        automation_capex, new_automation_tier_arr, reported_target_arr = gated_automation_capex(
            state, decoded_levers.automation_tier, self._config
        )
        # Cast back to plain Python-int lists (the pre-T1 loop's own types) so every
        # downstream consumer (the RetailerState seat below) sees the identical types
        # it always has — a numpy int64 element is numerically equal but a different
        # type, and RetailerState.automation_tier / .last_automation_tier_action are
        # both declared ``int``.
        new_automation_tier: list[int] = [int(t) for t in new_automation_tier_arr]
        last_automation_tier_action: list[int] = [int(t) for t in reported_target_arr]

        # 3. THE AFFORDABILITY CLAMP (MECHANIC 1; deterministic, NO RNG). Gated on
        #    ``cash_budget_enabled``: when OFF (the default) ``levers`` stays the raw
        #    decode and ``spend_clamp`` stays all-ones, so demand/accounting/seats are
        #    bit-for-bit what they were pre-CLAMP-T3.
        wage_cfg = self._config.wage
        warehouse_cfg = self._config.warehouse
        n_retailers = len(state.retailers)

        #
        # (Senior-review fix, fix/tier-b-core-review: gating additionally on
        # ``np.any(s_t != 0.0)`` looked reasonable but was wrong, because the NOT-armed
        # branch re-reads the SEATED ``wage_level`` — which the ARMED branch re-seats to
        # last tick's PAID wage, not ``wage_base``. That made a submitted
        # ``wage_spend=0.0`` STICKY, CLIFFY and RIVAL-DEPENDENT. Gating on the scale knob
        # alone makes every seat's paid wage a continuous function of its OWN ``s_t[k]``
        # only.)
        #
        # CLAMP-T3 extends that same reasoning to the clamped path: once the paid wage
        # can be CUT SHORT by the clamp and the CUT value is what gets seated as
        # ``wage_level``, reading the seat back as next tick's INTENT would ratchet the
        # wage policy permanently downward after a single cash crunch (and would make
        # an armed-vs-unarmed twin at ``wage_spend = 0`` diverge, since the armed branch
        # recomputes ``wage_base`` while the unarmed one would compound the clamp). So
        # under the clamp the INTENDED wage is always the policy formula
        # ``wage_base + s_t · scale`` — "base + Tier-B premium" — and only the PAID wage
        # (formula × clamp) is seated. At ``wage_spend_scale == 0`` the formula is
        # exactly ``wage_base``, which is what the seated read returns on every
        # unclamped tick, so this changes nothing until a clamp actually bites. (Edge
        # case, documented: a hand-built state that seats a ``wage_level`` other than
        # ``wage_base`` under an armed clamp is billed the policy wage, not its seeded
        # one. No production config does that — ``reset`` seats ``wage_base``.)
        wage_spend_t = decoded_levers.wage_spend  # [K], s_t
        armed_t = wage_cfg.wage_spend_scale != 0.0
        intended_paid_wage = wage_cfg.wage_base + wage_spend_t * wage_cfg.wage_spend_scale
        # Is the wage line billed at all? (Unchanged gate: a zero wage_base with no
        # seated wage anywhere and an unarmed scale means there is no wage cost line —
        # ``None`` keeps accounting's float-op order untouched on that path.)
        wage_line_live = (
            wage_cfg.wage_base != 0.0
            or any(r.wage_level != 0.0 for r in state.retailers)
            or armed_t
        )

        cash_t = np.array([r.cash for r in state.retailers], dtype=np.float64)
        committed_capex = capex + automation_capex
        levers = decoded_levers
        spend_clamp = np.ones(n_retailers, dtype=np.float64)
        intended_spend = np.zeros(n_retailers, dtype=np.float64)
        if wage_cfg.cash_budget_enabled:
            intended_spend = intended_discretionary_spend(
                decoded_levers, intended_paid_wage, self._config
            )
            spend_clamp = compute_spend_clamp(
                cash_t, intended_spend, committed_capex, credit_limit=wage_cfg.credit_limit
            )
            levers = scale_discretionary_levers(decoded_levers, spend_clamp)

        # 4. resolve_demand (draw #1): the demand noise, resolved against the levers
        #    decided above — the CLAMPED ones when the budget gate is on, the raw decode
        #    otherwise. ``DemandResult`` carries those same lever SoTs, so every
        #    downstream consumer (accounting's money, the awareness/fidelity/stockpile/
        #    capacity seats) reads the value that was actually paid for.
        demand_result = resolve_demand(
            state,
            joint_action,
            self._rng,
            self._config,
            self._schema,
            levers=levers,
            spend_clamp=spend_clamp,
        )

        # 5. NPC actions for the NEXT tick, in agent-index order (draw #2). The scripted
        #    archetypes consume no RNG, but we call them in-order so the contract holds
        #    when a stochastic NPC arrives.
        #
        #
        #
        #    ``npc_prices`` still exists because an NPC's price seat is its NEXT price
        #    (the value ``state.price_index`` carries into the following tick, and what
        #    rivals perceive), not the price it just charged — the pre-existing seat-kind
        #    asymmetry, deliberately unchanged here.
        npc_prices: dict[int, float] = {}
        npc_pending: dict[int, tuple[float, ...]] = {}
        for idx, retailer in enumerate(state.retailers):
            if retailer.is_npc:
                policy = self._npc_policies[idx]
                flat = policy.act(state, self._rng, idx)
                # Decoded ONCE here only for the price seat; the pending tuple keeps the
                # RAW flat action so next tick's decode is the seat's single source of
                # truth for all eleven levers (one decode path, no lever-by-lever copy).
                npc_prices[idx] = decode_action(flat, self._schema).levers["price_index"]
                npc_pending[idx] = tuple(float(x) for x in flat)

        paid_wage_t: npt.NDArray[np.float64]
        if wage_cfg.cash_budget_enabled:
            paid_wage_t = intended_paid_wage * spend_clamp
        elif armed_t:
            paid_wage_t = intended_paid_wage
        else:
            paid_wage_t = np.array([r.wage_level for r in state.retailers], dtype=np.float64)

        wage_bill_arr: npt.NDArray[np.float64] | None = paid_wage_t if wage_line_live else None
        overhead_arr: npt.NDArray[np.float64] | None = None
        if wage_cfg.overhead_superlinear_coeff != 0.0:
            stores_total_arr = np.array(
                [sum(r.stores_per_region) for r in state.retailers], dtype=np.float64
            )
            excess = np.maximum(0.0, stores_total_arr - wage_cfg.overhead_threshold_stores)
            if wage_cfg.overhead_reference_stores == 1.0:
                overhead_arr = wage_cfg.overhead_superlinear_coeff * (
                    excess**wage_cfg.overhead_exponent
                )
            else:
                overhead_arr = wage_cfg.overhead_superlinear_coeff * (
                    (excess / wage_cfg.overhead_reference_stores) ** wage_cfg.overhead_exponent
                )

        invest_t = levers.warehouse_invest  # [K], v_t
        warehouse_spend_arr: npt.NDArray[np.float64] | None = None
        if warehouse_cfg.cost_per_unit_invest != 0.0 and bool(np.any(invest_t != 0.0)):
            warehouse_spend_arr = invest_t * warehouse_cfg.cost_per_unit_invest

        accounting = apply_accounting(
            state,
            demand_result,
            self._config,
            capex=capex,
            automation_capex=automation_capex,
            wage_bill=wage_bill_arr,
            overhead=overhead_arr,
            warehouse_spend=warehouse_spend_arr,
        )

        next_reward_stats: dict[int, RewardNormalizer] = {}

        # 9. Build the NEXT WorldState: advance tick; seat new cash; SEAT per-region
        #    awareness (build-where-present, the lagged recurrence), per-region
        #    loyalty (from the accrual), per-region stores (from the gate above);
        #    write reporting last_*; write each NPC's chosen price. The agent's
        #    price_index for next tick is the price it just acted on. Assortment is
        #    memoryless; promotion is dual (memoryless lift/cost + the seated
        #    stockpile debt seated per-region below). No history is kept.
        decay = self._config.marketing.decay
        awareness_cap = 1.0 / decay
        market_share = accounting.market_share  # [R, K]
        new_loyalty = accounting.new_loyalty  # [R, K]

        ref_wage = _reference_wages(state)  # [R, K], present-others' mean wage
        ref_wage_k = ref_wage.mean(axis=0)  # [K]
        own_share_k = market_share.mean(axis=0)  # [K]
        required_wage = (
            wage_cfg.wage_base
            + wage_cfg.wage_own_share_coeff * own_share_k
            + wage_cfg.wage_coupling_coeff * ref_wage_k
        )  # [K]
        happiness_decay = wage_cfg.happiness_decay

        seated_capacity = np.array(
            [r.warehouse_capacity for r in state.retailers], dtype=np.float64
        )  # [K], capacity_t (SEATED, pre-update — the value demand read this tick)
        warehouse_capacity_next = seated_capacity.copy()
        if warehouse_cfg.capacity_per_unit_invest != 0.0:
            invest_mask = invest_t != 0.0
            warehouse_capacity_next[invest_mask] = (
                seated_capacity[invest_mask]
                + invest_t[invest_mask] * warehouse_cfg.capacity_per_unit_invest
            )

        #
        # 24-region arena support: utilization reads the SAME effective (provided-
        # capacity-topped-up) capacity ``demand.py``'s footprint-share split just
        # served against — ``effective_warehouse_capacity`` is the ONE definition, fed
        # by the SAME ``_footprint_totals`` (also n_regions-bounded) as that split and
        # the under-provisioning shortfall below, so none of the three can disagree
        # (senior-review M1/m1 — an earlier draft computed this footprint sum
        # separately here via the unbounded ``sum(stores_per_region)``, which silently
        # DISAGREED with ``demand.py``'s n_regions-bounded version whenever a
        # retailer's tuple was longer than n_regions). Inert at the
        # ``provided_capacity_per_store == 0.0`` default: the helper's ``!= 0.0`` guard
        # returns ``seated_capacity`` UNCHANGED — the SAME array object, so
        # ``effective_capacity_t`` ALIASES ``seated_capacity`` at the default and must
        # never be written in place (an in-place mutation would corrupt the seated
        # value the M2 shortfall / next tick's carry-forward still read).
        #
        stores_total_t = _capacity_footprint(state, n_regions, warehouse_cfg)  # [K]
        effective_capacity_t = effective_warehouse_capacity(
            self._config, seated_capacity, stores_total_t
        )  # [K]
        served_t = demand_result.units.sum(axis=0)  # [K]
        finite_positive_capacity = np.isfinite(effective_capacity_t) & (effective_capacity_t > 0.0)
        utilization_t = np.zeros_like(effective_capacity_t)
        utilization_t[finite_positive_capacity] = np.minimum(
            served_t[finite_positive_capacity] / effective_capacity_t[finite_positive_capacity],
            1.0,
        )

        ramp_ticks = warehouse_cfg.ramp_ticks
        next_retailers: list[RetailerState] = []
        for idx, retailer in enumerate(state.retailers):
            #
            # What this seat is FOR changed at NFD-1, though. From tick 1 on it is no
            # longer what demand charges the scripted seat: ``_decode_levers`` prefers
            # the pending action, whose ``price_index`` comes from the very same
            # ``policy.act`` call as ``npc_prices[idx]``, so the two agree numerically and
            # the pending one wins. The seat now survives as the source the PERCEIVED obs
            # block reads for ``competitor_price`` (and as the tick-0 hold-course price,
            # before any pending action exists). Keep them in sync: a future policy that
            # seated a price different from the one it made pending would show the human
            # one number and charge another.
            if retailer.is_npc:
                next_price = npc_prices[idx]
            elif idx in joint_action:
                next_price = float(demand_result.price_indices[idx])
            else:
                next_price = retailer.price_index

            # NFD-1: the discretionary reporting seats now read off ``demand_result``
            # for EVERY seat kind — the CLAMPED effective values actually played this
            # tick — because ``_decode_levers`` fed an NPC's from its pending action.
            # (Previously an NPC reported the RAW values of the action decided this tick
            # FOR NEXT tick, which is why its obs slot led its economics by a tick and
            # showed money it might not be able to fund.) A seat that played nothing —
            # an absent learning seat, or an NPC on its hold-course first tick — carries
            # the registry defaults here, i.e. the same 0.0 the old explicit branch set.
            spend = float(demand_result.spend_fractions[idx])
            acted_assortment = float(demand_result.assortment[idx])
            acted_promotion = float(demand_result.promotion[idx])

            stores_next = next_stores_per_region[idx]

            if ramp_ticks > 0:
                prior_ages = retailer.store_age_per_region
                prior_stores = retailer.stores_per_region
                age_next: list[int] = []
                for r in range(n_regions):
                    prior_age = prior_ages[r] if r < len(prior_ages) else 0
                    was_present = r < len(prior_stores) and prior_stores[r] > 0
                    if stores_next[r] <= 0:
                        age_next.append(prior_age)  # never opened (or not yet)
                    elif was_present:
                        age_next.append(min(prior_age + 1, ramp_ticks))  # ages, capped
                    else:
                        age_next.append(0)  # opened THIS tick — the speed tax starts at 0
                store_age_next: tuple[int, ...] = tuple(age_next)
            else:
                store_age_next = retailer.store_age_per_region

            awareness_next: list[float] = []
            for r in range(n_regions):
                prior = (
                    retailer.awareness_per_region[r]
                    if r < len(retailer.awareness_per_region)
                    else 0.0
                )
                build = spend if stores_next[r] > 0 else 0.0
                awareness_next.append(min(max((1.0 - decay) * prior + build, 0.0), awareness_cap))

            # Per-region loyalty from the accrual (the moat advancing, built from this
            # tick's per-region share; read next tick by demand).
            loyalty_next = tuple(float(new_loyalty[r, idx]) for r in range(n_regions))
            share_next = tuple(float(market_share[r, idx]) for r in range(n_regions))

            #
            #   service_score_next[r] = clip(
            #       (1 - decay) * prev[r]
            #         + decay * (1 - lost_sales_share_penalty * stockout_rate[r, k]),
            #       0.0, 1.0
            #   )
            #
            # At ``stockout_rate = 0`` and ``prev = 1.0`` the next value is EXACTLY 1.0
            # (the byte-identity fixed point; the value the lever-default produces). At
            # any positive stockout_rate the EMA pulls the score down toward
            # ``1 - lost_sales_share_penalty * stockout_rate`` and recovers toward 1.0 at
            # the ``service_score_decay`` rate when stockouts stop.
            scm_cfg = self._config.scm
            service_decay = scm_cfg.service_score_decay
            lost_penalty = scm_cfg.lost_sales_share_penalty
            service_score_next: list[float] = []
            for r in range(n_regions):
                prior = (
                    retailer.service_score_per_region[r]
                    if r < len(retailer.service_score_per_region)
                    else INITIAL_SERVICE_SCORE
                )
                sr = float(demand_result.stockout_rate[r, idx])
                ema = (1.0 - service_decay) * prior + service_decay * (1.0 - lost_penalty * sr)
                service_score_next.append(min(max(ema, 0.0), 1.0))

            paid_wage = float(paid_wage_t[idx])
            shortfall = paid_wage - float(required_wage[idx])
            if wage_cfg.wage_shortfall_scale != 1.0:
                shortfall = shortfall / wage_cfg.wage_shortfall_scale
            happiness_target = min(max(1.0 + shortfall, 0.0), 1.0)
            happiness_next = min(
                max(
                    (1.0 - happiness_decay) * retailer.employee_happiness
                    + happiness_decay * happiness_target,
                    0.0,
                ),
                1.0,
            )

            next_retailers.append(
                RetailerState(
                    name=retailer.name,
                    is_npc=retailer.is_npc,
                    cash=float(accounting.new_cash[idx]),
                    price_index=next_price,
                    stores_per_region=stores_next,
                    store_age_per_region=store_age_next,
                    awareness_per_region=tuple(awareness_next),
                    loyalty_per_region=loyalty_next,
                    last_profit=float(accounting.profit[idx]),
                    last_revenue=float(accounting.revenue[idx]),
                    # Aggregate (region-0) reporting share for the legacy obs slot.
                    last_market_share=float(market_share[0, idx]),
                    last_market_share_per_region=share_next,
                    last_assortment=acted_assortment,
                    last_promotion=acted_promotion,
                    # Reporting-only (Phase 0.4): the expansion choice acted this tick
                    # (the choice, even if it no-opped) surfaced in next-tick obs.
                    last_expansion=last_expansion[idx],
                    # Phase 1.1: the new seated per-region service_score (the EMA above).
                    service_score_per_region=tuple(service_score_next),
                    # Phase 1.2 (F3 = PERMANENT): the new seated per-retailer
                    # ``automation_tier`` integer (the gate's resolution — either the
                    # target on a valid upgrade or the prior tier on a no-op). The
                    # SoT the next-tick's demand reads (PRE-update / lagged) to apply
                    # the COGS multiplier via ``savings_per_tier[automation_tier_
                    # prev]``. Reporting-only ``last_automation_tier_action`` records
                    # the DECODED TARGET tier (before the mask) — surfaced in info /
                    # metrics for the gate's uptake measurement.
                    automation_tier=new_automation_tier[idx],
                    last_automation_tier_action=last_automation_tier_action[idx],
                    wage_level=paid_wage,
                    employee_happiness=happiness_next,
                    accumulated_overhead_basis=float(accounting.overhead[idx]),
                    last_turnover=float(accounting.revenue[idx]),
                    warehouse_capacity=float(warehouse_capacity_next[idx]),
                    last_warehouse_utilization=float(utilization_t[idx]),
                    last_stockout_rate=float(demand_result.lost_sales_fraction[idx]),
                    # Phase 4 MECHANIC 4 (RESTRUCTURING): carry BOTH counters forward
                    # VERBATIM here (the ``wage_level``/``warehouse_capacity`` seat-
                    # forward-then-mutate-in-place precedent) — step 12b below is the
                    # ONLY place either advances, gated on ``restructuring_gate_active``.
                    # At the byte-identity default this is the sole read of either field
                    # every tick: a plain carry, no arithmetic.
                    insolvent_ticks=retailer.insolvent_ticks,
                    restructurings=retailer.restructurings,
                    pending_action=npc_pending.get(idx),
                )
            )

        decay_s = self._config.promotion.stockpile_decay
        build_s = self._config.promotion.stockpile_build
        cap_s = 1.0 / decay_s
        promotion = demand_result.promotion  # [K]
        shares = demand_result.shares  # [R, K]
        next_regions = []
        for r, region in enumerate(state.regions):
            promo_agg_r = float(np.dot(promotion, shares[r]))
            next_regions.append(
                RegionState(
                    population=region.population,
                    base_regional_demand=region.base_regional_demand,
                    segment_mix=dict(region.segment_mix),
                    realized_regional_demand=float(demand_result.total_demand[r]),
                    stockpile=min(
                        max((1.0 - decay_s) * region.stockpile + build_s * promo_agg_r, 0.0),
                        cap_s,
                    ),
                )
            )

        next_state = WorldState(
            tick=state.tick + 1,
            regions=next_regions,
            retailers=next_retailers,
            segments=dict(state.segments),
            # Seat the advanced per-seat normalizers (the SINGLE seat site, like every
            # other seated dynamic). Empty for mode="profit" (the normalizer is never
            # advanced — the profit reward stays byte-identical).
            reward_stats=next_reward_stats,
        )

        # 10. DRAW #4 (NEW, LAST): the perception noise + seat the per-observer perceived
        #     competitor view + research_fidelity on next_state. Drawn AFTER the
        #     next-state build (so it noises the TRUE next-state competitor scalars the
        #     agent would have seen with perfect info) and AFTER all economic draws, as
        #     ONE constant-size vectorized call whose COUNT is independent of fidelity
        #     (fidelity scales magnitude only). research per observer is the decoded SoT
        #     on demand_result. Because #4 is LAST + constant-size, draws #1–#3 (the
        #     economic trajectory) are byte-identical to 0.5 at every research level.
        research_by_seat = {
            idx: float(demand_result.research[idx]) for idx in range(len(state.retailers))
        }
        _seat_perceived_views(next_state, research_by_seat, self._config, self._rng)

        writeoff_by_seat: dict[int, float] = {}
        if wage_cfg.writeoff_gate_active:
            n_ret = len(state.retailers)
            draw = self._rng.uniform(
                wage_cfg.writeoff_frac_min, wage_cfg.writeoff_frac_max, size=n_ret
            )
            for idx in range(n_ret):
                seated = next_state.retailers[idx]
                severity = wage_cfg.writeoff_severity_scale * (1.0 - seated.employee_happiness)
                prior_turnover = state.retailers[idx].last_turnover
                size = float(sum(seated.stores_per_region))
                leader_factor = size ** (wage_cfg.leader_weight_exponent - 1.0)
                writeoff = float(draw[idx]) * severity * prior_turnover * leader_factor
                if writeoff != 0.0:
                    seated.last_profit -= writeoff
                    seated.cash -= writeoff
                    writeoff_by_seat[idx] = writeoff

        penalty_by_seat: dict[int, float] = {}
        if warehouse_cfg.penalty_gate_active:
            n_ret = len(state.retailers)
            # INTENTIONAL: one fixed-shape [K] draw covering ALL seats whenever the gate is
            # armed (even seats with inf capacity, whose severity resolves to 0). Drawing for
            # all seats keeps the RNG cursor advance deterministic regardless of which seats
            # are under-provisioned — mirrors the draw-#5 write-off gate.
            draw = self._rng.uniform(
                warehouse_cfg.penalty_frac_min, warehouse_cfg.penalty_frac_max, size=n_ret
            )
            # 24-region arena support (senior-review M2): the "how much you actually
            # HAVE" side of the shortfall must read the SAME effective (provided-
            # capacity-topped-up) value the throughput cap and the utilization
            # readout use — one mechanic must not hold two readings of the same
            # state. Hoisted inside this gate so the default (``penalty_severity_
            # scale == 0.0``) path gains no extra float ops and the draw order is
            # untouched. Both sides stay on the NEXT state, as before.
            seated_capacity_next = np.array(
                [r.warehouse_capacity for r in next_state.retailers], dtype=np.float64
            )  # [K]
            stores_total_next = _capacity_footprint(next_state, n_regions, warehouse_cfg)  # [K]
            effective_next = effective_warehouse_capacity(
                self._config, seated_capacity_next, stores_total_next
            )  # [K]
            for idx in range(n_ret):
                seated = next_state.retailers[idx]
                # The REQUIRED side keeps the pre-existing, UNBOUNDED footprint sum
                # (matching the growth-overhead mechanic's own convention for a
                # "how big should this be" computation) — only the ACTUAL/provided
                # side above switches to the n_regions-bounded, effective-capacity
                # reading. See ``WarehouseConfig.provided_capacity_per_store``'s
                # docstring for the resulting `capacity_per_store >
                # provided_capacity_per_store` residual requirement.
                stores_total = float(sum(seated.stores_per_region))
                required_capacity = warehouse_cfg.capacity_per_store * stores_total
                shortfall = max(0.0, required_capacity - float(effective_next[idx]))
                severity = warehouse_cfg.penalty_severity_scale * (
                    shortfall / (shortfall + warehouse_cfg.penalty_halfsat)
                )
                prior_turnover = state.retailers[idx].last_turnover
                penalty = float(draw[idx]) * severity * prior_turnover
                if penalty != 0.0:
                    seated.last_profit -= penalty
                    seated.cash -= penalty
                    penalty_by_seat[idx] = penalty

        #
        #
        #
        restructured_by_seat: dict[int, bool] = {}
        if wage_cfg.restructuring_gate_active:
            debt_line = wage_cfg.credit_limit + wage_cfg.restructure_debt_threshold
            keep_stores = wage_cfg.restructure_keep_stores
            for idx in range(len(next_state.retailers)):
                seated = next_state.retailers[idx]
                if seated.cash < -debt_line:
                    seated.insolvent_ticks += 1
                else:
                    seated.insolvent_ticks = 0
                if seated.insolvent_ticks < wage_cfg.restructure_after_ticks:
                    continue
                # CLOSEABLE regions only (r >= 1) -- region 0 is the action-schema's
                # stranded no-op slot (see _restructuring_survivor_regions's docstring)
                # and is never a candidate for closure, so it never counts against
                # keep_stores and the closure loop below never touches it.
                present_count = sum(
                    1 for r in range(1, n_regions) if seated.stores_per_region[r] > 0
                )
                if present_count > keep_stores:
                    kept = _restructuring_survivor_regions(
                        seated.stores_per_region,
                        seated.last_market_share_per_region,
                        n_regions,
                        keep_stores,
                    )
                    new_stores = list(seated.stores_per_region)
                    new_ages = list(seated.store_age_per_region)
                    for r in range(1, n_regions):
                        if new_stores[r] > 0 and r not in kept:
                            new_stores[r] = 0
                            new_ages[r] = 0  # the field default -- "no history yet"
                    seated.stores_per_region = tuple(new_stores)
                    seated.store_age_per_region = tuple(new_ages)
                seated.cash = 0.0
                seated.insolvent_ticks = 0
                seated.restructurings += 1
                restructured_by_seat[idx] = True

        #
        #
        reward_cfg = self._config.reward
        rewards: dict[int, float] = {}
        observations: dict[int, npt.NDArray[np.float32]] = {}
        infos: dict[int, dict[str, object]] = {}
        terminated: dict[int, bool] = {}
        truncated: dict[int, bool] = {}
        mature_store_equivalents = _capacity_footprint(next_state, n_regions, warehouse_cfg)
        for idx in range(len(next_state.retailers)):
            observations[idx] = encode_observation(next_state, idx)

            components = reward_components(
                accounting,
                demand_result,
                idx,
                writeoff=writeoff_by_seat.get(idx, 0.0),
                penalty=penalty_by_seat.get(idx, 0.0),
            )

            # A weighted reward is computed ONLY for a LEARNING seat (one carrying a
            # normalizer in reward_stats). Seats without one — every NPC seat, and
            # every seat when mode="profit" — take the post-penalty profit directly
            # (their reward is unused if scripted, and byte-identical to 0.0–0.4 in
            # profit mode). This keeps the per-seat reward well-defined without forcing
            # a normalizer on NPC seats, and keeps ``reward == info['components']
            # ['profit']`` for EVERY seat, learning or not.
            has_normalizer = reward_cfg.mode == "weighted" and idx in state.reward_stats
            if has_normalizer:
                # Copy so this tick's fold does not mutate the input state's stats;
                # the advanced copy becomes next_state's seated normalizer.
                normalizer = RewardNormalizer.from_triples(state.reward_stats[idx].to_triples())
                rewards[idx] = compute_reward_for_seat(components, reward_cfg, normalizer, seat=idx)
                next_reward_stats[idx] = normalizer
            else:
                rewards[idx] = float(components.profit)

            infos[idx] = {
                "components": components.as_dict(),
                # CLAMP-T3 (design note Q6): the affordability clamp this seat's spend
                # was scaled by, in [0, 1] — 1.0 whenever the budget gate is off, so the
                # key is always present and always meaningful. A DX/info signal only:
                # NO observation-schema change (an obs field would be a schema bump and
                # is a separate decision), but without it an agent whose spend was cut
                # short has no way to tell that from a spend that simply did not work.
                "spend_clamp": float(demand_result.spend_clamp[idx]),
                # C-wage-probe-decomposition: this seat's write-off THIS tick (draw #5,
                # already subtracted from ``last_profit``/``cash`` in step 11 and from the
                # scored ``components['profit']`` above) — 0.0 whenever the gate is inert
                # or the draw resolved to 0 (``writeoff_by_seat`` only ever holds nonzero
                # entries), so the key is always present. A DX/info signal only, exactly
                # like ``spend_clamp`` above: purely ADDITIVE (never changes an existing
                # value), so it does not touch the byte-identity guardrail (state, RNG,
                # obs, components) — no observation-schema change either. Without it the
                # probe tooling that decomposes "ignoring wages beats complying" could
                # only read the NET effect (the scored profit), never isolate how much of
                # that gap the write-off itself accounts for.
                "writeoff": float(writeoff_by_seat.get(idx, 0.0)),
                "mature_store_equivalents": float(mature_store_equivalents[idx]),
                # Phase 4 MECHANIC 4 (RESTRUCTURING): DX signal only, exactly like
                # ``spend_clamp``/``writeoff``/``mature_store_equivalents`` above —
                # purely ADDITIVE (no observation-schema change, no byte-identity
                # impact). ``restructured`` is True ONLY on the tick THIS seat was
                # restructured (the cumulative count is ``RetailerState.
                # restructurings`` — a reporting field, not surfaced here);
                # ``insolvent_ticks`` is the freshly-updated seated counter (0
                # whenever the gate is inert or the seat is solvent). The reward is
                # UNTOUCHED by design (see ``WageConfig``'s MECHANIC 4 docstring): the
                # write-off here is a balance-sheet event, not income — every loss
                # that put the seat underwater already hit profit/reward on the way
                # down, and crediting it again here would double-count.
                "restructured": bool(restructured_by_seat.get(idx, False)),
                "insolvent_ticks": int(next_state.retailers[idx].insolvent_ticks),
                "action_mask": {
                    "expansion": compute_expansion_mask(next_state, idx, self._config),
                    "automation": compute_automation_mask(next_state, idx, self._config),
                },
            }
            terminated[idx] = False
            truncated[idx] = False

        return StepResult(
            next_state=next_state,
            observations=observations,
            rewards=rewards,
            infos=infos,
            terminated=terminated,
            truncated=truncated,
        )
