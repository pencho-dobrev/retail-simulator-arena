"""``RetailEnv`` — the Gymnasium single-agent view over the pure ``World`` seam.

This is a thin adapter: it constructs a :class:`~retail_simulator.core.world.World`,
threads the latest ``WorldState`` between calls (the seam is stateless w.r.t. the
state argument), translates the agent's flat-Box action via
:mod:`~retail_simulator.envs.action_wrapper`, and presents the Gymnasium
``(obs, reward, terminated, truncated, info)`` 5-tuple. All world logic lives in
``core``; nothing here reimplements a transition.

Episode boundaries are NOT modeled here: ``terminated`` is always ``False`` (the
world is continuing) and truncation is applied EXTERNALLY by
``gymnasium.wrappers.TimeLimit`` (use ``gym.make("RetailSim-v0",
max_episode_steps=...)``). The library stays silent on import — no logging
handlers are added.

Phase 0.1 adds two DX keys so the marketing lever's effect is visible in
``metrics.jsonl``: ``awareness`` (the agent's seated awareness stock, read off the
next ``RetailerState``) and ``marketing_spend`` (the money the agent COMMITTED to
this tick). ``marketing_spend`` is derived from the SAME action the env passed to
the seam, priced by the SAME rule (``decoded spend_fraction`` raised to
``MarketingConfig.marketing_cost_exponent`` then times ``cost_per_unit_spend`` —
linear at the 1.0 default) — i.e. "the action taken this tick", honoring the
lagged-model single-source discipline; it is 0.0 at
``reset`` (no action has been taken). Under an armed affordability clamp
(``WageConfig.cash_budget_enabled``, CLAMP-T3) the seam charges that amount TIMES
the seat's ``info['spend_clamp']``, which is emitted alongside it; at
``CoreConfig.default()`` the clamp is always 1.0 and the two coincide. See
``envs/info.py`` for the full intent-vs-charge contract.

Phase 0.2 adds one more DX key, ``assortment`` (the decoded breadth this tick), so
the contemporaneous assortment lever's effect is visible in ``metrics.jsonl``. Like
``marketing_spend`` it is derived from the SAME action the env passed to the seam
(``decode(flat_action).levers["assortment"]``) — "the action taken this tick",
single-sourced (the seam reads the identical decoded breadth for utility + opex);
it is 0.0 at ``reset`` (no action taken).

Phase 0.3 adds two more DX keys for the dual promotion lever. ``promotion`` is the
decoded promo intensity this tick (the SAME action the env passed; the seam reads
the identical decoded value for the contemporaneous lift + opex cost), 0.0 at
``reset``. ``stockpile`` is the REAL seated ``RegionState.stockpile`` read off the
NEXT state — the consumer forward-buy debt currently suppressing the pie — so the
``metrics.jsonl`` can show the debt building/decaying (the analog of ``awareness``
for the marketing lever). It is the world's seated state, not a re-derived value.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.registration import register

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.schema import ActionSchema, default_action_schema
from retail_simulator.core.world import AGENT_INDEX, World, WorldState, compute_expansion_mask
from retail_simulator.envs.action_wrapper import decode, mask_expansion_logits, to_core_action
from retail_simulator.envs.info import enrich_agent_info
from retail_simulator.envs.spaces import build_action_space, build_observation_space

# The registered Gymnasium id for the Phase 0.0 single-agent environment.
ENV_ID: str = "RetailSim-v0"


class RetailEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Single-agent Gymnasium view: the learning agent vs a discounter NPC.

    Wraps one :class:`World`. ``reset`` builds and persists the initial
    ``WorldState``; ``step`` advances the seam by one tick with the agent seated
    at index 0 and returns its observation/reward/info. The observation always
    lies within :attr:`observation_space` (bounds are derived from the schema).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: CoreConfig | None = None,
        *,
        seed: int | None = None,
        mask_expansion: bool = False,
    ) -> None:
        """ """
        super().__init__()
        self._config = config if config is not None else CoreConfig.default()
        self._world = World(config=self._config, seed=seed)
        # The seam validates actions against the default action schema internally;
        # the adapter uses the same schema so its spaces/translation stay in lockstep.
        self._schema: ActionSchema = default_action_schema()
        self._mask_expansion = mask_expansion

        self.observation_space = build_observation_space(self._config)
        self.action_space = build_action_space(self._schema)

        # Threaded between calls: the seam is stateless w.r.t. the state arg, so
        # the env owns the "current" WorldState. None until reset() is called.
        self._state: WorldState | None = None
        # Episode-step counter (core has no episode notion); 0 right after reset.
        self._episode_step: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset the world and return the initial ``(observation, info)``.

        Re-seeds the world's RNG when ``seed`` is given (so two envs reset with
        the same seed start identical trajectories). ``options`` is accepted for
        Gymnasium API compatibility but unused in Phase 0.0.
        """
        super().reset(seed=seed)
        self._state = self._world.reset(seed)
        self._episode_step = 0
        obs, info = self._world.agent_observation(self._state, AGENT_INDEX)
        return obs, self._enrich_info(
            info,
            self._state,
            marketing_spend=0.0,
            assortment=0.0,
            promotion=0.0,
            expansion_action=0.0,
            research_action=0.0,
            # Phase 1.1: at reset no action has been taken, so the SCM lever defaults
            # to the registry default (1.0 — the byte-identity anchor); no prior state
            # exists either, so the per-region stockout rate is 0.0 (the truth).
            service_level_action=1.0,
            prior_service_score_per_region=None,
            # Phase 1.2: no automation action at reset — the registry default tier 0
            # (the no-investment baseline + the byte-identity anchor).
            automation_choice=0.0,
            loyalty_spend_action=0.0,
            # Phase 5.0 (B-2): no wage/warehouse action at reset — the registry
            # default 0.0 for both (mirrors loyalty_spend_action).
            wage_spend_action=0.0,
            warehouse_invest_action=0.0,
        )

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Advance one tick with the agent acting; return the Gym 5-tuple.

        ``terminated`` is always ``False`` (continuing world); ``truncated`` is
        always ``False`` here (apply ``TimeLimit`` externally via ``gym.make``).
        """
        if self._state is None:
            raise RuntimeError("step() called before reset(); call reset() first")

        flat_action = to_core_action(action, self._schema)
        if self._mask_expansion:
            # S8: mask off the PRE-step state (self._state — still unchanged here,
            # BEFORE self._world.step below) — the mask the agent's decision actually
            # faces. NOT result.infos[AGENT_INDEX]['action_mask'] below, which is
            # computed by the seam for next_state (advisory for the NEXT action).
            expansion_mask = compute_expansion_mask(self._state, AGENT_INDEX, self._config)
            flat_action = mask_expansion_logits(flat_action, expansion_mask, self._schema)
        # Phase 1.1: capture the PRIOR per-region service_score BEFORE stepping so
        # ``_enrich_info`` can recover the per-tick stockout rate from the deterministic
        # EMA recurrence (a pure read off the seam — no recompute / no RNG).
        prior_service_score = self._state.retailers[AGENT_INDEX].service_score_per_region
        # Phase 1.2 (F3 = PERMANENT): capture the PRIOR seated automation_tier BEFORE
        # stepping so ``_enrich_info`` can recover the DIFFERENTIAL capex
        # (``capex_per_tier[new] − capex_per_tier[prev]``) charged this tick — a pure
        # read off the seam's authoritative state, no recompute.
        prior_automation_tier = int(self._state.retailers[AGENT_INDEX].automation_tier)
        result = self._world.step(self._state, {AGENT_INDEX: flat_action})
        self._state = result.next_state
        self._episode_step += 1

        # Decode the action ONCE for the DX info derivations below (marketing
        # spend + assortment breadth + promotion), all "the action taken this tick".
        levers = decode(flat_action, self._schema).levers
        # Marketing charged this tick = the action just taken (decoded
        # spend_fraction) * cost_per_unit_spend — the SAME quantity the seam
        # charges (the single-source "charged == action taken this tick"), so the
        # env never recomputes it from awareness or re-bills it.
        spend_fraction = levers.get("marketing", 0.0)
        marketing_exponent = self._config.marketing.marketing_cost_exponent
        if marketing_exponent == 1.0:
            marketing_spend = spend_fraction * self._config.marketing.cost_per_unit_spend
        else:
            marketing_spend = (
                spend_fraction**marketing_exponent * self._config.marketing.cost_per_unit_spend
            )
        # Assortment breadth set this tick = the decoded action (contemporaneous,
        # memoryless) — the SAME breadth the seam reads for utility + opex.
        assortment = levers.get("assortment", 0.0)
        # Promotion intensity set this tick = the decoded action (contemporaneous
        # lift + per-unit opex cost) — the SAME value the seam reads for utility,
        # the promo cost, and seating the next stockpile.
        promotion = levers.get("promotion", 0.0)
        # Expansion choice decoded this tick = the argmax of the action's discrete
        # block (0 = no-op, r = open region r) — the SAME choice the seam's gate
        # reads to seat presence (single-source "the action taken this tick").
        # Whether it actually opened depends on the seam's cash/presence gate; this
        # is the chosen action, the analog of the other reporting-only lever keys.
        expansion_action = levers.get("expansion", 0.0)
        # Research spend decoded this tick = the decoded continuous lever (Phase 1.0)
        # — the SAME value the seam reads for the research opex + the perception draw
        # scale (single-source "the action taken this tick"). The resulting per-observer
        # fidelity is SEATED state (read off next_state by enrich_agent_info), not
        # recomputed here — the perception layer is wholly in the seam.
        research_action = levers.get("research", 0.0)
        # Phase 1.1: the decoded SCM service_level this tick = the SAME value the seam
        # reads for the fill-rate cap + the per-retailer cogs_fraction (single-source
        # "the action taken this tick"). At the registry default (1.0) it is the
        # byte-identity baseline; below 1.0 it bites economics when cogs_premium > 0.
        service_level_action = levers.get("service_level", 1.0)
        # Phase 1.2 (F3 = PERMANENT): the decoded TARGET automation tier this tick =
        # the SAME argmax-decoded choice in {0, 1, 2} the seam reads at the upgrade
        # gate (the SoT for the monotonicity + differential-affordability check +
        # the differential capex billing). The SEATED tier (after the gate) is the
        # seam's authority and is read off ``RetailerState.automation_tier`` by
        # ``enrich_agent_info`` — this env never recomputes it.
        automation_choice = levers.get("automation", 0.0)
        loyalty_spend_action = levers.get("loyalty_spend", 0.0)
        # Phase 5.0 (B-2): the decoded wage_spend/warehouse_invest this tick = the
        # SAME values the seam reads for the paid-wage formula + the capacity-carry
        # recurrence (single-source "the action taken this tick"; mirrors
        # loyalty_spend_action). At the registry default (0.0) both are the
        # byte-identity anchor for ANY config.
        wage_spend_action = levers.get("wage_spend", 0.0)
        warehouse_invest_action = levers.get("warehouse_invest", 0.0)

        obs = np.asarray(result.observations[AGENT_INDEX], dtype=np.float32)
        reward = float(result.rewards[AGENT_INDEX])
        terminated = bool(result.terminated[AGENT_INDEX])
        truncated = bool(result.truncated[AGENT_INDEX])
        # result.infos[AGENT_INDEX] already carries the REAL action_mask the seam
        # computed for next_state (compute_expansion_mask); _enrich_info forwards it.
        info = self._enrich_info(
            result.infos[AGENT_INDEX],
            self._state,
            marketing_spend,
            assortment,
            promotion,
            expansion_action,
            research_action,
            service_level_action=service_level_action,
            prior_service_score_per_region=prior_service_score,
            automation_choice=automation_choice,
            prior_automation_tier=prior_automation_tier,
            loyalty_spend_action=loyalty_spend_action,
            wage_spend_action=wage_spend_action,
            warehouse_invest_action=warehouse_invest_action,
        )
        return obs, reward, terminated, truncated, info

    def _enrich_info(
        self,
        base_info: dict[str, Any],
        state: WorldState,
        marketing_spend: float,
        assortment: float,
        promotion: float,
        expansion_action: float,
        research_action: float,
        *,
        service_level_action: float = 1.0,
        prior_service_score_per_region: tuple[float, ...] | None = None,
        automation_choice: float = 0.0,
        prior_automation_tier: int | None = None,
        loyalty_spend_action: float = 0.0,
        wage_spend_action: float = 0.0,
        warehouse_invest_action: float = 0.0,
    ) -> dict[str, Any]:
        """ """
        return enrich_agent_info(
            base_info,
            state,
            AGENT_INDEX,
            self._episode_step,
            marketing_spend=marketing_spend,
            assortment=assortment,
            promotion=promotion,
            expansion_action=expansion_action,
            research_action=research_action,
            service_level_action=service_level_action,
            prior_service_score_per_region=prior_service_score_per_region,
            config=self._config,
            automation_choice=automation_choice,
            prior_automation_tier=prior_automation_tier,
            loyalty_spend_action=loyalty_spend_action,
            wage_spend_action=wage_spend_action,
            warehouse_invest_action=warehouse_invest_action,
        )


def register_env() -> None:
    """Register ``RetailSim-v0`` with Gymnasium (idempotent).

    Called on ``import retail_simulator`` so ``gym.make("RetailSim-v0")`` works
    without an explicit import of this module. ``max_episode_steps`` is left
    unset here on purpose: truncation is a caller concern
    (``gym.make("RetailSim-v0", max_episode_steps=104)`` applies ``TimeLimit``).
    Re-registration is skipped so repeated imports do not warn.
    """
    if ENV_ID in gym.registry:
        return
    register(id=ENV_ID, entry_point="retail_simulator.envs.gym_env:RetailEnv")
