"""``RetailParallelEnv`` — the PettingZoo Parallel multi-agent view over the seam.

The default ``RetailParallelEnv()`` (``n_learning_agents=2``, ``npc_archetypes=()``)
is a K=2 ALL-LEARNING duopoly (both seats take PettingZoo actions). Scripted NPCs fill
the remaining seats only when ``npc_archetypes`` is non-empty.

Layer note: this is the adapter layer, so importing ``pettingzoo``/``gymnasium`` here
is allowed (CI forbids them only under ``core/``). All world logic comes from ``core``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from pettingzoo import ParallelEnv

from retail_simulator.core.config import CoreConfig, SeatSpec
from retail_simulator.core.schema import ActionSchema, default_action_schema
from retail_simulator.core.world import World, WorldState, compute_expansion_mask
from retail_simulator.envs.action_wrapper import decode, mask_expansion_logits, to_core_action
from retail_simulator.envs.info import enrich_agent_info
from retail_simulator.envs.spaces import build_action_space, build_observation_space


def _agent_id(seat_index: int) -> str:
    """The PettingZoo agent id for a learning seat (the stable ordered name)."""
    return f"retailer_{seat_index}"


def _one_hot_presence(home: int, n_regions: int) -> tuple[int, ...]:
    """ """
    return tuple(1 if r == home else 0 for r in range(n_regions))


class RetailParallelEnv(ParallelEnv):
    """PettingZoo Parallel multi-agent env: N learning agents over the shared seam.

    See the module docstring. ``observation_space``/``action_space`` are HOMOGENEOUS
    across learning agents (the same per-agent v5 Box the Gym env uses — the
    param-sharing precondition). ``reset``/``step`` return per-agent dicts keyed by
    ``possible_agents``; ``infos[agent]`` carries the real ``action_mask`` + the
    un-normalized reward ``components`` + the DX flat metric keys.
    """

    metadata: dict[str, Any] = {"name": "retail_sim_parallel_v0", "render_modes": []}

    def __init__(
        self,
        config: CoreConfig | None = None,
        *,
        n_learning_agents: int = 2,
        npc_archetypes: tuple[str, ...] = (),
        home_regions: tuple[int, ...] | None = None,
        seed: int | None = None,
        mask_expansion_seats: tuple[int, ...] = (),
    ) -> None:
        """ """
        if n_learning_agents < 1:
            raise ValueError(f"n_learning_agents must be >= 1, got {n_learning_agents}")
        for seat in mask_expansion_seats:
            if (
                not isinstance(seat, (int, np.integer))
                or isinstance(seat, bool)
                or not (0 <= seat < n_learning_agents)
            ):
                raise ValueError(
                    f"mask_expansion_seats entry {seat!r} (type {type(seat).__name__}) is "
                    "invalid; must be an integer LEARNING seat index satisfying "
                    f"0 <= seat < n_learning_agents ({n_learning_agents})"
                )

        base_config = config if config is not None else CoreConfig.default()
        region_cfgs = base_config.demand.regions
        n_regions = len(region_cfgs) if region_cfgs else 1
        seat_plan = self._build_seat_plan(
            n_learning_agents, npc_archetypes, n_regions, home_regions
        )
        # Inject the seat plan into the config (additive): the World builds N+M seats
        # from it; the absent-plan default world is untouched (this env always supplies
        # one). dataclasses.replace keeps every other CoreConfig field as given.
        self._config = replace(base_config, seats=seat_plan)
        self._schema: ActionSchema = default_action_schema()
        self._world = World(config=self._config, seed=seed)

        self._n_learning_agents = n_learning_agents
        self.possible_agents: list[str] = [_agent_id(i) for i in range(n_learning_agents)]
        self._seat_of: dict[str, int] = {_agent_id(i): i for i in range(n_learning_agents)}
        self.agents: list[str] = list(self.possible_agents)
        # S8: the learning-seat indices whose decode gets illegal-expansion masking
        # (see the constructor docstring); validated above. Stored verbatim (a small
        # tuple — membership via `in` is fine at this size).
        self._mask_expansion_seats: tuple[int, ...] = mask_expansion_seats

        # Homogeneous per-agent Boxes (the SAME spaces the Gym env builds from the
        # schema). PettingZoo reads these via the observation_space/action_space methods
        # and/or the *_spaces dicts; expose both for bridge compatibility.
        self._observation_space: gym.spaces.Box = build_observation_space(self._config)
        self._action_space: gym.spaces.Box = build_action_space(self._schema)
        self.observation_spaces: dict[str, gym.spaces.Box] = {
            aid: self._observation_space for aid in self.possible_agents
        }
        self.action_spaces: dict[str, gym.spaces.Box] = {
            aid: self._action_space for aid in self.possible_agents
        }

        # Threaded between calls: the seam is stateless w.r.t. the state arg, so this
        # env owns the "current" WorldState. None until reset().
        self._state: WorldState | None = None
        # Episode-step counter (core has no episode notion); 0 right after reset.
        self._episode_step: int = 0
        # The raw per-SEAT infos from the most recent ``World.step`` (ALL seats,
        # learning AND npc) — the seam's authoritative, un-normalized
        # ``info['components']`` (``reward_components(...).as_dict()``) for EVERY seat,
        # before the public ``step`` slices it down to the learning agents. The public
        # API exposes only learning seats (NPCs are not PettingZoo agents), but the
        # hot-seat needs the NPC rival's CANONICAL components to score it on the
        # cycle's metric (``weighted_objective``); state-derived reconstruction is NOT
        # faithful (the seated ``last_market_share``/``loyalty_stock`` are region-0
        # scalars, not the demand-weighted aggregate the cycle is measured on). This is
        # a READ-ONLY by-seat-index mirror; ``None`` until the first ``step``.
        self._last_raw_infos: dict[int, dict[str, Any]] | None = None

    @staticmethod
    def _build_seat_plan(
        n_learning_agents: int,
        npc_archetypes: tuple[str, ...],
        n_regions: int,
        home_regions: tuple[int, ...] | None = None,
    ) -> tuple[SeatSpec, ...]:
        """ """
        n_seats = n_learning_agents + len(npc_archetypes)
        if home_regions is not None:
            if len(home_regions) != n_seats:
                raise ValueError(
                    f"home_regions must have exactly {n_seats} entries (one per seat: "
                    f"{n_learning_agents} learning + {len(npc_archetypes)} NPC), got "
                    f"{len(home_regions)}"
                )
            for seat_index, home in enumerate(home_regions):
                if (
                    not isinstance(home, (int, np.integer))
                    or isinstance(home, bool)
                    or not (0 <= home < n_regions)
                ):
                    raise ValueError(
                        f"home_regions[{seat_index}]={home!r} (type "
                        f"{type(home).__name__}) is invalid; must be an integer region "
                        f"index satisfying 0 <= r < n_regions ({n_regions})"
                    )
            seats: list[SeatSpec] = [
                SeatSpec(is_npc=False, presence=_one_hot_presence(home_regions[i], n_regions))
                for i in range(n_learning_agents)
            ]
            seats.extend(
                SeatSpec(
                    is_npc=True,
                    archetype=name,
                    presence=_one_hot_presence(home_regions[n_learning_agents + i], n_regions),
                )
                for i, name in enumerate(npc_archetypes)
            )
            return tuple(seats)

        cover_with_seat0 = len(npc_archetypes) == 0
        learning: list[SeatSpec] = []
        for i in range(n_learning_agents):
            presence = (1,) * n_regions if (i == 0 and cover_with_seat0) else None
            learning.append(SeatSpec(is_npc=False, presence=presence))
        npc = tuple(SeatSpec(is_npc=True, archetype=name) for name in npc_archetypes)
        return tuple(learning) + npc

    def observation_space(self, agent: str) -> gym.spaces.Box:
        """The homogeneous observation Box for an agent (PettingZoo requires a method)."""
        return self._observation_space

    def action_space(self, agent: str) -> gym.spaces.Box:
        """The homogeneous action Box for an agent (PettingZoo requires a method)."""
        return self._action_space

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, dict[str, Any]]]:
        """Reset the world; return per-learning-agent ``(observations, infos)`` dicts.

        Re-seeds the world's RNG when ``seed`` is given (two envs reset with the same
        seed start identical trajectories). ``options`` is accepted for the PettingZoo
        API but unused. Each agent's ``info`` carries the REAL ``action_mask`` from the
        seam's ``agent_observation`` plus the DX flat keys (0.0 for the per-lever
        quantities — no action taken at reset). ``agents`` is set to ``possible_agents``.
        """
        self._state = self._world.reset(seed)
        self._episode_step = 0
        self.agents = list(self.possible_agents)

        observations: dict[str, npt.NDArray[np.float32]] = {}
        infos: dict[str, dict[str, Any]] = {}
        for aid in self.possible_agents:
            seat = self._seat_of[aid]
            obs, base_info = self._world.agent_observation(self._state, seat)
            observations[aid] = np.asarray(obs, dtype=np.float32)
            infos[aid] = enrich_agent_info(
                base_info,
                self._state,
                seat,
                self._episode_step,
                marketing_spend=0.0,
                assortment=0.0,
                promotion=0.0,
                expansion_action=0.0,
                research_action=0.0,
                # Phase 1.1: at reset no action has been taken; the SCM lever defaults
                # to the registry 1.0 and there is no prior state, so the per-region
                # stockout rate is 0.0 (the truth — no stockouts have occurred).
                service_level_action=1.0,
                prior_service_score_per_region=None,
                config=self._config,
                # Phase 1.2: no automation action at reset — tier 0 (the no-investment
                # baseline + the byte-identity anchor).
                automation_choice=0.0,
                loyalty_spend_action=0.0,
                # Phase 5.0 (B-2): no wage/warehouse action at reset — the registry
                # default 0.0 for both (mirrors loyalty_spend_action).
                wage_spend_action=0.0,
                warehouse_invest_action=0.0,
            )
        return observations, infos

    def step(
        self, actions: dict[str, npt.NDArray[np.float32]]
    ) -> tuple[
        dict[str, npt.NDArray[np.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """ """
        if self._state is None:
            raise RuntimeError("step() called before reset(); call reset() first")

        # 1. Build the joint action for the LEARNING seats only (the seam fills NPCs).
        #    Decode each action ONCE for the per-agent DX info derivations below.
        joint_action: dict[int, npt.NDArray[np.float32]] = {}
        agent_levers: dict[str, dict[str, float]] = {}
        for aid in self.possible_agents:
            seat = self._seat_of[aid]
            flat_action = to_core_action(actions[aid], self._schema)
            if seat in self._mask_expansion_seats:
                # S8: mask THIS seat's flat action off the PRE-step state (the mask
                # the agent's decision actually faces — self._state is still the
                # state BEFORE self._world.step runs, below) before it reaches EITHER
                # consumer: the joint action the seam decodes internally AND the local
                # `decode` call for DX info, so both agree on the (now legal) choice.
                # Per-seat: every OTHER seat's flat_action is untouched even when it
                # shares this episode with a masked seat (an unmasked scripted
                # archetype's own illegal picks must survive, module docstring).
                expansion_mask = compute_expansion_mask(self._state, seat, self._config)
                flat_action = mask_expansion_logits(flat_action, expansion_mask, self._schema)
            joint_action[seat] = flat_action
            agent_levers[aid] = decode(flat_action, self._schema).levers

        # Phase 1.1: capture each learning seat's PRIOR per-region service_score BEFORE
        # stepping so the per-tick stockout rate can be recovered from the deterministic
        # EMA recurrence after the step (a pure read off the seam — no recompute, no RNG).
        prior_service_scores: dict[str, tuple[float, ...]] = {
            aid: self._state.retailers[self._seat_of[aid]].service_score_per_region
            for aid in self.possible_agents
        }
        # Phase 1.2 (F3 = PERMANENT): capture each learning seat's PRIOR seated
        # ``automation_tier`` BEFORE stepping so the DIFFERENTIAL capex charged
        # this tick can be recovered as ``capex_per_tier[new] − capex_per_tier[prev]``
        # (the SAME quantity ``apply_accounting`` charged — single-sourced from
        # the seam's state, no recompute).
        prior_automation_tiers: dict[str, int] = {
            aid: int(self._state.retailers[self._seat_of[aid]].automation_tier)
            for aid in self.possible_agents
        }

        # 2. Advance the world EXACTLY ONCE.
        result = self._world.step(self._state, joint_action)
        self._state = result.next_state
        self._episode_step += 1
        # Stash the raw per-seat infos (ALL seats, incl. NPCs) before the public slice
        # below drops the NPC seats. See ``_last_raw_infos`` — the hot-seat reads the
        # NPC rival's canonical ``info['components']`` from here to score it on the
        # cycle's ``weighted_objective`` metric. Purely additive; the public return is
        # unchanged.
        self._last_raw_infos = result.infos

        # 3. Slice the seam's per-retailer outputs back into per-agent dicts (learning
        #    seats only) — no recomputation, the seam's outputs pass through unchanged.
        observations: dict[str, npt.NDArray[np.float32]] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}
        for aid in self.possible_agents:
            seat = self._seat_of[aid]
            levers = agent_levers[aid]
            observations[aid] = np.asarray(result.observations[seat], dtype=np.float32)
            rewards[aid] = float(result.rewards[seat])
            terminations[aid] = False
            truncations[aid] = False
            marketing_spend_fraction = levers.get("marketing", 0.0)
            marketing_exponent = self._config.marketing.marketing_cost_exponent
            marketing_spend = (
                marketing_spend_fraction * self._config.marketing.cost_per_unit_spend
                if marketing_exponent == 1.0
                else marketing_spend_fraction**marketing_exponent
                * self._config.marketing.cost_per_unit_spend
            )
            infos[aid] = enrich_agent_info(
                result.infos[seat],
                self._state,
                seat,
                self._episode_step,
                marketing_spend=marketing_spend,
                assortment=levers.get("assortment", 0.0),
                promotion=levers.get("promotion", 0.0),
                expansion_action=levers.get("expansion", 0.0),
                # Phase 1.0: the decoded research spend this tick (the action taken);
                # research_fidelity is the SEATED per-observer view enrich reads off
                # next_state (the seam's authority — never recomputed in the env).
                research_action=levers.get("research", 0.0),
                # Phase 1.1: the decoded SCM service this tick (the action taken; the
                # SAME value the seam reads for the fill-rate cap + per-retailer
                # cogs_fraction) + the prior per-region service_score (captured before
                # the step, above) so enrich recovers the per-tick stockout rate from
                # the deterministic EMA recurrence (no RNG in the env).
                service_level_action=levers.get("service_level", 1.0),
                prior_service_score_per_region=prior_service_scores[aid],
                config=self._config,
                # Phase 1.2 (F3 = PERMANENT): the decoded TARGET automation tier
                # this tick (the action taken; the SAME argmax-decoded choice the
                # seam reads at the upgrade gate). The SEATED tier + the DIFFERENTIAL
                # capex debited this tick are read off ``next_state`` / ``config`` by
                # ``enrich_agent_info``, using the PRE-step seated tier captured
                # above (no recompute in the env).
                automation_choice=levers.get("automation", 0.0),
                prior_automation_tier=prior_automation_tiers[aid],
                # Phase 1.3: the decoded loyalty-program intensity this tick (the
                # SAME value the seam reads for the same-tick brand-loyal utility
                # boost + the per-tick opex line — the single-source "the action
                # taken this tick"). At the registry default (0.0) the lever is the
                # no-program baseline + the byte-identity anchor.
                loyalty_spend_action=levers.get("loyalty_spend", 0.0),
                # Phase 5.0 (B-2): the decoded wage_spend/warehouse_invest this tick
                # (the SAME values the seam reads for the paid-wage formula + the
                # capacity-carry recurrence — single-source "the action taken this
                # tick"). At the registry default (0.0) both are the byte-identity
                # anchor for ANY config.
                wage_spend_action=levers.get("wage_spend", 0.0),
                warehouse_invest_action=levers.get("warehouse_invest", 0.0),
            )

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        """No-op render (no render modes are supported)."""
        return None

    def close(self) -> None:
        """No-op close (the env holds no external resources)."""
        return None
