"""The SuperSuit↔SB3 bridge + the multi-agent self-play learnability gate (Phase 0.5).

Layer note: ``harness -> {envs, core, stable_baselines3, supersuit, gymnasium,
pettingzoo}``; importing these here is allowed (forbidden only under ``core/``). The
heavy RL imports (supersuit, sb3) are done LAZILY inside the functions so
``import retail_simulator.harness`` stays cheap and does not hard-require ``[train]``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig, RewardConfig
from retail_simulator.core.reward import RewardComponents, weighted_objective
from retail_simulator.harness.learnability import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_MAX_EPISODE_STEPS,
    FULL_TOTAL_STEPS,
    GATE_SIGMA_MULTIPLE,
    GateReport,
    _curve_trends_up,
)
from retail_simulator.harness.sweep import (
    RESEARCH_OPPONENT_ARCHETYPE,
    automation_gate_config,
    loyalty_gate_config,
    research_gate_segments,
    scm_gate_config,
)

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.harness.curriculum import ScenarioDistribution

_RESEARCH_LEVER_NAME: str = "research"

_SCM_LEVER_NAME: str = "service_level"

_AUTOMATION_LEVER_NAME: str = "automation"

_LOYALTY_LEVER_NAME: str = "loyalty_spend"


# The default self-play competition: a clean all-learning duopoly (F1 default 2),
# scored on the weighted objective. Two homogeneous learning seats exercise the
# param-share bridge + simultaneous actions + the per-seat normalizer.
DEFAULT_N_LEARNING_AGENTS: int = 2
BRIDGE_SMOKE_STEPS: int = 10_000
DIVERSITY_WITHIN_FRACTION: float = 0.80
DIVERSITY_MAX_RATIO: float = 2.0
LSTM_HIDDEN_SIZE: int = 64


def competition_config(*, reward_mode: str = "weighted") -> CoreConfig:
    """ """
    return CoreConfig(reward=RewardConfig(mode=reward_mode))


def build_parallel_vec_env(
    config: CoreConfig | None = None,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = (),
    num_vec_envs: int = 1,
    num_cpus: int = 1,
    seed: int | None = None,
    curriculum: "ScenarioDistribution | None" = None,
    curriculum_seed: int = 0,
    home_regions: tuple[int, ...] | None = None,
    mask_expansion_seats: tuple[int, ...] = (),
) -> Any:
    """ """
    import supersuit as ss

    from retail_simulator.envs.parallel_env import RetailParallelEnv

    if curriculum is not None and home_regions is not None:
        raise ValueError(
            "build_parallel_vec_env: curriculum and home_regions cannot both be "
            "given — CurriculumEnvWrapper has no seating seam of its own"
        )
    if curriculum is not None and mask_expansion_seats:
        raise ValueError(
            "build_parallel_vec_env: curriculum and mask_expansion_seats cannot both "
            "be given — CurriculumEnvWrapper has no masking seam of its own"
        )

    base_config = config if config is not None else competition_config()
    penv: Any
    if curriculum is not None:
        # Lazy import — keeps ``import retail_simulator.harness.parallel_gate``
        # cheap (CurriculumEnvWrapper pulls envs.parallel_env which pulls
        # pettingzoo). The wrapper's surface mirrors RetailParallelEnv enough for
        # SuperSuit's MarkovVectorEnv (possible_agents, observation_space(agent),
        # action_space(agent), reset(seed, options), step, agents, metadata,
        # unwrapped.render_mode).
        from retail_simulator.harness.curriculum import CurriculumEnvWrapper

        penv = CurriculumEnvWrapper(
            curriculum,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            curriculum_seed=curriculum_seed,
            core_seed=seed,
        )
    else:
        penv = RetailParallelEnv(
            base_config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            seed=seed,
            home_regions=home_regions,
            mask_expansion_seats=mask_expansion_seats,
        )
    # SuperSuit's MarkovVectorEnv reads ``par_env.unwrapped.render_mode``. The env
    # declares ``render_modes: []`` (no rendering) but does not carry a ``render_mode``
    # attribute; set it to None here (a bridge-only concern — the env itself never
    # renders) so the wrap does not AttributeError. unwrapped already returns self.
    if not hasattr(penv, "render_mode"):
        penv.render_mode = None
    vec = ss.pettingzoo_env_to_vec_env_v1(penv)
    vec = ss.concat_vec_envs_v1(
        vec,
        num_vec_envs,
        num_cpus=num_cpus,
        base_class="stable_baselines3",
    )
    return vec


@dataclass(frozen=True)
class BridgeSmokeResult:
    """ """

    api_test_passed: bool
    learn_steps: int
    n_learning_agents: int


def run_bridge_smoke(
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    learn_steps: int = BRIDGE_SMOKE_STEPS,
    num_cycles: int = 100,
    seed: int = 0,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
) -> BridgeSmokeResult:
    """ """
    from pettingzoo.test import parallel_api_test

    from retail_simulator.envs.parallel_env import RetailParallelEnv

    parallel_api_test(RetailParallelEnv(n_learning_agents=n_learning_agents), num_cycles=num_cycles)

    # 2. The bridge builds + PPO trains end-to-end (the real plumbing check).
    vec = build_parallel_vec_env(n_learning_agents=n_learning_agents, seed=seed)
    model = _make_shared_ppo(vec, seed=seed, policy_kind=policy_kind)
    model.learn(total_timesteps=learn_steps, progress_bar=False)
    vec.close()

    return BridgeSmokeResult(
        api_test_passed=True,
        learn_steps=learn_steps,
        n_learning_agents=n_learning_agents,
    )


def _episode_stationary_score(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    predict: Any,
) -> float:
    """ """
    observations, _infos = env.reset(seed=seed)
    per_seat_totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    horizon = DEFAULT_MAX_EPISODE_STEPS
    for _ in range(horizon):
        actions = {aid: predict(observations[aid]) for aid in env.agents}
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            components = _components_from_info(infos[aid])
            per_seat_totals[aid] += weighted_objective(components, cfg)
    return float(np.mean(list(per_seat_totals.values())))


def _components_from_info(info: dict[str, Any]) -> RewardComponents:
    """ """
    raw = info["components"]
    return RewardComponents(
        profit=float(raw["profit"]),
        revenue=float(raw["revenue"]),
        market_share=float(raw["market_share"]),
        loyalty=float(raw["loyalty"]),
    )


def evaluate_shared_policy_stationary(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> list[float]:
    """ """
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    env = RetailParallelEnv(config, n_learning_agents=n_learning_agents, seed=seed)

    def predict(obs: Any) -> Any:
        action, _state = model.predict(obs, deterministic=True)
        return action

    scores = [
        _episode_stationary_score(env, cfg, seed=seed + episode, predict=predict)
        for episode in range(episodes)
    ]
    env.close()
    return scores


def evaluate_random_stationary(
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> list[float]:
    """ """
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    env = RetailParallelEnv(config, n_learning_agents=n_learning_agents, seed=seed)
    action_space = env.action_space(env.possible_agents[0])
    action_space.seed(seed)

    def predict(_obs: Any) -> Any:
        return action_space.sample()

    scores = [
        _episode_stationary_score(env, cfg, seed=seed + episode, predict=predict)
        for episode in range(episodes)
    ]
    env.close()
    return scores


def _mixed_game_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    trained_agent: str,
    action_space: Any,
) -> dict[str, float]:
    """One episode where ONLY ``trained_agent`` uses the trained policy; rest are random.

    The trained seat plays its deterministic ``predict``; every OTHER learning seat samples
    its action space (the random opponent). Returns ``{agent_id: Σ_t weighted_objective}``
    — the stationary score of EACH seat in this single shared game. This is the head-to-head
    primitive: trained and random compete in the SAME market, so the score difference is a
    real competitive-skill signal (unlike comparing two separate all-trained / all-random
    games on the absolute objective — see :func:`evaluate_head_to_head`).
    """
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions = {
            aid: (
                model.predict(observations[aid], deterministic=True)[0]
                if aid == trained_agent
                else action_space.sample()
            )
            for aid in env.agents
        }
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_head_to_head(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """ """
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    env = RetailParallelEnv(config, n_learning_agents=n_learning_agents, seed=seed)
    action_space = env.action_space(env.possible_agents[0])

    trained_scores: list[float] = []
    random_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        action_space.seed(episode_seed + 5_000)
        trained_seat_vals: list[float] = []
        random_seat_vals: list[float] = []
        # The trained policy occupies each learning seat once (random fills the rest).
        for trained_agent in env.possible_agents:
            totals = _mixed_game_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                trained_agent=trained_agent,
                action_space=action_space,
            )
            trained_seat_vals.append(totals[trained_agent])
            random_seat_vals.extend(v for aid, v in totals.items() if aid != trained_agent)
        trained_scores.append(float(np.mean(trained_seat_vals)))
        random_scores.append(float(np.mean(random_seat_vals)))
    env.close()
    return trained_scores, random_scores


def _research_offset(schema: Any) -> int:
    """The flat-action offset of the ``research`` lever (read from the schema layout).

    The force-off mechanism pins exactly this dim to 0. Reading the offset from
    ``flatten_action_space_layout`` (the layout authority) rather than hardcoding 6
    keeps the force-off correct if the action layout ever changes.
    """
    from retail_simulator.core.encoding import flatten_action_space_layout

    layout = flatten_action_space_layout(schema)
    for slot in layout.levers:
        if slot.spec.name == _RESEARCH_LEVER_NAME:
            return slot.offset
    raise ValueError(f"no {_RESEARCH_LEVER_NAME!r} lever in the action schema")


def research_force_off_action(
    action: npt.NDArray[np.float32], research_offset: int
) -> npt.NDArray[np.float32]:
    """ """
    forced = np.array(action, dtype=np.float32).copy()
    forced[research_offset] = 0.0
    return forced


def scm_force_default_action(
    action: npt.NDArray[np.float32],
    scm_offset: int,
    scm_default: float = 1.0,
) -> npt.NDArray[np.float32]:
    """ """
    forced = np.array(action, dtype=np.float32).copy()
    forced[scm_offset] = scm_default
    return forced


def _scm_offset_and_default(schema: Any) -> tuple[int, float]:
    """The flat offset + the REGISTRY DEFAULT of the ``service_level`` lever.

    Reading both the offset AND the registry default from
    ``flatten_action_space_layout`` (the layout authority) rather than hardcoding 7
    and 1.0 keeps the force-default correct if the action layout or the registry
    default ever change.
    """
    from retail_simulator.core.encoding import flatten_action_space_layout

    layout = flatten_action_space_layout(schema)
    for slot in layout.levers:
        if slot.spec.name == _SCM_LEVER_NAME:
            return slot.offset, float(slot.spec.default)
    raise ValueError(f"no {_SCM_LEVER_NAME!r} lever in the action schema")


def _research_on_off_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    research_on_agent: str,
    research_offset: int,
) -> dict[str, float]:
    """ """
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions: dict[str, Any] = {}
        for aid in env.agents:
            base_action, _state = model.predict(observations[aid], deterministic=True)
            if aid == research_on_agent:
                actions[aid] = base_action
            else:
                actions[aid] = research_force_off_action(base_action, research_offset)
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_research_on_vs_off(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = (RESEARCH_OPPONENT_ARCHETYPE,),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """ """
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    if n_learning_agents < 2:
        raise ValueError(
            f"evaluate_research_on_vs_off requires n_learning_agents >= 2 (got "
            f"{n_learning_agents}): the position-controlled rotation puts every OTHER "
            "seat in the research-off role, which is empty (⇒ silent NaN mean) at "
            "n=1. For a true single-agent research gate, run two separate rollouts "
            "(same trained model, action unchanged vs research_force_off_action'd) "
            "and compare — that path is the orchestrator's runner, not this function."
        )
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    research_offset = _research_offset(env._schema)  # noqa: SLF001 - the env's action schema

    research_on_scores: list[float] = []
    research_off_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        on_seat_vals: list[float] = []
        off_seat_vals: list[float] = []
        # The research-ON role occupies each learning seat once (rest forced off).
        for research_on_agent in env.possible_agents:
            totals = _research_on_off_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                research_on_agent=research_on_agent,
                research_offset=research_offset,
            )
            on_seat_vals.append(totals[research_on_agent])
            off_seat_vals.extend(v for aid, v in totals.items() if aid != research_on_agent)
        research_on_scores.append(float(np.mean(on_seat_vals)))
        research_off_scores.append(float(np.mean(off_seat_vals)))
    env.close()
    return research_on_scores, research_off_scores


def _scm_on_off_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    scm_on_agent: str,
    scm_offset: int,
    scm_default: float,
) -> dict[str, float]:
    """ """
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions: dict[str, Any] = {}
        for aid in env.agents:
            base_action, _state = model.predict(observations[aid], deterministic=True)
            if aid == scm_on_agent:
                actions[aid] = base_action
            else:
                actions[aid] = scm_force_default_action(base_action, scm_offset, scm_default)
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_scm_on_vs_off(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """ """
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    if n_learning_agents < 2:
        raise ValueError(
            f"""evaluate_scm_on_vs_off requires n_learning_agents >= 2 (got {n_learning_agents}): the position-controlled rotation puts every OTHER seat in the scm-off role, which is empty (⇒ silent NaN mean) at n=1. Phase 1.1 calls for the single-agent vs Discounter substrate; the current implementation uses n=2 + Discounter as a close equivalent (SCM is first-order competitor-independent). A literal n=1 path would compare two separate rollouts (same trained model, action unchanged vs scm_force_default_action'd) — that is a Phase 1.x follow-up runner, not this function."""
        )
    cfg = config.reward
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    scm_offset, scm_default = _scm_offset_and_default(env._schema)  # noqa: SLF001

    scm_on_scores: list[float] = []
    scm_off_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        on_seat_vals: list[float] = []
        off_seat_vals: list[float] = []
        # The SCM-ON role occupies each learning seat once (rest forced to default).
        for scm_on_agent in env.possible_agents:
            totals = _scm_on_off_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                scm_on_agent=scm_on_agent,
                scm_offset=scm_offset,
                scm_default=scm_default,
            )
            on_seat_vals.append(totals[scm_on_agent])
            off_seat_vals.extend(v for aid, v in totals.items() if aid != scm_on_agent)
        scm_on_scores.append(float(np.mean(on_seat_vals)))
        scm_off_scores.append(float(np.mean(off_seat_vals)))
    env.close()
    return scm_on_scores, scm_off_scores


@dataclass(frozen=True)
class SCMLeverUsage:
    """ """

    mean: float
    std: float
    uses_lever: bool
    default: float
    n_samples: int


USAGE_MARGIN_MEAN: float = 0.1
USAGE_MARGIN_STD: float = 0.05


def measure_scm_lever_usage(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> SCMLeverUsage:
    """ """
    from retail_simulator.envs.action_wrapper import decode
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    scm_offset, scm_default = _scm_offset_and_default(env._schema)  # noqa: SLF001

    service_levels: list[float] = []
    for episode in range(episodes):
        observations, _infos = env.reset(seed=seed + episode)
        for _ in range(DEFAULT_MAX_EPISODE_STEPS):
            actions: dict[str, Any] = {}
            for aid in env.agents:
                base_action, _state = model.predict(observations[aid], deterministic=True)
                actions[aid] = base_action
                # The decoded service_level (post-clip, the SAME value the seam reads
                # for the fill-rate cap + per-retailer cogs_fraction) — the lever's
                # USE signal, not the raw network output.
                decoded = decode(np.asarray(base_action, dtype=np.float32), env._schema).levers  # noqa: SLF001
                service_levels.append(float(decoded.get(_SCM_LEVER_NAME, scm_default)))
            observations, _rewards, _term, _trunc, _infos = env.step(actions)
    env.close()

    arr = np.asarray(service_levels, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float(scm_default)
    std = float(arr.std()) if arr.size > 1 else 0.0
    uses_lever = abs(mean - scm_default) > USAGE_MARGIN_MEAN and std > USAGE_MARGIN_STD
    return SCMLeverUsage(
        mean=mean,
        std=std,
        uses_lever=uses_lever,
        default=scm_default,
        n_samples=int(arr.size),
    )


AUTOMATION_UPTAKE_TAU: float = 0.5


def automation_force_default_action(
    action: npt.NDArray[np.float32], automation_offset: int, n_tiers: int
) -> npt.NDArray[np.float32]:
    """ """
    forced = np.array(action, dtype=np.float32).copy()
    # Zero the whole 3-wide block, then write a one-hot at the tier-0 slot — the
    # argmax-decoded value is 0 regardless of the surrounding action's magnitude.
    forced[automation_offset : automation_offset + n_tiers] = 0.0
    forced[automation_offset] = 1.0
    return forced


def _automation_offset_and_width(schema: Any) -> tuple[int, int]:
    """The flat offset + the WIDTH of the ``automation`` lever (read from the schema).

    Reading the offset AND the width from ``flatten_action_space_layout`` (the layout
    authority) rather than hardcoding 8 and 3 keeps the force-default correct if the
    action layout ever changes. The default automation lever's width equals
    :data:`retail_simulator.core.schema.N_AUTOMATION_TIERS`.
    """
    from retail_simulator.core.encoding import flatten_action_space_layout

    layout = flatten_action_space_layout(schema)
    for slot in layout.levers:
        if slot.spec.name == _AUTOMATION_LEVER_NAME:
            return slot.offset, slot.width
    raise ValueError(f"no {_AUTOMATION_LEVER_NAME!r} lever in the action schema")


def _automation_on_off_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    automation_on_agent: str,
    automation_offset: int,
    n_tiers: int,
) -> dict[str, float]:
    """ """
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions: dict[str, Any] = {}
        for aid in env.agents:
            base_action, _state = model.predict(observations[aid], deterministic=True)
            if aid == automation_on_agent:
                actions[aid] = base_action
            else:
                actions[aid] = automation_force_default_action(
                    base_action, automation_offset, n_tiers
                )
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_automation_on_vs_off(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """ """
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    if n_learning_agents < 2:
        raise ValueError(
            f"""evaluate_automation_on_vs_off requires n_learning_agents >= 2 (got {n_learning_agents}): the position-controlled rotation puts every OTHER seat in the automation-off role, which is empty (⇒ silent NaN mean) at n=1. Phase 1.2 calls for the single-agent vs Discounter substrate; the current implementation uses n=2 + Discounter as a close equivalent (automation is first-order competitor-independent, — identical to SCM's structure). A literal n=1 path would compare two separate rollouts (same trained model, action unchanged vs automation_force_default_action'd) — that is a Phase 1.x follow-up runner, not this function."""
        )
    cfg = config.reward
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    automation_offset, n_tiers = _automation_offset_and_width(env._schema)  # noqa: SLF001

    automation_on_scores: list[float] = []
    automation_off_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        on_seat_vals: list[float] = []
        off_seat_vals: list[float] = []
        # The automation-ON role occupies each learning seat once (rest forced to tier 0).
        for automation_on_agent in env.possible_agents:
            totals = _automation_on_off_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                automation_on_agent=automation_on_agent,
                automation_offset=automation_offset,
                n_tiers=n_tiers,
            )
            on_seat_vals.append(totals[automation_on_agent])
            off_seat_vals.extend(v for aid, v in totals.items() if aid != automation_on_agent)
        automation_on_scores.append(float(np.mean(on_seat_vals)))
        automation_off_scores.append(float(np.mean(off_seat_vals)))
    env.close()
    return automation_on_scores, automation_off_scores


@dataclass(frozen=True)
class AutomationUptake:
    """ """

    uptake_fraction: float
    tier_distribution: tuple[float, ...]
    uses_lever: bool
    n_tiers: int
    n_samples: int
    n_episodes: int


def measure_automation_uptake(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
    uptake_tau: float = AUTOMATION_UPTAKE_TAU,
) -> AutomationUptake:
    """ """
    from retail_simulator.envs.action_wrapper import decode
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    _automation_offset, n_tiers = _automation_offset_and_width(env._schema)  # noqa: SLF001

    decoded_tiers: list[int] = []
    # Per-episode end-of-episode seated tier (across learning seats). A seat-episode
    # counts as "reached tier >= 1" if its FINAL seated automation_tier is >= 1
    # (the monotonicity guarantee means this is true iff any upgrade ever fired).
    seat_episode_outcomes: list[bool] = []
    for episode in range(episodes):
        observations, _infos = env.reset(seed=seed + episode)
        episode_tiers: list[int] = []
        final_seats: dict[str, int] = {aid: 0 for aid in env.possible_agents}
        for _ in range(DEFAULT_MAX_EPISODE_STEPS):
            actions: dict[str, Any] = {}
            for aid in env.agents:
                base_action, _state = model.predict(observations[aid], deterministic=True)
                actions[aid] = base_action
                decoded = decode(np.asarray(base_action, dtype=np.float32), env._schema).levers  # noqa: SLF001
                # The DECODED target tier — diagnostic over the rollout for the
                # tier distribution channel.
                episode_tiers.append(int(decoded.get(_AUTOMATION_LEVER_NAME, 0.0)))
            observations, _rewards, _term, _trunc, infos = env.step(actions)
            # Track each learning seat's CURRENT seated tier (monotone non-decreasing
            # — only the last value matters for the end-of-episode check; we overwrite
            # rather than accumulate so we always carry the latest seated value).
            for aid in env.agents:
                final_seats[aid] = int(infos[aid].get("automation_tier", 0))
        decoded_tiers.extend(episode_tiers)
        # One outcome per learning seat × episode: True iff the final seated tier is
        # >= 1 (i.e. some upgrade fired by end of episode under monotonicity).
        for aid in env.possible_agents:
            seat_episode_outcomes.append(final_seats[aid] >= 1)
    env.close()

    arr = np.asarray(decoded_tiers, dtype=np.int64)
    n_samples = int(arr.size)
    uptake_fraction = (
        float(sum(seat_episode_outcomes) / len(seat_episode_outcomes))
        if seat_episode_outcomes
        else 0.0
    )
    if n_samples > 0:
        counts = np.bincount(arr, minlength=n_tiers)
        tier_distribution = tuple(float(c) / n_samples for c in counts[:n_tiers])
    else:
        tier_distribution = tuple(0.0 for _ in range(n_tiers))
    uses_lever = uptake_fraction > uptake_tau
    return AutomationUptake(
        uptake_fraction=uptake_fraction,
        tier_distribution=tier_distribution,
        uses_lever=uses_lever,
        n_tiers=n_tiers,
        n_samples=n_samples,
        n_episodes=int(episodes),
    )


LOYALTY_USAGE_MARGIN_MEAN: float = 0.1
LOYALTY_USAGE_MARGIN_STD: float = 0.05


def loyalty_force_default_action(
    action: npt.NDArray[np.float32],
    loyalty_offset: int,
    loyalty_default: float = 0.0,
) -> npt.NDArray[np.float32]:
    """ """
    forced = np.array(action, dtype=np.float32).copy()
    forced[loyalty_offset] = loyalty_default
    return forced


def _loyalty_offset_and_default(schema: Any) -> tuple[int, float]:
    """The flat offset + the REGISTRY DEFAULT of the ``loyalty_spend`` lever.

    Reading both the offset AND the registry default from
    ``flatten_action_space_layout`` (the layout authority) rather than hardcoding
    11 and 0.0 keeps the force-default correct if the action layout or the
    registry default ever change.
    """
    from retail_simulator.core.encoding import flatten_action_space_layout

    layout = flatten_action_space_layout(schema)
    for slot in layout.levers:
        if slot.spec.name == _LOYALTY_LEVER_NAME:
            return slot.offset, float(slot.spec.default)
    raise ValueError(f"no {_LOYALTY_LEVER_NAME!r} lever in the action schema")


def _loyalty_on_off_episode_scores(
    env: "RetailParallelEnv",
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    loyalty_on_agent: str,
    loyalty_offset: int,
    loyalty_default: float,
) -> dict[str, float]:
    """ """
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions: dict[str, Any] = {}
        for aid in env.agents:
            base_action, _state = model.predict(observations[aid], deterministic=True)
            if aid == loyalty_on_agent:
                actions[aid] = base_action
            else:
                actions[aid] = loyalty_force_default_action(
                    base_action, loyalty_offset, loyalty_default
                )
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_loyalty_on_vs_off(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """ """
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    if n_learning_agents < 2:
        raise ValueError(
            f"""evaluate_loyalty_on_vs_off requires n_learning_agents >= 2 (got {n_learning_agents}): the position-controlled rotation puts every OTHER seat in the loyalty-off role, which is empty (⇒ silent NaN mean) at n=1. Phase 1.3 calls for the single-agent vs Discounter substrate; the current implementation uses n=2 + Discounter as a close equivalent (loyalty is first-order competitor-independent, — identical to SCM / automation). A literal n=1 path would compare two separate rollouts (same trained model, action unchanged vs loyalty_force_default_action'd) — that is a Phase 1.x follow-up runner, not this function."""
        )
    cfg = config.reward
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    loyalty_offset, loyalty_default = _loyalty_offset_and_default(env._schema)  # noqa: SLF001

    loyalty_on_scores: list[float] = []
    loyalty_off_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        on_seat_vals: list[float] = []
        off_seat_vals: list[float] = []
        # The loyalty-ON role occupies each learning seat once (rest forced to 0.0).
        for loyalty_on_agent in env.possible_agents:
            totals = _loyalty_on_off_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                loyalty_on_agent=loyalty_on_agent,
                loyalty_offset=loyalty_offset,
                loyalty_default=loyalty_default,
            )
            on_seat_vals.append(totals[loyalty_on_agent])
            off_seat_vals.extend(v for aid, v in totals.items() if aid != loyalty_on_agent)
        loyalty_on_scores.append(float(np.mean(on_seat_vals)))
        loyalty_off_scores.append(float(np.mean(off_seat_vals)))
    env.close()
    return loyalty_on_scores, loyalty_off_scores


@dataclass(frozen=True)
class LoyaltyLeverUsage:
    """ """

    mean: float
    std: float
    uses_lever: bool
    default: float
    n_samples: int


def measure_loyalty_lever_usage(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
    usage_margin_mean: float = LOYALTY_USAGE_MARGIN_MEAN,
    usage_margin_std: float = LOYALTY_USAGE_MARGIN_STD,
) -> LoyaltyLeverUsage:
    """ """
    from retail_simulator.envs.action_wrapper import decode
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    loyalty_offset, loyalty_default = _loyalty_offset_and_default(env._schema)  # noqa: SLF001

    loyalty_spends: list[float] = []
    for episode in range(episodes):
        observations, _infos = env.reset(seed=seed + episode)
        for _ in range(DEFAULT_MAX_EPISODE_STEPS):
            actions: dict[str, Any] = {}
            for aid in env.agents:
                base_action, _state = model.predict(observations[aid], deterministic=True)
                actions[aid] = base_action
                # The decoded loyalty_spend (post-clip, the SAME value the seam
                # reads for the same-tick brand-loyal utility boost + the
                # per-tick opex line) — the lever's USE signal, not the raw
                # network output.
                decoded = decode(np.asarray(base_action, dtype=np.float32), env._schema).levers  # noqa: SLF001
                loyalty_spends.append(float(decoded.get(_LOYALTY_LEVER_NAME, loyalty_default)))
            observations, _rewards, _term, _trunc, _infos = env.step(actions)
    env.close()

    arr = np.asarray(loyalty_spends, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float(loyalty_default)
    std = float(arr.std()) if arr.size > 1 else 0.0
    uses_lever = mean > usage_margin_mean and std > usage_margin_std
    return LoyaltyLeverUsage(
        mean=mean,
        std=std,
        uses_lever=uses_lever,
        default=loyalty_default,
        n_samples=int(arr.size),
    )


def run_parallel_gate(
    *,
    total_steps: int = FULL_TOTAL_STEPS,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    seed: int = 42,
    num_vec_envs: int = 1,
    num_cpus: int = 1,
    reward_mode: str = "weighted",
    research_gate: bool = False,
    research_opponent: str = RESEARCH_OPPONENT_ARCHETYPE,
    scm_gate: bool = False,
    automation_gate: bool = False,
    loyalty_gate: bool = False,
    curriculum: "ScenarioDistribution | None" = None,
    curriculum_seed: int = 0,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
) -> GateReport:
    """ """
    # The four gates are mutually exclusive (each selects a different training
    # substrate + on-vs-off eval). The validation surfaces an explicit error so a typo
    # (e.g. setting two flags at once) fails fast at the boundary.
    if sum(int(g) for g in (research_gate, scm_gate, automation_gate, loyalty_gate)) > 1:
        raise ValueError(
            "research_gate, scm_gate, automation_gate, and loyalty_gate are mutually "
            "exclusive — each selects a different training substrate and a different "
            "on-vs-off eval (research vs the wandering opponent; SCM vs the "
            "Discounter in the SCM scenario; automation vs the Discounter in the "
            "automation scenario; loyalty vs the Discounter in the loyalty scenario)"
        )

    config = competition_config(reward_mode=reward_mode)
    if research_gate:
        config = replace(config, segments=research_gate_segments())
    elif scm_gate:
        config = scm_gate_config(config)
    elif automation_gate:
        config = automation_gate_config(config)
    elif loyalty_gate:
        config = loyalty_gate_config(config)
    npc_archetypes: tuple[str, ...]
    if research_gate:
        npc_archetypes = (research_opponent,)
    elif scm_gate:
        npc_archetypes = ("discounter",)
    elif automation_gate:
        npc_archetypes = ("discounter",)
    elif loyalty_gate:
        npc_archetypes = ("discounter",)
    else:
        npc_archetypes = ()

    vec = build_parallel_vec_env(
        config,
        n_learning_agents=n_learning_agents,
        npc_archetypes=npc_archetypes,
        num_vec_envs=num_vec_envs,
        num_cpus=num_cpus,
        seed=seed,
        curriculum=curriculum,
        curriculum_seed=curriculum_seed,
    )
    curve: list[tuple[int, float]] = []
    callback = _make_curve_callback(curve)
    model = _make_shared_ppo(vec, seed=seed, policy_kind=policy_kind)
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
    vec.close()

    if research_gate:
        on_scores, off_scores = evaluate_research_on_vs_off(
            model,
            config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            episodes=eval_episodes,
            seed=seed,
        )
        return build_parallel_report(
            trained_scores=on_scores,
            random_scores=off_scores,
            learning_curve=list(curve),
            total_steps=total_steps,
            eval_episodes=eval_episodes,
        )

    if scm_gate:
        on_scores, off_scores = evaluate_scm_on_vs_off(
            model,
            config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            episodes=eval_episodes,
            seed=seed,
        )
        return build_parallel_report(
            trained_scores=on_scores,
            random_scores=off_scores,
            learning_curve=list(curve),
            total_steps=total_steps,
            eval_episodes=eval_episodes,
        )

    if automation_gate:
        on_scores, off_scores = evaluate_automation_on_vs_off(
            model,
            config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            episodes=eval_episodes,
            seed=seed,
        )
        return build_parallel_report(
            trained_scores=on_scores,
            random_scores=off_scores,
            learning_curve=list(curve),
            total_steps=total_steps,
            eval_episodes=eval_episodes,
        )

    if loyalty_gate:
        on_scores, off_scores = evaluate_loyalty_on_vs_off(
            model,
            config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            episodes=eval_episodes,
            seed=seed,
        )
        return build_parallel_report(
            trained_scores=on_scores,
            random_scores=off_scores,
            learning_curve=list(curve),
            total_steps=total_steps,
            eval_episodes=eval_episodes,
        )

    trained_scores, random_scores = evaluate_head_to_head(
        model, config, n_learning_agents=n_learning_agents, episodes=eval_episodes, seed=seed
    )

    return build_parallel_report(
        trained_scores=trained_scores,
        random_scores=random_scores,
        learning_curve=list(curve),
        total_steps=total_steps,
        eval_episodes=eval_episodes,
    )


def build_parallel_report(
    *,
    trained_scores: list[float],
    random_scores: list[float],
    learning_curve: list[tuple[int, float]],
    total_steps: int,
    eval_episodes: int,
) -> GateReport:
    """ """
    trained_mean = float(np.mean(trained_scores))
    random_mean = float(np.mean(random_scores))
    random_std = float(np.std(random_scores))
    margin = trained_mean - random_mean
    threshold = GATE_SIGMA_MULTIPLE * random_std
    beats_random = margin >= threshold
    trends_up = _curve_trends_up(learning_curve)
    return GateReport(
        ppo_mean_reward=trained_mean,
        random_mean_reward=random_mean,
        random_std_reward=random_std,
        margin=margin,
        threshold=threshold,
        beats_random=beats_random,
        curve_trends_up=trends_up,
        total_steps=total_steps,
        eval_episodes=eval_episodes,
        learning_curve=tuple((int(s), float(r)) for s, r in learning_curve),
    )


def _make_shared_ppo(
    vec_env: Any,
    *,
    seed: int,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
) -> Any:
    """ """
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    if policy_kind == "mlp":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy", vec_env, seed=None, verbose=0, ent_coef=ent_coef, gae_lambda=gae_lambda
        )
    if policy_kind == "lstm":
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:  # pragma: no cover - exercised when [recurrent] extra is missing
            raise ImportError(
                "policy_kind='lstm' requires the [recurrent] extra; "
                "install with: pip install 'retail-simulator[recurrent]' "
                "(or: pip install sb3-contrib>=2.3,<3.0)"
            ) from exc
        return RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            seed=None,
            verbose=0,
            ent_coef=ent_coef,
            gae_lambda=gae_lambda,
            policy_kwargs={"lstm_hidden_size": LSTM_HIDDEN_SIZE},
        )
    raise ValueError(f"unknown policy_kind {policy_kind!r}; expected 'mlp' or 'lstm'")


def _make_curve_callback(sink: list[tuple[int, float]]) -> Any:
    """SB3 callback recording the shared-policy rolling mean episode reward.

    Mirrors the single-agent gate's curve callback: at each rollout end it appends
    ``(timestep, mean_reward)`` from SB3's ``ep_info_buffer`` (the shared policy's
    episode returns under self-play). ``BaseCallback`` is imported lazily (only the
    ``[train]`` extra has it). The curve drives ``curve_trends_up`` (criterion 2).
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _CurveCallback(BaseCallback):
        def __init__(self, curve_sink: list[tuple[int, float]]) -> None:
            super().__init__()
            self._sink = curve_sink

        def _on_rollout_end(self) -> None:
            buffer = getattr(self.model, "ep_info_buffer", None)
            if buffer:
                mean_reward = float(np.mean([ep["r"] for ep in buffer if "r" in ep]))
                self._sink.append((int(self.num_timesteps), mean_reward))

        def _on_step(self) -> bool:
            return True

    return _CurveCallback(sink)


@dataclass(frozen=True)
class DiversityReport:
    """ """

    best_score: float
    min_score: float
    fraction_within: float
    n_within: int
    max_min_ratio: float
    meets_soft_goal: bool


def diversity_report(policy_scores: dict[str, float]) -> DiversityReport:
    """ """
    if not policy_scores:
        raise ValueError("policy_scores must not be empty")

    scores = list(policy_scores.values())
    best = max(scores)
    worst = min(scores)

    # Fraction within 80% of best. With a non-positive best (every policy <= 0 — a
    # degenerate/failed roster) no policy "competes": the within-fraction is 0.
    if best > 0.0:
        threshold = DIVERSITY_WITHIN_FRACTION * best
        n_within = sum(1 for s in scores if s >= threshold)
    else:
        n_within = 0
    fraction_within = n_within / len(scores)

    if worst > 0.0:
        max_min_ratio = best / worst
    else:
        max_min_ratio = float("inf") if best > 0.0 else 0.0

    meets_soft_goal = n_within >= 2 and max_min_ratio <= DIVERSITY_MAX_RATIO

    return DiversityReport(
        best_score=float(best),
        min_score=float(worst),
        fraction_within=float(fraction_within),
        n_within=int(n_within),
        max_min_ratio=float(max_min_ratio),
        meets_soft_goal=bool(meets_soft_goal),
    )
