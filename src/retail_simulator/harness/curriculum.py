"""Phase 1.4 scenario curriculum: 6-segment world + training-time scenario sampler.

The module sits in the harness layer (``harness -> {envs, core, gymnasium,
pettingzoo, ...}``); no schema bump (F6-A schema UNCHANGED at v10); no new
RNG in core; no obs/action delta.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig, DemandConfig, RegionConfig
from retail_simulator.core.state import (
    BARGAIN_SEEKER,
    BRAND_LOYAL,
    CONVENIENCE,
    PREMIUM_HUNTER,
    PRICE_SENSITIVE,
    QUALITY_SEEKING,
    SegmentParams,
    default_segments,
)
from retail_simulator.harness.sweep import (
    automation_gate_config,
    automation_verdict_for_timing,
    run_sweep,
    scm_gate_config,
    scm_verdict_for_service,
)

if TYPE_CHECKING:  # type-only imports — never pulled in at runtime
    from retail_simulator.envs.parallel_env import RetailParallelEnv

# --- 6-segment world ---------------------------------------------------------

PHASE_1_4_PREMIUM_HUNTER_PROFILE: SegmentParams = SegmentParams(
    beta_price=0.3,
    beta_assortment=1.5,
    beta_marketing=1.8,
    beta_promotion=0.1,
    beta_loyalty=0.0,
    beta_under_reference_aversion=1.2,
    beta_reference=0.0,
    beta_service=0.4,
    beta_program=0.0,
)
PHASE_1_4_BARGAIN_SEEKER_PROFILE: SegmentParams = SegmentParams(
    beta_price=1.6,
    beta_assortment=0.3,
    beta_marketing=0.0,
    beta_promotion=2.5,
    beta_loyalty=0.0,
    beta_under_reference_aversion=0.0,
    beta_reference=0.5,
    beta_service=0.0,
    beta_program=0.0,
)


def phase_1_4_six_segments() -> dict[str, SegmentParams]:
    """ """
    segments = default_segments()
    segments[PREMIUM_HUNTER] = PHASE_1_4_PREMIUM_HUNTER_PROFILE
    segments[BARGAIN_SEEKER] = PHASE_1_4_BARGAIN_SEEKER_PROFILE
    return segments


_VARIANT_A_REGION_0_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.30,
    QUALITY_SEEKING: 0.20,
    CONVENIENCE: 0.20,
    BRAND_LOYAL: 0.10,
    PREMIUM_HUNTER: 0.10,
    BARGAIN_SEEKER: 0.10,
}
_VARIANT_A_REGION_1_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.15,
    QUALITY_SEEKING: 0.10,
    CONVENIENCE: 0.10,
    BRAND_LOYAL: 0.40,
    PREMIUM_HUNTER: 0.15,
    BARGAIN_SEEKER: 0.10,
}

# Variant B (the "premium-leaning" mix): region 0 premium-hunter-heavy, region 1 a
# more even brand-loyal/PREMIUM_HUNTER split — the curriculum exercises a regime
# where the new under-reference-aversion axis is load-bearing.
_VARIANT_B_REGION_0_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.15,
    QUALITY_SEEKING: 0.20,
    CONVENIENCE: 0.15,
    BRAND_LOYAL: 0.10,
    PREMIUM_HUNTER: 0.30,
    BARGAIN_SEEKER: 0.10,
}
_VARIANT_B_REGION_1_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.15,
    QUALITY_SEEKING: 0.10,
    CONVENIENCE: 0.15,
    BRAND_LOYAL: 0.25,
    PREMIUM_HUNTER: 0.25,
    BARGAIN_SEEKER: 0.10,
}

# Variant C (the "bargain-leaning" mix): region 0 bargain-seeker-heavy, region 1
# brand-loyal-heavy with non-trivial BARGAIN_SEEKER — exercises the promotion-dominant
# regime + the new forward-buying amplification.
_VARIANT_C_REGION_0_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.20,
    QUALITY_SEEKING: 0.10,
    CONVENIENCE: 0.20,
    BRAND_LOYAL: 0.10,
    PREMIUM_HUNTER: 0.10,
    BARGAIN_SEEKER: 0.30,
}
_VARIANT_C_REGION_1_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.20,
    QUALITY_SEEKING: 0.10,
    CONVENIENCE: 0.10,
    BRAND_LOYAL: 0.35,
    PREMIUM_HUNTER: 0.05,
    BARGAIN_SEEKER: 0.20,
}

# Region 0 (large) / region 1 (smaller) pie sizes — kept at the 0.4 defaults so the
# curriculum varies the MIX, not the SIZE (per F3-A: static per-region variants).
_PHASE_1_4_REGION_0_DEMAND: float = 1000.0
_PHASE_1_4_REGION_1_DEMAND: float = 600.0


def _phase_1_4_demand_config(
    region_0_mix: dict[str, float],
    region_1_mix: dict[str, float],
    *,
    base: DemandConfig | None = None,
) -> DemandConfig:
    """Build a 2-region ``DemandConfig`` with the given per-region 6-segment mixes.

    Other ``DemandConfig`` fields (price_elasticity, noise_sigma) come from ``base`` —
    the regression sweep passes ``CoreConfig.default().demand`` so the rest of the
    world's economics are unchanged; only the per-region mix changes. The fallback
    scalar ``segment_mix`` field is also set to the region-0 mix so a single-region
    fallback (an empty ``regions`` tuple) would reproduce region 0.
    """
    base_demand = base if base is not None else CoreConfig.default().demand
    return replace(
        base_demand,
        base_regional_demand=_PHASE_1_4_REGION_0_DEMAND,
        segment_mix=dict(region_0_mix),
        regions=(
            RegionConfig(
                base_regional_demand=_PHASE_1_4_REGION_0_DEMAND,
                segment_mix=dict(region_0_mix),
            ),
            RegionConfig(
                base_regional_demand=_PHASE_1_4_REGION_1_DEMAND,
                segment_mix=dict(region_1_mix),
            ),
        ),
    )


def phase_1_4_six_segment_config(
    base_config: CoreConfig | None = None,
    *,
    region_0_mix: dict[str, float] | None = None,
    region_1_mix: dict[str, float] | None = None,
) -> CoreConfig:
    """ """
    base = base_config if base_config is not None else CoreConfig.default()
    r0 = region_0_mix if region_0_mix is not None else _VARIANT_A_REGION_0_MIX
    r1 = region_1_mix if region_1_mix is not None else _VARIANT_A_REGION_1_MIX
    return replace(
        base,
        demand=_phase_1_4_demand_config(r0, r1, base=base.demand),
        segments=phase_1_4_six_segments(),
    )


# --- ScenarioDistribution protocol + UniformScenarioDistribution -------------


@runtime_checkable
class ScenarioDistribution(Protocol):
    """ """

    scenarios: tuple[CoreConfig, ...]

    def sample(self, rng: np.random.Generator) -> CoreConfig:  # pragma: no cover - Protocol
        ...


@dataclass(frozen=True)
class UniformScenarioDistribution:
    """ """

    scenarios: tuple[CoreConfig, ...]

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError(
                "UniformScenarioDistribution requires at least one scenario "
                "(empty tuple makes sample() ill-defined)"
            )

    def sample(self, rng: np.random.Generator) -> CoreConfig:
        """Draw ONE ``CoreConfig`` uniformly at random via ``rng``."""
        index = int(rng.integers(0, len(self.scenarios)))
        return self.scenarios[index]


# --- The basic Phase 1.4 curriculum (3 variants) -----------------------------


PHASE_1_4_BASIC_CURRICULUM_NAME: str = "phase_1_4_basic"


def phase_1_4_basic_curriculum(
    base_config: CoreConfig | None = None,
) -> UniformScenarioDistribution:
    """Build the basic curriculum: 3 variants of the 6-segment world (F4-A + F3-A).

    Three per-region 6-segment mix variants — balanced / premium-leaning /
    bargain-leaning (each a different "competitive regime" the learner sees). The
    curriculum samples one uniformly at each ``CurriculumEnvWrapper.reset()``;
    PPO is trained across the distribution and learns a policy robust over the
    six-segment competitive regimes. ``base_config`` lets the caller layer extra
    knobs (e.g. ``loyalty_gate_config(...)``) on the base BEFORE the per-region
    mix is swapped in.
    """
    return UniformScenarioDistribution(
        scenarios=(
            phase_1_4_six_segment_config(
                base_config,
                region_0_mix=_VARIANT_A_REGION_0_MIX,
                region_1_mix=_VARIANT_A_REGION_1_MIX,
            ),
            phase_1_4_six_segment_config(
                base_config,
                region_0_mix=_VARIANT_B_REGION_0_MIX,
                region_1_mix=_VARIANT_B_REGION_1_MIX,
            ),
            phase_1_4_six_segment_config(
                base_config,
                region_0_mix=_VARIANT_C_REGION_0_MIX,
                region_1_mix=_VARIANT_C_REGION_1_MIX,
            ),
        )
    )


_CURRICULUM_REGISTRY: dict[str, Any] = {
    PHASE_1_4_BASIC_CURRICULUM_NAME: phase_1_4_basic_curriculum,
}


def known_curricula() -> tuple[str, ...]:
    """The names of curricula the CLI's ``--curriculum=<name>`` flag accepts.

    A small registry of name → builder. The CLI looks up the builder, calls it
    (with the same ``base_config`` it would otherwise pass straight through to the
    gate), and threads the returned ``ScenarioDistribution`` into
    :func:`harness.parallel_gate.run_parallel_gate`. Default ``None`` ⇒ no
    curriculum (the 1.3 CLI behavior is preserved).
    """
    return tuple(sorted(_CURRICULUM_REGISTRY))


def build_curriculum(name: str, base_config: CoreConfig | None = None) -> ScenarioDistribution:
    """Look up a curriculum builder by name and call it.

    Used by the CLI to map ``--curriculum=<name>`` onto a ``ScenarioDistribution``.
    An unknown name raises ``ValueError`` listing the known names — boundary
    validation, never a silent fallback. ``base_config`` is passed straight to
    the builder so the curriculum can layer the gate scenario's knobs on top.
    """
    if name not in _CURRICULUM_REGISTRY:
        raise ValueError(f"unknown curriculum {name!r}; known curricula: {list(known_curricula())}")
    builder = _CURRICULUM_REGISTRY[name]
    distribution = builder(base_config)
    if not isinstance(distribution, ScenarioDistribution):
        # Defensive: a registry entry that returns the wrong type would silently
        # break the gate. Surface it at the boundary instead.
        raise TypeError(
            f"curriculum builder for {name!r} returned {type(distribution).__name__}, "
            f"expected a ScenarioDistribution"
        )
    return distribution


# --- CurriculumEnvWrapper ----------------------------------------------------


# Lazy / module-top import: pettingzoo's ``ParallelEnv`` is the base class
# SuperSuit's ``pettingzoo_env_to_vec_env_v1`` requires via an ``isinstance``
# check (see ``supersuit/vector/vector_constructors.py`` — the bridge does NOT
# duck-type). Inheriting it here makes the curriculum wrapper a first-class
# PettingZoo Parallel env so ``build_parallel_vec_env(curriculum=...)`` and any
# Slice B composition (e.g. ``LeagueEnvWrapper(CurriculumEnvWrapper(env))``)
# bridge end-to-end without a downstream adapter shim. Mirrors the import
# locality :mod:`retail_simulator.harness.league` adopts for the league wrapper
# (the same SuperSuit-isinstance gap surfaced in 1.5 Slice A).
from pettingzoo import ParallelEnv  # noqa: E402 - kept near the wrapper for locality


class CurriculumEnvWrapper(ParallelEnv):  # type: ignore[misc, unused-ignore]
    """ """

    metadata: dict[str, Any] = {"name": "retail_sim_curriculum_parallel_v0", "render_modes": []}

    def __init__(
        self,
        distribution: ScenarioDistribution,
        *,
        n_learning_agents: int = 2,
        npc_archetypes: tuple[str, ...] = (),
        curriculum_seed: int = 0,
        core_seed: int | None = None,
    ) -> None:
        """Seed the curriculum RNG and build the FIRST scenario's parallel env.

        ``curriculum_seed`` keys the harness-only ``np.random.default_rng`` used
        for scenario picks; ``core_seed`` is the per-episode core RNG seed for the
        FIRST built env (the wrapper rebuilds on every ``reset()``, so subsequent
        episodes use the seed passed to ``reset(seed=...)``).
        """
        # Lazy import to keep ``harness.curriculum`` cheap to import: the parallel
        # env pulls pettingzoo at module load (a 1.5+s import).
        from retail_simulator.envs.parallel_env import RetailParallelEnv

        self._distribution = distribution
        self._n_learning_agents = n_learning_agents
        self._npc_archetypes = npc_archetypes
        self._curriculum_rng: np.random.Generator = np.random.default_rng(curriculum_seed)
        self._curriculum_seed = curriculum_seed
        # Build an INITIAL env from the first scenario so ``observation_space``/
        # ``action_space`` are queryable BEFORE the first reset (the PettingZoo
        # API surface convention). The first scenario is drawn from a SNAPSHOT of
        # the curriculum rng so it is reproducible without consuming the rng's
        # state (the first ``reset()`` will redraw — the natural pattern).
        first_config = self._distribution.sample(np.random.default_rng(curriculum_seed))
        self._env: RetailParallelEnv = RetailParallelEnv(
            first_config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            seed=core_seed,
        )
        # The PettingZoo agent interface delegates straight to the wrapped env.
        self.possible_agents: list[str] = list(self._env.possible_agents)
        self.agents: list[str] = list(self.possible_agents)
        # SuperSuit's MarkovVectorEnv reads ``par_env.unwrapped.render_mode`` (the
        # 0.5 bridge documented gap; same fix applied at the parallel-env layer).
        # Setting render_mode = None on the wrapper makes the bridge see a
        # well-defined attribute regardless of whether the wrapper is treated as
        # the env or unwrapped (we expose ``unwrapped`` returning self below).
        self.render_mode: Any = None
        # Track the last sampled config (a DX hook for ``info`` / diagnostics).
        self._last_scenario: CoreConfig = first_config

    @property
    def unwrapped(self) -> "CurriculumEnvWrapper":
        """The "innermost" env — we are the outermost wrapper, so return self.

        SuperSuit's ``MarkovVectorEnv`` accesses ``par_env.unwrapped.render_mode``;
        a wrapper that returned the underlying ``RetailParallelEnv`` here would
        leak the inner env's reference (which is rebuilt every ``reset()``), so
        we return ``self`` and expose ``render_mode`` on the wrapper. The wrapper
        IS the conceptual "env" the bridge sees.
        """
        return self

    @property
    def curriculum_rng(self) -> np.random.Generator:
        """ """
        return self._curriculum_rng

    @property
    def last_scenario(self) -> CoreConfig:
        """The most recently sampled ``CoreConfig`` (set by the last ``reset()``)."""
        return self._last_scenario

    def observation_space(self, agent: str) -> Any:
        """The homogeneous observation Box (schema v10 — UNCHANGED across scenarios)."""
        return self._env.observation_space(agent)

    def action_space(self, agent: str) -> Any:
        """The homogeneous action Box (schema v10 — UNCHANGED across scenarios)."""
        return self._env.action_space(agent)

    @property
    def observation_spaces(self) -> dict[str, Any]:
        """The per-agent observation Box dict (PettingZoo / SuperSuit attribute surface)."""
        return self._env.observation_spaces

    @property
    def action_spaces(self) -> dict[str, Any]:
        """The per-agent action Box dict (PettingZoo / SuperSuit attribute surface)."""
        return self._env.action_spaces

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, dict[str, Any]]]:
        """Draw the next scenario, rebuild the env, and return its ``(observations, infos)``.

        The curriculum draw happens HERE (NOT inside ``World.step`` — F5-A): the
        next ``CoreConfig`` is sampled from ``distribution`` via
        ``curriculum_rng``; a fresh ``RetailParallelEnv`` is constructed from
        that config (same seat plan, same N/M); the new env's ``reset(seed)`` is
        called and its output returned. The ``seed`` argument seeds the new env's
        CORE PCG64 (the per-tick draw stream) — the curriculum_rng is NOT advanced
        by reset's seed argument. This is the isolation contract.
        """
        from retail_simulator.envs.parallel_env import RetailParallelEnv

        scenario = self._distribution.sample(self._curriculum_rng)
        self._last_scenario = scenario
        # Close the prior env (frees its World resources); rebuild from the new
        # config. PettingZoo's parallel envs are cheap to construct (one World
        # object + numpy buffers).
        self._env.close()
        self._env = RetailParallelEnv(
            scenario,
            n_learning_agents=self._n_learning_agents,
            npc_archetypes=self._npc_archetypes,
            seed=seed,
        )
        self.possible_agents = list(self._env.possible_agents)
        self.agents = list(self.possible_agents)
        return self._env.reset(seed=seed, options=options)

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
        observations, rewards, term, trunc, infos = self._env.step(actions)
        self.agents = list(self._env.agents)
        return observations, rewards, term, trunc, infos

    def close(self) -> None:
        """Close the wrapped env (the curriculum carries no resources of its own)."""
        self._env.close()


@dataclass(frozen=True)
class RecipeIVRegressionResult:
    """ """

    scm_degenerate: bool
    scm_reason: str
    automation_degenerate: bool
    automation_reason: str
    best_scm_service: float
    best_automation_policy: str

    @property
    def passed(self) -> bool:
        """True iff both the SCM and the automation verdicts are non-degenerate."""
        return not (self.scm_degenerate or self.automation_degenerate)


def phase_1_4_recipe_iv_regression_sweep(
    six_segment_config: CoreConfig | None = None,
    *,
    grid_points: int = 11,
    ticks_per_point: int = 200,
    seeds: tuple[int, ...] = (0, 1, 2),
    start_present_both: bool = True,
) -> RecipeIVRegressionResult:
    """ """
    base = six_segment_config if six_segment_config is not None else phase_1_4_six_segment_config()

    # SCM leg: layer the SCM scenario knobs on top of the 6-segment base (the
    # ``base_segments`` keyword preserves the 6-segment dict and adds CONVENIENCE-
    # beta_service on top — the regression "extension point" introduced for 1.4).
    scm_config = scm_gate_config(base, base_segments=phase_1_4_six_segments())
    scm_result = run_sweep(
        scm_config,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        seeds=seeds,
        start_present_both=start_present_both,
        run_research=False,
        run_expansion_timing=False,
        run_scm=True,
        run_automation=False,
        run_loyalty=False,
    )
    if not scm_result.service_rows:
        raise RuntimeError(
            "phase_1_4_recipe_iv_regression_sweep: SCM leg returned no rows; the regression sweep needs a multi-region 6-segment scenario for the SCM leg to run (see scm_verdict_for_service / of)"
        )
    scm_degenerate, scm_reason, best_constant_service, _scm_profit = scm_verdict_for_service(
        scm_result.service_rows,
        scm_result.service_variable_mean if scm_result.service_variable_mean is not None else 0.0,
        (
            scm_result.service_variable_std_error
            if scm_result.service_variable_std_error is not None
            else 0.0
        ),
    )

    automation_config = automation_gate_config(base)
    automation_result = run_sweep(
        automation_config,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        seeds=seeds,
        start_present_both=start_present_both,
        run_research=False,
        run_expansion_timing=False,
        run_scm=False,
        run_automation=True,
        run_loyalty=False,
    )
    if not automation_result.automation_rows:
        raise RuntimeError(
            "phase_1_4_recipe_iv_regression_sweep: automation leg returned no rows; the regression sweep needs the automation scenario for the leg to run (see automation_verdict_for_timing / of)"
        )
    (
        automation_degenerate,
        automation_reason,
        best_automation_policy,
        _automation_profit,
    ) = automation_verdict_for_timing(automation_result.automation_rows)

    return RecipeIVRegressionResult(
        scm_degenerate=scm_degenerate,
        scm_reason=scm_reason,
        automation_degenerate=automation_degenerate,
        automation_reason=automation_reason,
        best_scm_service=best_constant_service,
        best_automation_policy=best_automation_policy,
    )


@dataclass(frozen=True)
class GeneralizationReport:
    """ """

    mean_curriculum_score: float
    mean_fixed_score: float
    per_scenario_curriculum_means: tuple[float, ...]
    per_scenario_fixed_means: tuple[float, ...]
    margin: float
    n_scenarios: int
    n_episodes_per_scenario: int


def _evaluate_policy_in_config(
    policy: Any,
    config: CoreConfig,
    *,
    n_episodes: int,
    n_learning_agents: int,
    npc_archetypes: tuple[str, ...],
    seed: int,
) -> float:
    """ """
    from retail_simulator.harness.parallel_gate import _episode_stationary_score

    # Lazy import: the parallel env pulls pettingzoo (a 1.5+s import).
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )

    def predict(obs: Any) -> Any:
        action, _state = policy.predict(obs, deterministic=True)
        return action

    scores = [
        _episode_stationary_score(env, cfg, seed=seed + episode, predict=predict)
        for episode in range(n_episodes)
    ]
    env.close()
    return float(np.mean(scores)) if scores else 0.0


def evaluate_generalization(
    curriculum_model: Any,
    fixed_model: Any,
    scenarios: tuple[CoreConfig, ...] | list[CoreConfig],
    *,
    episodes_per_scenario: int = 5,
    n_learning_agents: int = 2,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    seed: int = 0,
) -> GeneralizationReport:
    """ """
    scenarios_tuple = tuple(scenarios)
    if not scenarios_tuple:
        raise ValueError(
            "evaluate_generalization requires at least one held-out scenario; "
            "the diagnostic measures per-scenario means"
        )
    if episodes_per_scenario < 1:
        raise ValueError(
            f"episodes_per_scenario must be >= 1, got {episodes_per_scenario}; "
            "at least one episode per scenario is needed for a mean"
        )

    curriculum_means: list[float] = []
    fixed_means: list[float] = []
    for index, scenario in enumerate(scenarios_tuple):
        scenario_seed = seed + index * 10_000
        curriculum_means.append(
            _evaluate_policy_in_config(
                curriculum_model,
                scenario,
                n_episodes=episodes_per_scenario,
                n_learning_agents=n_learning_agents,
                npc_archetypes=npc_archetypes,
                seed=scenario_seed,
            )
        )
        fixed_means.append(
            _evaluate_policy_in_config(
                fixed_model,
                scenario,
                n_episodes=episodes_per_scenario,
                n_learning_agents=n_learning_agents,
                npc_archetypes=npc_archetypes,
                seed=scenario_seed,
            )
        )

    mean_curriculum = float(np.mean(curriculum_means))
    mean_fixed = float(np.mean(fixed_means))
    return GeneralizationReport(
        mean_curriculum_score=mean_curriculum,
        mean_fixed_score=mean_fixed,
        per_scenario_curriculum_means=tuple(curriculum_means),
        per_scenario_fixed_means=tuple(fixed_means),
        margin=mean_curriculum - mean_fixed,
        n_scenarios=len(scenarios_tuple),
        n_episodes_per_scenario=int(episodes_per_scenario),
    )


__all__ = [
    "CurriculumEnvWrapper",
    "GeneralizationReport",
    "PHASE_1_4_BARGAIN_SEEKER_PROFILE",
    "PHASE_1_4_BASIC_CURRICULUM_NAME",
    "PHASE_1_4_PREMIUM_HUNTER_PROFILE",
    "RecipeIVRegressionResult",
    "ScenarioDistribution",
    "UniformScenarioDistribution",
    "build_curriculum",
    "evaluate_generalization",
    "known_curricula",
    "phase_1_4_basic_curriculum",
    "phase_1_4_recipe_iv_regression_sweep",
    "phase_1_4_six_segment_config",
    "phase_1_4_six_segments",
]
