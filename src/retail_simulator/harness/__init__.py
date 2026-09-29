"""Learnability + competition tooling (NOT in the env import path).

* :mod:`~retail_simulator.harness.sweep` — an analytical action-space sweep (degeneracy
  detection) against the pure :class:`~retail_simulator.core.world.World` (no RL, fast).
* :mod:`~retail_simulator.harness.learnability` — the SINGLE-agent gate: SB3 PPO vs a
  random baseline → a PASS/FAIL :class:`GateReport`.
* :mod:`~retail_simulator.harness.parallel_gate` (Phase 0.5) — the SuperSuit↔SB3 bridge,
  the param-shared self-play MULTI-agent gate (scored on the STATIONARY weighted
  objective), and the soft strategic-diversity report.
* :mod:`~retail_simulator.harness.calibrate` (Phase 0.5) — the NPC calibrate loop
  (each archetype's beat-random margin on its primary metric, on the fast seam) + the
  reward variance-balance health check.

Layer note: the harness sits ABOVE the env adapter (``harness -> {envs, core,
stable_baselines3, supersuit, gymnasium, pettingzoo}``); it always goes through
``RetailEnv``/``RetailParallelEnv``/``World`` and never reimplements world logic. Heavy
RL imports are done LAZILY inside functions so importing this package stays cheap.
"""

from __future__ import annotations

from retail_simulator.harness.calibrate import (
    ArchetypeCalibration,
    calibrate_all,
    calibrate_archetype,
    reward_variance_check,
)
from retail_simulator.harness.curriculum import (
    CurriculumEnvWrapper,
    GeneralizationReport,
    RecipeIVRegressionResult,
    ScenarioDistribution,
    UniformScenarioDistribution,
    build_curriculum,
    evaluate_generalization,
    known_curricula,
    phase_1_4_basic_curriculum,
    phase_1_4_recipe_iv_regression_sweep,
    phase_1_4_six_segment_config,
    phase_1_4_six_segments,
)
from retail_simulator.harness.ladder import (
    ARCHETYPE_SPEC_PREFIX,
    CALIBRATED_ENV_LABEL,
    DEFAULT_ENV_LABEL,
    LADDER_ARCHETYPE_LABELS,
    LADDER_SPIKE_ARCHETYPE_LABELS,
    RANDOM_SPEC_PREFIX,
    LadderEntry,
    LadderResult,
    PairwiseMargin,
    ParticipantKind,
    ParticipantSpec,
    parse_participant_spec,
    run_ladder,
)
from retail_simulator.harness.learnability import GateReport, run_gate
from retail_simulator.harness.parallel_gate import (
    BridgeSmokeResult,
    DiversityReport,
    build_parallel_report,
    diversity_report,
    run_bridge_smoke,
    run_parallel_gate,
)
from retail_simulator.harness.sweep import (
    SweepResult,
    SweepRow1D,
    SweepRow2D,
    assortment_verdict_for_axis,
    run_sweep,
    verdict_for_sweep,
)

__all__ = [
    "ARCHETYPE_SPEC_PREFIX",
    "ArchetypeCalibration",
    "BridgeSmokeResult",
    "CALIBRATED_ENV_LABEL",
    "CurriculumEnvWrapper",
    "DEFAULT_ENV_LABEL",
    "DiversityReport",
    "GateReport",
    "GeneralizationReport",
    "LADDER_ARCHETYPE_LABELS",
    "LADDER_SPIKE_ARCHETYPE_LABELS",
    "LadderEntry",
    "LadderResult",
    "PairwiseMargin",
    "ParticipantKind",
    "ParticipantSpec",
    "RANDOM_SPEC_PREFIX",
    "RecipeIVRegressionResult",
    "ScenarioDistribution",
    "SweepResult",
    "SweepRow1D",
    "SweepRow2D",
    "UniformScenarioDistribution",
    "assortment_verdict_for_axis",
    "build_curriculum",
    "build_parallel_report",
    "calibrate_all",
    "calibrate_archetype",
    "diversity_report",
    "evaluate_generalization",
    "known_curricula",
    "parse_participant_spec",
    "phase_1_4_basic_curriculum",
    "phase_1_4_recipe_iv_regression_sweep",
    "phase_1_4_six_segment_config",
    "phase_1_4_six_segments",
    "reward_variance_check",
    "run_bridge_smoke",
    "run_gate",
    "run_ladder",
    "run_parallel_gate",
    "run_sweep",
    "verdict_for_sweep",
]
