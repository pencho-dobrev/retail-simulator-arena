"""SB3 PPO vs random baseline — the Phase 0.0 learnability gate.

PASS (the exact rule) requires BOTH:

1. **Beats random by >= 2 sigma:** ``ppo_mean >= random_mean + 2 * random_std``.
2. **Non-degenerate learning:** the smoothed training curve trends up (the final
   window's mean return exceeds the initial window's, i.e. the run actually
   learned rather than starting already-good or flat).

A JSON report is written for provenance; a learning-curve PNG is saved only if
``matplotlib`` is importable (it is NOT a hard dependency of this package — the
plot is best-effort and silently skipped when absent).

Layer note: ``harness -> envs -> core``; importing ``gymnasium`` and
``stable-baselines3`` here is allowed (forbidden only under ``core/``). Heavy RL
imports are done lazily inside functions so ``import retail_simulator.harness``
stays cheap and does not hard-require the ``[train]`` extra.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # type-only imports; never pulled in at runtime import time
    import gymnasium as gym

# The store count of an agent present in region 0 only (the reset state). An
# episode "opens region 1" the tick its open-store count first exceeds this — the
# seam no-ops an unaffordable/illegal open, so a rising ``stores`` is the
# authoritative signal that an open was actually ACCEPTED (not merely requested).
_REGION_0_ONLY_STORES: int = 1

# Default scoring/eval horizon (~2 simulated years) — mirrors the core hint.
DEFAULT_MAX_EPISODE_STEPS: int = 104
# Default number of evaluation episodes for both PPO and the random baseline.
DEFAULT_EVAL_EPISODES: int = 20
# "Quick" smoke budget for CI (~200k steps); the full research gate is ~2M.
QUICK_TOTAL_STEPS: int = 200_000
FULL_TOTAL_STEPS: int = 2_000_000
# The "beats random" margin: PPO mean must clear random mean by this many sigma.
GATE_SIGMA_MULTIPLE: float = 2.0

ENV_ID: str = "RetailSim-v0"

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpansionUptake:
    """ """

    fraction_opening: float
    median_open_tick: float | None
    episodes: int


@dataclass(frozen=True)
class GateReport:
    """ """

    ppo_mean_reward: float
    random_mean_reward: float
    random_std_reward: float
    margin: float
    threshold: float
    beats_random: bool
    curve_trends_up: bool
    total_steps: int
    eval_episodes: int
    learning_curve: tuple[tuple[int, float], ...]
    expansion_uptake: ExpansionUptake | None = None

    @property
    def passed(self) -> bool:
        """ """
        return self.beats_random and self.curve_trends_up

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (tuples become lists; uptake None-safe)."""
        data = asdict(self)
        data["learning_curve"] = [list(point) for point in self.learning_curve]
        data["passed"] = self.passed
        # asdict already turned the nested ExpansionUptake into a dict (or left it
        # None) — JSON-serializable as-is; median_open_tick stays None when never.
        return data


def evaluate_random_baseline(
    env: gym.Env[Any, Any],
    *,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> list[float]:
    """Roll out the random policy (``action_space.sample()``) for N episodes.

    Returns the per-episode total (undiscounted) returns. The env is reset with a
    distinct seed per episode (derived from ``seed``) and the action space's RNG
    is seeded once for reproducibility, so the baseline is deterministic given
    ``seed``.
    """
    returns: list[float] = []
    env.action_space.seed(seed)
    for episode in range(episodes):
        _obs, _info = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            done = bool(terminated) or bool(truncated)
        returns.append(total)
    return returns


def evaluate_policy_returns(
    model: Any,
    env: gym.Env[Any, Any],
    *,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> list[float]:
    """Roll out a trained SB3 model deterministically for N episodes.

    Returns per-episode total returns. Uses ``model.predict(..,
    deterministic=True)`` so the evaluation reflects the learned policy, not its
    exploration noise. Same per-episode seeding scheme as the baseline.
    """
    returns: list[float] = []
    for episode in range(episodes):
        obs, _info = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            done = bool(terminated) or bool(truncated)
        returns.append(total)
    return returns


def measure_expansion_uptake(
    model: Any,
    env: gym.Env[Any, Any],
    *,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> ExpansionUptake:
    """ """
    open_ticks: list[int] = []
    for episode in range(episodes):
        obs, _info = env.reset(seed=seed + episode)
        done = False
        opened_at: int | None = None
        step_index = 0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            step_index += 1
            if opened_at is None and int(info.get("stores", 0)) > _REGION_0_ONLY_STORES:
                # First tick the open-store count rose => region 1 was opened. Use
                # the env's own episode_step when present (its provenance for the
                # tick), else fall back to the loop counter.
                opened_at = int(info.get("episode_step", step_index))
            done = bool(terminated) or bool(truncated)
        if opened_at is not None:
            open_ticks.append(opened_at)

    fraction_opening = len(open_ticks) / episodes if episodes else 0.0
    median_open_tick = float(np.median(open_ticks)) if open_ticks else None
    return ExpansionUptake(
        fraction_opening=fraction_opening,
        median_open_tick=median_open_tick,
        episodes=episodes,
    )


def _curve_trends_up(curve: list[tuple[int, float]]) -> bool:
    """Is the learning curve non-degenerate (final window > initial window)?

    With < 4 samples (a very short smoke), we cannot judge a trend, so we treat
    it as non-degenerate (do not block on too little data — the >=2 sigma test
    still governs). Otherwise compare the mean of the last quarter of logged
    points against the mean of the first quarter.
    """
    if len(curve) < 4:
        return True
    rewards = [reward for _step, reward in curve]
    window = max(1, len(rewards) // 4)
    initial = float(np.mean(rewards[:window]))
    final = float(np.mean(rewards[-window:]))
    return final > initial


def run_gate(
    *,
    total_steps: int = QUICK_TOTAL_STEPS,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    seed: int = 42,
    n_envs: int = 1,
    report_path: Path | None = None,
    save_plot: bool = True,
    progress: bool = False,
) -> GateReport:
    """Train PPO, compare to random, and return the PASS/FAIL :class:`GateReport`.

    Trains ``PPO("MlpPolicy", env)`` for ``total_steps`` with no custom wrappers,
    then evaluates both PPO and the random baseline over ``eval_episodes`` episodes
    at a fixed ``seed``. A learning curve is captured via SB3's ``Monitor`` + a
    logging callback.

    ``n_envs`` controls training parallelism: with ``n_envs > 1`` PPO collects
    rollouts from ``n_envs`` env instances in parallel (SB3 ``SubprocVecEnv``),
    multiplying wall-clock throughput across CPU cores without changing the core
    env. Sub-envs are seeded deterministically from ``seed`` (seed, seed+1, ...),
    so the gate stays reproducible given ``(seed, n_envs)``. Evaluation always runs
    on a single env. (The core sim is untouched; this is purely the training path.)

    If ``report_path`` is given, the report JSON (and, when ``matplotlib`` is
    available and ``save_plot`` is true, a ``learning_curve.png`` beside it) is
    written. The heavy RL imports happen here so the module stays import-light.
    """
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    import retail_simulator  # noqa: F401  (registers RetailSim-v0)

    train_env: Any
    if n_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        # make_vec_env wraps each sub-env in Monitor, so the curve callback's
        # ep_info_buffer is populated exactly as in the single-env path.
        train_env = make_vec_env(
            ENV_ID,
            n_envs=n_envs,
            seed=seed,
            env_kwargs={"max_episode_steps": max_episode_steps},
            vec_env_cls=SubprocVecEnv,
            # fork avoids re-importing __main__ (forkserver/spawn break when the
            # caller isn't an importable module); safe here — the env has no
            # background threads.
            vec_env_kwargs={"start_method": "fork"},
        )
    else:
        train_env = Monitor(gym.make(ENV_ID, max_episode_steps=max_episode_steps))
    curve: list[tuple[int, float]] = []
    callback = _make_curve_callback(curve)

    model = PPO("MlpPolicy", train_env, seed=seed, verbose=1 if progress else 0)
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)

    eval_env = gym.make(ENV_ID, max_episode_steps=max_episode_steps)
    ppo_returns = evaluate_policy_returns(model, eval_env, episodes=eval_episodes, seed=seed)
    expansion_uptake = measure_expansion_uptake(model, eval_env, episodes=eval_episodes, seed=seed)
    random_returns = evaluate_random_baseline(eval_env, episodes=eval_episodes, seed=seed)
    eval_env.close()
    train_env.close()

    report = build_report(
        ppo_returns=ppo_returns,
        random_returns=random_returns,
        learning_curve=list(curve),
        total_steps=total_steps,
        eval_episodes=eval_episodes,
        expansion_uptake=expansion_uptake,
    )

    if report_path is not None:
        _write_report(report, report_path, save_plot=save_plot)

    return report


def build_report(
    *,
    ppo_returns: list[float],
    random_returns: list[float],
    learning_curve: list[tuple[int, float]],
    total_steps: int,
    eval_episodes: int,
    expansion_uptake: ExpansionUptake | None = None,
) -> GateReport:
    """ """
    ppo_mean = float(np.mean(ppo_returns))
    random_mean = float(np.mean(random_returns))
    # Population std of the baseline returns; the ">= 2 sigma" rule wants the
    # spread of the random baseline itself.
    random_std = float(np.std(random_returns))
    margin = ppo_mean - random_mean
    threshold = GATE_SIGMA_MULTIPLE * random_std
    beats_random = margin >= threshold
    trends_up = _curve_trends_up(learning_curve)
    return GateReport(
        ppo_mean_reward=ppo_mean,
        random_mean_reward=random_mean,
        random_std_reward=random_std,
        margin=margin,
        threshold=threshold,
        beats_random=beats_random,
        curve_trends_up=trends_up,
        total_steps=total_steps,
        eval_episodes=eval_episodes,
        learning_curve=tuple((int(s), float(r)) for s, r in learning_curve),
        expansion_uptake=expansion_uptake,
    )


def _make_curve_callback(sink: list[tuple[int, float]]) -> Any:
    """Build an SB3 callback that records rolling mean episode reward.

    SB3's ``BaseCallback`` is imported lazily (it is only available with the
    ``[train]`` extra). At each rollout end the callback appends ``(timestep,
    mean_reward)`` to ``sink`` from the ``Monitor``-tracked episode-reward
    buffer, so the learning curve has a handful of points even on a short run.
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


def _write_report(report: GateReport, report_path: Path, *, save_plot: bool) -> None:
    """Write the gate report JSON and (best-effort) a learning-curve PNG."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if save_plot and report.learning_curve:
        _maybe_save_curve_plot(report, report_path.parent / "learning_curve.png")


def _maybe_save_curve_plot(report: GateReport, plot_path: Path) -> bool:
    """Save a learning-curve PNG if matplotlib is importable; else skip silently.

    Returns ``True`` if a plot was written. ``matplotlib`` is intentionally NOT a
    dependency of this package (only the ``[viz]`` extra), so this degrades to a
    no-op when it is absent.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend; no display required
        import matplotlib.pyplot as plt
    except ImportError:
        _log.debug("matplotlib not installed; skipping learning-curve plot")
        return False

    steps = [s for s, _ in report.learning_curve]
    rewards = [r for _, r in report.learning_curve]
    fig, ax = plt.subplots()
    ax.plot(steps, rewards, label="PPO mean episode reward")
    ax.axhline(
        report.random_mean_reward,
        color="grey",
        linestyle="--",
        label="random baseline mean",
    )
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean episode reward")
    ax.set_title("Learnability: PPO vs random")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return True
