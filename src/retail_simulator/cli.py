"""``retail-sim`` — the Typer CLI for the retail-simulator.

* ``check-env``         — run ``gymnasium.utils.env_checker.check_env`` -> [OK]/[FAIL]
* ``train``             — SB3 PPO; write ``runs/<run-id>/{config.json,
                          metrics.jsonl, metrics.csv, policy.zip, run_summary.txt}``
* ``sweep``             — analytical pricing sweep; write a report under ``runs/``
* ``learnability-check``— run the single-agent gate; print the margin + PASS/FAIL
* ``validate``          — load + validate a scenario YAML (aggregated errors) (0.5)
* ``calibrate``         — measure each NPC archetype's beat-random margin (0.5)
* ``competition``       — the multi-agent competition sub-app (run|replay|
                          register|leaderboard) (0.5)
* ``version``           — print the package version

Layer note: the CLI sits in the harness tier (``cli -> harness -> envs -> core``)
and may import gymnasium/sb3 — always through ``RetailEnv``/``World``/the harness/
``scenarios``, never reimplementing world/gate/loader logic (the 0.5 commands are a
THIN surface: ``validate`` routes through ``scenarios.load_scenario``, ``calibrate``
through ``harness.calibrate``, ``competition`` through ``harness.parallel_gate``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import typer

from retail_simulator import __version__

if TYPE_CHECKING:
    # Type-only import for `_build_serve_config`'s return annotation. The real
    # import lives inside command bodies so the live wrapper (and its lazy
    # `websockets` contract) is not paid for at CLI module-import time.
    from collections.abc import Callable

    from retail_simulator.core.config import CoreConfig
    from retail_simulator.harness.ladder import LadderResult
    from retail_simulator.live import SeriesResult, ServeConfig, ServeResult

# `DEFAULT_LEAGUE_NAME` is needed as the default value of the `league train`
# `--league` Typer option (Typer resolves Option defaults at function-definition
# time, so the constant must be importable at module load). The league module is
# lazy w.r.t. SB3/SuperSuit — importing this name does NOT pull the [train] extra.
from retail_simulator.harness.league import DEFAULT_LEAGUE_NAME

# `LADDER_ARCHETYPE_LABELS` (RLB-8b) is needed to build `league ladder`'s
# `--participants` help text at module-definition time (same Typer-defaults-
# resolved-at-def-time constraint as `DEFAULT_LEAGUE_NAME` above), so the name
# list can never drift from the real allowlist. `harness.ladder`'s own
# module-level imports are core + spike_archetypes + geo_archetypes only (no
# envs/SB3) — importing this name is as cheap as the league one above.
from retail_simulator.harness.ladder import LADDER_ARCHETYPE_LABELS

# Global state populated by the root callback and read by subcommands. Typer has
# no first-class "context object" we need beyond these few resolved options.
_state: dict[str, Any] = {"seed": 42, "quiet": False, "verbose": False, "run_dir": Path("runs")}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="retail-simulator CLI — check, train, sweep, validate, calibrate, and gate the env.",
)

competition_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Multi-agent competition — run|replay the self-play gate; register|leaderboard.",
)
app.add_typer(competition_app, name="competition")

league_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Phase 1.5 self-play tooling — league train|evaluate|generalization-diagnostic.",
)
app.add_typer(league_app, name="league")

live_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Phase 2.0 demo server — live multiplayer over WebSocket.",
)
app.add_typer(live_app, name="live")

_log = logging.getLogger("retail_simulator")


def _use_color() -> bool:
    """ """
    if _state["quiet"]:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


# ANSI codes used only for the status prefixes; gated behind _use_color().
_COLORS = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "reset": "\033[0m",
}


def _status(prefix: str, message: str, color: str) -> str:
    """Format a ``[PREFIX] message`` status line, colored only when allowed."""
    if _use_color():
        return f"{_COLORS[color]}{prefix}{_COLORS['reset']} {message}"
    return f"{prefix} {message}"


def _echo(message: str) -> None:
    """Write a result line to stdout (suppressed entirely under --quiet)."""
    if not _state["quiet"]:
        typer.echo(message)


def _configure_logging() -> None:
    """Attach a stderr handler at the resolved level (CLI owns logging setup).

    The library adds no handlers on import; here the CLI wires one so INFO/DEBUG
    progress goes to *stderr* (stdout is reserved for results). ``--quiet`` =>
    errors only; ``--verbose`` => DEBUG; default => INFO.
    """
    if _state["quiet"]:
        level = logging.ERROR
    elif _state["verbose"]:
        level = logging.DEBUG
    else:
        level = logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _log.handlers.clear()
    _log.addHandler(handler)
    _log.setLevel(level)
    _log.propagate = False


def _new_run_dir(kind: str) -> Path:
    """Create and return a fresh ``<run-dir>/<kind>-<timestamp>`` directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(_state["run_dir"]) / f"{kind}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@app.callback()
def main(
    seed: int = typer.Option(42, "--seed", help="Global RNG seed (default 42)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Errors only; no color/results."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG-level logging to stderr."),
    run_dir: Path = typer.Option(
        Path("runs"), "--run-dir", help="Root directory for run artifacts (default ./runs)."
    ),
) -> None:
    """Resolve global options and configure logging before any subcommand runs."""
    _state["seed"] = seed
    _state["quiet"] = quiet
    _state["verbose"] = verbose
    _state["run_dir"] = run_dir
    _configure_logging()


@app.command()
def version() -> None:
    """Print the package version."""
    _echo(__version__)


@app.command("check-env")
def check_env_command() -> None:
    """Run the Gymnasium env checker on ``RetailEnv`` -> [OK]/[FAIL], exit 0/1.

    Example: ``retail-sim check-env``
    """
    import warnings

    from gymnasium.utils.env_checker import check_env

    from retail_simulator.envs.gym_env import RetailEnv

    try:
        with warnings.catch_warnings():
            # The Box-action-space symmetry warning is expected (price box is
            # [0.5, 2.0] per contract); env_checker must still raise no error.
            warnings.simplefilter("ignore")
            check_env(RetailEnv())
    except Exception as exc:  # noqa: BLE001  (boundary: surface a clean failure, not a traceback)
        _echo(_status("[FAIL]", f"env_checker reported: {exc}", "red"))
        _echo("This is a bug in RetailEnv, not your agent.")
        raise typer.Exit(code=1) from exc
    _echo(_status("[OK]", "RetailEnv passes gymnasium.utils.env_checker.check_env", "green"))


@app.command()
def sweep(
    grid_points: int = typer.Option(
        11,
        "--grid-points",
        help="Grid points per lever (2-D price x marketing => grid_points^2 cells, "
        "plus a grid_points-cell assortment axis at the joint optimum).",
    ),
    ticks: int = typer.Option(1000, "--ticks", help="Ticks per cell per seed (incl. burn-in)."),
    seeds: int = typer.Option(3, "--seeds", help="Number of seeds to average over."),
) -> None:
    """ """
    from retail_simulator.harness.sweep import (
        assortment_axis_csv_rows,
        automation_axis_csv_rows,
        expansion_timing_csv_rows,
        format_sweep_table,
        loyalty_axis_csv_rows,
        promotion_axis_csv_rows,
        research_axis_csv_rows,
        run_sweep,
        scm_axis_csv_rows,
        sweep_csv_rows,
    )

    seed_tuple = tuple(range(_state["seed"], _state["seed"] + seeds))
    _log.info(
        "running sweep: 2-D %dx%d price x marketing + %d-cell assortment axis "
        "+ %d-cell promotion axis (steady state) present-in-both-regions "
        "+ expansion-timing leg x %d ticks x %d seeds",
        grid_points,
        grid_points,
        grid_points,
        grid_points,
        ticks,
        seeds,
    )
    result = run_sweep(
        grid_points=grid_points,
        ticks_per_point=ticks,
        seeds=seed_tuple,
        start_present_both=True,
    )

    table = format_sweep_table(result)
    _echo(table)

    run_dir = _new_run_dir("sweep")
    (run_dir / "sweep_report.txt").write_text(table + "\n", encoding="utf-8")
    _write_csv(run_dir / "sweep.csv", sweep_csv_rows(result))
    _write_csv(run_dir / "sweep_assortment.csv", assortment_axis_csv_rows(result))
    _write_csv(run_dir / "sweep_promotion.csv", promotion_axis_csv_rows(result))
    # The expansion-timing leg's CSV (one row per full-trajectory policy). Empty
    # when the leg was not run (single region / disabled) — _write_csv handles [].
    _write_csv(run_dir / "sweep_expansion_timing.csv", expansion_timing_csv_rows(result))
    # Phase 1.0: the research (value-of-information) leg's CSV (one row per research
    # spend vs the Balanced NPC). Empty when the leg was not run — _write_csv handles [].
    _write_csv(run_dir / "sweep_research.csv", research_axis_csv_rows(result))
    # Phase 1.1: the SCM (value-of-service) leg's CSV (one row per service_level vs
    # the Discounter in the SCM scenario). Empty when the leg was not run / single-
    # region — _write_csv handles [].
    _write_csv(run_dir / "sweep_scm.csv", scm_axis_csv_rows(result))
    # Phase 1.2: the automation (value-of-automation) leg's CSV (one row per constant
    # tier vs the Discounter in the automation scenario, each carrying the steady-state
    # seated automation_tier (the upgrade-timing signal under F3=PERMANENT)). Empty when the leg was not run —
    # _write_csv handles [].
    _write_csv(run_dir / "sweep_automation.csv", automation_axis_csv_rows(result))
    # Phase 1.3: the loyalty-program leg's CSV (one row per constant loyalty_spend
    # cell vs the Discounter in the loyalty scenario). Empty when the leg was not
    # run — _write_csv handles [].
    _write_csv(run_dir / "sweep_loyalty.csv", loyalty_axis_csv_rows(result))
    _log.info("sweep artifacts written to %s", run_dir)

    if result.passed:
        _echo(
            _status(
                "PASS",
                "no dominant/degenerate strategy (every lever incl. expansion-timing "
                "+ research value-of-information + SCM + Phase-1.2 automation pays)",
                "green",
            )
        )
    else:
        _echo(_status("FAIL", "degenerate/dominant strategy detected", "red"))
        raise typer.Exit(code=1)


@app.command("learnability-check")
def learnability_check(
    quick: bool = typer.Option(False, "--quick", help="~200k-step smoke (CI); else full ~2M."),
    total_steps: int = typer.Option(
        0, "--total-steps", help="Override the training budget (0 = use --quick/full default)."
    ),
    eval_episodes: int = typer.Option(20, "--eval-episodes", help="Eval episodes per policy."),
    n_envs: int = typer.Option(
        1, "--n-envs", help="Parallel training envs (SubprocVecEnv) — multiplies throughput."
    ),
) -> None:
    """ """
    from retail_simulator.harness.learnability import (
        FULL_TOTAL_STEPS,
        QUICK_TOTAL_STEPS,
        run_gate,
    )

    if total_steps > 0:
        steps = total_steps
    elif quick:
        steps = QUICK_TOTAL_STEPS
    else:
        steps = FULL_TOTAL_STEPS

    run_dir = _new_run_dir("learnability")
    _log.info(
        "running learnability gate: %d steps, %d eval episodes, %d env(s)",
        steps,
        eval_episodes,
        n_envs,
    )
    report = run_gate(
        total_steps=steps,
        eval_episodes=eval_episodes,
        seed=_state["seed"],
        n_envs=n_envs,
        report_path=run_dir / "gate_report.json",
        progress=_state["verbose"],
    )

    _echo(_format_gate_summary(report))
    _log.info("gate artifacts written to %s", run_dir)

    if report.passed:
        _echo(_status("PASS", f"PPO beats random by {report.margin:.2f} (>= 2 sigma)", "green"))
    else:
        _echo(
            _status(
                "FAIL",
                f"margin {report.margin:.2f} vs threshold {report.threshold:.2f} "
                f"(beats_random={report.beats_random}, trends_up={report.curve_trends_up})",
                "red",
            )
        )
        raise typer.Exit(code=1)


@app.command()
def validate(
    scenario: Path = typer.Argument(..., help="Path to the scenario YAML to validate."),
) -> None:
    """ """
    from retail_simulator.scenarios import ScenarioConfig
    from retail_simulator.scenarios.loader import ScenarioValidationError, load_scenario

    try:
        core_config = load_scenario(scenario)
    except ScenarioValidationError as exc:
        # Aggregated, numeric per-field message (the loader already formatted it). To
        # stderr (it is an error/diagnostic, not a result) + a FAIL status line.
        typer.echo(str(exc), err=True)
        _echo(_status("FAIL", f"scenario {scenario} is invalid", "red"))
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        _echo(_status("FAIL", f"scenario file not found: {scenario}", "red"))
        raise typer.Exit(code=1) from exc

    # Re-derive the operator-facing summary from the validated ScenarioConfig (the
    # same object the loader built). The CoreConfig is the load-bearing artifact; the
    # ScenarioConfig surface is re-parsed here only to report the operator's fields.
    surface = ScenarioConfig.from_yaml(scenario)
    n_regions = len(core_config.demand.regions) if core_config.demand.regions else 1
    _echo(_status("INFO", f"scenario {surface.name!r}", "blue"))
    _echo(_format_scenario_summary(surface, n_regions))
    _echo(_status("PASS", f"scenario {scenario} is valid", "green"))


def _format_scenario_summary(surface: Any, n_regions: int) -> str:
    """Render the validated scenario's operator-facing fields as a small ASCII table."""
    archetypes = ", ".join(surface.npc_archetypes) if surface.npc_archetypes else "(none)"
    rows = [
        ("name", str(surface.name)),
        ("seed", str(surface.seed)),
        ("n_regions", str(n_regions)),
        ("n_learning_agents", str(surface.n_learning_agents)),
        ("npc_archetypes", archetypes),
        ("reward mode", str(surface.reward_mode)),
    ]
    lines = [f"{'field':>18} {'value':>20}", "-" * 39]
    lines += [f"{label:>18} {value:>20}" for label, value in rows]
    return "\n".join(lines)


@app.command()
def calibrate(
    seeds: int = typer.Option(5, "--seeds", help="Seeds averaged per archetype (taming noise)."),
    ticks: int = typer.Option(104, "--ticks", help="Scoring horizon per rollout (~2 years)."),
) -> None:
    """ """
    from retail_simulator.harness.calibrate import calibrate_all

    seed_tuple = tuple(range(_state["seed"], _state["seed"] + seeds))
    _log.info("calibrating NPC archetypes: %d seeds x %d ticks on the fast seam", seeds, ticks)
    results = calibrate_all(seeds=seed_tuple, horizon=ticks)

    _echo(_format_calibration_table(results))
    out_of_band = [name for name, cal in results.items() if not cal.in_band]
    if out_of_band:
        _echo(
            _status(
                "WARN",
                f"archetype(s) {out_of_band} are out of the 20-65% competence band "
                "(tune their lever constants; this is a report, not a gate failure)",
                "yellow",
            )
        )
    else:
        _echo(_status("INFO", "all archetypes within the 20-65% competence band", "blue"))


def _format_calibration_table(results: dict[str, Any]) -> str:
    """Render the per-archetype beat-random margins as an ASCII table (DX convention)."""
    header = (
        f"{'archetype':>12} {'metric':>14} {'archetype':>12} "
        f"{'random':>12} {'margin':>10} {'in band':>9}"
    )
    lines = [header, "-" * len(header)]
    for name, cal in results.items():
        margin_str = "inf" if cal.margin == float("inf") else f"{cal.margin:.3f}"
        lines.append(
            f"{name:>12} {cal.primary_metric:>14} {cal.archetype_score:>12.4f} "
            f"{cal.random_score:>12.4f} {margin_str:>10} {str(cal.in_band):>9}"
        )
    return "\n".join(lines)


@app.command()
def train(
    total_steps: int = typer.Option(100_000, "--total-steps", help="PPO training timesteps."),
    max_episode_steps: int = typer.Option(
        104, "--max-episode-steps", help="TimeLimit horizon (scoring window)."
    ),
    n_envs: int = typer.Option(
        1, "--n-envs", help="Parallel training envs (SubprocVecEnv) — multiplies throughput."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan and exit without training."
    ),
) -> None:
    """ """
    import gymnasium as gym
    from stable_baselines3 import PPO

    import retail_simulator  # noqa: F401  (registers RetailSim-v0)

    if dry_run:
        _echo(
            f"[dry-run] would train PPO for {total_steps} steps "
            f"(max_episode_steps={max_episode_steps}, seed={_state['seed']}, n_envs={n_envs})"
        )
        return

    run_dir = _new_run_dir("train")
    _write_config_receipt(run_dir, total_steps, max_episode_steps)

    # Single env for the post-training metrics rollout (one deterministic trajectory).
    env = gym.make("RetailSim-v0", max_episode_steps=max_episode_steps)
    if n_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        train_env: Any = make_vec_env(
            "RetailSim-v0",
            n_envs=n_envs,
            seed=_state["seed"],
            env_kwargs={"max_episode_steps": max_episode_steps},
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"},  # robust regardless of launcher
        )
    else:
        train_env = env
    _log.info("training PPO for %d steps (%d env(s)) -> %s", total_steps, n_envs, run_dir)
    model = PPO("MlpPolicy", train_env, seed=_state["seed"], verbose=1 if _state["verbose"] else 0)
    model.learn(total_timesteps=total_steps, progress_bar=False)
    if train_env is not env:
        train_env.close()
    model.save(str(run_dir / "policy.zip"))

    summary = _rollout_and_record_metrics(model, env, run_dir, max_episode_steps)
    env.close()

    (run_dir / "run_summary.txt").write_text(summary, encoding="utf-8")
    _echo(_status("[OK]", f"trained policy + metrics written to {run_dir}", "green"))
    _echo(str(run_dir))


def _rollout_and_record_metrics(
    model: Any,
    env: Any,
    run_dir: Path,
    max_episode_steps: int,
) -> str:
    """ """
    import time

    fieldnames = [
        "tick",
        "episode",
        "episode_step",
        "reward",
        "revenue",
        "profit",
        "market_share",
        "loyalty",
        "cash",
        "stores",
        "price_index",
        "assortment",
        "promotion",
        "stockpile",
        "expansion_action",
        "research_action",
        "research_fidelity",
        # Phase 1.1: the SCM action this tick + the recovered per-region stockout
        # rate + the seated per-region service_score (the 4th seated dynamic, the
        # per-region liability — pure reads via enrich_agent_info; no RNG).
        "service_level_action",
        "stockout_rate_region_0",
        "stockout_rate_region_1",
        "service_score_region_0",
        "service_score_region_1",
        "automation_tier_action",
        "automation_tier",
        "last_automation_tier_action",
        "automation_capex",
        # Phase 1.3: the decoded active-loyalty-program intensity this tick (the
        # action taken; 0.0 at the byte-identity default). NO new seated obs —
        # the existing per-region loyalty stocks already surface the moat.
        "loyalty_spend_action",
        "npc_market_share",
        "seed",
        "algo",
        "wall_time",
    ]
    rows: list[dict[str, Any]] = []
    obs, info = env.reset(seed=_state["seed"])
    done = False
    total_reward = 0.0
    start = time.time()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        rows.append(
            {
                "tick": int(info.get("tick", 0)),
                "episode": 0,
                "episode_step": int(info.get("episode_step", 0)),
                "reward": float(reward),
                "revenue": float(info.get("revenue", 0.0)),
                "profit": float(info.get("profit", 0.0)),
                "market_share": float(info.get("market_share", 0.0)),
                "loyalty": float(info.get("loyalty", 0.0)),
                "cash": float(info.get("cash", 0.0)),
                # Phase 0.4: the env's `stores` is now the MULTI-REGION open-store
                # count (sum across regions), so this column tracks expansion.
                "stores": int(info.get("stores", 0)),
                "price_index": float(np.asarray(action).reshape(-1)[0]),
                # Phase 0.2: the decoded assortment breadth this tick, surfaced by
                # the env in info (the action taken this tick; 0.0 at reset).
                "assortment": float(info.get("assortment", 0.0)),
                # Phase 0.3: the decoded promotion intensity this tick (the action
                # taken this tick; 0.0 at reset) and the seated regional stockpile
                # debt (read off the next state; surfaces the intertemporal effect).
                "promotion": float(info.get("promotion", 0.0)),
                "stockpile": float(info.get("stockpile", 0.0)),
                # Phase 0.4: the decoded expansion choice this tick (0 = no-op, r =
                # open region r), surfaced by the env in info (the action taken this
                # tick; 0.0 at reset) — the multi-region story alongside `stores`.
                "expansion_action": float(info.get("expansion_action", 0.0)),
                # Phase 1.0: the decoded research spend this tick + the seated
                # per-observer perception fidelity, surfaced by the env in info (the
                # research metrics for metrics.jsonl; 0.0 at reset).
                "research_action": float(info.get("research_action", 0.0)),
                "research_fidelity": float(info.get("research_fidelity", 0.0)),
                # Phase 1.1: the SCM action this tick + the recovered per-region
                # stockout rate + the seated per-region service_score (the per-
                # region liability — pure reads via enrich_agent_info; no RNG;
                # 1.0/0.0 sentinels at reset).
                "service_level_action": float(info.get("service_level_action", 1.0)),
                "stockout_rate_region_0": float(info.get("stockout_rate_region_0", 0.0)),
                "stockout_rate_region_1": float(info.get("stockout_rate_region_1", 0.0)),
                "service_score_region_0": float(info.get("service_score_region_0", 1.0)),
                "service_score_region_1": float(info.get("service_score_region_1", 1.0)),
                "automation_tier_action": float(info.get("automation_tier_action", 0.0)),
                "automation_tier": int(info.get("automation_tier", 0)),
                "last_automation_tier_action": int(info.get("last_automation_tier_action", 0)),
                "automation_capex": float(info.get("automation_capex", 0.0)),
                "loyalty_spend_action": float(info.get("loyalty_spend_action", 0.0)),
                "npc_market_share": float(1.0 - float(info.get("market_share", 0.0))),
                "seed": int(_state["seed"]),
                "algo": "PPO",
                "wall_time": round(time.time() - start, 6),
            }
        )
        done = bool(terminated) or bool(truncated)

    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    _write_csv(run_dir / "metrics.csv", rows, fieldnames=fieldnames)

    return (
        f"algo: PPO\nseed: {_state['seed']}\nmax_episode_steps: {max_episode_steps}\n"
        f"eval_episode_steps: {len(rows)}\neval_total_reward: {total_reward:.4f}\n"
        f"final_cash: {rows[-1]['cash'] if rows else 0.0:.2f}\n"
    )


def _write_config_receipt(run_dir: Path, total_steps: int, max_episode_steps: int) -> None:
    """Write ``config.json`` — the reproducibility receipt (seed + versions)."""
    import gymnasium
    import numpy
    import stable_baselines3

    from retail_simulator.core.config import CoreConfig

    receipt = {
        "seed": _state["seed"],
        "total_steps": total_steps,
        "max_episode_steps": max_episode_steps,
        "algo": "PPO",
        "policy": "MlpPolicy",
        "env_id": "RetailSim-v0",
        "core_config": asdict(CoreConfig.default()),
        "versions": {
            "retail_simulator": __version__,
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write a list of flat dicts as CSV (pandas-readable). Empty => header-less."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames if fieldnames is not None else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _format_gate_summary(report: Any) -> str:
    """ """
    lines = [
        f"{'metric':>22} {'value':>14}",
        "-" * 38,
        f"{'PPO mean reward':>22} {report.ppo_mean_reward:>14.2f}",
        f"{'random mean reward':>22} {report.random_mean_reward:>14.2f}",
        f"{'random std reward':>22} {report.random_std_reward:>14.2f}",
        f"{'margin':>22} {report.margin:>14.2f}",
        f"{'threshold (2 sigma)':>22} {report.threshold:>14.2f}",
        f"{'beats random':>22} {str(report.beats_random):>14}",
        f"{'curve trends up':>22} {str(report.curve_trends_up):>14}",
        f"{'total steps':>22} {report.total_steps:>14}",
    ]
    uptake = getattr(report, "expansion_uptake", None)
    if uptake is not None:
        median = uptake.median_open_tick
        median_str = f"{median:.1f}" if median is not None else "never"
        lines.append(f"{'expansion uptake':>22} {uptake.fraction_opening:>14.3f}")
        lines.append(f"{'median open tick':>22} {median_str:>14}")
        if uptake.fraction_opening == 0.0:
            # A passing gate where PPO never opens region 1 means the lever is
            # learnable-to-ignore — a CALIBRATION SMELL to surface, not an auto-fail.
            lines.append(
                "note: PPO never opened region 1 (uptake 0.0) — expansion is "
                "learnable-to-ignore; a calibration smell to review (not a gate failure)"
            )
    return "\n".join(lines)


@app.command("play-human")
def play_human(
    seed: int = typer.Option(0, "--seed", help="Seed: picks the hidden opponent + the world."),
    ticks: int = typer.Option(50, "--ticks", help="Game length in ticks (default 50)."),
    debrief: Path = typer.Option(
        Path("hotseat_debrief.jsonl"),
        "--debrief",
        help="Local .jsonl file the game record is appended to.",
    ),
) -> None:
    """ """
    # Lazy: keep the cheap CLI-import path free of the env/pettingzoo stack (mirrors the
    # other heavy subcommands; the hot-seat driver itself lazily imports the seam).
    from retail_simulator.harness.hotseat import (
        append_debrief,
        play_hotseat_game,
        print_rps_primer,
    )
    from retail_simulator.harness.spike import calibrated_spike_config

    if ticks < 1:
        _echo(_status("FAIL", f"--ticks must be >= 1, got {ticks}", "red"))
        raise typer.Exit(code=2)

    def _input(prompt: str) -> str:
        return input(prompt)

    def _output(message: str) -> None:
        typer.echo(message)

    print_rps_primer(_output)
    result = play_hotseat_game(
        config=calibrated_spike_config(),
        seed=seed,
        n_ticks=ticks,
        input_fn=_input,
        output_fn=_output,
    )
    append_debrief(result, debrief)
    _echo(_status("PASS", f"game recorded to {debrief}", "green"))


@app.command("serve-web")
def serve_web(
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP bind host (default 127.0.0.1)."),
    port: int = typer.Option(8000, "--port", help="HTTP bind port (default 8000)."),
    dist_dir: Path = typer.Option(
        None,
        "--dist-dir",
        help=(
            "Built SPA dir to serve same-origin (default: frontend/dist at the repo root). "
            "Build it first with `npm --prefix frontend run build`. If absent, the API still "
            "serves and `/` shows a 'not built' note."
        ),
    ),
) -> None:
    """Serve the web platform: the built SPA + the FastAPI JSON API as ONE same-origin app.

    Runs uvicorn over ``retail_simulator.web.app:app`` — the FastAPI transport wrapping the
    in-memory session core. When a built ``frontend/dist`` exists, the SPA is served at
    ``/`` (with ``/api`` precedence + a history fallback so a hard refresh on ``/board`` /
    ``/docs`` boots the app); when it is absent, only the JSON API serves and ``/`` shows a
    friendly "not built" note. ``uvicorn`` is imported LAZILY inside the command body so the
    rest of the CLI (and ``import retail_simulator.web``) never pays for the heavy ``[web]``
    extra unless an operator actually serves.

    BUILD FIRST: ``npm --prefix frontend run build`` (emits ``frontend/dist``), then
    ``retail-sim serve-web``. ``--dist-dir`` overrides where the SPA is read from.

    SINGLE-PROCESS ONLY: the session manager holds all live games in memory, so the server
    is pinned to ``workers=1`` (multiple workers would each own a disjoint, unreachable
    session registry — a started game would be unreachable from a later request landing on
    another worker). This is a hard product constraint, not a default.

    Example: ``retail-sim serve-web --host 127.0.0.1 --port 8000``
    """
    # Lazy: uvicorn (the heavy [web] extra) is paid for only when an operator serves.
    import uvicorn

    # The app is created via the import string (uvicorn owns its lifecycle), so pass the
    # dist dir to it through the env var the factory reads — keeps the import string clean
    # and the lazy-import contract intact (no FastAPI imported here).
    if dist_dir is not None:
        os.environ["RETAIL_SIM_WEB_DIST"] = str(dist_dir)

    _log.info("serving web platform on http://%s:%d (single-process, workers=1)", host, port)
    # workers=1 is load-bearing — the in-memory SessionManager is per-process (see the
    # web.app module docstring). The import string lets uvicorn own the app lifecycle.
    uvicorn.run("retail_simulator.web.app:app", host=host, port=port, workers=1)


_PER_AGENT_METRIC_FIELDS: tuple[str, ...] = (
    "reward",
    "profit",
    "market_share",
    "loyalty",
    "cash",
    "stores",
    "expansion_action",
    # Phase 1.0: the research metrics (the decoded research spend + the seated
    # perception fidelity), consistent with the per-lever metric fields.
    "research_action",
    "research_fidelity",
    # Phase 1.1: per-agent SCM metrics — the decoded service this tick + the
    # recovered per-region stockout rate + the seated per-region service_score.
    "service_level_action",
    "stockout_rate_region_0",
    "stockout_rate_region_1",
    "service_score_region_0",
    "service_score_region_1",
    "automation_tier_action",
    "automation_tier",
    "last_automation_tier_action",
    "automation_capex",
    "loyalty_spend_action",
)


def _scenario_seat_plan(scenario: Path | None) -> tuple[int, str]:
    """Resolve ``(n_learning_agents, reward_mode)`` from a scenario YAML (or defaults).

    Routes through ``scenarios.load_scenario`` so the CLI reimplements NO loading. The
    multi-agent gate (``harness.parallel_gate.run_parallel_gate``) currently parametrizes
    on ``n_learning_agents`` + ``reward_mode`` only (it builds its own competition
    ``CoreConfig``), so a ``--scenario`` threads those two derived fields; the full
    per-region economics of a custom scenario are NOT yet forwarded to the gate (a
    harness limitation, reported — the CLI does not work around it). Absent a scenario,
    returns the competition defaults (2 learning agents, weighted reward).
    """
    if scenario is None:
        from retail_simulator.harness.parallel_gate import DEFAULT_N_LEARNING_AGENTS

        return DEFAULT_N_LEARNING_AGENTS, "weighted"

    from retail_simulator.scenarios import ScenarioConfig
    from retail_simulator.scenarios.loader import load_scenario

    load_scenario(scenario)  # validate at the boundary (raises ScenarioValidationError)
    surface = ScenarioConfig.from_yaml(scenario)
    return surface.n_learning_agents, surface.reward_mode


@competition_app.command("run")
def competition_run(
    scenario: Path = typer.Option(
        None, "--scenario", help="Scenario YAML (threads n_learning_agents + reward_mode)."
    ),
    total_steps: int = typer.Option(
        0, "--total-steps", help="Override the shared-policy training budget (0 = default)."
    ),
    n_envs: int = typer.Option(
        1, "--n-envs", help="Parallel vec env copies (num_vec_envs) — multiplies throughput."
    ),
    eval_episodes: int = typer.Option(20, "--eval-episodes", help="Eval episodes per policy."),
    quick: bool = typer.Option(False, "--quick", help="~200k-step smoke (CI); else full ~2M."),
    regional_breakdown: bool = typer.Option(
        False,
        "--regional-breakdown",
        help="Add a per-region split to the per-agent metrics.jsonl (opt-in; off by default).",
    ),
    research_gate: bool = typer.Option(
        False,
        "--research-gate",
        help="Phase 1.0 value-of-information gate: PPO WITH research vs the SAME policy with research FORCED OFF, in a VARYING-opponent substrate ((B)).",
    ),
    research_opponent: str = typer.Option(
        "wandering",
        "--research-opponent",
        help="The varying opponent the research gate runs against (NEVER the constant discounter): default 'wandering' (exogenous, non-inferable —); 'balanced' is the partly-inferable reported-secondary.",
    ),
    scm_gate: bool = typer.Option(
        False,
        "--scm-gate",
        help="Phase 1.1 value-of-SCM gate: PPO WITH SCM vs the SAME policy with SCM FORCED-TO-DEFAULT (service_level=1.0) in the SCM scenario substrate (scm_gate_config — cogs_premium>0 AND lost_sales_share_penalty>0 AND CONVENIENCE-beta_service>0) vs the default Discounter (-B of).",
    ),
    automation_gate: bool = typer.Option(
        False,
        "--automation-gate",
        help="Phase 1.2 value-of-automation gate (F3 = PERMANENT): PPO WITH automation vs the SAME policy with the 3-wide automation block FORCED-TO-TIER-0 (the no-upgrade baseline + the byte-identity anchor) in the automation scenario substrate (automation_gate_config — capex_per_tier>0 AND savings_per_tier>0) vs the default Discounter (-B of).",
    ),
    loyalty_gate: bool = typer.Option(
        False,
        "--loyalty-gate",
        help="Phase 1.3 value-of-active-loyalty-program gate: PPO WITH loyalty vs the SAME policy with loyalty_spend FORCED-TO-0.0 (the no-program baseline + the byte-identity anchor) in the loyalty scenario substrate (loyalty_gate_config — cost_per_unit_spend>0 AND BRAND_LOYAL beta_program>0) vs the default Discounter (-B of).",
    ),
    curriculum: str = typer.Option(
        "",
        "--curriculum",
        help="Phase 1.4 (F4-A + F5-A): training-time scenario curriculum name (e.g. 'phase_1_4_basic' — the 6-segment 3-variant curriculum). Training samples a fresh CoreConfig from the named distribution at each episode reset (a SEPARATE rng — NEVER the core PCG64); the eval / gate scenario is unchanged. Empty (the default) = no curriculum (1.3 CLI behavior).",
    ),
    policy_kind: str = typer.Option(
        "mlp",
        "--policy-kind",
        help="Phase 1.6 (F-LOCUS=A): policy class — 'mlp' (default; vanilla SB3 PPO, preserves every 0–1.5 byte-identity) or 'lstm' (RecurrentPPO with MlpLstmPolicy; requires the [recurrent] extra — sb3-contrib). The lstm path is opt-in; the default 'mlp' threads through to harness.parallel_gate._make_shared_ppo unchanged.",
    ),
) -> None:
    """ """
    from retail_simulator.harness.curriculum import (
        ScenarioDistribution,
        build_curriculum,
        known_curricula,
    )
    from retail_simulator.harness.learnability import FULL_TOTAL_STEPS, QUICK_TOTAL_STEPS
    from retail_simulator.harness.parallel_gate import diversity_report, run_parallel_gate

    n_learning_agents, reward_mode = _scenario_seat_plan(scenario)
    if policy_kind not in ("mlp", "lstm"):
        _echo(
            _status(
                "[FAIL]",
                f"unknown policy-kind {policy_kind!r}; expected 'mlp' or 'lstm'",
                "red",
            )
        )
        raise typer.Exit(code=1)
    selected_curriculum: ScenarioDistribution | None
    if curriculum:
        try:
            selected_curriculum = build_curriculum(curriculum)
        except ValueError as exc:
            _echo(_status("[FAIL]", str(exc), "red"))
            _echo(f"Known curricula: {list(known_curricula())}")
            raise typer.Exit(code=1) from exc
    else:
        selected_curriculum = None

    if total_steps > 0:
        steps = total_steps
    elif quick:
        steps = QUICK_TOTAL_STEPS
    else:
        steps = FULL_TOTAL_STEPS

    run_dir = _new_run_dir("competition")
    if research_gate:
        gate_label = "research-on-vs-off"
    elif scm_gate:
        gate_label = "scm-on-vs-off"
    elif automation_gate:
        gate_label = "automation-on-vs-off"
    elif loyalty_gate:
        gate_label = "loyalty-on-vs-off"
    else:
        gate_label = "self-play vs random"
    curriculum_suffix = f", curriculum={curriculum}" if curriculum else ""
    _log.info(
        "running multi-agent gate (%s): %d steps, %d eval episodes, %d learning agent(s), "
        "%d vec env(s), reward=%s%s%s",
        gate_label,
        steps,
        eval_episodes,
        n_learning_agents,
        n_envs,
        reward_mode,
        f", opponent={research_opponent}" if research_gate else "",
        curriculum_suffix,
    )
    policy_kind_literal = cast(Literal["mlp", "lstm"], policy_kind)
    report = run_parallel_gate(
        total_steps=steps,
        eval_episodes=eval_episodes,
        n_learning_agents=n_learning_agents,
        seed=_state["seed"],
        num_vec_envs=n_envs,
        reward_mode=reward_mode,
        research_gate=research_gate,
        research_opponent=research_opponent,
        scm_gate=scm_gate,
        automation_gate=automation_gate,
        loyalty_gate=loyalty_gate,
        curriculum=selected_curriculum,
        policy_kind=policy_kind_literal,
    )

    # Write the gate report (the reused GateReport.to_dict — JSON-serializable).
    (run_dir / "gate_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    # Per-agent metrics.jsonl: a thin eval rollout of a RANDOM field through the
    # parallel env (the gate returns only the report, not the trained model — so the
    # recorded trajectory is the random baseline's, sufficient to honor the per-agent
    # schema contract; the gate's learned scores live in gate_report.json).
    _write_competition_metrics(
        run_dir,
        n_learning_agents=n_learning_agents,
        reward_mode=reward_mode,
        regional_breakdown=regional_breakdown,
    )

    _echo(_format_gate_summary(report))
    roster = {"trained_selfplay": report.ppo_mean_reward, "random": report.random_mean_reward}
    _echo(_format_diversity_report(diversity_report(roster)))
    _log.info("competition artifacts written to %s", run_dir)

    if report.passed:
        if research_gate:
            winner, beaten = "research-on", "research-off"
        elif scm_gate:
            winner, beaten = "scm-on", "scm-off-at-default"
        elif automation_gate:
            winner, beaten = "automation-on", "automation-off-at-tier-0"
        elif loyalty_gate:
            winner, beaten = "loyalty-on", "loyalty-off-at-zero"
        else:
            winner, beaten = "self-play", "random"
        _echo(
            _status(
                "PASS",
                f"{winner} beats {beaten} by {report.margin:.2f} (>= 2 sigma)",
                "green",
            )
        )
    else:
        _echo(
            _status(
                "FAIL",
                f"margin {report.margin:.2f} vs threshold {report.threshold:.2f} "
                f"(beats_random={report.beats_random}, trends_up={report.curve_trends_up})",
                "red",
            )
        )
        raise typer.Exit(code=1)


@competition_app.command("replay")
def competition_replay(
    report: Path = typer.Argument(
        ..., help="A gate_report.json (or a competition run dir containing one)."
    ),
) -> None:
    """Re-print a saved competition gate report (no training) — exit 0 / 1 on PASS/FAIL.

    Reads a ``gate_report.json`` written by ``competition run`` (a run DIR is also
    accepted — it resolves ``<dir>/gate_report.json``) and re-prints the headline
    numbers + PASS/FAIL. Pure I/O: no env, no training, no gate logic. Exits 0 if the
    saved report PASSED, 1 if it FAILED, so replay is scriptable in CI.

    Example: ``retail-sim competition replay runs/competition-.../gate_report.json``
    """
    report_path = report / "gate_report.json" if report.is_dir() else report
    if not report_path.exists():
        typer.echo(f"gate report not found: {report_path}", err=True)
        _echo(_status("FAIL", f"no gate report at {report_path}", "red"))
        raise typer.Exit(code=1)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    _echo(_format_saved_gate_report(data))
    if bool(data.get("passed")):
        _echo(_status("PASS", f"saved report at {report_path} PASSED", "green"))
    else:
        _echo(_status("FAIL", f"saved report at {report_path} FAILED", "red"))
        raise typer.Exit(code=1)


@competition_app.command("register")
def competition_register(
    policy: Path = typer.Argument(..., help="A trained policy.zip to register as an entrant."),
) -> None:
    """ """
    _echo(
        _status(
            "INFO",
            f"register is a Phase-2 feature (persistent competition roster); {policy} "
            "was NOT registered. 0.5 has no persistence — use `competition run` + "
            "`competition replay` (gate_report.json) instead.",
            "blue",
        )
    )


@competition_app.command("leaderboard")
def competition_leaderboard() -> None:
    """ """
    _echo(
        _status(
            "INFO",
            "leaderboard is a Phase-2 feature (it ranks a persistent entrant roster, "
            "which 0.5 does not build). Use `competition replay <gate_report.json>` for a "
            "run's standing, or `retail-sim calibrate` for archetype competence.",
            "blue",
        )
    )


def _write_competition_metrics(
    run_dir: Path,
    *,
    n_learning_agents: int,
    reward_mode: str,
    regional_breakdown: bool,
) -> None:
    """ """
    from retail_simulator.core.config import CoreConfig, RewardConfig
    from retail_simulator.envs import RetailParallelEnv

    config = CoreConfig(reward=RewardConfig(mode=reward_mode))
    env = RetailParallelEnv(config, n_learning_agents=n_learning_agents, seed=_state["seed"])
    action_space = env.action_space(env.possible_agents[0])
    action_space.seed(_state["seed"])

    observations, _infos = env.reset(seed=_state["seed"])
    horizon = 104  # the ~2-year scoring window (matches the gate's eval horizon)
    rows: list[dict[str, Any]] = []
    for _ in range(horizon):
        actions = {aid: action_space.sample() for aid in env.agents}
        observations, rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            info = infos[aid]
            row: dict[str, Any] = {
                "agent": aid,
                "tick": int(info.get("tick", 0)),
                "episode_step": int(info.get("episode_step", 0)),
                "reward": float(rewards[aid]),
                "profit": float(info.get("profit", 0.0)),
                "market_share": float(info.get("market_share", 0.0)),
                "loyalty": float(info.get("loyalty", 0.0)),
                "cash": float(info.get("cash", 0.0)),
                "stores": int(info.get("stores", 0)),
                "expansion_action": float(info.get("expansion_action", 0.0)),
                # Phase 1.0: the per-agent research metrics (decoded spend + seated
                # fidelity), surfaced by the env in each agent's info.
                "research_action": float(info.get("research_action", 0.0)),
                "research_fidelity": float(info.get("research_fidelity", 0.0)),
                # Phase 1.1: per-agent SCM metrics — the decoded service this tick +
                # the recovered per-region stockout rate + the seated per-region
                # service_score (pure reads via enrich_agent_info; no RNG).
                "service_level_action": float(info.get("service_level_action", 1.0)),
                "stockout_rate_region_0": float(info.get("stockout_rate_region_0", 0.0)),
                "stockout_rate_region_1": float(info.get("stockout_rate_region_1", 0.0)),
                "service_score_region_0": float(info.get("service_score_region_0", 1.0)),
                "service_score_region_1": float(info.get("service_score_region_1", 1.0)),
                # Phase 1.2 (F3 = PERMANENT): per-agent automation metrics — the
                # decoded TARGET tier this tick + the SEATED INTEGER tier (the
                # seam's authority, the SoT for the next-tick demand + the uptake
                # signal) + the last decoded target (reporting) + the DIFFERENTIAL
                # cash debited this tick (zero on a no-op upgrade).
                "automation_tier_action": float(info.get("automation_tier_action", 0.0)),
                "automation_tier": int(info.get("automation_tier", 0)),
                "last_automation_tier_action": int(info.get("last_automation_tier_action", 0)),
                "automation_capex": float(info.get("automation_capex", 0.0)),
                # Phase 1.3: the decoded active-loyalty-program intensity this
                # tick (the action taken; 0.0 at the byte-identity default).
                "loyalty_spend_action": float(info.get("loyalty_spend_action", 0.0)),
            }
            if regional_breakdown:
                row["regional_breakdown"] = _regional_breakdown_for_seat(env, aid)
            rows.append(row)
    env.close()

    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _regional_breakdown_for_seat(env: Any, agent_id: str) -> list[int]:
    """ """
    seat = env._seat_of[agent_id]
    retailer = env._state.retailers[seat]
    return [int(n) for n in retailer.stores_per_region]


def _format_diversity_report(report: Any) -> str:
    """"""
    ratio = "inf" if report.max_min_ratio == float("inf") else f"{report.max_min_ratio:.2f}"
    lines = [
        f"{'diversity (soft)':>22} {'value':>14}",
        "-" * 38,
        f"{'best score':>22} {report.best_score:>14.2f}",
        f"{'min score':>22} {report.min_score:>14.2f}",
        f"{'fraction within 80%':>22} {report.fraction_within:>14.3f}",
        f"{'count within 80%':>22} {report.n_within:>14}",
        f"{'max/min ratio':>22} {ratio:>14}",
        f"{'meets soft goal':>22} {str(report.meets_soft_goal):>14}",
        "note: diversity is a SOFT goal — REPORTED, never part of any gate's PASS/FAIL",
    ]
    return "\n".join(lines)


def _format_saved_gate_report(data: dict[str, Any]) -> str:
    """Render a saved ``gate_report.json`` dict as the gate summary table (replay)."""
    lines = [
        f"{'metric':>22} {'value':>14}",
        "-" * 38,
        f"{'trained mean':>22} {float(data.get('ppo_mean_reward', 0.0)):>14.2f}",
        f"{'random mean':>22} {float(data.get('random_mean_reward', 0.0)):>14.2f}",
        f"{'random std':>22} {float(data.get('random_std_reward', 0.0)):>14.2f}",
        f"{'margin':>22} {float(data.get('margin', 0.0)):>14.2f}",
        f"{'threshold (2 sigma)':>22} {float(data.get('threshold', 0.0)):>14.2f}",
        f"{'beats random':>22} {str(data.get('beats_random')):>14}",
        f"{'curve trends up':>22} {str(data.get('curve_trends_up')):>14}",
        f"{'total steps':>22} {int(data.get('total_steps', 0)):>14}",
        f"{'passed':>22} {str(data.get('passed')):>14}",
    ]
    return "\n".join(lines)


@league_app.command("train")
def league_train(
    pool_dir: Path = typer.Option(
        ...,
        "--pool-dir",
        help="On-disk pool directory (created if missing). The seeded archetypes + each "
        "round's SB3 checkpoint zip are written here; `league evaluate` reads it back.",
    ),
    rounds: int = typer.Option(
        3,
        "--rounds",
        help="Number of league rounds (>=1). Round-0 is the random-init snapshot (no "
        "`.learn` call — the AlphaStar canonical seed); rounds 1..N each train a fresh "
        "shared PPO against an opponent sampled from the GROWN pool.",
    ),
    train_steps_per_round: int = typer.Option(
        500_000,
        "--train-steps-per-round",
        help="PPO `learn` budget per round-1..N (round-0 does not call `.learn`).",
    ),
    league_seed: int = typer.Option(
        0,
        "--league-seed",
        help="Seeds the league's opponent-picker Generator (kept SEPARATE from the curriculum + core RNGs — three-RNG isolation contract).",
    ),
    curriculum_seed: int = typer.Option(
        0,
        "--curriculum-seed",
        help="Seeds the curriculum wrapper's harness Generator under F3-A composition.",
    ),
    core_seed: int = typer.Option(
        42,
        "--core-seed",
        help="Seeds the core PCG64 + the SB3 PPO constructor (training reproducibility).",
    ),
    curriculum: str = typer.Option(
        "",
        "--curriculum",
        help="Phase 1.4 curriculum NAME (e.g. 'phase_1_4_basic') to compose under the "
        "league (F3-A stacked). Empty (default) = no curriculum.",
    ),
    league: str = typer.Option(
        DEFAULT_LEAGUE_NAME,
        "--league",
        help="Named league (pool seed). Defaults to the F5-A seeded basic league.",
    ),
    distribution: str = typer.Option(
        "uniform",
        "--distribution",
        help=(
            "Opponent distribution: 'uniform' (default; backward compat with Slice B) "
            "or 'pfsp' (Slice C; win-rate-weighted via PFSPOpponentDistribution)."
        ),
    ),
    pfsp_alpha: float = typer.Option(
        2.0,
        "--pfsp-alpha",
        help=(
            "PFSP hardness exponent for (1-x)^alpha weighting; ignored under "
            "--distribution uniform."
        ),
    ),
    round_end_eval_episodes: int = typer.Option(
        5,
        "--round-end-eval-episodes",
        help=(
            "Episodes per opponent in the round-end head-to-head eval (populates "
            "the win-rate matrix)."
        ),
    ),
    reset_pool: bool = typer.Option(
        False,
        "--reset-pool",
        help=(
            "Before running, clear the pool dir of any prior win_rate_matrix.json "
            "+ round_*.zip artifacts (operator UX for fresh PFSP runs)."
        ),
    ),
    policy_kind: str = typer.Option(
        "mlp",
        "--policy-kind",
        help="Phase 1.6 (F-LOCUS=A): policy class for every per-round _make_shared_ppo call — 'mlp' (default; vanilla SB3 PPO, preserves every Slice C byte-identity) or 'lstm' (RecurrentPPO with MlpLstmPolicy; requires the [recurrent] extra — sb3-contrib). The lstm path is opt-in; the default 'mlp' threads through to harness.league.run_league_training unchanged.",
    ),
) -> None:
    """ """
    from retail_simulator.harness.curriculum import (
        ScenarioDistribution,
        build_curriculum,
        known_curricula,
    )
    from retail_simulator.harness.league import (
        FrozenPolicyPool,
        OpponentDistribution,
        PFSPOpponentDistribution,
        UniformOpponentDistribution,
        build_league,
        known_leagues,
        run_league_training,
    )

    # -- Boundary validation (T-C1-CLI) -------------------------------------------------
    # Unknown --distribution / bad --pfsp-alpha / non-positive --round-end-eval-episodes
    # are operator typos; surface them as clean CLI failures (NOT stack traces) before
    # we do any heavy work (build_league, run_league_training). Mirror the
    # `competition run --curriculum nonsuch` pattern: [FAIL] line + exit 1.
    if distribution not in ("uniform", "pfsp"):
        _echo(
            _status(
                "[FAIL]",
                f"unknown distribution {distribution!r}; expected 'uniform' or 'pfsp'",
                "red",
            )
        )
        raise typer.Exit(code=1)
    if policy_kind not in ("mlp", "lstm"):
        _echo(
            _status(
                "[FAIL]",
                f"unknown policy-kind {policy_kind!r}; expected 'mlp' or 'lstm'",
                "red",
            )
        )
        raise typer.Exit(code=1)
    if pfsp_alpha < 0:
        _echo(
            _status(
                "[FAIL]",
                f"--pfsp-alpha must be >= 0, got {pfsp_alpha}",
                "red",
            )
        )
        raise typer.Exit(code=1)
    if round_end_eval_episodes < 1:
        _echo(
            _status(
                "[FAIL]",
                (f"--round-end-eval-episodes must be >= 1, got {round_end_eval_episodes}"),
                "red",
            )
        )
        raise typer.Exit(code=1)

    # -- Reset-pool (T-C1-CLI / N1) ----------------------------------------------------
    # Clear any prior win-rate matrix + round_*.zip artifacts so a fresh PFSP run
    # starts from a clean win-rate history without manual cleanup. The manifest
    # (if present) is left alone — build_league below rebuilds the pool from
    # scratch against the resolved league_name.
    if reset_pool:
        pool_dir.mkdir(parents=True, exist_ok=True)
        wr_path = pool_dir / "win_rate_matrix.json"
        if wr_path.exists():
            wr_path.unlink()
        for zip_path in pool_dir.glob("round_*.zip"):
            zip_path.unlink()
        _echo(f"--reset-pool: cleared win_rate_matrix.json + round_*.zip from {pool_dir}")

    # Resolve the named league (build the seeded pool at the operator-supplied dir).
    # An unknown name raises ValueError listing the known names — surfaced as a clean
    # CLI failure, never a stack trace (boundary validation, mirrors `competition run`).
    try:
        pool = build_league(league, pool_dir)
    except ValueError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        _echo(f"Known leagues: {list(known_leagues())}")
        raise typer.Exit(code=1) from exc

    # Resolve the optional curriculum (training-time scenario distribution under F3-A).
    selected_curriculum: ScenarioDistribution | None
    if curriculum:
        try:
            selected_curriculum = build_curriculum(curriculum)
        except ValueError as exc:
            _echo(_status("[FAIL]", str(exc), "red"))
            _echo(f"Known curricula: {list(known_curricula())}")
            raise typer.Exit(code=1) from exc
    else:
        selected_curriculum = None

    _log.info(
        "running league training: league=%s, rounds=%d, train_steps_per_round=%d, "
        "seeds=(league=%d, curriculum=%d, core=%d), curriculum=%s, "
        "distribution=%s, pfsp_alpha=%s, round_end_eval_episodes=%d, pool_dir=%s",
        league,
        rounds,
        train_steps_per_round,
        league_seed,
        curriculum_seed,
        core_seed,
        curriculum or "none",
        distribution,
        pfsp_alpha,
        round_end_eval_episodes,
        pool_dir,
    )

    # Slice C.1 T-C1-CLI — resolve the opponent distribution at the boundary so the
    # CLI passes a concrete instance into the harness (rather than the Slice B
    # factory-only path). The factory is still required by the harness signature; we
    # supply a closure that returns the same instance each round (the harness's
    # T-C1-RUNLOOP loop wraps it with `_MatrixAwareDistribution` per round, so
    # cross-round growth is handled inside the harness without the factory needing
    # to rebuild).
    dist_instance: OpponentDistribution
    if distribution == "uniform":
        dist_instance = UniformOpponentDistribution()
    elif distribution == "pfsp":
        dist_instance = PFSPOpponentDistribution(alpha=pfsp_alpha)
    else:
        # Defensive — the boundary validation above already exited; if a future
        # refactor adds a new value, this branch surfaces the omission immediately
        # instead of silently falling through to one of the above.
        _echo(_status("[FAIL]", f"unknown distribution {distribution!r}", "red"))
        raise typer.Exit(code=1)

    def _distribution_factory(_pool: FrozenPolicyPool) -> OpponentDistribution:
        return dist_instance

    # T-C1-CLI / N3 — capture the pre-run checkpoint count so the final echo can
    # report HOW MUCH the pool grew (the operator-visible value the orchestrator's
    # human-in-the-loop tunes against). Filter to kind=="checkpoint" so the
    # scripted seed archetypes the pool always carries are not counted.
    n_existing_checkpoints = sum(1 for m in pool.members() if m.kind == "checkpoint")

    policy_kind_literal = cast(Literal["mlp", "lstm"], policy_kind)
    result = run_league_training(
        initial_pool=pool,
        n_rounds=rounds,
        distribution_factory=_distribution_factory,
        train_steps_per_round=train_steps_per_round,
        seeds=(league_seed, curriculum_seed, core_seed),
        scenario_distribution=selected_curriculum,
        distribution=dist_instance,
        pfsp_alpha=pfsp_alpha,
        round_end_eval_episodes=round_end_eval_episodes,
        policy_kind=policy_kind_literal,
    )

    # T-C1-CLI / N3 — operator-visible final-pool-size echo (BEFORE the JSON dump so
    # it doesn't clutter the JSON payload — the JSON stays clean for downstream
    # tooling consumers, the human-readable summary goes to stderr-style chatter).
    _echo(
        f"final pool size: {len(result.checkpoints)} checkpoints "
        f"(was {n_existing_checkpoints} before this run)"
    )

    # The new ``win_rate_matrix_snapshot`` field is a WinRateMatrix instance whose
    # ``_entries`` dict has tuple keys; ``json.dumps`` does NOT support tuple keys
    # natively, and ``dataclasses.asdict`` preserves that dict as-is. Strip the
    # snapshot from the printed payload and emit a single line summarizing the
    # entry count + the on-disk path (the source of truth the operator can grep).
    printable_result = replace(result, win_rate_matrix_snapshot=None)
    snapshot = result.win_rate_matrix_snapshot
    if snapshot is not None:
        _echo(
            f"win_rate_matrix_snapshot: {len(snapshot)} entries at "
            f"{result.pool_dir / 'win_rate_matrix.json'}"
        )

    # `default=str` handles non-JSON-native types (Path on pool_dir, PoolMember inside
    # checkpoints) without us reaching into the dataclass. The same convention the
    # 1.5 evaluator + diagnostic commands below use.
    _echo(json.dumps(asdict(printable_result), indent=2, default=str))


@league_app.command("evaluate")
def league_evaluate(
    pool_dir: Path = typer.Option(
        ...,
        "--pool-dir",
        help="On-disk pool directory previously written by `league train`.",
    ),
    n_episodes: int = typer.Option(
        20,
        "--n-episodes",
        help="Position-controlled head-to-head episode budget for the round-N vs round-1 comparison (-A).",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Base seed for the head-to-head episodes (per-episode seed = base + index).",
    ),
) -> None:
    """ """
    from retail_simulator.harness.arena import arena_lock_config
    from retail_simulator.harness.league import (
        FrozenPolicyPool,
        LeagueTrainingResult,
        evaluate_pool_snapshot_improvement,
    )

    # A missing pool is operator error (typo / wrong dir) — surface cleanly and exit 1.
    try:
        pool = FrozenPolicyPool.load(pool_dir)
    except FileNotFoundError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    # m6 (senior review): an arena-trained pool records its seating on the
    # manifest (`pool.home_regions`) -- evaluating it on the evaluator's own
    # default `competition_config()` world would seat these SAME home indices
    # on a config with far fewer regions (a ValueError from RetailParallelEnv's
    # own range check) or, worse, silently on the WRONG geometry entirely.
    # `arena_lock_config()` is the one documented arena config a
    # home_regions-bearing pool can have trained under (harness.arena's module
    # docstring: it is the only env `--env arena` ever trains under) --
    # mirrors `phase4_rl_confirm.py`'s own `_resolve_env` inference.
    home_regions = pool.home_regions
    eval_config = arena_lock_config() if home_regions is not None else None

    # The evaluator consumes ONLY `training_result.checkpoints` for round selection
    # (it picks round-1 + max-round_index from this tuple). The other fields
    # (`learning_curves`, `seeds`, `train_steps_per_round`) are provenance the
    # evaluator does not read; we pass zero-valued placeholders so re-running
    # `evaluate` against a previously-trained pool does NOT require re-running
    # `train` to reconstruct the result object.
    checkpoints = tuple(
        sorted(
            (m for m in pool.members() if m.kind == "checkpoint"),
            key=lambda m: m.round_index,
        )
    )
    training_result = LeagueTrainingResult(
        pool_dir=pool.pool_dir,
        rounds_completed=len(checkpoints),
        checkpoints=checkpoints,
        learning_curves=(),
        seeds=(0, 0, 0),
        train_steps_per_round=0,
    )

    _log.info(
        "evaluating pool snapshot improvement: pool_dir=%s, checkpoints=%d, n_episodes=%d",
        pool_dir,
        len(checkpoints),
        n_episodes,
    )
    try:
        report = evaluate_pool_snapshot_improvement(
            pool,
            training_result,
            n_eval_episodes=n_episodes,
            seed=seed,
            config=eval_config,
            home_regions=home_regions,
        )
    except ValueError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    _echo(json.dumps(asdict(report), indent=2, default=str))


@league_app.command("generalization-diagnostic")
def league_generalization(
    curriculum: str = typer.Option(
        "phase_1_4_basic",
        "--curriculum",
        help="Phase 1.4 curriculum NAME the curriculum-trained PPO is trained against.",
    ),
    train_steps: int = typer.Option(
        2_000_000,
        "--train-steps",
        help="PPO `learn` budget for BOTH the curriculum-PPO and the fixed-PPO (same "
        "hyperparameters, same budget — the symmetric comparison).",
    ),
    n_eval_episodes: int = typer.Option(
        5,
        "--n-eval-episodes",
        help="Eval episodes per held-out scenario; -B mini-eval budget.",
    ),
    curriculum_seed: int = typer.Option(
        0,
        "--curriculum-seed",
        help="Seeds the curriculum wrapper's harness Generator (SEPARATE from core).",
    ),
    core_seed: int = typer.Option(
        42,
        "--core-seed",
        help="Seeds the core PCG64 + the SB3 PPO constructors (training reproducibility).",
    ),
) -> None:
    """ """
    from retail_simulator.harness.curriculum import (
        build_curriculum,
        known_curricula,
    )
    from retail_simulator.harness.league import (
        default_holdout_scenarios,
        run_real_ppo_generalization_diagnostic,
    )

    try:
        distribution = build_curriculum(curriculum)
    except ValueError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        _echo(f"Known curricula: {list(known_curricula())}")
        raise typer.Exit(code=1) from exc

    _log.info(
        "running real-PPO generalization diagnostic: curriculum=%s, train_steps=%d, "
        "n_eval_episodes=%d, seeds=(curriculum=%d, core=%d)",
        curriculum,
        train_steps,
        n_eval_episodes,
        curriculum_seed,
        core_seed,
    )
    report = run_real_ppo_generalization_diagnostic(
        curriculum=distribution,
        holdout_scenarios=default_holdout_scenarios(),
        train_steps=train_steps,
        seeds=(curriculum_seed, core_seed),
        n_eval_episodes_per_scenario=n_eval_episodes,
    )

    _echo(json.dumps(asdict(report), indent=2, default=str))


def _split_participant_label(spec: str) -> tuple[str, str | None]:
    """Split an optional trailing ``:label`` off a participant ``spec``.

    Two grammars (RLB-2's mixed-field ``--participants``):

    * a spec starting with ``archetype:`` or ``random:`` carries its OWN colon
      already, so a further label must be a THIRD segment:
      ``"archetype:lean"`` / ``"random:3"`` (no label) or
      ``"archetype:lean:mylabel"`` / ``"random:3:mylabel"`` (explicit label, the
      LAST segment) — anything else (wrong part count, or an empty segment, e.g.
      the trailing-colon typo ``"archetype:lean:"``) is a ``ValueError``.
    * anything else keeps the pre-RLB-2 rule: split on the LAST ``:`` if present
      (a checkpoint path never contains one on POSIX), rejecting an empty label
      after a trailing ``:``.

    Returns ``(raw, explicit_label)`` — ``raw`` is what
    :func:`~retail_simulator.harness.ladder.parse_participant_spec` should parse;
    ``explicit_label`` is ``None`` when the caller didn't supply one (the spec's
    own ``default_label`` applies).
    """
    from retail_simulator.harness.ladder import (  # noqa: PLC0415
        ARCHETYPE_SPEC_PREFIX,
        RANDOM_SPEC_PREFIX,
    )

    if spec.startswith(ARCHETYPE_SPEC_PREFIX) or spec.startswith(RANDOM_SPEC_PREFIX):
        parts = spec.split(":")
        if len(parts) == 2:
            return spec, None
        if len(parts) == 3 and all(parts):
            return f"{parts[0]}:{parts[1]}", parts[2]
        raise ValueError(
            f"participant {spec!r} must be '<prefix>:<value>' or "
            "'<prefix>:<value>:<label>' (no empty segments)"
        )
    if ":" in spec:
        raw, _, label = spec.rpartition(":")
        if not label:
            raise ValueError(f"participant {spec!r} has an empty label after ':'")
        return raw, label
    return spec, None


def _format_ladder_table(result: "LadderResult") -> str:
    """Render a :class:`LadderResult` as an ASCII ranking table (presentation-only).

    Same computed object as the JSON path — an alternate view for an operator
    scanning a terminal. Three blocks:

    1. The ranking table: rank, label, kind, field_margin, [ci_low, ci_high], plus
       (RLB-3, battle scoreboard) the field's revenue/market-share/profit
       component margins (``rev_fm``/``share_fm``/``profit_fm`` — ``"n/a"`` when
       ``entry.component_field_margins`` is ``None``; share to 4 decimals, money
       to 1).
    2. "battle pairs (A over B)": every :class:`PairwiseMargin` — label_a,
       label_b, the score mean + ``[ci_low, ci_high]``, and the same
       revenue/share/profit breakdown (``"n/a"`` when ``component_mean_margins``
       is ``None``).
    3. A footer: the REPORTED split-half Kendall-τ line, then (RLB-3) the
       tournament summary — Condorcet winner (or ``none``), the cyclic-triple
       count over the total, and whether the (first-only, per
       :attr:`~retail_simulator.harness.ladder.TournamentSummary.decisive_cycle`'s
       own docstring) realized cycle is decisive, then (RLB-7) the COMPLETE
       decisive-triple count and its checkpoint-inclusive subset count, plus one
       ``A > B > C > A`` line per decisive triple (a trailing ``*`` marks a triple
       containing >= 1 ``"checkpoint"``-kind participant) — the piece that makes
       the runbook's C3 criterion decidable straight off this table, not just the
       first-triple-only ``decisive_cycle`` flag.

    (m8) ``loyalty`` is part of :data:`~retail_simulator.harness.league.
    HEAD_TO_HEAD_COMPONENTS` and is serialized in every ``component_*`` dict, but
    is deliberately NOT rendered here — the operator-facing table surfaces only
    revenue/share/profit, the field's three commercial headline numbers.

    (RLB-8a) A ``homes=[...]`` line is printed right after ``env=`` ONLY when
    ``result.home_regions`` is not ``None`` — a pre-RLB-8a / non-seated ladder's
    table is completely unaffected (no blank/placeholder line appears).

    The output is plain ASCII — no color codes are emitted — so it is safe to
    pipe or redirect regardless of the ``NO_COLOR`` environment.
    """
    lines = [
        f"ladder_id={result.ladder_id}",
        f"env={result.config_label}",
    ]
    if result.home_regions is not None:
        lines.append(f"homes={list(result.home_regions)}")
    lines.append(
        f"seeds: n={result.n_seeds} set={list(result.seed_set)} n_episodes={result.n_episodes}"
    )
    lines.append("")
    lines.append(
        f"{'rank':>4} {'label':<24} {'kind':<10} {'field_margin':>14} "
        f"{'ci_low':>14} {'ci_high':>14} {'rev_fm':>12} {'share_fm':>10} {'profit_fm':>12}"
    )
    for entry in result.entries:
        fm = entry.component_field_margins
        rev_fm = "n/a" if fm is None else f"{fm['revenue']:.1f}"
        share_fm = "n/a" if fm is None else f"{fm['market_share']:.4f}"
        profit_fm = "n/a" if fm is None else f"{fm['profit']:.1f}"
        lines.append(
            f"{entry.rank:>4} {entry.label:<24} {entry.kind:<10} {entry.field_margin:>14.6f} "
            f"{entry.ci_low:>14.6f} {entry.ci_high:>14.6f} {rev_fm:>12} {share_fm:>10} "
            f"{profit_fm:>12}"
        )

    lines.append("")
    lines.append("battle pairs (A over B):")
    lines.append(
        f"{'label_a':<24} {'label_b':<24} {'score_mean':>14} {'score_ci':>34} "
        f"{'revenue':>12} {'share':>10} {'profit':>12}"
    )
    for pair in result.pairwise:
        score_ci = (
            "n/a"
            if pair.ci_low is None or pair.ci_high is None
            else f"[{pair.ci_low:.6f}, {pair.ci_high:.6f}]"
        )
        cm = pair.component_mean_margins
        revenue = "n/a" if cm is None else f"{cm['revenue']:.1f}"
        share = "n/a" if cm is None else f"{cm['market_share']:.4f}"
        profit = "n/a" if cm is None else f"{cm['profit']:.1f}"
        lines.append(
            f"{pair.label_a:<24} {pair.label_b:<24} {pair.mean_margin:>14.6f} {score_ci:>34} "
            f"{revenue:>12} {share:>10} {profit:>12}"
        )

    lines.append("")
    tau = result.kendall_tau_split_half
    tau_str = "n/a (n_seeds < 4)" if tau is None else f"{tau:.6f}"
    lines.append(f"kendall_tau_split_half (REPORTED): {tau_str}")

    tournament = result.tournament
    if tournament is None:
        lines.append("tournament: n/a")
    else:
        condorcet_winner = (
            "none" if tournament.condorcet_winner is None else tournament.condorcet_winner
        )
        lines.append(
            f"tournament: condorcet_winner={condorcet_winner} "
            f"cyclic_triples={tournament.n_cyclic_triples}/{tournament.total_triples} "
            f"decisive_cycle={tournament.decisive_cycle} "
            f"decisive_triples={len(tournament.decisive_triples)} "
            f"(checkpoint-inclusive: {tournament.checkpoint_inclusive_decisive_triples})"
        )
        if tournament.decisive_triples:
            kind_by_label = {entry.label: entry.kind for entry in result.entries}
            for x, y, z in tournament.decisive_triples:
                is_checkpoint_inclusive = any(
                    kind_by_label[label] == "checkpoint" for label in (x, y, z)
                )
                marker = " *" if is_checkpoint_inclusive else ""
                lines.append(f"  {x} > {y} > {z} > {x}{marker}")
    return "\n".join(lines)


def _ladder_env_default() -> tuple[CoreConfig | None, str]:
    """The ``default`` ``--env`` value: no config override (``run_ladder`` falls
    back to its own ``competition_config()`` default — the CLI never imports that
    builder itself)."""
    from retail_simulator.harness.ladder import DEFAULT_ENV_LABEL  # noqa: PLC0415

    return None, DEFAULT_ENV_LABEL


def _ladder_env_calibrated() -> tuple[CoreConfig | None, str]:
    """The ``calibrated`` ``--env`` value: the Phase-4 ``calibrated_spike_config``."""
    from retail_simulator.harness.ladder import CALIBRATED_ENV_LABEL  # noqa: PLC0415
    from retail_simulator.harness.spike import calibrated_spike_config  # noqa: PLC0415

    return calibrated_spike_config(), CALIBRATED_ENV_LABEL


def _ladder_env_arena() -> tuple[CoreConfig | None, str]:
    """The ``arena`` ``--env`` value (RLB-8a): the 25-region candidate-lock arena.

    Only the ``(config, config_label)`` pair — this dict's own established
    shape. The matching ``home_regions=arena_home_regions(2)`` seating is
    resolved separately in :func:`league_ladder` (a league game is two seats:
    learner + opponent), NOT folded into this builder's return shape, so every
    OTHER ``--env`` value stays a plain 2-tuple.
    """
    from retail_simulator.harness.arena import ARENA_ENV_LABEL, arena_lock_config  # noqa: PLC0415

    return arena_lock_config(), ARENA_ENV_LABEL


def _ladder_env_arena_balanced() -> tuple[CoreConfig | None, str]:
    """The ``arena-balanced`` ``--env`` value (credit-line-and-rebalance): the
    re-balanced 25-region arena — the candidate lock plus a working-capital
    credit line and a smaller warehouse footprint (``harness/arena.py::
    arena_balanced_config``). Same seating shape as ``arena`` (see
    :func:`_ladder_env_arena`'s own docstring for why ``home_regions`` is
    resolved separately in :func:`league_ladder`, not folded in here) — both
    envs share the SAME 25-region table, only the economics knobs differ.
    """
    from retail_simulator.harness.arena import (  # noqa: PLC0415
        ARENA_BALANCED_ENV_LABEL,
        arena_balanced_config,
    )

    return arena_balanced_config(), ARENA_BALANCED_ENV_LABEL


# `--env`'s SINGLE SOURCE OF TRUTH: name -> zero-arg ``(config, config_label)``
# builder. Referencing these functions here costs nothing at CLI module-import
# time (each builder's own heavy/lazy imports run only when INVOKED) — so both
# `league_ladder`'s `--env` help text and `_resolve_ladder_env`'s [FAIL] allowlist
# read this dict's keys instead of hand-maintaining a separate name list that could
# drift from the actual dispatch. "arena" (RLB-8a) is the first env wired in per
# this comment's own standing instruction: add its builder function + an entry here.
_LADDER_ENV_BUILDERS: dict[str, Callable[[], tuple[CoreConfig | None, str]]] = {
    "default": _ladder_env_default,
    "calibrated": _ladder_env_calibrated,
    "arena": _ladder_env_arena,
    "arena-balanced": _ladder_env_arena_balanced,
}


def _resolve_ladder_env(env: str) -> tuple[CoreConfig | None, str]:
    """Resolve ``--env``'s short name to a ``(config, config_label)`` pair.

    Dispatches through :data:`_LADDER_ENV_BUILDERS`. Raises ``ValueError`` on an
    unknown ``env`` name (the command's own [FAIL] + exit-1 boundary-validation
    pattern turns this into a clean CLI failure), listing the SAME allowlist the
    builders dict defines — never a hand-maintained, driftable copy of it.
    """
    try:
        builder = _LADDER_ENV_BUILDERS[env]
    except KeyError:
        allowed = ", ".join(_LADDER_ENV_BUILDERS)
        raise ValueError(f"unknown --env {env!r}; must be one of: {allowed}") from None
    return builder()


@league_app.command("ladder")
def league_ladder(
    participants: list[str] = typer.Option(
        ...,
        "--participants",
        help="Repeatable participant spec: a checkpoint zip `path` (or `path:label`), "
        "`archetype:<name>` for a calibrated or geo archetype "
        f"({'/'.join(LADDER_ARCHETYPE_LABELS)}), or "
        "`random:<seed>` for a random baseline — the latter two optionally suffixed "
        "`:label` (e.g. `random:3:baseline`; a checkpoint path keeps its own `path:label` "
        "form). When no label is given one is derived (the zip stem, the archetype name, "
        "or `random_<seed>`). A geo archetype's expansion targeting only makes sense on a "
        "multi-region world (`--env arena`) — on the smaller default/calibrated env it "
        "still runs, just as degenerately as a calibrated archetype does on the arena. "
        "Require >=2; a checkpoint path must exist and end in `.zip`; labels must be "
        "unique.",
    ),
    n_seeds: int = typer.Option(
        30,
        "--n-seeds",
        min=1,
        help="Fixed shared seed-set size; every unordered pair is played position-controlled over seeds 0..n_seeds-1 (the fairness contract).",
    ),
    n_episodes: int = typer.Option(
        1,
        "--n-episodes",
        min=1,
        help="Position-controlled head-to-head episodes per (pair, seed).",
    ),
    bootstrap_seed: int = typer.Option(
        0,
        "--bootstrap-seed",
        help="Seeds the bootstrap-CI seed-resample RNG (kept SEPARATE from the rollout seeds; "
        "fixing it makes the CIs — and the whole ladder.json — bit-reproducible).",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Path for the ladder.json artifact (default runs/<ladder_id>/ladder.json).",
    ),
    table: bool = typer.Option(
        False,
        "--table",
        help="Print an ASCII ranking table instead of the JSON payload (the artifact is "
        "still written either way).",
    ),
    env: str = typer.Option(
        "default",
        "--env",
        help="Which CoreConfig the ladder rolls out on — one of: "
        + ", ".join(_LADDER_ENV_BUILDERS)
        + " ('default' = competition_config, the pre-RLB-2 behavior; 'calibrated' = "
        "the Phase-4 calibrated_spike_config; 'arena' = the 25-region candidate-lock "
        "arena, RLB-8a; 'arena-balanced' = the same 25-region arena with a "
        "working-capital credit line + a smaller warehouse footprint (credit-line-"
        "and-rebalance) — both arena envs seat both participants at "
        "arena_home_regions(2)). Recorded in the artifact as config_label; folds "
        "into the default ladder_id only when non-default, so every pre-RLB-2 "
        "ladder_id stays byte-identical.",
    ),
) -> None:
    """ """
    from retail_simulator.harness.arena import arena_home_regions
    from retail_simulator.harness.ladder import parse_participant_spec, run_ladder

    # -- Boundary validation: --env ------------------------------------------------------
    try:
        config, config_label = _resolve_ladder_env(env)
    except ValueError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    # RLB-8a: the arena's seating is resolved HERE, not inside _LADDER_ENV_BUILDERS
    # (whose shape every other env value shares — see _ladder_env_arena's own
    # docstring) — a league game is two seats (learner + opponent), matching the
    # seat count run_ladder's real default path always builds (n_learning_agents=2).
    # credit-line-and-rebalance: "arena-balanced" shares the exact same 25-region
    # table (only the economics knobs differ), so it seats identically.
    home_regions = arena_home_regions(2) if env in ("arena", "arena-balanced") else None

    # -- Boundary validation: --participants ---------------------------------------------
    # Operator typos (too few participants, a malformed spec, a missing/non-zip
    # checkpoint path, duplicate labels) are surfaced as clean CLI failures BEFORE
    # any rollout, mirroring the other `league` commands' [FAIL] + exit-1 pattern.
    if len(participants) < 2:
        _echo(
            _status(
                "[FAIL]",
                f"--participants requires at least 2 entries; got {len(participants)}",
                "red",
            )
        )
        raise typer.Exit(code=1)

    parsed: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for spec in participants:
        try:
            raw, explicit_label = _split_participant_label(spec)
            participant_spec = parse_participant_spec(raw)
        except ValueError as exc:
            _echo(_status("[FAIL]", str(exc), "red"))
            raise typer.Exit(code=1) from exc
        label = explicit_label if explicit_label is not None else participant_spec.default_label
        # The `.zip` suffix + existence checks are a CHECKPOINT-only concern — an
        # archetype/random spec has no filesystem footprint to validate. Normalizing
        # through `Path` (pre-RLB-2 behavior, preserved here) is a checkpoint-only
        # concern too — an archetype/random spec's source string is passed through
        # VERBATIM (`Path("archetype:lean")` would just be noise).
        source = participant_spec.source
        if participant_spec.kind == "checkpoint":
            path = Path(source)
            if path.suffix != ".zip":
                _echo(_status("[FAIL]", f"participant path {path} must end in '.zip'", "red"))
                raise typer.Exit(code=1)
            if not path.is_file():
                _echo(_status("[FAIL]", f"participant checkpoint not found: {path}", "red"))
                raise typer.Exit(code=1)
            source = str(path)
        if label in seen_labels:
            _echo(_status("[FAIL]", f"duplicate participant label {label!r}", "red"))
            raise typer.Exit(code=1)
        seen_labels.add(label)
        parsed.append((source, label))

    _log.info(
        "running ladder: participants=%d, n_seeds=%d, n_episodes=%d, bootstrap_seed=%d, env=%s",
        len(parsed),
        n_seeds,
        n_episodes,
        bootstrap_seed,
        config_label,
    )
    result = run_ladder(
        parsed,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        bootstrap_seed=bootstrap_seed,
        config=config,
        config_label=config_label,
        home_regions=home_regions,
    )

    # Default artifact path derives from the resolved ladder_id (mirrors the other
    # commands' runs/<id>/ layout); honor the global --run-dir root.
    out_path = (
        out if out is not None else Path(_state["run_dir"]) / result.ladder_id / "ladder.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        out_path,
        json.dumps(asdict(result), indent=2, sort_keys=True),
    )
    _echo(f"wrote ladder artifact: {out_path}")

    if table:
        _echo(_format_ladder_table(result))
    else:
        _echo(json.dumps(asdict(result), indent=2, sort_keys=True))


def _resolve_token(no_auth: bool) -> str | None:
    """Resolve the F2-A shared-secret token from the operator's flags/env.

    The ONE owner of the ``--no-auth`` / ``RETAIL_SIM_TOKEN`` posture, shared by
    every live-game-touching command (``live serve``, ``live run-series``, and
    ``live play``): ``--no-auth`` skips the env-var check (local-only
    convenience and returns ``None``); otherwise ``RETAIL_SIM_TOKEN`` MUST be
    set or the command exits 1 with a clean operator diagnostic (never a stack
    trace — boundary validation). Both the server-side config builder and the
    client command call this so the resolution logic is not duplicated.
    """
    if no_auth:
        return None
    token = os.environ.get("RETAIL_SIM_TOKEN")
    if token is None:
        _echo(
            _status(
                "[FAIL]",
                "RETAIL_SIM_TOKEN env var must be set (or use --no-auth for local testing)",
                "red",
            )
        )
        raise typer.Exit(code=1)
    return token


def _build_serve_config(
    *,
    n_agents: int,
    tick_interval: float,
    n_ticks: int,
    host: str,
    port: int,
    registration_timeout: float,
    no_auth: bool,
    npc_archetypes: str,
    scenario_name: str,
    runs_dir: Path,
    seed: int,
) -> "ServeConfig":
    """Resolve the live-game operator flags into a :class:`ServeConfig`.

    The ONE owner of the live-game boundary rules shared by ``live serve`` and
    ``live run-series`` (DRY): comma-separated ``--npc-archetypes`` parsing +
    unknown-name rejection (fail fast at the boundary, exit 1 with the known
    list) and ``--no-auth`` / ``RETAIL_SIM_TOKEN`` resolution (the F2-A
    shared-secret posture). Raises ``typer.Exit(code=1)`` on operator error —
    never a stack trace — so both commands surface identical diagnostics.
    """
    from retail_simulator.core.config import NPC_ARCHETYPES
    from retail_simulator.live import ServeConfig

    # Parse comma-separated archetypes; empty string -> empty tuple (the F3-A
    # all-remote mode the server's seat plan handles natively).
    parsed_archetypes = tuple(name.strip() for name in npc_archetypes.split(",") if name.strip())
    unknown = [a for a in parsed_archetypes if a not in NPC_ARCHETYPES]
    if unknown:
        _echo(_status("[FAIL]", f"unknown NPC archetype(s): {unknown}", "red"))
        _echo(f"Known archetypes: {list(NPC_ARCHETYPES)}")
        raise typer.Exit(code=1)

    # Token resolution: the F2-A shared-secret posture lives in the shared
    # `_resolve_token` helper (ONE owner — `live play` reuses it verbatim).
    token = _resolve_token(no_auth)

    return ServeConfig(
        n_agents=n_agents,
        n_ticks=n_ticks,
        tick_interval_seconds=tick_interval,
        host=host,
        port=port,
        registration_timeout_seconds=registration_timeout,
        token=token,
        npc_archetypes=parsed_archetypes,
        scenario_name=scenario_name,
        runs_dir=runs_dir,
        seed=seed,
    )


@live_app.command("serve")
def live_serve(
    n_agents: int = typer.Option(2, "--n-agents", help="Number of learning seats."),
    tick_interval: float = typer.Option(
        5.0,
        "--tick-interval",
        help="Seconds per tick; 0 = CI fast mode (F5=A — no wall-clock sleep).",
    ),
    n_ticks: int = typer.Option(104, "--n-ticks", help="Game length (per)."),
    host: str = typer.Option("localhost", "--host", help="WebSocket bind host."),
    port: int = typer.Option(8765, "--port", help="WebSocket bind port."),
    registration_timeout: float = typer.Option(
        60.0,
        "--registration-timeout",
        help="F4=B abort after this many seconds with insufficient clients.",
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Disable shared-token auth. Token via RETAIL_SIM_TOKEN env var when set; F2=A.",
    ),
    npc_archetypes: str = typer.Option(
        "",
        "--npc-archetypes",
        help="F3=B comma-separated NPC archetypes (e.g. 'discounter,premium'); empty = all-remote.",
    ),
    scenario_name: str = typer.Option(
        "default", "--scenario", help="Scenario name for SeatAssigned + manifest."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Where to write runs/<game-id>/."
    ),
    seed: int = typer.Option(0, "--seed", help="Core PCG64 seed."),
) -> None:
    """Run a single live game over WebSocket — Phase 2.0 demo server.

    Resolves the operator's flags into a :class:`ServeConfig`, awaits
    :func:`serve_game` once, prints the resulting :class:`ServeResult` as
    JSON on stdout, and exits with ``result.exit_status`` (0 on a clean
    game, 1 on registration timeout, 2 on server error).

    Unknown ``--npc-archetypes`` names fail fast at the boundary with the
    known list and exit 1 (mirrors ``competition run``'s ``--curriculum``
    handling). When ``--no-auth`` is NOT set, the shared-secret token is
    read from the ``RETAIL_SIM_TOKEN`` env var; an unset env var with
    ``--no-auth`` absent exits 1 with a clean operator diagnostic (never
    a stack trace — boundary validation).

    Example: ``retail-sim live serve --n-agents 2 --tick-interval 0 --no-auth``
    """
    import asyncio

    from retail_simulator.live import serve_game

    config = _build_serve_config(
        n_agents=n_agents,
        tick_interval=tick_interval,
        n_ticks=n_ticks,
        host=host,
        port=port,
        registration_timeout=registration_timeout,
        no_auth=no_auth,
        npc_archetypes=npc_archetypes,
        scenario_name=scenario_name,
        runs_dir=runs_dir,
        seed=seed,
    )

    _log.info(
        "running live serve: n_agents=%d, n_ticks=%d, tick_interval=%.2fs, "
        "host=%s, port=%d, npc_archetypes=%s, scenario=%s, runs_dir=%s, seed=%d",
        n_agents,
        n_ticks,
        tick_interval,
        host,
        port,
        list(config.npc_archetypes),
        scenario_name,
        runs_dir,
        seed,
    )

    result = asyncio.run(serve_game(config))

    # JSON payload mirrors the league commands — operator-greppable + machine-readable.
    # ``default=str`` handles the Path on ``run_dir`` without reaching into the dataclass.
    _echo(json.dumps(asdict(result), indent=2, default=str))

    # Honor the server's exit status: 0 clean game, 1 registration timeout, 2 error.
    if result.exit_status != 0:
        raise typer.Exit(code=result.exit_status)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a sibling ``.tmp`` + ``os.replace``.

    Mirrors the live server's own ``_atomic_write_text`` idiom (a crash
    mid-write leaves either the prior or the new contents, never a partial
    file) without reaching into ``server.py``'s private helper. The temp file
    is in the same directory so ``os.replace`` is a same-filesystem rename
    (atomic on POSIX).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _format_series_table(series: "SeriesResult") -> str:
    """Render a :class:`SeriesResult` as an ASCII table (presentation-only).

    Same computed object as the JSON path — this is just an alternate view
    for an operator scanning a terminal. Color is suppressed via the same
    ``_use_color()`` gate the rest of the CLI honors (``NO_COLOR`` etc.).
    """
    lines = [
        f"series_id={series.series_id}",
        (
            f"games: requested={series.games_requested} "
            f"played={series.games_played} failed={series.games_failed}"
        ),
        "",
    ]

    seat_header = f"{'seat':>12} {'mean':>14} {'stddev':>14}"
    lines.append(seat_header)
    lines.append("-" * len(seat_header))
    for seat in sorted(series.per_seat_mean):
        mean = series.per_seat_mean[seat]
        stddev = series.per_seat_stddev[seat]
        mean_str = "n/a" if mean is None else f"{mean:.4f}"
        stddev_str = "n/a" if stddev is None else f"{stddev:.4f}"
        lines.append(f"{seat:>12} {mean_str:>14} {stddev_str:>14}")

    lines.append("")
    game_header = f"{'idx':>4} {'game_id':>18} {'seed':>8} {'exit':>6}"
    lines.append(game_header)
    lines.append("-" * len(game_header))
    for game in series.games:
        lines.append(
            f"{game.game_index:>4} {game.game_id:>18} {game.seed:>8} {game.exit_status:>6}"
        )
    return "\n".join(lines)


async def _wait_for_listener(host: str, port: int, *, timeout_seconds: float = 10.0) -> None:
    """Poll ``host:port`` until a TCP connect succeeds or the timeout elapses.

    A bind-probe readiness check (mirrors the live test suite's
    ``wait_for_server_ready``): far more robust than a fixed sleep when this runs
    hands-free on an arbitrary/loaded box, where a fixed delay could let a client
    race the listener and see ``ConnectionRefusedError``. Raises ``TimeoutError``
    if the listener never comes up — surfaced by the caller as a failed game.
    """
    import asyncio
    import contextlib

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        try:
            _reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            await asyncio.sleep(0.02)
            continue
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return
    raise TimeoutError(f"self-play: server at {host}:{port} did not bind within {timeout_seconds}s")


async def _serve_with_clients(config: "ServeConfig", policy: str) -> "ServeResult":
    """Run one game's server AND its ``n_agents`` internal clients concurrently.

    The self-play primitive for ``run-series --self-play`` (Slice 4): a single
    asyncio event loop hosts :func:`serve_game` plus ``config.n_agents`` internal
    :func:`play_one_game` clients (one per learning seat), so a game runs
    hands-free with NO external clients. NPC seats are filled by the server from
    ``config.npc_archetypes`` as configured; only the learning seats need an
    internal client. Returns the SERVER's :class:`ServeResult` (the authoritative
    per-seat scores + exit status that :func:`run_series` aggregates) — the
    clients' own returned scores are discarded.

    Server-readiness: the server task is created first, then a bounded TCP
    bind-probe (:func:`_wait_for_listener`) blocks until ``websockets.serve`` is
    actually accepting before any client connects — robust on slow/loaded boxes
    (a fixed sleep would risk a client racing the listener). The clients then
    drive every tick to FinalScores and exit; awaiting the server task yields its
    ServeResult. Internal-client exceptions are gathered (never masking the
    authoritative server result) and logged at WARNING so a failed-to-connect
    internal client is diagnosable rather than surfacing only as an unexplained
    registration timeout.
    """
    import asyncio
    import contextlib

    from retail_simulator.live import play_one_game, serve_game

    uri = f"ws://{config.host}:{config.port}"
    # The internal clients use whatever token the server expects — the same
    # `config.token` (`None` under --no-auth) the shared `_resolve_token` posture
    # already resolved into the base ServeConfig.
    token = config.token

    server_task = asyncio.create_task(serve_game(config))
    client_tasks: list[asyncio.Task[Any]] = []
    try:
        # Block until the listener is actually bound (not a fixed sleep) before
        # connecting clients. If the server died during startup, await it below
        # to surface its ServeResult rather than a bare TimeoutError.
        with contextlib.suppress(TimeoutError):
            await _wait_for_listener(config.host, config.port)

        client_tasks = [
            asyncio.create_task(play_one_game(uri, f"selfplay-{i}", token, policy=policy))
            for i in range(config.n_agents)
        ]

        # Await the server last — it returns the authoritative ServeResult once
        # the game ends.
        result: ServeResult = await server_task
    finally:
        # Always drain the clients (even if the server task raised) so no client
        # task is orphaned across the sequential games of a series. return_exceptions
        # keeps a client teardown error from masking the server result.
        outcomes = await asyncio.gather(*client_tasks, return_exceptions=True)
        for i, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                _log.warning("self-play internal client selfplay-%d failed: %s", i, outcome)
    return result


def _make_self_play_serve_fn(policy: str) -> "Callable[[ServeConfig], ServeResult]":
    """Build the ``serve_fn`` that :func:`run_series` injects under ``--self-play``.

    Each per-game call runs :func:`_serve_with_clients` under its own
    ``asyncio.run`` (one fresh event loop per game — games stay SEQUENTIAL, the
    series runs one at a time; the only added concurrency is the per-game
    server+clients gather). Mirrors the default ``serve_fn``'s
    ``asyncio.run(serve_game(cfg))`` shape, differing only in that the clients
    run in-process alongside the server.
    """
    import asyncio

    def _serve_fn(cfg: "ServeConfig") -> "ServeResult":
        return asyncio.run(_serve_with_clients(cfg, policy))

    return _serve_fn


@live_app.command("run-series")
def live_run_series(
    games: int = typer.Option(..., "--games", help="Number of games in the series."),
    n_agents: int = typer.Option(2, "--n-agents", help="Number of learning seats."),
    tick_interval: float = typer.Option(
        5.0,
        "--tick-interval",
        help="Seconds per tick; 0 = CI fast mode (F5=A — no wall-clock sleep).",
    ),
    n_ticks: int = typer.Option(104, "--n-ticks", help="Game length (per)."),
    host: str = typer.Option("localhost", "--host", help="WebSocket bind host."),
    port: int = typer.Option(8765, "--port", help="WebSocket bind port."),
    registration_timeout: float = typer.Option(
        60.0,
        "--registration-timeout",
        help="F4=B abort after this many seconds with insufficient clients.",
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Disable shared-token auth. Token via RETAIL_SIM_TOKEN env var when set; F2=A.",
    ),
    npc_archetypes: str = typer.Option(
        "",
        "--npc-archetypes",
        help="F3=B comma-separated NPC archetypes (e.g. 'discounter,premium'); empty = all-remote.",
    ),
    scenario_name: str = typer.Option(
        "default", "--scenario", help="Scenario name for SeatAssigned + manifest."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Where to write runs/<game-id>/."
    ),
    seed: int = typer.Option(0, "--seed", help="Core PCG64 seed (game i uses seed+i)."),
    table: bool = typer.Option(
        False, "--table", help="Render an ASCII summary table instead of JSON."
    ),
    self_play: bool = typer.Option(
        False,
        "--self-play",
        help="Run each game's server AND its n-agents internal default-policy clients "
        "concurrently in-process — a hands-free series with NO external clients. "
        "Off (default) waits for external clients exactly as before.",
    ),
    policy: str = typer.Option(
        "default",
        "--policy",
        help="Per-tick action policy for the internal --self-play clients. "
        "'default' submits the registry-default action every tick.",
    ),
) -> None:
    """Run a SERIES of ``--games`` live games and aggregate per-seat scores.

    Reuses every ``live serve`` flag to build the base :class:`ServeConfig`
    (game ``i`` runs with ``seed + i``; the port is held fixed — games are
    sequential, not concurrent), then drives :func:`run_series`. Writes a
    sidecar ``runs/<series-id>/series_summary.json`` (the flat per-game
    ``runs/<game-id>/`` dirs are UNCHANGED) and prints the
    :class:`SeriesResult` as JSON on stdout (``--table`` for the ASCII view).

    Exit code: 0 if the series ran to completion — EVEN with some failed
    games (a failed game is recorded and the series continues); 1 ONLY if
    ZERO games finished cleanly (``games_played == 0``), with a clear message.

    Example: ``retail-sim live run-series --games 3 --n-agents 2 --no-auth``
    """
    from retail_simulator.live import run_series

    # Boundary validation: this guard is the single owner of the ``--games``
    # domain rule, emitting the spec'd exit-1 + a clear operator message
    # (preferred over Typer's bare ``min=`` usage error, which exits 2 with no
    # domain context).
    if games < 1:
        _echo(_status("[FAIL]", f"--games must be >= 1 (got {games})", "red"))
        raise typer.Exit(code=1)

    # The internal-client policy is only consulted under --self-play; validate
    # it up front either way so the diagnostic is the same shape as `live play`
    # and a bad value never reaches the serve_fn.
    if self_play and policy != "default":
        _echo(
            _status(
                "[FAIL]",
                f"unknown --policy {policy!r}; only 'default' is supported in this slice",
                "red",
            )
        )
        raise typer.Exit(code=1)

    base_config = _build_serve_config(
        n_agents=n_agents,
        tick_interval=tick_interval,
        n_ticks=n_ticks,
        host=host,
        port=port,
        registration_timeout=registration_timeout,
        no_auth=no_auth,
        npc_archetypes=npc_archetypes,
        scenario_name=scenario_name,
        runs_dir=runs_dir,
        seed=seed,
    )

    _log.info(
        "running live run-series: games=%d, n_agents=%d, n_ticks=%d, tick_interval=%.2fs, "
        "host=%s, port=%d, npc_archetypes=%s, scenario=%s, runs_dir=%s, base_seed=%d",
        games,
        n_agents,
        n_ticks,
        tick_interval,
        host,
        port,
        list(base_config.npc_archetypes),
        scenario_name,
        runs_dir,
        seed,
    )

    # Without --self-play the series uses run_series's default serve_fn (waits
    # for external clients — behavior UNCHANGED). With --self-play, each game is
    # served alongside its own in-process clients via the injected serve_fn;
    # run_series's aggregation core is identical in both paths (self-play is
    # PURELY a serve_fn swap — no change to the reducer).
    if self_play:
        series = run_series(base_config, games=games, serve_fn=_make_self_play_serve_fn(policy))
    else:
        series = run_series(base_config, games=games)

    # Sidecar lives under the SAME runs_dir as the per-game dirs, namespaced by
    # series id so a series never collides with a flat per-game dir.
    series_dir = runs_dir / series.series_id
    series_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "series_id": series.series_id,
        "games_requested": series.games_requested,
        "games_played": series.games_played,
        "games_failed": series.games_failed,
        "per_seat_mean": series.per_seat_mean,
        "per_seat_stddev": series.per_seat_stddev,
        "games": [
            {
                "game_index": g.game_index,
                "game_id": g.game_id,
                "seed": g.seed,
                "exit_status": g.exit_status,
                "final_scores": g.final_scores,
            }
            for g in series.games
        ],
    }
    _atomic_write_text(series_dir / "series_summary.json", json.dumps(summary, indent=2))

    if table:
        _echo(_format_series_table(series))
    else:
        # ``default=str`` handles the Path on each game's ``run_dir`` (none here,
        # but matches `live serve`'s echo for consistency).
        _echo(json.dumps(asdict(series), indent=2, default=str))

    # The series is a SUCCESS as long as at least one game finished cleanly —
    # individual failed games are recorded and listed, never fatal. A zero-clean
    # series is the only exit-1 case (nothing aggregable; likely misconfiguration).
    if series.games_played == 0:
        _echo(
            _status(
                "[FAIL]",
                f"series {series.series_id}: 0/{series.games_requested} games completed cleanly",
                "red",
            )
        )
        raise typer.Exit(code=1)


@live_app.command("replay")
def live_replay(
    game_dir: Path = typer.Option(
        ...,
        "--game-dir",
        help="Per-game directory (runs/<game-id>/) containing game_manifest.json "
        "+ action_log.jsonl.",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="D-2 strict mode: truncated logs count as divergence. --no-strict accepts "
        "truncation for crashed-server replays.",
    ),
) -> None:
    """Replay a recorded game offline; verify AC-4 bit-identity — exit 0 / 1.

    Calls :func:`replay_game_dir` on the per-game directory and prints the
    resulting :class:`ReplayResult` as JSON. Exits 0 on a clean (bit-identical)
    replay; 1 on a divergence or (under ``--strict``) a truncated action log.

    A missing ``--game-dir`` (or missing manifest / action_log under it) surfaces
    as a clean operator diagnostic via :class:`FileNotFoundError` from
    :func:`replay_game_dir` — exit 1, no stack trace.

    Example: ``retail-sim live replay --game-dir runs/game-abc123``
    """
    from retail_simulator.live import replay_game_dir

    try:
        result = replay_game_dir(game_dir, strict=strict)
    except FileNotFoundError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    _echo(json.dumps(asdict(result), indent=2, default=str))

    if not result.ok:
        raise typer.Exit(code=1)


@live_app.command("play")
def live_play(
    host: str = typer.Option("localhost", "--host", help="WebSocket server host."),
    port: int = typer.Option(8765, "--port", help="WebSocket server port."),
    agent_name: str = typer.Option(
        "player", "--agent-name", help="Client name echoed to the server + logs."
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Connect without a token. Otherwise the token is read from RETAIL_SIM_TOKEN; F2=A.",
    ),
    policy: str = typer.Option(
        "default",
        "--policy",
        help="Per-tick action policy. 'default' submits the registry-default action every tick.",
    ),
    seed: int = typer.Option(
        0, "--seed", help="Policy RNG seed (reserved for future stochastic policies)."
    ),
) -> None:
    """Connect ONE client to a running live game, play it, print final scores.

    Constructs a :class:`LiveClient`, connects (printing the human-readable seat
    assignment), then submits one action per observed tick until the server
    sends FinalScores, which are printed as JSON on stdout. The client is the
    PRIMITIVE — to fill a 2-seat game an operator launches two ``play`` clients
    (or one + an NPC seat); this command does NOT orchestrate multiple clients.

    Exit 0 when the game ends cleanly (FinalScores received). Exit 1 on a
    :class:`ConnectionError` (server unreachable, connection rejected, or game
    aborted) with a clear ``[FAIL]`` message — never a stack trace.

    The ``default`` policy submits a writable copy of the read-only
    ``REGISTRY_DEFAULT_ACTION`` each tick (the passive baseline; the env outcome
    is still seed-dependent, so across a seed-varying ``run-series`` it yields
    cross-seed variance data). ``--policy random`` is intentionally NOT shipped
    in this slice — it is deferred until a stochastic baseline is needed.

    Example: ``retail-sim live play --host localhost --port 8765 --no-auth``
    """
    import asyncio

    from retail_simulator.live import SeatAssigned, play_one_game

    if policy != "default":
        _echo(
            _status(
                "[FAIL]",
                f"unknown --policy {policy!r}; only 'default' is supported in this slice",
                "red",
            )
        )
        raise typer.Exit(code=1)

    token = _resolve_token(no_auth)
    uri = f"ws://{host}:{port}"

    # `seed` is accepted now so the flag surface is stable when a stochastic
    # policy is added, but it does not affect the passive default policy
    # `play_one_game` submits (a writable copy of REGISTRY_DEFAULT_ACTION).

    _log.info(
        "running live play: uri=%s, agent_name=%s, policy=%s, auth=%s",
        uri,
        agent_name,
        policy,
        "off" if token is None else "on",
    )

    # Print the human-readable seat assignment at the original timing — right
    # after the server assigns the seat, before the first action — via the
    # play-loop's `on_seat_assigned` seam (the loop itself has ONE owner now:
    # `play_one_game`).
    def _print_seat(seat: SeatAssigned) -> None:
        _echo(
            _status(
                "[OK]",
                (f"seat_index={seat.seat_index} game_id={seat.game_id} n_ticks={seat.n_ticks}"),
                "green",
            )
        )

    try:
        final = asyncio.run(
            play_one_game(uri, agent_name, token, policy=policy, on_seat_assigned=_print_seat)
        )
    except ConnectionError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    # `play_one_game` returns None when the socket closed without an end-of-game
    # frame (a fatal error would have raised ConnectionError) — treat it as an
    # aborted game, mirroring the prior inline behavior.
    if final is None:
        _echo(_status("[FAIL]", "game ended without final scores", "red"))
        raise typer.Exit(code=1)

    _echo(json.dumps(asdict(final), indent=2, default=str))
