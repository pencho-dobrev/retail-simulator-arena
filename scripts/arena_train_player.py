"""Train one RL player for the 25-region arena with a named recipe.

The league driver (``scripts/phase4_rl_confirm.py``) is reused verbatim. Only its env resolver
is replaced, because the driver still exposes no flags for the wage-mechanic knobs (it does have
its own ``--reward`` now, passed through in ``driver_argv`` below as a no-op consistency check).
The full effective recipe is written to ``recipe.json`` in the pool directory, since the
driver's own summary records only part of it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from retail_simulator.core.config import CoreConfig
from retail_simulator.harness.arena import (
    arena_balanced_config,
    arena_home_regions,
    arena_lock_config,
    arena_region_table_hash,
)

WORLDS: tuple[str, ...] = ("lock", "balanced")
REWARDS: tuple[str, ...] = ("profit", "weighted")
_SCRIPTS_DIR = Path(__file__).resolve().parent
_DEFAULT_POOL_ROOT = _SCRIPTS_DIR.parent / "runs" / "phase4_rl_confirm"


def build_config(
    *,
    world: str,
    reward: str,
    happiness_decay: float | None = None,
    writeoff_frac_min: float | None = None,
    writeoff_frac_max: float | None = None,
    wage_shortfall_scale: float | None = None,
    ramp_ticks: int | None = None,
    overhead_reference_stores: float | None = None,
    restructure_after_ticks: int | None = None,
    restructure_keep_stores: int | None = None,
    max_share_per_region: float | None = None,
    marketing_cost_exponent: float | None = None,
    promotion_contest_weight: float | None = None,
    promo_cost_per_unit: float | None = None,
    stockpile_build: float | None = None,
    stockpile_decay: float | None = None,
) -> CoreConfig:
    """ """
    if world not in WORLDS:
        raise ValueError(f"world must be one of {WORLDS}, got {world!r}")
    if reward not in REWARDS:
        raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")
    cfg = arena_balanced_config() if world == "balanced" else arena_lock_config()
    if reward == "weighted":
        cfg = replace(cfg, reward=replace(cfg.reward, mode="weighted"))
    wage_overrides = {
        field: value
        for field, value in (
            ("happiness_decay", happiness_decay),
            ("writeoff_frac_min", writeoff_frac_min),
            ("writeoff_frac_max", writeoff_frac_max),
            ("wage_shortfall_scale", wage_shortfall_scale),
            ("overhead_reference_stores", overhead_reference_stores),
            ("restructure_after_ticks", restructure_after_ticks),
            ("restructure_keep_stores", restructure_keep_stores),
        )
        if value is not None
    }
    if wage_overrides:
        # The override field names come from the caller (CLI flags), so they are not
        # statically known; ``WageConfig`` mixes bool/int/float fields (MECHANIC 4
        # added the int-typed pair above), which mypy cannot verify a same-key-set
        # ``**dict`` unpack against. The cast says exactly that -- kept to this one
        # line rather than loosening ``build_config``'s own signature -- mirroring
        # ``harness/arena_match.py``'s ``_apply()`` helper, which exists for the
        # identical reason.
        cfg = replace(cfg, wage=replace(cast(Any, cfg.wage), **wage_overrides))
    if ramp_ticks is not None:
        cfg = replace(cfg, warehouse=replace(cfg.warehouse, ramp_ticks=ramp_ticks))
    if max_share_per_region is not None:
        cfg = replace(cfg, demand=replace(cfg.demand, max_share_per_region=max_share_per_region))
    if marketing_cost_exponent is not None:
        cfg = replace(
            cfg, marketing=replace(cfg.marketing, marketing_cost_exponent=marketing_cost_exponent)
        )
    if promotion_contest_weight is not None:
        cfg = replace(
            cfg,
            promotion=replace(cfg.promotion, promotion_contest_weight=promotion_contest_weight),
        )
    if promo_cost_per_unit is not None:
        cfg = replace(
            cfg, promotion=replace(cfg.promotion, promo_cost_per_unit=promo_cost_per_unit)
        )
    if stockpile_build is not None:
        cfg = replace(cfg, promotion=replace(cfg.promotion, stockpile_build=stockpile_build))
    if stockpile_decay is not None:
        cfg = replace(cfg, promotion=replace(cfg.promotion, stockpile_decay=stockpile_decay))
    return cfg


def arena_knobs(cfg: CoreConfig) -> dict[str, float]:
    """ """
    knobs: dict[str, float] = {
        "overhead_reference_stores": cfg.wage.overhead_reference_stores,
        "overhead_exponent": cfg.wage.overhead_exponent,
        "provided_capacity_per_store": cfg.warehouse.provided_capacity_per_store,
        "credit_limit": cfg.wage.credit_limit,
        "overdraft_rate": cfg.wage.overdraft_rate,
        "writeoff_frac_max": cfg.wage.writeoff_frac_max,
    }
    if cfg.wage.restructure_after_ticks:
        knobs["restructure_after_ticks"] = cfg.wage.restructure_after_ticks
        knobs["restructure_keep_stores"] = cfg.wage.restructure_keep_stores
        knobs["restructure_debt_threshold"] = cfg.wage.restructure_debt_threshold
    if cfg.demand.max_share_per_region < 1.0:
        knobs["max_share_per_region"] = cfg.demand.max_share_per_region
    if cfg.marketing.marketing_cost_exponent != 1.0:
        knobs["marketing_cost_exponent"] = cfg.marketing.marketing_cost_exponent
    if cfg.promotion.promotion_contest_weight != 0.0:
        knobs["promotion_contest_weight"] = cfg.promotion.promotion_contest_weight
    if cfg.promotion.promo_cost_per_unit != 0.009:
        knobs["promo_cost_per_unit"] = cfg.promotion.promo_cost_per_unit
    if cfg.promotion.stockpile_build != 0.17:
        knobs["stockpile_build"] = cfg.promotion.stockpile_build
    if cfg.promotion.stockpile_decay != 0.7:
        knobs["stockpile_decay"] = cfg.promotion.stockpile_decay
    return knobs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one arena RL player with a named recipe.")
    parser.add_argument("label", help="player name, also used as its pool directory name")
    parser.add_argument("--world", choices=WORLDS, default="balanced")
    parser.add_argument("--reward", choices=REWARDS, default="profit")
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--seeds", default="0,0,42", help="league,curriculum,core seeds")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument(
        "--train-steps-per-round", type=int, default=None, help="default: the driver's own"
    )
    parser.add_argument(
        "--episode-ticks",
        type=int,
        default=None,
        help="decision 8 (taken 2026-09-18): the league training-episode horizon, forwarded to the driver's own --league-max-episode-steps -> run_league_training(max_episode_steps=...); default: None (no truncation, one opponent per round, byte-identical to every prior run). Recorded in recipe.json's episode_ticks (null when unset)",
    )
    parser.add_argument(
        "--pfsp-mix-uniform",
        type=float,
        default=None,
        help="decision 10 (taken 2026-09-19): the PFSP uniform-mixing floor, forwarded to the driver's own --pfsp-mix-uniform -> PFSPOpponentDistribution(mix_uniform=...); default: None (no floor, byte-identical to every prior run). Recorded in recipe.json's pfsp_mix_uniform (null when unset)",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="owner decision 12a (taken 2026-09-19, generation 11's training lever): "
        "forwarded to the driver's own --warm-start -> "
        "run_league_training(warm_start=True), which continues round k>=2 from "
        "round (k-1)'s just-trained checkpoint instead of an independent "
        "from-scratch init; default: off, byte-identical to every prior run. "
        "Recorded in recipe.json's warm_start (false when unset)",
    )
    parser.add_argument("--happiness-decay", type=float, default=None)
    parser.add_argument("--writeoff-frac-min", type=float, default=None)
    parser.add_argument("--writeoff-frac-max", type=float, default=None)
    parser.add_argument("--wage-shortfall-scale", type=float, default=None)
    parser.add_argument(
        "--ramp-ticks",
        type=int,
        default=None,
        help="warehouse.ramp_ticks (store-ramp-up); default: inherit the world's own value (0, inert)",
    )
    parser.add_argument(
        "--overhead-reference-stores",
        type=float,
        default=None,
        help="wage.overhead_reference_stores (K-arm-overhead-5, re-tuned by the v4 promotion-combo re-tune); default: inherit the world's own value (7.15 for --world lock, 4.5 for --world balanced); pass 7.15 to reproduce a balanced-world recipe recorded before 2026-09-17, or 5.0 to reproduce one recorded under before this re-tune",
    )
    parser.add_argument(
        "--restructure-after-ticks",
        type=int,
        default=None,
        help="wage.restructure_after_ticks (Phase 4 MECHANIC 4, RESTRUCTURING -- the zombie-seat recovery path, decision 7); default: inherit the world's own value (0, inert, for --world lock; 50, armed, for --world balanced since 2026-09-18); pass 0 to reproduce a balanced-world recipe recorded before that date",
    )
    parser.add_argument(
        "--restructure-keep-stores",
        type=int,
        default=None,
        help="wage.restructure_keep_stores; default: inherit the world's own value (1 for --world lock, 3 for --world balanced since 2026-09-18)",
    )
    parser.add_argument(
        "--max-share-per-region",
        type=float,
        default=None,
        help="demand.max_share_per_region (Phase 4 MECHANIC 5, REGIONAL SHARE CEILING — decision 9); default: inherit the world's own value (1.0 for --world lock; 0.6 for --world balanced since 2026-09-19); pass 1.0 to reproduce a balanced-world recipe recorded before that date -- this knob is NOT match/eval-only, it changes the world the recipe TRAINS in",
    )
    parser.add_argument(
        "--marketing-cost-exponent",
        type=float,
        default=None,
        help="marketing.marketing_cost_exponent (MECHANIC 6 candidate, CONVEX MARKETING COST — a dueling experiment); default: inherit the world's own value (1.0, linear/off, for both --world lock and --world balanced -- NOT armed by this task). NOT match/eval-only, like --max-share-per-region: changes the world the recipe TRAINS in, so a policy must be RETRAINED at a value > 1.0 to reveal a new, interior spend optimum",
    )
    parser.add_argument(
        "--promotion-contest-weight",
        type=float,
        default=None,
        help="promotion.promotion_contest_weight (MECHANIC-6b candidate, PROMOTION CONTEST — a dueling experiment); default: inherit the world's own value (0.0, absolute/off, for --world lock; 1.0, armed, for --world balanced since K-arm-v4-promotion-combo); pass 0.0 to reproduce a balanced-world recipe recorded before that. NOT match/eval-only, like --marketing-cost-exponent: changes the world the recipe TRAINS in, so a policy must be RETRAINED at a value > 0.0 to reveal the new, reactive equilibrium",
    )
    parser.add_argument(
        "--promo-cost-per-unit",
        type=float,
        default=None,
        help="promotion.promo_cost_per_unit (promotion combo #2b); default: inherit the world's own value (0.009, the class default, for --world lock; 0.0, armed, for --world balanced since K-arm-v4-promotion-combo); pass 0.009 to reproduce a balanced-world recipe recorded before that",
    )
    parser.add_argument(
        "--stockpile-build",
        type=float,
        default=None,
        help="promotion.stockpile_build (promotion combo #2b); default: inherit the world's own value (0.17, the class default, for --world lock; 0.0, armed, for --world balanced since K-arm-v4-promotion-combo); pass 0.17 to reproduce a balanced-world recipe recorded before that",
    )
    parser.add_argument(
        "--stockpile-decay",
        type=float,
        default=None,
        help="promotion.stockpile_decay (promotion combo #2b); default: inherit the world's own value (0.7, the class default, for both --world lock and --world balanced -- NOT armed by this task)",
    )
    parser.add_argument("--pool-root", type=Path, default=_DEFAULT_POOL_ROOT)
    return parser


def _load_driver() -> ModuleType:
    """Import the league driver, which lives beside this script rather than in the package."""
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    import phase4_rl_confirm  # noqa: PLC0415

    return phase4_rl_confirm


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = build_config(
            world=args.world,
            reward=args.reward,
            happiness_decay=args.happiness_decay,
            writeoff_frac_min=args.writeoff_frac_min,
            writeoff_frac_max=args.writeoff_frac_max,
            wage_shortfall_scale=args.wage_shortfall_scale,
            ramp_ticks=args.ramp_ticks,
            overhead_reference_stores=args.overhead_reference_stores,
            restructure_after_ticks=args.restructure_after_ticks,
            restructure_keep_stores=args.restructure_keep_stores,
            max_share_per_region=args.max_share_per_region,
            marketing_cost_exponent=args.marketing_cost_exponent,
            promotion_contest_weight=args.promotion_contest_weight,
            promo_cost_per_unit=args.promo_cost_per_unit,
            stockpile_build=args.stockpile_build,
            stockpile_decay=args.stockpile_decay,
        )
    except ValueError as exc:
        parser.error(str(exc))

    pool_dir = args.pool_root / args.label
    pool_dir.mkdir(parents=True, exist_ok=True)
    knobs = arena_knobs(cfg)
    home_regions = arena_home_regions(2)
    recipe: dict[str, Any] = {
        "label": args.label,
        "world": args.world,
        "reward_mode": cfg.reward.mode,
        "ent_coef": args.ent_coef,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "train_steps_per_round": args.train_steps_per_round,
        "episode_ticks": args.episode_ticks,
        "pfsp_mix_uniform": args.pfsp_mix_uniform,
        "warm_start": args.warm_start,
        "gae_lambda": 0.98,
        "seed_archetypes": True,
        "mask_expansion": True,
        "home_regions": list(home_regions),
        "happiness_decay": cfg.wage.happiness_decay,
        "writeoff_frac_min": cfg.wage.writeoff_frac_min,
        "wage_shortfall_scale": cfg.wage.wage_shortfall_scale,
        "ramp_ticks": cfg.warehouse.ramp_ticks,
        "knobs": knobs,
    }
    (pool_dir / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")

    driver = _load_driver()
    region_table_hash = arena_region_table_hash(cfg)

    def resolve_env(_env: str) -> tuple[CoreConfig, tuple[int, ...], str, dict[str, float]]:
        return cfg, home_regions, region_table_hash, knobs

    driver._resolve_env = resolve_env
    driver_argv = [
        "--rounds",
        str(args.rounds),
        "--seed-archetypes",
        "--gae-lambda",
        "0.98",
        "--env",
        "arena-balanced" if args.world == "balanced" else "arena",
        # H5: passed explicitly so the driver's own --reward agrees with what
        # build_config() already baked into cfg above -- a no-op confirmation.
        "--reward",
        args.reward,
        "--mask-expansion",
        "--ent-coef",
        str(args.ent_coef),
        "--seeds",
        args.seeds,
        "--label",
        args.label,
        "--pool-dir",
        str(pool_dir),
        "--summary",
        str(pool_dir / "summary.json"),
    ]
    if args.train_steps_per_round is not None:
        driver_argv += ["--train-steps-per-round", str(args.train_steps_per_round)]
    if args.episode_ticks is not None:
        driver_argv += ["--league-max-episode-steps", str(args.episode_ticks)]
    if args.pfsp_mix_uniform is not None:
        driver_argv += ["--pfsp-mix-uniform", str(args.pfsp_mix_uniform)]
    if args.warm_start:
        driver_argv += ["--warm-start"]
    return int(driver.run(driver._build_parser().parse_args(driver_argv)))


if __name__ == "__main__":
    raise SystemExit(main())
