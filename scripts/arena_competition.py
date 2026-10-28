"""Multi-seat arena competition: learned players against scripted archetypes, one unbroken match.

This script is argument parsing plus CSV/JSON writing. The seating, the per-seat expansion
masking and the "log the EFFECTIVE decision" rule are OWNED by
:mod:`retail_simulator.envs.arena_match` (``ArenaMatch`` / ``resolve_arena_world``), which
the leaderboard generator and the live demo drive too — change a rule there, not here. A
match needs at least two seats, so a lone ``--player``/``--npc`` is rejected.

For a masked seat the logged decision is the effective one -- the post-mask action the world
executed. Decoding the raw action would record a preference the world never acted on.

Both the learned policies and the archetypes read regions by index, so a single seating
decides much of the mid-field. Rotate ``--homes`` across replicates and aggregate before
reporting any rank below first.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.arena import arena_home_regions
from retail_simulator.envs.arena_match import (
    LOGGED_LEVERS,
    MIN_SEATS,
    ArenaMatch,
    ArenaSeatSpec,
    resolve_arena_world,
)
from retail_simulator.envs.geo_archetypes import GEO_ARCHETYPE_LABELS
from retail_simulator.logging_setup import configure_logging

__all__ = ["LOGGED_LEVERS", "main"]

_log = logging.getLogger("arena_competition")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play one unbroken arena match and log every seat's decisions each round."
    )
    parser.add_argument("--rounds", type=int, default=1000, help="rounds to play (at least 1)")
    parser.add_argument("--seed", type=int, default=0, help="world seed")
    parser.add_argument(
        "--player",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="a trained checkpoint, seated first with expansion masking on; repeatable",
    )
    parser.add_argument(
        "--npc",
        action="append",
        default=[],
        metavar="NAME",
        help="a geographic archetype, seated unmasked after the players; repeatable; one of "
        + ", ".join(GEO_ARCHETYPE_LABELS),
    )
    parser.add_argument("--out", required=True, type=Path, help="standings JSON to write")
    parser.add_argument("--csv", required=True, type=Path, help="per-round decision log to write")
    parser.add_argument(
        "--homes", default=None, help="comma-separated home region for each seat, in seat order"
    )
    parser.add_argument("--quiet", action="store_true", help="write the files, print no table")

    world = parser.add_argument_group("world")
    world.add_argument(
        "--balanced",
        action="store_true",
        help="play in arena_balanced_config(), the world the competition is played in; "
        "without this flag the match plays the lock world",
    )
    world.add_argument(
        "--overhead",
        type=float,
        default=None,
        help="the LOCK-world sweep: build the world from "
        "arena_config(overhead_reference_stores=VALUE); cannot be combined with --balanced "
        "(see --overhead-reference-stores below to override the knob on a chosen world "
        "instead)",
    )

    brakes = parser.add_argument_group(
        "growth-brake overrides (each defaults to the chosen world's own value)"
    )
    brakes.add_argument(
        "--provided-capacity",
        type=float,
        default=None,
        help="warehouse.provided_capacity_per_store",
    )
    brakes.add_argument(
        "--capacity-per-store",
        type=float,
        default=None,
        help="warehouse.capacity_per_store, the REQUIRED capacity; a shortfall exists only "
        "when it exceeds the provided capacity",
    )
    brakes.add_argument(
        "--penalty-scale", type=float, default=None, help="warehouse.penalty_severity_scale"
    )
    brakes.add_argument(
        "--penalty-frac-max", type=float, default=None, help="warehouse.penalty_frac_max"
    )
    brakes.add_argument(
        "--ramp-ticks",
        type=int,
        default=None,
        help="warehouse.ramp_ticks — the new-store SPEED tax: ticks for a freshly opened store to reach full contribution to the provided warehouse top-up",
    )
    brakes.add_argument(
        "--overdraft",
        type=float,
        default=None,
        help="wage.overdraft_rate; needs a world with a credit line - see --balanced",
    )
    brakes.add_argument(
        "--writeoff-frac-max", type=float, default=None, help="wage.writeoff_frac_max"
    )
    brakes.add_argument(
        "--own-share-coeff", type=float, default=None, help="wage.wage_own_share_coeff"
    )
    brakes.add_argument(
        "--overhead-reference-stores",
        type=float,
        default=None,
        help="wage.overhead_reference_stores, the growth-overhead footprint-span brake — overrides the CHOSEN world's own value in place; distinct from --overhead above, which builds an entirely different world and stays incompatible with --balanced",
    )
    brakes.add_argument(
        "--restructure-after-ticks",
        type=int,
        default=None,
        help="wage.restructure_after_ticks (RESTRUCTURING — the insolvent-seat recovery path): consecutive ticks a seat's cash must stay below -(credit_limit + restructure_debt_threshold) before it is restructured; 0 turns the rule off; default: inherit the chosen world's own value (0, off, for the lock; the balanced world arms it at 50); pass 0 to turn the rule off",
    )
    brakes.add_argument(
        "--restructure-keep-stores",
        type=int,
        default=None,
        help="wage.restructure_keep_stores: how many of the seat's best regions (by market share) survive a restructuring; default: inherit the chosen world's own value (1 for the lock, 3 for --balanced)",
    )
    brakes.add_argument(
        "--max-share-per-region",
        type=float,
        default=None,
        help="demand.max_share_per_region (REGIONAL SHARE CEILING -- the anti-concentration brake): the maximum per-region choice share any one present retailer may hold; the excess flows to the region's other present, uncapped retailers, and whatever nobody present can absorb goes unserved; default: inherit the chosen world's own value (1.0, off, for the lock; 0.6 for --balanced)",
    )
    brakes.add_argument(
        "--marketing-cost-exponent",
        type=float,
        default=None,
        help="marketing.marketing_cost_exponent (CONVEX MARKETING COST): raises spend_fraction to this power before pricing it, so a high spend costs disproportionately more at the margin; default: inherit the chosen world's own value (1.0, linear/off, for both the lock and --balanced)",
    )
    brakes.add_argument(
        "--promotion-contest-weight",
        type=float,
        default=None,
        help="promotion.promotion_contest_weight (PROMOTION CONTEST): nets each retailer's promotion against its present rivals' mean promotion before the utility reads it, so out-promoting rivals steals share and a matched raid cancels; default: inherit the chosen world's own value (0.0, absolute/off, for the lock; 1.0, armed, for --balanced); pass 0.0 to turn the contest off",
    )
    brakes.add_argument(
        "--promo-cost-per-unit",
        type=float,
        default=None,
        help="promotion.promo_cost_per_unit (the calibration partner to --promotion-contest-weight): default: inherit the chosen world's own value (0.009, the class default, for the lock; 0.0, armed, for --balanced); pass 0.009 to restore the class default",
    )
    brakes.add_argument(
        "--stockpile-build",
        type=float,
        default=None,
        help="promotion.stockpile_build: default: inherit the chosen world's own value (0.17, the class default, for the lock; 0.0, armed, for --balanced); pass 0.17 to restore the class default",
    )
    brakes.add_argument(
        "--stockpile-decay",
        type=float,
        default=None,
        help="promotion.stockpile_decay: default: inherit the chosen world's own value (0.7, the class default, for both the lock and --balanced)",
    )
    return parser


def _resolve_world(args: argparse.Namespace) -> tuple[CoreConfig, str, dict[str, float]]:
    """Build the match's world, a label naming its base world, and the overrides applied.

    A thin adapter over :func:`~retail_simulator.envs.arena_match.resolve_arena_world`,
    which owns the rule (``scripts/gen_leaderboard_artifact.py`` resolves its world through
    the same function, so the two scripts cannot drift).
    """
    return resolve_arena_world(
        balanced=args.balanced,
        overhead=args.overhead,
        provided_capacity_per_store=args.provided_capacity,
        capacity_per_store=args.capacity_per_store,
        penalty_severity_scale=args.penalty_scale,
        penalty_frac_max=args.penalty_frac_max,
        ramp_ticks=args.ramp_ticks,
        overdraft_rate=args.overdraft,
        writeoff_frac_max=args.writeoff_frac_max,
        wage_own_share_coeff=args.own_share_coeff,
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


def _effective_knobs(cfg: CoreConfig) -> dict[str, float]:
    """The effective knob values of the match's world, as a flat dict for the CSV/JSON."""
    knobs: dict[str, float] = {
        "overhead_reference_stores": cfg.wage.overhead_reference_stores,
        "overhead_exponent": cfg.wage.overhead_exponent,
        "provided_capacity_per_store": cfg.warehouse.provided_capacity_per_store,
        "capacity_per_store": cfg.warehouse.capacity_per_store,
        "penalty_severity_scale": cfg.warehouse.penalty_severity_scale,
        "penalty_frac_max": cfg.warehouse.penalty_frac_max,
        "credit_limit": cfg.wage.credit_limit,
        "overdraft_rate": cfg.wage.overdraft_rate,
        "writeoff_frac_max": cfg.wage.writeoff_frac_max,
        "wage_own_share_coeff": cfg.wage.wage_own_share_coeff,
    }
    if cfg.warehouse.ramp_ticks:
        knobs["ramp_ticks"] = cfg.warehouse.ramp_ticks
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


def _parse_players(specs: Sequence[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    players: dict[str, str] = {}
    for spec in specs:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            parser.error(f"--player expects LABEL=PATH, got {spec!r}")
        if label in players:
            parser.error(f"--player label {label!r} is used twice; every seat needs its own label")
        players[label] = path
    return players


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.overhead is not None and args.balanced:
        parser.error("--overhead builds its own world from arena_config(); drop --balanced")
    if args.overhead is not None and args.overhead_reference_stores is not None:
        # --overhead already sets overhead_reference_stores (it builds the world from that value), so
        # combining it with the brake would build one world but label another -- a silent mismatch.
        # Reject the pair, like --balanced above.
        parser.error(
            "--overhead already sets overhead_reference_stores; drop --overhead-reference-stores"
        )
    unknown = [name for name in args.npc if name not in GEO_ARCHETYPE_LABELS]
    if unknown:
        parser.error(f"unknown --npc {unknown}; choose from {list(GEO_ARCHETYPE_LABELS)}")
    players = _parse_players(args.player, parser)
    seat_names = list(players) + list(args.npc)
    if len(seat_names) < MIN_SEATS:
        # A match is a competition: ArenaMatch enforces the same floor, but raising it
        # here keeps a bad field an argparse error (exit 2) like every other one.
        parser.error(f"seat at least {MIN_SEATS} of --player / --npc")
    duplicated = sorted({name for name in seat_names if seat_names.count(name) > 1})
    if duplicated:
        # Standings are keyed by name, so two seats sharing one would silently merge.
        parser.error(f"seat names must be unique, but {duplicated} appear more than once")

    n_seats = len(seat_names)
    if args.homes:
        try:
            homes = tuple(int(token) for token in args.homes.split(","))
        except ValueError:
            parser.error(f"--homes expects comma-separated integers, got {args.homes!r}")
        if len(homes) != n_seats:
            parser.error(f"--homes names {len(homes)} regions for {n_seats} seats")
    else:
        homes = arena_home_regions(n_seats)

    try:
        cfg, env_label, overrides = _resolve_world(args)
    except ValueError as exc:
        parser.error(str(exc))
    seats = [
        ArenaSeatSpec(name=label, kind="rl", checkpoint=path) for label, path in players.items()
    ]
    seats += [ArenaSeatSpec(name=name, kind="npc", archetype=name) for name in args.npc]
    try:
        match = ArenaMatch(cfg, seats, seed=args.seed, homes=homes)
    except ValueError as exc:
        parser.error(str(exc))
    masked = match.masked_seats

    _log.info(
        "arena match starting: %s world · %d seat(s) (%d masked) · %d round(s) · seed=%d",
        env_label,
        n_seats,
        len(masked),
        args.rounds,
        args.seed,
    )
    rows: list[dict[str, Any]] = []
    totals = {name: {"score": 0.0, "revenue": 0.0, "profit": 0.0} for name in seat_names}
    log_every = max(args.rounds // 10, 1)

    for round_index in range(args.rounds):
        result = match.step()
        for seat, seat_round in enumerate(result.seats):
            name = seat_names[seat]
            totals[name]["score"] += seat_round.reward
            totals[name]["revenue"] += seat_round.revenue
            totals[name]["profit"] += seat_round.profit
            row: dict[str, Any] = {
                "round": result.round,
                "seat": seat,
                "name": name,
                "kind": seats[seat].kind,
                "score": round(seat_round.reward, 3),
                "revenue": round(seat_round.revenue, 2),
                "profit": round(seat_round.profit, 2),
                "market_share": round(seat_round.market_share, 5),
                "cash": round(seat_round.cash, 2),
                "stores": seat_round.stores,
                # A region can only be opened when it holds no store yet, so the count of
                # regions opened this round IS the round's store growth.
                "opened": len(seat_round.opened_regions),
                "employee_happiness": round(seat_round.employee_happiness, 4),
            }
            for lever in LOGGED_LEVERS:
                value = seat_round.levers.get(lever)
                row[lever] = None if value is None else round(float(value), 4)
            rows.append(row)
        if (round_index + 1) % log_every == 0 or round_index + 1 == args.rounds:
            leader = max(seat_names, key=lambda name: totals[name]["score"])
            _log.info(
                "round %d/%d · leader: %s (%s score)",
                round_index + 1,
                args.rounds,
                leader,
                f"{totals[leader]['score']:,.0f}",
            )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    standings: dict[str, dict[str, Any]] = {}
    for seat, name in enumerate(seat_names):
        last = rows[len(rows) - n_seats + seat]
        standings[name] = {
            "kind": seats[seat].kind,
            "seat": seat,
            "home_region": homes[seat],
            "total_score": round(totals[name]["score"], 2),
            "total_revenue": round(totals[name]["revenue"], 2),
            "total_profit": round(totals[name]["profit"], 2),
            "final_market_share": last["market_share"],
            "final_cash": last["cash"],
            "final_stores": last["stores"],
        }
    payload = {
        "rounds": args.rounds,
        "seed": args.seed,
        "n_seats": n_seats,
        "env": env_label,
        "overrides": overrides,
        "knobs": _effective_knobs(cfg),
        "home_regions": list(homes),
        "masked_seats": list(masked),
        "standings": standings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    _log.info("match done: decisions -> %s · results -> %s", args.csv, args.out)

    if not args.quiet:
        order = sorted(standings, key=lambda name: -standings[name]["total_score"])
        print(
            f"{'rank':>4} {'name':<14} {'kind':<4} {'score':>14} {'revenue':>14} "
            f"{'profit':>13} {'share':>7} {'stores':>7}"
        )
        for rank, name in enumerate(order, start=1):
            entry = standings[name]
            print(
                f"{rank:>4} {name:<14} {entry['kind']:<4} {entry['total_score']:>14,.0f} "
                f"{entry['total_revenue']:>14,.0f} {entry['total_profit']:>13,.0f} "
                f"{entry['final_market_share']:>7.3f} {entry['final_stores']:>7d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
