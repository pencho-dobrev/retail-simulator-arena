"""Phase 2.1 — the game-series runner (the unit-testable heart of multi-game runs).

A *series* is a sequence of independent live games sharing a base
:class:`~retail_simulator.live.server.ServeConfig`, differing only by seed
(``base.seed + i`` for game ``i``; the port stays fixed — games run one at a
time, not concurrently). This module owns the PURE reduction: drive each game
via an injectable ``serve_fn`` (defaulting to a real :func:`serve_game` run),
keep only a reduced :class:`SeriesGameResult` per game, and aggregate per-seat
mean / population-stddev over the CLEAN games.

Deliberately socket-free and filesystem-free at this layer: all I/O lives
behind ``serve_fn``. That keeps the aggregation logic (seed derivation, clean
vs failed bookkeeping, the N=1 / all-failed stddev edge cases) directly
unit-testable by injecting synthetic :class:`ServeResult`s — no live server
required. The lazy-``websockets`` contract holds: this module imports only
``ServeConfig`` / ``ServeResult`` / ``serve_game`` from ``live.server`` (already
imported by the package ``__init__``), so ``import retail_simulator.live`` stays
cheap; ``websockets`` is paid for only when the default ``serve_fn`` actually
opens a socket inside :func:`serve_game`.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from retail_simulator.live.server import ServeConfig, ServeResult, serve_game


@dataclass(frozen=True)
class SeriesGameResult:
    """The reduced record of one game in a series — the ONLY per-game state retained.

    Holds the seed used and the game's final per-seat scores (``{}`` for a
    failed game). The heavyweight :class:`ServeResult` / env handles / per-tick
    data are dropped after this reduction so a long series stays bounded in
    memory.
    """

    game_index: int
    game_id: str
    seed: int
    exit_status: int
    final_scores: dict[str, float]  # {} for a failed game


@dataclass(frozen=True)
class SeriesResult:
    """Aggregate outcome of a whole series.

    ``per_seat_mean`` / ``per_seat_stddev`` are computed over CLEAN games only
    (``exit_status == 0``); the stddev is the POPULATION stddev (``ddof=0``) so
    a single clean game yields ``0.0`` rather than NaN. Both maps are empty when
    no game finished cleanly (no seat keys are derivable). ``games`` lists every
    attempted game — clean and failed alike, in order.
    """

    series_id: str
    games_requested: int
    games_played: int  # clean (exit_status == 0)
    games_failed: int
    per_seat_mean: dict[str, float | None]
    per_seat_stddev: dict[str, float | None]
    games: list[SeriesGameResult]


def _default_serve_fn(cfg: ServeConfig) -> ServeResult:
    """Run one real game by driving the async :func:`serve_game` to completion.

    The default ``serve_fn``: the only place the live server's async entry point
    is awaited. Tests inject a synthetic replacement so the series logic is
    exercised without a socket.
    """
    return asyncio.run(serve_game(cfg))


def run_series(
    base_config: ServeConfig,
    *,
    games: int,
    series_id: str | None = None,
    serve_fn: Callable[[ServeConfig], ServeResult] = _default_serve_fn,
) -> SeriesResult:
    """Run ``games`` independent games and aggregate their per-seat scores.

    Each game ``i`` runs with ``seed = base_config.seed + i`` (the port is held
    fixed — games are sequential, not concurrent). A failed game
    (``exit_status != 0``) is recorded and the series CONTINUES; it never aborts.
    Per-seat mean / population-stddev are aggregated over the clean games only.
    """
    resolved_series_id = series_id if series_id is not None else f"series-{uuid4().hex[:12]}"

    game_results: list[SeriesGameResult] = []
    for i in range(games):
        cfg_i = dataclasses.replace(base_config, seed=base_config.seed + i)
        result = serve_fn(cfg_i)
        game_results.append(
            SeriesGameResult(
                game_index=i,
                game_id=result.game_id,
                seed=cfg_i.seed,
                exit_status=result.exit_status,
                final_scores=result.final_scores,
            )
        )

    clean_results = [g for g in game_results if g.exit_status == 0]
    games_played = len(clean_results)
    games_failed = games - games_played

    # Collect per-seat score lists over the union of agent ids seen in clean
    # games (a failed game contributes no scores — its final_scores is {}).
    scores_by_seat: dict[str, list[float]] = {}
    for clean in clean_results:
        for agent_id, score in clean.final_scores.items():
            scores_by_seat.setdefault(agent_id, []).append(score)

    per_seat_mean: dict[str, float | None] = {}
    per_seat_stddev: dict[str, float | None] = {}
    for agent_id, scores in scores_by_seat.items():
        per_seat_mean[agent_id] = float(np.mean(scores))
        # Population stddev (ddof=0): N=1 -> 0.0, never NaN.
        per_seat_stddev[agent_id] = float(np.std(scores))

    return SeriesResult(
        series_id=resolved_series_id,
        games_requested=games,
        games_played=games_played,
        games_failed=games_failed,
        per_seat_mean=per_seat_mean,
        per_seat_stddev=per_seat_stddev,
        games=game_results,
    )


__all__ = ["SeriesGameResult", "SeriesResult", "run_series"]
