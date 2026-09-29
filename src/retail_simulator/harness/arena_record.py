"""Turn a running :class:`~retail_simulator.harness.arena_match.ArenaMatch` into
``arena-match-v1``.

**Cumulative vs window.** ``score_total`` / ``revenue_total`` / ``profit_total`` run from
round 1; ``profit_window`` and ``opened_regions`` cover only ``(round - snapshot_every,
round]``, the animation step a player actually watches.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from retail_simulator.core.config import CoreConfig
from retail_simulator.harness.arena import ARENA_REGION_CLASSES
from retail_simulator.harness.arena_match import LOGGED_LEVERS, ArenaSeatSpec, RoundResult

SCHEMA: str = "arena-match-v1"

# Levers carried as integers in a snapshot's ``decision`` — both are choice INDICES.
_INTEGER_LEVERS: frozenset[str] = frozenset({"expansion", "automation"})

_CASH_SCALE_DECIMALS: int = 2
_UNIT_SCALE_DECIMALS: int = 4


def region_class_name(index: int) -> str:
    """The region class at ``index`` under the arena's fixed cycle.

    Region 0 is the stranded FRONTIER slot; regions 1.. cycle
    :data:`~retail_simulator.harness.arena.ARENA_REGION_CLASSES` as ``(index - 1) mod 4``
    — the same rule :func:`~retail_simulator.harness.arena.arena_regions` builds the table
    with, read off the SAME constant rather than a second hardcoded list of four names.
    """
    if index == 0:
        return ARENA_REGION_CLASSES[-1].name
    return ARENA_REGION_CLASSES[(index - 1) % len(ARENA_REGION_CLASSES)].name


def region_table(config: CoreConfig) -> list[dict[str, Any]]:
    """ """
    return [
        {
            "index": index,
            "class": region_class_name(index),
            "base_demand": round(float(region.base_regional_demand), _UNIT_SCALE_DECIMALS),
            "enterable": index >= 1,
        }
        for index, region in enumerate(config.demand.regions)
    ]


def build_header(
    *,
    env_label: str,
    seed: int,
    rounds: int,
    snapshot_every: int,
    seats: Sequence[ArenaSeatSpec],
    homes: Sequence[int],
    config: CoreConfig,
) -> dict[str, Any]:
    """Everything an ``arena-match-v1`` document says about a match except its snapshots.

    ``rounds`` is the number PLANNED (a cancelled or still-running demo carries fewer
    snapshots than that implies — read ``rounds_done`` from the demo envelope, never the
    header, to know how far a match actually got).
    """
    if len(homes) != len(seats):
        raise ValueError(f"homes names {len(homes)} regions for {len(seats)} seats")
    return {
        "schema": SCHEMA,
        "env": env_label,
        "seed": int(seed),
        "rounds": int(rounds),
        "snapshot_every": int(snapshot_every),
        "n_regions": len(config.demand.regions),
        "regions": region_table(config),
        "seats": [
            {
                "seat": index,
                "name": seat.name,
                "kind": seat.kind,
                "player": seat.player,
                "archetype": seat.archetype,
                "home_region": int(homes[index]),
            }
            for index, seat in enumerate(seats)
        ],
    }


class MatchRecorder:
    """Fold every round into the schema's snapshots, emitting one per window.

    Feed it every :class:`~retail_simulator.harness.arena_match.RoundResult` in order via
    :meth:`observe`; it returns the Snapshot dict on a window boundary and ``None``
    otherwise. A window boundary is ``round % snapshot_every == 0`` — PLUS the header's
    final round, so a match whose length is not a multiple of ``snapshot_every`` still ends
    on a snapshot instead of silently dropping its tail.
    """

    def __init__(self, header: dict[str, Any], snapshot_every: int) -> None:
        if snapshot_every < 1:
            raise ValueError(f"snapshot_every must be >= 1, got {snapshot_every}")
        self._header = header
        self._snapshot_every = snapshot_every
        self._final_round = int(header["rounds"])
        n_seats = len(header["seats"])
        self._score_total = [0.0] * n_seats
        self._revenue_total = [0.0] * n_seats
        self._profit_total = [0.0] * n_seats
        self._profit_window = [0.0] * n_seats
        self._opened_window: list[list[int]] = [[] for _ in range(n_seats)]
        self._snapshots: list[dict[str, Any]] = []

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        """The snapshots emitted so far, ascending by round."""
        return self._snapshots

    def observe(self, result: RoundResult) -> dict[str, Any] | None:
        """Accumulate one round; return its Snapshot when the round closes a window."""
        for seat, seat_round in enumerate(result.seats):
            self._score_total[seat] += seat_round.reward
            self._revenue_total[seat] += seat_round.revenue
            self._profit_total[seat] += seat_round.profit
            self._profit_window[seat] += seat_round.profit
            self._opened_window[seat].extend(seat_round.opened_regions)

        closes_window = result.round % self._snapshot_every == 0
        if not closes_window and result.round != self._final_round:
            return None

        snapshot = {
            "round": result.round,
            "seats": [
                self._seat_snapshot(seat, seat_round)
                for seat, seat_round in enumerate(result.seats)
            ],
            "regions": [
                {
                    "index": region,
                    "share": [
                        round(share, _UNIT_SCALE_DECIMALS) for share in result.region_share[region]
                    ],
                    "stores": list(result.region_stores[region]),
                }
                for region in range(len(result.region_share))
            ],
        }
        self._reset_window()
        self._snapshots.append(snapshot)
        return snapshot

    def to_record(self) -> dict[str, Any]:
        """The full ``arena-match-v1`` document: the header plus every snapshot so far."""
        return {**self._header, "snapshots": self._snapshots}

    # -- internals --

    def _seat_snapshot(self, seat: int, seat_round: Any) -> dict[str, Any]:
        return {
            "seat": seat,
            "cash": round(seat_round.cash, _CASH_SCALE_DECIMALS),
            "stores": seat_round.stores,
            "market_share": round(seat_round.market_share, _UNIT_SCALE_DECIMALS),
            "score_total": round(self._score_total[seat], _CASH_SCALE_DECIMALS),
            "revenue_total": round(self._revenue_total[seat], _CASH_SCALE_DECIMALS),
            "profit_total": round(self._profit_total[seat], _CASH_SCALE_DECIMALS),
            "profit_window": round(self._profit_window[seat], _CASH_SCALE_DECIMALS),
            "employee_happiness": round(seat_round.employee_happiness, _UNIT_SCALE_DECIMALS),
            "opened_regions": sorted(self._opened_window[seat]),
            "decision": _decision(seat_round.levers),
        }

    def _reset_window(self) -> None:
        self._profit_window = [0.0] * len(self._profit_window)
        self._opened_window = [[] for _ in self._opened_window]


def _decision(levers: dict[str, float]) -> dict[str, Any]:
    """The snapshot's ``decision`` block: every logged lever, choice indices as ints."""
    decision: dict[str, Any] = {}
    for lever in LOGGED_LEVERS:
        value = levers.get(lever)
        if value is None:
            decision[lever] = None
        elif lever in _INTEGER_LEVERS:
            decision[lever] = int(value)
        else:
            decision[lever] = round(float(value), _UNIT_SCALE_DECIMALS)
    return decision
