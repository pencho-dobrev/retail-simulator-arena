"""Phase 2.1 Slice 4 — the reusable play-one-game coroutine.

This module owns the single client play loop that Slice 3 inlined in the
``live play`` CLI command: connect a :class:`LiveClient`, submit the passive
default-policy action every observed tick, capture the end-of-game
:class:`FinalScores`, and close. BOTH callers now share it:

  * ``live play`` (``cli.py``) — one operator client against a running game;
  * the Slice 4 ``--self-play`` ``serve_fn`` — ``cfg.n_agents`` internal clients
    driven concurrently with their own in-process ``serve_game`` server, so a
    multi-game ``run-series`` runs hands-free with no external clients.

DRY: the play loop has exactly ONE owner (this coroutine). ``cli.py`` keeps the
operator-facing concerns (printing the seat assignment + final scores, exit
codes); :func:`play_one_game` only drives the client and RETURNS the scores.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from retail_simulator.live.client import LiveClient
from retail_simulator.live.protocol import REGISTRY_DEFAULT_ACTION, FinalScores, SeatAssigned


async def play_one_game(
    uri: str,
    agent_name: str,
    token: str | None,
    *,
    policy: str = "default",
    on_seat_assigned: Callable[[SeatAssigned], None] | None = None,
) -> FinalScores | None:
    """Connect one :class:`LiveClient`, play a whole game, return its scores.

    Constructs a :class:`LiveClient`, :meth:`~LiveClient.connect`\\ s, then plays
    the ``policy`` action each observed tick until the server broadcasts
    FinalScores (which terminates :meth:`~LiveClient.iter_observations`), and
    returns :attr:`LiveClient.final_scores` (``None`` if the socket closed
    without an end-of-game frame — an aborted game). The client is ALWAYS
    closed in a ``finally`` so a caller error does not leak the socket.

    The ``default`` policy is obs-independent: a fresh WRITABLE float32 copy
    (shape matching the current action schema's flat width) of the read-only
    :data:`REGISTRY_DEFAULT_ACTION` singleton,
    matching :meth:`~LiveClient.submit_action`'s shape/dtype contract (the
    passive baseline; the env outcome is still seed-dependent, so across a
    seed-varying series it yields cross-seed variance data). ``policy`` values
    other than ``"default"`` raise :class:`ValueError` — a stochastic / learned
    policy is deferred (same reason as Slice 3).

    Note that this returns the scores rather than printing them: the operator
    presentation (seat assignment, JSON scores, exit codes) lives in the
    ``live play`` CLI command, while the self-play ``serve_fn`` discards the
    return value (it reads the SERVER's :class:`ServeResult`, not the client's).
    The optional ``on_seat_assigned`` callback fires once, right after the
    server assigns the seat and BEFORE the first action — the seam ``live play``
    uses to print the human-readable seat assignment at the original timing
    without this coroutine taking on presentation concerns.
    """
    if policy != "default":
        raise ValueError(f"unknown policy {policy!r}; only 'default' is supported in this slice")

    client = LiveClient(uri=uri, agent_name=agent_name, token=token)
    try:
        seat = await client.connect()
        if on_seat_assigned is not None:
            on_seat_assigned(seat)
        async for _obs in client.iter_observations():
            await client.submit_action(_default_action())
        return client.final_scores
    finally:
        await client.close()


def _default_action() -> np.ndarray[Any, Any]:
    """A fresh WRITABLE float32 copy of the registry-default action.

    Shape matches the current action schema's flat width (derived from
    ``flatten_action_space_layout``, never a hardcoded literal).

    The default policy submits a fresh copy each tick rather than the shared
    read-only singleton: ``submit_action`` does not mutate the array, but a
    writable copy keeps the action decoupled from the global and matches the
    contract Slice 3's ``live play`` established.
    """
    return np.array(REGISTRY_DEFAULT_ACTION, dtype=np.float32, copy=True)
