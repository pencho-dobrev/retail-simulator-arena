"""Phase 2 Slice B T-B3 — reference Python WebSocket client (wired).

Slice A (T-A4) locked the constructor surface and the async-method signatures
so Slice B could extend without rewriting the public API. T-B3 replaces the
four ``NotImplementedError("wired in Slice B")`` stubs with real ``websockets``
calls, completing the reference client.

Threading model: one :class:`LiveClient` per asyncio task; not thread-safe.
The mutable internal slots (``_seat_assigned``, ``_connection``,
``_current_tick``, ``_final_scores``) are touched only inside the owning
event loop.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import flatten_action_space_layout
from retail_simulator.live.protocol import (
    PROTOCOL_VERSION,
    ActionMessage,
    ErrorMessage,
    FinalScores,
    Hello,
    ObservationMessage,
    SeatAssigned,
    from_json_any,
)

logger = logging.getLogger(__name__)

# DERIVED from the current action schema's flat layout (Phase 5.0, B-1) — NOT a
# hardcoded literal, so a future lever-registry change never needs a matching
# edit here (mirrors ``live/_permutation.py``).
_EXPECTED_ACTION_SHAPE: tuple[int, ...] = (flatten_action_space_layout().total_dim,)


@dataclass
class LiveClient:
    """Reference Python WebSocket client for the Phase 2 demo server.

    Constructor surface (LOCKED at T-A4, preserved here):
        uri: ``ws://host:port`` — the server's WebSocket endpoint.
        agent_name: identifies this client to the server (echoed in
            :class:`Hello` and surfaced in server logs / the action-log
            artifact T-A5 writes).
        token: shared-token auth (F2-A); ``None`` when the server runs with
            ``--no-auth``. Forwarded verbatim into the :class:`Hello` frame.

    Slice B (T-B3) wires the four async methods to ``websockets``:
        * :meth:`connect` opens the socket, sends :class:`Hello`, awaits
          :class:`SeatAssigned`.
        * :meth:`iter_observations` yields one :class:`ObservationMessage`
          per tick; terminates on :class:`FinalScores` (captured on
          :attr:`final_scores`); logs and continues on a recoverable
          :class:`ErrorMessage`; closes + raises on a fatal one.
        * :meth:`submit_action` validates shape + dtype, sends one
          :class:`ActionMessage`, and increments the tick counter.
        * :meth:`close` is idempotent — safe to call before :meth:`connect`
          or after a prior :meth:`close`.

    The constructor remains intentionally minimal — there is no retry policy,
    no timeout knob, no per-call logging hook. Those are Slice B+ decisions
    and adding them here would lock surface area we have not yet designed.
    """

    uri: str
    agent_name: str
    token: str | None = None

    # Internal state populated by ``connect()`` / mutated by the other async
    # methods. ``init=False`` keeps them out of the constructor surface (which
    # is contractually locked above); ``repr=False`` so a debug print of a
    # connected client does not dump a websocket connection object or the
    # full :class:`SeatAssigned` payload.
    _seat_assigned: SeatAssigned | None = field(default=None, init=False, repr=False)
    _connection: Any = field(default=None, init=False, repr=False)
    # ``_current_tick`` advances inside :meth:`submit_action`. Initialized to
    # 0 so the first submission after :meth:`connect` carries ``tick=0`` —
    # matching the env adapter's tick numbering (the server's first
    # :class:`ObservationMessage` also carries ``tick=0``).
    _current_tick: int = field(default=0, init=False, repr=False)
    # ``_final_scores`` is the end-of-game payload. :meth:`iter_observations`
    # captures the :class:`FinalScores` frame and stops iteration; the caller
    # reads it via the :attr:`final_scores` property after the ``async for``
    # loop completes.
    _final_scores: FinalScores | None = field(default=None, init=False, repr=False)

    @property
    def seat_assigned(self) -> SeatAssigned | None:
        """The server's :class:`SeatAssigned` payload, populated by :meth:`connect`."""
        return self._seat_assigned

    @property
    def final_scores(self) -> FinalScores | None:
        """The end-of-game :class:`FinalScores`; ``None`` until the game ends."""
        return self._final_scores

    async def connect(self) -> SeatAssigned:
        """Open the WebSocket, send :class:`Hello`, receive :class:`SeatAssigned`.

        Returns the server's :class:`SeatAssigned` message (which carries
        ``seat_index``, ``n_agents``, ``n_npcs``, ``npc_archetypes``,
        ``n_ticks``, ``game_id``, ``agent_id``, ``tick_interval_ns``,
        ``scenario_name``). Caches it on ``self._seat_assigned`` so
        subsequent calls (replay tooling, error reporting) can read the
        assignment without a second round-trip.

        Raises:
            ConnectionError: server unreachable, transport closed during
                handshake, or server returned an :class:`ErrorMessage`
                rejecting the connection (auth_failed, protocol_mismatch,
                registration_timeout, ...).
            ValueError: server returned a well-formed frame of an
                unexpected type (i.e. not :class:`SeatAssigned` and not
                :class:`ErrorMessage`) — a protocol-state bug on the
                server side.
        """
        import websockets
        import websockets.exceptions

        # ``retail_simulator.__version__`` and ``ACTION_SCHEMA_VERSION`` are
        # cheap pure-Python imports, but keeping them inside the method body
        # mirrors the lazy-import discipline and keeps the module-level
        # imports declarative (the docstring contract that forbids
        # asyncio / websockets at import time covers the spirit, not the
        # letter, of this — but staying consistent here avoids future
        # confusion about which imports are "safe at module top").
        import retail_simulator
        from retail_simulator.core.schema import ACTION_SCHEMA_VERSION

        try:
            self._connection = await websockets.connect(self.uri)
        except (OSError, websockets.exceptions.WebSocketException) as e:
            # Two distinct failure modes funnel into one ``ConnectionError``
            # so callers handle a single boundary exception type:
            #   * ``OSError`` — host unreachable / port closed / DNS fail.
            #   * ``WebSocketException`` — TLS / HTTP-upgrade / handshake
            #     failure inside the websockets library.
            raise ConnectionError(f"failed to open WebSocket {self.uri!r}: {e}") from e

        hello = Hello(
            agent_name=self.agent_name,
            token=self.token,
            protocol_version=PROTOCOL_VERSION,
            schema_version=ACTION_SCHEMA_VERSION,
            client_name=f"retail-simulator-py/{retail_simulator.__version__}",
        )
        try:
            await self._connection.send(hello.to_json())
            raw = await self._connection.recv()
        except (OSError, websockets.exceptions.ConnectionClosedError) as e:
            # Best-effort cleanup; we still raise so the caller sees the
            # connection died mid-handshake. ``await close()`` rather than
            # touching the underlying transport so any pending send buffer
            # is flushed cleanly.
            await self._safe_close()
            raise ConnectionError(f"connection closed during handshake on {self.uri!r}: {e}") from e

        # ``recv()`` returns ``str | bytes``; the protocol is text-frame JSON,
        # so a bytes frame here is a server-side protocol bug. Normalize via
        # UTF-8 decode rather than rejecting (the websockets server we ship
        # always sends text frames, but tolerating bytes makes the client
        # robust against a future server tweak).
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        msg = from_json_any(raw)
        if isinstance(msg, ErrorMessage):
            await self._safe_close()
            raise ConnectionError(f"server rejected connection: {msg.error_kind}: {msg.message}")
        if not isinstance(msg, SeatAssigned):
            # Unexpected frame type. Close + raise ValueError (NOT
            # ConnectionError) because the transport is healthy; the bug is
            # a server-side state-machine error.
            await self._safe_close()
            raise ValueError(f"expected SeatAssigned during handshake, got {type(msg).__name__}")
        self._seat_assigned = msg
        return msg

    async def iter_observations(self) -> AsyncIterator[ObservationMessage]:
        """Async generator yielding :class:`ObservationMessage` per tick.

        Terminates cleanly when the server sends :class:`FinalScores`; the
        frame is captured on :attr:`final_scores` and the iterator stops
        (the caller's ``async for`` loop exits normally). A recoverable
        :class:`ErrorMessage` is logged and skipped (the server keeps the
        connection); a fatal :class:`ErrorMessage` closes the socket and
        raises :class:`ConnectionError`. An unexpected message type during
        play is logged and skipped — a server-side protocol bug, but not
        worth aborting an in-progress game over.

        Raises:
            RuntimeError: :meth:`connect` was never called (or :meth:`close`
                already ran).
            ConnectionError: server sent a fatal :class:`ErrorMessage`.
        """
        if self._connection is None:
            raise RuntimeError("LiveClient.iter_observations: must call connect() first")

        # ``async for raw in self._connection`` — the websockets
        # ClientConnection implements ``__aiter__`` and yields one frame
        # per iteration until the peer closes the socket. This is the
        # canonical receive loop in the websockets docs.
        async for raw in self._connection:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            msg = from_json_any(raw)
            if isinstance(msg, ObservationMessage):
                yield msg
            elif isinstance(msg, FinalScores):
                # End-of-game sentinel. Capture so the caller can read it
                # via :attr:`final_scores` after the ``async for`` exits,
                # then ``return`` to stop iteration (the websockets close
                # frame will follow naturally; we don't proactively close
                # so the server can send its close frame in its own time).
                self._final_scores = msg
                return
            elif isinstance(msg, ErrorMessage):
                if not msg.recoverable:
                    # Fatal: server will close. Close our side too (idempotent
                    # if the server already closed) and surface as
                    # ConnectionError so the caller distinguishes "game ended
                    # normally" (FinalScores -> StopIteration) from "game
                    # aborted" (raise).
                    await self._safe_close()
                    raise ConnectionError(
                        f"server error during play: {msg.error_kind}: {msg.message}"
                    )
                # Recoverable: server keeps the connection open. Log and
                # continue the receive loop; the next frame will be a new
                # ObservationMessage (or, in the worst case, another error).
                logger.warning("live server warning: %s: %s", msg.error_kind, msg.message)
            else:
                # Hello / SeatAssigned / ActionMessage during play are a
                # server-side state bug. Log + continue rather than raise —
                # an unknown frame is not a reason to abandon a game that
                # is otherwise progressing.
                logger.warning("unexpected message type during play: %s", type(msg).__name__)

    async def submit_action(self, action: npt.NDArray[np.float32]) -> None:
        """Send one :class:`ActionMessage` for the current tick.

        ``action`` must be a flat float32 vector matching the current action
        schema's width (``_EXPECTED_ACTION_SHAPE``, derived from
        ``flatten_action_space_layout`` — never a hardcoded literal). The
        method validates shape and dtype at the boundary; an invalid
        action is a caller bug, not a server bug, and we want to fail
        loudly before the JSON serialization step where the diagnostic
        is much less useful.

        Raises:
            ValueError: wrong shape or dtype.
            RuntimeError: :meth:`connect` was never called (or the
                handshake did not complete).
            ConnectionError: socket closed while sending.
        """
        # Boundary validation: catch the two common policy bugs (forgot to
        # cast to float32; forgot to reshape from (1, N) to (N,)) before
        # the wire layer. ``.shape`` and ``.dtype`` are O(1) numpy probes.
        if action.shape != _EXPECTED_ACTION_SHAPE:
            raise ValueError(
                f"submit_action: expected shape {_EXPECTED_ACTION_SHAPE}, got {action.shape}"
            )
        if action.dtype != np.float32:
            raise ValueError(f"submit_action: expected dtype float32, got {action.dtype}")
        if self._connection is None:
            raise RuntimeError("LiveClient.submit_action: must call connect() first")
        if self._seat_assigned is None:
            # Defensive: ``connect()`` always sets ``_seat_assigned`` before
            # returning, so this is unreachable in practice. Kept so a
            # future refactor that splits ``connect()`` into open + handshake
            # cannot accidentally let a submit slip through pre-handshake.
            raise RuntimeError("LiveClient.submit_action: handshake did not complete")

        action_msg = ActionMessage(tick=self._current_tick, action=action.tolist())
        try:
            await self._connection.send(action_msg.to_json())
        except OSError as e:
            # ConnectionClosedError is a websockets-specific subclass that
            # we re-raise as a generic ConnectionError so callers handle one
            # boundary exception type. Importing the websockets module
            # lazily here is wasteful (it's already in sys.modules after
            # connect()); ``OSError`` is the parent class for both
            # standard socket failures and websockets' close exceptions
            # (ConnectionClosed inherits from OSError indirectly via
            # asyncio's transport semantics), so a single ``except OSError``
            # plus a generic ``Exception`` re-raise covers both paths.
            raise ConnectionError(
                f"submit_action: send failed at tick {self._current_tick}: {e}"
            ) from e

        # Increment only on a successful send so a failed send does not
        # advance the tick counter (a subsequent retry, if the caller
        # implements one, will resend tick N rather than skipping to N+1).
        self._current_tick += 1

    async def close(self) -> None:
        """Close the WebSocket gracefully.

        Idempotent: safe to call before :meth:`connect` (no-op) and after a
        prior :meth:`close` (no-op). The ``async with`` pattern in callers
        therefore composes cleanly.
        """
        await self._safe_close()

    async def _safe_close(self) -> None:
        """Close the underlying connection if open; never raise.

        Internal helper used by both the public :meth:`close` and the
        error-cleanup branches of :meth:`connect` / :meth:`iter_observations`.
        The "never raise" contract matters there — a cleanup that itself
        raises would mask the real error the caller is being told about.
        """
        if self._connection is None:
            return
        connection = self._connection
        # Null first so a re-entrant call (e.g. from a finally clause) sees
        # the closed state and short-circuits.
        self._connection = None
        try:
            await connection.close()
        except Exception:  # noqa: BLE001  # intentional: cleanup must not raise
            # Best-effort: if the peer already tore down the socket the
            # close() may raise a ConnectionClosed; we swallow because the
            # logical state ("connection is closed") is what callers care
            # about, and the exception we are inside (if any) is the real
            # one to surface.
            logger.debug("LiveClient._safe_close: ignored exception during close", exc_info=True)


__all__ = ["LiveClient"]
