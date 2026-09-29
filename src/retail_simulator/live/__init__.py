"""This is the project's THIRD transport over the single ``World.step`` seam, sibling
to ``core/`` (the pure world model), ``envs/`` (the Gymnasium + PettingZoo
adapters), and ``harness/`` (the training tier). The live wrapper reimplements
no economics — the seam is called exactly once per tick by the Slice B asyncio
server, mirroring the ``envs/parallel_env.py`` relationship to ``core/``.

Slice A (this package, as shipped now) is SYNC and PURE-PYTHON: the wire
protocol dataclasses + JSON dispatch + the registry-default action vector +
the offline replay validator + the client skeleton. Slice B adds the asyncio
server loop, the action-log writer, the ``websockets`` connection layer, and
wires :class:`LiveClient` to a real socket.

**Lazy-import contract.** The optional ``websockets`` dependency (the
``[serve]`` extra per F-DEP=A) is HEAVY and OPTIONAL; it is imported ONLY
inside Slice B's ``server.py`` / ``client.py`` connection methods at the
point of first use — NEVER at package-import time. Mirrors the Phase 1.5
lazy-SB3 contract; verified by a smoke test in the Slice A test suite.
``import retail_simulator.live`` must remain cheap and stdlib + numpy only.
"""

from __future__ import annotations

from retail_simulator.live.client import LiveClient
from retail_simulator.live.play import play_one_game
from retail_simulator.live.protocol import (
    PROTOCOL_VERSION,
    REGISTRY_DEFAULT_ACTION,
    ActionMessage,
    ErrorMessage,
    FinalScores,
    Hello,
    ObservationMessage,
    SeatAssigned,
    from_json_any,
)
from retail_simulator.live.replay import (
    ActionLogRecord,
    ReplayManifest,
    ReplayResult,
    read_action_log,
    read_manifest,
    replay_game_dir,
)
from retail_simulator.live.replay import replay as run_replay
from retail_simulator.live.series import SeriesGameResult, SeriesResult, run_series
from retail_simulator.live.server import ServeConfig, ServeResult, serve_game

# Avoid shadowing the `replay` submodule on attribute access: re-export the
# function as ``run_replay`` (Slice B M1 cleanup per senior review). The
# submodule remains reachable via ``import retail_simulator.live.replay`` and
# ``retail_simulator.live.replay`` attribute access; CLI callers and Slice B's
# wrapper use ``run_replay``.


__all__ = [
    "ActionLogRecord",
    "ActionMessage",
    "ErrorMessage",
    "FinalScores",
    "Hello",
    "LiveClient",
    "ObservationMessage",
    "PROTOCOL_VERSION",
    "REGISTRY_DEFAULT_ACTION",
    "play_one_game",
    "ReplayManifest",
    "ReplayResult",
    "SeatAssigned",
    "SeriesGameResult",
    "SeriesResult",
    "ServeConfig",
    "ServeResult",
    "from_json_any",
    "run_series",
    "read_action_log",
    "read_manifest",
    "replay_game_dir",
    "run_replay",
    "serve_game",
]
