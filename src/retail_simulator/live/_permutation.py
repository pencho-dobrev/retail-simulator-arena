"""Phase 2 Slice A — the fixed agent-index permutation helper (the AC-3 invariant).

Module is package-internal (leading underscore); Slice B's ``server.py``
imports it directly rather than going through ``live/__init__.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import flatten_action_space_layout

# DERIVED from the current action schema's flat layout (Phase 5.0, B-1) — NOT a
# hardcoded literal, so a future lever-registry change (a new schema version)
# never needs a matching edit here. Every flat action vector must have this
# width + dtype float32. The helper validates at the boundary so a malformed
# entry surfaces here — not inside ``env.step`` with an opaque shape mismatch
# ten frames down.
_EXPECTED_ACTION_SHAPE: tuple[int, ...] = (flatten_action_space_layout().total_dim,)
_EXPECTED_ACTION_DTYPE: np.dtype[np.float32] = np.dtype(np.float32)


def _validate_action(seat: int, action: npt.NDArray[np.float32]) -> None:
    """Validate an action's shape + dtype; raise ``ValueError`` with the seat index."""
    if action.shape != _EXPECTED_ACTION_SHAPE:
        raise ValueError(
            f"seat {seat}: action has shape {action.shape}, expected {_EXPECTED_ACTION_SHAPE}"
        )
    if action.dtype != _EXPECTED_ACTION_DTYPE:
        raise ValueError(
            f"seat {seat}: action has dtype {action.dtype}, expected {_EXPECTED_ACTION_DTYPE}"
        )


def permute_actions_by_seat(
    arrived: dict[int, npt.NDArray[np.float32]],
    last_valid: dict[int, npt.NDArray[np.float32]],
    n_seats: int,
    registry_default_action: npt.NDArray[np.float32],
) -> tuple[dict[str, npt.NDArray[np.float32]], list[int]]:
    """ """
    if n_seats < 1:
        raise ValueError(f"n_seats must be >= 1, got {n_seats}")

    # Boundary validation: every action ndarray that will land in the joint
    # action (whether from ``arrived`` or ``last_valid``) must satisfy the
    # current action schema's shape/dtype contract. We do NOT validate the
    # registry default — the
    # T-A1 constant is built via ``encode_action`` and is shape/dtype-correct
    # by construction; revalidating would be cycle waste on every tick.
    for seat, action in arrived.items():
        _validate_action(seat, action)
    for seat, action in last_valid.items():
        _validate_action(seat, action)

    joint_action: dict[str, npt.NDArray[np.float32]] = {}
    missing_seats: list[int] = []

    for seat in range(n_seats):
        agent_id = f"retailer_{seat}"
        if seat in arrived:
            joint_action[agent_id] = arrived[seat]
        elif seat in last_valid:
            joint_action[agent_id] = last_valid[seat]
            missing_seats.append(seat)
        else:
            # Tick-0 miss: the registry default is shared + read-only; copy
            # so a downstream writer cannot corrupt the singleton through
            # aliasing across ticks.
            joint_action[agent_id] = registry_default_action.copy()
            missing_seats.append(seat)

    return joint_action, missing_seats


__all__ = ["permute_actions_by_seat"]
