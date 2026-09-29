"""Flat-Box <-> structured action translation for the single-Box SB3 convention.

In Phase 0.0 the action is just the 1-D ``price_index`` Box, so this is thin — but
it routes through ``core.encoding.decode_action`` for clipping/validation (the
single boundary authority) rather than reimplementing bounds, and keeps the
single-Box shape contract so the masked-expansion design extends cleanly later.

Adapter layer: numpy + core only (no gymnasium needed here).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import (
    OBS_DTYPE,
    StructuredAction,
    decode_action,
    flatten_action_space_layout,
)
from retail_simulator.core.schema import ActionSchema, default_action_schema


def to_core_action(
    action: npt.NDArray[np.float32] | list[float],
    schema: ActionSchema | None = None,
) -> npt.NDArray[np.float32]:
    """Normalize a raw agent action into the flat float32 vector the seam takes.

    The core seam (``World.step``) validates/clips via ``decode_action`` itself,
    so this only guarantees shape/dtype: a 1-D float32 array of the expected flat
    length. Raising here on a wrong-length action keeps the failure at the adapter
    boundary with a clear message (``decode_action`` would also raise).
    """
    if schema is None:
        schema = default_action_schema()
    layout = flatten_action_space_layout(schema)

    flat = np.asarray(action, dtype=OBS_DTYPE).ravel()
    if flat.shape[0] != layout.total_dim:
        raise ValueError(
            f"action has flat length {flat.shape[0]}, expected {layout.total_dim} "
            f"for schema v{schema.version}"
        )
    return flat


def decode(
    action: npt.NDArray[np.float32] | list[float],
    schema: ActionSchema | None = None,
) -> StructuredAction:
    """Decode a raw agent action into validated, clipped structured levers.

    Thin pass-through to ``core.encoding.decode_action`` (the single clipping /
    validation authority) so callers that want the structured view (e.g. metrics)
    do not import core directly.
    """
    if schema is None:
        schema = default_action_schema()
    return decode_action(to_core_action(action, schema), schema)


def mask_expansion_logits(
    action: npt.NDArray[np.float32] | list[float] | tuple[float, ...],
    expansion_mask: npt.NDArray[np.bool_],
    schema: ActionSchema | None = None,
) -> npt.NDArray[np.float32]:
    """ """
    if schema is None:
        schema = default_action_schema()
    layout = flatten_action_space_layout(schema)
    expansion_slot = layout.slot("expansion")

    mask = np.asarray(expansion_mask)
    if mask.dtype != np.bool_:
        raise ValueError(
            f"expansion_mask has dtype {mask.dtype}, expected bool — a non-bool "
            "array is silently truthy-cast instead of validated, which can "
            "misclassify a legal slot (e.g. a 0.0 entry, including hold) as illegal"
        )
    if mask.shape != (expansion_slot.width,):
        raise ValueError(
            f"expansion_mask has shape {mask.shape}, expected ({expansion_slot.width},) "
            f"to match the schema's expansion lever block width"
        )

    flat = np.array(action, dtype=OBS_DTYPE).ravel()  # a COPY — never mutate the caller's array
    if flat.shape[0] != layout.total_dim:
        raise ValueError(
            f"action has flat length {flat.shape[0]}, expected {layout.total_dim} "
            f"for schema v{schema.version}"
        )

    block = flat[expansion_slot.offset : expansion_slot.offset + expansion_slot.width]
    block[~mask] = -np.inf
    return flat
