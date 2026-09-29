"""Each ``make_*`` returns a ``predict(obs, *, episode_start=bool) -> np.ndarray(44,)
float32`` closure matching the F-MATERIALIZE-API=A contract the head-to-head primitive
drives (:func:`retail_simulator.harness.league._head_to_head_episode_scores` passes
``episode_start`` and the action is fed straight to ``World.step`` via the parallel
env). The action is built ONCE per ``make_*`` call (constructed via
:func:`retail_simulator.core.encoding.encode_action` so it respects the registry layout
+ the discrete-masked expansion/automation one-hot encoding — the encoder derives the
flat width from the schema, so these archetypes never hardcode a vector length) and the
SAME read-only array is returned on every call — deterministic across calls and ticks.

Layer note: pure ``core`` reads only (encoding/schema) — no envs, no RL framework, no
network. Safe under the ``harness`` import-linter contract.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import StructuredAction, decode_action, encode_action
from retail_simulator.core.schema import N_AUTOMATION_TIERS, N_REGIONS

# The predict closure signature the head-to-head primitive drives: an obs plus an
# optional ``episode_start`` kwarg (absorbed-and-ignored here — these archetypes are
# OPEN-LOOP), returning a flat float32 action (44-D as of schema v12; the width is
# derived from the schema, never hardcoded here).
ArchetypePredict = Callable[..., npt.NDArray[np.float32]]

AGGRESSOR: str = "aggressor"
LEAN: str = "lean"
PATIENT: str = "patient"


def _structured_action(
    *,
    price: float,
    marketing: float,
    assortment: float,
    promotion: float,
    expansion_choice: int,
    research: float,
    service_level: float,
    automation_tier: int,
    loyalty_spend: float,
    wage_spend: float = 0.0,
    warehouse_invest: float = 0.0,
) -> StructuredAction:
    """ """
    if not (0 <= expansion_choice < N_REGIONS):
        raise ValueError(f"expansion_choice must be in [0, {N_REGIONS}); got {expansion_choice}")
    if not (0 <= automation_tier < N_AUTOMATION_TIERS):
        raise ValueError(
            f"automation_tier must be in [0, {N_AUTOMATION_TIERS}); got {automation_tier}"
        )
    return StructuredAction(
        levers={
            "price_index": price,
            "marketing": marketing,
            "assortment": assortment,
            "promotion": promotion,
            "expansion": float(expansion_choice),
            "research": research,
            "service_level": service_level,
            "automation": float(automation_tier),
            "loyalty_spend": loyalty_spend,
            "wage_spend": wage_spend,
            "warehouse_invest": warehouse_invest,
        }
    )


def _fixed_posture(structured: StructuredAction) -> ArchetypePredict:
    """Freeze ``structured`` into a deterministic open-loop predict closure.

    Encodes ONCE, marks the array read-only (so a downstream ``+=`` in the seam can
    never mutate the shared posture — the env makes its own working copy), and returns
    the SAME array on every call regardless of obs / ``episode_start``.
    """
    action = encode_action(structured)
    action.setflags(write=False)

    def predict(_obs: object, *, episode_start: bool = False) -> npt.NDArray[np.float32]:
        del _obs, episode_start  # open-loop: posture is fixed, ignores obs + reset flag
        return action

    return predict


def make_aggressor() -> ArchetypePredict:
    """ """
    return _fixed_posture(
        _structured_action(
            price=0.75,  # aggressive low price to grab share
            marketing=0.9,  # heavy marketing push
            assortment=0.9,  # broad assortment
            promotion=0.7,  # heavy promotion
            expansion_choice=1,
            research=0.0,  # the spike archetypes are open-loop; no perception spend
            service_level=1.0,  # registry default / baseline
            automation_tier=N_AUTOMATION_TIERS - 1,  # top tier — spend cash early
            loyalty_spend=0.8,  # spend on loyalty early
        )
    )


def make_lean() -> ArchetypePredict:
    """ """
    return _fixed_posture(
        _structured_action(
            price=1.6,  # high price to defend margin
            marketing=0.05,  # minimal marketing
            assortment=0.1,  # narrow assortment
            promotion=0.0,  # no promotion
            expansion_choice=0,  # NO expansion (stay lean)
            research=0.0,
            service_level=1.0,  # registry default / baseline
            automation_tier=0,  # no automation capex
            loyalty_spend=0.05,  # minimal spend
        )
    )


def make_patient() -> ArchetypePredict:
    """ """
    return _fixed_posture(
        _structured_action(
            price=1.05,  # moderate price near baseline
            marketing=0.4,  # moderate marketing
            assortment=0.45,  # moderate assortment
            promotion=0.2,  # light promotion
            expansion_choice=1,  # measured expansion into region 1
            research=0.0,
            service_level=1.0,  # registry default / baseline
            automation_tier=1,  # mid automation tier (measured)
            loyalty_spend=0.4,  # balanced loyalty spend
        )
    )


def default_archetypes() -> dict[str, ArchetypePredict]:
    """ """
    return {
        LEAN: make_lean(),
        AGGRESSOR: make_aggressor(),
        PATIENT: make_patient(),
    }


def calibrated_archetypes() -> dict[str, ArchetypePredict]:
    """ """
    aggressor = _fixed_posture(
        _structured_action(
            price=0.8,
            marketing=0.9,
            assortment=0.9,
            promotion=0.7,
            expansion_choice=1,
            research=0.0,
            service_level=0.5,  # squeezer — accepts a lower fill rate to cut COGS
            automation_tier=2,
            loyalty_spend=0.6,
        )
    )
    lean = _fixed_posture(
        _structured_action(
            price=1.5,
            marketing=0.05,
            assortment=0.1,
            promotion=0.0,
            expansion_choice=0,
            research=0.0,
            service_level=0.2,  # deep squeezer — cheapest supply, lowest reliability
            automation_tier=0,
            loyalty_spend=0.05,
        )
    )
    patient = _fixed_posture(
        _structured_action(
            price=1.0,
            marketing=0.4,
            assortment=0.45,
            promotion=0.2,
            expansion_choice=1,
            research=0.0,
            service_level=1.0,  # partner — full fill rate, at full COGS
            automation_tier=1,
            loyalty_spend=0.4,
        )
    )
    return {
        LEAN: lean,
        AGGRESSOR: aggressor,
        PATIENT: patient,
    }


def _assert_decodable(action: npt.NDArray[np.float32]) -> StructuredAction:
    """Decode a posture action — the in-module guard that a posture is valid (test aid).

    Reused by the test suite to confirm every archetype returns a flat action
    ``decode_action`` accepts (correct length, in-bounds continuous values, valid
    discrete one-hot blocks). Exposed here so the round-trip lives next to the encoder.
    """
    return decode_action(action)
