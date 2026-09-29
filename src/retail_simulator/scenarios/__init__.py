"""The operator-facing layer. ``ScenarioConfig`` (pydantic, boundary-validated) maps a
small set of operator knobs onto the framework-free ``CoreConfig`` + a seat plan;
``load_scenario(path)`` parses + validates a YAML and returns the ``CoreConfig`` the
world consumes.
"""

from __future__ import annotations

from retail_simulator.scenarios.loader import ScenarioValidationError, load_scenario
from retail_simulator.scenarios.models import ScenarioConfig

__all__ = ["ScenarioConfig", "ScenarioValidationError", "load_scenario"]
