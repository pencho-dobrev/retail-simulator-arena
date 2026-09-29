"""``load_scenario(path)`` is the operator entry point the CLI (``validate``,
``competition run``) consumes: it parses a scenario YAML, validates it at the boundary
(pydantic, collecting ALL errors), and returns the framework-free
:class:`~retail_simulator.core.config.CoreConfig` — the seat plan is carried on
``CoreConfig.seats`` (so ``world.reset`` consumes the single returned object, matching
the as-shipped reset path).

On invalid input the loader raises :class:`ScenarioValidationError` carrying a clean,
aggregated message (never a bare pydantic traceback) so the CLI can print it and exit
non-zero with numeric detail (DX: "bad config: collect all errors").
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from retail_simulator.core.config import CoreConfig
from retail_simulator.scenarios.models import ScenarioConfig


class ScenarioValidationError(ValueError):
    """A scenario YAML failed boundary validation — carries the aggregated message.

    Raised by :func:`load_scenario` instead of letting a raw pydantic
    ``ValidationError`` propagate, so callers (the CLI) get one operator-facing string
    listing every problem (DX) rather than a framework traceback.
    """


def _format_validation_error(path: str | Path, exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` as a single aggregated operator message.

    Lists every collected problem (one per line) with its field location, so a YAML
    with several issues reports them all at once.
    """
    lines = [f"invalid scenario config at {path}:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_scenario(path: str | Path) -> CoreConfig:
    """Load a scenario YAML and return its validated ``CoreConfig`` (seat plan on it).

    Parses + validates via :meth:`ScenarioConfig.from_yaml` (pydantic collects ALL
    errors), then maps to ``CoreConfig`` via :meth:`ScenarioConfig.to_core_config`. A
    validation failure is re-raised as :class:`ScenarioValidationError` with a clean
    aggregated message; a missing file surfaces as ``FileNotFoundError`` (the operator's
    path is wrong — distinct from a malformed-but-present config).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"scenario file not found: {path}")
    try:
        scenario = ScenarioConfig.from_yaml(path)
    except ValidationError as exc:
        raise ScenarioValidationError(_format_validation_error(path, exc)) from exc
    return scenario.to_core_config()


__all__ = ["ScenarioValidationError", "load_scenario"]
