"""retail-simulator: a deterministic multi-agent RL environment for retail.

Public surface (Gym/PettingZoo-idiomatic; everything under ``core/`` is private):

* :class:`RetailEnv` — the Gymnasium single-agent environment.
* :class:`RetailParallelEnv` — the PettingZoo Parallel multi-agent environment.
* :class:`ScenarioConfig` — the operator-facing YAML scenario config (``scenarios``).
* ``__version__`` — package version.

Importing this package registers the Gymnasium id ``RetailSim-v0`` (so
``gym.make("RetailSim-v0")`` works without importing ``retail_simulator.envs``
directly) *when gymnasium is installed*. To keep a bare ``import retail_simulator``
import-light, the gymnasium/pettingzoo-dependent adapter layer is imported lazily:
``RetailEnv``/``RetailParallelEnv`` are resolved on first access via module
``__getattr__``, and registration degrades to a no-op if the optional ``[rl]`` extra
(gymnasium) is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = ["RetailEnv", "RetailParallelEnv", "ScenarioConfig", "__version__"]

if TYPE_CHECKING:  # import only for type-checkers; never at runtime import time
    from retail_simulator.envs import RetailEnv, RetailParallelEnv
    from retail_simulator.scenarios import ScenarioConfig


def register_environments() -> bool:
    """Register ``RetailSim-v0`` with Gymnasium if the adapter layer is available.

    Returns ``True`` if registration ran (gymnasium installed), ``False`` if the
    optional ``[rl]`` extra is absent — in which case a bare ``import
    retail_simulator`` still succeeds (numpy-only core). Importing
    ``retail_simulator.envs`` performs the actual registration.
    """
    try:
        import retail_simulator.envs  # noqa: F401  (import for its registration side effect)
    except ImportError:
        # gymnasium not installed (bare core install): nothing to register.
        return False
    return True


def __getattr__(name: str) -> Any:
    """Lazily expose ``RetailEnv`` without importing gymnasium at module load.

    Keeps ``import retail_simulator`` cheap and dependency-light; the adapter
    (and thus gymnasium) is only imported when ``retail_simulator.RetailEnv`` is
    actually accessed.
    """
    if name == "RetailEnv":
        from retail_simulator.envs import RetailEnv

        return RetailEnv
    if name == "RetailParallelEnv":
        from retail_simulator.envs import RetailParallelEnv

        return RetailParallelEnv
    if name == "ScenarioConfig":
        from retail_simulator.scenarios import ScenarioConfig

        return ScenarioConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Best-effort registration on import so `import retail_simulator; gym.make(...)`
# works per the DX contract. Silent no-op when gymnasium is absent.
register_environments()
