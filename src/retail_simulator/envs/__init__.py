"""RL adapter layer: Gymnasium Spaces, action translation, and ``RetailEnv``.

Importing this package pulls in ``gymnasium`` (the adapter framework) and
registers ``RetailSim-v0`` so ``gym.make("RetailSim-v0")`` works. The top-level
``retail_simulator`` package imports this lazily (inside ``register_environments``)
so a bare ``import retail_simulator`` does not hard-require gymnasium unless the
env surface is actually used.
"""

from __future__ import annotations

from retail_simulator.envs.gym_env import ENV_ID, RetailEnv, register_env
from retail_simulator.envs.parallel_env import RetailParallelEnv

# Register on import of the adapter package: any path that reaches envs/ (a direct
# import, or `import retail_simulator` which calls register_environments) makes the
# gym id available.
register_env()

__all__ = ["ENV_ID", "RetailEnv", "RetailParallelEnv", "register_env"]
