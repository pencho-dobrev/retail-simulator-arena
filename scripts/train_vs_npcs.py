"""Example trainer: one learning seat against the NPC field, with your own loop.

``scripts/arena_train_player.py`` is two-seat league self-play at fixed home regions. This
script is the other common baseline: ONE learning seat in the same five-seat field the
self-check plays (``cherry-picker``, ``fortress``, ``fast-follower``, ``sprawler``), written as a
plain Gymnasium env so any Stable-Baselines3 algorithm can train on it. It is a starting point
to read and edit, not a tuned entry.

Choices worth knowing, and why they are made:

* Seating. The learner sits in seat 0 with expansion masking ON, like a ``--player`` in
  ``arena_competition.py``. The NPC seats are driven by ``geo_archetypes`` exactly as
  ``ArenaMatch`` drives them, and are unmasked.
* Homes and seeds. Every episode shuffles the six published metro homes (1, 5, 9, 13, 17, 21)
  across the five seats and draws a fresh world seed, so the policy cannot memorise one seating.
* Truncation. The raw env never ends an episode, so the wrapper truncates at ``--episode-ticks``
  (the match length, 1000). SB3 bootstraps through a truncation instead of treating it as a loss.
* Bounded logits. The 35 argmax-decoded logit dims (expansion, automation) are bounded +/-1e9
  in the env's action space. SAC squashes through tanh to the space's bounds, so for SAC they are
  bounded to [-1, 1], which decodes identically. Continuous levers keep their true bounds.
  ``--bound-logits`` / ``--no-bound-logits`` overrides the default (on for SAC only).
* Scoring never applies ``VecNormalize`` statistics, so nothing here normalises observations
  except SAC's fixed scale, which is folded into the network's first layer before saving.
  Rewards are scaled for training only: ``VecNormalize(norm_obs=False)`` for the PPO family, a
  fixed factor for SAC (SB3's off-policy algorithms store the un-normalised reward, which would
  make ``VecNormalize`` do nothing there).
* Seeding follows the README: ``set_random_seed``, not the ``seed=`` argument. Each env copy is
  built separately with its own seed; cloning one env would replay identical worlds.

Train, then self-check the zip against the same NPC field:

    python scripts/train_vs_npcs.py my_sac --algo sac --steps 300000 --out runs/my_sac.zip
    python scripts/arena_competition.py --balanced --rounds 1000 \\
        --player MySac=runs/my_sac.zip \\
        --npc cherry-picker --npc fortress --npc fast-follower --npc sprawler \\
        --out results.json --csv decisions.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from retail_simulator.core.encoding import flatten_action_space_layout
from retail_simulator.core.schema import LeverKind
from retail_simulator.envs.arena import arena_balanced_config, arena_home_regions
from retail_simulator.envs.geo_archetypes import (
    CHERRY_PICKER,
    FAST_FOLLOWER,
    FORTRESS,
    SPRAWLER,
    geo_archetypes,
)
from retail_simulator.envs.parallel_env import RetailParallelEnv
from retail_simulator.envs.spaces import build_action_space, build_observation_space

NPC_FIELD: tuple[str, ...] = (CHERRY_PICKER, FORTRESS, FAST_FOLLOWER, SPRAWLER)
LEARNER_SEAT = 0
N_SEATS = 1 + len(NPC_FIELD)
N_METRO_HOMES = 6  # the published rotation: one home per seat of a full six-seat match
SAC_REWARD_SCALE = 1e-3  # per-tick profit is in the hundreds to thousands
OBS_SCALE_WARMUP_TICKS = 3000


def bounded_logit_space(native: spaces.Box) -> spaces.Box:
    """Copy of ``native`` with every argmax-decoded logit dim bounded to [-1, 1]."""
    low, high = native.low.copy(), native.high.copy()
    for slot in flatten_action_space_layout().levers:
        if slot.spec.kind is LeverKind.DISCRETE:
            low[slot.offset : slot.offset + slot.width] = -1.0
            high[slot.offset : slot.offset + slot.width] = 1.0
    return spaces.Box(low=low, high=high, dtype=np.float32)


class NpcFieldEnv(gym.Env):
    """Seat 0 learns (expansion-masked) against the four NPC archetypes."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        seed: int,
        *,
        episode_ticks: int = 1000,
        bound_logits: bool = False,
        reward_scale: float = 1.0,
        obs_scale: npt.NDArray[np.float32] | None = None,
    ) -> None:
        super().__init__()
        self.np_random, _ = seeding.np_random(seed)
        self._config = arena_balanced_config()
        self._regions = self._config.demand.regions
        self._metro_homes = arena_home_regions(N_METRO_HOMES, n_regions=len(self._regions))
        self._episode_ticks = episode_ticks
        self._reward_scale = reward_scale
        self._obs_scale = obs_scale
        native = build_action_space()
        self.action_space = bounded_logit_space(native) if bound_logits else native
        self.observation_space = build_observation_space(self._config)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        homes = tuple(int(h) for h in self.np_random.permutation(self._metro_homes)[:N_SEATS])
        world_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._world = RetailParallelEnv(
            self._config,
            n_learning_agents=N_SEATS,
            home_regions=homes,
            seed=world_seed,
            mask_expansion_seats=(LEARNER_SEAT,),
        )
        self._agents = self._world.possible_agents
        # A fresh roster per episode: each archetype closure carries its own tick counter.
        roster = geo_archetypes(len(self._regions), self._regions)
        self._npcs = [roster[name] for name in NPC_FIELD]
        self._obs, _ = self._world.reset(seed=world_seed)
        self._tick = 0
        return self._learner_obs(), {"homes": homes, "world_seed": world_seed}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        actions = {self._agents[LEARNER_SEAT]: np.asarray(action, dtype=np.float32)}
        for seat, npc in enumerate(self._npcs, start=1):
            agent = self._agents[seat]
            actions[agent] = np.asarray(
                npc(self._obs[agent], episode_start=self._tick == 0), dtype=np.float32
            ).ravel()
        self._obs, rewards, _, _, _ = self._world.step(actions)
        self._tick += 1
        reward = float(rewards[self._agents[LEARNER_SEAT]]) * self._reward_scale
        return self._learner_obs(), reward, False, self._tick >= self._episode_ticks, {}

    def _learner_obs(self) -> npt.NDArray[np.float32]:
        obs = self._obs[self._agents[LEARNER_SEAT]]
        return obs if self._obs_scale is None else (obs / self._obs_scale).astype(np.float32)


def estimate_obs_scale(
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Per-dim scale (peak |obs| over a random-action rollout, floored at 1) and sample obs.

    Raw observations reach ~2e4 (cash), which a freshly initialised network handles badly. The
    scale only has to be roughly right; trained play may exceed it.
    """
    env = NpcFieldEnv(seed, bound_logits=True)
    env.action_space.seed(seed)
    obs, _ = env.reset()
    peak, samples = np.abs(obs), [obs]
    for tick in range(OBS_SCALE_WARMUP_TICKS):
        obs, _, _, truncated, _ = env.step(env.action_space.sample())
        peak = np.maximum(peak, np.abs(obs))
        if tick % 100 == 0:
            samples.append(obs)
        if truncated:
            obs, _ = env.reset()
    return np.maximum(peak, 1.0).astype(np.float32), np.stack(samples)


def fold_obs_scale(model: SAC, obs_scale: npt.NDArray[np.float32]) -> None:
    """Bake ``obs / obs_scale`` into the first Linear layer of the actor and critics, in place.

    The saved zip then takes RAW observations, as the scorer feeds them. A critic's input is
    (obs, action), so only its first ``len(obs_scale)`` columns are scaled.
    """
    inv = torch.as_tensor(1.0 / obs_scale, dtype=torch.float32)
    policy = model.policy
    critics = [*policy.critic.q_networks, *policy.critic_target.q_networks]
    with torch.no_grad():
        for first in [policy.actor.latent_pi[0], *(net[0] for net in critics)]:
            first.weight[:, : len(inv)] *= inv


def make_env_fns(args: argparse.Namespace, **env_kwargs: Any) -> list[Callable[[], gym.Env]]:
    """One factory per env copy, each with its own seed (cloning one env would repeat worlds)."""
    seeds = [
        int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.n_envs)
    ]
    return [
        lambda s=s: Monitor(
            NpcFieldEnv(
                s, episode_ticks=args.episode_ticks, bound_logits=args.bound_logits, **env_kwargs
            )
        )
        for s in seeds
    ]


def build_model(args: argparse.Namespace, venv: VecEnv) -> Any:
    if args.algo == "ppo":
        return PPO("MlpPolicy", venv, verbose=1)
    if args.algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO("MlpLstmPolicy", venv, verbose=1)
    # gradient_steps=-1: one gradient step per env step, however many envs. The default 1e6
    # buffer is far larger than a short run needs.
    return SAC(
        "MlpPolicy", venv, buffer_size=min(args.steps, 1_000_000), gradient_steps=-1, verbose=1
    )


def train(args: argparse.Namespace) -> Any:
    set_random_seed(args.seed)
    obs_scale = samples = None
    env_kwargs: dict[str, Any] = {}
    if args.algo == "sac":
        obs_scale, samples = estimate_obs_scale(args.seed)
        env_kwargs = {"reward_scale": SAC_REWARD_SCALE, "obs_scale": obs_scale}
    venv: VecEnv = DummyVecEnv(make_env_fns(args, **env_kwargs))
    if args.algo != "sac":
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)
    # SB3 does not seed the action space; SAC's random warm-up actions are drawn from it.
    venv.action_space.seed(args.seed)
    model = build_model(args, venv)
    model.learn(total_timesteps=args.steps)
    if obs_scale is not None:
        before, _ = model.predict(samples / obs_scale, deterministic=True)
        fold_obs_scale(model, obs_scale)
        after, _ = model.predict(samples, deterministic=True)
        drift = float(np.max(np.abs(before - after)))
        if drift > 1e-5:
            raise RuntimeError(f"folding the obs scale changed predictions by {drift:.2e}")
        print(f"obs scale folded into the network (max prediction drift {drift:.1e})")
    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("label", help="run name; names the default --out")
    parser.add_argument("--algo", choices=("ppo", "recurrent_ppo", "sac"), default="ppo")
    parser.add_argument("--steps", type=int, default=300_000, help="env steps summed over copies")
    parser.add_argument("--n-envs", type=int, default=4, help="separately seeded env copies")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-ticks", type=int, default=1000)
    parser.add_argument(
        "--out", type=Path, default=None, help="zip path (default: runs/<label>.zip)"
    )
    parser.add_argument(
        "--bound-logits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="bound the argmax logit dims to [-1, 1] (default: on for sac, off otherwise)",
    )
    args = parser.parse_args(argv)
    if args.bound_logits is None:
        args.bound_logits = args.algo == "sac"
    args.out = args.out or Path("runs") / f"{args.label}.zip"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = train(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
