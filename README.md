# retail-simulator arena — competitor edition

This is the runnable 25-region arena for training and self-checking a
reinforcement-learning entry. It ships the environment, the training CLI, and
the self-check tooling; the balance study and internal tooling behind the
world's tuning are intentionally not included.

## Setup

Use Python 3.11 or 3.12. Python 3.13 and newer are not supported: `numpy<2.0`
publishes no prebuilt wheels for them, so pip stops with
`No matching distribution found for numpy` or has to compile numpy from source.
If `python3` is newer than 3.12, create the environment with
`python3.12 -m venv .venv` instead.

```bash
git clone https://github.com/pencho-dobrev/retail-simulator-arena.git
cd retail-simulator-arena

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[rl,train,recurrent]"
```

## Train

`scripts/arena_train_player.py` builds the training world, trains one PPO
player, and writes the full effective recipe to `recipe.json` in the pool
directory. Checkpoints land under `runs/phase4_rl_confirm/<label>/`, one
snapshot per round — `runs/` is gitignored, so a checkpoint exists only on
your own machine until you submit one.

Always train in the balanced world (`--world balanced`) — that is the world
the competition is played in.

`--rounds N` writes `round_0.zip` through `round_{N-1}.zip`. `round_0.zip` is
the untrained, randomly initialised snapshot, not a trained agent, so train with
`--rounds 2` or more. The checkpoint to use is the last one: `round_5.zip` at the
default of 6 rounds.

Rounds grow the pool of opponents the next round trains against; they do not add
training steps to a checkpoint. Each round trains a fresh policy from scratch
for `--train-steps-per-round` steps (default 300,000), so a checkpoint's
training budget is that number, not rounds times steps. With `--warm-start`,
round 2 onward continues from the previous round's checkpoint instead.

Three reference recipes — they differ by objective, not by seed:

```bash
# Explorer -- profit-only reward with an entropy bonus.
python scripts/arena_train_player.py my_explorer --world balanced \
    --ent-coef 0.02 --seeds 2,2,44

# Margin -- profit-only reward (the default), no entropy bonus.
python scripts/arena_train_player.py my_margin --world balanced --seeds 0,0,42

# Share -- the weighted objective instead of profit only.
python scripts/arena_train_player.py my_share --world balanced \
    --reward weighted --seeds 1,1,43
```

Training with your own loop or another algorithm? Seed Stable-Baselines3 with
`stable_baselines3.common.utils.set_random_seed(seed)`, not the `seed=` argument.
The vector environment SuperSuit builds has no `seed()` method, so
`PPO("MlpPolicy", venv, seed=0)` raises
`AttributeError: 'ConcatVecEnv' object has no attribute 'seed'`, with every
SuperSuit version this repo installs. `build_parallel_vec_env` in
`retail_simulator.envs.parallel_gate.bridge` builds the vector environment the
arena trainer uses.

### Example: train against the NPC field

`scripts/train_vs_npcs.py` is a short, readable example of your own training loop.
One learning seat plays the four NPCs from the self-check, with home regions
shuffled and a new world seed every episode, for PPO, RecurrentPPO or SAC. It is a
starting point to read and change, not a tuned entry.

```bash
python scripts/train_vs_npcs.py my_ppo --algo ppo --steps 300000 --out runs/my_ppo.zip
```

## Self-check

Seat your checkpoint against the NPC field with
`scripts/arena_competition.py` and play a full match. This is practice, not how
finals are scored (see [Rules](#rules)).

```bash
python scripts/arena_competition.py --balanced --rounds 1000 \
    --player MyAgent=path/to/model.zip \
    --npc cherry-picker --npc fortress --npc fast-follower --npc sprawler \
    --out results.json --csv decisions.csv
```

## Submit

Open an entry at
https://github.com/pencho-dobrev/retail-simulator-arena/issues/new with your
checkpoint and an `entry.json`. The checkpoint is a standard Stable-Baselines3 or
sb3-contrib `.zip` written by `model.save(...)`, from one of these algorithms:

| Algorithm | Policy | Library |
| --- | --- | --- |
| PPO, A2C | `MlpPolicy` | stable-baselines3 |
| RecurrentPPO | `MlpLstmPolicy` (stateful) | sb3-contrib |
| SAC, TD3, DDPG | `MlpPolicy` | stable-baselines3 |
| TQC | `MlpPolicy` | sb3-contrib |

The tooling detects the algorithm from the zip. A RecurrentPPO policy keeps its
recurrent state across the whole match.

The observation is a fixed 318-dimensional vector and the action a 44-dimensional
continuous vector, so `CnnPolicy` and `MultiInputPolicy` do not apply, and neither do
discrete-only algorithms such as DQN.

Two things hold for every algorithm:

- Scoring always predicts deterministically, whatever the policy was trained with.
- `VecNormalize` statistics are stored outside the zip and are not applied at scoring,
  so a policy trained with normalization must bake it into the network or not use it.

## Rules

The world is fixed — `arena_balanced_config()`, not competitor-chosen.

> **Final scoring is a tournament among the entries — there are no NPCs in
> it.** The organizer groups all entrants into matches of up to six seats.
> Across several matches the groups are re-drawn (when there are more than six
> entrants) and the seats and home regions are rotated, so a rank does not rest
> on a single seating or a single starting region. Every entrant plays the same
> number of matches, and entrants are ranked by their aggregate `total_profit`
> across them. The seed and the home rotations are undisclosed.

The self-check above is local practice and is not how finals are scored: the
opponents at final scoring are the other entries, so a strong result against the
NPC field is not a prediction of your rank. The published home rotation is
`1, 5, 9, 13, 17, 21` and the tools default to `--seed 0`, but an agent tuned to a
fixed seed, a fixed set of starting regions or a fixed set of opponents will not
necessarily score well. Train for robustness across seeds, starting regions and
opponents.
