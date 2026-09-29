# retail-simulator arena — competitor edition

This is the runnable 25-region arena for training and self-checking a
reinforcement-learning entry. It ships the environment, the training CLI, and
the self-check tooling; the balance study and internal tooling behind the
world's tuning are intentionally not included.

## Setup

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

## Self-check

Seat your checkpoint against the NPC field with
`scripts/arena_competition.py` and play a full match:

```bash
python scripts/arena_competition.py --balanced --rounds 1000 \
    --player MyAgent=path/to/model.zip \
    --npc cherry-picker --npc fortress --npc fast-follower --npc sprawler \
    --out results.json --csv decisions.csv
```

## Submit

Open an entry at
https://github.com/pencho-dobrev/retail-simulator-arena/issues/new with your
Stable-Baselines3 `.zip` checkpoint and an `entry.json`.

## Rules

The world is fixed — `arena_balanced_config()`, not competitor-chosen.

> **Final scoring is run by the organizer on undisclosed seeds and home
> rotations.** The published home rotation is `1, 5, 9, 13, 17, 21` and the
> tools default to `--seed 0`, but an agent tuned to a fixed seed or a fixed
> set of starting regions will not necessarily score well. Train for
> robustness across seeds and starting regions.
