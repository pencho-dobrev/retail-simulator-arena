# retail-simulator arena — competitor edition

This is the runnable 25-region arena for training and self-checking a
reinforcement-learning entry. It ships the environment, the training CLI, and
the self-check tooling; the balance study and internal tooling behind the
world's tuning are intentionally not included.

## Setup

Use Python 3.11 or 3.12. Python 3.13 and newer are not supported (`numpy<2.0`
publishes no prebuilt wheels for them), so the package's `requires-python` makes
`pip install` refuse them up front:

```text
ERROR: Package 'retail-simulator' requires a different Python: 3.14.6 not in '<3.13,>=3.11'
```

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

### The end-of-run output

When training finishes, the trainer plays the pool's own checkpoints against each
other and prints a table ending in a `VERDICT: rl_confirms=…` line. That line is
a diagnostic about the pool of checkpoints, not a check on your agent:
`rl_confirms=False` ("RL does NOT confirm") does not mean training failed, and it
has no bearing on whether a checkpoint is a valid entry. The same goes for the
warning `only N distinct trained checkpoint(s)` on short runs, which ends with
`(Expected with --smoke's tiny round count; …)`. `--smoke` is a flag of the
underlying driver, `scripts/phase4_rl_confirm.py`, and
`arena_train_player.py` does not take it; ignore the warning. Whether an entry
is valid comes down to the checklist on the entry form, starting with the
[self-check](#self-check).

### Your own training loop

Training with your own loop or another algorithm? Seed Stable-Baselines3 with
`stable_baselines3.common.utils.set_random_seed(seed)`, not the `seed=` argument.
The vector environment SuperSuit builds has no `seed()` method, so
`PPO("MlpPolicy", venv, seed=0)` raises
`AttributeError: 'ConcatVecEnv' object has no attribute 'seed'`, with every
SuperSuit version this repo installs.

`build_parallel_vec_env` in `retail_simulator.envs.parallel_gate.bridge` wraps a
`RetailParallelEnv` in the SuperSuit bridge the trainer uses, but its defaults
are not the arena. Called with no arguments it builds a 2-region world with the
weighted reward and a 42-dimensional observation; a policy trained there cannot
play the 318-dimensional arena. Pass the arena explicitly:

| Argument | Default | In the arena |
| --- | --- | --- |
| `config` | 2-region world, weighted reward | `arena_balanced_config()` from `retail_simulator.envs.arena`: 25 regions, profit reward unless you change it, 318-dimensional observation |
| `home_regions` | none; with no `npc_archetypes` either, seat 0 then starts with a store in **every** region | `arena_home_regions(n_seats)`: one home per seat, regions `1, 5, 9, …` in seat order |
| `mask_expansion_seats` | none | your learning seat(s). The trainer and the self-check both play your seat with expansion masking on |
| `n_learning_agents` | 2 | seats played by your policy. Every seat this function builds is played by the one shared policy |
| `npc_archetypes` | none | leave it empty; see the first trap below |

Four traps to know about:

- **`npc_archetypes` cannot seat the self-check field.** It takes the engine's
  built-in NPC names (`discounter`, `premium`, `balanced`, `wandering`). The
  self-check's `cherry-picker`, `fortress`, `fast-follower` and `sprawler` are a
  different family, so passing one raises
  `ValueError: unknown NPC archetype 'fortress'`. The self-check seats them as
  ordinary seats and drives each with a predict function from
  `geo_archetypes(n_regions, config.demand.regions)`. To train against that
  field, seat it the same way in your own wrapper around `RetailParallelEnv`
  rather than through `build_parallel_vec_env`. `ArenaMatch` in
  `src/retail_simulator/envs/arena_match.py` is the reference: every seat,
  NPCs included, is a `RetailParallelEnv` learning seat; the self-check puts the
  players first and the NPCs after them; homes come from `arena_home_regions`;
  expansion masking is on for the player seats only; and each NPC is called with
  its own seat's observation and `episode_start=True` on its first call after a
  reset. That flag matters: each NPC counts rounds internally and only restarts
  the count when it sees it. The four predict functions are in
  `src/retail_simulator/envs/geo_archetypes.py`.
- **Leaving out `home_regions` gives seat 0 the whole map.** With no
  `home_regions` and no `npc_archetypes`, seat 0 starts with a store in all 25
  regions and every other seat with one, so a policy trained there never sees
  the one-store start it is scored from. The self-check seats every seat in a
  single home region, and final scoring rotates those home regions (see
  [Rules](#rules)).
- **The environment never ends an episode.** `terminated` and `truncated` are
  always `False`, so SB3 sees one endless episode and never logs an episode
  return. A self-check match is 1,000 rounds long. If you want episodes,
  truncate them in your own wrapper.
- **`num_vec_envs=N` does not give N independent worlds.** SuperSuit makes the
  copies by cloning one environment, random state included, so all N start from
  the same random stream: given the same actions, two copies follow the same
  trajectory. To get independent copies, build each environment yourself with
  its own seed.

### Example: train against the NPC field

`scripts/train_vs_npcs.py` is a short, readable example of your own training loop.
One learning seat plays the four NPCs from the self-check, with home regions
shuffled and a new world seed every episode, for PPO, RecurrentPPO or SAC. It is a
starting point to read and change, not a tuned entry.

```bash
python scripts/train_vs_npcs.py my_ppo --algo ppo --steps 300000 --out runs/my_ppo.zip
python scripts/train_vs_npcs.py my_sac --algo sac --steps 300000 --out runs/my_sac.zip
python scripts/train_vs_npcs.py my_rppo --algo recurrent_ppo --steps 300000 --out runs/my_rppo.zip
```

Declare the matching `policy_kind` in `entry.json` (see [Submit](#submit)). To
confirm a saved zip holds the policy class you expect, print it with the
one-liner below. It reads the `policy_class` entry of the zip's `data` file and
decodes its text; nothing is unpickled, and neither torch nor Stable-Baselines3
needs to be installed.

| `--algo` | `policy_kind` | The one-liner prints |
| --- | --- | --- |
| `ppo` | `ppo-mlp` | `stable_baselines3.common.policies ActorCriticPolicy` |
| `sac` | `sac` | `stable_baselines3.sac.policies SACPolicy` |
| `recurrent_ppo` | `recurrent-ppo-lstm` | `sb3_contrib.common.recurrent.policies RecurrentActorCriticPolicy` |

```bash
python -c "import base64,json,re,sys,zipfile; d=json.loads(zipfile.ZipFile(sys.argv[1]).read('data')); print(*re.findall(r'[A-Za-z_][\w.]{3,}', base64.b64decode(d['policy_class'][':serialized:']).decode('latin-1')))" runs/my_sac.zip
```

It tells policy families apart, not every algorithm: A2C also prints
`ActorCriticPolicy`, and DDPG prints `TD3Policy`.

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

> **Pass `--balanced`.** `scripts/arena_competition.py` plays the lock world
> unless you pass `--balanced`, while the trainer defaults to
> `--world balanced`. Without the flag the match still runs, but in a different
> world from the one you trained in and the one the competition is played in, so
> its numbers say nothing about your entry. The first log line names the world:
> it should read `arena match starting: arena-balanced (…) world`.

## Submit

Open an entry with the
[competition entry form](https://github.com/pencho-dobrev/retail-simulator-arena/issues/new?template=competition-entry.yml)
and attach two files to its Checkpoint field: your checkpoint `.zip` (the one you
self-checked) and an `entry.json`.

`entry.json` describes the checkpoint. It has six fields, the same six the form
asks for:

| Field | Value | Notes |
| --- | --- | --- |
| `team` | string | Your team name, as it should appear in the standings. |
| `agent_name` | string | The name for this checkpoint's seat. |
| `policy_kind` | `"ppo-mlp"`, `"a2c-mlp"`, `"recurrent-ppo-lstm"`, `"sac"`, `"td3"`, `"ddpg"` or `"tqc"` | The algorithm and policy the zip was trained with. The tooling detects it from the zip, but when it cannot read the zip it falls back to `ppo-mlp` without failing. That only goes unnoticed for a `recurrent-ppo-lstm` entry, which would then play without its recurrent state, so your declaration is what makes the fallback visible. |
| `contact` | string | How the organizer reaches you about the entry. |
| `training_recipe` | string or object | Free text, or the contents of the `recipe.json` that `arena_train_player.py` wrote into your pool directory. |
| `sb3_version` | string | The stable-baselines3 version that saved the zip (the repo pins `>=2.3,<3.0`). |

Read `sb3_version` from the zip itself rather than from your current
environment: `model.save(...)` writes a `system_info.txt` into the zip, and its
`Stable-Baselines3:` line is the version that saved it. This is the same for
sb3-contrib algorithms.

```bash
unzip -p path/to/model.zip system_info.txt
```

```json
{
  "team": "your-team",
  "agent_name": "MyAgent",
  "policy_kind": "ppo-mlp",
  "contact": "you@example.com",
  "training_recipe": "arena_train_player.py my_agent --world balanced --ent-coef 0.02 --seeds 2,2,44",
  "sb3_version": "2.8.0"
}
```

The checkpoint is a standard Stable-Baselines3 or sb3-contrib `.zip` written by
`model.save(...)`, from one of these algorithms:

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

Checkpoints are compared at scoring time. If two entries' checkpoint files are
byte-identical (same SHA-256), only the entry whose issue was filed first is
scored; the later one is disqualified.
