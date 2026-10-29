# Param tuning hints

Where to look for a stronger agent. These are directions to investigate, not settings to copy —
what works depends on your algorithm, your budget and your training setup.

## 1. Measure before you tune

- Judge your agent the way it will be scored: deterministic play, full-length matches, several
  seeds and starting regions. A single match tells you almost nothing.
- Train each setting with more than one random seed before deciding it beats another.
- Change one thing at a time and write down what you changed.

## 2. Train longer

- The training budget is usually the biggest single gain. Short runs are nowhere near converged.
  Plan for millions of steps, not a few hundred thousand.

## 3. Choose the algorithm on purpose, and try more than one

- On-policy methods (the PPO family) tend to be stable but need a lot of data. Off-policy methods
  (the SAC family) use data more efficiently but are touchier to configure.
- Which algorithm comes out on top changes a lot with the setup, so there is no safe default.
- Recurrent (memory) policies can remember what happened earlier in a match. Consider them if
  that history matters to your strategy.

## 4. Understand the action

- Each turn mixes adjustable levers (price, marketing, spending levels) with pick-one choices
  (where to open a store, the automation level). Check that your agent can actually express a
  preference on the pick-one choices.
- Watch for all-or-nothing play. If every lever sits at its minimum or maximum, the agent isn't
  really learning to adjust, which usually points to exploration or learning-rate settings.

## 5. Scale inputs and rewards

- Observation values range from small fractions to millions. Normalizing them usually helps, but
  the normalization has to be built into the saved network. Normalization statistics stored
  outside the model are not used at scoring.
- Scaling rewards during training is fine.
- After you change any scaling, retune exploration, because the two interact.

## 6. Think about the time horizon

- A match lasts 1,000 rounds, and many decisions pay off slowly. The usual default discount only
  looks about 100 rounds ahead, so try longer horizons.
- Train on episodes that look like a real match, including the same length.

## 7. Balance exploration

- Too little exploration and the agent locks onto the first decent strategy it finds. Too much
  and it never commits.
- Each algorithm has its own exploration settings. Automatic tuning of them may not settle in a
  short run.
- Scoring always plays deterministically, so exploration only matters while training.

## 8. Network size and learning rate

- The default networks are small for 318 inputs and 44 outputs, so try larger ones.
- The learning rate, and whether it decreases over training, is worth a sweep.

## 9. Train for the field you will actually face

- The final has no NPCs. It is up to six seats of other entries, and the seats, regions and
  seeds rotate and are not disclosed. Beating the practice NPCs is not the goal.
- Vary opponents, seats and starting regions while training, so the agent doesn't overfit one
  situation. Self-play against earlier versions of your agent and a mix of play styles both help.
- Self-play can go in circles or collapse. Keep a pool of past versions to train against.

## 10. Keep your entry valid

- Use one of the supported algorithms with its standard policy type.
- Don't put custom code in the saved model. Built-in settings of any value are fine. Your own
  network classes get the entry rejected, and your own schedule functions get it held for
  manual review.
- Run the self-check on the exact file you submit.
