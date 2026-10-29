# Param tuning: study guide

The background reading for [Param tuning hints](PARAM_TUNING_HINTS.md). Each section matches a
section of the hints, with a two-sentence summary and sources to learn from. Sources are listed
roughly from the most approachable to the most advanced.

## Start here

- [Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)
  — Sutton & Barto, 2018 · *free book*. The standard textbook; chapters 1–6 and 13 cover what you
  need for this competition.
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) — OpenAI · *tutorial*. A
  short, clear path from the basics to modern policy-gradient methods.
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
  · *free course*. Hands-on units that use Stable-Baselines3, the library the arena runs on.
- [CS 285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) — UC
  Berkeley · *university course*. Lectures and slides if you want the full theory.
- [RL Course by David Silver](https://www.davidsilver.uk/teaching/) — UCL/DeepMind · *lecture
  series*. A classic ten-lecture introduction.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) · *code to read*. Each algorithm in one readable
  file, the easiest way to see what an algorithm actually does.

## 1. Measure before you tune

The same algorithm with the same settings can score very differently from one random seed to the
next, so a single run cannot tell you whether a change helped. Learn how to compare agents over
several seeds with confidence intervals before trusting any improvement.

- [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
  — Irpan, 2018 · *blog*. A readable tour of why RL results are noisy and fragile.
- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560) — Henderson et
  al., 2017 · *paper*. Shows how seeds, settings and code details swing results.
- [How Many Random Seeds?](https://arxiv.org/abs/1806.08295) — Colas et al., 2018 · *paper*. How
  many runs you need before a difference is real.
- [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) — Agarwal
  et al., 2021 · *paper*, with the [rliable](https://github.com/google-research/rliable) library.
  Better ways to summarise a handful of runs.
- [Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315) — Patterson et
  al., 2023 · *long guide*. How to design experiments you can trust.

## 2. Train longer

Deep RL needs far more experience than most people expect, and agents that look stuck early
often keep improving much later. Learn why sample efficiency is hard and what can make long runs
stall.

- [Reinforcement Learning Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
  — Stable-Baselines3 docs · *guide*. Practical advice on budgets, evaluation and common mistakes.
- [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
  — Irpan, 2018 · *blog*. Its section on sample inefficiency explains why millions of steps are
  normal.
- [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/abs/2205.07802) — Nikishin
  et al., 2022 · *paper*. Why agents can overfit their early experience and stop improving.

## 3. Choose the algorithm on purpose, and try more than one

The algorithms fall into on-policy methods (PPO, A2C, RecurrentPPO), which learn only from fresh
experience, and off-policy methods (SAC, TQC, TD3, DDPG), which reuse old experience from a
memory. Knowing how each one learns and explores tells you which settings matter for it.

- [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) —
  Weng, 2018 · *blog*. One clear overview of nearly every algorithm in the arena.
- [Stable-Baselines3 algorithm overview](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html)
  · *docs*. What each algorithm supports, side by side.
- [Spinning Up key papers](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) ·
  *reading list*. Where each algorithm came from.
- On-policy originals: [PPO](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) and
  [A3C/A2C](https://arxiv.org/abs/1602.01783) (Mnih et al., 2016) · *papers*.
- Off-policy originals: [DDPG](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2015),
  [TD3](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018),
  [SAC](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) and
  [TQC](https://arxiv.org/abs/2005.04269) (Kuznetsov et al., 2020) · *papers*.
- Memory: [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://arxiv.org/abs/2110.05038)
  (Ni et al., 2021) and [Deep Recurrent Q-Learning](https://arxiv.org/abs/1507.06527) (Hausknecht
  & Stone, 2015) · *papers*. When and why a policy with memory helps.

## 4. Understand the action

Each arena action mixes continuous levers with pick-one choices, and an algorithm's output
distribution decides how well it can express each kind. Watch for policies that only ever push
levers to their limits, a known behaviour with its own research.

- [Is Bang-Bang Control All You Need?](https://arxiv.org/abs/2111.02552) — Seyde et al., 2021 ·
  *paper*. Why trained agents often end up all-or-nothing on continuous controls.
- [Improving Stochastic Policy Gradients with the Beta Distribution](https://proceedings.mlr.press/v70/chou17a.html)
  — Chou et al., 2017 · *paper*. How clipping a Gaussian at the action bounds biases learning.
- [A Closer Look at Invalid Action Masking](https://arxiv.org/abs/2006.14171) — Huang & Ontañón,
  2020 · *paper*. How masking unavailable choices affects policy gradients.
- [Reinforcement Learning with Parameterized Actions](https://arxiv.org/abs/1509.01644) — Masson
  et al., 2015, and [Hybrid Actor-Critic RL](https://arxiv.org/abs/1903.01344) — Fan et al., 2019
  · *papers*. Learning with mixed discrete and continuous actions.

## 5. Scale inputs and rewards

Neural networks learn badly when inputs and rewards span many orders of magnitude, so
normalising them is one of the most reliable improvements. Because scaling interacts with
learning rates and exploration, it is worth understanding rather than just switching on.

- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  — Huang et al., 2022 · *blog*. Walks through observation and reward normalisation among many
  other small details that matter.
- [What Matters in On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990) —
  Andrychowicz et al., 2020 · *paper*. A large study of which choices, normalisation included,
  actually change results.
- [Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — Engstrom
  et al., 2020 · *paper*. Shows code-level tricks can matter as much as the algorithm.
- [Learning Values Across Many Orders of Magnitude](https://arxiv.org/abs/1602.07714) — van
  Hasselt et al., 2016 · *paper*. Adaptive reward scaling (PopArt).

## 6. Think about the time horizon

The discount factor sets how far ahead an agent effectively plans, and a horizon that is too
short makes slow-paying decisions look worthless. How episodes end during training also matters:
cutting a match short is not the same as the game ending.

- Sutton & Barto, [chapter 3](http://incompleteideas.net/book/the-book-2nd.html) (returns and
  discounting) and chapter 12 (λ-returns) · *book*. The foundations.
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015 ·
  *paper*. How γ and λ trade bias for variance in on-policy learning.
- [Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378) — Pardo et al., 2017
  · *paper*. Why a truncated episode must be treated differently from a finished one.
- [Discount Factor as a Regularizer](https://arxiv.org/abs/2007.02040) — Amit et al., 2020 ·
  *paper*. What changing γ really does to learning.

## 7. Balance exploration

An agent only learns about actions it tries, so too little exploration locks it into its first
decent habit while too much keeps it from ever committing. Each algorithm explores differently:
entropy bonuses, a learned temperature, or added action noise.

- [Exploration Strategies in Deep RL](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)
  — Weng, 2020 · *blog*. A broad, readable overview.
- [Understanding the Impact of Entropy on Policy Optimization](https://arxiv.org/abs/1811.11214)
  — Ahmed et al., 2018 · *paper*. What the entropy bonus does to the learning landscape.
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) — Haarnoja et
  al., 2018 · *paper*. How SAC tunes its own exploration temperature.
- [Smooth Exploration for Robotic RL (gSDE)](https://arxiv.org/abs/2005.05719) — Raffin et al.,
  2020 · *paper*. A smoother exploration option built into Stable-Baselines3.
- [Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905) — Plappert et al.,
  2017 · *paper*. Exploring by perturbing the network instead of the action.

## 8. Network size and learning rate

The default networks and learning rates are generic starting points, not values tuned for this
game. Systematic sweeps beat guesswork, and RL is more sensitive to these settings than most
machine learning.

- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) · *tuned examples*. Published
  hyperparameters for many tasks, useful as starting points.
- [Optuna](https://optuna.org/) · *tool*. A widely used library for automated hyperparameter
  search.
- [Hyperparameters in RL and How To Tune Them](https://arxiv.org/abs/2306.01324) — Eimer et al.,
  2023 · *paper*. Practical recommendations for tuning RL fairly.
- [What Matters in On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990) —
  Andrychowicz et al., 2020 · *paper*. Covers network size and learning rate among its findings.
- [Automated Reinforcement Learning (AutoRL)](https://arxiv.org/abs/2201.03916) — Parker-Holder et
  al., 2022 · *survey*. The wider field of automatic tuning.

## 9. Train for the field you will actually face

An agent trained against one set of opponents learns to beat those opponents, not opponents in
general. Training against varied and changing opponents, including past versions of yourself,
is the standard way to build robust competitive agents.

- [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748) — Bansal et
  al., 2017 · *paper*. Why sampling from past versions of yourself keeps self-play stable.
- [A Unified Game-Theoretic Approach to Multiagent RL](https://arxiv.org/abs/1711.00832) — Lanctot
  et al., 2017 · *paper*. Shows how agents overfit to the opponents they trained with.
- [AlphaStar: Grandmaster level in StarCraft II](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
  — DeepMind, 2019 · *blog*, with the [Nature paper](https://www.nature.com/articles/s41586-019-1724-z).
  A league of varied opponents at scale.
- More in the [training tournament study guide](TRAINING_TOURNAMENT_STUDY.md).

## 10. Keep your entry valid

A saved model is a file of settings and network weights. Loading one that contains arbitrary
code is a security risk, which is why entries are screened and custom code inside them is
refused.

- [Stable-Baselines3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)
  · *docs*. How saving and loading a model works.
- [Python pickle documentation](https://docs.python.org/3/library/pickle.html) · *docs*. The
  warning at the top explains why loading untrusted files can run code.
