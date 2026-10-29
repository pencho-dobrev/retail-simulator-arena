# Training tournaments: study guide

The background reading for [Training tournament hints](TRAINING_TOURNAMENT_HINTS.md). Each
section matches a section of the hints, with a two-sentence summary and sources to learn from.
Sources are listed roughly from the most approachable to the most advanced.

## Start here

- [AlphaStar: Grandmaster level in StarCraft II](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
  — DeepMind, 2019 · *blog*. The best-known example of growing a champion through a league of
  agents; read the blog first, then the [Nature paper](https://www.nature.com/articles/s41586-019-1724-z).
- [A Survey and Critique of Multiagent Deep RL](https://arxiv.org/abs/1810.05587) —
  Hernandez-Leal et al., 2018 · *survey*. A map of the multi-agent field and its pitfalls.
- [OpenSpiel](https://github.com/google-deepmind/open_spiel) — DeepMind · *code to read*.
  Reference implementations of many of the league and evaluation methods below.

## 1. Build a candidate pool

A good pool mixes agents from different runs, settings and stages of training, because the
newest checkpoint is often not the strongest. Population methods formalise this by training many
agents at once and copying the winners.

- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) — Jaderberg et
  al., 2017 · *paper*. Train a population, keep the best, mutate their settings.
- [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748) — Bansal et
  al., 2017 · *paper*. Training against a mix of past versions instead of only the latest.
- [Dota 2 with Large Scale Deep RL (OpenAI Five)](https://arxiv.org/abs/1912.06680) — OpenAI, 2019
  · *paper*. How a long-running project kept old versions in its opponent mix.

## 2. Copy the final's format

How you evaluate decides which agent you pick, so the evaluation should match the real contest:
the same number of seats, match length, rotation and scoring. In multi-player games a single
head-to-head score says little, and rating systems summarise many mixed matches.

- [TrueSkill](https://trueskill.org/) · *docs*. A rating system built for multi-player matches,
  with a readable explanation of how it works.
- [Re-evaluating Evaluation](https://arxiv.org/abs/1806.02643) — Balduzzi et al., 2018 · *paper*.
  Why the choice of opponents in an evaluation can change the winner.
- [α-Rank: Multi-Agent Evaluation by Evolution](https://arxiv.org/abs/1903.01373) — Omidshafiei et
  al., 2019 · *paper*. Ranking agents when no single one beats all the others.

## 3. Play enough matches to trust the result

Tournament results are noisy, and small gaps between agents often disappear when the matches are
replayed with new seeds. Confidence intervals tell you whether a ranking is real or luck.

- [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) — Agarwal
  et al., 2021 · *paper*, with the [rliable](https://github.com/google-research/rliable) library.
  Bootstrap confidence intervals for small numbers of runs.
- [How Many Random Seeds?](https://arxiv.org/abs/1806.08295) — Colas et al., 2018 · *paper*.
  Estimating how many repeats a comparison needs.
- [Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315) — Patterson et
  al., 2023 · *long guide*. Designing comparisons you can trust.

## 4. Avoid the common traps

Multi-agent results hide several traps: agents overfit to the partners they trained with, copies
of one agent can fight over the same resources, and strategies can beat each other in circles.
A champion can also hide a narrow weakness that one well-aimed opponent exploits.

- [A Unified Game-Theoretic Approach to Multiagent RL](https://arxiv.org/abs/1711.00832) — Lanctot
  et al., 2017 · *paper*. Measures how badly agents do against opponents they did not train with.
- [Multi-agent RL in Sequential Social Dilemmas](https://arxiv.org/abs/1702.03037) — Leibo et al.,
  2017 · *paper*. How competition between similar agents changes with scarce resources.
- [Real World Games Look Like Spinning Tops](https://arxiv.org/abs/2004.09468) — Czarnecki et al.,
  2020 · *paper*. Why many games contain "A beats B beats C beats A" cycles.
- [Open-ended Learning in Symmetric Zero-sum Games](https://arxiv.org/abs/1901.08106) — Balduzzi et
  al., 2019 · *paper*. Using diverse populations to escape those cycles.
- [Adversarial Policies Beat Superhuman Go AIs](https://arxiv.org/abs/2211.00241) — Wang et al.,
  2022 · *paper*. A weak but targeted agent beats a champion, a vivid case for diverse testing.

## 5. Grow a champion over several generations

Strong agents are usually grown in rounds: each new generation trains against the best so far,
and the pool keeps older champions so skills are not forgotten. League methods add agents whose
only job is to find and exploit the champion's weaknesses.

- [AlphaStar Nature paper](https://www.nature.com/articles/s41586-019-1724-z) — Vinyals et al.,
  2019 · *paper*. Main agents, exploiters and prioritised opponent sampling in one league.
- [A Unified Game-Theoretic Approach to Multiagent RL (PSRO)](https://arxiv.org/abs/1711.00832) —
  Lanctot et al., 2017 · *paper*. Adding best responses to a growing population, generation by
  generation.
- [Deep RL from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121) —
  Heinrich & Silver, 2016 · *paper*. Mixing a best response with an average of past play.
- [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528) — Baker et
  al., 2019 · *paper*. How rounds of competition produce new strategies on their own.
- [Autocurricula and the Emergence of Innovation](https://arxiv.org/abs/1903.00742) — Leibo et al.,
  2019 · *essay*. Why competing populations keep generating harder challenges.
