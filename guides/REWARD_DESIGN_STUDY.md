# Reward design: study guide

The background reading for [Reward design hints](REWARD_DESIGN_HINTS.md). Each section matches a
section of the hints, with a two-sentence summary and sources to learn from. Sources are listed
roughly from the most approachable to the most advanced.

## Start here

- [Specification gaming: the flip side of AI ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
  — Krakovna et al., DeepMind, 2020 · *blog*. The clearest short introduction to how agents
  exploit the reward they are given.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) —
  Sutton & Barto, 2018 · *free book*. Section 3.2 on goals and rewards, and section 17.4 on
  designing reward signals.
- [A Practical Guide to Multi-Objective RL and Planning](https://arxiv.org/abs/2103.09568) — Hayes
  et al., 2021 · *guide*. How to think about rewards that combine several goals.

## 1. Change the reward to change the goal

The reward is the only description of the goal the agent ever sees, so it is the main way to
make an agent want something different. The loss only controls how the network is updated
towards that goal.

- [Spinning Up: Key concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — OpenAI ·
  *tutorial*. Where rewards, returns and policy losses each fit.
- Sutton & Barto, [section 3.2](http://incompleteideas.net/book/the-book-2nd.html) · *book*. The
  "reward hypothesis": that any goal can be expressed as a reward to maximise.

## 2. Give each sparring partner a clear personality

Agents trained on different mixtures of goals behave like different players, which gives a
champion varied opponents to learn from. Weighting several goals into one reward is a topic of
its own.

- [A Practical Guide to Multi-Objective RL and Planning](https://arxiv.org/abs/2103.09568) — Hayes
  et al., 2021 · *guide*. Weighting goals and what happens when you combine them.
- [Human-level Performance in First-Person Multiplayer Games](https://arxiv.org/abs/1807.01281) —
  Jaderberg et al., 2018 · *paper*. Each agent in a population learns its own internal reward,
  producing varied teammates and opponents.
- [AlphaStar Nature paper](https://www.nature.com/articles/s41586-019-1724-z) — Vinyals et al.,
  2019 · *paper*. Exploiter agents trained with different objectives from the main agent.

## 3. Make the style strong enough to show

A small change in the reward often produces no visible change in behaviour, so diversity has to
be pushed deliberately and then measured. Research on diversity offers ways to make agents
behave differently on purpose.

- [Diversity is All You Need](https://arxiv.org/abs/1802.06070) — Eysenbach et al., 2018 ·
  *paper*. Learning distinct behaviours by rewarding distinguishability.
- [Effective Diversity in Population Based RL](https://arxiv.org/abs/2002.00632) — Parker-Holder et
  al., 2020 · *paper*. Rewarding a population for behaving differently from each other.
- [A Practical Guide to Multi-Objective RL and Planning](https://arxiv.org/abs/2103.09568) — Hayes
  et al., 2021 · *guide*. Why the scale of each term decides which one the agent listens to.

## 4. Beware of agents taking the reward literally

Agents optimise exactly what is measured, and often find shortcuts that score well while
missing the intended goal. Learning to spot this early saves wasted training runs.

- [Specification gaming examples in AI](https://vkrakovna.wordpress.com/2018/04/02/specification-gaming-examples-in-ai/)
  — Krakovna, 2018 · *list*. A long, entertaining collection of real reward exploits.
- [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
  — Weng, 2024 · *blog*. A thorough overview of causes and fixes.
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) — Amodei et al., 2016 ·
  *paper*. The reward hacking section is the classic framing.
- [Reward (Mis)design for Autonomous Driving](https://arxiv.org/abs/2104.13906) — Knox et al.,
  2021 · *paper*. A practical checklist of reward design mistakes, using driving as the example.
- [Defining and Characterizing Reward Hacking](https://arxiv.org/abs/2209.13085) — Skalse et al.,
  2022 · *paper*. The formal side, for those who want it.

## 5. Reward progress, not standing still

There is a well-founded way to add helpful rewards without changing which strategy is best:
reward the change in a measure, not its level. How often the reward arrives also shapes how far
ahead the agent learns to look.

- [Policy Invariance Under Reward Transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
  — Ng, Harada & Russell, 1999 · *paper*. The original result on potential-based shaping: reward
  progress and the best strategy stays the same.
- Sutton & Barto, [section 17.4](http://incompleteideas.net/book/the-book-2nd.html) · *book*.
  Sparse versus frequent rewards, and shaping.

## 6. Use shaping as a stepping stone

Extra reward terms can teach basic skills early, and then be removed so the agent ends up
optimising the real goal. This is one form of curriculum learning.

- [Curriculum Learning for RL Domains](https://arxiv.org/abs/2003.04960) — Narvekar et al., 2020 ·
  *survey*. Ways to order and fade training tasks and rewards.
- [Policy Invariance Under Reward Transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
  — Ng, Harada & Russell, 1999 · *paper*. Which shaping terms are safe to leave in.

## 7. Other places to shape behaviour

Rewards are not the only lever: exploration settings, extra training terms that pull towards a
reference style, and the choice of training opponents all change how an agent plays. Training
against one fixed opponent reliably produces a specialist counter to it.

- [Modeling Strong and Human-Like Gameplay with KL-Regularized Search](https://arxiv.org/abs/2112.07544)
  — Jacob et al., 2021 · *paper*. Pulling an agent towards a reference style while still letting
  it play well.
- [Understanding the Impact of Entropy on Policy Optimization](https://arxiv.org/abs/1811.11214) —
  Ahmed et al., 2018 · *paper*. How the entropy bonus changes what the agent learns.
- [Adversarial Policies: Attacking Deep RL](https://arxiv.org/abs/1905.10615) — Gleave et al., 2019
  · *paper*. Training against a frozen opponent finds strategies that beat it specifically.

## 8. Put the personas to work

Personas pay off when they become permanent members of the training pool and the evaluation
tournament. The champion should be judged against all of them, not just the latest.

- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) — Jaderberg et
  al., 2017 · *paper*. Keeping and evolving a population of agents.
- [AlphaStar](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
  — DeepMind, 2019 · *blog*. A league that keeps varied agents in play permanently.
- More in the [training tournament study guide](TRAINING_TOURNAMENT_STUDY.md).
