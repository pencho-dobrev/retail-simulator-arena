# Reward design hints

How to shape what an agent is trying to achieve, so you can build sparring partners with
distinct, strongly expressed styles, and a stronger champion trained against them.

## 1. Change the reward to change the goal

- The reward decides what the agent wants. The algorithm's loss only decides how it learns.
  To get a different kind of player, change what it is paid for.
- Your entry is ranked on total profit. Style rewards are mainly for building sparring partners.
  Whatever your submitted agent was trained on, it has to end up good at profit.

## 2. Give each sparring partner a clear personality

The simulation tracks revenue, profit, cash, market share, store count, regions opened and
employee happiness for every seat, including your rivals. Mix these into the reward to create
characters such as:

- **Expander:** paid for new stores and regions, so it pushes into territory early.
- **Defender:** paid for holding share in its home regions, so it is hard to dislodge.
- **Price fighter:** paid for market share, even at thinner margins.
- **Cautious operator:** paid for profit with a penalty for low cash, so it rarely takes risks.
- **Spoiler:** paid for how far it beats its rivals rather than its own result alone, so it
  plays aggressively against whoever leads.
- **Employer of choice:** paid for keeping staff happy alongside profit.

The point isn't that any of these wins. A champion that has learned to beat all of them is much
harder to surprise.

## 3. Make the style strong enough to show

- If two personas end up playing the same way, the difference in their rewards was too small.
  Push it until their behaviour visibly separates: different store counts, prices, regions and
  spending.
- Check the scale of every term. A term that is tiny next to profit gets ignored. A term that
  is huge drowns everything else.
- Measure the behaviour you get instead of predicting it from the formula. Agents often respond
  to a reward change in ways you didn't expect, sometimes the opposite.

## 4. Beware of agents taking the reward literally

- An agent maximises exactly what you pay for, not what you meant. Paid for store count, it may
  open stores until it goes broke. Paid for revenue, it may sell at a loss.
- Always watch profit, cash and actual play next to the reward. A rising reward with collapsing
  profit means the agent found a loophole.
- Keep penalties bounded. A penalty that grows with the very thing it punishes can spiral and
  break training.

## 5. Reward progress, not standing still

- Paying for a level ("has lots of share") pays the agent just for being there. Paying for a
  change ("gained share this round") rewards actually doing something.
- Rewards spread across the match teach faster than one big end-of-match payout, but they can
  make the agent short-sighted. Balance them against your discount horizon.

## 6. Use shaping as a stepping stone

- A shaped reward can teach basic skills quickly. Fade the extra terms out over training until
  the agent is optimising profit alone.
- Or keep shaped agents purely as opponents, and train your entry on profit against a pool full
  of them.

## 7. Other places to shape behaviour

- Exploration settings change how adventurous an agent is while training, which also yields
  different styles.
- An extra training term can pull an agent toward a reference style, such as one of the practice
  NPCs, without fixing it to that style.
- Training against a particular persona tends to produce a counter to it. That is useful for
  growing a varied pool.

## 8. Put the personas to work

- Add them to your training pool and to your own tournaments as fixed yardsticks.
- Keep several versions of each persona from different stages of training. Early and late
  versions often play quite differently.
- A new champion should beat every persona, not just the last one it trained against.
