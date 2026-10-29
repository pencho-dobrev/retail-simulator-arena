# Training tournament hints

How to run your own tournaments to pick and grow a champion. Training reward is a poor way to
choose your entry: it depends on who each checkpoint happened to train against. Choose by playing
your candidates against each other in a tournament that looks like the final.

## 1. Build a candidate pool

- Save checkpoints throughout every run. The last one is not always the best.
- Make the pool varied: different algorithms, settings, seeds and training opponents.
- Keep the practice NPCs and a few simple fixed strategies in the pool as yardsticks that never
  change. They show whether your pool is improving, but they shouldn't decide the winner.

## 2. Copy the final's format

- Matches of up to six seats, full length (1,000 rounds), ranked by total profit summed across
  matches.
- Rotate seats and starting regions so every candidate plays every position. A rank shouldn't
  depend on one seating or one region.
- Re-draw the groups between matches so candidates meet many different line-ups.
- Use seeds you never trained on, and fresh ones again for the deciding round.
- Play deterministically, the way scoring does.

## 3. Play enough matches to trust the result

- Differences between mid-field candidates are small and noisy. Re-run the tournament with new
  seeds and see how much the order moves. If two candidates' results overlap, treat them as
  tied.
- Results at a different match length can crown a different winner, so evaluate at the real
  length.

## 4. Avoid the common traps

- Copies of the same agent can behave very differently when they share a match: one can crowd
  the other out. Don't judge a candidate only against its own siblings.
- A can beat B, B can beat C, and C can still beat A. Pick the candidate with the best average
  against the whole field, not the winner of one head-to-head.
- Results from two-seat matches don't predict six-seat matches. Evaluate at full size.
- A champion that only beats your own pool may lose to other teams' entries. Keep the pool
  diverse so it isn't just variations of one idea.

## 5. Grow a champion over several generations

- Train new candidates against the current top few and against past champions, then run the
  tournament again. Repeat.
- Keep a hall of fame of past champions in the pool, so a new candidate can't win by forgetting
  how to beat older styles.
- Promote a new champion only if it clearly beats the current one on fresh seeds, not on one
  lucky run.
- Before submitting, freeze the champion, run one last tournament on brand-new seeds and regions,
  then self-check the exact file.
