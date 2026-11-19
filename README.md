# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ydhikaaa/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ydhikaaa/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Dilution + progression forecast of GTA VI PS5 trophy earn rates on Exophase at 2026-11-26: per-trophy exponential maturity delays fit to the launch-week reads, players-tracked growth extrapolated, honest logit spreads.

## Model

A trophy's earn rate is earners / players tracked, and the two counts grow on different clocks after launch: the tracked base grows as buyers link their accounts, while earners grow only as players actually reach that trophy. So the program models the two counts separately and divides:

- **Players P(t):** the true tracked-player curve, linearly interpolated between the reads where the published count moved (a jump is a stale-page catch-up — the players it newly counts started during the window since the previous refresh), extrapolated past the last read with a power-law fit on those change points. A perfectly flat history near release means the page is stale and a refresh is due: the rules state the first refresh took the count 638 → 3,868, so a ~6x jump is the prior for that first refresh.
- **Earners E_i(t):** players mature into a trophy with an exponential delay of mean D_i hours after being tracked. D_i and the eventual share R_i are fit per trophy by least squares on the observed rates (objective in logit space, the scorer's scale), with a description/tier prior on D_i (first mission ~minutes, story chapters hours, grinds/completion hundreds of hours) that carries flat histories and yields to data when the history moves. Reads are evaluated at the refresh they actually show, not at their clock time.
- **Forecast:** rate_i(target) = R_i × C_i(target), where C_i is the matured share of the tracked base at the target.
- **Spread:** logit sd grows with horizon (random-walk-like) plus the per-trophy fit residual and a stale-page term when the history is flat, so the 90% band stays honest about refresh shocks. A trophy every tracked player holds stays pinned at 99.99% (point mass at 100% with the model's diluted value as the low tail).

On the practice cases the medians sit below the last-read rates (dilution, as the rules' first refresh showed) with bands wider than the house baseline's; on a synthetic final-like case (120 reads, different ground-truth delays) it scored a pinball loss of 0.19 vs the house baseline's 0.38, with 90% coverage of 0.96. Deterministic, no network, Node 22 standard library only, ~0.3s per case.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=nmjakjk9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
