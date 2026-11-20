# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ydhikaaa/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ydhikaaa/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts the Exophase earn rate (percent of tracked players who have
earned each trophy) of all 55 GTA VI PS5 trophies at a case's target
time, as five quantiles (5/25/50/75/95) per trophy. Scored by mean
pinball loss on the logit scale.

**Round 3** of the make.fun competition `gta6-trophy-rarity-forecast-01`.
Round 1 (submission `01a1f3c3-aed2-703b-a289-426a6042a063`) scored a
practice loss of **0.782**; round 2 (submission
`01a1f4bd-f11a-72f0-b0d3-5518f3d91910`) scored **0.202** with 90%
coverage of 0.927. Round 3 keeps the round-2 model and fixes the one
regime both earlier rounds overshot — the stale page — see "What
changed in round 3" below.

## Run

```
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<dir> node main.ts
```

Node 22, standard library only, no network. Deterministic: the same
case always produces byte-identical output. One `.out` file per `.in`
file, `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`.

## The model

A trophy's earn rate is earners / players tracked, and the two sides of
that fraction are modelled separately.

**Players.** A published count is the true count at the moment Exophase
generated the snapshot, which lags the read that first shows it, so each
refresh point is placed at an estimated snapshot time (half a refresh
gap before the read). Between snapshot points the true curve grows on a
concave ramp — a catch-up jump is spread back over the window in which
those players actually arrived, not dumped at the refresh instant — and
past the last snapshot it is extrapolated with a power-law fit on the
change points. A perfectly flat history near release means the page is
stale and a refresh is due; the rules state the first refresh took the
count 638 → 3,868, so a ~6x jump is the prior there.

**Earners.** Each inflow cohort matures by a **lognormal required-time
kernel**: the share of a cohort of age `a` that has earned a trophy with
median required time `D` is `Phi(ln(a/D)/sigma)`. The thin early tail is
the point — almost nobody earns a 200-hour grind in their first hours,
so slow trophies dilute almost fully when the base jumps, while fast
trophies fill in quickly. Per trophy, `D` and the eventual share `R`
are fit by least squares in logit space (the scorer's scale) over the
refresh points, with a description/tier prior on `D` that carries flat
histories and yields to data when the history moves. The kernel spread
`sigma` is fit once per case (one value for all trophies, chosen by
total fit quality over the trophies whose histories resolve), because a
case whose trophies turn on sharply identifies it.

**Censored observations.** The page prints two decimals, so a printed
0.00 or 0.01 is an interval (true rate below ~0.015), not a point. The
fit scores such observations with a hinge cost in logit space — free
inside the printed interval, squared distance outside it. Treating them
as exact points at the clamp makes slow trophies indistinguishable from
dead ones and can collapse their forecasts to zero; a weak prior on `R`
(toward a tier-typical eventual share) breaks the remaining ties, and a
trophy whose observations are mostly censored gets an honestly wide
spread.

**Floor.** Earners never un-earn and the tracked base never shrinks, so
the median never falls below `r_last * P_last / P_target`, and the 5%
quantile stays near that floor. This is model-free, and it is what keeps
trophies whose snapshot rate is an anomalous fast tail (a handful of
pre-release players) from being forecast to vanish.

**Spread.** The logit standard deviation combines a horizon term
(growing like sqrt(days)), the per-trophy fit residual, a count-noise
term for trophies with few observed earners, and a refresh-jump term
when the page is stale at the checkpoint — a refresh inside the horizon
moves every rate at once, and its size is the dominant uncertainty
there. A trophy at 100% in the last read is treated as universal: its
median and upper quantiles sit at 99.99 (the house's own first refresh
kept it at 100% while a 6x larger base arrived).

## What changed in round 3, and why

Rounds 1 and 2 share one failure: on the launch-night practice cases
the page is **stale** — one snapshot, sat unchanged for hours — and
the answer is the refresh that lands inside the horizon. Both rounds
projected the kernel's full earner growth over that window; round 2's
medians sat systematically above the refreshed page (server: loss
0.202, coverage 0.927, against round 1's 0.782 / 0.073). The refresh
pairs a fresh player count with trophy data from an older crawl wave,
so only part of the projected growth is in the answering snapshot —
while the pure-dilution floor (earners frozen) undershoots by the
aging the pre-refresh base genuinely did.

Round 3 therefore discounts the stale regime, and only that regime
(one snapshot, or the last snapshot ≥ 1.5 h old at the checkpoint):

- **Growth discount.** The median is pulled toward the
  earner-conservation floor in logit space, keeping 35% of the
  model's projected logit growth over the floor.
- **Spread rescale.** The stale-regime spread is scaled to 0.45 of
  round 2's, which over-covered once the medians came down.

The constants were chosen against a family of candidate answer pages
built to reproduce the server's known round-1 and round-2 moments
(loss and coverage jointly): pure dilution, geometric blends toward
round 2's medians, and an old-cohort-aging reconstruction, shrunk and
unshrunk. Across that family round 3 scores **0.071–0.098** (round 2:
0.143–0.229 on the same family), with coverage 0.91–1.00. There is
still no local scorer for the real practice answers, so these are
calibration measurements against constructed truths, not a claimed
practice score.

The dense-history regime is untouched by construction: on a synthetic
final-like case (2.5 days of 30-minute snapshots, checkpoint at the
deadline, target 4.5 days later) round 3's output is byte-identical
to round 2's, so the final's long-history behaviour — the paid case —
is exactly the model that scored 0.218 there in local regression.

## What changed in round 2 vs round 1, and why

Round 1 used an exponential maturity delay for earners. That kernel lets
earner counts grow in proportion to player-hours, and on the launch-night
practice cases it showed up as a uniform bias: every non-100% trophy's
median came out at ~3.6x the pure-dilution reference (snapshot rate x
638/3868), as if a 200-hour grind earned as many new players in 90
minutes as a story trophy. Slow-trophy medians sat roughly 0.8-1.1 logit
too high, the band was too narrow for the refresh regime (the house
baseline's ~0 coverage on the same jump is the extreme version), and
round 1's scored practice loss was 0.782 against the baseline's ~1.51.

Round 2 replaces the kernel (lognormal, above), adds the
earner-conservation floor, the censored-observation fit, the per-case
spread fit, and the refresh-aware snapshot timing and spread. A
half-octave refinement of the fitted `D` was implemented, measured, and
**dropped**: it helped marginally in four synthetic worlds and hurt
materially in one (0.202 → 0.260), so the factor-of-2 grid stays.

## Validation

Local checks on the kit's practice cases (p00, p01): output shape exact
(55 rows, ids unique, exactly the required keys, five ascending finite
quantiles in 0-100, 2,717 / 2,718 bytes against the 65,536 limit);
two runs byte-identical; ~0.3 s for both cases together, ~0.5 s for a
synthetic 2.5-day-history case — far under the 8 s limit. Degenerate
inputs (empty history, missing rates) produce valid output instead of
crashing.

Round 3 re-verified: practice outputs 2,758 / 2,744 bytes, two runs
byte-identical, ~0.7 s for both cases; degenerate inputs (empty
history, all rates null, a single read) produce valid 55-row output;
the synthetic final-like case above (120 reads) runs in ~0.9 s and is
byte-identical to round 2's output on it.

There is no local scorer for the real practice cases, so no practice
loss is claimed for round 2. Calibration was instead measured on a
synthetic harness: worlds with known ground truth (a launch-night case
pair and a final-like case with ~2 days of reads, a ~95k → ~160k player
ramp and a refresh ~3 h before the target), scored with the exact
pinball-on-logit rule. Mean loss over the three cases, round 1 vs
round 2:

| world | round 1 | round 2 |
|---|---|---|
| base (lognormal truth) | 1.072 | 0.209 |
| trophies 30% faster | 0.929 | 0.215 |
| trophies 40% slower | 1.219 | 0.240 |
| wider time spread (sigma 1.4) | 0.761 | 0.202 |
| tighter spread (0.8) + 7 h refresh gaps | 1.662 | 0.362 |
| Weibull truth (different kernel family) | 0.956 | 0.275 |
| snapshot = pre-release pool only | 1.217 | 0.302 |

Round 2's 90% bands covered the truth for 96-100% of trophies in the
base world (round 1: 2-13%). These worlds are my own constructions —
they cannot prove the real score, only that the model is calibrated in
regimes that resemble both the practice cases and the final's
multi-day history, and that round 1's failure modes do not survive in
any of them.
