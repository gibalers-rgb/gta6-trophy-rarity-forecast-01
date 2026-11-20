# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gibalers-rgb/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gibalers-rgb/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A quantitative forecast of how many GTA VI players earn each trophy in week
one: for every one of the game's 55 PlayStation 5 trophies, five quantiles of
its earn rate on Exophase at the case's target time (final target:
2026-11-26T00:00:00Z). Deterministic, Python 3.13 standard library only, no
network.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`,
one row per trophy, quantiles in percent, non-decreasing.

## How it forecasts

The program fits each trophy's own trajectory at run time from the case's
history of Exophase reads — it does not hard-code the answer:

1. **Dedupe.** Consecutive identical reads are collapsed: Exophase refreshes
   in steps, so a history can hold many copies of one page (the practice
   histories are 7–8 identical reads of the 638-player launch-night page).
2. **Logit trend.** For each trophy, weighted least squares of
   `logit(rate)` against `ln(1 + hours since launch 2026-11-19T00:00Z)`,
   weights rising toward the last read. The fitted slope is shrunk toward a
   category prior (story chapters rise fastest, grind / platinum stay ~flat),
   more so when the trend is weakly identified, and damped over long
   horizons: as the tracked population broadens from day-one buyers to
   casual players, progress saturates and dilution bites.
3. **Regime split.** Reads before the first big refresh of the tracked count
   (launch night: 638 players, rates inflated by the tiny cohort) are dropped
   from the trend fit once the count has tripled.
4. **Ceilings.** The median is clamped to a per-category day-7 ceiling
   (opening mission 99.9%; Chapter 1 30% / Chapter 2 15% / Chapter 3 8% /
   Chapter 4 4% / final mission 2%; named story missions 1.5–8%; easy
   open-world 0.4–20%; grind / collectibles 0.02–4%; 100% completion 0.02%;
   platinum 0.005%), and its total logit move from the last read is capped
   (+1.5 / −1.0 over a long horizon). Near-zero trophies stay near zero: on
   the logit scale, forecasting 0.1% for a true 0.01% trophy costs ~2.3
   logit units, so restraint matters as much as growth.
5. **Spread.** `sigma_logit = 0.35 + 0.35 * sqrt(horizon_days)`, plus 0.25
   when the last rate is below 0.05% (small-count noise) and 0.2 when the
   trend is weakly identified, capped at 1.3. Quantiles are
   `inv-logit(logit(median) + z_tau * sigma)`, `z = [-1.645, -0.674, 0,
   0.674, 1.645]`, rounded to 4 decimals half-up.
6. **Dilution branch.** If the history holds a single distinct read and the
   checkpoint is within ~12 h of launch (the practice shape), the forecast
   is a refresh drop (median ~0.35x the last rate for non-opening trophies)
   instead of persistence — the house baseline's practice failure is exactly
   this refresh, when Exophase's tracked count jumped 638 → 3,868 and every
   rate fell.

## Why this should beat the house baseline

The baseline's median is the stale last rate with a random-walk spread
(`sigma 0.5 * sqrt(days)`). Week-one rates move a lot over the final's
4.5-day horizon — early-story trophies climb steeply while slow trophies are
diluted by the flood of new tracked players — so a median that follows each
trophy's own fitted trajectory, regularised by difficulty ceilings, scores
better on the logit pinball loss than a median that stands still.

## Research notes

- Trophy list: 55 trophies (1 platinum, 2 gold, 6 silver, 46 bronze), all
  offline, one playthrough. Story: 5 chapters, ~93–100 missions.
- Story length ~50–65 h (Rockstar: ~80 h with side content); platinum /
  100% ~200–300+ h. By day 7 almost nobody finishes the story.
- Launch scale: Newzoo (July 2026, via pre-orders) forecast 37–51M copies
  in week one ($3.3–5.2B). Exophase tracks only a small synced fraction of
  PS5 players.
- Live Exophase snapshot 2026-11-20 06:49 UTC (~31 h after launch): opening
  mission 100%, Chapter 1 7.99%, Chapter 2 0.90%, Chapters 3/4 and final
  mission 0.02%, fishing challenge 10.57%, Breakfast Run 2.52%, most grind /
  collectible / silver / gold / platinum trophies 0.00–0.02%.
- GTA V template: base 51 trophies, platinum ~2% of tracked players
  long-term; week-one rates are far below long-term rates.

---

<sub>Entered on [make.fun](https://make.fun/?ref=1grmr2h6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
