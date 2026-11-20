"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at a case's target time.

Research behind the model (see README.md for sources):
  - 55 trophies (1 platinum, 2 gold, 6 silver, 46 bronze), all offline.
  - Main story ~50-65 h (Rockstar: ~80 h with side content); platinum / 100%
    ~200-300+ h. By day 7 almost nobody finishes the story, so late-story
    and grind trophies stay near 0% while early-story trophies rise fastest.
  - Launch week forecast 37-51M copies (Newzoo, Jul 2026). Exophase tracks
    only a small synced fraction of PS5 players, and that tracked count grows
    fast (638 -> 3,868 players in the first refresh on launch night), which
    dilutes the rates of slow trophies while early trophies' earners grow.
  - Live Exophase snapshot 2026-11-20 06:49 UTC (~31 h after launch):
    opening mission 100%, Chapter 1 7.99%, Chapter 2 0.90%, Chapters 3/4 and
    final mission 0.02%, fishing challenge 10.57%, Breakfast Run 2.52%,
    most grind / collectible / silver / gold / platinum trophies 0.00-0.02%.
  - GTA V template: platinum ~2% of tracked players long-term, first-mission
    trophy eventually very common; week-one rates are far below long-term.

Model (deterministic, standard library only, no network):
  1. Dedupe consecutive identical reads -- Exophase refreshes in steps, so a
     history can hold many copies of one page (the practice histories are
     7-8 identical reads of the 638-player page).
  2. Work in logit space. For each trophy fit weighted least squares
     logit(rate) ~ a + b * ln(1 + hours since launch 2026-11-19T00:00Z)
     over the distinct reads, weights rising toward the last read.
  3. Shrink the fitted slope toward a category prior (story chapters rise
     fastest in logit space, grind / platinum stay ~flat). The shrinkage
     weight grows when the trend is weakly identified (few distinct reads,
     short x-span). This is what beats the house baseline: the baseline's
     median is the stale last rate, while week-one rates move a lot over
     the final's 4.5-day horizon.
  4. Median at target = last rate carried forward along the shrunk slope,
     blended with a category prior, then clamped to a per-category day-7
     ceiling. Near-zero trophies stay near zero: on the logit scale a
     forecast of 0.1% for a true 0.01% trophy costs ~2.3 logit units, so
     restraint matters as much as growth.
  5. Spread: sigma_logit = 0.35 + 0.35 * sqrt(horizon_days), plus 0.25 when
     the last rate is < 0.05% (small-count noise) and 0.2 when the trend is
     weakly identified, capped at 1.3. The opening mission gets a tiny
     spread. Quantiles are inv-logit(logit(median) + z_tau * sigma) with
     z = [-1.645, -0.674, 0, 0.674, 1.645], rounded to 4 decimals half-up
     (the starter's convention).
  6. Dilution branch: if the history holds a single distinct read and the
     checkpoint is within ~12 h of launch (the practice shape), forecast a
     refresh drop (median ~0.35x the last rate for non-opening trophies)
     instead of persistence. The house baseline's practice failure is
     exactly this refresh: every rate fell when Exophase's tracked count
     jumped 638 -> 3,868.

Run: python3 main.py, reading $MAKEFUN_INPUT_DIR (else /input) and writing
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Grand Theft Auto VI released 2026-11-19 (UTC). All time is measured from
# the start of launch day; progress is concave in clock time, so the trend
# is fitted against ln(1 + hours since launch).
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# A rate Exophase prints as 0.00% is below its 0.01% print floor; treat it
# as half the floor when a logit is needed. The scorer clamps rates to
# 0.01..99.99 before the logit, so a forecast below 0.01% is scored as 0.01%.
PRINT_FLOOR = 0.01
ZERO_RATE = 0.005


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def hours_since_launch(t: datetime) -> float:
    return max(0.0, (t - LAUNCH).total_seconds() / 3600.0)


def x_of(t: datetime) -> float:
    """Concave clock: ln(1 + hours since launch)."""
    return math.log1p(hours_since_launch(t))


# ---------------------------------------------------------------------------
# Trophy categories, from the trophy's tier and description. Each category
# carries a prior logit-slope (per unit of x) and a day-7 ceiling (percent):
# the highest earn rate the trophy can plausibly reach by 2026-11-26, given
# a 50-65 h story, a 200-300+ h platinum, and the live day-1 rates above.
# ---------------------------------------------------------------------------

def classify(trophy: dict) -> tuple[str, float, float]:
    """Return (category, prior_slope, day7_ceiling_pct)."""
    tier = trophy.get("tier", "bronze")
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    tid = trophy.get("id", 0)

    if tier == "platinum":
        return ("platinum", 0.0, 0.005)
    if "100% completion" in desc:
        return ("completion", 0.0, 0.02)
    if desc.startswith("complete chapter"):
        stage = {"1": 1, "2": 2, "3": 3, "4": 4}.get(desc[-2:-1], 1)
        # Chapter 1 is reached in the first hours of play; later chapters
        # need tens of hours, so their logit rises faster once players
        # arrive, but from a much lower base and ceiling.
        ceilings = {1: 30.0, 2: 15.0, 3: 8.0, 4: 4.0}
        slopes = {1: 2.2, 2: 2.8, 3: 3.4, 4: 3.8}
        return (f"chapter{stage}", slopes[stage], ceilings[stage])
    if "final mission for" in desc:
        return ("stranger", 1.5, 1.2)
    if "complete the final mission" in desc:
        return ("final", 4.2, 2.0)
    if desc.startswith("complete ") and desc.endswith(".") and tid in (2, 8, 9, 10, 11, 12, 13):
        # Named story missions (Breakfast Run, The Cartel Heist, ...). The
        # opening mission is id 2; the first named mission is early story.
        if tid == 2:
            return ("opening", 0.0, 99.9)
        ceilings = {8: 8.0, 9: 3.5, 10: 3.0, 11: 2.5, 12: 2.0, 13: 1.5}
        return ("mission", 2.0, ceilings.get(tid, 2.5))
    if "mission strands for strangers" in desc:
        return ("stranger", 1.2, 0.4)
    if "250 stunts" in desc:
        return ("grind", 0.2, 0.02)
    if "gold medal in all shooting range" in desc:
        return ("grind", 0.2, 0.05)
    if "drone race at every location" in desc:
        return ("grind", 0.2, 0.05)
    if "win 20 different races" in desc:
        return ("grind", 0.5, 0.3)
    if "collectible card sets" in desc:
        return ("grind", 0.3, 0.1)
    if "waink it at 100" in desc:
        return ("grind", 0.3, 0.2)
    if "complete 50 challenges" in desc:
        return ("grind", 0.4, 0.4)
    if "meditate at every spot" in desc:
        return ("grind", 0.3, 0.15)
    if "explore all of leonida" in desc:
        return ("grind", 0.4, 0.3)
    if "purchase all garage" in desc:
        return ("grind", 0.3, 0.15)
    if "register 10 vehicles" in desc:
        return ("grind", 0.5, 0.4)
    if "discover 100 vehicles" in desc:
        return ("easy", 1.2, 4.0)
    if "laundering marked bills" in desc:
        return ("grind", 0.3, 0.2)
    if "spend $100,000" in desc:
        return ("grind", 0.6, 0.6)
    if "rob 5 bank vaults" in desc:
        return ("grind", 0.3, 0.1)
    if "illegal operation belonging to each gang" in desc:
        return ("grind", 0.3, 0.15)
    if "try each type of activity" in desc:
        return ("easy", 1.0, 2.0)
    if "complete an arcade game" in desc:
        return ("easy", 1.2, 4.0)
    if "fishing challenge" in desc:
        return ("easy", 1.0, 20.0)
    if "wanted level with your partner" in desc:
        return ("easy", 1.0, 2.5)
    if "rob 3 shops" in desc:
        return ("easy", 0.8, 0.6)
    if "combat focus" in desc:
        return ("easy", 1.0, 1.2)
    if "unscheduled departure" in desc:
        return ("easy", 1.0, 2.0)
    if "rhinos" in desc:
        return ("easy", 1.0, 2.0)
    if "jewelry store without being detected" in desc:
        return ("easy", 0.8, 0.5)
    if "armored truck" in desc:
        return ("easy", 0.8, 0.5)
    if "cocktail with a view" in desc:
        return ("easy", 0.8, 1.5)
    if "vehicle showroom" in desc:
        return ("easy", 0.6, 0.4)
    if "bowling alley" in desc:
        return ("easy", 1.0, 3.0)
    if "melee drive-by" in desc:
        return ("easy", 0.8, 0.6)
    # Hidden / vague trophies (ids 47-50 and anything unclassified): cheap
    # open-world moments, low ceilings, modest slopes.
    if tier == "silver":
        return ("grind", 0.4, 0.5)
    if tier == "gold":
        return ("grind", 0.2, 0.1)
    return ("misc", 0.9, 2.0)


def distinct_reads(history: list) -> list:
    """Drop consecutive reads that repeat the previous read whole."""
    out = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if out and out[-1][0] == key:
            continue
        out.append((key, read))
    return [read for _, read in out]


def fit_trend(points: list) -> tuple[float, float, float]:
    """Weighted least squares logit ~ a + b*x. Returns (a, b, x_span)."""
    n = len(points)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (points[0][1], 0.0, 0.0)
    # Weights rise toward the last read: recent pages matter most.
    weights = [0.5 + (i / max(1, n - 1)) for i in range(n)]
    sw = sum(weights)
    mx = sum(w * x for w, (x, _) in zip(weights, points)) / sw
    my = sum(w * y for w, (_, y) in zip(weights, points)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, (x, _) in zip(weights, points))
    if sxx <= 1e-9:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, (x, y) in zip(weights, points))
    b = sxy / sxx
    a = my - b * mx
    xs = [x for x, _ in points]
    return (a, b, max(xs) - min(xs))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_t = parse_time(case["target"])
    at_t = parse_time(case["at"])
    x_target = x_of(target_t)
    x_last = x_of(parse_time(history[-1]["at"]))

    reads = distinct_reads(history)
    n_distinct = len(reads)
    last = history[-1]
    last_rates = last["rates"]

    # The practice shape: one distinct read, checkpoint within ~12 h of
    # launch. Exophase is about to refresh its tracked count upward by
    # several times, diluting every rate except the opening mission's.
    # The house baseline holds the last rate and pays for it; forecast
    # the refresh drop instead.
    single_read_launch = n_distinct <= 1 and hours_since_launch(at_t) <= 12.0

    rows = []
    for i, trophy in enumerate(trophies):
        category, prior_slope, ceiling = classify(trophy)
        raw_last = last_rates[i]
        r_last = raw_last if raw_last is not None else 0.0
        r_last = max(r_last, ZERO_RATE) if r_last <= 0 else r_last
        logit_last = logit(r_last)

        if category == "opening":
            # Everybody who plays finishes the first mission; Exophase
            # already prints 100%. Median stays ~100% with a tiny spread.
            median = 99.9
            sigma = 0.05
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(logit(median) + z * sigma))
                               for z in Z]})
            continue

        if single_read_launch:
            median = min(ceiling, max(ZERO_RATE, r_last * 0.35))
            sigma = 0.55
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(logit(median) + z * sigma))
                               for z in Z]})
            continue

        # Trend points for this trophy over the distinct reads. Reads
        # before the first big refresh of the tracked count belong to a
        # different regime: the launch-night page covered only 638 players
        # (reviewers / early players, rates inflated by the tiny cohort),
        # then the count jumped several-fold and every rate fell. Once the
        # count has tripled, drop the pre-jump reads from the trend fit so
        # the refresh drop does not drag the slope down.
        points = []
        fit_reads = reads
        if len(reads) >= 4:
            base_n = reads[0].get("players_tracked") or 0
            for j, read in enumerate(reads):
                if base_n and (read.get("players_tracked") or 0) >= 3 * base_n:
                    if len(reads) - j >= 3:
                        fit_reads = reads[j:]
                    break
        for read in fit_reads:
            rate = read["rates"][i]
            if rate is None:
                continue
            rate = max(rate, ZERO_RATE) if rate <= 0 else rate
            points.append((x_of(parse_time(read["at"])), logit(rate)))
        a, b_fit, x_span = fit_trend(points)

        # How much the fitted slope can be trusted: more distinct reads
        # and a wider x-span mean more trust; the prior slope fills the gap.
        # An extreme fitted slope is not credible over a long horizon.
        trust = 0.0
        if len(points) >= 3 and x_span > 0.05:
            trust = min(0.85, 0.15 * len(points) + 0.35 * min(x_span, 1.5))
        if abs(b_fit) > 3.0:
            trust *= 0.6
        b = trust * b_fit + (1.0 - trust) * prior_slope
        # A fitted slope is never allowed to run away: week-one logit
        # slopes above ~5 per unit of ln(1+h) are not credible.
        b = max(-1.0, min(5.0, b))

        # Trend median: last rate carried along the shrunk slope. Over a
        # long horizon the slope is damped: as the tracked population
        # broadens from day-one buyers to casual buyers, per-trophy
        # progress saturates and dilution bites, so the early explosive
        # slope does not run to the target unchanged.
        dx = x_target - x_last
        damp = 1.0 if dx <= 0.8 else 0.75
        logit_trend = logit_last + b * dx * damp
        # Prior median: last rate grown at the prior slope, capped.
        logit_prior = logit_last + prior_slope * dx * damp
        w_trend = 0.35 + 0.5 * trust
        logit_med = w_trend * logit_trend + (1.0 - w_trend) * logit_prior
        # Over a long horizon a trophy's logit may not run away from its
        # last read: at most +1.5 logit units up (about 4.5x the odds) and
        # 1.0 down, however steep the early slope looked. The tracked
        # population keeps broadening to casual players, which brakes
        # every rate's rise and can dilute slow trophies.
        if dx > 0.8:
            logit_med = max(logit_last - 1.0, min(logit_last + 1.5, logit_med))
        # Ceiling: no trophy may be forecast above its day-7 ceiling, and
        # no trophy may fall below the print floor.
        median = min(inv_logit(logit_med), ceiling)
        median = max(median, ZERO_RATE)
        # A trophy already at/above its ceiling stays there (it is not
        # going to fall back below it over 4.5 days of rising play) -- but
        # only for categories whose rates genuinely rise. The launch-night
        # page's 0.63% for the platinum / 100% trophies is a tiny-cohort
        # artifact (4 of 638 players), not a rate to defend.
        if r_last >= ceiling and category in ("opening", "chapter1",
                "chapter2", "chapter3", "chapter4", "final", "mission",
                "easy", "misc"):
            median = min(max(r_last, median), 99.9)

        sigma = 0.35 + 0.35 * math.sqrt(max(horizon_days, 0.0))
        if r_last < 0.05:
            sigma += 0.25
        if trust < 0.4:
            sigma += 0.2
        sigma = min(sigma, 1.3)

        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(logit(median) + z * sigma))
                           for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
