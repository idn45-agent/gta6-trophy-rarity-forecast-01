"""GTA VI trophy rarity forecaster, v2 (gta6-trophy-rarity-forecast-01).

For every trophy in a case, forecast its Exophase earn rate at the
case's target as five quantiles (5/25/50/75/95 percent), scored by
pinball loss on the logit scale.  Python 3.13, standard library only,
fully deterministic: the same case always yields the same bytes.

Two regimes, chosen per case from its history:

TREND (the history actually moves).  A rate is earners over players
tracked, r = E / P, and the two parts move for different reasons:
players flood in on their own schedule while earners accumulate as
people play.  So on a linear axis x = days since 2026-11-19T00:00Z
three recency-weighted least-squares lines are fitted over the last
18 hours of distinct reads -- log P, log E per trophy, and logit(r)
directly -- and the forecast median is a blend, in logit space, of
the decomposition 100 * E_target / P_target with the direct line read
at the target.  The blend is what lets a rate fall when the player
base grows faster than its earners and rise when progression outruns
the flood.  The spread adds the direct fit's residual scatter, a
random walk over the horizon, and the slope's standard error
stretched over the extrapolation distance, floored and capped.  (The
linear axis is deliberate: on synthetic worlds of every shape tried
-- rising, diluting, saturating and power-law growth -- it
extrapolates the 4.5-day horizon far better than the log-time axis
the house curve baseline uses, which bends the wrong way whenever
player growth is anywhere near exponential in real time.)

FLAT (the page never changed in the recorded history).  Exophase
refreshes a game's page on its own cycle and every read between
refreshes repeats the same stale page, so the question is whether a
refresh lands before the target.  Refresh cycles are near-periodic,
not memoryless: a page already stale for hours is *due*, so the
refresh probability comes from a renewal calculation -- average over
a prior on the cycle length T of the forward-recurrence probability
given the page's observed age -- instead of a Poisson hazard, which
badly understates it (the launch-night page sat unchanged for over
four hours and then refreshed; a memoryless model still gives that
event only ~26% in the next 90 minutes).  If a refresh lands, the
player count jumps and rates dilute: newcomers have barely earned
anything, so trophy i's post-refresh rate is centred on its current
rate times (1 + beta_i) / M, with M the player multiplier and beta_i
the newcomers' earns for trophy i over its current earners.  The one
refresh the rules disclose (638 -> 3,868 players tracked, 943 ->
4,276 trophies earned, so 103 new earns spread over the 54 trophies
not already at 100%) pins the newcomers' total earn rate; beta_i
averages three plausible splits of that total (proportional to r^2,
to r, and concentrated on the already-common trophies), and the
branch spread is wide because the true split is unknown.  Quantiles
are read off the two-branch mixture CDF by bisection.  A trophy at
100% in every read stays at 100%: it survived a sixfold influx of
new players at 100%, so it is earned by starting the game.

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_OF_LEVEL = (-1.6448536269514722, -0.6744897501960817, 0.0,
              0.6744897501960817, 1.6448536269514722)

# Time axis origin: the game's release day.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- trend options (chosen by the synthetic-world sweep; see README)
WINDOW_HOURS = 18.0        # reads older than this are not fitted
BLEND_DECOMP = 0.5         # weight of the decomposition in the blend
DAMP = 1.0                 # shrink on the extrapolation step in x
WEIGHT_TAU_HOURS = 6.0     # recency weighting time constant (hours)
RW_SIGMA = 0.35            # random-walk sigma per sqrt(day)
SIGMA_FLOOR = 0.30
SIGMA_CAP = 1.60

# --- flat options ---------------------------------------------------
# Refresh cycle prior: the cycle length T (hours) is averaged over a
# grid with weights peaked where the one disclosed cycle sits (the
# page was stale from before 2026-11-18T23:48Z until its refresh by
# 2026-11-19T04:30Z, a spell of at least 4.2 hours).
CYCLE_GRID = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0)
CYCLE_WEIGHTS = (0.4, 0.7, 1.0, 1.2, 1.2, 1.1, 1.0, 0.7, 0.5, 0.35)
# The disclosed refresh: players 638 -> 3,868, and of the 3,333 new
# earns 3,230 belong to the trophy already at 100%, leaving 103 earns
# among 3,230 newcomers over the other 54 trophies.
REFRESH_PLAYER_MULT = 3868.0 / 638.0
NEW_EARNS_PER_NEWCOMER = 103.0 / 3230.0
REFRESH_SIGMA = 0.35       # spread of the refresh branch (logit)
FLAT_SIGMA = 0.05          # spread if no refresh: the page is the page


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the starter's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_epoch(dt):
    return (dt - EPOCH).total_seconds() / 3600.0


def axis_x(t_hours):
    """Trend-fit axis: days since the epoch (linear in real time)."""
    return max(0.0, t_hours) / 24.0


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, resid_sd, slope_se).  Weights are pre-normalised by
    the caller; resid_sd and slope_se use the effective sample size so
    a heavily down-weighted tail cannot fake a tight fit.
    """
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    n_eff = sw * sw / sum(w * w for w in ws)
    dof = max(1.0, n_eff - 2.0)
    var = sum(w * r * r for r, w in zip(resid, ws)) / sw * (n_eff / dof)
    return a, b, math.sqrt(max(0.0, var)), math.sqrt(max(0.0, var) / sxx)


def distinct_reads(history):
    """Collapse consecutive identical reads: a repeated read is the
    same stale page, not a new observation."""
    out = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            out.append(read)
            prev_key = key
    return out


def refresh_probability(age_hours, horizon_hours):
    """P(at least one refresh within the horizon), renewal model.

    The page's current state has been in place for `age_hours` (a
    lower bound: it was already there at the first recorded read if
    the history never changes).  For a cycle of length T the age is
    uniform on [age, T] given no refresh was seen, so the residual
    life is uniform on [0, T - age] and P(residual <= h) is
    min(1, h / (T - age)); cycles shorter than the observed age would
    already have fired, so for them the refresh is overdue (p = 1 is
    NOT used -- instead those cycles are dropped and the remaining
    weight renormalised, since a cycle that short is refuted by the
    flat history itself).
    """
    num = 0.0
    den = 0.0
    for t, w in zip(CYCLE_GRID, CYCLE_WEIGHTS):
        if t <= age_hours:
            continue
        num += w * min(1.0, horizon_hours / (t - age_hours))
        den += w
    if den <= 0:
        return 0.95
    return num / den


def expected_refreshes(age_hours, horizon_hours):
    """Expected number of refreshes inside the horizon (first one at
    its expected residual life, then one per cycle)."""
    num = 0.0
    den = 0.0
    mean_t = 0.0
    wsum = 0.0
    for t, w in zip(CYCLE_GRID, CYCLE_WEIGHTS):
        mean_t += w * t
        wsum += w
        if t <= age_hours:
            continue
        resid = 0.5 * (t - age_hours)
        n = 1.0 + max(0.0, (horizon_hours - resid) / t)
        num += w * n
        den += w
    if den <= 0:
        return max(1.0, horizon_hours / max(1.0, mean_t / wsum))
    return num / den


def newcomer_betas(rates, players_now):
    """beta_i = (newcomers' earns of trophy i) / (current earners).

    Averages three splits of the newcomers' total earns (proportional
    to r^2, to r, and concentrated on trophies already at >= 1.10%),
    each normalised to the disclosed total earn rate per newcomer.
    Trophies at 100% are excluded: every newcomer earns those too,
    which is exactly why their rate does not dilute.
    """
    idx = [i for i, r in enumerate(rates) if r is not None and r < 100.0]
    betas = [0.0] * len(rates)
    if not idx or players_now <= 0:
        return betas
    newcomers = (REFRESH_PLAYER_MULT - 1.0) * players_now
    total_new = newcomers * NEW_EARNS_PER_NEWCOMER

    def split(weight_of):
        ws = [weight_of(rates[i]) for i in idx]
        sw = sum(ws)
        if sw <= 0:
            return {i: 0.0 for i in idx}
        return {i: total_new * w / sw for i, w in zip(idx, ws)}

    parts = [split(lambda r: r * r),
             split(lambda r: r),
             split(lambda r: r if r >= 1.10 else 0.0)]
    for i in idx:
        earners_now = max(0.5, rates[i] / 100.0 * players_now)
        betas[i] = sum(p[i] for p in parts) / 3.0 / earners_now
    return betas


def mixture_quantiles(mu_stay, sig_stay, mu_ref, sig_ref, p_ref):
    """The five quantiles of a two-branch normal mixture on the logit
    scale, by bisection on its CDF (monotone, so this is exact)."""
    lo = min(mu_stay, mu_ref) - 8.0 * max(sig_stay, sig_ref) - 0.5
    hi = max(mu_stay, mu_ref) + 8.0 * max(sig_stay, sig_ref) + 0.5

    def cdf(g):
        return ((1.0 - p_ref) * norm_cdf((g - mu_stay) / sig_stay)
                + p_ref * norm_cdf((g - mu_ref) / sig_ref))

    qs = []
    prev = None
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        q = round4(clamp_pct(inv_logit(0.5 * (a + b))))
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def trend_forecast(case, win, trophies):
    """Fit the three lines over the window's distinct reads and read
    them at the target.  Returns a list of (median_logit, sigma) per
    trophy, or None where a trophy has no usable series."""
    target_hours = hours_since_epoch(parse_time(case["target"]))
    x_target = axis_x(target_hours)
    horizon_days = case["horizon_minutes"] / 1440.0

    xs, ts, log_p = [], [], []
    for read in win:
        t = hours_since_epoch(parse_time(read["at"]))
        ts.append(t)
        xs.append(axis_x(t))
        log_p.append(math.log(max(1.0, float(read["players_tracked"]))))
    if WEIGHT_TAU_HOURS:
        ws = [math.exp(-(ts[-1] - t) / WEIGHT_TAU_HOURS) for t in ts]
    else:
        ws = [1.0] * len(ts)
    p_fit = None
    if len(xs) >= 3 and max(xs) - min(xs) > 1e-9 and max(log_p) - min(log_p) > 1e-9:
        p_fit = wls(xs, log_p, ws)

    out = []
    for i in range(len(trophies)):
        pts = [(xs[j], ts[j], float(win[j]["rates"][i]),
                float(win[j]["players_tracked"]), ws[j])
               for j in range(len(win)) if win[j]["rates"][i] is not None]
        med = None
        sigma = None
        if len(pts) >= 3 and p_fit is not None:
            x_min = min(p[0] for p in pts)
            x_max = max(p[0] for p in pts)
            if x_max - x_min > 1e-9:
                e_xs = [p[0] for p in pts]
                e_ys = [math.log(max(0.5, p[2] / 100.0 * p[3])) for p in pts]
                l_ys = [logit(p[2]) for p in pts]
                pws = [p[4] for p in pts]
                a_e, b_e, _, _ = wls(e_xs, e_ys, pws)
                a_l, b_l, sd_l, se_l = wls(e_xs, l_ys, pws)
                # Read the lines at the target, optionally shrinking
                # the step past the last read (damping the bend).
                x_eff = x_max + DAMP * max(0.0, x_target - x_max)
                r_decomp = 100.0 * math.exp((a_e + b_e * x_eff)
                                            - (p_fit[0] + p_fit[1] * x_eff))
                r_direct = inv_logit(a_l + b_l * x_eff)
                med = (BLEND_DECOMP * logit(r_decomp)
                       + (1.0 - BLEND_DECOMP) * logit(r_direct))
                dx = max(0.0, x_target - x_max)
                sigma = math.sqrt(sd_l * sd_l
                                  + (RW_SIGMA * math.sqrt(horizon_days)) ** 2
                                  + (se_l * dx) ** 2)
                sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        out.append((med, sigma))
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = distinct_reads(case["history"])
    at_dt = parse_time(case["at"])
    at_hours = hours_since_epoch(at_dt)
    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0

    # Trend window: the last WINDOW_HOURS of distinct reads; fall back
    # to the whole history when the window holds too few.
    win = [r for r in history
           if at_hours - hours_since_epoch(parse_time(r["at"])) <= WINDOW_HOURS]
    if len(win) < 3:
        win = history
    trend = trend_forecast(case, win, trophies) if len(win) >= 3 else None

    # Flat fallback facts: the current page state's age (the last
    # distinct read is when this state first appeared) and its rates.
    last = history[-1] if history else None
    if last is not None:
        age = at_hours - hours_since_epoch(parse_time(last["at"]))
        age = max(0.0, age)
        players_now = float(last["players_tracked"])
        last_rates = [None if r is None else float(r) for r in last["rates"]]
    else:
        age, players_now, last_rates = 0.0, 0.0, [None] * len(trophies)
    p_ref = refresh_probability(age, horizon_hours)
    n_ref = expected_refreshes(age, horizon_hours)
    betas = newcomer_betas(last_rates, players_now) if last else [0.0] * len(trophies)
    sig_ref = math.sqrt(REFRESH_SIGMA ** 2
                        + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)

    rows = []
    for i, trophy in enumerate(trophies):
        series_rates = [r["rates"][i] for r in history if r["rates"][i] is not None]
        if series_rates and all(float(r) == 100.0 for r in series_rates):
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        med = sigma = None
        if trend is not None:
            med, sigma = trend[i]
        if med is None:
            rate = last_rates[i] if last_rates[i] is not None else 1.0
            # Compound the dilution over the expected refresh count:
            # the first refresh dilutes by (1 + beta) / M, later ones
            # by less as the base grows, so the exponent is shrunk.
            exponent = 1.0 + 0.5 * max(0.0, min(2.0, n_ref - 1.0))
            mult = ((1.0 + betas[i]) / REFRESH_PLAYER_MULT) ** exponent
            rows.append({"id": trophy["id"], "q": mixture_quantiles(
                logit(rate), FLAT_SIGMA,
                logit(rate * mult), sig_ref, p_ref)})
            continue
        qs = []
        prev = None
        for z in Z_OF_LEVEL:
            q = round4(clamp_pct(inv_logit(med + z * sigma)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
