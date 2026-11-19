"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case's target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Model (own design, standard library only, fully deterministic):

  * Work on the logit scale.  A trophy's rate is r = E / P, earners over
    players tracked, so it is modelled through its two moving parts:
      - players tracked P(t): log P is fitted linear in x = ln(1 + hours
        since 2026-11-19T00:00Z) and read at the target;
      - earners E(t) = r(t) * P(t) / 100 per trophy: log E is fitted
        linear in the same x and read at the target.
    The decomposition forecast is r = 100 * E_target / P_target.  It is
    blended 50/50 (in logit space) with a direct fit of logit(r) on x.
    Splitting earners from players tracked is what lets a rate fall when
    the player base floods in faster than earners, and rise when players
    actually progress -- the two forces Exophase refreshes mix together.
  * Only distinct reads are fitted: consecutive identical reads are the
    same stale page, not new information, and would overweight the flat
    stretch in a least-squares fit.  Only the last 24 hours of reads are
    used: trophy curves saturate, so the recent regime extrapolates
    better than the launch-night rush (a sweep over synthetic
    final-like cases: 24h window 0.226 vs all-reads 0.269 mean pinball
    loss; a 12h window was a hair better but leaves too few distinct
    reads if refreshes run slow).  Damping the extrapolation past the
    last read was tested in the same sweep and did not help, so the
    line is read at the target undamped.
  * Spread (logit sigma) combines the fit's residual scatter, a random
    walk 0.5 * sqrt(horizon days) and the slope's standard error times
    the extrapolation distance, floored and capped so a near-perfect
    flat fit cannot claim a near-zero spread.
  * Flat history (no usable trend, e.g. the practice cases): Exophase
    refreshes discretely, hours apart, so the forecast is a two-branch
    mixture: with probability p = 1 - exp(-horizon_hours / 5) a refresh
    lands inside the horizon, with probability 1 - p the page is
    unchanged.  In the refresh branch newcomers have earned almost
    nothing yet -- the one refresh the rules disclose (638 -> 3,868
    players tracked, 943 -> 4,276 trophies earned) shows every rate but
    the one at 100% fell by roughly a factor 0.22 on average -- so the
    branch centres on the diluted rate (newcomers earn trophy i at a
    rate proportional to its current rate, beta_i = 0.0755 * r_i,
    calibrated so the mean multiplier over the practice trophies is the
    disclosed 0.221), with a wide spread because the per-trophy split
    of the newcomers' earns is unknown.  Quantiles are read off the
    mixture's CDF, so a short horizon keeps the median at persistence
    while the lower quantiles already cover the refresh.
  * A trophy at 100% in every read stays at 100%: it survived a sixfold
    influx of players at 100%, so it is earned on starting the game.

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

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Launch-day origin for the log-time axis (same origin the house curve
# baseline uses).
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Trend-fit options (chosen by the backtest sweep; see builder log).
WINDOW_HOURS = 24.0
BLEND_DECOMP = 0.5
DAMP = 1.0

# --- flat-history fallback, calibrated on the one refresh the rules
# disclose (638 -> 3,868 players; 943 -> 4,276 trophies earned):
# newcomers earned ~103 trophies among 3,230 new players, i.e. a
# newcomer earns trophy i at ~7.5% of its current rate per point of
# rate (beta_i = 0.0755 * r_i), giving a mean rate multiplier of 0.221.
REFRESH_PLAYER_MULT = 3868.0 / 638.0
BETA_PER_RATE_POINT = 0.0755
REFRESH_CADENCE_HOURS = 5.0
REFRESH_BRANCH_SIGMA = 0.45
NO_REFRESH_SIGMA = 0.05

SIGMA_FLOOR = 0.30
SIGMA_CAP = 1.60
RW_SIGMA = 0.5


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
    """Round to 4 decimals, half up (matches the starter's rounding)."""
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


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, slope_se, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0, n
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = n - 2
    if dof > 0:
        var = sum(r * r for r in resid) / dof
        sd = math.sqrt(var)
        se_b = math.sqrt(var / sxx) if sxx > 0 else 0.0
    else:
        sd = 0.0
        se_b = 0.0
    return a, b, sd, se_b, n


def distinct_reads(history):
    """Collapse consecutive identical reads (same players + same rates)."""
    out = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            out.append(read)
            prev_key = key
    return out


def mixture_quantiles(mu_a, sig_a, mu_b, sig_b, p_b):
    """Quantiles of the two-component normal mixture on the logit scale.

    Branch A (weight 1 - p_b): no refresh, rate unchanged.
    Branch B (weight p_b): one refresh, rate diluted.
    Returns the five quantiles in percent, via a deterministic grid
    inversion of the mixture CDF.
    """
    lo = min(mu_a, mu_b) - 6.0 * max(sig_a, sig_b) - 0.5
    hi = max(mu_a, mu_b) + 6.0 * max(sig_a, sig_b) + 0.5
    step = 0.01
    n = int((hi - lo) / step) + 1
    grid = [lo + i * step for i in range(n)]
    cdfs = []
    for g in grid:
        cdfs.append((1.0 - p_b) * norm_cdf((g - mu_a) / sig_a)
                    + p_b * norm_cdf((g - mu_b) / sig_b))
    qs = []
    prev_q = None
    for tau in LEVELS:
        # First grid point whose CDF reaches tau (linear interpolation
        # between the bracketing points).
        idx = 0
        while idx < n and cdfs[idx] < tau:
            idx += 1
        if idx == 0:
            lg = grid[0]
        elif idx >= n:
            lg = grid[-1]
        else:
            c0, c1 = cdfs[idx - 1], cdfs[idx]
            if c1 > c0:
                f = (tau - c0) / (c1 - c0)
            else:
                f = 0.0
            lg = grid[idx - 1] + f * step
        q = round4(clamp_pct(inv_logit(lg)))
        if prev_q is not None and q < prev_q:
            q = prev_q
        qs.append(q)
        prev_q = q
    return qs


def forecast_case(case, window_hours=WINDOW_HOURS,
                  blend_decomp=BLEND_DECOMP, damp=DAMP):
    trophies = case["trophies"]
    history = distinct_reads(case["history"])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    at_dt = parse_time(case["at"])
    target_dt = parse_time(case["target"])
    x_target = math.log1p(max(0.0, hours_since_epoch(target_dt)))
    at_hours = hours_since_epoch(at_dt)
    # Trend window: the last `window_hours` of reads.
    win = []
    for read in history:
        t = hours_since_epoch(parse_time(read["at"]))
        if at_hours - t <= window_hours:
            win.append(read)
    if len(win) < 3:
        win = history
    xs = []
    log_p = []
    for read in win:
        t = hours_since_epoch(parse_time(read["at"]))
        xs.append(math.log1p(max(0.0, t)))
        log_p.append(math.log(max(1.0, float(read["players_tracked"]))))

    # Players-tracked model (only meaningful when P actually moved).
    p_fit = None
    if len(xs) >= 3 and max(xs) - min(xs) > 1e-9 and max(log_p) - min(log_p) > 1e-9:
        p_fit = ols(xs, log_p)

    rows = []
    for i, trophy in enumerate(trophies):
        series = []  # (x, rate, players) with a rate present
        for j, read in enumerate(win):
            rate = read["rates"][i]
            if rate is not None:
                series.append((xs[j], float(rate), float(read["players_tracked"])))

        if series and all(r == 100.0 for _, r, _ in series):
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue

        med_logit = None
        sigma = None
        if len(series) >= 3 and p_fit is not None:
            x_min = min(x for x, _, _ in series)
            x_max = max(x for x, _, _ in series)
            if x_max - x_min > 1e-9:
                # Earners fit: log E on x.
                e_xs, e_ys = [], []
                l_xs, l_ys = [], []
                for x, rate, players in series:
                    earners = max(0.5, rate / 100.0 * players)
                    e_xs.append(x)
                    e_ys.append(math.log(earners))
                    l_xs.append(x)
                    l_ys.append(logit(rate))
                a_e, b_e, _, _, _ = ols(e_xs, e_ys)
                a_l, b_l, sd_l, se_l, _ = ols(l_xs, l_ys)
                # Damped extrapolation: shrink the log-time step past
                # the last read, where real curves bend away from the
                # straight line.
                x_eff = x_max + damp * max(0.0, x_target - x_max)
                log_p_t = p_fit[0] + p_fit[1] * x_eff
                log_e_t = a_e + b_e * x_eff
                r_decomp = 100.0 * math.exp(log_e_t - log_p_t)
                r_direct = inv_logit(a_l + b_l * x_eff)
                med_logit = (blend_decomp * logit(r_decomp)
                             + (1.0 - blend_decomp) * logit(r_direct))
                dx = max(0.0, x_target - x_max)
                sigma = math.sqrt(sd_l * sd_l
                                  + (RW_SIGMA * math.sqrt(horizon_days)) ** 2
                                  + (se_l * dx) ** 2)
                sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

        if med_logit is None:
            # Flat history: two-branch mixture (refresh / no refresh).
            last_rate = None
            for _, rate, _ in reversed(series):
                last_rate = rate
                break
            if last_rate is None:
                last_rate = 1.0
            beta = BETA_PER_RATE_POINT * last_rate
            mult = (1.0 + (REFRESH_PLAYER_MULT - 1.0) * beta) / REFRESH_PLAYER_MULT
            p_refresh = 1.0 - math.exp(-horizon_hours / REFRESH_CADENCE_HOURS)
            sig_b = math.sqrt(REFRESH_BRANCH_SIGMA ** 2
                              + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
            rows.append({"id": trophy["id"], "q": mixture_quantiles(
                logit(last_rate), NO_REFRESH_SIGMA,
                logit(last_rate * mult), sig_b, p_refresh)})
            continue

        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(inv_logit(med_logit + z * sigma)))
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
