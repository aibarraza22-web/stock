"""
Monte Carlo scenario engine — 10-trading-day horizon, P(+10%) screen.
Run date: 2026-06-11. Window: close 2026-06-11 -> close 2026-06-25
(trading days: Jun 12, 15, 16, 17, 18, 19, 22, 23, 24, 25).

MODEL
  Daily log-return r_t = mu(state) + sigma(state) * eps_t + J_bg + J_event
    eps_t  : Student-t(df) innovation, variance-normalized (fat diffusive tails)
    J_bg   : background Poisson jump (news shocks), lambda per day, N(0, s_bg)
    J_event: scheduled-catalyst jump, fired ONCE on the known event day:
             direction ~ Bernoulli(p_up); magnitude |N(m_up,s_up)| or -|N(m_dn,s_dn)|
             scaled so E|J_event| ~= implied_move * (1 - HAIRCUT), since options-
             implied earnings moves historically overstate realized moves ~15-20%.
    FOMC 2026-06-17 (day 4): common macro layer — extra vol that day plus a 5%
             chance of a +-2.5%-scale macro jump (sign symmetric).
  Regime layer (per day, Markov): momentum / neutral / mean-revert.
    momentum:  drift  +mu_mom, vol x 1.15
    neutral:   drift   0,      vol x 1.00
    mean-rev:  drift  -k * cum_return_so_far (pull toward start), vol x 1.05
  Post-event: vol multiplied by CRUSH (<1) for remaining days (IV crush analog),
    and regime re-drawn conditioned on jump sign (continuation vs reversion).

UNCERTAINTY
  Two layers, reported separately:
    1. Monte Carlo SE (made small: N_PATHS per parameter draw).
    2. Parameter ensemble: every ASSUMED parameter is drawn from its stated
       range each ensemble member; the spread of P(+10%) across members is the
       honest band. This dominates, and it should.

All SOURCED vs ASSUMED parameters are tagged in CALIB below.
"""

import json
import numpy as np

rng = np.random.default_rng(7)

N_DAYS = 10
FOMC_DAY = 4            # Jun 17 is the 4th trading day of the window
N_PATHS = 25_000        # paths per ensemble member
N_ENS = 48              # parameter draws
HAIRCUT_RANGE = (0.10, 0.20)   # SOURCED: IV overstates realized earnings move 15-20%
DF_RANGE = (3.5, 8.0)          # ASSUMED: Student-t tail df for daily returns

# ---------------------------------------------------------------------------
# CALIBRATION. u(a,b) = drawn uniformly per ensemble member (ASSUMED range).
# Fixed numbers are SOURCED (see report for citations) unless marked ASSUMED.
# ---------------------------------------------------------------------------
CALIB = {
    "ADBE": dict(
        s0=234.0,            # SOURCED ~Jun 9-10 close/pre-mkt; stale +-2%
        s0_unc=0.02,
        event_day=1,         # reports Jun 11 AMC -> reaction Jun 12
        implied_move=(0.087, 0.0945),  # SOURCED: Bloomberg 8.7% / TipRanks 9.45%
        p_up=(0.35, 0.50),   # ASSUMED: AI-competition fear, negative recent reactions
        ann_vol_pre=(0.30, 0.40),      # ASSUMED (event is day 1; barely matters)
        ann_vol_post=(0.25, 0.38),     # ASSUMED post-crush
        crush=(0.55, 0.75),
        mom_prob=0.15, rev_prob=0.35,  # ASSUMED initial regime mix: downtrend name
        bg_lambda=0.03, bg_sigma=0.04,
    ),
    "MU": dict(
        s0=1040.0,           # SOURCED ~Jun 2-10 range $1010-1047; stale +-4%
        s0_unc=0.04,
        event_day=10,        # reports Jun 24 AMC -> reaction Jun 25 = final day
        implied_move=(0.09, 0.15),     # ASSUMED — could not retrieve live straddle
        p_up=(0.45, 0.62),   # ASSUMED: momentum + sold-out HBM vs. extreme expectations
        ann_vol_pre=(0.55, 0.85),      # ASSUMED from 300->1040 1-yr price action
        ann_vol_post=(0.55, 0.85),     # event on last day; post-vol unused
        crush=(0.6, 0.8),
        mom_prob=0.60, rev_prob=0.15,  # ASSUMED: strong momentum regime
        bg_lambda=0.06, bg_sigma=0.05, # AI-news shock layer
        jump_widener=(1.2, 1.6),       # NOVEL: analyst-PT-dispersion tail widener
    ),
    "ORCL": dict(
        s0=199.0,            # SOURCED: after-hours Jun 10 post-earnings
        s0_unc=0.02,
        event_day=None,      # catalyst already happened (Jun 10 AMC)
        implied_move=None,
        p_up=None,
        ann_vol_pre=(0.28, 0.45),      # ASSUMED: post-event elevated-but-crushed vol
        ann_vol_post=(0.28, 0.45),
        crush=(1.0, 1.0),
        mom_prob=0.20, rev_prob=0.40,  # ASSUMED: post-shock reversion-tilted
        bg_lambda=0.05, bg_sigma=0.05, # capex-narrative headline risk
    ),
    "LEN": dict(
        s0=100.0,            # NORMALIZED — live price not confirmed; % outputs only
        s0_unc=0.02,
        event_day=1,         # reports Jun 11 AMC -> reaction Jun 12
        implied_move=(0.043, 0.043),   # SOURCED: 4.3% options-implied
        p_up=(0.35, 0.55),   # ASSUMED: 4 straight misses but lowered bar
        ann_vol_pre=(0.25, 0.35),
        ann_vol_post=(0.20, 0.32),
        crush=(0.6, 0.8),
        mom_prob=0.15, rev_prob=0.30,
        bg_lambda=0.02, bg_sigma=0.03,
    ),
}

def u(rng, rg):
    return rng.uniform(*rg)

def simulate_member(p, rng):
    """One parameter draw -> N_PATHS 10-day paths -> stats."""
    df = u(rng, DF_RANGE)
    tnorm = np.sqrt(df / (df - 2.0))
    vol_pre = u(rng, p["ann_vol_pre"]) / np.sqrt(252)
    vol_post = u(rng, p["ann_vol_post"]) / np.sqrt(252) * u(rng, p["crush"])
    s0_shift = rng.normal(0.0, p["s0_unc"] / 2)   # stale-price uncertainty
    widener = u(rng, p.get("jump_widener", (1.0, 1.0)))

    ev_day = p["event_day"]
    if ev_day is not None:
        im = u(rng, p["implied_move"]) * (1 - u(rng, HAIRCUT_RANGE))
        p_up = u(rng, p["p_up"])
        # mixture means/sds; mean abs jump ~= im, sd gives dispersion around it
        m_up, s_up = im * 0.95, im * 0.55 * widener
        m_dn, s_dn = im * 1.05, im * 0.60 * widener

    mom_p, rev_p = p["mom_prob"], p["rev_prob"]
    mu_mom = u(rng, (0.000, 0.004))   # ASSUMED momentum drift 0-40bp/day
    k_rev = 0.10                       # mean-reversion pull strength (ASSUMED)

    n = N_PATHS
    logret = np.full(n, s0_shift)
    # initial regime: 0=mom 1=neutral 2=rev
    state = rng.choice(3, size=n, p=[mom_p, 1 - mom_p - rev_p, rev_p])
    # regime persistence 0.85, else re-draw from initial mix (ASSUMED)
    for day in range(1, N_DAYS + 1):
        stay = rng.random(n) < 0.85
        redraw = rng.choice(3, size=n, p=[mom_p, 1 - mom_p - rev_p, rev_p])
        state = np.where(stay, state, redraw)

        vol = vol_pre if (ev_day is None or day <= ev_day) else vol_post
        volv = np.where(state == 0, vol * 1.15, np.where(state == 2, vol * 1.05, vol))
        mu = np.where(state == 0, mu_mom, 0.0) - np.where(state == 2, k_rev * logret, 0.0)

        eps = rng.standard_t(df, size=n) / tnorm
        r = mu + volv * eps

        # background news jumps
        nj = rng.random(n) < p["bg_lambda"]
        r += np.where(nj, rng.normal(0, p["bg_sigma"], size=n), 0.0)

        # FOMC macro layer (common scheduled macro event)
        if day == FOMC_DAY:
            r += rng.normal(0, 0.004, size=n)
            mshock = rng.random(n) < 0.05
            r += np.where(mshock, rng.normal(0, 0.025, size=n), 0.0)

        # scheduled catalyst jump
        if ev_day is not None and day == ev_day:
            up = rng.random(n) < p_up
            jup = np.abs(rng.normal(m_up, s_up, size=n))
            jdn = -np.abs(rng.normal(m_dn, s_dn, size=n))
            jump = np.where(up, jup, jdn)
            r += np.log1p(jump)  # jump quoted in simple-return space
            # post-event regime conditioned on jump sign: winners drift, losers chop
            state = np.where(up & (rng.random(n) < 0.45), 0, state)
            state = np.where(~up & (rng.random(n) < 0.45), 2, state)

        logret += r

    simple = np.expm1(logret)
    return simple

def run(name):
    p = CALIB[name]
    members = []
    pool = []
    for _ in range(N_ENS):
        s = simulate_member(p, rng)
        members.append(dict(
            p_up10=float(np.mean(s >= 0.10)),
            p_dn10=float(np.mean(s <= -0.10)),
        ))
        pool.append(s)
    allp = np.concatenate(pool)
    ups = np.array([m["p_up10"] for m in members])
    dns = np.array([m["p_dn10"] for m in members])
    res = dict(
        ticker=name,
        median=float(np.median(allp)),
        mean=float(np.mean(allp)),
        p5=float(np.percentile(allp, 5)),
        p95=float(np.percentile(allp, 95)),
        p_up10=float(np.median(ups)),
        p_up10_band=[float(np.percentile(ups, 10)), float(np.percentile(ups, 90))],
        p_dn10=float(np.median(dns)),
        p_dn10_band=[float(np.percentile(dns, 10)), float(np.percentile(dns, 90))],
        cvar5=float(np.mean(allp[allp <= np.percentile(allp, 5)])),
        mc_se=float(np.std(ups) / np.sqrt(N_ENS)),
        n_paths=int(N_ENS * N_PATHS),
    )
    return res

if __name__ == "__main__":
    out = {}
    for t in CALIB:
        r = run(t)
        out[t] = r
        print(f"{t:5s} med={r['median']:+.1%} 5-95%=[{r['p5']:+.1%},{r['p95']:+.1%}] "
              f"P(+10%)={r['p_up10']:.1%} band[{r['p_up10_band'][0]:.1%},{r['p_up10_band'][1]:.1%}] "
              f"P(-10%)={r['p_dn10']:.1%} CVaR5={r['cvar5']:+.1%}")
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
