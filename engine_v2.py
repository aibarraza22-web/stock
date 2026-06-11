"""
Monte Carlo scenario engine v2 — wide-funnel finalists.
Run date: 2026-06-11. Horizon: 20 trading days, close 2026-06-11 -> close 2026-07-10
(Jul 3 market holiday observed). Target: P(finish >= +10%).

FUNNEL (honest accounting):
  Stage 1: published market-wide screens (monthly gainer rankings, most-shorted
    lists, FDA/PDUFA calendars, weekly earnings calendars, unusual-options
    reports) — these rank the full ~5,000-name US listed universe; coverage of
    high-potential sectors (semis/AI hardware, optical, energy, uranium,
    consumer discretionary, biotech) far exceeds 500 names, but the per-name
    features are the publishers', not ours (sandbox network policy blocks bulk
    price data: Stooq/Yahoo/Wikipedia all unreachable).
  Stage 2: ~30 surfaced candidates -> hard constraints (NYSE/Nasdaq, >$5,
    ADDV >= $25M, no fresh SPAC/IPO, >=6mo history) + snippet-verifiable
    forward signals -> 6 finalists. Biotech binaries EXCLUDED only because
    specific in-window PDUFA tickers were unretrievable, not by choice.

MODEL: as engine.py (t jump-diffusion + regimes + catalyst jumps + FOMC layer),
  plus: per-name drift range (down-momentum allowed), event-date uncertainty
  (CCL's earnings date is only estimated June 23-30 -> drawn per ensemble
  member), quarter-end (Jun 30, day 13) minor vol bump.
"""

import json
import numpy as np

rng = np.random.default_rng(11)

N_DAYS = 20
FOMC_DAY = 4          # Jun 17
QTR_END_DAY = 13      # Jun 30
N_PATHS = 25_000
N_ENS = 48
HAIRCUT_RANGE = (0.10, 0.20)
DF_RANGE = (3.5, 8.0)

CALIB = {
    # MU $1,040 (SOURCED, stale +-4%). Earnings Wed Jun 24 AMC (SOURCED) ->
    # reaction day 10. Implied earnings move ASSUMED 9-15% (not retrievable).
    "MU": dict(s0_unc=0.04, event_day=(10, 10), implied_move=(0.09, 0.15),
               p_up=(0.45, 0.62), ann_vol=(0.55, 0.85), crush=(0.6, 0.8),
               mom_prob=0.60, rev_prob=0.15, mu_mom=(0.000, 0.004),
               bg_lambda=0.06, bg_sigma=0.05, jump_widener=(1.2, 1.6)),
    # AAOI ~$197 (SOURCED Jun 5-8, choppy +-5%). 15x in 12mo, 12% off ATH,
    # high short interest. NO scheduled event in window (Q2 report ~Aug).
    "AAOI": dict(s0_unc=0.05, event_day=None, implied_move=None, p_up=None,
                 ann_vol=(0.85, 1.30), crush=(1.0, 1.0),
                 mom_prob=0.45, rev_prob=0.25, mu_mom=(-0.002, 0.005),
                 bg_lambda=0.10, bg_sigma=0.08),
    # SMCI $33.87 (SOURCED Jun 10 close, -16.7% on $7B dilution news).
    # 18% short interest. Post-shock: bounce vs. continued slide both live.
    "SMCI": dict(s0_unc=0.03, event_day=None, implied_move=None, p_up=None,
                 ann_vol=(0.70, 1.00), crush=(1.0, 1.0),
                 mom_prob=0.30, rev_prob=0.35, mu_mom=(-0.004, 0.003),
                 bg_lambda=0.08, bg_sigma=0.06),
    # UEC $9.45 (SOURCED Jun 10). Earnings PASSED Jun 9 (+13.6% then faded);
    # 50%+ below Jan ATH. High-vol, no in-window catalyst. Control name.
    "UEC": dict(s0_unc=0.03, event_day=None, implied_move=None, p_up=None,
                ann_vol=(0.55, 0.80), crush=(1.0, 1.0),
                mom_prob=0.20, rev_prob=0.35, mu_mom=(-0.003, 0.003),
                bg_lambda=0.05, bg_sigma=0.05),
    # CCL $26.06 (SOURCED Jun ~10). Earnings date only ESTIMATED Jun 23-30
    # -> event day drawn 9-14 per member. Implied move ASSUMED 6-9%.
    # Last reaction -4.3%; prior reactions mixed.
    "CCL": dict(s0_unc=0.02, event_day=(9, 14), implied_move=(0.06, 0.09),
                p_up=(0.40, 0.55), ann_vol=(0.35, 0.50), crush=(0.65, 0.85),
                mom_prob=0.25, rev_prob=0.25, mu_mom=(0.000, 0.003),
                bg_lambda=0.03, bg_sigma=0.04),
    # CRK price NOT CONFIRMED -> normalized; % outputs only. Q1 missed (-10%),
    # but 35x call volume buy-side flow flag (Schwab, ~Jun 10) + Texas power
    # narrative. No scheduled event in window.
    "CRK": dict(s0_unc=0.04, event_day=None, implied_move=None, p_up=None,
                ann_vol=(0.45, 0.70), crush=(1.0, 1.0),
                mom_prob=0.35, rev_prob=0.25, mu_mom=(0.000, 0.004),
                bg_lambda=0.06, bg_sigma=0.05),
}

def u(rng, rg):
    return rng.uniform(*rg)

def simulate_member(p, rng):
    df = u(rng, DF_RANGE)
    tnorm = np.sqrt(df / (df - 2.0))
    vol_d = u(rng, p["ann_vol"]) / np.sqrt(252)
    s0_shift = rng.normal(0.0, p["s0_unc"] / 2)

    ev = p["event_day"]
    ev_day = int(rng.integers(ev[0], ev[1] + 1)) if ev is not None else None
    if ev_day is not None:
        im = u(rng, p["implied_move"]) * (1 - u(rng, HAIRCUT_RANGE))
        p_up = u(rng, p["p_up"])
        w = u(rng, p.get("jump_widener", (1.0, 1.0)))
        m_up, s_up = im * 0.95, im * 0.55 * w
        m_dn, s_dn = im * 1.05, im * 0.60 * w
        crush = u(rng, p["crush"])

    mom_p, rev_p = p["mom_prob"], p["rev_prob"]
    mu_mom = u(rng, p["mu_mom"])
    k_rev = 0.10

    n = N_PATHS
    logret = np.full(n, s0_shift)
    state = rng.choice(3, size=n, p=[mom_p, 1 - mom_p - rev_p, rev_p])
    for day in range(1, N_DAYS + 1):
        stay = rng.random(n) < 0.85
        redraw = rng.choice(3, size=n, p=[mom_p, 1 - mom_p - rev_p, rev_p])
        state = np.where(stay, state, redraw)

        v = vol_d if (ev_day is None or day <= ev_day) else vol_d * crush
        volv = np.where(state == 0, v * 1.15, np.where(state == 2, v * 1.05, v))
        mu = np.where(state == 0, mu_mom, 0.0) - np.where(state == 2, k_rev * logret, 0.0)

        r = mu + volv * (rng.standard_t(df, size=n) / tnorm)
        nj = rng.random(n) < p["bg_lambda"]
        r += np.where(nj, rng.normal(0, p["bg_sigma"], size=n), 0.0)

        if day == FOMC_DAY:
            r += rng.normal(0, 0.004, size=n)
            mshock = rng.random(n) < 0.05
            r += np.where(mshock, rng.normal(0, 0.025, size=n), 0.0)
        if day == QTR_END_DAY:
            r += rng.normal(0, 0.003, size=n)

        if ev_day is not None and day == ev_day:
            up = rng.random(n) < p_up
            jump = np.where(up,
                            np.abs(rng.normal(m_up, s_up, size=n)),
                            -np.abs(rng.normal(m_dn, s_dn, size=n)))
            r += np.log1p(jump)
            state = np.where(up & (rng.random(n) < 0.45), 0, state)
            state = np.where(~up & (rng.random(n) < 0.45), 2, state)

        logret += r

    return np.expm1(logret)

def run(name):
    p = CALIB[name]
    pool, ups, dns = [], [], []
    for _ in range(N_ENS):
        s = simulate_member(p, rng)
        ups.append(np.mean(s >= 0.10))
        dns.append(np.mean(s <= -0.10))
        pool.append(s)
    allp = np.concatenate(pool)
    ups, dns = np.array(ups), np.array(dns)
    return dict(
        ticker=name,
        median=float(np.median(allp)), mean=float(np.mean(allp)),
        p5=float(np.percentile(allp, 5)), p95=float(np.percentile(allp, 95)),
        p_up10=float(np.median(ups)),
        p_up10_band=[float(np.percentile(ups, 10)), float(np.percentile(ups, 90))],
        p_up20=float(np.mean(allp >= 0.20)),
        p_dn10=float(np.median(dns)),
        p_dn10_band=[float(np.percentile(dns, 10)), float(np.percentile(dns, 90))],
        cvar5=float(np.mean(allp[allp <= np.percentile(allp, 5)])),
        n_paths=int(N_ENS * N_PATHS),
    )

if __name__ == "__main__":
    out = {}
    for t in CALIB:
        r = run(t)
        out[t] = r
        print(f"{t:5s} med={r['median']:+.1%} 5-95=[{r['p5']:+.1%},{r['p95']:+.1%}] "
              f"P(+10%)={r['p_up10']:.1%} [{r['p_up10_band'][0]:.1%},{r['p_up10_band'][1]:.1%}] "
              f"P(+20%)={r['p_up20']:.1%} P(-10%)={r['p_dn10']:.1%} CVaR5={r['cvar5']:+.1%}")
    with open("results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
