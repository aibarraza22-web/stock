"""
Forecast scoring & calibration tracker.

Workflow (each new session, or manually):
  1. For any row in forecasts.csv whose window_end has passed and whose
     resolved_close is blank, look up the OFFICIAL closing price for the
     ticker on window_end (and on forecast_date if ref_price_est is blank
     or flagged unconfirmed — realized_ret must be close-to-close).
  2. Enter it in resolved_close (or realized_ret directly for rows with no
     reference price). Then run:  python3 score_forecasts.py
  3. The script fills realized_ret, outcomes, Brier scores, and prints a
     calibration report. It never overwrites a hand-entered value.

Scoring:
  outcome_up10 = 1 if realized_ret >= +0.10 else 0   (same for dn10)
  brier_up10   = (p_up10 - outcome_up10)^2
     - 0.0 is perfect, 0.25 is what always-saying-50% scores, lower = better.
  The calibration report buckets forecasts by predicted probability and
  compares predicted vs. realized frequency. Meaningful only after ~10+
  resolved forecasts; before that, treat everything as anecdote.
"""

import csv
import sys

PATH = "forecasts.csv"

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def main():
    with open(PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = rows[0].keys() if rows else []

    resolved, pending = [], []
    for r in rows:
        ret = f(r["realized_ret"])
        if ret is None:
            close, ref = f(r["resolved_close"]), f(r["ref_price_est"])
            if close is not None and ref:
                ret = close / ref - 1.0
                r["realized_ret"] = f"{ret:.4f}"
        if ret is None:
            pending.append(r)
            continue
        up, dn = int(ret >= 0.10), int(ret <= -0.10)
        r["outcome_up10"], r["outcome_dn10"] = str(up), str(dn)
        r["brier_up10"] = f"{(f(r['p_up10']) - up) ** 2:.4f}"
        resolved.append(r)

    with open(PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Resolved: {len(resolved)}   Pending: {len(pending)}")
    for r in pending:
        print(f"  pending: run{r['run_id']} {r['ticker']:5s} window ends {r['window_end']}"
              + ("  [needs realized_ret entered directly]" if not f(r["ref_price_est"]) else ""))
    if not resolved:
        return

    briers = [f(r["brier_up10"]) for r in resolved]
    hits = sum(int(r["outcome_up10"]) for r in resolved)
    print(f"\nMean Brier (up10): {sum(briers)/len(briers):.4f}  "
          f"(0.25 = no-skill baseline of always saying 50%)")
    print(f"+10% events: {hits}/{len(resolved)}  "
          f"avg predicted P: {sum(f(r['p_up10']) for r in resolved)/len(resolved):.1%}")

    # calibration buckets
    buckets = {}
    for r in resolved:
        b = int(f(r["p_up10"]) * 100 // 10) * 10
        buckets.setdefault(b, []).append(int(r["outcome_up10"]))
    print("\nCalibration (predicted bucket -> realized frequency):")
    for b in sorted(buckets):
        o = buckets[b]
        print(f"  {b}-{b+10}%: predicted, realized {sum(o)}/{len(o)} = {sum(o)/len(o):.0%}")
    if len(resolved) < 10:
        print("\nNOTE: fewer than 10 resolved forecasts — calibration stats are "
              "anecdotes, not evidence. Do not draw conclusions yet.")

if __name__ == "__main__":
    sys.exit(main())
