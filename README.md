# Stock probability research scaffolding

Monte Carlo scenario engines plus a forecast log that keeps the engines honest.
**Research scaffolding only — not financial advice.** The engine's most common
correct answer is "no edge found."

## Files

| File | What it is |
|---|---|
| `engine.py` | Run 1 (2026-06-11): 10-trading-day window, ADBE / MU / ORCL / LEN |
| `results.json` | Run 1 output |
| `engine_v2.py` | Run 2 (2026-06-11): 20-trading-day window, wide-funnel finalists MU / AAOI / SMCI / UEC / CCL / CRK |
| `results_v2.json` | Run 2 output |
| `forecasts.csv` | **The forecast log.** Every probability the engine has put on record, with resolution columns |
| `score_forecasts.py` | Fills outcomes/Brier scores once closes are entered; prints calibration report |

## Model (both engines)

Student-t jump-diffusion (df 3.5–8) + 3-state regime layer (momentum / neutral
/ mean-revert) + explicitly modeled scheduled-catalyst jumps sized from
options-implied earnings moves (haircut 10–20%) + FOMC macro-shock layer.
Every assumed parameter is drawn from a stated range across a 48-member
ensemble, so probabilities are reported as bands. All SOURCED vs ASSUMED
parameters are tagged in the `CALIB` dict of each engine.

## Resolving forecasts (do this after each window closes)

1. After a row's `window_end` date passes, look up the ticker's official
   closing price on `window_end` and enter it in `resolved_close`.
   Rows flagged `unconfirmed` in `ref_price_flag` (LEN, CRK) have no usable
   reference price — compute close-to-close return from `forecast_date` to
   `window_end` yourself and enter it in `realized_ret` directly.
2. Run `python3 score_forecasts.py` — it fills outcomes and Brier scores and
   prints a calibration report. It never overwrites hand-entered values.
3. Commit the updated `forecasts.csv`.

Upcoming resolution dates: **2026-06-25** (run 1) and **2026-07-10** (run 2).

## Known limitations (read before trusting anything)

- Built in a network-restricted sandbox: no options chains, no bulk price
  data, no short-interest feed. Several parameters (MU implied move, AAOI/SMCI
  vol, CRK price) are flagged estimates.
- Nothing is backtested yet. The forecast log exists precisely to accumulate
  the track record that would justify (or kill) trust in the engine.
  Under ~10 resolved forecasts, calibration output is anecdote, not evidence.
- Reference prices were 0–9 days stale at forecast time; resolution should use
  official closes for both endpoints where possible.
