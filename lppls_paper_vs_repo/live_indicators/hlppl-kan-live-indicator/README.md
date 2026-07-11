# HLPPL-KAN Live Bubble Indicator

HLPPL bubble score with the LPPL trajectory fitted by a Kolmogorov-Arnold network (M-LNN-KAN) instead of classical least squares.

Part of a family of live LPPLS bubble indicators (one repository per model)
built on real-data comparisons of LPPLS calibration methods. Companion
models: hlppl / hlppl-dae / hlppl-kan / hlppl-dae-kan / mono-lppls /
mono-kan / mono-dae.

## Usage

```bash
pip install -r requirements.txt
python indicator.py --csv PRICES.csv                # analyse ln(price)
python indicator.py --csv PRICES.csv --series equity  # analyse ln(price x volume)
```

The CSV can be a plain `Date,Close[,Volume]` file or a raw yfinance export
(3-row header). Output is a JSON state dump: current bubble score /
(t_c, critical level) ensemble, and for HLPPL-family models the threshold
trading action of the HLPPL paper (entry |score| >= 0.7, exit at 0.3,
reversal exit on horizon sign flips).

## Method

As the base HLPPL, but the LPPL trajectory is produced by a Kolmogorov-Arnold network (learnable B-spline activations) trained per window (PINN-style loss).

Known limitations (disclosed by design):
* the paper's news-based Hype index is proxied by abnormal trading volume,
  and Sentiment is set to 0 (no news feed in this package);
* score forecasts use ridge regression over lagged scores - a deliberately
  small stand-in for the paper's dual-stream transformer;
* DAE variants train their cleaner on synthetic LPPLS shapes at first run
  (~20 s); declining windows are mirrored before cleaning.

Generated from the research repo `rasheedovsky/projects`
(branch `claude/lppls-paper-code-compare-hfda5d`), where all validation
results (TASI, SPY, Nasdaq, Alcoa) live.
