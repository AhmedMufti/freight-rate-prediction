# Freight rate prediction

Predict `posted_rate` for 12,000 Nov-Dec 2025 loads and a 31-day December chart for Lexington to Fort Wayne
(360 mi, Dry Van, 32,000 lb). LightGBM on log(rate per mile) with a leakage-safe temporal validation scheme.

## Quickstart

### Docker (what was used to produce the results)
```powershell
docker build -t freight-ml .
scripts\dr.ps1 prepare-data                 # creates data/*.csv with the brief's underscore names from raw/
scripts\dr.ps1 all                          # fixed-config pipeline (reads config/decisions.json), a few minutes
scripts\dr.ps1 -Raw python -m pytest -q -p no:cacheprovider
scripts\dr.ps1 -Raw python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

### Plain Python (3.10+)
```bash
pip install -r requirements.txt
pip install -e .            # or: export PYTHONPATH=src
python -m freight.cli all
python -m pytest -q
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

### Data layouts
The brief expects `data/train_test.csv`, `data/validation.csv`, `data/validation_predictions_template.csv`,
`data/december_chart_inputs.csv`. The originals are the hyphenated files (`train-test.csv`, ...). Put either layout
in `raw/` (or `data/`) and run `prepare-data`; it copies to the underscore names and never overwrites a filled
December file. `data/` and `raw/` are git-ignored.

## Pipeline commands (`python -m freight.cli <cmd>`)
`prepare-data`, `quality`, `eda`, `stability`, `cv`, `cv --sweep {outlier_target,market,recency} --write-decisions`,
`cv --seeds 3 --write-decisions`, `tune --write-decisions`, `train`, `predict`, `december`, `score`, `figures`,
`report`, `manifest`, `all`. `all` is a fixed-config run: sweeps and tuning are one-off steps whose outcome is
committed in `config/decisions.json` (each choice has a `_why` with the CV evidence).

## Validation approach
Expanding-window temporal CV with 2-month test windows (F1 May-Jun, F2 Jul-Aug, F3 Sep-Oct), a gap fold (Oct only),
an unseen-city fold (8 train cities masked) and a random 5-fold contrast reported only to show its optimism. All
transforms are fitted inside the fit window; boosting rounds come from an inner holdout (last fit month).
Decisions are sequential (outliers/target, market features, recency, tuning) and use the rule
"mean MAE gain >= gate and F3 not worse and gap fold not worse", with gate = max(0.5%, 2 x seed noise).
F3 is a selection fold, not an untouched holdout.

## Results
Temporal CV MAE in dollars (full table with RMSE/MAPE/WAPE/R2 in `outputs/cv_summary.md`):

| model | F1 | F2 | F3 (selection) | mean | gap (Oct) | unseen-city |
|---|---|---|---|---|---|---|
| b0 global rate/mile | 241.8 | 249.0 | 257.0 | 249.3 | 253.2 | n/a |
| b1 equipment x distance band | 201.3 | 134.8 | 134.9 | 157.0 | 138.8 | 130.4 |
| b2 lane + equipment | 222.4 | 175.4 | 150.9 | 182.9 | 154.8 | 159.3 |
| ridge (log-log) | 200.2 | 113.5 | 109.9 | 141.2 | 112.2 | 104.6 |
| **LightGBM (final)** | 180.9 | 106.9 | 98.4 | **128.7** | 99.3 | 92.7 |

Mean MAPE 5.4% (b0 11.6%). Random 5-fold gives 117.7, i.e. optimistic versus temporal CV.
Chosen configuration (`config/decisions.json`): log rate-per-mile target, label outliers dropped from fitting only,
no market_index / quote_signal inputs (the CV rule rejected both: the market index helps early folds but hurts F3 and the gap fold; the quote signal alone helps F3 and the gap fold but badly hurts early folds, so mean MAE is worse),
recency half-life 3 months, Huber objective, 31 leaves, min_child_samples 80, 315 rounds.
December (Lexington to Fort Wayne): mean $836, range $827-$843, weekly shape from day of week.

Market-index re-check (`scripts\dr.ps1 -Raw python scripts/mi_audit.py`, results in `outputs/mi_audit.md`, 3 seeds):
raw, date-mean, monotone and log-offset MI improve F1/F2 but make F3 26-30% and an Oct-only fold fitted on Jan-Sep
11-15% worse (MI level is confounded with an upward rate drift); short-run-only MI offsets are neutral. MI stays out.

## Layout
```
src/freight/   config io clean features splits models evaluate stability train december eda figures report manifest cli
config/decisions.json   chosen settings with evidence
tests/         pytest suite
outputs/       cv_summary.md, stability.md, data_quality.json, december_*.csv, figures/, eda/
reports/       report.docx, loom_script.md, candidate_december.png
validation_predictions.csv, Dockerfile, scripts/dr.ps1
```

## Notes
`score.py` and `raw/` are untouched. Nothing is pushed by the pipeline: push to GitHub and record the Loom yourself
(script in `reports/loom_script.md`).
