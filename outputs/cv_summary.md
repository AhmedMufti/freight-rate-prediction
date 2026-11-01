# CV summary

MAE in dollars. Folds are expanding-window temporal splits; F3 is the *selection fold* (decisions were made on it, so it is not an untouched holdout). `gap` = Oct-only test with the Jan-Aug fit (proxy for the 1-month gap before December).

## Final configuration (config/decisions.json)

| model | F1 | F2 | F3 | mean MAE | sd | gap(Oct) | mean RMSE | mean MAPE % | mean MedAPE % | mean WAPE % | mean R2 | unseen-city MAE | random 5-fold MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0_global_rpm | 241.8 | 249.0 | 257.0 | 249.3 | 6.2 | 253.2 | 676.9 | 11.56 | 8.04 | 10.35 | 0.802 | nan | nan |
| b1_equip_distband_rpm | 201.3 | 134.8 | 134.9 | 157.0 | 31.3 | 138.8 | 644.7 | 6.84 | 4.14 | 6.50 | 0.820 | 130.4 | nan |
| b2_lane_equip_rpm | 222.4 | 175.4 | 150.9 | 182.9 | 29.6 | 154.8 | 719.5 | 7.86 | 4.22 | 7.58 | 0.775 | 159.3 | nan |
| ridge_loglog | 200.2 | 113.5 | 109.9 | 141.2 | 41.7 | 112.2 | 641.6 | 5.86 | 3.33 | 5.84 | 0.822 | 104.6 | nan |
| lgbm | 180.9 | 106.9 | 98.4 | 128.7 | 37.0 | 99.3 | 635.5 | 5.40 | 2.88 | 5.32 | 0.825 | 92.7 | 117.7 |

## Market-index range per fold (daily MI): fit window vs test window

| fold | fit MI min..max (mean) | test MI min..max (mean) |
|---|---|---|
| F1 | 0.80..1.37 (1.05) | 1.18..1.40 (1.29) |
| F2 | 0.80..1.40 (1.13) | 0.84..1.35 (1.09) |
| F3 | 0.80..1.40 (1.12) | 0.77..1.07 (0.93) |
| validation (Nov-Dec) | train Jan-Oct 0.77..1.40 | 0.82..1.04 (0.93) |

## Decision stage: seeds

Seed sd of mean MAE = 0.744% over MAEs [163.54, 164.5, 165.98]; gate = 1.487%.

## Decision stage: outlier_target

| variant | mean MAE | F1 | F2 | F3 | gap | gain vs incumbent % | decision |
|---|---|---|---|---|---|---|---|
| keep|log_rpm | 134.3 | 186.0 | 111.0 | 105.9 | 106.7 | +nan | baseline |
| keep|log_rate | 145.1 | 198.2 | 120.2 | 117.0 | 118.4 | -8.07 | reject |
| drop|log_rpm | 131.9 | 187.6 | 107.4 | 100.6 | 102.2 | +1.79 | ADOPT |
| drop|log_rate | 136.9 | 192.3 | 111.6 | 106.7 | 108.2 | -3.78 | reject |
| weight|log_rpm | 131.6 | 187.3 | 106.8 | 100.7 | 102.0 | +0.22 | reject |
| weight|log_rate | 137.1 | 192.9 | 111.6 | 106.7 | 108.2 | -3.93 | reject |

Winner: **drop|log_rpm**. 
- keep|log_rate vs keep|log_rpm: gain -8.07% (gate 1.49%), F3 117.0 vs 105.9, gap 118.4 vs 106.7 -> reject
- drop|log_rpm vs keep|log_rpm: gain +1.79% (gate 1.49%), F3 100.6 vs 105.9, gap 102.2 vs 106.7 -> ADOPT
- drop|log_rate vs drop|log_rpm: gain -3.78% (gate 1.49%), F3 106.7 vs 100.6, gap 108.2 vs 102.2 -> reject
- weight|log_rpm vs drop|log_rpm: gain +0.22% (gate 1.49%), F3 100.7 vs 100.6, gap 102.0 vs 102.2 -> reject
- weight|log_rate vs drop|log_rpm: gain -3.93% (gate 1.49%), F3 106.7 vs 100.6, gap 108.2 vs 102.2 -> reject

## Decision stage: market

| variant | mean MAE | F1 | F2 | F3 | gap | gain vs incumbent % | decision |
|---|---|---|---|---|---|---|---|
| no_qs_no_mi | 136.9 | 193.3 | 111.1 | 106.3 | 108.2 | +nan | baseline |
| no_qs | 129.8 | 133.8 | 110.6 | 145.0 | 143.7 | +5.19 | reject |
| no_mi | 156.8 | 237.7 | 136.0 | 96.7 | 103.2 | -14.50 | reject |
| full | 163.5 | 236.7 | 129.7 | 124.3 | 129.0 | -19.44 | reject |
| qs_dev | 169.9 | 236.4 | 125.1 | 148.3 | 136.1 | -24.10 | reject |

Winner: **no_qs_no_mi**. 
- no_qs vs no_qs_no_mi: gain +5.19% (gate 1.49%), F3 145.0 vs 106.3, gap 143.7 vs 108.2 -> reject
- no_mi vs no_qs_no_mi: gain -14.50% (gate 1.49%), F3 96.7 vs 106.3, gap 103.2 vs 108.2 -> reject
- full vs no_qs_no_mi: gain -19.44% (gate 1.49%), F3 124.3 vs 106.3, gap 129.0 vs 108.2 -> reject
- qs_dev vs no_qs_no_mi: gain -24.10% (gate 1.49%), F3 148.3 vs 106.3, gap 136.1 vs 108.2 -> reject

## Decision stage: recency

| variant | mean MAE | F1 | F2 | F3 | gap | gain vs incumbent % | decision |
|---|---|---|---|---|---|---|---|
| h=0 | 134.3 | 193.8 | 107.6 | 101.4 | 104.0 | +nan | baseline |
| h=3 | 130.8 | 181.5 | 109.7 | 101.4 | 102.1 | +2.56 | ADOPT |
| h=6 | 131.9 | 187.6 | 107.4 | 100.6 | 102.2 | -0.80 | reject |

Winner: **h=3**. 
- h=3 vs h=0: gain +2.56% (gate 1.49%), F3 101.4 vs 101.4, gap 102.1 vs 104.0 -> ADOPT
- h=6 vs h=3: gain -0.80% (gate 1.49%), F3 100.6 vs 101.4, gap 102.2 vs 102.1 -> reject

## Decision stage: tune

tune: incumbent nl63_mcs40_l2 mean 130.84; best grid nl31_mcs80_huber mean 128.72 (gain +1.62%, gate 1.49%, F3 98.4 vs 101.4, gap 99.3 vs 102.1) -> ADOPT; n_final = round(1.1 * median inner iters [286, 160, 1980]) = 315

## Current decisions

```
{
 "outlier_mode": "drop",
 "target": "log_rpm",
 "qs_variant": "none",
 "use_mi": false,
 "mi_date_mean": false,
 "recency_halflife": 3,
 "lgbm_params": {
  "learning_rate": 0.03,
  "num_leaves": 31,
  "min_child_samples": 80,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.8,
  "bagging_freq": 1,
  "lambda_l2": 1.0,
  "objective": "huber"
 },
 "n_final": 315,
 "seed_sd_mae_pct": 0.7437,
 "gate_pct": 1.4874,
 "dec_qs_method": "none",
 "dec_mi_smoothing": false
}
```
