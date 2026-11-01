# Stability checks

Residual = log(rate / b1 prediction), b1 = median rpm per (equipment, distance decile), fitted on all train.

## Residual correlation by month

| month | n | corr(res, qs) | corr(res, abs(qs - median)) | corr(res, qs_dev) | corr(res, MI) |
|---|---|---|---|---|---|
| 1 | 4918 | +0.109 | +0.060 | +0.105 | +0.044 |
| 2 | 4337 | +0.108 | +0.053 | +0.099 | +0.058 |
| 3 | 5036 | +0.147 | +0.116 | +0.133 | +0.094 |
| 4 | 4819 | -0.094 | +0.041 | -0.082 | +0.091 |
| 5 | 4913 | -0.099 | +0.061 | -0.095 | +0.049 |
| 6 | 4783 | +0.135 | +0.130 | +0.121 | +0.053 |
| 7 | 4912 | -0.100 | +0.060 | -0.086 | +0.106 |
| 8 | 4759 | +0.001 | +0.025 | +0.002 | +0.050 |
| 9 | 4670 | +0.136 | +0.105 | +0.126 | +0.033 |
| 10 | 4853 | -0.110 | +0.085 | -0.104 | +0.078 |

Linear qs correlation: 6 positive and 4 negative months (sign flips -> unstable). MI correlation positive in 10 of 10 months.

## qs and MI quantiles by month (months 11-12 = validation)

| month | qs q05 | qs q50 | qs q95 | MI q05 | MI q50 | MI q95 |
|---|---|---|---|---|---|---|
| 1 | 1.73 | 2.03 | 2.55 | 0.80 | 0.93 | 1.04 |
| 2 | 1.76 | 2.06 | 2.57 | 0.87 | 0.99 | 1.13 |
| 3 | 1.81 | 2.14 | 2.70 | 0.95 | 1.06 | 1.20 |
| 4 | 1.46 | 2.00 | 2.32 | 1.04 | 1.21 | 1.37 |
| 5 | 1.39 | 1.96 | 2.27 | 1.19 | 1.30 | 1.41 |
| 6 | 1.92 | 2.25 | 2.86 | 1.17 | 1.27 | 1.40 |
| 7 | 1.42 | 1.96 | 2.29 | 0.97 | 1.21 | 1.35 |
| 8 | 1.69 | 2.05 | 2.42 | 0.86 | 0.98 | 1.10 |
| 9 | 1.82 | 2.16 | 2.72 | 0.77 | 0.89 | 1.02 |
| 10 | 1.42 | 1.99 | 2.30 | 0.84 | 0.96 | 1.06 |
| 11 | 1.69 | 2.06 | 2.42 | 0.81 | 0.91 | 1.03 |
| 12 | 1.69 | 2.05 | 2.41 | 0.82 | 0.93 | 1.05 |

## Adversarial validation (Aug-Oct train vs validation, 5-fold LightGBM AUC)

- All model features: AUC 0.807
- Without market_index and qs: AUC 0.560
- Interpretation: material shift (>0.65); the coordinate columns can separate the 8 cities that only appear in validation.
