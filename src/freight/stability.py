"""Stability checks (reporting only): residual correlations by month, MI/qs quantiles, adversarial AUC."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .evaluate import load_clean
from .features import FeatureBuilder
from .models import B1


def run() -> dict:
    tr, va, _ = load_clean()
    rate = tr["posted_rate"]
    b1 = B1().fit(tr, rate)
    resid = np.log(rate.values / b1.predict(tr))
    qsm = tr["quote_signal"].median()
    rows = []
    for m in range(1, 11):
        k = (tr["month"] == m).values
        r = resid[k]
        q = tr.loc[k, "quote_signal"].values
        rows.append({"month": m, "n": int(k.sum()),
                     "corr_qs": np.corrcoef(r, q)[0, 1],
                     "corr_abs_qs_dev_median": np.corrcoef(r, np.abs(q - qsm))[0, 1],
                     "corr_qs_dev": np.corrcoef(r, tr.loc[k, "qs_dev"].values)[0, 1],
                     "corr_mi": np.corrcoef(r, tr.loc[k, "market_index"].values)[0, 1]})
    cor = pd.DataFrame(rows)
    qs_pos, qs_neg = int((cor.corr_qs > 0).sum()), int((cor.corr_qs < 0).sum())
    mi_pos = int((cor.corr_mi > 0).sum())

    # quantiles per month, train vs validation Nov/Dec
    q = []
    allf = pd.concat([tr[["month", "quote_signal", "market_index"]], va[["month", "quote_signal", "market_index"]]])
    for m, g in allf.groupby("month"):
        q.append({"month": m, "qs_q05": g.quote_signal.quantile(.05), "qs_q50": g.quote_signal.median(),
                  "qs_q95": g.quote_signal.quantile(.95), "mi_q05": g.market_index.quantile(.05),
                  "mi_q50": g.market_index.median(), "mi_q95": g.market_index.quantile(.95)})
    qt = pd.DataFrame(q)

    # adversarial validation: Aug-Oct train vs validation, features only
    dec = C.load_decisions()
    a = tr[tr["month"] >= 8]
    both = pd.concat([a, va], ignore_index=True)
    lab = np.r_[np.zeros(len(a)), np.ones(len(va))]
    out = {}
    for name, sub in (("all_features", None), ("without_market_cols", ["market_index", "qs"])):
        fb = FeatureBuilder({**dec, "qs_variant": "qs", "use_mi": True}).fit(a)
        X = fb.transform(both)
        if sub:
            X = X.drop(columns=sub)
        # city-level coords are kept: they reveal the 8 unseen cities (reported as such)
        oof = np.zeros(len(X))
        for tri, tei in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(X, lab):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=C.SEED,
                                   verbose=-1, n_jobs=4, deterministic=True, force_row_wise=True)
            m.fit(X.iloc[tri], lab[tri])
            oof[tei] = m.predict_proba(X.iloc[tei])[:, 1]
        out[name] = float(roc_auc_score(lab, oof))
    auc = out["all_features"]

    L = ["# Stability checks", "",
         "Residual = log(rate / b1 prediction), b1 = median rpm per (equipment, distance decile), fitted on all train.", "",
         "## Residual correlation by month", "",
         "| month | n | corr(res, qs) | corr(res, abs(qs - median)) | corr(res, qs_dev) | corr(res, MI) |", "|---|---|---|---|---|---|"]
    for _, r in cor.iterrows():
        L.append(f"| {int(r.month)} | {int(r.n)} | {r.corr_qs:+.3f} | {r.corr_abs_qs_dev_median:+.3f} | {r.corr_qs_dev:+.3f} | {r.corr_mi:+.3f} |")
    L += ["", f"Linear qs correlation: {qs_pos} positive and {qs_neg} negative months (sign flips -> unstable). "
          f"MI correlation positive in {mi_pos} of 10 months.", "",
          "## qs and MI quantiles by month (months 11-12 = validation)", "",
          "| month | qs q05 | qs q50 | qs q95 | MI q05 | MI q50 | MI q95 |", "|---|---|---|---|---|---|---|"]
    for _, r in qt.iterrows():
        L.append(f"| {int(r.month)} | {r.qs_q05:.2f} | {r.qs_q50:.2f} | {r.qs_q95:.2f} | {r.mi_q05:.2f} | {r.mi_q50:.2f} | {r.mi_q95:.2f} |")
    L += ["", "## Adversarial validation (Aug-Oct train vs validation, 5-fold LightGBM AUC)", "",
          f"- All model features: AUC {out['all_features']:.3f}",
          f"- Without market_index and qs: AUC {out['without_market_cols']:.3f}",
          f"- Interpretation: {'material shift (>0.65)' if auc > 0.65 else 'no material shift (<=0.65)'}; "
          "the coordinate columns can separate the 8 cities that only appear in validation.", ""]
    (C.OUT / "stability.md").write_text("\n".join(L), encoding="utf-8")
    res = {"qs_pos_months": qs_pos, "qs_neg_months": qs_neg, "mi_pos_months": mi_pos, "auc": out}
    (C.OUT / "stability.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
