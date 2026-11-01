r"""Market-index audit: does any way of using market_index beat the no-market model on the temporal folds?

Folds F1-F3 and gap as in splits.py, plus oct1 (fit Jan-Sep, test Oct: the closest proxy for fit-to-Oct, predict Nov-Dec).
Uses the fold runner of evaluate.py unchanged (inner-holdout early stopping, recency weights, outlier drop).
Variants: base (current decisions), mi (raw MI feature), mi_datemean, mi_mono (monotone +1 on MI),
offset_est (log-rate offset beta*log(MI), beta estimated inside the fit window from within-month daily residuals),
off_devK (offset on log MI minus its trailing K-day mean: short-run moves only), mi_dev28 (that deviation as a tree feature).
Usage (Docker): scripts\dr.ps1 -Raw python scripts/mi_audit.py [variants comma-separated] [seeds comma-separated]
Appends rows to outputs/mi_audit.csv and rewrites outputs/mi_audit.md.
"""
import sys, json, copy
import numpy as np, pandas as pd
import freight.models as FM
import freight.evaluate as E
import freight.splits as S
from freight import config as C
from freight.features import FeatureBuilder
from freight.models import LGBM, B1

S.SPECS["oct1"] = (9, (10,))
tr, va, mi_daily = E.load_clean()
rate = tr["posted_rate"]
lmi_d = np.log(mi_daily)
trail = (lmi_d - lmi_d.shift(1).rolling(28, min_periods=7).mean()).bfill()
tr["mi_dev28"] = tr["date"].map(trail).values
for w in (7, 14):
    tr[f"mi_dev{w}"] = tr["date"].map((lmi_d - lmi_d.shift(1).rolling(w, min_periods=3).mean()).bfill()).values


class FB2(FeatureBuilder):
    def __init__(self, d=None):
        super().__init__(d)
        self.extra = (d or {}).get("_extra", [])

    def _base(self, df):
        X = super()._base(df)
        for c in self.extra:
            X[c] = df[c].astype(float)
        return X


FM.FeatureBuilder = FB2


def est_beta(df, r):
    b1 = B1().fit(df, r)
    d = pd.DataFrame({"res": np.log(r.values / b1.predict(df)), "lmi": np.log(df["mi_date_mean"].values),
                      "date": df["date"].values, "month": df["month"].values})
    g = d.groupby("date").agg(res=("res", "median"), lmi=("lmi", "mean"), month=("month", "first"))
    g[["res", "lmi"]] -= g.groupby("month")[["res", "lmi"]].transform("mean")
    return float(np.polyfit(g.lmi, g.res, 1)[0])


class OffsetLGBM(LGBM):
    def fit(self, df, rate, w=None, es=None):
        if self.dec.get("_beta") is None or self.dec.get("_beta") == "est":
            self.beta = est_beta(df, rate)
        else:
            self.beta = float(self.dec["_beta"])
        return super().fit(df, rate, w, es)

    def _y(self, df, rate):
        return super()._y(df, rate) - self.beta * self._x(df)

    def _x(self, df):
        c = self.dec.get("_offcol")
        return df[c].values if c else np.log(df["mi_date_mean"].values)

    def _inv(self, yhat, df):
        return super()._inv(yhat + self.beta * self._x(df), df)


base = C.load_decisions()
cols_mi = ["distance", "equipment", "weight", "weight_missing_flag", "market_index",
           "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "circuity", "dow"]
mono = [1 if c == "market_index" else 0 for c in cols_mi]
VARS = {
    "base": ({}, LGBM),
    "mi": ({"use_mi": True}, LGBM),
    "mi_datemean": ({"use_mi": True, "mi_date_mean": True}, LGBM),
    "mi_mono": ({"use_mi": True, "lgbm_params": {**base["lgbm_params"], "monotone_constraints": mono}}, LGBM),
    "offset_est": ({"_beta": "est"}, OffsetLGBM),
    "off_dev28": ({"_beta": "est", "_offcol": "mi_dev28"}, OffsetLGBM),
    "off_dev14": ({"_beta": "est", "_offcol": "mi_dev14"}, OffsetLGBM),
    "off_dev7": ({"_beta": "est", "_offcol": "mi_dev7"}, OffsetLGBM),
    "mi_dev28": ({"_extra": ["mi_dev28"]}, LGBM),
    "offset_est_dev28": ({"_beta": "est", "_extra": ["mi_dev28"]}, OffsetLGBM),
}


def summarize():
    d = pd.read_csv(C.OUT / "mi_audit.csv")
    cols = ["F1", "F2", "F3", "gap", "oct1", "mean"]
    g = d.groupby("variant", sort=False)[cols].mean().round(1)
    g.insert(0, "seeds", d.groupby("variant", sort=False)["seed"].count())
    b = g.loc["base"]
    g["vs base mean %"] = (100 * (b["mean"] - g["mean"]) / b["mean"]).round(2)
    g["vs base oct1 %"] = (100 * (b["oct1"] - g["oct1"]) / b["oct1"]).round(2)
    lines = ["# Market-index audit (MAE $, mean over seeds; positive % = better than base)", "",
             "| variant | " + " | ".join(g.columns) + " |", "|---" * (len(g.columns) + 1) + "|"]
    lines += ["| " + k + " | " + " | ".join(str(v) for v in r.values) + " |" for k, r in g.iterrows()]
    (C.OUT / "mi_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return g


if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
    print(summarize().to_string())
    sys.exit(0)
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARS)
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 7, 123]
folds = ["F1", "F2", "F3", "oct1"]
out = []
for v in sel:
    ov, cls = VARS[v]
    dec = {**copy.deepcopy(base), **ov}
    E.LGBM = cls
    for s in seeds:
        res = {}
        for fn in folds:
            f = S.make_fold(tr, fn)
            m, info = E.fit_lgbm_window(tr, rate, f.fit, f.inner_fit, f.es, dec, s)
            p = m.predict(tr.iloc[f.test])
            y = rate.iloc[f.test].values
            res[fn] = float(np.mean(np.abs(p - y)))
            res[fn + "_it"] = info["inner_iter"]
            if fn == "F3":
                g = S.make_fold(tr, "gap")
                pos = {t: i for i, t in enumerate(f.test)}
                ix = np.array([pos[t] for t in g.test])
                res["gap"] = float(np.mean(np.abs(p[ix] - rate.iloc[g.test].values)))
            if hasattr(m, "beta"):
                res[fn + "_beta_x1000"] = round(m.beta * 1000)
        res["mean"] = float(np.mean([res[k] for k in ("F1", "F2", "F3")]))
        row = {"variant": v, "seed": s, **{k: (round(x, 1) if isinstance(x, float) else x) for k, x in res.items()}}
        print(json.dumps(row), flush=True)
        out.append(row)
P = C.OUT / "mi_audit.csv"
new = pd.DataFrame(out)
if P.exists():
    old = pd.read_csv(P)
    old = old[~old.set_index(["variant", "seed"]).index.isin(new.set_index(["variant", "seed"]).index)]
    new = pd.concat([old, new], ignore_index=True)
new.to_csv(P, index=False, encoding="utf-8", lineterminator="\n")
summarize()
