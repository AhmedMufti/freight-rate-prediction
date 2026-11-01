"""train (fit on all Jan-Oct with n_final) and predict (validation_predictions.csv with asserts)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import io as fio
from .clean import OutlierModel
from .evaluate import load_clean, train_weights
from .models import LGBM

MODEL_PATH = C.MODELS / "model.joblib"
PRED_PATH = C.ROOT / "validation_predictions.csv"


def fit_final(tr: pd.DataFrame, dec: dict, seed=C.SEED, n_iter=None, rate=None) -> LGBM:
    rate = tr["posted_rate"] if rate is None else rate
    n = n_iter or dec["n_final"]
    assert n, "n_final missing in decisions.json: run `tune --write-decisions` first"
    om = OutlierModel().fit(tr, rate)
    w = train_weights(dec, om, tr, rate, tr["date"].max())
    k = w > 0
    return LGBM(dec, seed=seed, n_iter=n).fit(tr[k], rate[k], w[k])


def train() -> str:
    tr, _, _ = load_clean()
    dec = C.load_decisions()
    m = fit_final(tr, dec)
    C.MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": m, "decisions": dec}, MODEL_PATH)
    m.booster.save_model(str(C.MODELS / "model.txt"), num_iteration=m.best_iter)
    return f"trained lgbm on {len(tr)} rows, n_iter={m.best_iter}"


def predict() -> str:
    _, va, _ = load_clean()
    m = joblib.load(MODEL_PATH)["model"]
    p = m.predict(va)
    tpl = fio.load_template()
    out = pd.DataFrame({"load_id": va["load_id"].values, "predicted_rate": p})
    out = tpl[["load_id"]].merge(out, on="load_id", how="left")
    assert list(out.columns) == ["load_id", "predicted_rate"]
    assert len(out) == 12000 and out["load_id"].is_unique
    assert out["load_id"].iloc[0] == "TE-000001" and out["load_id"].iloc[-1] == "TE-012000"
    assert np.isfinite(out["predicted_rate"]).all() and (out["predicted_rate"] > 0).all()
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv(PRED_PATH, index=False, encoding="utf-8", lineterminator="\n")
    corr = float(np.corrcoef(out["predicted_rate"], va.set_index("load_id").loc[out["load_id"], "distance"])[0, 1])
    return f"wrote {PRED_PATH.name}: median ${out['predicted_rate'].median():.0f}, corr(pred,distance)={corr:.3f}"
