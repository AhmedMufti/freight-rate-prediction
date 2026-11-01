"""December forecast for Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb."""
from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from . import io as fio  # noqa: E402
from .evaluate import load_clean  # noqa: E402
from .train import MODEL_PATH, fit_final  # noqa: E402


def build_rows(coords, d: pd.Timestamp, mi: float, qs: np.ndarray, qs_dev: np.ndarray) -> pd.DataFrame:
    """Full feature rows for date d (one per qs draw) with the same columns as cleaned train/validation frames."""
    pu, de = C.DEC_LANE
    n = len(qs)
    df = pd.DataFrame({
        "pickup": pu, "delivery": de,
        "pickup_lat": coords.loc[pu, "lat"], "pickup_lon": coords.loc[pu, "lon"],
        "delivery_lat": coords.loc[de, "lat"], "delivery_lon": coords.loc[de, "lon"],
        "distance": C.DEC_DISTANCE, "equipment": C.DEC_EQUIP, "weight": C.DEC_WEIGHT,
        "weight_missing_flag": 0, "market_index": mi, "mi_date_mean": mi,
        "quote_signal": qs, "qs_dev": qs_dev, "dow": d.dayofweek, "date": d,
    }, index=range(n))
    df["lane"] = pu + "|" + de
    return df


def predict_days(model, coords, va, days, mi_by_day, qs_mode, qs_train):
    out = []
    dec = model.dec
    has_qs = dec["qs_variant"] != "none"
    for d in days:
        rows = va[(va["date"] == d)]
        sel = rows[rows["distance"] < 600]
        qs, dev = sel["quote_signal"].values, sel["qs_dev"].values
        if not has_qs or qs_mode == "marginal":
            pass
        elif qs_mode == "train_median":
            qs, dev = np.array([qs_train["median"]]), np.array([0.0])
        elif qs_mode == "day_median":
            qs, dev = np.array([np.median(qs)]), np.array([np.median(dev)])
        elif qs_mode == "grid":
            qs = qs_train["grid"]
            dev = qs_train["grid_dev"]
        X = build_rows(coords, d, mi_by_day[d], qs, dev)
        out.append(float(np.mean(model.predict(X))))
    return np.array(out)


def run(write_decisions: bool = False) -> dict:
    tr, va, mi_daily = load_clean()
    coords = fio.city_coords()
    dec = C.load_decisions()
    dfile = fio.load_december()
    days = pd.to_datetime(dfile["date"], format="%Y-%m-%d")
    model = joblib.load(MODEL_PATH)["model"]

    # MI per day: validation date mean (raw values, missing excluded) = mi_daily; smoothing rule
    dec_mi = mi_daily.reindex(days)
    diffs_dec = mi_daily.loc["2025-12-01":"2025-12-31"].diff().dropna()
    diffs_tr = mi_daily.loc[:"2025-10-31"].diff().dropna()
    ratio = float(diffs_dec.std() / diffs_tr.std())
    smooth = ratio > 2
    mi_use = mi_daily.rolling(3, center=True, min_periods=1).mean().reindex(days) if smooth else dec_mi
    mi_by_day = dict(zip(days, mi_use.values))

    qs_train = {"median": float(tr["quote_signal"].median())}
    qs_train["grid"] = np.quantile(tr["quote_signal"], np.linspace(0.05, 0.95, 19))
    qs_train["grid_dev"] = np.quantile(tr["qs_dev"], np.linspace(0.05, 0.95, 19))

    primary = predict_days(model, coords, va, days, mi_by_day, "marginal", qs_train)
    variants = {"primary (marginalise qs over day's <600mi loads)": primary,
                "qs: train-median plug-in": predict_days(model, coords, va, days, mi_by_day, "train_median", qs_train),
                "qs: daily band-median plug-in": predict_days(model, coords, va, days, mi_by_day, "day_median", qs_train),
                "qs: train 5-95pct grid mean": predict_days(model, coords, va, days, mi_by_day, "grid", qs_train)}
    oct_mean = float(tr.loc[tr["month"] == 10, "mi_date_mean"].mean())
    variants["MI: October mean"] = predict_days(model, coords, va, days, {d: oct_mean for d in days}, "marginal", qs_train)
    roll = mi_daily.rolling(7, min_periods=1).mean().reindex(days)
    variants["MI: 7-day rolling mean"] = predict_days(model, coords, va, days, dict(zip(days, roll.values)), "marginal", qs_train)
    nm_dec = {**dec, "qs_variant": "none", "use_mi": False}
    nm = fit_final(tr, nm_dec, n_iter=dec["n_final"])
    variants["no-market model"] = predict_days(nm, coords, va, days, mi_by_day, "marginal", qs_train)
    # market-aware alternatives (not chosen by the CV rule; shown so the reader can see what the market index would do)
    for label, ov in (("alt model: MI only", {"qs_variant": "none", "use_mi": True}),
                      ("alt model: MI + qs", {"qs_variant": "qs", "use_mi": True})):
        if (ov["qs_variant"], ov["use_mi"]) != (dec["qs_variant"], dec["use_mi"]):
            alt = fit_final(tr, {**dec, **ov}, n_iter=dec["n_final"])
            variants[label] = predict_days(alt, coords, va, days, mi_by_day, "marginal", qs_train)

    pred = np.round(primary, 2)
    assert np.isfinite(pred).all() and (pred > 0).all()
    out = dfile.copy()
    out["predicted_rate"] = [f"{x:.2f}" for x in pred]
    out.to_csv(C.DATA / "december_chart_inputs.csv", index=False, encoding="utf-8", lineterminator="\n")
    out.to_csv(C.OUT / "december_predictions.csv", index=False, encoding="utf-8", lineterminator="\n")

    sens = pd.DataFrame([{"variant": k, "mean": v.mean(), "min": v.min(), "max": v.max(),
                          "corr_with_primary": float(np.corrcoef(v, primary)[0, 1]) if np.std(v) > 0 else float("nan")}
                         for k, v in variants.items()])
    sens.round(2).to_csv(C.OUT / "december_sensitivity.csv", index=False, encoding="utf-8", lineterminator="\n")
    wide = pd.DataFrame({"date": dfile["date"], "mi": mi_use.values, **{k: np.round(v, 2) for k, v in variants.items()}})
    wide.to_csv(C.OUT / "december_daily_variants.csv", index=False, encoding="utf-8", lineterminator="\n")

    corr_mi = float(np.corrcoef(primary, mi_use.values)[0, 1])
    info = {"mean": float(pred.mean()), "min": float(pred.min()), "max": float(pred.max()),
            "corr_with_mi": corr_mi, "mi_diff_sd_ratio_dec_vs_train": ratio, "mi_smoothing": bool(smooth),
            "dec_qs_method": "marginalise" if dec["qs_variant"] != "none" else "none"}
    (C.OUT / "december_summary.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    sanity_plot(tr, days, pred, mi_use.values, C.OUT / "december_sanity.png")
    if write_decisions:
        C.set_decision("dec_mi_smoothing", bool(smooth), f"Dec MI day-to-day sd is {ratio:.2f}x the train sd (rule: smooth if > 2x)")
        C.set_decision("dec_qs_method", info["dec_qs_method"],
                       "qs marginalised over the day's validation loads under 600 mi (Jensen) when qs is a feature")
    return info


def sanity_plot(tr, days, pred, mi, path):
    h = tr[(tr["pickup"] == C.DEC_LANE[0]) & (tr["delivery"] == C.DEC_LANE[1]) & (tr["equipment"] == C.DEC_EQUIP)]
    scaled = h["posted_rate"] * C.DEC_DISTANCE / h["distance"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(h["market_index"], scaled, s=22, c="#4C78A8", label=f"history, Jan-Oct (n={len(h)}), scaled to 360 mi")
    ax[0].scatter(mi, pred, s=22, c="#F58518", label="December predictions")
    ax[0].set_xlabel("market index"); ax[0].set_ylabel("rate, $ (360 mi, Dry Van)")
    ax[0].set_title("Lexington to Fort Wayne vs market index"); ax[0].legend(fontsize=8)
    ax[1].plot(days, pred, c="#F58518", marker="o", ms=3)
    ax[1].axhspan(scaled.min(), scaled.max(), color="#4C78A8", alpha=0.15, label="historic range")
    ax[1].set_title("December daily prediction"); ax[1].tick_params(axis="x", rotation=45); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
