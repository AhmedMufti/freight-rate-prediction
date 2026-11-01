"""EDA figures and summary json."""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from .clean import OutlierModel  # noqa: E402
from .evaluate import load_clean  # noqa: E402

COL = {"Dry Van": "#4C78A8", "Reefer": "#54A24B", "Flatbed": "#F58518"}


def _save(fig, name):
    C.EDA.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.EDA / name, dpi=90)
    plt.close(fig)


def run() -> dict:
    tr, va, mi = load_clean()
    tr = tr.copy()
    tr["rpm"] = tr["posted_rate"] / tr["distance"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(np.log10(tr["posted_rate"]), bins=60, color="#4C78A8")
    ax[0].set_xlabel("log10 posted_rate ($)"); ax[0].set_title("Target distribution")
    ax[1].hist(tr["distance"], bins=60, color="#54A24B"); ax[1].set_xlabel("distance (mi)"); ax[1].set_title("Distance")
    _save(fig, "eda_1_target_distance.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for e, g in tr.sample(8000, random_state=C.SEED).groupby("equipment"):
        ax.scatter(g["distance"], g["posted_rate"], s=3, alpha=.4, c=COL[e], label=e)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("distance (mi)"); ax.set_ylabel("posted_rate ($)")
    ax.set_title("Rate vs distance (log-log)"); ax.legend(markerscale=4)
    _save(fig, "eda_2_rate_vs_distance.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    eq = [e for e in C.EQUIP]
    ax[0].boxplot([tr.loc[tr.equipment == e, "rpm"].clip(upper=8) for e in eq], tick_labels=eq, showfliers=False)
    ax[0].set_title("Rate per mile by equipment")
    bands = pd.cut(tr["distance"], [0, 300, 600, 1000, 1500, 2200, 3500])
    ax[1].plot([str(b) for b in tr.groupby(bands, observed=True)["rpm"].median().index],
               tr.groupby(bands, observed=True)["rpm"].median().values, marker="o")
    ax[1].set_title("Median rate per mile by distance band"); ax[1].tick_params(axis="x", rotation=30)
    _save(fig, "eda_3_rpm_equipment_distance.png")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(mi.index, mi.values, c="#4C78A8"); ax.axvline(pd.Timestamp("2025-11-01"), c="k", ls="--")
    ax.set_title("Market index, daily mean (dashed line = start of validation)"); ax.set_ylabel("MI")
    _save(fig, "eda_4_market_index_daily.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    dec_ = pd.qcut(tr["quote_signal"], 10)
    ax[0].plot(range(10), tr.groupby(dec_, observed=True)["rpm"].median().values, marker="o", c="#E45756")
    ax[0].set_xlabel("quote_signal decile"); ax[0].set_ylabel("median rpm"); ax[0].set_title("qs vs rpm: U-shape")
    ax[1].hist(tr["quote_signal"], bins=60, alpha=.6, label="train", density=True, color="#4C78A8")
    ax[1].hist(va["quote_signal"], bins=60, alpha=.6, label="validation", density=True, color="#F58518")
    ax[1].legend(); ax[1].set_title("quote_signal distribution")
    _save(fig, "eda_5_quote_signal.png")

    raw_w = pd.read_csv(C.ROOT / "data" / "train_test.csv", usecols=["weight"])["weight"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(raw_w.dropna(), bins=80, color="#72B7B2")
    ax.set_title(f"Weight: {int((raw_w < 0).sum())} negative, {int(raw_w.isna().sum())} missing, {int((raw_w.abs() >= 47500).sum())} at 47,500 cap")
    ax.set_xlabel("raw weight (lb)")
    _save(fig, "eda_6_weight_quality.png")

    cities = pd.concat([tr[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["c", "lat", "lon"], axis=1),
                        va[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["c", "lat", "lon"], axis=1)]).drop_duplicates("c")
    seen = set(tr.pickup) | set(tr.delivery)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    k = cities.c.isin(seen)
    ax.scatter(cities[k].lon, cities[k].lat, c="#4C78A8", label="in train")
    ax.scatter(cities[~k].lon, cities[~k].lat, c="#E45756", marker="*", s=120, label="validation only")
    ax.legend(); ax.set_title("City coordinates (8 cities are validation-only)")
    _save(fig, "eda_7_cities.png")

    om = OutlierModel().fit(tr, tr["posted_rate"])
    ratio = np.log10(tr["posted_rate"].values / om.expected(tr))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(ratio, bins=100, color="#B279A2"); ax.axvline(np.log10(3), c="k", ls="--"); ax.axvline(-np.log10(3), c="k", ls="--")
    ax.set_title("log10(rate / lane-equipment median): outliers beyond dashed lines"); ax.set_xlabel("log10 ratio")
    _save(fig, "eda_8_label_outliers.png")

    s = {"rows_train": len(tr), "rows_val": len(va),
         "corr_loglog": float(np.corrcoef(np.log(tr.posted_rate), np.log(tr.distance))[0, 1]),
         "rpm_median_by_equipment": tr.groupby("equipment")["rpm"].median().round(3).to_dict(),
         "mi_lag1_autocorr": float(mi.autocorr(1)),
         "mi_sd_within_date": float(tr.groupby("date")["market_index"].std().mean()),
         "mi_sd_between_dates": float(mi.std()),
         "train_rows_per_day": float(tr.groupby("date").size().mean()),
         "val_rows_per_day": float(va.groupby("date").size().mean())}
    (C.OUT / "eda_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s
