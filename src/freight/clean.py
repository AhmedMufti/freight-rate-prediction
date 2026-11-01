"""Cleaning: pure functions applied identically to train, validation and December."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CAP = 47500.0
BASE_COLS = ["load_id", "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
             "distance", "equipment", "weight", "date", "market_index", "quote_signal"]


def schema_check(df: pd.DataFrame, labelled: bool) -> None:
    need = BASE_COLS + (["posted_rate"] if labelled else [])
    assert list(df.columns) == need, f"unexpected columns: {list(df.columns)}"
    assert (df["distance"] > 0).all(), "distance must be > 0"
    assert set(df["equipment"]) <= set(C.EQUIP), "unknown equipment"
    pd.to_datetime(df["date"], format="%Y-%m-%d")


def mi_daily_series(*dfs: pd.DataFrame) -> pd.Series:
    """Daily mean of the non-missing market index over all supplied files (MI is a feature, not a label)."""
    allv = pd.concat([d[["date", "market_index"]] for d in dfs])
    allv["date"] = pd.to_datetime(allv["date"])
    s = allv.groupby("date")["market_index"].mean()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(idx).interpolate(limit_direction="both")


def weight_medians(df: pd.DataFrame) -> dict:
    w = df["weight"].abs()
    return w.groupby(df["equipment"]).median().to_dict()


def impute_weight(df: pd.DataFrame, med: dict) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["weight"].fillna(df["equipment"].map(med))
    return df


def clean_frame(df: pd.DataFrame, mi_daily: pd.Series, weight_med: dict | None = None) -> pd.DataFrame:
    """Flags, abs(weight), MI fill, daily helper columns. Weight NaN is imputed only if weight_med is given."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["weight_neg_flag"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs()
    df["weight_missing_flag"] = df["weight"].isna().astype(int)
    df["weight_capped_flag"] = (df["weight"] >= CAP).astype(int)
    df["mi_missing_flag"] = df["market_index"].isna().astype(int)
    df["mi_date_mean"] = df["date"].map(mi_daily).astype(float)
    df["market_index"] = df["market_index"].fillna(df["mi_date_mean"])
    df["qs_daily_median"] = df.groupby("date")["quote_signal"].transform("median")
    df["qs_dev"] = df["quote_signal"] - df["qs_daily_median"]
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["lane"] = df["pickup"] + "|" + df["delivery"]
    if weight_med is not None:
        df = impute_weight(df, weight_med)
    return df


class OutlierModel:
    """Label-outlier detector fitted on fit rows only: rate vs median of (lane, equipment), else rpm band."""

    def fit(self, df: pd.DataFrame, rate: pd.Series):
        d = pd.DataFrame({"lane": df["lane"].values, "eq": df["equipment"].values,
                          "dist": df["distance"].values, "rate": rate.values})
        g = d.groupby(["lane", "eq"])["rate"].agg(["median", "size"])
        self.lane_med = g[g["size"] >= 3]["median"]
        d["rpm"] = d["rate"] / d["dist"]
        self.edges = np.unique(np.quantile(d["dist"], np.linspace(0, 1, 11)))
        d["band"] = np.clip(np.searchsorted(self.edges, d["dist"], side="right") - 1, 0, len(self.edges) - 2)
        self.band_rpm = d.groupby(["eq", "band"])["rpm"].median()
        self.glob_rpm = d["rpm"].median()
        return self

    def expected(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["lane"].values, df["equipment"].values])
        lm = self.lane_med.reindex(idx).values
        band = np.clip(np.searchsorted(self.edges, df["distance"].values, side="right") - 1, 0, len(self.edges) - 2)
        bi = pd.MultiIndex.from_arrays([df["equipment"].values, band])
        br = self.band_rpm.reindex(bi).fillna(self.glob_rpm).values * df["distance"].values
        return np.where(np.isnan(lm), br, lm)

    def flag(self, df: pd.DataFrame, rate: pd.Series) -> np.ndarray:
        ratio = rate.values / self.expected(df)
        return (ratio > 3) | (ratio < 1 / 3)


def data_quality(train: pd.DataFrame, val: pd.DataFrame, mi_daily: pd.Series) -> dict:
    out = {}
    med = weight_medians(train)
    for name, raw in (("train", train), ("validation", val)):
        c = clean_frame(raw, mi_daily, med)
        out[name] = {
            "rows": int(len(raw)),
            "weight_negative": int((raw["weight"] < 0).sum()),
            "weight_missing": int(raw["weight"].isna().sum()),
            "weight_at_cap_47500": int((raw["weight"] >= CAP).sum()),
            "market_index_missing": int(raw["market_index"].isna().sum()),
            "nan_weight_after_clean": int(c["weight"].isna().sum()),
            "nan_mi_after_clean": int(c["market_index"].isna().sum()),
        }
    ct = clean_frame(train, mi_daily, med)
    om = OutlierModel().fit(ct, ct["posted_rate"])
    ratio = ct["posted_rate"].values / om.expected(ct)
    fl = om.flag(ct, ct["posted_rate"])
    out["train"]["label_outliers"] = int(fl.sum())
    out["train"]["label_outlier_pct"] = round(100 * float(fl.mean()), 3)
    out["train"]["rate_above_3x"] = int((ratio > 3).sum())
    out["train"]["rate_below_third"] = int((ratio < 1 / 3).sum())
    coords = pd.concat([train[["pickup_lat", "pickup_lon"]].set_axis(["lat", "lon"], axis=1),
                        train[["delivery_lat", "delivery_lon"]].set_axis(["lat", "lon"], axis=1)])
    out["train"]["coord_rows_lon_clipped_at_-69.5"] = int((coords["lon"] >= -69.5 - 1e-9).sum())
    out["train"]["coord_rows_lat_clipped_at_25.5"] = int((coords["lat"] <= 25.5 + 1e-9).sum())
    tc, vc = set(train.pickup) | set(train.delivery), set(val.pickup) | set(val.delivery)
    new = sorted(vc - tc)
    touch = val["pickup"].isin(new) | val["delivery"].isin(new)
    out["validation"].update({"cities": len(vc), "unseen_cities": new, "rows_touching_unseen": int(touch.sum()),
                              "pct_rows_touching_unseen": round(100 * float(touch.mean()), 2)})
    out["train"]["cities"] = len(tc)
    return out
