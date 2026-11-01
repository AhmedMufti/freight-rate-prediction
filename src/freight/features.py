"""FeatureBuilder: one fit/transform used for train, validation and December."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .clean import weight_medians

GEO = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


class FeatureBuilder:
    def __init__(self, decisions: dict | None = None):
        d = decisions or C.load_decisions()
        self.qs_variant = d["qs_variant"]
        self.use_mi = d["use_mi"]
        self.mi_date_mean = d["mi_date_mean"]

    def fit(self, df: pd.DataFrame):
        self.weight_med = weight_medians(df)
        self.qs_median = float(df["quote_signal"].median())
        return self

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["distance"] = df["distance"].astype(float)
        X["equipment"] = pd.Categorical(df["equipment"], categories=C.EQUIP)
        w = df["weight"].astype(float).abs()
        X["weight"] = w.fillna(df["equipment"].map(self.weight_med)).astype(float)
        X["weight_missing_flag"] = df["weight_missing_flag"].astype(int)
        if self.use_mi:
            X["market_index"] = (df["mi_date_mean"] if self.mi_date_mean else df["market_index"]).astype(float)
        if self.qs_variant == "qs":
            X["qs"] = df["quote_signal"].astype(float)
        elif self.qs_variant == "qs_dev":
            X["qs_dev"] = df["qs_dev"].astype(float)
        for c in GEO:
            X[c] = df[c].astype(float)
        hv = haversine(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
        X["circuity"] = df["distance"] / np.maximum(hv, 1.0)
        X["dow"] = df["dow"].astype(int)
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._base(df)

    def transform_ridge(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._base(df).drop(columns=["distance", "equipment", "qs", "qs_dev"], errors="ignore")
        X["log_distance"] = np.log(df["distance"].astype(float))
        for e in C.EQUIP[1:]:
            X[f"eq_{e}"] = (df["equipment"] == e).astype(float)
        if self.qs_variant != "none":
            X["abs_qs_dev"] = (df["quote_signal"] - self.qs_median).abs()
        X["weight"] = X["weight"] / 10000.0
        return X
