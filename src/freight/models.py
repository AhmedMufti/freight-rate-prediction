"""Model registry. CLI names == registry names. Every model: fit(df, rate, w, es) / predict(df) -> dollars."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import FeatureBuilder


class B0:
    name = "b0_global_rpm"

    def fit(self, df, rate, w=None, es=None):
        self.rpm = float(np.median(rate.values / df["distance"].values))
        return self

    def predict(self, df):
        return self.rpm * df["distance"].values


class B1:
    name = "b1_equip_distband_rpm"

    def fit(self, df, rate, w=None, es=None):
        rpm = rate.values / df["distance"].values
        self.edges = np.unique(np.quantile(df["distance"].values, np.linspace(0, 1, 11)))
        d = pd.DataFrame({"eq": df["equipment"].values, "band": self._band(df), "rpm": rpm})
        self.med = d.groupby(["eq", "band"])["rpm"].median()
        self.glob = float(np.median(rpm))
        return self

    def _band(self, df):
        return np.clip(np.searchsorted(self.edges, df["distance"].values, side="right") - 1, 0, len(self.edges) - 2)

    def predict(self, df):
        idx = pd.MultiIndex.from_arrays([df["equipment"].values, self._band(df)])
        return self.med.reindex(idx).fillna(self.glob).values * df["distance"].values


class B2:
    name = "b2_lane_equip_rpm"

    def fit(self, df, rate, w=None, es=None):
        self.b1 = B1().fit(df, rate)
        d = pd.DataFrame({"lane": df["lane"].values, "pu": df["pickup"].values, "eq": df["equipment"].values,
                          "rpm": rate.values / df["distance"].values})
        self.lane = d.groupby(["lane", "eq"])["rpm"].median()
        self.pu = d.groupby(["pu", "eq"])["rpm"].median()
        return self

    def predict(self, df):
        li = pd.MultiIndex.from_arrays([df["lane"].values, df["equipment"].values])
        pi = pd.MultiIndex.from_arrays([df["pickup"].values, df["equipment"].values])
        rpm = self.lane.reindex(li).values
        rpm = np.where(np.isnan(rpm), self.pu.reindex(pi).values, rpm)
        fb = self.b1.predict(df) / df["distance"].values
        return np.where(np.isnan(rpm), fb, rpm) * df["distance"].values


class RidgeLogLog:
    name = "ridge_loglog"

    def __init__(self, decisions=None, seed=C.SEED):
        self.dec = decisions or C.load_decisions()

    def fit(self, df, rate, w=None, es=None):
        self.fb = FeatureBuilder(self.dec).fit(df)
        X = self.fb.transform_ridge(df)
        self.cols = list(X.columns)
        self.sc = StandardScaler().fit(X)
        y = np.log(rate.values / df["distance"].values)
        self.m = Ridge(alpha=1.0).fit(self.sc.transform(X), y, sample_weight=w)
        return self

    def predict(self, df):
        X = self.fb.transform_ridge(df)[self.cols]
        return np.exp(self.m.predict(self.sc.transform(X))) * df["distance"].values


class LGBM:
    name = "lgbm"
    MAX_ROUNDS = 4000

    def __init__(self, decisions=None, seed=C.SEED, n_iter=None, params=None):
        self.dec = decisions or C.load_decisions()
        self.seed = seed
        self.n_iter = n_iter
        self.params = dict(self.dec["lgbm_params"])
        if params:
            self.params.update(params)
        self.best_iter = n_iter

    def _y(self, df, rate):
        r = rate.values.astype(float)
        return np.log(r / df["distance"].values) if self.dec["target"] == "log_rpm" else np.log(r)

    def _inv(self, yhat, df):
        e = np.exp(yhat)
        return e * df["distance"].values if self.dec["target"] == "log_rpm" else e

    def fit(self, df, rate, w=None, es=None):
        """es = (df_es, rate_es, w_es) enables early stopping (200 rounds) and sets best_iter."""
        self.fb = FeatureBuilder(self.dec).fit(df)
        X = self.fb.transform(df)
        p = {**self.params, "metric": "l1", "seed": self.seed, "bagging_seed": self.seed,
             "feature_fraction_seed": self.seed, "verbose": -1, "deterministic": True,
             "force_row_wise": True, "num_threads": 4}
        if p["objective"] == "huber":
            p.setdefault("alpha", 0.3)
        ds = lgb.Dataset(X, self._y(df, rate), weight=w, categorical_feature=["equipment"])
        if es is not None:
            des, res, wes = es
            dv = lgb.Dataset(self.fb.transform(des), self._y(des, res), weight=wes,
                             categorical_feature=["equipment"], reference=ds)
            self.booster = lgb.train(p, ds, self.MAX_ROUNDS, valid_sets=[dv],
                                     callbacks=[lgb.early_stopping(200, verbose=False)])
            self.best_iter = int(self.booster.best_iteration)
        else:
            assert self.n_iter, "n_iter required without early stopping"
            self.booster = lgb.train(p, ds, int(self.n_iter))
            self.best_iter = int(self.n_iter)
        return self

    def predict(self, df):
        X = self.fb.transform(df)
        return self._inv(self.booster.predict(X, num_iteration=self.best_iter), df)

    def predict_log(self, df):
        return self.booster.predict(self.fb.transform(df), num_iteration=self.best_iter)

    def importance(self):
        return pd.Series(self.booster.feature_importance("gain"), index=self.booster.feature_name())


REGISTRY = {c.name: c for c in (B0, B1, B2, RidgeLogLog, LGBM)}


def make_model(name: str, decisions=None, **kw):
    cls = REGISTRY[name]
    if cls in (RidgeLogLog, LGBM):
        return cls(decisions, **kw)
    return cls()
