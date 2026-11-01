"""Expanding-window temporal CV (2-month test windows), gap fold, unseen-city masks, random-KFold contrast."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config as C

# name -> (last fit month, test months)
SPECS = {"F1": (4, (5, 6)), "F2": (6, (7, 8)), "F3": (8, (9, 10)), "gap": (8, (10,))}
MAIN = ["F1", "F2", "F3"]


@dataclass
class Fold:
    name: str
    fit: np.ndarray        # positional indices
    inner_fit: np.ndarray  # fit minus last fit month
    es: np.ndarray         # last fit month (early-stopping holdout)
    test: np.ndarray

    def fit_end(self, df) -> pd.Timestamp:
        return df["date"].iloc[self.fit].max()


def make_fold(df: pd.DataFrame, name: str) -> Fold:
    last, tm = SPECS[name]
    m = df["month"].values
    return Fold(name, np.where(m <= last)[0], np.where(m < last)[0], np.where(m == last)[0],
                np.where(np.isin(m, tm))[0])


def make_folds(df: pd.DataFrame, names=("F1", "F2", "F3", "gap")) -> dict[str, Fold]:
    return {n: make_fold(df, n) for n in names}


def unseen_city_fold(df: pd.DataFrame, seed: int, n_mask: int = 8) -> tuple[Fold, list[str]]:
    """F3 with n_mask train cities masked: every fit row touching them is dropped; test = F3 test rows touching them."""
    base = make_fold(df, "F3")
    rng = np.random.default_rng(seed)
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    # only cities that occur in the F3 test window can be evaluated
    test_cities = sorted((set(df["pickup"].iloc[base.test]) | set(df["delivery"].iloc[base.test])))
    masked = list(rng.choice(test_cities, size=n_mask, replace=False))
    touch = (df["pickup"].isin(masked) | df["delivery"].isin(masked)).values
    keep = lambda idx: idx[~touch[idx]]  # noqa: E731
    f = Fold("unseen", keep(base.fit), keep(base.inner_fit), keep(base.es), base.test[touch[base.test]])
    return f, masked


def random_kfold_contrast(n: int, k: int = 5, seed: int = C.SEED):
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(np.arange(n)):
        yield tr, te
