"""Report figures (the five named in the plan)."""
from __future__ import annotations

import shutil

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from .evaluate import BASELINES, get, load_clean, read_results  # noqa: E402
from .train import MODEL_PATH  # noqa: E402


def _save(fig, name):
    C.FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG / name, dpi=100)
    plt.close(fig)


def run() -> None:
    tr, va, mi = load_clean()
    # timeline
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].bar(tr.groupby("date").size().index, tr.groupby("date").size().values, color="#4C78A8", label="train")
    ax[0].bar(va.groupby("date").size().index, va.groupby("date").size().values, color="#F58518", label="validation")
    ax[0].set_ylabel("loads / day"); ax[0].legend(); ax[0].set_title("Data timeline")
    ax[1].plot(mi.index, mi.values, c="#54A24B"); ax[1].set_ylabel("market index")
    _save(fig, "fig_timeline.png")

    # folds
    res = read_results()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    spans = {"F1": (0, 4, 4, 6), "F2": (0, 6, 6, 8), "F3": (0, 8, 8, 10), "gap": (0, 8, 9, 10)}
    for i, (n, (a, b, c, d)) in enumerate(spans.items()):
        ax[0].barh(i, b - a, left=a, color="#4C78A8")
        ax[0].barh(i, d - c if n != "gap" else 1, left=c if n != "gap" else 9, color="#F58518")
    ax[0].set_yticks(range(4)); ax[0].set_yticklabels(spans); ax[0].invert_yaxis()
    ax[0].set_xlabel("months of 2025 (blue=fit, orange=test)"); ax[0].set_title("Expanding-window temporal folds")
    w = 0.16
    for j, m in enumerate(BASELINES + ["lgbm"]):
        v = [get(res, "final", "default", m, f, "MAE") for f in ("F1", "F2", "F3", "gap")]
        ax[1].bar(np.arange(4) + j * w, v, w, label=m)
    ax[1].set_xticks(np.arange(4) + 2 * w); ax[1].set_xticklabels(["F1", "F2", "F3", "gap"])
    ax[1].set_ylabel("MAE ($)"); ax[1].legend(fontsize=7); ax[1].set_title("MAE by fold")
    _save(fig, "fig_folds.png")

    # residuals
    f3 = pd.read_csv(C.OUT / "f3_predictions.csv")
    f3["err"] = f3.pred - f3.y
    f3["band"] = pd.cut(f3.distance, [0, 300, 800, 1500, 1e9], labels=["<300", "300-800", "800-1500", ">1500"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(f3.y, f3.pred, s=3, alpha=.3); lim = [50, 20000]
    ax[0].plot(lim, lim, c="k", lw=1); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("actual ($)"); ax[0].set_ylabel("predicted ($)"); ax[0].set_title("F3 (Sep-Oct): predicted vs actual")
    ax[1].boxplot([f3.loc[f3.band == b, "err"] for b in f3.band.cat.categories], tick_labels=f3.band.cat.categories, showfliers=False)
    ax[1].axhline(0, c="k", lw=1); ax[1].set_title("F3 error ($) by distance band")
    _save(fig, "fig_residuals.png")

    # importance
    m = joblib.load(MODEL_PATH)["model"]
    imp = m.importance().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp.values / imp.sum() * 100, color="#4C78A8"); ax.set_xlabel("share of total gain (%)")
    ax.set_title("LightGBM feature importance (final model)")
    _save(fig, "fig_importance.png")

    shutil.copyfile(C.OUT / "december_sanity.png", C.FIG / "fig_december_history.png")
