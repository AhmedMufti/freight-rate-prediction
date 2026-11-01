"""Metrics, fold runner, CV results store, decision rule and sweeps."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config as C
from . import io as fio
from .clean import OutlierModel, clean_frame, mi_daily_series
from .models import LGBM, make_model
from .splits import MAIN, make_fold, make_folds, random_kfold_contrast, unseen_city_fold

log = logging.getLogger("freight")
RESULTS = C.OUT / "cv_results.csv"
DLOG = C.OUT / "decision_log.json"
BASELINES = ["b0_global_rpm", "b1_equip_distband_rpm", "b2_lane_equip_rpm", "ridge_loglog"]


# ----------------------------------------------------------------------------- data
def load_clean():
    tr, va = fio.load_train(), fio.load_val()
    mi = mi_daily_series(tr, va)
    return clean_frame(tr, mi).reset_index(drop=True), clean_frame(va, mi).reset_index(drop=True), mi


# ----------------------------------------------------------------------------- metrics
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ape = np.abs(e) / y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE": float(100 * np.mean(ape)), "MedAPE": float(100 * np.median(ape)),
            "WAPE": float(100 * np.sum(np.abs(e)) / np.sum(y)),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


# ----------------------------------------------------------------------------- weights / fold runner
def train_weights(dec, om, df, rate, fit_end, recency=True):
    w = np.ones(len(df))
    mode = dec["outlier_mode"]
    if mode != "keep":
        fl = om.flag(df, rate)
        w[fl] = 0.0 if mode == "drop" else 0.1
    h = dec["recency_halflife"]
    if recency and h:
        months = (fit_end - df["date"]).dt.days.values / 30.44
        w = w * 0.5 ** (months / h)
    return w


def fit_lgbm_window(df, rate, fit_idx, inner_idx, es_idx, dec, seed):
    """Early-stop on the inner holdout, refit on the full fit window with that count. Returns (model, info)."""
    inner = df.iloc[inner_idx]
    r_in = rate.iloc[inner_idx]
    om = OutlierModel().fit(inner, r_in)
    w_in = train_weights(dec, om, inner, r_in, inner["date"].max())
    keep = w_in > 0
    es_df, es_r = df.iloc[es_idx], rate.iloc[es_idx]
    w_es = train_weights(dec, om, es_df, es_r, None, recency=False)
    m1 = LGBM(dec, seed=seed).fit(inner[keep], r_in[keep], w_in[keep], es=(es_df, es_r, w_es))
    pe = m1.predict_log(es_df)
    ye = m1._y(es_df, es_r)
    sel = w_es > 0
    smear = float(np.mean(np.exp(ye[sel] - pe[sel])))
    fit = df.iloc[fit_idx]
    r_fit = rate.iloc[fit_idx]
    om2 = OutlierModel().fit(fit, r_fit)
    w = train_weights(dec, om2, fit, r_fit, fit["date"].max())
    keep = w > 0
    m2 = LGBM(dec, seed=seed, n_iter=m1.best_iter).fit(fit[keep], r_fit[keep], w[keep])
    return m2, {"inner_iter": m1.best_iter, "smear": smear}


def run_fold(df, rate, fold, model_name, dec, seed=C.SEED):
    """Fit on fold.fit, predict fold.test. Returns (preds in dollars, info)."""
    test = df.iloc[fold.test]
    if model_name == "lgbm":
        m, info = fit_lgbm_window(df, rate, fold.fit, fold.inner_fit, fold.es, dec, seed)
        p = m.predict(test)
        info["pred_smear"] = p * info["smear"]
        return p, info
    m = make_model(model_name, dec).fit(df.iloc[fold.fit], rate.iloc[fold.fit])
    return m.predict(test), {}


def evaluate_model(df, rate, model_name, dec, seed=C.SEED, names=("F1", "F2", "F3")):
    """Returns rows [(fold, metric, value)] for F1-F3, mean, gap (derived from the F3 fit), plus raw F3 preds."""
    rows, preds = [], {}
    for n in names:
        f = make_fold(df, n)
        p, info = run_fold(df, rate, f, model_name, dec, seed)
        y = rate.iloc[f.test].values
        for k, v in metrics(y, p).items():
            rows.append((n, k, v))
        if "inner_iter" in info:
            rows.append((n, "inner_iter", info["inner_iter"]))
            rows.append((n, "MAE_smear", metrics(y, info["pred_smear"])["MAE"]))
        preds[n] = (f.test, p)
        if n == "F3":  # gap fold: same fit (Jan-Aug), Oct-only test -> subset of F3 predictions
            g = make_fold(df, "gap")
            pos = {t: i for i, t in enumerate(f.test)}
            sel = np.array([pos[t] for t in g.test])
            for k, v in metrics(rate.iloc[g.test].values, p[sel]).items():
                rows.append(("gap", k, v))
    for k in ("MAE", "RMSE", "MAPE", "MedAPE", "WAPE", "R2"):
        vals = [v for (f, m, v) in rows if m == k and f in MAIN]
        rows.append(("mean", k, float(np.mean(vals))))
        rows.append(("sd", k, float(np.std(vals, ddof=0))))
    return rows, preds


# ----------------------------------------------------------------------------- results store
def add_results(stage, variant, model, rows):
    C.OUT.mkdir(exist_ok=True)
    new = pd.DataFrame([{"stage": stage, "variant": variant, "model": model, "fold": f, "metric": m, "value": v}
                        for f, m, v in rows])
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        key = ((old.stage == stage) & (old.variant == variant) & (old.model == model)
               & old.fold.isin(new.fold.unique()))
        old = old[~key]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(RESULTS, index=False, encoding="utf-8", lineterminator="\n")


def read_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(
        columns=["stage", "variant", "model", "fold", "metric", "value"])


def get(res, stage, variant, model, fold, metric):
    s = res[(res.stage == stage) & (res.variant == variant) & (res.model == model) & (res.fold == fold)
            & (res.metric == metric)]["value"]
    return float(s.iloc[0]) if len(s) else float("nan")


# ----------------------------------------------------------------------------- decision rule / sweeps
def summarize_rows(rows) -> dict:
    d = {(f, m): v for f, m, v in rows}
    return {"mean": d[("mean", "MAE")], "F1": d[("F1", "MAE")], "F2": d[("F2", "MAE")], "F3": d[("F3", "MAE")],
            "gap": d[("gap", "MAE")]}


def adopt(cand: dict, inc: dict, gate_pct: float):
    gain = 100 * (inc["mean"] - cand["mean"]) / inc["mean"]
    ok = gain >= gate_pct and cand["F3"] <= inc["F3"] and cand["gap"] <= inc["gap"]
    return ok, gain


def apply_ladder(stage, variants, evaluate, gate):
    """variants: ordered [(name, overrides)], simplest first. evaluate(name, overrides)->summary dict.
    Each challenger must beat the current incumbent by the rule. Returns (winner_name, table, notes)."""
    table, notes = {}, []
    inc_name, inc = None, None
    for name, ov in variants:
        s = evaluate(name, ov)
        table[name] = s
        if inc is None:
            inc_name, inc = name, s
            continue
        ok, gain = adopt(s, inc, gate)
        s["gain_vs_incumbent_pct"] = gain
        s["vs"] = inc_name
        s["adopted"] = bool(ok)
        notes.append(f"{name} vs {inc_name}: gain {gain:+.2f}% (gate {gate:.2f}%), F3 {s['F3']:.1f} vs {inc['F3']:.1f}, "
                     f"gap {s['gap']:.1f} vs {inc['gap']:.1f} -> {'ADOPT' if ok else 'reject'}")
        if ok:
            inc_name, inc = name, s
    return inc_name, table, notes


def log_decision(stage, record):
    d = json.loads(DLOG.read_text(encoding="utf-8")) if DLOG.exists() else {}
    d[stage] = record
    DLOG.write_text(json.dumps(d, indent=2), encoding="utf-8")


def sweep_variants(stage, dec):
    if stage == "outlier_target":
        out = []
        for mode in ("keep", "drop", "weight"):
            for tgt in ("log_rpm", "log_rate"):
                out.append((f"{mode}|{tgt}", {"outlier_mode": mode, "target": tgt}))
        return out
    if stage == "market":
        return [("no_qs_no_mi", {"qs_variant": "none", "use_mi": False}),
                ("no_qs", {"qs_variant": "none", "use_mi": True}),
                ("no_mi", {"qs_variant": "qs", "use_mi": False}),
                ("full", {"qs_variant": "qs", "use_mi": True}),
                ("qs_dev", {"qs_variant": "qs_dev", "use_mi": True})]
    if stage == "recency":
        return [("h=0", {"recency_halflife": 0}), ("h=3", {"recency_halflife": 3}), ("h=6", {"recency_halflife": 6})]
    raise ValueError(stage)


def run_sweep(stage: str, write: bool, seed=C.SEED):
    tr, _, _ = load_clean()
    rate = tr["posted_rate"]
    dec = C.load_decisions()
    gate = dec["gate_pct"]
    cache = {}

    def evaluate(name, ov):
        d = {**dec, **ov}
        rows, _ = evaluate_model(tr, rate, "lgbm", d, seed)
        add_results(stage, name, "lgbm", rows)
        s = summarize_rows(rows)
        sm = {(f, m): v for f, m, v in rows}
        s["F3_smear"] = sm.get(("F3", "MAE_smear"), float("nan"))
        s["mean_inner_iter"] = float(np.mean([sm[(f, "inner_iter")] for f in MAIN]))
        log.info("%s %s: %s", stage, name, {k: round(v, 2) for k, v in s.items()})
        print(f"  {stage}/{name}: mean MAE {s['mean']:.2f}  F3 {s['F3']:.2f}  gap {s['gap']:.2f}", flush=True)
        cache[name] = ov
        return s

    winner, table, notes = apply_ladder(stage, sweep_variants(stage, dec), evaluate, gate)
    updates = dict(cache[winner])
    extra = ""
    if stage == "market" and updates.get("use_mi"):
        ov = {**updates, "mi_date_mean": True}
        inc = table[winner]
        s = evaluate("mi_date_mean", ov)
        ok, gain = adopt(s, inc, gate)
        notes.append(f"mi_date_mean vs {winner}: gain {gain:+.2f}% -> {'ADOPT' if ok else 'reject'}")
        table["mi_date_mean"] = {**s, "gain_vs_incumbent_pct": gain, "adopted": bool(ok), "vs": winner}
        updates["mi_date_mean"] = bool(ok)
        extra = f"; mi_date_mean={'on' if ok else 'off'}"
    if stage == "outlier_target":
        sm = table[winner]
        extra = (f"; smearing check on F3 for {winner}: MAE {sm['F3']:.1f} plain vs {sm['F3_smear']:.1f} with smear factor"
                 " (factor from inner holdout residuals)")
    why = f"{stage} sweep (gate {gate:.2f}%): winner {winner}. " + " | ".join(notes) + extra
    log_decision(stage, {"winner": winner, "gate_pct": gate, "table": table, "notes": notes, "updates": updates})
    if write:
        for k, v in updates.items():
            C.set_decision(k, v, why)
        if stage == "outlier_target":
            pass
    return winner, notes


def run_seeds(n: int, write: bool):
    tr, _, _ = load_clean()
    rate = tr["posted_rate"]
    dec = C.load_decisions()
    means = []
    for i in range(n):
        rows, _ = evaluate_model(tr, rate, "lgbm", dec, seed=C.SEED + i)
        means.append(summarize_rows(rows)["mean"])
        print(f"  seed {C.SEED + i}: mean MAE {means[-1]:.3f}", flush=True)
    sd_pct = 100 * float(np.std(means, ddof=1)) / float(np.mean(means))
    gate = max(0.5, 2 * sd_pct)
    log_decision("seeds", {"maes": means, "sd_pct": sd_pct, "gate_pct": gate})
    if write:
        why = f"seed sd of mean-MAE over {n} seeds = {sd_pct:.3f}% (MAEs {[round(m, 2) for m in means]}); gate = max(0.5, 2*sd)"
        C.set_decision("seed_sd_mae_pct", round(sd_pct, 4), why)
        C.set_decision("gate_pct", round(gate, 4), why)
    return sd_pct, gate


# ----------------------------------------------------------------------------- default cv
def segments_f3(df, rate, fold, preds: dict) -> pd.DataFrame:
    """MAE by segment on F3 for each model. preds: model -> array aligned with fold.test."""
    t = df.iloc[fold.test]
    y = rate.iloc[fold.test].values
    fit_cities = set(df["pickup"].iloc[fold.fit]) | set(df["delivery"].iloc[fold.fit])
    unseen = (~t["pickup"].isin(fit_cities) | ~t["delivery"].isin(fit_cities)).values
    om = OutlierModel().fit(df.iloc[fold.fit], rate.iloc[fold.fit])
    outlier = om.flag(t, rate.iloc[fold.test])
    band = pd.cut(t["distance"], [0, 300, 800, 1500, 1e9], labels=["<300", "300-800", "800-1500", ">1500"]).astype(str).values
    segs = {"equipment": t["equipment"].values, "distance_band": band,
            "city": np.where(unseen, "unseen", "seen"), "label": np.where(outlier, "outlier", "clean")}
    out = []
    for m, p in preds.items():
        for sname, sv in segs.items():
            for level in pd.unique(sv):
                k = sv == level
                if k.sum() == 0:
                    continue
                mm = metrics(y[k], p[k])
                out.append({"model": m, "segment": sname, "level": level, "n": int(k.sum()),
                            "MAE": mm["MAE"], "MAPE": mm["MAPE"]})
    return pd.DataFrame(out)


def run_cv_default(seed=C.SEED):
    tr, _, _ = load_clean()
    rate = tr["posted_rate"]
    dec = C.load_decisions()
    f3 = make_fold(tr, "F3")
    f3preds = {}
    inner = []
    for name in BASELINES + ["lgbm"]:
        rows, preds = evaluate_model(tr, rate, name, dec, seed)
        add_results("final", "default", name, rows)
        f3preds[name] = preds["F3"][1]
        if name == "lgbm":
            inner = [v for (f, m, v) in rows if m == "inner_iter"]
        print(f"  {name}: mean MAE {summarize_rows(rows)['mean']:.2f}", flush=True)
    # unseen-city check, 3 seeds
    for name in ["b1_equip_distband_rpm", "b2_lane_equip_rpm", "ridge_loglog", "lgbm"]:
        mae, mape, ns = [], [], []
        for s in range(3):
            f, masked = unseen_city_fold(tr, seed=C.SEED + s)
            p, _ = run_fold(tr, rate, f, name, dec, seed)
            mm = metrics(rate.iloc[f.test].values, p)
            mae.append(mm["MAE"]); mape.append(mm["MAPE"]); ns.append(len(f.test))
        add_results("final", "default", name, [("unseen", "MAE", float(np.mean(mae))), ("unseen", "MAPE", float(np.mean(mape))),
                                              ("unseen", "n_test", float(np.mean(ns))), ("unseen", "MAE_sd", float(np.std(mae)))])
        print(f"  unseen {name}: MAE {np.mean(mae):.2f}", flush=True)
    # random-KFold contrast (core lgbm with fixed iteration count)
    n_it = int(dec["n_final"] or np.median(inner))
    d2 = dec
    maes, y_all, p_all = [], [], []
    for tri, tei in random_kfold_contrast(len(tr)):
        fit = tr.iloc[tri]
        r = rate.iloc[tri]
        m = LGBM(d2, seed=seed, n_iter=n_it).fit(fit, r)
        p = m.predict(tr.iloc[tei])
        maes.append(metrics(rate.iloc[tei].values, p)["MAE"])
    add_results("final", "default", "lgbm", [("randomkfold", "MAE", float(np.mean(maes))),
                                            ("randomkfold", "MAE_sd", float(np.std(maes)))])
    print(f"  random 5-fold lgbm MAE {np.mean(maes):.2f}", flush=True)
    seg = segments_f3(tr, rate, f3, {k: v for k, v in f3preds.items() if k in ("b1_equip_distband_rpm", "b2_lane_equip_rpm", "lgbm")})
    seg.to_csv(C.OUT / "segments_f3.csv", index=False, encoding="utf-8", lineterminator="\n")
    t = tr.iloc[f3.test]
    pd.DataFrame({"load_id": t["load_id"].values, "equipment": t["equipment"].values, "distance": t["distance"].values,
                  "date": t["date"].dt.strftime("%Y-%m-%d").values, "y": rate.iloc[f3.test].values,
                  "pred": f3preds["lgbm"]}).to_csv(C.OUT / "f3_predictions.csv", index=False, encoding="utf-8",
                                                  lineterminator="\n", float_format="%.4f")
    write_summary()


# ----------------------------------------------------------------------------- tuning
def run_tune(write: bool, seed=C.SEED):
    tr, _, _ = load_clean()
    rate = tr["posted_rate"]
    dec = C.load_decisions()
    gate = dec["gate_pct"]
    base_params = dict(dec["lgbm_params"])
    grid = [(nl, mcs, obj) for nl in (31, 63, 127) for mcs in (20, 80) for obj in ("l2", "huber")]
    rows_out, summ, iters = [], {}, {}
    for nl, mcs, obj in grid:
        name = f"nl{nl}_mcs{mcs}_{obj}"
        p = {**base_params, "num_leaves": nl, "min_child_samples": mcs, "objective": obj}
        d = {**dec, "lgbm_params": p}
        rows, _ = evaluate_model(tr, rate, "lgbm", d, seed)
        add_results("tune", name, "lgbm", rows)
        s = summarize_rows(rows)
        sm = {(f, m): v for f, m, v in rows}
        iters[name] = [sm[(f, "inner_iter")] for f in MAIN]
        summ[name] = (s, p)
        rows_out.append({"config": name, **s, "inner_iters": iters[name]})
        print(f"  tune {name}: mean {s['mean']:.2f} F3 {s['F3']:.2f} gap {s['gap']:.2f}", flush=True)
    pd.DataFrame(rows_out).to_csv(C.OUT / "tuning.csv", index=False, encoding="utf-8", lineterminator="\n")
    # incumbent = the current decisions config (grid cell with the same params if present, else evaluate it)
    cur = f"nl{base_params['num_leaves']}_mcs{base_params['min_child_samples']}_{base_params['objective']}"
    if cur in summ:
        inc_name = cur
    else:
        rows, _ = evaluate_model(tr, rate, "lgbm", dec, seed)
        summ[cur] = (summarize_rows(rows), base_params)
        sm = {(f, m): v for f, m, v in rows}
        iters[cur] = [sm[(f, "inner_iter")] for f in MAIN]
        inc_name = cur
    inc = summ[inc_name][0]
    best = min((k for k in summ if k != inc_name), key=lambda k: summ[k][0]["mean"])
    ok, gain = adopt(summ[best][0], inc, gate)
    win = best if ok else inc_name
    n_final = int(round(1.1 * float(np.median(iters[win]))))
    note = (f"tune: incumbent {inc_name} mean {inc['mean']:.2f}; best grid {best} mean {summ[best][0]['mean']:.2f} "
            f"(gain {gain:+.2f}%, gate {gate:.2f}%, F3 {summ[best][0]['F3']:.1f} vs {inc['F3']:.1f}, gap "
            f"{summ[best][0]['gap']:.1f} vs {inc['gap']:.1f}) -> {'ADOPT' if ok else 'keep incumbent'}; "
            f"n_final = round(1.1 * median inner iters {iters[win]}) = {n_final}")
    log_decision("tune", {"winner": win, "note": note, "n_final": n_final})
    if write:
        C.set_decision("lgbm_params", summ[win][1], note)
        C.set_decision("n_final", n_final, note)
    return win, n_final, note


# ----------------------------------------------------------------------------- summary markdown
def mi_ranges(tr):
    out = []
    for n in MAIN:
        f = make_fold(tr, n)
        a, b = tr["mi_date_mean"].iloc[f.fit], tr["mi_date_mean"].iloc[f.test]
        out.append((n, a.min(), a.max(), b.min(), b.max(), b.mean(), a.mean()))
    return out


def write_summary():
    res = read_results()
    tr, va, _ = load_clean()
    L = ["# CV summary", "", "MAE in dollars. Folds are expanding-window temporal splits; F3 is the *selection fold* "
         "(decisions were made on it, so it is not an untouched holdout). `gap` = Oct-only test with the Jan-Aug fit "
         "(proxy for the 1-month gap before December).", ""]
    fin = res[res.stage == "final"]
    models = [m for m in BASELINES + ["lgbm"] if m in set(fin.model)]
    L += ["## Final configuration (config/decisions.json)", "",
          "| model | F1 | F2 | F3 | mean MAE | sd | gap(Oct) | mean RMSE | mean MAPE % | mean MedAPE % | mean WAPE % | mean R2 | unseen-city MAE | random 5-fold MAE |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        g = lambda f, k: get(res, "final", "default", m, f, k)  # noqa: E731
        L.append(f"| {m} | {g('F1','MAE'):.1f} | {g('F2','MAE'):.1f} | {g('F3','MAE'):.1f} | {g('mean','MAE'):.1f} | "
                 f"{g('sd','MAE'):.1f} | {g('gap','MAE'):.1f} | {g('mean','RMSE'):.1f} | {g('mean','MAPE'):.2f} | "
                 f"{g('mean','MedAPE'):.2f} | {g('mean','WAPE'):.2f} | {g('mean','R2'):.3f} | {g('unseen','MAE'):.1f} | "
                 f"{g('randomkfold','MAE'):.1f} |")
    L += ["", "## Market-index range per fold (daily MI): fit window vs test window", "",
          "| fold | fit MI min..max (mean) | test MI min..max (mean) |", "|---|---|---|"]
    for n, a0, a1, b0, b1, bm, am in mi_ranges(tr):
        L.append(f"| {n} | {a0:.2f}..{a1:.2f} ({am:.2f}) | {b0:.2f}..{b1:.2f} ({bm:.2f}) |")
    vm = va["mi_date_mean"]
    L.append(f"| validation (Nov-Dec) | train Jan-Oct {tr['mi_date_mean'].min():.2f}..{tr['mi_date_mean'].max():.2f} | "
             f"{vm.min():.2f}..{vm.max():.2f} ({vm.mean():.2f}) |")
    dl = json.loads(DLOG.read_text(encoding="utf-8")) if DLOG.exists() else {}
    for stage in ("seeds", "outlier_target", "market", "recency", "tune"):
        if stage not in dl:
            continue
        rec = dl[stage]
        L += ["", f"## Decision stage: {stage}", ""]
        if stage == "seeds":
            L.append(f"Seed sd of mean MAE = {rec['sd_pct']:.3f}% over MAEs {[round(x, 2) for x in rec['maes']]}; "
                     f"gate = {rec['gate_pct']:.3f}%.")
            continue
        if stage == "tune":
            L.append(rec["note"])
            continue
        L += ["| variant | mean MAE | F1 | F2 | F3 | gap | gain vs incumbent % | decision |", "|---|---|---|---|---|---|---|---|"]
        for k, s in rec["table"].items():
            dcs = "baseline" if "adopted" not in s else ("ADOPT" if s["adopted"] else "reject")
            L.append(f"| {k} | {s['mean']:.1f} | {s['F1']:.1f} | {s['F2']:.1f} | {s['F3']:.1f} | {s['gap']:.1f} | "
                     f"{s.get('gain_vs_incumbent_pct', float('nan')):+.2f} | {dcs} |")
        L += ["", f"Winner: **{rec['winner']}**. " + " ".join(f"- {n}" for n in rec["notes"][:0])]
        L += [f"- {n}" for n in rec["notes"]]
    d = C.load_decisions()
    L += ["", "## Current decisions", "", "```", json.dumps({k: v for k, v in d.items() if k != "_why"}, indent=1), "```"]
    (C.OUT / "cv_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
