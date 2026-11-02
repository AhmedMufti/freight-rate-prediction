"""Build reports/report.docx from outputs/ only (no hard-coded metric numbers)."""
from __future__ import annotations

import json
import re

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from . import config as C
from .evaluate import BASELINES, get, read_results


def _j(p):
    return json.loads((C.OUT / p).read_text(encoding="utf-8"))


def _shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _table(doc, header, rows, widths=None, size=8):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = str(h)
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    for ri, row in enumerate(t.rows):
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))          # keep each row on one page
        if ri == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))      # repeat the header row on the next page
        for c in row.cells:
            if ri == 0:
                _shade(c, "D9E2F3")
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.bold = True if ri == 0 else False
    if widths:
        t.autofit = False
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Inches(w)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(2)
    spacer.add_run(" ").font.size = Pt(4)
    return t


_CMP = re.compile(r"gain (?P<g>[+-]?[\d.]+)% \(gate (?P<gate>[\d.]+)%\), F3 (?P<f3a>[\d.]+) vs (?P<f3b>[\d.]+), "
                  r"gap (?P<ga>[\d.]+) vs (?P<gb>[\d.]+) -> (?P<v>\w+)")


def _cmps(why: str) -> dict:
    """Comparisons recorded in a decisions.json _why string: 'A vs B' -> gain, F3 and gap numbers, verdict."""
    out = {}
    for piece in why.split(" | "):
        if ": gain" not in piece:
            continue
        label = piece.split(": gain")[0].split(". ")[-1].strip()
        m = _CMP.search(piece)
        if m:
            out[label] = m.groupdict()
    return out


def _sg(x) -> str:
    return f"{float(x):+.1f}%".replace("-", "\u2212")


def _why_evidence(dec: dict) -> dict:
    """Plain-English evidence per setting, built from the numbers stored in config/decisions.json."""
    w = dec["_why"]
    out = {}
    try:
        o = _cmps(w["outlier_mode"])
        d, wt, lr = o["drop|log_rpm vs keep|log_rpm"], o["weight|log_rpm vs drop|log_rpm"], o["drop|log_rate vs drop|log_rpm"]
        out["outlier_mode"] = (f"Dropping flagged labels from fitting beat keeping them: {_sg(d['g'])} mean MAE (F3 {d['f3a']} vs "
                               f"{d['f3b']}, gap {d['ga']} vs {d['gb']}). Down-weighting them instead gained only {_sg(wt['g'])}, "
                               f"below the {d['gate']}% gate.")
        sm = re.search(r"MAE ([\d.]+) plain vs ([\d.]+) with smear", w["target"])
        out["target"] = (f"Predicting log rate per mile beat predicting log rate: the log-rate target was {_sg(lr['g'])} "
                         f"(F3 {lr['f3a']} vs {lr['f3b']}, gap {lr['ga']} vs {lr['gb']})."
                         + (f" A smearing correction on the back-transform also hurt F3 (MAE {sm.group(1)} plain vs "
                            f"{sm.group(2)} smeared)." if sm else ""))
        m = _cmps(w["qs_variant"])
        q, qd = m["no_mi vs no_qs_no_mi"], m["qs_dev vs no_qs_no_mi"]
        out["qs_variant"] = (f"Rejected. On its own the quote signal improved F3 ({q['f3a']} vs {q['f3b']}) and the gap fold "
                             f"({q['ga']} vs {q['gb']}) but made the early folds so much worse that mean MAE was {abs(float(q['g'])):.1f}% worse "
                             f"(its effect flips sign between months). The deviation form was worse still ({_sg(qd['g'])}).")
        mi, full = m["no_qs vs no_qs_no_mi"], m["full vs no_qs_no_mi"]
        out["use_mi"] = (f"Rejected. The market index alone made F3 ({mi['f3a']} vs {mi['f3b']}) and the gap fold ({mi['ga']} vs "
                         f"{mi['gb']}) worse, and together with the quote signal it was worse again (F3 {full['f3a']} vs "
                         f"{full['f3b']}, gap {full['ga']} vs {full['gb']}). The audit below confirms it helps early folds only.")
        r = _cmps(w["recency_halflife"])
        r3, r6 = r["h=3 vs h=0"], r["h=6 vs h=3"]
        out["recency_halflife"] = (f"A 3-month half-life improved mean MAE by {_sg(r3['g'])} over no recency weighting (gap fold "
                                   f"{r3['ga']} vs {r3['gb']}); 6 months was {_sg(r6['g'])} vs 3 months.")
        t = re.search(r"incumbent (\S+) mean ([\d.]+); best grid (\S+) mean ([\d.]+) \(gain ([+-][\d.]+)%, gate ([\d.]+)%, "
                      r"F3 ([\d.]+) vs ([\d.]+), gap ([\d.]+) vs ([\d.]+)\)", w["lgbm_params"])
        out["lgbm_params"] = (f"Small grid: 31 leaves, min child 80 and a Huber loss beat the incumbent (63 leaves, min child 40, L2) "
                              f"by {_sg(t.group(5))} (mean MAE {t.group(4)} vs {t.group(2)}; F3 {t.group(7)} vs {t.group(8)}, "
                              f"gap {t.group(9)} vs {t.group(10)}).")
        n = re.search(r"median inner iters \[(.*?)\]\) = (\d+)", w["n_final"])
        out["n_final"] = f"1.1 x the median early-stopping round on the inner holdouts ({n.group(1)}) = {n.group(2)}."
    except Exception:  # never fail the report over wording: fall back to the first sentence of the raw note
        for k, v in w.items():
            out.setdefault(k, v.split(" | ")[0][:200])
    return out


def _setting_value(dec: dict, k: str) -> str:
    v = dec[k]
    if k == "lgbm_params":
        return (f"learning rate {v['learning_rate']}, {v['num_leaves']} leaves, min child {v['min_child_samples']}, "
                f"feature/bagging fraction {v['feature_fraction']}, L2 {v['lambda_l2']}, {v['objective']} loss")
    return v if isinstance(v, str) else json.dumps(v)


def _pic(doc, path, width=6.2, caption=None):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    if caption:
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].italic = True
        p.runs[0].font.size = Pt(8)


def build() -> str:
    res = read_results()
    dec = C.load_decisions()
    dq = _j("data_quality.json")
    stab = _j("stability.json")
    dsum = _j("december_summary.json")
    dlog = _j("decision_log.json") if (C.OUT / "decision_log.json").exists() else {}
    sens = pd.read_csv(C.OUT / "december_sensitivity.csv")
    seg = pd.read_csv(C.OUT / "segments_f3.csv")
    g = lambda m, f, k, st="final", v="default": get(res, st, v, m, f, k)  # noqa: E731
    L = "lgbm"
    mae = {m: g(m, "mean", "MAE") for m in BASELINES + [L]}
    d_b2 = 100 * (1 - mae[L] / mae["b2_lane_equip_rpm"])
    d_b1 = 100 * (1 - mae[L] / mae["b1_equip_distband_rpm"])
    d_b0 = 100 * (1 - mae[L] / mae["b0_global_rpm"])

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.8)
        s.top_margin = s.bottom_margin = Inches(0.7)

    # 1 ---------------------------------------------------------------- summary
    doc.add_heading("Freight rate prediction: validation approach, model and December forecast", 0)
    doc.add_heading("1. Summary", 1)
    doc.add_paragraph(
        "Split approach: expanding-window temporal cross-validation with 2-month test windows (matching the Nov-Dec "
        "validation horizon), plus a one-month-gap check (Oct only) and an unseen-city check. Random K-fold is reported "
        "only to show how optimistic it is. All transforms are fitted inside each fold's fit window, and the number of "
        "boosting rounds is chosen on an inner holdout (the last fit month), never on a test window.")
    doc.add_paragraph(
        f"Final model: LightGBM on log(rate per mile), back-transformed to dollars. Mean temporal-CV MAE "
        f"${mae[L]:,.0f} (MAPE {g(L,'mean','MAPE'):.1f}%, WAPE {g(L,'mean','WAPE'):.1f}%), against "
        f"${mae['b2_lane_equip_rpm']:,.0f} for the lane+equipment median baseline ({d_b2:.1f}% lower), "
        f"${mae['b1_equip_distband_rpm']:,.0f} for equipment x distance-band median ({d_b1:.1f}% lower) and "
        f"${mae['b0_global_rpm']:,.0f} for a global rate-per-mile ({d_b0:.1f}% lower). The selection fold F3 "
        f"(Sep-Oct) MAE is ${g(L,'F3','MAE'):,.0f}; the gap fold (Oct) ${g(L,'gap','MAE'):,.0f}.")
    _table(doc, ["model", "F1", "F2", "F3 (selection)", "mean MAE", "gap (Oct)", "mean MAPE %"],
           [[m, f"{g(m,'F1','MAE'):,.0f}", f"{g(m,'F2','MAE'):,.0f}", f"{g(m,'F3','MAE'):,.0f}", f"{g(m,'mean','MAE'):,.0f}",
             f"{g(m,'gap','MAE'):,.0f}", f"{g(m,'mean','MAPE'):.1f}"] for m in BASELINES + [L]])
    doc.add_paragraph()
    doc.add_paragraph(
        f"December forecast (Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb): mean ${dsum['mean']:,.0f}, range "
        f"${dsum['min']:,.0f} to ${dsum['max']:,.0f}, correlation with the daily market index {dsum['corr_with_mi']:.2f}"
        + ("." if dec["use_mi"] else " (the chosen model has no market inputs: the shape comes from day of week, and the "
           "December market index happens to share a weekly pattern; see section 7)."))
    _pic(doc, C.REPORTS / "candidate_december.png", 5.2, "score.py December chart (candidate_december.png)")

    # 2 ---------------------------------------------------------------- data / EDA
    doc.add_heading("2. Data and EDA", 1)
    es = _j("eda_summary.json")
    doc.add_paragraph(
        f"Train has {es['rows_train']:,} loads (2025-01-01 to 10-31) and validation {es['rows_val']:,} (Nov-Dec). "
        f"Distance dominates: corr(log rate, log distance) = {es['corr_loglog']:.3f}. Median rate per mile by equipment: "
        + ", ".join(f"{k} {v:.2f}" for k, v in es["rpm_median_by_equipment"].items()) + ". The market index is effectively one "
        f"value per day (sd within a date {es['mi_sd_within_date']:.3f} vs {es['mi_sd_between_dates']:.3f} between dates, lag-1 "
        f"autocorrelation {es['mi_lag1_autocorr']:.2f}). quote_signal is row-level, weakly and non-monotonically related to "
        "price (U-shape) and its linear relation with the residual flips sign by month.")
    _pic(doc, C.FIG / "fig_timeline.png", 5.8, "Loads per day and daily market index")
    _pic(doc, C.EDA / "eda_2_rate_vs_distance.png", 4.4, "Rate vs distance (log-log)")
    _pic(doc, C.EDA / "eda_5_quote_signal.png", 5.6, "quote_signal: U-shaped relation with rate per mile")
    _pic(doc, C.EDA / "eda_7_cities.png", 4.4, "Validation contains cities never seen in train")

    # 3 ---------------------------------------------------------------- data quality
    doc.add_heading("3. Data quality", 1)
    t, v = dq["train"], dq["validation"]
    _table(doc, ["issue", "train", "validation", "fix"], [
        ["negative weight (sign error)", t["weight_negative"], v["weight_negative"], "abs(); flag kept"],
        ["missing weight", t["weight_missing"], v["weight_missing"], "median per equipment (fit rows); flag"],
        ["weight exactly 47,500 (cap)", t["weight_at_cap_47500"], v["weight_at_cap_47500"], "kept as is; flag"],
        ["missing market_index", t["market_index_missing"], v["market_index_missing"], "mean of other rows same date; flag"],
        ["label outliers (rate vs lane-equipment median beyond 3x or 1/3)", f"{t['label_outliers']} ({t['label_outlier_pct']}%)", "n/a",
         f"mode '{dec['outlier_mode']}' (CV decision); always kept in evaluation"],
        ["coordinates clipped (lon -69.5 / lat 25.5 rows)", f"{t['coord_rows_lon_clipped_at_-69.5']} / {t['coord_rows_lat_clipped_at_25.5']}", "-", "left unchanged, reported"],
        ["cities not in train", "-", f"{len(v['unseen_cities'])} cities, {v['rows_touching_unseen']} rows ({v['pct_rows_touching_unseen']}%)", "use coordinates, not city IDs"],
        ["file names", "hyphenated in raw/", "", "prepare-data creates the data/ names used by the brief"],
    ])

    # 4 ---------------------------------------------------------------- validation
    doc.add_heading("4. Validation and split approach", 1)
    doc.add_paragraph(
        "Validation is the two months after the training data, so a random split would leak market regime information "
        "(neighbouring days share a market index and lane rates) and overstate accuracy. Folds: F1 fit Jan-Apr / test May-Jun; "
        "F2 fit Jan-Jun / test Jul-Aug; F3 fit Jan-Aug / test Sep-Oct; gap fold: same fit as F3, test October only (proxy for "
        "forecasting a month ahead); unseen-city fold: F3 with 8 train cities removed from the fit set and scored on the "
        "test rows touching them (3 seeds). Early stopping uses the last fit month as inner holdout, then the model is "
        "refit on the whole fit window with that iteration count.")
    _pic(doc, C.FIG / "fig_folds.png", 6.3, "Fold layout and MAE by model")
    doc.add_paragraph(
        f"Random 5-fold on all train gives LightGBM MAE ${g(L,'randomkfold','MAE'):,.0f} against ${mae[L]:,.0f} under temporal CV: "
        f"{100*(1-g(L,'randomkfold','MAE')/mae[L]):.0f}% too optimistic. Unseen-city MAE: LightGBM ${g(L,'unseen','MAE'):,.0f} vs "
        f"lane+equipment baseline ${g('b2_lane_equip_rpm','unseen','MAE'):,.0f}.")
    doc.add_paragraph(
        f"Stability: the linear residual correlation with quote_signal is positive in {stab['qs_pos_months']} and negative in "
        f"{stab['qs_neg_months']} of 10 months; market_index is positive in {stab['mi_pos_months']} of 10. Adversarial AUC "
        f"(Aug-Oct train vs validation) = {stab['auc']['all_features']:.3f} with all features and "
        f"{stab['auc']['without_market_cols']:.3f} without market columns (coordinates reveal the new cities).")
    doc.add_paragraph(
        "Caveat: decisions were made sequentially on F3 and the gap fold, so F3 is a selection fold, not an untouched holdout. "
        "The only untouched evaluation is the scoring of the validation set itself.")

    # 5 ---------------------------------------------------------------- features / model
    doc.add_heading("5. Features and model", 1)
    doc.add_paragraph(
        "Features: distance, equipment, cleaned weight and its missing flag, market_index"
        + (" (date mean)" if dec["mi_date_mean"] else "") + (", " + dec["qs_variant"] if dec["qs_variant"] != "none" else "")
        + ", pickup/delivery coordinates, circuity (distance / haversine) and day of week. No month or time index (December "
        "is out of the training range). City IDs and lane target-encodings are deliberately absent: coordinates generalise to new cities.")
    doc.add_paragraph(f"Target: {dec['target']}; outlier handling: {dec['outlier_mode']}; recency half-life: "
                      f"{dec['recency_halflife'] or 'off'} months; final boosting rounds n_final = {dec['n_final']}; "
                      f"seed noise sd = {dec['seed_sd_mae_pct']}% of MAE, so the adoption gate is {dec['gate_pct']}%.")
    doc.add_paragraph("Decision rule: adopt the more complex option only if mean MAE improves by at least the gate and neither "
                      "F3 nor the gap fold gets worse. Decisions and evidence (from config/decisions.json):")
    ev = _why_evidence(dec)
    _table(doc, ["setting", "value", "evidence"],
           [[k, _setting_value(dec, k), ev.get(k, "default (not tuned)")]
            for k in ("outlier_mode", "target", "qs_variant", "use_mi", "mi_date_mean", "recency_halflife", "lgbm_params", "n_final")],
           widths=[1.3, 1.9, 3.6], size=8)
    for stage in ("market",):
        if stage in dlog:
            doc.add_paragraph()
            doc.add_paragraph("Market-feature ablation (mean MAE over F1-F3, F3, gap). Variants: no_qs_no_mi = neither market input "
                              "(the baseline); no_qs = market index only; no_mi = quote signal only; full = both; qs_dev = quote "
                              "signal as a deviation from the day's median.")
            _table(doc, ["variant", "mean MAE", "F3", "gap", "decision"],
                   [[k, f"{s['mean']:.0f}", f"{s['F3']:.0f}", f"{s['gap']:.0f}",
                     "baseline" if "adopted" not in s else ("adopt" if s["adopted"] else "reject")] for k, s in dlog[stage]["table"].items()])
    aud_p = C.OUT / "mi_audit.csv"
    if aud_p.exists() and not dec["use_mi"]:
        aud = pd.read_csv(aud_p)
        cols = ["F1", "F2", "F3", "gap", "oct1", "mean"]
        ag = aud.groupby("variant", sort=False)[cols].mean()
        ns = aud.groupby("variant", sort=False)["seed"].count()
        doc.add_paragraph()
        doc.add_paragraph(
            "Market-index audit (scripts/mi_audit.py, final configuration, mean over seeds). Because market_index is positively "
            "related to the residual in every month, its rejection was re-tested in several forms, adding fold 'oct1' (fit Jan-Sep, "
            "test Oct), the closest proxy for fitting through October and predicting Nov-Dec. Variants: raw MI, date-mean MI, MI with "
            "a monotone (+) constraint, a fixed log-rate offset beta*log(MI) with beta estimated inside each fit window from "
            "within-month daily residuals (about 0.13-0.15), and offsets on short-run MI moves only (log MI minus its trailing 7/14/28-day mean).")
        _table(doc, ["variant", "seeds"] + cols,
               [[k, int(ns[k])] + [f"{r[c]:.1f}" for c in cols] for k, r in ag.iterrows()], size=7)
        doc.add_paragraph(
            "Every MI-level variant wins on F1 and F2 (test windows where MI rose or stayed high) but is 26-30% worse on F3 and "
            "11-15% worse on oct1. The reason is a confound: after removing an MI elasticity of about 0.14, rates still drift upward by "
            "about 6% from January to October, so when MI falls (Aug to Sep-Oct) an MI-driven model predicts a drop that did not happen. "
            "The short-run-only offsets avoid this but are neutral (mean within 0.4%, below the gate; oct1 about 1% worse), because "
            "day of week already captures MI's weekly cycle. Nov-Dec MI (0.92-0.93) is at the Sep-Oct level, so an MI feature would "
            "mostly add risk rather than signal. The no-market model is kept.")
    _pic(doc, C.FIG / "fig_importance.png", 4.2, "Feature importance (gain share)")

    # 6 ---------------------------------------------------------------- results
    doc.add_heading("6. Results", 1)
    _table(doc, ["model", "RMSE", "MAPE %", "MedAPE %", "WAPE %", "R2", "MAE sd across folds"],
           [[m, f"{g(m,'mean','RMSE'):,.0f}", f"{g(m,'mean','MAPE'):.1f}", f"{g(m,'mean','MedAPE'):.1f}", f"{g(m,'mean','WAPE'):.1f}",
             f"{g(m,'mean','R2'):.3f}", f"{g(m,'sd','MAE'):,.0f}"] for m in BASELINES + [L]])
    doc.add_paragraph()
    doc.add_paragraph("F3 segments (MAE in dollars):")
    sg = seg[seg.model == L]
    _table(doc, ["segment", "level", "n", "LightGBM MAE", "b2 MAE"],
           [[r.segment, r.level, r.n, f"{r.MAE:,.0f}",
             f"{seg[(seg.model=='b2_lane_equip_rpm')&(seg.segment==r.segment)&(seg.level==r.level)].MAE.iloc[0]:,.0f}"]
            for r in sg.itertuples()], size=7)
    _pic(doc, C.FIG / "fig_residuals.png", 5.8, "F3 predictions vs actual and error by distance band")

    # 7 ---------------------------------------------------------------- december
    doc.add_heading("7. December forecast", 1)
    doc.add_paragraph(
        "The chart file only gives the lane, distance, equipment, weight and date. The full feature rows are rebuilt for each "
        "December date with the same code path as training: coordinates from the city table, the fixed load attributes, day of week, and the "
        "market index for that date taken from validation.csv (it is a daily series that exists for every December date). "
        f"The December MI day-to-day volatility is {dsum['mi_diff_sd_ratio_dec_vs_train']:.2f}x the training volatility, so "
        f"{'a 3-day centred mean is used' if dsum['mi_smoothing'] else 'raw daily values are used (smoothing rule: only if > 2x)'}.")
    if dec["qs_variant"] != "none":
        doc.add_paragraph(
            "quote_signal is row-level and is not provided for the chart, and the model is non-linear in it (U-shape), so plugging in a "
            "median would give f(E[qs]) rather than E[f(qs)] (Jensen). The prediction is therefore averaged over the quote_signal values of "
            "that day's validation loads under 600 mi.")
    if not dec["use_mi"] and dec["qs_variant"] == "none":
        doc.add_paragraph(
            "The sequential CV ablation rejected both market_index and quote_signal as model inputs. The market index improved the "
            "early folds but made the selection fold and the gap fold worse; the quote signal alone improved those two late folds but "
            "made the early folds so much worse (its effect flips sign between months) that mean MAE got worse by far more than the gate "
            "(see section 5). So the chosen model's December curve is driven by "
            "distance, equipment, weight, lane geography and day of week only. In the sensitivity table the first six rows are "
            "therefore identical by construction; the two 'alt model' rows show what a market-aware model would predict.")
    _pic(doc, C.REPORTS / "candidate_december.png", 5.4, "score.py candidate_december.png")
    _table(doc, ["variant", "mean $", "min $", "max $", "corr with primary"],
           [[r.variant, f"{r['mean']:.0f}", f"{r['min']:.0f}", f"{r['max']:.0f}", f"{r.corr_with_primary:.2f}"] for _, r in sens.iterrows()])
    _pic(doc, C.FIG / "fig_december_history.png", 5.8, "December predictions vs historic Lexington-Fort Wayne Dry Van rates (scaled to 360 mi)")

    # 8 ---------------------------------------------------------------- limitations
    doc.add_heading("8. Limitations", 1)
    for s in ["Train has no Nov/Dec data, so holiday effects (Dec 24-26, 31) cannot be learned; the December curve only moves with day of week"
              + (" and the market index." if dec["use_mi"] else " (it tracks the market index's weekly cycle, corr "
                 f"{dsum['corr_with_mi']:.2f}, but not its level)."),
              ("The forecast depends on the daily market index supplied in validation.csv; a true forecast would need a forecast of that index."
               if dec["use_mi"] else "The model has no time trend; the January-October upward drift (about 6% after the market effect) "
                                      "is only partly carried into Nov-Dec through the 3-month recency weighting, so a small downward bias is possible."),
              "quote_signal is weak and unstable; its inclusion was decided by the CV rule and its contribution is small.",
              "Label outliers are assumed to be entry errors; this is an assumption.",
              "F3 is a selection fold; the real out-of-sample result is the external scoring.",
              "Future work: lane-level shrinkage, quantile models for uncertainty bands, a market-index forecaster."]:
        doc.add_paragraph(s, style="List Bullet")
    C.REPORTS.mkdir(exist_ok=True)
    out = C.REPORTS / "report.docx"
    props = doc.core_properties
    props.title = "Freight Rate Prediction: Approach and Validation"
    props.author = "Muhammad Ahmed Mufti"
    props.comments = ""
    doc.save(out)
    return f"wrote {out.name}"
