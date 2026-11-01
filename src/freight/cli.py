"""Command line entry point: python -m freight.cli <cmd>."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import traceback

from . import config as C

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

log = logging.getLogger("freight")


def say(msg: str) -> None:
    print(msg, flush=True)
    log.info(msg)


def c_prepare(a):
    from . import io
    say(io.prepare_data())


def c_quality(a):
    from .clean import data_quality
    from .evaluate import fio, mi_daily_series
    tr, va = fio.load_train(), fio.load_val()
    q = data_quality(tr, va, mi_daily_series(tr, va))
    C.OUT.mkdir(exist_ok=True)
    (C.OUT / "data_quality.json").write_text(json.dumps(q, indent=1), encoding="utf-8")
    say("data_quality.json written: " + json.dumps(q["train"]["label_outlier_pct"]) + "% label outliers")


def c_eda(a):
    from . import eda
    c_quality(a)
    eda.run()
    say("eda figures written")


def c_stability(a):
    from . import stability
    say("stability: " + json.dumps(stability.run()))


def c_cv(a):
    from . import evaluate as E
    if a.seeds:
        sd, gate = E.run_seeds(a.seeds, a.write_decisions)
        say(f"seed sd {sd:.3f}%  gate {gate:.3f}%")
        E.write_summary()
    elif a.sweep:
        w, notes = E.run_sweep(a.sweep, a.write_decisions)
        for n in notes:
            say("  " + n)
        say(f"sweep {a.sweep}: winner {w} (write_decisions={a.write_decisions})")
        E.write_summary()
    else:
        E.run_cv_default()
        say("cv summary written to outputs/cv_summary.md")


def c_tune(a):
    from . import evaluate as E
    win, n, note = E.run_tune(a.write_decisions)
    say(note)
    E.write_summary()


def c_train(a):
    from . import train
    say(train.train())


def c_predict(a):
    from . import train
    say(train.predict())


def c_december(a):
    from . import december
    say("december: " + json.dumps(december.run(a.write_decisions)))


def c_score(a):
    cmd = [sys.executable, "score.py", "--predictions", "validation_predictions.csv",
           "--december-predictions", str(C.DATA / "december_chart_inputs.csv")]
    r = subprocess.run(cmd, cwd=C.ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    say(r.stdout.strip())
    C.REPORTS.mkdir(exist_ok=True)
    shutil.copyfile(C.ROOT / "scorer_results" / "candidate_december.png", C.REPORTS / "candidate_december.png")


def c_figures(a):
    from . import figures
    figures.run()
    say("figures written")


def c_report(a):
    from . import report
    say(report.build())


def c_manifest(a):
    from . import manifest
    manifest.run()
    say("run_manifest.json written")


def c_all(a):
    """Fixed-config pipeline. No sweeps, no tuning: relies on config/decisions.json."""
    a.sweep = a.seeds = None
    a.write_decisions = False
    for name, fn in [("prepare-data", c_prepare), ("eda", c_eda), ("stability", c_stability), ("cv", c_cv),
                     ("train", c_train), ("predict", c_predict), ("december", c_december), ("score", c_score),
                     ("figures", c_figures), ("report", c_report), ("manifest", c_manifest)]:
        say(f"== {name}")
        fn(a)


COMMANDS = {"prepare-data": c_prepare, "eda": c_eda, "stability": c_stability, "cv": c_cv, "tune": c_tune,
            "train": c_train, "predict": c_predict, "december": c_december, "score": c_score,
            "figures": c_figures, "report": c_report, "manifest": c_manifest, "quality": c_quality, "all": c_all}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="freight", description="Freight rate prediction pipeline")
    p.add_argument("cmd", choices=COMMANDS)
    p.add_argument("--sweep", choices=["outlier_target", "market", "recency"])
    p.add_argument("--seeds", type=int)
    p.add_argument("--write-decisions", action="store_true")
    a = p.parse_args(argv)
    C.LOGS.mkdir(exist_ok=True)
    h = logging.FileHandler(C.LOGS / f"{a.cmd}.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.setLevel(logging.INFO)
    log.addHandler(h)
    try:
        COMMANDS[a.cmd](a)
    except Exception as e:  # noqa: BLE001
        log.error(traceback.format_exc())
        print(f"FAILED {a.cmd}: {type(e).__name__}: {e}", flush=True)
        return 1
    print(f"DONE {a.cmd}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
