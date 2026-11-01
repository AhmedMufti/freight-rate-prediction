"""run_manifest.json: versions, SHA256 of data/outputs, git sha, decisions."""
from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata

from . import config as C


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run() -> dict:
    try:
        g = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        g = None
    files = {}
    for p in [*sorted(C.DATA.glob("*.csv")), C.ROOT / "validation_predictions.csv", C.ROOT / "score.py",
              C.OUT / "december_predictions.csv", C.DECISIONS]:
        if p.exists():
            files[str(p.relative_to(C.ROOT)).replace("\\", "/")] = sha(p)
    pk = {n: metadata.version(n) for n in ("pandas", "numpy", "scikit-learn", "lightgbm", "matplotlib", "python-docx", "joblib")}
    m = {"git_sha": g, "packages": pk, "sha256": files, "decisions": C.load_decisions(), "seed": C.SEED}
    (C.OUT / "run_manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m
