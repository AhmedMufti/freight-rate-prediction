"""Paths, constants and the decisions.json registry."""
from __future__ import annotations

import copy
import json
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = ROOT / "raw"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
EDA = OUT / "eda"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
DECISIONS = CONFIG / "decisions.json"

# canonical name -> (data/ name, raw/ name, expected rows)
FILES = {
    "train": ("train_test.csv", "train-test.csv", 48000),
    "val": ("validation.csv", "validation.csv", 12000),
    "template": ("validation_predictions_template.csv", "validation-predictions-template.csv", 12000),
    "december": ("december_chart_inputs.csv", "december-chart-inputs.csv", 31),
}

EQUIP = ["Dry Van", "Flatbed", "Reefer"]
DEC_LANE = ("Lexington", "Fort Wayne")
DEC_DISTANCE = 360.0
DEC_EQUIP = "Dry Van"
DEC_WEIGHT = 32000.0

LGBM_DEFAULT = {
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "objective": "l2",
}

DEFAULT_DECISIONS = {
    "outlier_mode": "keep",
    "target": "log_rpm",
    "qs_variant": "qs",
    "use_mi": True,
    "mi_date_mean": False,
    "recency_halflife": 0,
    "lgbm_params": LGBM_DEFAULT,
    "n_final": None,
    "seed_sd_mae_pct": None,
    "gate_pct": 0.5,
    "dec_qs_method": "marginalise",
    "dec_mi_smoothing": False,
    "_why": {},
}


def load_decisions() -> dict:
    d = copy.deepcopy(DEFAULT_DECISIONS)
    if DECISIONS.exists():
        d.update(json.loads(DECISIONS.read_text(encoding="utf-8")))
    return d


def save_decisions(d: dict) -> None:
    CONFIG.mkdir(exist_ok=True)
    DECISIONS.write_text(json.dumps(d, indent=2, sort_keys=False), encoding="utf-8")


def set_decision(key: str, value, why: str) -> dict:
    d = load_decisions()
    d[key] = value
    d.setdefault("_why", {})[key] = why
    save_decisions(d)
    return d
