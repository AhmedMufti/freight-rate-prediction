from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_predictions_format():
    p = ROOT / "validation_predictions.csv"
    if not p.exists():
        pytest.skip("run the pipeline first")
    df = pd.read_csv(p)
    assert list(df.columns) == ["load_id", "predicted_rate"] and len(df) == 12000
    assert df["load_id"].is_unique and df["load_id"].iloc[0] == "TE-000001" and df["load_id"].iloc[-1] == "TE-012000"
    assert df["predicted_rate"].notna().all() and (df["predicted_rate"] > 0).all()


def test_december_format():
    p = ROOT / "data" / "december_chart_inputs.csv"
    if not p.exists():
        pytest.skip("data missing")
    df = pd.read_csv(p, dtype=str)
    assert list(df.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if df["predicted_rate"].isna().all():
        pytest.skip("december not filled yet")
    assert len(df) == 31 and df["date"].is_unique and df["date"].min() == "2025-12-01"
    assert (df["predicted_rate"].astype(float) > 0).all()
