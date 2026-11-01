import numpy as np
import pandas as pd
import pytest

from freight import io
from freight.clean import OutlierModel, clean_frame, data_quality, mi_daily_series, weight_medians


@pytest.fixture(scope="module")
def frames():
    tr, va = io.load_train(), io.load_val()
    return tr, va, mi_daily_series(tr, va)


def test_counts(frames):
    tr, va, mi = frames
    q = data_quality(tr, va, mi)
    assert (q["train"]["weight_negative"], q["validation"]["weight_negative"]) == (292, 145)
    assert (q["train"]["weight_missing"], q["validation"]["weight_missing"]) == (300, 165)
    assert (q["train"]["weight_at_cap_47500"], q["validation"]["weight_at_cap_47500"]) == (1191, 297)
    assert (q["train"]["market_index_missing"], q["validation"]["market_index_missing"]) == (374, 249)
    assert 0.6 <= q["train"]["label_outlier_pct"] <= 1.0
    for s in ("train", "validation"):
        assert q[s]["nan_weight_after_clean"] == 0 and q[s]["nan_mi_after_clean"] == 0


def test_weight_abs_and_flags(frames):
    tr, _, mi = frames
    c = clean_frame(tr, mi, weight_medians(tr))
    assert (c["weight"] > 0).all()
    assert c["weight_neg_flag"].sum() == 292


def test_outlier_model_flags_extremes():
    df = pd.DataFrame({"lane": ["a|b"] * 6, "equipment": ["Dry Van"] * 6, "distance": [100.0] * 6})
    rate = pd.Series([1000, 1010, 990, 1005, 4000, 200.0])
    fl = OutlierModel().fit(df, rate).flag(df, rate)
    assert list(fl) == [False, False, False, False, True, True]
