import numpy as np
import pandas as pd
import pytest

from freight import config as C
from freight import io
from freight.december import build_rows
from freight.evaluate import load_clean
from freight.features import FeatureBuilder


@pytest.fixture(scope="module")
def frames():
    return load_clean()


def test_same_columns_and_dtypes(frames):
    tr, va, _ = frames
    dec = C.load_decisions()
    fb = FeatureBuilder(dec).fit(tr)
    a, b = fb.transform(tr), fb.transform(va)
    assert list(a.columns) == list(b.columns)
    assert (a.dtypes == b.dtypes).all()
    coords = io.city_coords()
    d = pd.Timestamp("2025-12-05")
    x = fb.transform(build_rows(coords, d, 0.9, np.array([2.0, 2.1]), np.array([0.0, 0.1])))
    assert list(x.columns) == list(a.columns) and (x.dtypes == a.dtypes).all()
    assert not x.isna().any().any()


def test_no_calendar_except_dow(frames):
    tr, _, _ = frames
    cols = FeatureBuilder(C.load_decisions()).fit(tr).transform(tr).columns
    assert "dow" in cols
    assert not {"month", "day", "doy", "date", "week", "time_index"} & set(cols)


def test_city_coords():
    assert len(io.city_coords()) == 72


def test_daily_median_qs(frames):
    tr, va, _ = frames
    for df in (tr, va):
        d = df["date"].iloc[100]
        m = df.loc[df["date"] == d, "quote_signal"].median()
        assert np.allclose(df.loc[df["date"] == d, "qs_daily_median"], m)
        assert np.allclose(df.loc[df["date"] == d, "qs_dev"], df.loc[df["date"] == d, "quote_signal"] - m)
