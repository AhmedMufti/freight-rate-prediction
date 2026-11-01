import pytest

from freight.evaluate import load_clean
from freight.splits import make_folds, random_kfold_contrast, unseen_city_fold


@pytest.fixture(scope="module")
def tr():
    return load_clean()[0]


def test_no_leakage_and_inner_holdout(tr):
    for n, f in make_folds(tr).items():
        assert tr["date"].iloc[f.fit].max() < tr["date"].iloc[f.test].min(), n
        assert tr["month"].iloc[f.es].nunique() == 1
        assert tr["month"].iloc[f.es].iloc[0] == tr["month"].iloc[f.fit].max()
        assert tr["month"].iloc[f.inner_fit].max() < tr["month"].iloc[f.es].iloc[0]


def test_test_sizes(tr):
    f = make_folds(tr)
    for n in ("F1", "F2", "F3"):
        assert 9000 < len(f[n].test) < 10500


def test_unseen_masks(tr):
    f, masked = unseen_city_fold(tr, seed=42)
    fit = tr.iloc[f.fit]
    assert not (fit["pickup"].isin(masked) | fit["delivery"].isin(masked)).any()
    t = tr.iloc[f.test]
    assert (t["pickup"].isin(masked) | t["delivery"].isin(masked)).all()
    assert len(masked) == 8


def test_random_kfold(tr):
    parts = list(random_kfold_contrast(len(tr)))
    assert len(parts) == 5 and sum(len(te) for _, te in parts) == len(tr)
