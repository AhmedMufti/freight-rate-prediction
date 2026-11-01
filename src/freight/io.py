"""Data loading. Handles both the brief's underscore names (data/) and the original hyphen names (raw/)."""
from __future__ import annotations

import shutil

import pandas as pd

from . import config as C


def path_of(kind: str):
    us, hy, _ = C.FILES[kind]
    p = C.DATA / us
    if p.exists():
        return p
    q = C.RAW / hy
    if q.exists():
        return q
    # original flat layout (files next to the repo root)
    return C.ROOT / hy


def _read(kind: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path_of(kind), **kw)


def load_train() -> pd.DataFrame:
    return _read("train")


def load_val() -> pd.DataFrame:
    return _read("val")


def load_template() -> pd.DataFrame:
    return _read("template", dtype=str)


def load_december() -> pd.DataFrame:
    return _read("december", dtype=str)


def city_coords(train: pd.DataFrame | None = None, val: pd.DataFrame | None = None) -> pd.DataFrame:
    """One (lat, lon) per city from the union of train and validation (both pickup and delivery)."""
    train = load_train() if train is None else train
    val = load_val() if val is None else val
    parts = []
    for df in (train, val):
        parts.append(df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).groupby("city")[["lat", "lon"]].first()


def prepare_data() -> str:
    """Copy raw files to data/ with underscore names. Idempotent; never overwrites a good file."""
    C.DATA.mkdir(exist_ok=True)
    msgs = []
    for kind, (us, hy, n) in C.FILES.items():
        dst = C.DATA / us
        if dst.exists() and len(pd.read_csv(dst, usecols=[0])) == n:
            msgs.append(f"{us}: ok")
            continue
        src = C.RAW / hy
        if not src.exists():
            src = C.ROOT / hy
        shutil.copyfile(src, dst)
        msgs.append(f"{us}: copied")
    return "; ".join(msgs)
