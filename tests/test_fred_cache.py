"""Fetch and cache behaviour (offline; the network is blocked by conftest)."""

import os
from datetime import date, timedelta

import pandas as pd
import pytest

from fedrates import fred


@pytest.fixture
def sample() -> pd.Series:
    """A small known series for cache round-trips."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="TEST")


def test_cache_round_trip(tmp_cache, write_series, sample):
    path = write_series("TEST", sample)
    out = fred.fetch("TEST", cache_dir=tmp_cache)
    pd.testing.assert_series_equal(out, sample, check_freq=False)
    assert out.dtype == "float64"
    assert out.name == "TEST"
    assert out.index.name == "date"
    assert isinstance(out.index, pd.DatetimeIndex)
    assert path.exists()


def test_is_stale(tmp_cache, write_series, sample):
    path = write_series("TEST", sample)
    assert not fred.is_stale(path, max_age_days=1.0)
    old = path.stat().st_mtime - timedelta(days=2).total_seconds()
    os.utime(path, (old, old))
    assert fred.is_stale(path, max_age_days=1.0)
    assert not fred.is_stale(path, max_age_days=3.0)
    assert fred.is_stale(tmp_cache / "MISSING.csv")


def test_cache_path_namespaces():
    root = fred.cache_path("TEST")
    first = fred.cache_path("TEST", "first_release")
    dated = fred.cache_path("TEST", date(2020, 1, 2))
    iso = fred.cache_path("TEST", "2020-01-02")
    assert len({root, first, dated}) == 3
    assert root != first and first != dated and root != dated
    assert root.name == "TEST.csv"
    assert first.parts[-3:-1] == ("vintage", "first")
    # A date object and its ISO string name the same vintage.
    assert dated == iso
    assert dated.parts[-3] == "vintage" and dated.parts[-2] == "2020-01-02"


def test_fresh_cache_never_builds_client(tmp_cache, write_series, sample, monkeypatch):
    write_series("TEST", sample)

    def _boom(*args, **kwargs):
        raise AssertionError("client constructed on fresh cache")

    monkeypatch.setattr(fred, "client", _boom)
    out = fred.fetch("TEST", cache_dir=tmp_cache)
    pd.testing.assert_series_equal(out, sample, check_freq=False)


def test_force_raises_without_client(tmp_cache, write_series, sample, monkeypatch):
    write_series("TEST", sample)

    def _boom(*args, **kwargs):
        raise RuntimeError("FRED_API_KEY not set; add it to .env or the environment")

    monkeypatch.setattr(fred, "client", _boom)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        fred.fetch("TEST", force=True, cache_dir=tmp_cache)
    # The cached file is left intact.
    assert (tmp_cache / "TEST.csv").exists()


def test_vintage_fetch_not_implemented(tmp_cache, write_series, sample):
    write_series("TEST", sample)
    with pytest.raises(NotImplementedError, match="vintage"):
        fred.fetch("TEST", vintage=date(2020, 1, 1), cache_dir=tmp_cache)
    with pytest.raises(NotImplementedError, match="vintage"):
        fred.fetch("TEST", vintage="first_release", cache_dir=tmp_cache)
    with pytest.raises(NotImplementedError, match="vintage"):
        fred.fetch_many(["TEST"], vintage="2020-01-01", cache_dir=tmp_cache)


def test_fetch_many_survives_bad_id(tmp_cache, write_series, sample, monkeypatch):
    write_series("GOOD", sample)

    def _boom(*args, **kwargs):
        raise RuntimeError("no client in tests")

    monkeypatch.setattr(fred, "client", _boom)
    with pytest.warns(UserWarning, match="BAD"):
        out = fred.fetch_many(["GOOD", "BAD"], cache_dir=tmp_cache, on_error="warn")
    assert set(out) == {"GOOD"}
    pd.testing.assert_series_equal(out["GOOD"], sample.rename("GOOD"), check_freq=False)

    with pytest.raises(RuntimeError, match="no client"):
        fred.fetch_many(["GOOD", "BAD"], cache_dir=tmp_cache, on_error="raise")


def test_clear_cache(tmp_cache, write_series, sample, monkeypatch):
    write_series("A", sample)
    write_series("B", sample)
    monkeypatch.setattr(fred, "CACHE_DIR", tmp_cache)
    assert fred.clear_cache(["A"]) == 1
    assert not (tmp_cache / "A.csv").exists()
    assert (tmp_cache / "B.csv").exists()
    assert fred.clear_cache() == 1
    assert list(tmp_cache.glob("*.csv")) == []
