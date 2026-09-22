"""Shared fixtures: offline guarantee, cache helpers, synthetic frame."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import requests

__all__ = ["tmp_cache", "write_series", "no_network", "synthetic_monthly"]


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """Isolated cache directory for fetch tests.

    Returns:
        Path to an empty cache directory.
    """
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def write_series(tmp_cache: Path) -> Callable[[str, pd.Series], Path]:
    """Factory writing a series into the cache as a date,value CSV.

    Args:
        tmp_cache: Cache directory fixture.

    Returns:
        Callable(series_id, series) -> path to the written CSV.
    """

    def _write(series_id: str, series: pd.Series) -> Path:
        path = tmp_cache / f"{series_id}.csv"
        df = pd.DataFrame({"value": series.astype("float64")})
        df.index.name = "date"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        return path

    return _write


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Block any network access; tests marked pytest.mark.network are exempt."""
    import fredapi

    if request.node.get_closest_marker("network") is not None:
        return

    def _no_init(self, *args, **kwargs):
        raise AssertionError("network access in test")

    def _no_get(*args, **kwargs):
        raise AssertionError("network access in test")

    monkeypatch.setattr(fredapi.Fred, "__init__", _no_init)
    monkeypatch.setattr(requests, "get", _no_get)


@pytest.fixture
def synthetic_monthly() -> pd.DataFrame:
    """Hand-built monthly frame with exactly-known values.

    Columns: pce_core_yoy, u_gap, y_gap, policy_rate, nrou, unrate. Index is
    MS, named date, 24 rows from 2022-01 to 2023-12. Values follow one rule
    per column so tests can compute expectations exactly:

    - pce_core_yoy: 5.0 decreasing by 0.25 each month (5.00 -> 0.75)
    - u_gap: unrate - nrou
    - y_gap: 1.0 + row number / 10 (1.1 -> 3.4)
    - policy_rate: 0.25 + row number * 0.25 (0.25 -> 5.75)
    - nrou: 4.0 constant
    - unrate: 6.0 - row number / 10 (6.0 -> 3.7)
    """
    idx = pd.date_range("2022-01-01", periods=24, freq="MS", name="date")
    n = pd.RangeIndex(24)
    return pd.DataFrame(
        {
            "pce_core_yoy": 5.0 - 0.25 * n,
            "u_gap": (6.0 - 0.1 * n) - 4.0,
            "y_gap": 1.0 + 0.1 * n,
            "policy_rate": 0.25 + 0.25 * n,
            "nrou": 4.0,
            "unrate": 6.0 - 0.1 * n,
        },
        index=idx,
    )
