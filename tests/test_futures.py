"""Tests for fedrates.futures: Kuttner surprises and FedWatch-style probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedrates import futures
from fedrates.futures import CLASSES, benchmark, expected_proba, kuttner
from fedrates.sources import RAW_DIR

_WORKBOOK = RAW_DIR / "monetary-policy-surprises-data.xlsx"


# --- kuttner ------------------------------------------------------------------


def test_kuttner_scales_by_days_remaining() -> None:
    """Mid-month surprises scale FF1 by D/(D-d); remaining 7 is not end of month."""
    dates = pd.DatetimeIndex(["2022-06-10", "2023-03-24"])  # remaining 20 and 7
    out = kuttner(dates, [0.25, 0.07], [np.nan, 99.0])
    assert out.to_numpy() == pytest.approx([37.5, 7.0 * 31 / 7])


def test_kuttner_switches_to_next_month_contract_at_month_end() -> None:
    """The last 7 days of the month read unscaled FF2."""
    dates = pd.DatetimeIndex(["2019-07-31", "2021-07-28"])  # remaining 0 and 3
    out = kuttner(dates, [0.25, 0.25], [-0.10, 0.08])
    assert out.to_numpy() == pytest.approx([-10.0, 8.0])


def test_kuttner_nan_where_contract_missing() -> None:
    """A missing print on the contract in use gives NaN."""
    dates = pd.DatetimeIndex(["2022-06-10", "2022-06-30"])
    out = kuttner(dates, [np.nan, np.nan], [0.05, 0.05])
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(5.0)


# --- expected_proba -------------------------------------------------------------


def test_expected_proba_splits_between_bracketing_multiples() -> None:
    """-12.5bp splits 50/50 between cut25 and hold; 37.5 between the hikes."""
    index = pd.DatetimeIndex(["2024-01-01", "2024-02-01"], name="date")
    p = expected_proba(pd.Series([-12.5, 37.5], index=index))
    assert list(p.columns) == list(CLASSES)
    assert p.to_numpy() == pytest.approx(
        np.array([[0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.5]])
    )


def test_expected_proba_point_masses_at_multiples() -> None:
    """Each multiple of 25 maps to a point mass on its class."""
    p = expected_proba(np.array([-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0]))
    expected = np.array(
        [
            [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
        ]
    )
    assert p.to_numpy() == pytest.approx(expected)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_expected_proba_clips_and_skips_non_finite() -> None:
    """Expectations clip to [-100, +100]; non-finite values give all-NaN rows."""
    p = expected_proba(pd.Series([-300.0, 250.0, np.nan]))
    assert p.iloc[0].to_numpy() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])
    assert p.iloc[1].to_numpy() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])
    assert p.iloc[2].isna().all()


# --- surprises and benchmark ------------------------------------------------------


def _write_workbook(path: Path, rows: list[dict]) -> None:
    """Write a surprises worksheet in the SF Fed layout."""
    pd.DataFrame(rows, columns=["Date", "FF1", "FF2"]).to_excel(
        path, sheet_name="FOMC (update 2023)", index=False
    )


def test_surprises_parses_worksheet(tmp_path: Path) -> None:
    """The parser reads ff1/ff2 by decision date and keeps the last duplicate."""
    path = tmp_path / "surprises.xlsx"
    _write_workbook(path, [
        {"Date": "2010-01-27", "FF1": 0.000, "FF2": -0.005},
        {"Date": "2022-06-15", "FF1": 0.0215, "FF2": 0.0400},
        {"Date": "2022-06-15", "FF1": 0.0100, "FF2": 0.0200},
    ])
    sur = futures.surprises(path)
    assert list(sur.columns) == ["ff1", "ff2"]
    assert sur.index.name == "date"
    assert sur.loc["2022-06-15", "ff1"] == pytest.approx(0.0100)


def test_benchmark_shape_and_drops(tmp_path: Path) -> None:
    """Hand-built workbook: alignment on date, uncovered meetings dropped.

    2010-01-27 (remaining 4) reads FF2 -0.5bp, so a hold has an expected
    +0.5bp split 98/2 between hold and hike25; 2022-06-15 (remaining 15)
    scales FF1 2.15bp by 30/15 to 4.3bp, so a 75bp hike expects +70.7bp ->
    hike50+; 2019-07-31 (remaining 0) reads FF2 -10bp, so a 25bp cut expects
    -15bp -> 0.6 cut25 / 0.4 hold. The 2009 meeting has no print and the
    2024 meeting has no row, so both are dropped.
    """
    path = tmp_path / "surprises.xlsx"
    _write_workbook(path, [
        {"Date": "2010-01-27", "FF1": 0.000, "FF2": -0.005},
        {"Date": "2019-07-31", "FF1": np.nan, "FF2": -0.100},
        {"Date": "2022-06-15", "FF1": 0.0215, "FF2": 0.0400},
        {"Date": "2009-12-16", "FF1": np.nan, "FF2": np.nan},
    ])
    meetings = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2009-12-16", "2010-01-27", "2019-07-31", "2022-06-15", "2024-01-31"]
            ),
            "size_bp": [0, 0, -25, 75, 0],
        }
    )
    out = benchmark(meetings, path)
    assert list(out.index) == list(pd.to_datetime(["2010-01-27", "2019-07-31", "2022-06-15"]))
    assert list(out.columns) == [*CLASSES, "label"]
    assert out["label"].tolist() == ["hold", "cut25", "hike50+"]
    assert out.loc["2010-01-27"].to_numpy()[:5] == pytest.approx(
        [0.0, 0.0, 0.98, 0.02, 0.0]
    )
    assert out.loc["2019-07-31"].to_numpy()[:5] == pytest.approx(
        [0.0, 0.6, 0.4, 0.0, 0.0]
    )
    assert out.loc["2022-06-15"].to_numpy()[:5] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 1.0]
    )


@pytest.mark.skipif(not _WORKBOOK.exists(), reason="surprises workbook not cached")
def test_surprises_cached_workbook(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real worksheet parses: ff1/ff2 by date, ending with the 2023 update."""
    monkeypatch.setattr(futures, "_download", lambda *a, **k: None)
    sur = futures.surprises()
    assert sur.index.max() == pd.Timestamp("2023-12-13")
    assert sur.loc["2022-06-15"].to_numpy() == pytest.approx([0.0215, 0.0400])
