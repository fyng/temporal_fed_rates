"""Tests for fedrates.dataset."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from fedrates import COLUMNS, dataset, fred, registry, rules, sources, theme, transforms

_START = "2008-01-01"
_END = "2026-06-30"


# --- Synthetic FRED universe -------------------------------------------------


def _elb(idx: pd.DatetimeIndex) -> np.ndarray:
    """Mask of the two synthetic ELB windows."""
    first = (idx >= "2009-01-01") & (idx <= "2015-12-31")
    second = (idx >= "2020-03-01") & (idx <= "2022-02-28")
    return first | second


def _synthetic_series(sid: str) -> pd.Series:
    """Hand-built FRED series for one registry id, with exactly-known values."""
    meta = registry.get(sid)
    freq = {"d": "D", "w": "W-MON", "m": "MS", "q": "QS"}[meta.freq]
    start = pd.Timestamp(meta.start or _START)
    end = pd.Timestamp(_END)
    if sid == "DFEDTAR":
        end = pd.Timestamp("2008-12-15")
    if sid == "IOER":
        end = pd.Timestamp("2021-07-28")
    idx = pd.date_range(start, end, freq=freq, name="date")
    n = len(idx)
    t = np.arange(n, dtype=float)
    values: np.ndarray
    if sid in {"DFF", "EFFR", "FEDFUNDS"}:
        values = np.where(_elb(idx), 0.0, 2.0)
    elif sid == "DFEDTAR":
        values = np.full(n, 3.5)
    elif sid == "DFEDTARU":
        values = np.full(n, 4.5)
    elif sid == "DFEDTARL":
        values = np.full(n, 3.5)
    elif sid == "IOER":
        values = np.full(n, 0.10)
    elif sid == "IORB":
        values = np.full(n, 0.10)
    elif sid == "RRPONTSYAWARD":
        values = np.full(n, 0.05)
    elif sid in {"CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"}:
        values = 100.0 * 1.002**t
    elif sid == "UNRATE":
        hot = (idx >= "2010-01-01") & (idx <= "2013-12-31")
        values = np.where(hot, 10.0, 5.0)
    elif sid == "MICH":
        values = np.full(n, 2.5)
    elif sid == "PAYEMS":
        values = 100_000.0 + 100.0 * t
    elif sid == "ICSA":
        values = np.full(n, 250_000.0)
    elif sid == "NFCI":
        values = np.zeros(n)
    elif sid == "ANFCI":
        values = np.full(n, 0.1)
    elif sid in {"T10YIE", "T5YIE", "T5YIFR"}:
        values = np.full(n, 2.0)
    elif sid == "T10Y2Y":
        values = np.full(n, 0.5)
    elif sid == "DGS10":
        values = np.full(n, 3.0)
    elif sid == "DGS2":
        values = np.full(n, 2.5)
    elif sid == "DTB3":
        values = np.full(n, 3.0)
    elif sid == "DTB6":
        values = np.full(n, 3.25)
    elif sid == "VIXCLS":
        values = np.full(n, 20.0)
    elif sid == "NROU":
        values = np.full(n, 4.5)
    elif sid == "GDPC1":
        values = 1000.0 * 1.005**t
    elif sid == "GDPPOT":
        values = 1000.0 * 1.004**t
    elif sid == "USREC":
        spans = ((idx >= "2008-01-01") & (idx <= "2009-06-01")) | (
            (idx >= "2020-02-01") & (idx <= "2020-04-01")
        )
        values = spans.astype(float)
    else:  # pragma: no cover - every registry id is handled above
        raise KeyError(f"no synthetic values for {sid}")
    return pd.Series(values.astype("float64"), index=idx, name=sid)


def _fake_fetch_many(series_ids=None, *, vintage=None, force=False, **kw):
    ids = list(series_ids) if series_ids is not None else registry.ids()
    return {sid: _synthetic_series(sid) for sid in ids}


def _fake_wu_xia(*args, **kw) -> pd.Series:
    idx = pd.date_range("2010-01-01", "2022-02-01", freq="MS", name="date")
    values = np.where(idx <= "2015-12-01", -1.5, np.where(idx <= "2019-12-01", 0.25, -0.5))
    return pd.Series(values.astype("float64"), index=idx, name="shadow_rate")


@pytest.fixture
def synthetic_raw(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Serve the synthetic universe in place of FRED and the Wu-Xia workbook."""
    monkeypatch.setattr(fred, "fetch_many", _fake_fetch_many)
    monkeypatch.setattr(sources, "wu_xia", _fake_wu_xia)
    yield


@pytest.fixture
def synthetic_frame(synthetic_raw) -> pd.DataFrame:
    """Monthly synthetic frame spanning the ELB windows."""
    return dataset.build_frame(start=_START)


# --- Contract: columns -------------------------------------------------------


def test_build_frame_emits_full_contract(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.name == "date"
    assert (frame.index.day == 1).all()
    assert (frame.dtypes == "float64").all()
    assert set(frame.columns) == set(COLUMNS)


def test_columns_documentation_agrees_both_ways(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame
    assert len(frame.columns) == len(COLUMNS)
    assert set(frame.columns) == set(COLUMNS)
    undocumented = set(frame.columns) - set(COLUMNS)
    missing = set(COLUMNS) - set(frame.columns)
    assert not undocumented and not missing


def test_column_count() -> None:
    assert len(COLUMNS) == 54


def test_quarterly_build(synthetic_raw) -> None:
    frame = dataset.build_frame("q", start=_START)
    assert (frame.index.day == 1).all()
    assert set(frame.columns) == set(COLUMNS)
    assert (frame.dtypes == "float64").all()


# --- policy_rate splice ------------------------------------------------------


def test_policy_rate_splice_windows(synthetic_frame: pd.DataFrame) -> None:
    pr = synthetic_frame["policy_rate"]
    assert pr.notna().all()
    assert (pr.loc["2009-01":"2009-12"] == 0.0).all()  # ELB, no shadow yet
    assert (pr.loc["2010-01":"2015-12"] == -1.5).all()
    assert (pr.loc["2016-01":"2019-12"] == 2.0).all()  # shadow exists, rate above ELB
    assert (pr.loc["2020-03":"2022-02"] == -0.5).all()
    assert (pr.loc["2022-03":] == 2.0).all()


def test_policy_rate_continuous_with_elb_stretch(synthetic_frame: pd.DataFrame) -> None:
    pr = synthetic_frame["policy_rate"]
    assert pr.index.is_monotonic_increasing
    assert pr.notna().all()


def test_shadow_off_keeps_effective_rate(synthetic_raw) -> None:
    frame = dataset.build_frame(start=_START, shadow=False)
    assert frame["policy_rate"].equals(frame["ffr"])
    assert frame["shadow_rate"].notna().any()


# --- ior splice --------------------------------------------------------------


def test_ior_splice_monthly(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame
    ior = frame["ior"].loc["2021-01-01":"2022-06-01"]
    assert ior.notna().all()
    assert ior.loc["2021-07-01"] == pytest.approx(0.10)  # IOER leg
    assert ior.loc["2021-08-01"] == pytest.approx(0.10)  # IORB leg
    assert ior.diff().abs().max() == pytest.approx(0.0)  # continuous across the cutover
    # IOER does not leak past its 2021-07-28 end
    assert frame["ioer"].last_valid_index() == pd.Timestamp("2021-07-01")
    # IORB contributes nothing before 2021-07-29
    assert frame["iorb"].first_valid_index() == pd.Timestamp("2021-07-01")


# --- r_star ------------------------------------------------------------------


def test_r_star_constant_column(synthetic_frame: pd.DataFrame) -> None:
    assert (synthetic_frame["r_star"] == 2.0).all()


def test_r_star_hlw_path(synthetic_raw, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2008-01-01", periods=74, freq="QS", name="date")
    hlw = pd.Series(1.0 + 0.1 * np.arange(74, dtype=float), index=idx)
    monkeypatch.setattr(sources, "hlw", lambda *a, **kw: hlw)
    frame = dataset.build_frame(start=_START, r_star="hlw")
    expected = transforms.quarterly_to_monthly(hlw, "ffill").reindex(frame.index)
    pd.testing.assert_series_equal(frame["r_star"], expected.rename("r_star"), check_freq=False)
    row = frame.loc["2015-06-01"]
    pi = row["pce_core_yoy"]
    assert row["rule_taylor93_unfloored"] == pytest.approx(
        row["r_star"] + pi + 0.5 * (pi - 2.0) - row["u_gap"]
    )


def test_r_star_invalid_string(synthetic_raw) -> None:
    with pytest.raises(ValueError, match="hlw"):
        dataset.build_frame(start=_START, r_star="nope")


# --- Rules -------------------------------------------------------------------


def test_rules_floor_and_pairing(synthetic_frame: pd.DataFrame) -> None:
    floored = [c for c in COLUMNS if c.startswith("rule_") and not c.endswith("_unfloored")]
    unfloored = [c for c in COLUMNS if c.endswith("_unfloored")]
    assert len(floored) == 5 and len(unfloored) == 5
    assert synthetic_frame[floored].dropna(how="all").ge(rules.ELB).all().all()
    assert (synthetic_frame[unfloored] < rules.ELB).any().any()


def test_taylor93_value(synthetic_frame: pd.DataFrame) -> None:
    row = synthetic_frame.loc["2015-06-01"]
    pi = row["pce_core_yoy"]
    assert pi == pytest.approx(100.0 * (1.002**12 - 1))
    assert row["rule_taylor93_unfloored"] == pytest.approx(
        2.0 + pi + 0.5 * (pi - 2.0) - row["u_gap"]
    )


# --- Persistence -------------------------------------------------------------


def test_save_load_roundtrip(tmp_path, monkeypatch, synthetic_frame) -> None:
    monkeypatch.setattr(dataset, "PROCESSED_DIR", tmp_path)
    path = dataset.save_frame(synthetic_frame, "roundtrip")
    assert path == tmp_path / "roundtrip.csv"
    loaded = dataset.load_frame("roundtrip")
    assert loaded.index.name == "date"
    assert (loaded.dtypes == "float64").all()
    pd.testing.assert_frame_equal(loaded, synthetic_frame, check_freq=False)


def test_load_frame_builds_when_missing(tmp_path, monkeypatch, synthetic_raw) -> None:
    monkeypatch.setattr(dataset, "PROCESSED_DIR", tmp_path)
    frame = dataset.load_frame("built", start=_START)
    assert (tmp_path / "built.csv").exists()
    assert set(frame.columns) == set(COLUMNS)
    again = dataset.load_frame("built")
    pd.testing.assert_frame_equal(again, frame)


def test_frame_path_default_name() -> None:
    assert dataset.frame_path().name == "analysis_monthly.csv"
    assert dataset.frame_path("x").name == "x.csv"


# --- tidy --------------------------------------------------------------------


def test_tidy_shape_labels_groups(synthetic_frame: pd.DataFrame) -> None:
    cols = ["ffr", "policy_rate", "pce_core_yoy", "rule_taylor93"]
    long = dataset.tidy(synthetic_frame, cols)
    assert list(long.columns) == ["date", "series", "value", "label", "group"]
    assert set(long["series"]) == set(cols)
    assert long["label"].notna().all()
    assert long["group"].notna().all()
    assert set(long["group"]) == {"policy", "derived", "rule"}
    assert long.groupby("series")["label"].nunique().eq(1).all()


def test_tidy_full_frame_no_orphans(synthetic_frame: pd.DataFrame) -> None:
    long = dataset.tidy(synthetic_frame)
    assert set(long["series"]) == set(COLUMNS)
    assert long["label"].notna().all()
    assert long["group"].notna().all()
    expected_groups = {"derived", "rule"} | set(registry.GROUPS)
    assert set(long["group"]) <= expected_groups


def test_tidy_unknown_column_raises(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="label"):
        dataset.tidy(synthetic_frame, ["not_a_column"])


# --- recession_spans ---------------------------------------------------------


def test_recession_spans_two_known_runs() -> None:
    idx = pd.date_range("2008-01-01", periods=24, freq="MS", name="date")
    rec = np.zeros(24)
    rec[0:6] = 1.0
    rec[14:17] = 1.0
    frame = pd.DataFrame({"recession": rec}, index=idx)
    spans = dataset.recession_spans(frame)
    assert spans == [
        (pd.Timestamp("2008-01-01"), pd.Timestamp("2008-06-01")),
        (pd.Timestamp("2009-03-01"), pd.Timestamp("2009-05-01")),
    ]


def test_recession_spans_default_path(synthetic_raw, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dataset, "PROCESSED_DIR", tmp_path)
    spans = dataset.recession_spans()
    assert spans == [
        (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-06-01")),
        (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")),
    ]


def test_recession_spans_feed_period_shading(synthetic_frame: pd.DataFrame) -> None:
    fig = go.Figure(go.Scatter(x=synthetic_frame.index, y=synthetic_frame["policy_rate"]))
    spans = dataset.recession_spans(synthetic_frame)
    assert all(a < b for a, b in spans)
    assert all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))
    out = theme.period_shading(fig, spans)
    assert len(out.layout.shapes) == len(spans)


# --- Real data (offline from the local cache) --------------------------------


@pytest.fixture
def cached_fred(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Serve strictly from the on-disk cache regardless of file age."""
    monkeypatch.setattr(fred, "is_stale", lambda path, max_age_days=1.0: False)
    monkeypatch.setattr(sources, "download", lambda url, path, **kw: path)
    yield


def test_real_frame_offline(cached_fred) -> None:
    frame = dataset.build_frame()
    assert frame.loc["2026-09-01", "target_upper"] == pytest.approx(4.0)
    assert (frame.loc["2009":"2015", "rule_taylor93_unfloored"] < -2).any()
    assert frame["policy_rate"].notna().all()
    assert (frame.loc["2010":"2015", "policy_rate"] < 0).any()
    assert (frame.loc["2020":"2021", "policy_rate"] < 0).any()
    spans = dataset.recession_spans(frame)
    assert all(a < b for a, b in spans)
    assert all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))


def test_ior_splice_offline(cached_fred) -> None:
    frame = dataset.build_frame()
    assert frame["ior"].loc["2013-09-01":].notna().all()
    ioer = fred.fetch("IOER")
    iorb = fred.fetch("IORB")
    assert ioer.index.max() == pd.Timestamp("2021-07-28")
    assert iorb.index.min() == pd.Timestamp("2021-07-29")
    ior = transforms.splice(iorb, ioer, at="2021-07-29")
    boundary = pd.Timestamp("2021-07-29")
    step = abs(ior.asof(boundary) - ior.asof(boundary - pd.Timedelta(days=1)))
    assert step <= 0.05  # set equal across the 2021 rename
    effr = fred.fetch("EFFR").reindex(ior.index)
    spread_bp = ((effr - ior) * 100.0).dropna().loc["2014":]
    assert len(spread_bp) > 2500
    assert (spread_bp < 0).mean() > 0.8  # EFFR prints below the rate on reserves
