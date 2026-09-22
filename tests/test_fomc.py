"""Tests for fedrates.fomc: committed record, labels, chairs, leakage boundary."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd
import pytest

from fedrates.fomc import (
    CHAIRS,
    CLASSES,
    MEETINGS_CSV,
    chair_at,
    label_actions,
    load_meetings,
    state_at_meetings,
)

# --- Committed resource ------------------------------------------------------


def test_resource_loads_via_importlib() -> None:
    """The committed CSV resolves inside the installed package."""
    path = files("fedrates") / MEETINGS_CSV
    assert path.is_file()


def test_schema_columns_and_dtypes() -> None:
    """Exact schema: columns, dtypes, parsed dates, end_date equals date."""
    df = load_meetings()
    assert tuple(df.columns) == (
        "date", "end_date", "intermeeting", "action", "size_bp", "target",
        "target_lower", "target_upper", "dissents", "dissent_names", "chair",
        "statement_url", "notes",
    )
    assert pd.api.types.is_datetime64_dtype(df["date"])
    assert pd.api.types.is_datetime64_dtype(df["end_date"])
    assert pd.api.types.is_bool_dtype(df["intermeeting"])
    assert pd.api.types.is_integer_dtype(df["size_bp"])
    for col in ("target", "target_lower", "target_upper"):
        assert pd.api.types.is_float_dtype(df[col])
    assert str(df["dissents"].dtype) == "Int64"
    assert (df["end_date"] == df["date"]).all()


def test_dates_sorted_unique_and_recent() -> None:
    """Dates strictly increasing, unique, within 1994-02-04 to today."""
    df = load_meetings()
    dates = pd.DatetimeIndex(df["date"])
    assert dates.is_monotonic_increasing
    assert dates.is_unique
    assert dates.min() >= pd.Timestamp("1994-01-01")
    assert dates.min() == pd.Timestamp("1994-02-04")
    assert dates.max() <= pd.Timestamp.today().normalize()


def test_action_sign_matches_size() -> None:
    """action agrees with sign(size_bp) on every row."""
    df = load_meetings()
    expect = pd.Series(
        pd.cut(df["size_bp"], [-float("inf"), -1e-9, 1e-9, float("inf")],
               labels=["cut", "hold", "hike"]),
        name="action",
    ).astype(str)
    assert (df["action"] == expect).all()


def test_target_regime_columns() -> None:
    """Single-point target before 2008-12-16, range after, never both."""
    df = load_meetings()
    era = pd.Timestamp("2008-12-16")
    old = df[df["date"] < era]
    new = df[df["date"] >= era]
    assert old["target"].notna().all()
    assert old["target_lower"].isna().all()
    assert old["target_upper"].isna().all()
    assert new["target"].isna().all()
    assert new["target_lower"].notna().all()
    assert new["target_upper"].notna().all()
    assert (new["target_upper"] >= new["target_lower"]).all()


@pytest.mark.parametrize(
    ("date", "size_bp", "intermeeting", "lo", "hi"),
    [
        ("1994-02-04", 25, False, None, None),
        ("2008-12-16", -88, False, 0.0, 0.25),
        ("2020-03-15", -100, True, 0.0, 0.25),
        ("2022-06-15", 75, False, 1.50, 1.75),
        ("2026-09-16", 25, False, 3.75, 4.00),
    ],
)
def test_spot_checks(
    date: str, size_bp: int, intermeeting: bool, lo: float | None, hi: float | None
) -> None:
    """Hand-checked rows anchor the record at known decisions."""
    row = load_meetings().set_index("date").loc[pd.Timestamp(date)]
    assert int(row["size_bp"]) == size_bp
    assert bool(row["intermeeting"]) == intermeeting
    assert row["action"] == ("hike" if size_bp > 0 else "cut")
    if lo is None:
        assert pd.isna(row["target_lower"])
    else:
        assert row["target_lower"] == lo
        assert row["target_upper"] == hi


def test_nine_intermeeting_dates() -> None:
    """All nine well-known intermeeting moves are present and flagged."""
    nine = [
        "1998-10-15", "2001-01-03", "2001-04-18", "2001-09-17", "2007-08-17",
        "2008-01-22", "2008-10-08", "2020-03-03", "2020-03-15",
    ]
    df = load_meetings().set_index("date")
    for d in nine:
        assert pd.Timestamp(d) in df.index
        assert bool(df.loc[pd.Timestamp(d), "intermeeting"])


def test_load_meetings_filters() -> None:
    """start/end/scheduled_only narrow the record as documented."""
    full = load_meetings()
    sub = load_meetings(start="2010-01-01", end="2019-12-31")
    assert sub["date"].min() >= pd.Timestamp("2010-01-01")
    assert sub["date"].max() <= pd.Timestamp("2019-12-31")
    sched = load_meetings(scheduled_only=True)
    assert not sched["intermeeting"].any()
    assert len(sched) == int((~full["intermeeting"]).sum())


# --- Labels ------------------------------------------------------------------


def test_label_actions_ordered_categories() -> None:
    """label_actions returns an ordered categorical over exactly CLASSES."""
    df = load_meetings()
    labels = label_actions(df)
    assert list(labels.cat.categories) == list(CLASSES)
    assert labels.cat.ordered
    assert labels.notna().all()


@pytest.mark.parametrize(
    ("size_bp", "label"),
    [(-100, "cut50+"), (-50, "cut50+"), (-25, "cut25"), (-1, "cut25"),
     (0, "hold"), (1, "hike25"), (49, "hike25"), (50, "hike50+"), (75, "hike50+")],
)
def test_label_edges(size_bp: int, label: str) -> None:
    """Class boundaries follow the documented bp cut-offs."""
    frame = pd.DataFrame({"size_bp": [size_bp]})
    assert label_actions(frame).iloc[0] == label


def test_label_distribution() -> None:
    """Holds dominate the 1994-2026 record and every class is populated."""
    counts = label_actions(load_meetings()).value_counts()
    assert counts["hold"] > len(load_meetings()) * 0.5
    assert set(counts.index) == set(CLASSES)


# --- Chairs ------------------------------------------------------------------


def test_chair_at_full_span_no_gaps() -> None:
    """chair_at covers 1988-2030 with no nulls and honours tenure boundaries."""
    dates = pd.date_range("1988-01-01", "2030-01-01", freq="D")
    chairs = chair_at(dates)
    assert chairs.notna().all()
    assert set(chairs.unique()) == {name for name, _, _ in CHAIRS}


@pytest.mark.parametrize(
    ("date", "chair"),
    [
        ("1987-08-11", "Greenspan"),
        ("1994-02-04", "Greenspan"),
        ("2006-01-31", "Greenspan"),
        ("2006-02-01", "Bernanke"),
        ("2014-01-31", "Bernanke"),
        ("2014-02-01", "Yellen"),
        ("2018-02-04", "Yellen"),
        ("2018-02-05", "Powell"),
        ("2026-05-21", "Powell"),
        ("2026-05-22", "Warsh"),
        ("2026-09-16", "Warsh"),
    ],
)
def test_chair_at_boundaries(date: str, chair: str) -> None:
    """CHAIRS start/end dates switch the chair exactly on the boundary."""
    assert chair_at(pd.DatetimeIndex([pd.Timestamp(date)])).iloc[0] == chair


def test_chair_at_before_first_start_raises() -> None:
    """Dates before Greenspan's start raise ValueError."""
    with pytest.raises(ValueError, match="before the first chair start"):
        chair_at(pd.DatetimeIndex([pd.Timestamp("1980-01-01")]))


def test_meetings_chair_column_matches_chair_at() -> None:
    """The committed chair column agrees with chair_at on every row."""
    df = load_meetings()
    assert (df["chair"].to_numpy() == chair_at(pd.DatetimeIndex(df["date"])).to_numpy()).all()


# --- Dissents ----------------------------------------------------------------


def test_dissents_non_negative_ints() -> None:
    """Non-null dissents are non-negative integers."""
    dissents = load_meetings()["dissents"].dropna()
    assert (dissents >= 0).all()


def test_dissent_names_count_matches() -> None:
    """dissent_names lists exactly dissents names where both are present."""
    df = load_meetings()
    both = df[df["dissents"].notna() & df["dissent_names"].notna()]
    for _, row in both.iterrows():
        assert len(str(row["dissent_names"]).split(";")) == int(row["dissents"])


# --- Leakage boundary --------------------------------------------------------


def _meetings_frame(dates: list[str]) -> pd.DataFrame:
    """Meetings frame with one decision per given date."""
    return pd.DataFrame({"date": pd.to_datetime(dates)})


def test_state_at_meetings_lags_one_month(synthetic_monthly: pd.DataFrame) -> None:
    """A March meeting sees February data, never March's own row."""
    meetings = _meetings_frame(["2022-03-16", "2023-06-14"])
    joined = state_at_meetings(synthetic_monthly, meetings, lag_months=1)
    assert list(joined.index) == list(pd.to_datetime(meetings["date"]))
    row = joined.loc[pd.Timestamp("2022-03-16")]
    source = synthetic_monthly.loc[pd.Timestamp("2022-02-01")]
    assert row["pce_core_yoy"] == source["pce_core_yoy"] == 4.75
    assert row["policy_rate"] == source["policy_rate"] == 0.50
    row = joined.loc[pd.Timestamp("2023-06-14")]
    source = synthetic_monthly.loc[pd.Timestamp("2023-05-01")]
    assert row["unrate"] == source["unrate"] == 4.4
    assert row["y_gap"] == source["y_gap"] == 2.6


def test_state_at_meetings_lag_three(synthetic_monthly: pd.DataFrame) -> None:
    """lag_months=3 reaches three months back and NaN outside the frame."""
    meetings = _meetings_frame(["2022-03-16", "2023-12-13"])
    joined = state_at_meetings(synthetic_monthly, meetings, lag_months=3)
    # 2022-03 - 3 months = 2021-12, before the frame starts
    assert joined.loc[pd.Timestamp("2022-03-16")].isna().all()
    # 2023-12 - 3 months = 2023-09, inside the frame
    source = synthetic_monthly.loc[pd.Timestamp("2023-09-01")]
    row = joined.loc[pd.Timestamp("2023-12-13")]
    assert row["pce_core_yoy"] == source["pce_core_yoy"]
    assert row["unrate"] == source["unrate"]


def test_state_at_meetings_never_uses_meeting_month_or_later(
    synthetic_monthly: pd.DataFrame,
) -> None:
    """No joined row is dated on or after the meeting's month: the leakage guard.

    For every meeting date d the joined source month is exactly d - lag_months
    and always strictly before d's own month.
    """
    dates = pd.date_range("2022-04-01", "2023-12-01", freq="MS")
    meetings = pd.DataFrame(
        {"date": dates + pd.Timedelta(days=15)}
    )  # mid-month meetings, every frame month from 2022-04 (lag-3 keys stay inside)
    for lag in (1, 3):
        joined = state_at_meetings(synthetic_monthly, meetings, lag_months=lag)
        meeting_months = pd.DatetimeIndex(meetings["date"]).to_period("M")
        source_months = (
            pd.DatetimeIndex(meetings["date"]) - pd.DateOffset(months=lag)
        ).to_period("M")
        # the joined values must equal the frame row for the lagged month
        for d, key in zip(meetings["date"], source_months, strict=True):
            expect = synthetic_monthly.loc[key.to_timestamp()]
            got = joined.loc[d]
            pd.testing.assert_series_equal(got, expect, check_names=False)
        # and the lagged month is always strictly before the meeting month
        assert (source_months < meeting_months).all()
        assert joined.notna().all().all()


def test_state_at_meetings_default_meetings_and_cols(
    synthetic_monthly: pd.DataFrame,
) -> None:
    """cols selects a subset; meetings=None loads the committed record."""
    meetings = _meetings_frame(["2023-01-18"])
    joined = state_at_meetings(synthetic_monthly, meetings, cols=["policy_rate"])
    assert list(joined.columns) == ["policy_rate"]
    # 2023-01 - 1 month = 2022-12 -> n = 11
    assert joined.loc[pd.Timestamp("2023-01-18"), "policy_rate"] == pytest.approx(3.0)
