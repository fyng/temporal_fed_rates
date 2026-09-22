"""FOMC decision record 1994-2026: the event spine for the decision model.

The Fed did not announce decisions before 1994-02-04 - markets inferred them
from Desk operations - so the record of announced decisions starts there. The
committed resource ``resources/fomc_meetings.csv`` is the hand-checked source
of truth; :func:`build_meetings` only proposes a rebuild for human review and
must never feed an automated path. The ``dissents`` columns are hand-curated
from statement pages: vote rolls appear on those pages from 2002-03-19 onward
(rows before that, and 2026-06-17/2026-09-16 where no roll was published, are
null); the builder emits them null.
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import fred

__all__ = ["MEETINGS_CSV", "CHAIRS", "CLASSES",
           "load_meetings", "chair_at", "label_actions", "state_at_meetings", "build_meetings"]

# --- Chairs and classes -------------------------------------------------------

CHAIRS: tuple[tuple[str, str, str], ...] = (
    ("Greenspan", "1987-08-11", "2006-01-31"),
    ("Bernanke", "2006-02-01", "2014-01-31"),
    ("Yellen", "2014-02-01", "2018-02-04"),
    ("Powell", "2018-02-05", "2026-05-21"),
    ("Warsh", "2026-05-22", ""),
)

CLASSES: tuple[str, ...] = ("cut50+", "cut25", "hold", "hike25", "hike50+")

# --- Committed record ---------------------------------------------------------

MEETINGS_CSV = "resources/fomc_meetings.csv"

_COLUMNS: tuple[str, ...] = (
    "date", "end_date", "intermeeting", "action", "size_bp", "target",
    "target_lower", "target_upper", "dissents", "dissent_names", "chair",
    "statement_url", "notes",
)
_READ_DTYPES: dict[str, str] = {
    "intermeeting": "bool",
    "size_bp": "int64",
    "target": "float64",
    "target_lower": "float64",
    "target_upper": "float64",
    "dissents": "Int64",
}
_RANGE_ERA = pd.Timestamp("2008-12-16")  # first day of the target-range regime
_PROPOSAL_PATH = Path(__file__).resolve().parents[2] / "data" / "interim" / "fomc_meetings.csv"


@lru_cache(maxsize=1)
def _read_resource() -> pd.DataFrame:
    """Read the committed meetings CSV with the schema dtypes.

    Returns:
        The full committed record; treat as read-only.
    """
    with as_file(files("fedrates") / MEETINGS_CSV) as path:
        frame = pd.read_csv(path, parse_dates=["date", "end_date"], dtype=_READ_DTYPES)
    return frame


def load_meetings(
    *, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None,
    scheduled_only: bool = False,
) -> pd.DataFrame:
    """Load the committed meetings record.

    Args:
        start: Inclusive lower bound on the decision date.
        end: Inclusive upper bound on the decision date.
        scheduled_only: Drop intermeeting decisions.

    Returns:
        Frame with the :data:`MEETINGS_CSV` schema, filtered as requested.
    """
    out = _read_resource()
    mask = pd.Series(True, index=out.index)
    if start is not None:
        mask &= out["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= out["date"] <= pd.Timestamp(end)
    if scheduled_only:
        mask &= ~out["intermeeting"]
    return out[mask].reset_index(drop=True)


def chair_at(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """Map dates to the sitting Fed chair.

    Args:
        dates: Dates to map.

    Returns:
        Chair name per date.

    Raises:
        ValueError: If any date precedes the first chair start in CHAIRS.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    starts = pd.DatetimeIndex([pd.Timestamp(start) for _, start, _ in CHAIRS])
    pos = starts.searchsorted(idx, side="right") - 1
    if (pos < 0).any():
        raise ValueError(f"date(s) before the first chair start {starts[0].date()}")
    names = [CHAIRS[int(p)][0] for p in pos]
    index = dates.index if isinstance(dates, pd.Series) else idx
    return pd.Series(names, index=index, name="chair", dtype=object)


def label_actions(meetings: pd.DataFrame) -> pd.Series:
    """Classify decisions by ``size_bp`` into the ordered CLASSES scale.

    Args:
        meetings: Frame with a ``size_bp`` column.

    Returns:
        Ordered categorical over CLASSES aligned with ``meetings``.
    """
    size = meetings["size_bp"].to_numpy(dtype="float64")
    labels = np.select(
        [size <= -50, size < 0, size == 0, size < 50],
        ["cut50+", "cut25", "hold", "hike25"],
        default="hike50+",
    )
    out = pd.Series(labels, index=meetings.index, name="label", dtype=object)
    return out.astype(pd.CategoricalDtype(CLASSES, ordered=True))


def state_at_meetings(
    frame: pd.DataFrame,
    meetings: pd.DataFrame | None = None,
    *,
    lag_months: int = 1,
    cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join analysis-frame rows known before each meeting; the leakage boundary.

    A meeting on date d is joined to the frame row for month d - lag_months,
    never the month containing d: CPI for month m is published mid-month m+1,
    so a naive as-of join hands the model data the committee did not have.

    Args:
        frame: Monthly analysis frame on a month-start DatetimeIndex.
        meetings: Meetings frame with a ``date`` column; load_meetings() when None.
        lag_months: Months of publication lag to enforce; 1 is the conservative
            default, 0 selects the meeting's own month and is not leak-safe.
        cols: Frame columns to join; all when None.

    Returns:
        Frame indexed by meeting date with the joined row per meeting; months
        outside the frame come back NaN.
    """
    meetings = load_meetings() if meetings is None else meetings
    dates = pd.DatetimeIndex(pd.to_datetime(meetings["date"]))
    keys = (dates - pd.DateOffset(months=lag_months)).to_period("M").to_timestamp()
    source = frame if cols is None else frame[list(cols)]
    joined = source.reindex(keys)
    joined.index = dates.rename("date")
    return joined


# --- Calendar scrape (rebuild only) -------------------------------------------

_FED = "https://www.federalreserve.gov"
_HISTORICAL_URL = _FED + "/monetarypolicy/fomchistorical{year}.htm"
_CALENDARS_URL = _FED + "/monetarypolicy/fomccalendars.htm"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research data collection"}
_MEETING_RE = re.compile(
    r"^(?:(?P<m0>[A-Za-z]+)\s*/\s*)?(?P<month>[A-Za-z]+)\.?\s+(?P<d1>\d{1,2})"
    r"(?:\s*[-–]\s*(?:(?P<m1>[A-Za-z]+)\.?\s+)?(?P<d2>\d{1,2}))?"
    r"(?:\s*\([^)]*\))?\s+(?P<kind>Meeting|Conference Call)\s*-\s*(?P<year>\d{4})$"
)
_ROW_DAYS_RE = re.compile(r"^(\d{1,2})(?:[-–](\d{1,2}))?\*?$")
_URL_DATE_RE = re.compile(r"(\d{8})")
_STATEMENT_HREF_RE = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass(frozen=True)
class _Panel:
    """One Fed-calendar event: a meeting or a conference call.

    Args:
        first: First day of the meeting.
        last: Last day of the meeting; the decision day for scheduled meetings.
        scheduled: False for conference calls and unscheduled video meetings.
        statement_url: Statement link from the calendar page, if any.
        statement_date: Announcement date parsed from the statement URL.
    """

    first: pd.Timestamp
    last: pd.Timestamp
    scheduled: bool
    statement_url: str | None
    statement_date: pd.Timestamp | None


def _get(url: str) -> str:
    """Fetch a page as text.

    Args:
        url: Page URL.

    Returns:
        The page HTML.
    """
    response = requests.get(url, headers=_HEADERS, timeout=60)
    response.raise_for_status()
    return response.text


def _statement_link(anchor: BeautifulSoup) -> tuple[str | None, pd.Timestamp | None]:
    """Extract a statement URL and its announcement date from a calendar block.

    Args:
        anchor: A calendar panel or row element.

    Returns:
        (absolute statement URL, announcement date); (None, None) if absent.
    """
    for a in anchor.select("a[href]"):
        href = a["href"]
        if a.get_text(strip=True).strip().lower() == "statement":
            found = _URL_DATE_RE.search(href)
            url = href if href.startswith("http") else _FED + href
            return url, (pd.Timestamp(found.group(1)) if found else None)
        found = _STATEMENT_HREF_RE.search(href)
        if found:
            return _FED + href, pd.Timestamp(found.group(1))
    return None, None


def _month(name: str) -> int:
    """Resolve a month name or abbreviation to its number.

    Args:
        name: Month name in any capitalisation.

    Returns:
        Month number, 1-12.

    Raises:
        KeyError: If the name is not a known month.
    """
    return _MONTHS[name.strip().lower()]


def _scrape_historical(year: int) -> list[_Panel]:
    """Scrape one fomchistoricalYYYY page into panels.

    Args:
        year: Page year.

    Returns:
        Panels for every meeting and conference call on the page.
    """
    soup = BeautifulSoup(_get(_HISTORICAL_URL.format(year=year)), "html.parser")
    out: list[_Panel] = []
    seen: set[str] = set()
    for panel in soup.select("div.panel"):
        head = panel.select_one("div.panel-heading") or panel.select_one("h5")
        text = head.get_text(" ", strip=True) if head else ""
        if not text or text in seen or "(cancelled)" in text or "(notation vote)" in text:
            continue
        seen.add(text)
        match = _MEETING_RE.match(text)
        if match is None:
            if "(unscheduled)" not in text:
                warnings.warn(f"unparsed FOMC panel heading: {text!r}", stacklevel=2)
            continue
        d1, d2 = int(match["d1"]), int(match["d2"] or match["d1"])

        url, statement_date = _statement_link(panel)
        out.append(
            _Panel(
                # "Jul/Aug 31-1" and "June 30-July 1" span two months of one year.
                first=pd.Timestamp(year=year, month=_month(match["m0"] or match["month"]), day=d1),
                last=pd.Timestamp(year=year, month=_month(match["m1"] or match["month"]), day=d2),
                scheduled="conference call" not in text.lower() and "(unscheduled)" not in text,
                statement_url=url,
                statement_date=statement_date,
            )
        )
    return out


def _scrape_calendars(years: set[int]) -> list[_Panel]:
    """Scrape the recent calendars page for the requested years.

    Args:
        years: Year numbers to keep.

    Returns:
        Panels for every listed meeting.
    """
    soup = BeautifulSoup(_get(_CALENDARS_URL), "html.parser")
    out: list[_Panel] = []
    seen: set[tuple[int, int, int]] = set()
    for panel in soup.select("div.panel"):
        head = panel.select_one("h5") or panel.select_one("div.panel-heading")
        text = head.get_text(" ", strip=True) if head else ""
        match = re.match(r"^(\d{4}) FOMC Meetings", text)
        if match is None or int(match.group(1)) not in years:
            continue
        year = int(match.group(1))
        for row in panel.select("div.row"):
            leaves = [d for d in row.find_all("div", recursive=True) if not d.find("div")]
            texts = [d.get_text(" ", strip=True) for d in leaves]
            # "Jan/Feb 31-1" labels a meeting spanning two months of one year.
            label = next((t for t in texts if t.lower() in _MONTHS or "/" in t), None)
            days = next((_ROW_DAYS_RE.match(t) for t in texts if _ROW_DAYS_RE.match(t)), None)
            if label is None or days is None:
                continue
            months = label.split("/")
            try:
                m1 = _month(months[-1])
                m0 = _month(months[0]) if len(months) > 1 else m1
            except KeyError:
                continue
            d1, d2 = int(days.group(1)), int(days.group(2) or days.group(1))
            if (year, m0, d1) in seen:
                continue
            seen.add((year, m0, d1))
            url, statement_date = _statement_link(row)
            out.append(
                _Panel(
                    first=pd.Timestamp(year=year, month=m0, day=d1),
                    last=pd.Timestamp(year=year, month=m1, day=d2),
                    scheduled="(unscheduled)" not in row.get_text(),
                    statement_url=url,
                    statement_date=statement_date,
                )
            )
    return out


# --- FRED derivation (rebuild only) -------------------------------------------


def _steps(series: pd.Series, start: pd.Timestamp) -> pd.Series:
    """Signed target steps (percent) at or after ``start``.

    Args:
        series: Spliced target series on a daily index.
        start: First date to consider.

    Returns:
        Series of step sizes indexed by the step date.
    """
    diff = series.diff()
    return diff[(diff.index >= start) & (diff.abs() > 1e-9)]


def _post_target(
    effective: pd.Timestamp, date: pd.Timestamp, spliced: pd.Series,
    lower: pd.Series, upper: pd.Series,
) -> tuple[float, float, float]:
    """Post-decision target values under the two regimes.

    Args:
        effective: Date the new value took effect (step date, or the meeting
            date for holds).
        date: Decision date; selects the regime at the 2008-12-16 splice.
        spliced: Spliced single-point target series.
        lower: DFEDTARL series.
        upper: DFEDTARU series.

    Returns:
        (target, target_lower, target_upper) with NaN where not applicable.
    """
    if date < _RANGE_ERA:
        return round(float(spliced.asof(effective)), 4), np.nan, np.nan
    return np.nan, round(float(lower.asof(effective)), 4), round(float(upper.asof(effective)), 4)


# --- Builder -------------------------------------------------------------------

# 2007-08-17 changed no target: the FOMC cut the discount rate 50bp intermeeting
# and flagged downside risks. No FRED step can derive it, so it is curated.
_CURATED_EVENTS: tuple[dict, ...] = (
    {
        "date": pd.Timestamp("2007-08-17"),
        "intermeeting": True,
        "action": "hold",
        "size_bp": 0,
        "statement_url": _FED + "/newsevents/press/monetary/20070817b.htm",
        "notes": "Intermeeting: discount rate cut 50bp; funds target unchanged at 5.25%",
    },
)


def build_meetings(
    out: Path | None = None, *, start: int = 1994, end: int = 2026
) -> pd.DataFrame:
    """Derive the meetings table from Fed calendar pages and FRED; proposes only.

    Decision dates come from the calendar's statement links, actions and sizes
    from a spliced DFEDTAR / (DFEDTARU+DFEDTARL)/2 series, holds from scheduled
    meetings with no step. Dissents are always null: they are hand-curated in
    the committed CSV, which is the source of truth - review before committing.

    Args:
        out: CSV path to write; no file is written when None.
        start: First year of meetings to include.
        end: Last year of meetings to include.

    Returns:
        Frame with the committed CSV schema. Needs network access and the
        FRED cache; never call it under the test-time network ban.
    """
    today = pd.Timestamp.today().normalize()
    panels: list[_Panel] = []
    for year in range(start, min(end, 2020) + 1):
        panels.extend(_scrape_historical(year))
    if end >= 2021:
        panels.extend(_scrape_calendars({y for y in range(2021, end + 1) if y <= today.year + 1}))

    old = fred.fetch("DFEDTAR")
    upper = fred.fetch("DFEDTARU")
    lower = fred.fetch("DFEDTARL")
    spliced = pd.concat([old, (upper + lower) / 2]).sort_index()
    steps = _steps(spliced, pd.Timestamp(year=start, month=1, day=1))

    scheduled = sorted(
        (p for p in panels if p.scheduled and (p.statement_date or p.last) <= today),
        key=lambda p: p.statement_date or p.last,
    )
    # Some Fed pages split one meeting into adjacent one-day panels; keep the
    # panel carrying the statement (the decision day).
    kept: list[_Panel] = []

    def _key(p: _Panel) -> pd.Timestamp:
        return p.statement_date or p.last

    for panel in scheduled:
        if kept and _key(panel) - _key(kept[-1]) <= pd.Timedelta(days=1):
            if panel.statement_date is not None or kept[-1].statement_date is None:
                kept[-1] = panel
            warnings.warn(f"adjacent scheduled panels kept {_key(kept[-1]).date()}", stacklevel=2)
            continue
        kept.append(panel)
    scheduled = kept
    statements = {
        p.statement_date: p.statement_url for p in panels if p.statement_date is not None
    }
    rows: list[dict] = []
    matched: set[pd.Timestamp] = set()
    for effective, move in steps.items():
        hit = next(
            (p for p in scheduled
             if effective - pd.Timedelta(days=1) <= _key(p) <= effective),
            None,
        )
        if hit is not None:
            date, intermeeting = hit.statement_date or hit.last, False
            matched.add(id(hit))
        else:
            call = next(
                (p for p in panels if not p.scheduled and p.statement_date is not None
                 and p.statement_date <= effective <= p.statement_date + pd.Timedelta(days=1)),
                None,
            )
            if call is None:
                warnings.warn(f"unmatched FRED step {effective.date()} {move:+.3f}", stacklevel=2)
                date, intermeeting = effective, True
            else:
                date, intermeeting = call.statement_date, True
        size_bp = int(round(move * 100))
        target, lo, hi = _post_target(effective, date, spliced, lower, upper)
        rows.append({
            "date": date,
            "end_date": date,
            "intermeeting": intermeeting,
            "action": "cut" if size_bp < 0 else ("hike" if size_bp > 0 else "hold"),
            "size_bp": size_bp,
            "target": target,
            "target_lower": lo,
            "target_upper": hi,
            "dissents": pd.NA,
            "dissent_names": None,
            "chair": chair_at(pd.DatetimeIndex([date])).iloc[0],
            "statement_url": statements.get(date),
            "notes": (
                "Midpoint move -87.5bp, recorded as -88bp"
                if date == _RANGE_ERA else None
            ),
        })
    for panel in scheduled:
        if id(panel) in matched:
            continue
        date = panel.statement_date or panel.last
        target, lo, hi = _post_target(date, date, spliced, lower, upper)
        rows.append({
            "date": date,
            "end_date": date,
            "intermeeting": False,
            "action": "hold",
            "size_bp": 0,
            "target": target,
            "target_lower": lo,
            "target_upper": hi,
            "dissents": pd.NA,
            "dissent_names": None,
            "chair": chair_at(pd.DatetimeIndex([date])).iloc[0],
            "statement_url": statements.get(date),
            "notes": None,
        })
    for event in _CURATED_EVENTS:
        if start <= event["date"].year <= end:
            date = event["date"]
            target, lo, hi = _post_target(date, date, spliced, lower, upper)
            rows.append({
                **event,
                "end_date": date,
                "target": target,
                "target_lower": lo,
                "target_upper": hi,
                "dissents": pd.NA,
                "dissent_names": None,
                "chair": chair_at(pd.DatetimeIndex([date])).iloc[0],
            })
    frame = pd.DataFrame(rows, columns=_COLUMNS).sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        warnings.warn("duplicate meeting dates in derived record", stacklevel=2)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, na_rep="")
    return frame


# --- CLI -----------------------------------------------------------------------


def _main() -> None:
    """CLI entry point proposing a rebuild of the committed CSV."""
    parser = argparse.ArgumentParser(
        description="Propose a rebuild of the FOMC meetings CSV for human review"
    )
    parser.add_argument("--rebuild", action="store_true", help="scrape calendars and rebuild")
    parser.add_argument("--out", type=Path, default=None, help="proposal path; "
                        f"default {_PROPOSAL_PATH}")
    args = parser.parse_args()
    if not args.rebuild:
        parser.error("pass --rebuild")
    frame = build_meetings(args.out)
    print(f"{len(frame)} decisions {frame['date'].min().date()} -> {frame['date'].max().date()}")
    print(f"proposal written to {args.out or _PROPOSAL_PATH}")
    print(label_actions(frame).value_counts().sort_index().to_string())
    print(f"intermeeting: {int(frame['intermeeting'].sum())}")


if __name__ == "__main__":
    _main()
