"""Catalogue of FRED series: ids, labels, groups and aggregation rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

__all__ = ["Series", "REGISTRY", "GROUPS", "get", "by_group", "ids", "labels"]

Frequency = Literal["d", "w", "m", "q"]
Aggregation = Literal["mean", "last", "sum", "ffill"]


@dataclass(frozen=True, slots=True)
class Series:
    """One FRED series and how it enters the analysis frame.

    Args:
        id: FRED series id.
        name: Chart-ready human label.
        column: Snake_case name in the analysis frame.
        group: One of GROUPS.
        freq: Native frequency code.
        agg: Aggregation rule when collapsing to a lower frequency.
        units: Units of the series.
        start: First date to request, when the full history is not wanted.
        source: Data source.
        notes: Short caveat.
    """

    id: str
    name: str
    column: str
    group: str
    freq: Frequency
    agg: Aggregation
    units: str = "percent"
    start: str | None = None
    source: str = "FRED"
    notes: str = ""


# Price indices use "last" because YoY is computed after the monthly collapse;
# policy targets use "last" (end-of-period stance); rates and VIX use "mean".
GROUPS: tuple[str, ...] = (
    "policy",
    "inflation",
    "activity",
    "expectations",
    "financial",
    "cycle",
)

REGISTRY: dict[str, Series] = {
    # --- policy -------------------------------------------------------------
    "DFF": Series(
        id="DFF",
        name="Effective fed funds rate (daily)",
        column="ffr",
        group="policy",
        freq="d",
        agg="mean",
    ),
    "FEDFUNDS": Series(
        id="FEDFUNDS",
        name="Effective fed funds rate (monthly)",
        column="ffr_m",
        group="policy",
        freq="m",
        agg="mean",
    ),
    "EFFR": Series(
        id="EFFR",
        name="Effective fed funds rate (EFFR)",
        column="effr",
        group="policy",
        freq="d",
        agg="mean",
        start="2000-07-03",
    ),
    "DFEDTAR": Series(
        id="DFEDTAR",
        name="Fed funds target (old)",
        column="target_old",
        group="policy",
        freq="d",
        agg="last",
        notes="Discontinued 2008-12-15; single-point target",
    ),
    "DFEDTARU": Series(
        id="DFEDTARU",
        name="Fed funds target, upper",
        column="target_upper",
        group="policy",
        freq="d",
        agg="last",
        start="2008-12-16",
    ),
    "DFEDTARL": Series(
        id="DFEDTARL",
        name="Fed funds target, lower",
        column="target_lower",
        group="policy",
        freq="d",
        agg="last",
        start="2008-12-16",
    ),
    "IOER": Series(
        id="IOER",
        name="Interest on excess reserves",
        column="ioer",
        group="policy",
        freq="d",
        agg="last",
        notes="Replaced by IORB 2021",
    ),
    "IORB": Series(
        id="IORB",
        name="Interest on reserve balances",
        column="iorb",
        group="policy",
        freq="d",
        agg="last",
        start="2021-07-29",
        notes="Replaced IOER 2021",
    ),
    "RRPONTSYAWARD": Series(
        id="RRPONTSYAWARD",
        name="Overnight reverse repo rate",
        column="onrrp",
        group="policy",
        freq="d",
        agg="last",
        start="2013-09-23",
    ),
    # --- inflation ----------------------------------------------------------
    "CPIAUCSL": Series(
        id="CPIAUCSL",
        name="CPI, all items",
        column="cpi",
        group="inflation",
        freq="m",
        agg="last",
        units="index",
    ),
    "CPILFESL": Series(
        id="CPILFESL",
        name="CPI, core",
        column="cpi_core",
        group="inflation",
        freq="m",
        agg="last",
        units="index",
    ),
    "PCEPI": Series(
        id="PCEPI",
        name="PCE price index",
        column="pce",
        group="inflation",
        freq="m",
        agg="last",
        units="index",
    ),
    "PCEPILFE": Series(
        id="PCEPILFE",
        name="PCE price index, core",
        column="pce_core",
        group="inflation",
        freq="m",
        agg="last",
        units="index",
    ),
    # --- activity -----------------------------------------------------------
    "UNRATE": Series(
        id="UNRATE",
        name="Unemployment rate",
        column="unrate",
        group="activity",
        freq="m",
        agg="mean",
    ),
    "NROU": Series(
        id="NROU",
        name="Natural rate of unemployment (CBO)",
        column="nrou",
        group="activity",
        freq="q",
        agg="ffill",
        notes="CBO; may be superseded",
    ),
    "GDPC1": Series(
        id="GDPC1",
        name="Real GDP",
        column="gdp",
        group="activity",
        freq="q",
        agg="ffill",
        units="$bn, 2017 chained",
    ),
    "GDPPOT": Series(
        id="GDPPOT",
        name="Real potential GDP (CBO)",
        column="gdp_pot",
        group="activity",
        freq="q",
        agg="ffill",
        units="$bn, 2017 chained",
    ),
    "PAYEMS": Series(
        id="PAYEMS",
        name="Nonfarm payrolls",
        column="payems",
        group="activity",
        freq="m",
        agg="last",
        units="thousands",
    ),
    "ICSA": Series(
        id="ICSA",
        name="Initial jobless claims",
        column="claims",
        group="activity",
        freq="w",
        agg="mean",
        units="persons",
    ),
    # --- expectations -------------------------------------------------------
    "T10YIE": Series(
        id="T10YIE",
        name="10-year breakeven inflation",
        column="bei10",
        group="expectations",
        freq="d",
        agg="mean",
        start="2003-01-02",
    ),
    "T5YIE": Series(
        id="T5YIE",
        name="5-year breakeven inflation",
        column="bei5",
        group="expectations",
        freq="d",
        agg="mean",
    ),
    "T5YIFR": Series(
        id="T5YIFR",
        name="5-year, 5-year forward inflation",
        column="fwd5y5y",
        group="expectations",
        freq="d",
        agg="mean",
        start="2003-01-02",
    ),
    "MICH": Series(
        id="MICH",
        name="Michigan inflation expectations",
        column="mich",
        group="expectations",
        freq="m",
        agg="last",
    ),
    # --- financial ----------------------------------------------------------
    "T10Y2Y": Series(
        id="T10Y2Y",
        name="10y-2y Treasury spread",
        column="spread_10y2y",
        group="financial",
        freq="d",
        agg="mean",
    ),
    "DGS10": Series(
        id="DGS10",
        name="10-year Treasury yield",
        column="dgs10",
        group="financial",
        freq="d",
        agg="mean",
    ),
    "DGS2": Series(
        id="DGS2",
        name="2-year Treasury yield",
        column="dgs2",
        group="financial",
        freq="d",
        agg="mean",
    ),
    "NFCI": Series(
        id="NFCI",
        name="Chicago Fed financial conditions",
        column="nfci",
        group="financial",
        freq="w",
        agg="last",
        units="index",
    ),
    "ANFCI": Series(
        id="ANFCI",
        name="Chicago Fed adjusted financial conditions",
        column="anfci",
        group="financial",
        freq="w",
        agg="last",
        units="index",
    ),
    "VIXCLS": Series(
        id="VIXCLS",
        name="VIX",
        column="vix",
        group="financial",
        freq="d",
        agg="mean",
        units="index",
    ),
    # --- cycle --------------------------------------------------------------
    "USREC": Series(
        id="USREC",
        name="NBER recession indicator",
        column="recession",
        group="cycle",
        freq="m",
        agg="last",
        units="0/1",
    ),
}


def get(series_id: str) -> Series:
    """Return the Series for a FRED id.

    Args:
        series_id: FRED series id.

    Returns:
        The registry entry.

    Raises:
        KeyError: If the id is not in the registry.
    """
    return REGISTRY[series_id]


def by_group(group: str) -> list[Series]:
    """Return registry entries in a group, in registry order.

    Args:
        group: One of GROUPS.

    Returns:
        List of Series.

    Raises:
        KeyError: If the group is unknown.
    """
    if group not in GROUPS:
        raise KeyError(f"unknown group {group!r}; expected one of {GROUPS}")
    return [s for s in REGISTRY.values() if s.group == group]


def ids(groups: Iterable[str] | None = None) -> list[str]:
    """Return FRED ids, optionally filtered to groups, in registry order.

    Args:
        groups: Groups to keep; all when None.

    Returns:
        List of FRED ids.
    """
    keep = None if groups is None else set(groups)
    return [s.id for s in REGISTRY.values() if keep is None or s.group in keep]


def labels(keys: Iterable[str] | None = None) -> dict[str, str]:
    """Map FRED ids and frame column names to human labels.

    Args:
        keys: Ids or column names to keep; all when None.

    Returns:
        Dict of key to chart-ready label.
    """
    keep = None if keys is None else set(keys)
    out: dict[str, str] = {}
    for s in REGISTRY.values():
        for key in (s.id, s.column):
            if keep is None or key in keep:
                out[key] = s.name
    return out
