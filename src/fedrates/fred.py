"""FRED fetching with a file cache under ``data/raw``."""

from __future__ import annotations

import argparse
import os
import warnings
from collections.abc import Iterable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from . import registry

__all__ = ["CACHE_DIR", "fetch", "fetch_many", "client", "cache_path", "is_stale", "clear_cache"]

Vintage = str | date | Literal["first_release"] | None

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@lru_cache(maxsize=1)
def client(api_key: str | None = None) -> Fred:
    """Build the FRED client, loading ``.env`` for the key.

    Args:
        api_key: Explicit key; ``FRED_API_KEY`` from the environment otherwise.

    Returns:
        A ``fredapi.Fred`` instance.

    Raises:
        RuntimeError: If no key is found.
    """
    load_dotenv()
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set; add it to .env or the environment")
    return Fred(api_key=key)


def cache_path(series_id: str, vintage: Vintage = None, cache_dir: Path | None = None) -> Path:
    """Resolve the cache file for a series and vintage.

    Args:
        series_id: FRED series id.
        vintage: None for the latest, ``"first_release"`` for first-vintage
            data, or a date for a point-in-time vintage.
        cache_dir: Cache root; CACHE_DIR when None.

    Returns:
        Path to the CSV cache file.
    """
    root = cache_dir if cache_dir is not None else CACHE_DIR
    if vintage is None:
        return root / f"{series_id}.csv"
    if vintage == "first_release":
        return root / "vintage" / "first" / f"{series_id}.csv"
    day = vintage if isinstance(vintage, date) else date.fromisoformat(str(vintage))
    return root / "vintage" / day.isoformat() / f"{series_id}.csv"


def is_stale(path: Path, max_age_days: float = 1.0) -> bool:
    """Whether a cache file is missing or older than the age limit.

    Args:
        path: Cache file path.
        max_age_days: Maximum age in days.

    Returns:
        True when the file is absent or too old.
    """
    if not path.exists():
        return True
    age = pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
    return age > pd.Timedelta(days=max_age_days)


def _read_cache(path: Path, series_id: str) -> pd.Series:
    """Read a cached CSV into a float series indexed by date.

    Args:
        path: Cache file path.
        series_id: Name given to the returned series.

    Returns:
        The cached series.
    """
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    out = df["value"].astype("float64")
    out.index = pd.DatetimeIndex(out.index, name="date")
    out.name = series_id
    return out


def _write_cache(series: pd.Series, path: Path) -> None:
    """Write a series to the cache as a date,value CSV.

    Args:
        series: Series to write.
        path: Cache file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"value": series.astype("float64")})
    df.index.name = "date"
    df.to_csv(path)


def _refetch(series_id: str, start, end) -> pd.Series:
    """Fetch one series from FRED, bypassing the cache.

    Args:
        series_id: FRED series id.
        start: Start date or None.
        end: End date or None.

    Returns:
        The fetched series.
    """
    kwargs: dict = {}
    if start is not None:
        kwargs["observation_start"] = start
    if end is not None:
        kwargs["observation_end"] = end
    raw = client().get_series(series_id, **kwargs)
    out = pd.Series(raw.astype("float64"), name=series_id)
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def fetch(
    series_id: str,
    *,
    start=None,
    end=None,
    vintage: Vintage = None,
    force: bool = False,
    max_age_days: float = 1.0,
    cache_dir: Path | None = None,
) -> pd.Series:
    """Fetch a series, serving from cache when fresh.

    A fresh (non-stale) cache never touches the network or the environment.
    The FRED client is constructed only when a refetch is required.

    Args:
        series_id: FRED series id.
        start: Start date for a refetch.
        end: End date for a refetch.
        vintage: None for latest; any other value is not yet implemented.
        force: Refetch even when the cache is fresh.
        max_age_days: Cache age limit in days.
        cache_dir: Cache root; CACHE_DIR when None.

    Returns:
        The series, float64, indexed by date.

    Raises:
        NotImplementedError: If a vintage is requested.
        RuntimeError: If a refetch is needed but no API key is available.
    """
    if vintage is not None:
        raise NotImplementedError("vintage fetching not yet implemented")
    path = cache_path(series_id, None, cache_dir)
    if not force and not is_stale(path, max_age_days):
        return _read_cache(path, series_id)
    out = _refetch(series_id, start, end)
    _write_cache(out, path)
    return out


def fetch_many(
    series_ids: Iterable[str] | None = None,
    *,
    vintage: Vintage = None,
    force: bool = False,
    max_age_days: float = 1.0,
    cache_dir: Path | None = None,
    on_error: Literal["raise", "warn"] = "warn",
) -> dict[str, pd.Series]:
    """Fetch several series at their native frequencies.

    Args:
        series_ids: FRED ids; all registry ids when None.
        vintage: None for latest; any other value is not yet implemented.
        force: Refetch even when the cache is fresh.
        max_age_days: Cache age limit in days.
        cache_dir: Cache root; CACHE_DIR when None.
        on_error: "raise" to stop at the first failure, "warn" to skip it.

    Returns:
        Dict of id to series; failed ids omitted under "warn".

    Raises:
        NotImplementedError: If a vintage is requested.
        RuntimeError: If a refetch is needed but no API key is available.
    """
    if vintage is not None:
        raise NotImplementedError("vintage fetching not yet implemented")
    out: dict[str, pd.Series] = {}
    for sid in series_ids if series_ids is not None else registry.ids():
        try:
            out[sid] = fetch(
                sid,
                vintage=None,
                force=force,
                max_age_days=max_age_days,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            if on_error == "raise":
                raise
            warnings.warn(f"fetch failed for {sid}: {exc}", stacklevel=2)
    return out


def clear_cache(series_ids: Iterable[str] | None = None) -> int:
    """Delete cache files for the given ids.

    Args:
        series_ids: FRED ids; all cached CSVs at the top level when None.

    Returns:
        Number of files deleted.
    """
    if series_ids is None:
        paths = list(CACHE_DIR.glob("*.csv"))
    else:
        paths = [CACHE_DIR / f"{sid}.csv" for sid in series_ids]
    n = 0
    for path in paths:
        if path.exists():
            path.unlink()
            n += 1
    return n


def _main() -> None:
    """CLI entry point for refreshing and inspecting the cache."""
    parser = argparse.ArgumentParser(description="Fetch FRED series into the data/raw cache")
    parser.add_argument("--all", action="store_true", help="fetch every registry series")
    parser.add_argument("--group", help="fetch one registry group")
    parser.add_argument("--force", action="store_true", help="refetch even when cached")
    args = parser.parse_args()
    if not args.all and args.group is None:
        parser.error("choose --all or --group GROUP")
    sids = registry.ids() if args.all else registry.ids([args.group])
    out = fetch_many(sids, force=args.force, on_error="warn")
    for sid, series in out.items():
        print(f"{sid:>14}  {len(series):>6} obs  {series.index.min().date()} -> "
              f"{series.index.max().date()}")


if __name__ == "__main__":
    _main()
