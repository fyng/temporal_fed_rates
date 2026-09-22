"""Registry invariants."""

import re

import pytest

from fedrates import registry

REQUIRED_IDS = frozenset(
    {
        # policy
        "DFF",
        "FEDFUNDS",
        "EFFR",
        "DFEDTAR",
        "DFEDTARU",
        "DFEDTARL",
        "IOER",
        "IORB",
        "RRPONTSYAWARD",
        # inflation
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
        "PCEPILFE",
        # activity
        "UNRATE",
        "NROU",
        "GDPC1",
        "GDPPOT",
        "PAYEMS",
        "ICSA",
        # expectations
        "T10YIE",
        "T5YIE",
        "T5YIFR",
        "MICH",
        # financial
        "T10Y2Y",
        "DGS10",
        "DGS2",
        "NFCI",
        "ANFCI",
        "VIXCLS",
        # cycle
        "USREC",
    }
)

FREQS = {"d", "w", "m", "q"}
AGGS = {"mean", "last", "sum", "ffill"}
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_keys_equal_ids():
    for sid, series in registry.REGISTRY.items():
        assert series.id == sid


def test_all_required_ids_present():
    assert registry.REGISTRY.keys() >= REQUIRED_IDS


def test_columns_unique_and_snake_case():
    columns = [s.column for s in registry.REGISTRY.values()]
    assert len(columns) == len(set(columns))
    for column in columns:
        assert SNAKE_CASE.match(column), f"not snake_case: {column}"


def test_groups_known():
    for series in registry.REGISTRY.values():
        assert series.group in registry.GROUPS


def test_freq_and_agg_within_literals():
    for series in registry.REGISTRY.values():
        assert series.freq in FREQS, series.id
        assert series.agg in AGGS, series.id


def test_labels_cover_ids_and_columns():
    all_keys = {s.id for s in registry.REGISTRY.values()}
    all_keys |= {s.column for s in registry.REGISTRY.values()}
    labels = registry.labels()
    assert set(labels) == all_keys
    for s in registry.REGISTRY.values():
        assert labels[s.id] == s.name
        assert labels[s.column] == s.name


def test_labels_subset():
    labels = registry.labels(["DFF", "unrate"])
    assert labels == {"DFF": "Effective fed funds rate (daily)", "unrate": "Unemployment rate"}


def test_get_and_by_group():
    with pytest.raises(KeyError):
        registry.get("NOPE")
    with pytest.raises(KeyError):
        registry.by_group("nope")
    policy = registry.by_group("policy")
    assert {s.group for s in policy} == {"policy"}
    assert registry.ids(["policy"]) == [s.id for s in policy]
