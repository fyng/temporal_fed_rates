"""Tests for the interactive web figures."""

import re
from pathlib import Path

import pytest

from fedrates.web import figures

ARTICLE = Path(__file__).resolve().parents[1] / "docs" / "article.md"


@pytest.fixture(scope="module")
def built():
    """Build every web figure once."""
    return figures.build_all()


def test_build_all_matches_article(built):
    """Every figure the article cites is built, and nothing else."""
    cited = set(re.findall(r"^::: figure (\S+)", ARTICLE.read_text(), flags=re.M))
    assert set(built) == cited


def test_figures_have_data_and_no_title(built):
    """Each figure has traces and leaves its title to the HTML."""
    for fid, fig in built.items():
        assert len(fig.data) > 0, fid
        assert not (fig.layout.title and fig.layout.title.text), fid
