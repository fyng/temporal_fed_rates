"""Tests for the self-contained web page builder."""

from html.parser import HTMLParser
from pathlib import Path

import plotly.graph_objects as go
import pytest

from fedrates.web.build import render_page

STUB_ARTICLE = """# A test headline

> standfirst: A standfirst for the stub article.

## First section

Body text with inline math $\\alpha + \\beta$ and an
[external link](https://example.com).

$$
y = mx + b
$$

::: figure test-fig
title: A test chart
subtitle: A test subtitle
source: Source: Test
:::

## Second section

| Col A | Col B |
|---|---|
| 1 | 2 |
"""


def _stub_figure() -> go.Figure:
    return go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))


class _RemoteTagScanner(HTMLParser):
    """Collect src/href attributes on script, link, img and iframe tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.remote: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"script", "link", "img", "iframe"}:
            return
        for name, value in attrs:
            if name in {"src", "href"} and value and value.startswith(("http://", "https://")):
                self.remote.append((tag, name, value))


def test_no_remote_resources() -> None:
    html = render_page(STUB_ARTICLE, {"test-fig": _stub_figure()}, fonts_dir=Path("/nonexistent"))
    scanner = _RemoteTagScanner()
    scanner.feed(html)
    assert scanner.remote == []


def test_figures_rendered_lazily() -> None:
    html = render_page(STUB_ARTICLE, {"test-fig": _stub_figure()}, fonts_dir=Path("/nonexistent"))
    assert 'id="chart-test-fig"' in html
    assert 'id="data-chart-test-fig"' in html
    assert "Plotly.newPlot" in html
    assert "IntersectionObserver" in html
    assert '<figcaption class="chart-title">A test chart</figcaption>' in html
    assert "Source: Test" in html


def test_mathml_present() -> None:
    html = render_page(STUB_ARTICLE, {"test-fig": _stub_figure()}, fonts_dir=Path("/nonexistent"))
    assert "<math" in html
    assert 'display="block"' in html
    assert "@@M" not in html and "@@FIG" not in html


def test_standfirst_and_headline() -> None:
    html = render_page(STUB_ARTICLE, {"test-fig": _stub_figure()}, fonts_dir=Path("/nonexistent"))
    assert '<h1 class="headline">A test headline</h1>' in html
    assert '<p class="standfirst">A standfirst for the stub article.</p>' in html


def test_missing_figure_id_raises() -> None:
    ghost_article = STUB_ARTICLE.replace("::: figure test-fig", "::: figure ghost")
    with pytest.raises(ValueError, match="ghost"):
        render_page(ghost_article, {"test-fig": _stub_figure()}, fonts_dir=Path("/nonexistent"))


def test_no_h1_raises() -> None:
    with pytest.raises(ValueError, match="H1"):
        render_page("no headline here", {}, fonts_dir=Path("/nonexistent"))
