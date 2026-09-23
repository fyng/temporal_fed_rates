"""Build a self-contained HTML page from docs/article.md and fedrates.web.figures.

Run ``uv run python -m fedrates.web.build`` to write ``site/fed_funds.html``.
The page inlines plotly.js, the stylesheet and all fonts, so it opens offline
with no network requests.
"""

from __future__ import annotations

import base64
import json
import re
from html import escape
from pathlib import Path
from typing import Any

import latex2mathml.converter
import markdown as md_pkg
import plotly.utils
from plotly.offline import get_plotlyjs

REPO = Path(__file__).resolve().parents[3]
ARTICLE = REPO / "docs" / "article.md"
FONTS_DIR = REPO / "assets" / "fonts"
OUTPUT = REPO / "site" / "fed_funds.html"

SECTION_KICKER = 'Finance &amp; economics <span class="kicker-sep">|</span> The Fed'

_FIGURE_RE = re.compile(r"^:::\s*figure\s+(\S+)[ \t]*\n(.*?)^:::[ \t]*$", re.MULTILINE | re.DOTALL)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\w)\$(?!\s)((?:\\.|[^$\\])+?)(?<!\s)\$(?!\w)", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)

_FONT_SPECS = (
    ("Barlow", "700", "normal", "Barlow-Bold.ttf"),
    ("Barlow", "600", "normal", "Barlow-SemiBold.ttf"),
    ("Barlow", "500", "normal", "Barlow-Medium.ttf"),
    ("Barlow Semi Condensed", "300", "normal", "BarlowSemiCondensed-Light.ttf"),
    ("Barlow Semi Condensed", "400", "normal", "BarlowSemiCondensed-Regular.ttf"),
    ("Barlow Semi Condensed", "700", "normal", "BarlowSemiCondensed-Bold.ttf"),
    ("Source Serif 4", "400", "normal", "SourceSerif4-Regular.ttf"),
    ("Source Serif 4", "400", "italic", "SourceSerif4-It.ttf"),
    ("Source Serif 4", "600", "normal", "SourceSerif4-Semibold.ttf"),
    ("Latin Modern Math", "400", "normal", "LatinModernMath-Regular.otf"),
)

_LOADER_JS = """
(function () {
  function render(div) {
    if (div.dataset.rendered) return;
    div.dataset.rendered = "1";
    var cfg = JSON.parse(document.getElementById("data-" + div.id).textContent);
    var layout = cfg.layout || {};
    var minWidth = layout.meta && layout.meta.minWidth;
    if (minWidth && div.clientWidth < minWidth) {
      var scroller = document.createElement("div");
      scroller.className = "chart-scroll";
      div.parentNode.insertBefore(scroller, div);
      scroller.appendChild(div);
      div.style.width = minWidth + "px";
    }
    var width = div.clientWidth;
    if (layout.width && width && width < layout.width) {
      var ratio = width / layout.width;
      layout.width = Math.round(width);
      if (layout.height) layout.height = Math.round(layout.height * ratio);
    }
    Plotly.newPlot(div, cfg.data, layout, {displayModeBar: false, responsive: true});
  }
  var charts = Array.prototype.slice.call(document.querySelectorAll(".chart-body"));
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            io.unobserve(entry.target);
            render(entry.target);
          }
        });
      },
      {rootMargin: "250px 0px"}
    );
    charts.forEach(function (div) { io.observe(div); });
  } else {
    charts.forEach(render);
  }
})();
"""

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
@@FONTS_CSS@@
@@CSS@@
</style>
</head>
<body>
<main class="page">
<div class="econ-tag" aria-hidden="true"></div>
<p class="kicker">@@KICKER@@</p>
<h1 class="headline">@@HEADLINE@@</h1>
@@STANDFIRST@@
@@TOC@@
<div class="article">
@@BODY@@
</div>
</main>
<script>@@PLOTLYJS@@</script>
<script>@@LOADER@@</script>
</body>
</html>
"""

_CSS = """
:root {
  --red: #E3120B; --black: #0C0C0C; --grey: #3F5661; --muted: #758D99;
  --rule: #B7C6CF; --panel: #E9EDF0; --link: #006BA2;
  --sans: "Barlow", "Helvetica Neue", Arial, sans-serif;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: #fff; color: var(--black);
       font-family: var(--serif); font-size: 18.5px; line-height: 1.55; }
.page { max-width: 680px; margin: 0 auto; padding: 26px 20px 80px; }
.econ-tag { width: 45px; height: 7px; background: var(--red); }
.kicker { font-family: var(--sans); font-weight: 600; font-size: 14px;
          letter-spacing: .05em; text-transform: uppercase; color: var(--red);
          margin: 14px 0 0; }
.kicker-sep { color: var(--muted); font-weight: 400; }
.headline { font-family: var(--sans); font-weight: 700; font-size: 40px;
            line-height: 1.1; letter-spacing: -.01em; margin: 8px 0 16px; }
.standfirst { font-family: var(--sans); font-weight: 500; font-size: 22px;
              line-height: 1.4; color: var(--grey); margin: 0 0 20px; }
.toc { font-family: var(--sans); font-weight: 500; font-size: 14.5px; line-height: 1.5;
       border-top: 1px solid var(--black); border-bottom: 1px solid var(--rule);
       padding: 10px 0 12px; margin: 0 0 30px; }
.toc-label { font-weight: 700; font-size: 12px; letter-spacing: .05em;
             text-transform: uppercase; color: var(--muted); margin: 0 0 4px; }
.toc ol { margin: 0; padding-left: 18px; }
.toc li { margin: 2px 0; }
.toc a { color: var(--link); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.article p { margin: 0 0 20px; }
.article h2 { font-family: var(--sans); font-weight: 700; font-size: 26px;
              line-height: 1.18; letter-spacing: -.005em; margin: 40px 0 14px; }
.article h3 { font-family: var(--sans); font-weight: 600; font-size: 19.5px;
              line-height: 1.25; margin: 30px 0 10px; }
.article a { color: var(--link); text-decoration: underline;
             text-underline-offset: 2px; }
.article ul, .article ol { margin: 0 0 20px; padding-left: 22px; }
.article li { margin: 0 0 6px; }
.article blockquote { background: var(--panel); font-family: var(--sans);
                      font-weight: 500; font-size: 16.5px; line-height: 1.45;
                      color: var(--grey); margin: 26px 0; padding: 16px 18px; }
.article blockquote p { margin: 0 0 10px; }
.article blockquote p:last-child { margin-bottom: 0; }
.article hr { border: 0; border-top: 1px solid var(--rule); margin: 30px 0; }
figure.chart { margin: 34px 0; border-top: 1px solid var(--rule); padding: 12px 0 0; }
.chart-title { font-family: var(--sans); font-weight: 700; font-size: 17px;
               line-height: 1.25; color: var(--black); margin: 0 0 2px; }
.chart-subtitle { font-family: var(--sans); font-weight: 500; font-size: 14px;
                  line-height: 1.35; color: var(--grey); margin: 0 0 10px; }
.chart-body { width: 100%; min-height: 60px; }
.chart-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.chart-source { font-family: var(--sans); font-weight: 500; font-size: 12.5px;
                color: var(--muted); margin: 4px 0 0; }
.table-wrap { overflow-x: auto; }
table { font-family: var(--sans); font-weight: 500; font-size: 14px; line-height: 1.4;
        border-collapse: collapse; width: 100%; margin: 24px 0;
        border-top: 1px solid var(--black); }
th, td { padding: 7px 12px 7px 0; text-align: left; vertical-align: top; }
thead th { font-weight: 700; border-bottom: 1px solid var(--black); }
tbody td { border-bottom: 1px solid var(--rule); }
tbody tr:last-child td { border-bottom: 1px solid var(--black); }
math { font-family: "Latin Modern Math", "STIX Two Math", "Cambria Math", math; font-size: 110%; }
math[display="block"] { display: block; margin: 4px 0 20px;
                        overflow-x: auto; overflow-y: hidden; }
@media (max-width: 560px) {
  body { font-size: 17.5px; }
  .headline { font-size: 30px; }
  .standfirst { font-size: 19px; }
  .article h2 { font-size: 22px; }
}
"""


def _fonts_css(fonts_dir: Path) -> str:
    """Build base64 @font-face rules for every bundled font found in fonts_dir.

    Args:
        fonts_dir: Directory holding the font files named in _FONT_SPECS.

    Returns:
        CSS text; empty when no font files exist, so the stacks fall back.
    """
    rules = []
    for family, weight, style, name in _FONT_SPECS:
        path = fonts_dir / name
        if not path.is_file():
            continue
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        mime, fmt = ("otf", "opentype") if path.suffix == ".otf" else ("ttf", "truetype")
        rules.append(
            f'@font-face{{font-family:"{family}";font-weight:{weight};font-style:{style};'
            f'src:url(data:font/{mime};base64,{payload}) format("{fmt}");'
            "font-display:swap;}"
        )
    return "\n".join(rules)


def _extract_standfirst(text: str) -> tuple[str, str | None]:
    """Pull the ``> standfirst:`` blockquote out of the article.

    Args:
        text: Raw article markdown.

    Returns:
        Article text without the standfirst block, and the standfirst text.
    """
    lines = text.split("\n")
    kept: list[str] = []
    standfirst_parts: list[str] | None = None
    i = 0
    while i < len(lines):
        marker = re.match(r"^\s*>\s*standfirst:\s*(.*)$", lines[i])
        if marker is None:
            kept.append(lines[i])
            i += 1
            continue
        parts = [marker.group(1)]
        i += 1
        while i < len(lines) and re.match(r"^\s*>", lines[i]):
            parts.append(re.sub(r"^\s*>\s?", "", lines[i]))
            i += 1
        standfirst_parts = [part.strip() for part in parts if part.strip()]
    if standfirst_parts is None:
        return text, None
    return "\n".join(kept), " ".join(standfirst_parts)


def _extract_figures(text: str) -> tuple[str, list[tuple[str, dict[str, str]]]]:
    """Replace ::: figure blocks with placeholders and collect their metadata.

    Args:
        text: Article markdown, with standfirst already removed.

    Returns:
        Article text with placeholders, and (id, metadata) pairs in order.
    """
    figures: list[tuple[str, dict[str, str]]] = []

    def replace(match: re.Match[str]) -> str:
        fid = match.group(1)
        meta: dict[str, str] = {}
        for line in match.group(2).strip().splitlines():
            key, _, value = line.partition(":")
            if key.strip():
                meta[key.strip().lower()] = value.strip()
        figures.append((fid, meta))
        return f"\n@@FIG{len(figures) - 1}@@\n"

    return _FIGURE_RE.sub(replace, text), figures


def _extract_math(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace display and inline math with placeholders.

    Args:
        text: Article markdown.

    Returns:
        Article text with placeholders, and (latex, display) pairs in order.
    """
    maths: list[tuple[str, str]] = []

    def stash(display: str) -> Any:
        def replace(match: re.Match[str]) -> str:
            maths.append((match.group(1).strip(), display))
            return f"@@M{len(maths) - 1}@@"

        return replace

    text = _DISPLAY_MATH_RE.sub(stash("block"), text)
    text = _INLINE_MATH_RE.sub(stash("inline"), text)
    return text, maths


def _mathml(latex: str, display: str) -> str:
    """Convert one LaTeX expression to MathML.

    Args:
        latex: LaTeX source.
        display: "inline" or "block".

    Returns:
        MathML markup.

    Raises:
        ValueError: If the expression cannot be parsed.
    """
    try:
        return latex2mathml.converter.convert(latex, display=display)
    except Exception as exc:
        raise ValueError(f"Failed to render LaTeX: {latex!r}") from exc


def _figure_html(index: int, fid: str, meta: dict[str, str], figure: Any) -> str:
    """Render one figure block as HTML with its plotly JSON embedded.

    Args:
        index: Placeholder index, used for the div id.
        fid: Figure id from the article.
        meta: Title, subtitle, source and optional footnote from the figure block.
        figure: Plotly Figure (or anything with ``to_plotly_json``).

    Returns:
        HTML for the whole <figure> element.
    """
    div_id = "chart-" + re.sub(r"[^A-Za-z0-9_-]", "-", fid)
    spec = figure.to_plotly_json() if hasattr(figure, "to_plotly_json") else figure
    payload = json.dumps(spec, cls=plotly.utils.PlotlyJSONEncoder).replace("</", "<\\/")
    parts = [f'<figure class="chart" id="fig-{div_id}">']
    if meta.get("title"):
        parts.append(f'<figcaption class="chart-title">{escape(meta["title"])}</figcaption>')
    if meta.get("subtitle"):
        parts.append(f'<p class="chart-subtitle">{escape(meta["subtitle"])}</p>')
    parts.append(f'<div class="chart-body" id="{div_id}"></div>')
    if meta.get("source"):
        parts.append(f'<p class="chart-source">{escape(meta["source"])}</p>')
    if meta.get("footnote"):
        parts.append(f'<p class="chart-source">{escape(meta["footnote"])}</p>')
    parts.append(f'<script type="application/json" id="data-{div_id}">{payload}</script>')
    parts.append("</figure>")
    return "\n".join(parts)


def _toc_html(toc_tokens: list[dict[str, Any]]) -> str:
    """Build the contents block from the article's H2 headings.

    Args:
        toc_tokens: Heading tree produced by the markdown toc extension.

    Returns:
        HTML for the nav, or "" when there are no H2s.
    """

    def h2s(tokens: list[dict[str, Any]]):
        for token in tokens:
            if token["level"] == 2:
                yield token
            yield from h2s(token.get("children", []))

    entries = list(h2s(toc_tokens))
    if not entries:
        return ""
    items = "\n".join(
        f'<li><a href="#{token["id"]}">{token["name"]}</a></li>' for token in entries
    )
    return f'<nav class="toc"><p class="toc-label">Contents</p>\n<ol>\n{items}\n</ol></nav>'


def _substitute(html: str, maths: list[tuple[str, str]], fig_htmls: list[str]) -> str:
    """Swap placeholders in the rendered HTML for MathML and figure markup.

    Args:
        html: Rendered article HTML still holding placeholders.
        maths: (latex, display) pairs from _extract_math.
        fig_htmls: Rendered figure HTML blocks, in article order.

    Returns:
        HTML with every placeholder replaced.

    Raises:
        ValueError: If a placeholder survives, which would mean a token leaked.
    """
    for i, (latex, display) in enumerate(maths):
        mathml = _mathml(latex, display)
        token = f"@@M{i}@@"
        if display == "block":
            html = html.replace(f"<p>{token}</p>", mathml)
        html = html.replace(token, mathml)
    for i, figure_html in enumerate(fig_htmls):
        token = f"@@FIG{i}@@"
        html = html.replace(f"<p>{token}</p>", figure_html)
        html = html.replace(token, figure_html)
    if "@@" in html:
        raise ValueError("Unresolved placeholder in rendered HTML")
    return html


def render_page(article: str, figures: dict[str, Any], fonts_dir: Path = FONTS_DIR) -> str:
    """Render the full self-contained HTML page.

    Args:
        article: Raw markdown of docs/article.md.
        figures: Map of figure id to plotly Figure (or plotly-json dict).
        fonts_dir: Directory with the TTFs to inline.

    Returns:
        Complete HTML document as a string.

    Raises:
        ValueError: If the article has no H1, references an unknown figure id,
            or holds LaTeX that cannot be converted.
    """
    body, standfirst_text = _extract_standfirst(article)

    h1 = _H1_RE.search(body)
    if h1 is None:
        raise ValueError("Article has no H1 headline")
    headline = h1.group(1)
    body = body[: h1.start()] + body[h1.end() :]

    body, fig_metas = _extract_figures(body)
    for fid, _meta in fig_metas:
        if fid not in figures:
            known = ", ".join(sorted(figures)) or "(none)"
            raise ValueError(
                f"Article references figure {fid!r} but build_all() provided: {known}"
            )

    body, maths = _extract_math(body)

    converter = md_pkg.Markdown(extensions=["tables", "fenced_code", "attr_list", "toc"])
    rendered = converter.convert(body)
    rendered = _substitute(rendered, maths, fig_htmls=[
        _figure_html(i, fid, meta, figures[fid]) for i, (fid, meta) in enumerate(fig_metas)
    ])
    rendered = rendered.replace("<table>", '<div class="table-wrap"><table>').replace(
        "</table>", "</table></div>"
    )

    standfirst_html = (
        f'<p class="standfirst">{escape(standfirst_text)}</p>' if standfirst_text else ""
    )

    page = _PAGE_TEMPLATE
    page = page.replace("@@FONTS_CSS@@", _fonts_css(fonts_dir))
    page = page.replace("@@CSS@@", _CSS)
    page = page.replace("@@TITLE@@", escape(headline))
    page = page.replace("@@KICKER@@", SECTION_KICKER)
    page = page.replace("@@HEADLINE@@", escape(headline))
    page = page.replace("@@STANDFIRST@@", standfirst_html)
    page = page.replace("@@TOC@@", _toc_html(converter.toc_tokens))
    page = page.replace("@@BODY@@", rendered)
    page = page.replace("@@PLOTLYJS@@", get_plotlyjs())
    page = page.replace("@@LOADER@@", _LOADER_JS)
    return page


def main() -> Path:
    """Build site/fed_funds.html from docs/article.md and fedrates.web.figures.

    Returns:
        Path to the written HTML file.
    """
    from fedrates.web.figures import build_all

    html = render_page(ARTICLE.read_text(encoding="utf-8"), build_all())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(main())
