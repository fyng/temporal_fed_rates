"""The Economist chart theme: colours, type, furniture and export helpers.

Single source of truth for every chart in the repo. Plotly is the primary
backend; matplotlib/seaborn is supported through :func:`mpl_style` and
:func:`mpl_furniture`.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from cycler import cycler
from matplotlib.patches import Rectangle

__all__ = [
    "ECON_RED",
    "PALETTE",
    "MAIN",
    "CATEGORICAL",
    "SCALES",
    "SEQUENTIAL",
    "DIVERGING",
    "DIVERGING_COLORSCALE",
    "SIZES",
    "FONT_HEAD",
    "FONT_BODY",
    "TYPE_SCALE",
    "PT_TO_PX",
    "diverging",
    "diverging_scale",
    "signed",
    "register",
    "install_fonts",
    "economist",
    "highlight",
    "period_shading",
    "annotate_event",
    "save",
    "mpl_style",
    "mpl_furniture",
]

# --- Colour: semantic palette (web values) ---------------------------------
PT_TO_PX = 1.7922
ECON_RED = "#E3120B"
BLACK = "#0C0C0C"
TEXT = "#3F5661"
MUTED = "#758D99"
RULE = "#B7C6CF"
BACKGROUND = "#FFFFFF"
PANEL = "#E9EDF0"
GREY_LABEL = "#A4BDC9"

PALETTE = {
    "ECON_RED": ECON_RED,
    "BLACK": BLACK,
    "TEXT": TEXT,
    "MUTED": MUTED,
    "RULE": RULE,
    "BACKGROUND": BACKGROUND,
    "PANEL": PANEL,
    "GREY_LABEL": GREY_LABEL,
}

# --- Colour: main series ----------------------------------------------------
MAIN = {
    "RED": "#DB444B",
    "BLUE": "#006BA2",
    "CYAN": "#3EBCD2",
    "GREEN": "#379A8B",
    "YELLOW": "#EBB434",
    "OLIVE": "#B4BA39",
    "PURPLE": "#9A607F",
    "GOLD": "#D1B07C",
    "GREY": "#758D99",
}

# Default cycle. RED is reserved for the series the chart is about.
CATEGORICAL = [
    MAIN["BLUE"],
    MAIN["CYAN"],
    MAIN["GREEN"],
    MAIN["YELLOW"],
    MAIN["OLIVE"],
    MAIN["PURPLE"],
    MAIN["GOLD"],
    MAIN["GREY"],
]

# --- Colour: equal-lightness scales, darkest to lightest --------------------
SCALES = {
    "RED": ["#A81829", "#C7303C", "#E64E53", "#FF6B6C", "#FF8785", "#FFA39F"],
    "BLUE": ["#00588D", "#1270A8", "#3D89C3", "#5DA4DF", "#7BBFFC", "#98DAFF"],
    "CYAN": ["#005F73", "#00788D", "#0092A7", "#25ADC2", "#4EC8DE", "#6FE4FB"],
    "GREEN": ["#005F52", "#00786B", "#2E9284", "#4DAD9E", "#69C9B9", "#86E5D4"],
    "YELLOW": ["#714C00", "#8D6300", "#AA7C00", "#C89608", "#E7B030", "#FFCB4D"],
    "OLIVE": ["#4C5900", "#667100", "#818A00", "#9DA521", "#BAC03F", "#D7DB5A"],
    "PURPLE": ["#78405F", "#925977", "#AD7291", "#C98CAC", "#E6A6C7", "#FFC2E3"],
    "GOLD": ["#674E1F", "#826636", "#9D7F4E", "#B99966", "#D5B480", "#F2CF9A"],
    "GREY": ["#3F5661", "#576E79", "#6F8793", "#89A2AE", "#A4BDC9", "#BFD8E5"],
}

# Light-to-dark ramps for chronological categories.
SEQUENTIAL = {hue: list(reversed(ramp)) for hue, ramp in SCALES.items()}

# Blue-red diverging scale through near white: the scale for rate changes.
_DIVERGING_MID = "#F0F3F5"
DIVERGING = [
    "#00588D",
    "#1270A8",
    "#5DA4DF",
    "#98DAFF",
    _DIVERGING_MID,
    "#FFA39F",
    "#FF6B6C",
    "#C7303C",
    "#A81829",
]
DIVERGING_COLORSCALE = [[i / (len(DIVERGING) - 1), c] for i, c in enumerate(DIVERGING)]

# --- Sizes: print points and web pixels (x1.7922) ---------------------------
SIZES = {
    "col1": {"pt": (160.0, 116.6), "px": (290, 209)},
    "col2": {"pt": (332.0, 238.8), "px": (595, 428)},
    "col3": {"pt": (504.0, 362.7), "px": (903, 650)},
    "leader": {"pt": (117.0, 83.5), "px": (290, 208)},
    "slide": {"pt": (893.0, 502.2), "px": (1600, 900)},
}
_TYPE_MULT = {"col1": 1.0, "col2": 1.0, "col3": 1.0, "leader": 1.0, "slide": 1600 / 595}

# --- Typography --------------------------------------------------------------
FONT_HEAD = "Barlow"
FONT_BODY = "Barlow Semi Condensed"

TYPE_SCALE = {
    "headline": {"family": FONT_HEAD, "weight": 700, "size": 9.5, "leading": 11.0},
    "subtitle": {"family": FONT_BODY, "weight": 400, "size": 8.0, "leading": 9.5},
    "sub-subtitle": {"family": FONT_BODY, "weight": 300, "size": 7.5, "leading": 9.0},
    "tick": {"family": FONT_BODY, "weight": 400, "size": 7.0, "leading": 7.5},
    "annotation": {"family": FONT_BODY, "weight": 300, "size": 7.0, "leading": 7.0},
    "source": {"family": FONT_BODY, "weight": 300, "size": 6.5, "leading": 7.0},
    "period": {"family": FONT_BODY, "weight": 300, "size": 6.5, "leading": 7.0},
}

_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_FONT_FILES = (
    "Barlow-Bold.ttf",
    "Barlow-Medium.ttf",
    "BarlowSemiCondensed-Bold.ttf",
    "BarlowSemiCondensed-Light.ttf",
    "BarlowSemiCondensed-Medium.ttf",
    "BarlowSemiCondensed-Regular.ttf",
)
_FONT_INSTALL_DIR = Path.home() / ".local" / "share" / "fonts" / "fedrates"
_FC_WARNED = False

_Y_ROOM = 44.0  # px reserved for right-hand y tick labels at type mult 1
_LINE_PX = 1.8  # ~1pt data line


def _font(role: str, mult: float = 1.0, color: str | None = None) -> dict:
    """Build a plotly font dict for a role in the type scale.

    Args:
        role: Key of TYPE_SCALE.
        mult: Multiplier for slide-scale rendering.
        color: Text colour; TEXT by default.

    Returns:
        Dict accepted by any plotly font attribute.
    """
    spec = TYPE_SCALE[role]
    return {
        "family": spec["family"],
        "weight": spec["weight"],
        "size": round(spec["size"] * PT_TO_PX * mult, 1),
        "color": color or TEXT,
    }


def _px(pt: float, mult: float = 1.0) -> float:
    """Convert print points to web pixels.

    Args:
        pt: Length in print points.
        mult: Type-scale multiplier.

    Returns:
        Length in web pixels.
    """
    return pt * PT_TO_PX * mult


def _ensure_mpl_fonts() -> None:
    """Register the bundled TTFs with matplotlib's font manager."""
    for name in _FONT_FILES:
        path = _FONT_DIR / name
        if path.exists():
            try:
                mpl.font_manager.fontManager.addfont(str(path))
            except Exception as exc:  # pragma: no cover - corrupt font only
                warnings.warn(f"Could not register {name}: {exc}", stacklevel=3)


def install_fonts() -> Path:
    """Install the bundled TTFs for Chromium/fontconfig (Kaleido export).

    Kaleido renders through headless Chromium, which reads OS fonts rather than
    matplotlib's cache. Copies the TTFs to ~/.local/share/fonts/fedrates and
    refreshes the font cache. No-op when already installed.

    Returns:
        Path to the installed font directory.
    """
    global _FC_WARNED
    _ensure_mpl_fonts()
    target = _FONT_INSTALL_DIR
    if all((target / name).exists() for name in _FONT_FILES):
        return target
    target.mkdir(parents=True, exist_ok=True)
    for name in _FONT_FILES:
        source = _FONT_DIR / name
        if source.exists():
            shutil.copy2(source, target / name)
    if shutil.which("fc-cache"):
        subprocess.run(["fc-cache", "-f", str(target)], capture_output=True, check=False)
    elif not _FC_WARNED:
        warnings.warn(
            "fc-cache not found; Chromium may not see Barlow until caches refresh", stacklevel=2
        )
        _FC_WARNED = True
    return target


def _template() -> go.layout.Template:
    """Build the plotly template with Economist axis, legend and line defaults.

    Returns:
        Plotly template named for registration.
    """
    t = go.layout.Template()
    t.layout.font = _font("tick")
    t.layout.colorway = CATEGORICAL
    t.layout.paper_bgcolor = BACKGROUND
    t.layout.plot_bgcolor = BACKGROUND
    t.layout.margin = dict(l=_px(5), r=_Y_ROOM, t=_px(43), b=_px(28))
    t.layout.xaxis = dict(
        showgrid=False,
        ticks="outside",
        ticklen=_px(3),
        tickwidth=0.72,
        tickcolor=BLACK,
        showline=True,
        linecolor=BLACK,
        linewidth=0.72,
        tickfont=_font("tick"),
        zeroline=False,
    )
    t.layout.yaxis = dict(
        side="right",
        showgrid=True,
        gridcolor=RULE,
        gridwidth=0.9,
        showline=False,
        ticks="",
        tickfont=_font("tick"),
        zeroline=False,
    )
    t.layout.legend = dict(
        orientation="h",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        groupclick="togglegroup",
        font=_font("tick"),
        tracegroupgap=2,
    )
    t.layout.hoverlabel = dict(font=_font("tick"), bgcolor=BACKGROUND, bordercolor=RULE)
    t.data.scatter = [go.Scatter(line=dict(width=_LINE_PX))]
    t.data.bar = [go.Bar(marker_line_width=0)]
    return t


def register(default: bool = True) -> None:
    """Register the Economist theme for plotly, matplotlib and seaborn.

    Installs fonts, creates the plotly template ``"economist"``, optionally
    makes it the default, and applies matching matplotlib rcParams and seaborn
    palette. Safe to call repeatedly.

    Args:
        default: Set ``"economist"`` as the default plotly template.
    """
    pio.templates["economist"] = _template()
    if default:
        pio.templates.default = "economist"
    try:
        install_fonts()
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"Font installation failed: {exc}", stacklevel=2)
    mpl.rcParams.update(_mpl_rc())
    sns.set_palette(CATEGORICAL)


def _mpl_rc() -> dict:
    """Build matplotlib rcParams mirroring the plotly theme.

    Returns:
        Dict of rcParams; lengths are in print points.
    """
    return {
        "font.family": [FONT_BODY, "DejaVu Sans"],
        "font.size": TYPE_SCALE["tick"]["size"],
        "axes.labelsize": TYPE_SCALE["tick"]["size"],
        "axes.labelcolor": TEXT,
        "xtick.labelsize": TYPE_SCALE["tick"]["size"],
        "ytick.labelsize": TYPE_SCALE["tick"]["size"],
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.direction": "out",
        "xtick.major.size": 3.0,
        "xtick.major.width": 0.4,
        "ytick.major.size": 0.0,
        "ytick.labelleft": False,
        "ytick.labelright": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.edgecolor": BLACK,
        "axes.linewidth": 0.4,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": RULE,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.facecolor": BACKGROUND,
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "axes.prop_cycle": cycler("color", CATEGORICAL),
        "legend.frameon": False,
        "legend.fontsize": TYPE_SCALE["tick"]["size"],
        "axes.unicode_minus": False,
    }


@contextmanager
def mpl_style():
    """Apply the Economist rcParams for the duration of the context.

    Yields:
        None.
    """
    _ensure_mpl_fonts()
    snapshot = dict(mpl.rcParams)
    try:
        mpl.rcParams.update(_mpl_rc())
        yield
    finally:
        mpl.rcParams.update(snapshot)


def mpl_furniture(ax, title, subtitle=None, source=None, footnote=None):
    """Apply Economist furniture to a matplotlib axes.

    Draws the red tag, the left-aligned title block above the plotting area,
    right-hand y tick labels, and the source/footnote band below the baseline.

    Args:
        ax: Axes to decorate.
        title: Headline text.
        subtitle: Secondary line under the headline.
        source: Source line, bottom-left.
        footnote: Footnote text, bottom-right.

    Returns:
        The decorated axes.
    """
    fig = ax.figure
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelright=True, labelleft=False, length=0, width=0)
    ax.tick_params(axis="x", direction="out", length=3, width=0.4, pad=3)
    ax.grid(True, axis="y", color=RULE, linewidth=0.5)
    ax.grid(False, axis="x")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["bottom"].set_linewidth(0.4)
    lo, hi = ax.get_ylim()
    if lo < 0 < hi:
        ax.axhline(0, color=ECON_RED, linewidth=0.4, zorder=1.5)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    kw = dict(
        xycoords="axes fraction", textcoords="offset points", annotation_clip=False, zorder=10
    )
    ax.annotate(
        title,
        xy=(0, 1),
        xytext=(0, 26),
        ha="left",
        va="bottom",
        fontsize=TYPE_SCALE["headline"]["size"],
        fontfamily=[FONT_HEAD, "DejaVu Sans"],
        fontweight=TYPE_SCALE["headline"]["weight"],
        color=TEXT,
        **kw,
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xytext=(0, 15),
            ha="left",
            va="bottom",
            fontsize=TYPE_SCALE["subtitle"]["size"],
            fontfamily=[FONT_BODY, "DejaVu Sans"],
            fontweight=TYPE_SCALE["subtitle"]["weight"],
            color=TEXT,
            **kw,
        )
    pos = ax.get_position()
    fh, fw = fig.get_figheight(), fig.get_figwidth()
    block_top = pos.y1 + (15 + 11 + 17) / 72 / fh
    fig.add_artist(
        Rectangle(
            (pos.x0, block_top),
            15 / 72 / fw,
            5 / 72 / fh,
            transform=fig.transFigure,
            facecolor=ECON_RED,
            edgecolor="none",
            clip_on=False,
            zorder=12,
        )
    )
    down = -(3 + 10 + 2)
    if source:
        ax.annotate(
            source,
            xy=(0, 0),
            xytext=(0, down),
            ha="left",
            va="top",
            fontsize=TYPE_SCALE["source"]["size"],
            fontfamily=[FONT_BODY, "DejaVu Sans"],
            fontweight=TYPE_SCALE["source"]["weight"],
            color=MUTED,
            **kw,
        )
    if footnote:
        ax.annotate(
            footnote,
            xy=(1, 0),
            xytext=(0, down),
            ha="right",
            va="top",
            fontsize=TYPE_SCALE["source"]["size"],
            fontfamily=[FONT_BODY, "DejaVu Sans"],
            fontweight=TYPE_SCALE["source"]["weight"],
            color=MUTED,
            **kw,
        )
    return ax


def _y_extent(fig: go.Figure) -> tuple[float | None, float | None]:
    """Compute the y data extent across traces, including bar baselines.

    Args:
        fig: Figure to inspect.

    Returns:
        (min, max) of finite y values, or (None, None) if there is no data.
    """
    lo, hi = np.inf, -np.inf
    for trace in fig.data:
        candidates = [0.0] if isinstance(trace, go.Bar) else []
        if trace.y is not None:
            arr = np.asarray(trace.y, dtype=float)
            candidates.extend(arr[np.isfinite(arr)].tolist())
        if not candidates:
            continue
        lo = min(lo, min(candidates))
        hi = max(hi, max(candidates))
    if np.isinf(lo):
        return None, None
    return lo, hi


_ROLE_TTF = {
    "headline": "Barlow-Bold.ttf",
    "subtitle": "BarlowSemiCondensed-Regular.ttf",
    "tick": "BarlowSemiCondensed-Regular.ttf",
    "annotation": "BarlowSemiCondensed-Light.ttf",
    "source": "BarlowSemiCondensed-Light.ttf",
    "period": "BarlowSemiCondensed-Light.ttf",
}
_ADVANCES: dict[str, tuple[dict, int, int]] = {}


def _advances(role: str):
    """Load per-character advance widths for a type-scale role.

    Args:
        role: Key of TYPE_SCALE.

    Returns:
        (mapping of character to advance in font units, units per em, fallback
        advance), or None when the font file is unavailable.
    """
    name = _ROLE_TTF.get(role, _ROLE_TTF["tick"])
    if name not in _ADVANCES:
        path = _FONT_DIR / name
        if not path.exists():
            return None
        from fontTools.ttLib import TTFont

        font = TTFont(str(path), lazy=True)
        cmap = font.getBestCmap()
        hmtx = font["hmtx"]
        upem = font["head"].unitsPerEm
        table = {}
        for code, glyph in cmap.items():
            if code < 0x2100:
                table[chr(code)] = hmtx[glyph][0]
        _ADVANCES[name] = (table, upem, hmtx["space"][0] if "space" in hmtx.metrics else upem // 4)
    return _ADVANCES[name]


def _text_width(text: str, role: str, mult: float = 1.0) -> float:
    """Measure rendered text width in pixels from the bundled font metrics.

    Args:
        text: String to measure.
        role: Key of TYPE_SCALE selecting size and font file.
        mult: Type-scale multiplier.

    Returns:
        Width in pixels; a coarse estimate if the font file is missing.
    """
    size = TYPE_SCALE[role]["size"] * PT_TO_PX * mult
    data = _advances(role)
    if data is None:
        return len(text) * 0.5 * size
    table, upem, fallback = data
    units = sum(table.get(ch, fallback) for ch in text)
    return units / upem * size


def _legend_width(fig: go.Figure, mult: float) -> float:
    """Measure the rendered width of the legend in pixels.

    Args:
        fig: Figure whose named traces become legend entries.
        mult: Type-scale multiplier.

    Returns:
        Width in pixels, including swatches and inter-item gaps.
    """
    names = [t.name for t in fig.data if t.name and t.showlegend is not False]
    if not names:
        return 0.0
    # Plotly reserves a swatch plus padding either side of each label.
    entry = 40.0 * mult
    return sum(_text_width(n, "tick", mult) + entry for n in names)


def economist(
    fig: go.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    source: str | None = None,
    footnote: str | None = None,
    size: str = "col2",
    legend: bool = True,
    zeroline: str | bool = "auto",
    yside: str = "right",
) -> go.Figure:
    """Apply the full Economist furniture to a plotly figure.

    Red tag, left-aligned title block above the plot, horizontal top-right
    legend with square swatches, right-hand y tick labels, horizontal-only
    gridlines, 0.4pt baseline, and the source/footnote band.

    Args:
        fig: Figure with data traces already added.
        title: Headline text; no terminal full stop.
        subtitle: Secondary line under the headline.
        source: Source line, bottom-left.
        footnote: Footnote text, bottom-right.
        size: Named preset from SIZES.
        legend: Show the legend.
        zeroline: True, False, or "auto" (draw when the scale crosses zero).
        yside: Side for y tick labels; "right" is the house style.

    Returns:
        The decorated figure.
    """
    if size not in SIZES:
        raise KeyError(f"size must be one of {sorted(SIZES)}, got {size!r}")
    w, h = SIZES[size]["px"]
    m = _TYPE_MULT[size]
    gap = _px(15, m)
    sub_band = _px(11, m)
    head_band = _px(17, m)
    tick_len = _px(3, m)
    label_band = _px(10, m)
    src_band = _px(10, m)
    pad = _px(5, m)
    # A legend too wide to sit beside the headline gets its own band between
    # the title block and the plot.
    stacked = bool(
        legend
        and _text_width(title, "headline", m) + _px(12, m) + _legend_width(fig, m)
        > w - _Y_ROOM * m - _px(5, m)
    )
    legend_band = _px(13, m) if stacked else 0.0
    top = gap + sub_band + head_band + legend_band
    bottom = tick_len + label_band + (src_band if (source or footnote) else 0) + pad
    y_room = _Y_ROOM * m
    if stacked:
        legend_pos = dict(
            x=pad / w,
            y=1 - (head_band + sub_band + legend_band / 2) / h,
            xanchor="left",
        )
    else:
        legend_pos = dict(
            x=1 - pad / w,
            y=1 - ((head_band + sub_band) / 2 if subtitle else head_band / 2) / h,
            xanchor="right",
        )
    fig.update_layout(
        width=w,
        height=h,
        margin=dict(l=pad, r=y_room, t=top, b=bottom),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=_font("tick", m),
        showlegend=legend,
        legend=dict(
            orientation="h",
            xref="container",
            yref="container",
            **legend_pos,
            yanchor="middle",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            groupclick="togglegroup",
            font=_font("tick", m),
            tracegroupgap=2,
        ),
    )
    # Red brand tag, flush to the top-left corner of the chart block.
    pw = w - pad - y_room
    ph = h - top - bottom
    tag_w, tag_h = _px(15, m), _px(5, m)
    x0 = -pad / pw
    y_top_paper = (h - bottom) / ph
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=x0,
        x1=x0 + tag_w / pw,
        y0=y_top_paper - tag_h / ph,
        y1=y_top_paper,
        fillcolor=ECON_RED,
        line_width=0,
        layer="above",
    )
    head_off = gap + legend_band + (sub_band if subtitle else 0)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=1,
        xanchor="left",
        yanchor="bottom",
        yshift=head_off,
        text=title,
        showarrow=False,
        align="left",
        font=_font("headline", m),
    )
    if subtitle:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            xanchor="left",
            yanchor="bottom",
            yshift=gap + legend_band,
            text=subtitle,
            showarrow=False,
            align="left",
            font=_font("subtitle", m),
        )
    fig.update_xaxes(
        showgrid=False,
        ticks="outside",
        ticklen=tick_len,
        tickwidth=0.72 * m,
        tickcolor=BLACK,
        showline=True,
        linecolor=BLACK,
        linewidth=0.72 * m,
        tickfont=_font("tick", m),
    )
    fig.update_yaxes(
        side=yside,
        showgrid=True,
        gridcolor=RULE,
        gridwidth=0.9 * m,
        showline=False,
        ticks="",
        tickfont=_font("tick", m),
    )
    lo, hi = _y_extent(fig)
    zl = zeroline
    if zl == "auto":
        zl = bool(lo is not None and lo < 0 < hi)
    if zl:
        # A scale crossing zero carries an Econ Red zero rule instead of a
        # black baseline; drawing both double-rules the chart.
        fig.update_yaxes(zeroline=True, zerolinecolor=ECON_RED, zerolinewidth=0.72 * m)
        fig.update_xaxes(showline=False, ticks="")
    else:
        fig.update_yaxes(zeroline=False)
        # Anchor at zero only when the data already sits against it, so the
        # baseline doubles as the zero line. A line chart that never
        # approaches zero keeps a tight range and a plain baseline.
        if lo is not None and hi is not None and 0 <= lo <= 0.05 * max(hi, 1e-12):
            fig.update_yaxes(rangemode="tozero")
    y_down = -(tick_len + label_band)
    if source:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            xanchor="left",
            yanchor="top",
            yshift=y_down,
            text=source,
            showarrow=False,
            font=_font("source", m, color=MUTED),
        )
    if footnote:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1,
            y=0,
            xanchor="right",
            yanchor="top",
            yshift=y_down,
            text=footnote,
            showarrow=False,
            font=_font("source", m, color=MUTED),
        )
    if m != 1.0:
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.line is not None:
                trace.line.width = (trace.line.width or _LINE_PX) * m
    if legend:
        _square_swatches(fig, m)
    return fig


def _square_swatches(fig: go.Figure, mult: float) -> None:
    """Replace line legend items with small filled square swatches.

    Plotly draws line samples for line traces; the house style uses squares.
    Adds invisible marker traces that carry the legend entries and links them
    to the data traces via legendgroup so HTML clicks still toggle the series.

    Args:
        fig: Figure whose visible line traces get square swatches.
        mult: Type-scale multiplier for swatch size.
    """
    for i, trace in enumerate(fig.data):
        if trace.showlegend is False:
            continue
        if isinstance(trace, go.Bar):
            continue
        color = trace.line.color if trace.line and trace.line.color else None
        if color is None:
            color = trace.marker.color if trace.marker else None
        if color is None:
            color = CATEGORICAL[i % len(CATEGORICAL)]
        key = trace.legendgroup or trace.name or f"trace{i}"
        trace.legendgroup = key
        trace.showlegend = False
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=trace.name,
                legendgroup=key,
                showlegend=True,
                hoverinfo="skip",
                marker=dict(symbol="square", size=6 * mult, color=color, line=dict(width=0)),
            )
        )


def highlight(fig: go.Figure, series: str, colour: str | None = None) -> go.Figure:
    """Emphasise one series and mute all others to grey.

    Args:
        fig: Figure with labelled traces; call before economist().
        series: Name of the trace to emphasise.
        colour: Focus colour; defaults to MAIN["BLUE"], use MAIN["RED"] to
            signal the series the chart is about.

    Returns:
        The restyled figure, focus trace drawn last.
    """
    focus_color = colour or MAIN["BLUE"]
    focus = None
    for trace in fig.data:
        if trace.name == series:
            focus = trace
            color = focus_color
            width = 3.0
        else:
            color = GREY_LABEL
            width = _LINE_PX
        if isinstance(trace, go.Scatter):
            trace.update(line=dict(color=color, width=width), marker=dict(color=color))
        elif isinstance(trace, go.Bar):
            trace.update(marker=dict(color=color))
    if focus is not None:
        others = [t for t in fig.data if t is not focus]
        fig.data = tuple(others + [focus])
    return fig


def period_shading(fig: go.Figure, spans, labels=None) -> go.Figure:
    """Shade time spans behind the data and mark them with uppercase labels.

    Args:
        fig: Figure to decorate; spans use the x axis of the first subplot.
        spans: Iterable of (x0, x1) axis-coordinate pairs.
        labels: Optional labels, one per span; rendered uppercase at the top
            of each span in Light 6.5pt.

    Returns:
        The decorated figure.
    """
    labels = list(labels) if labels is not None else []
    for i, span in enumerate(spans):
        x0, x1 = span
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            fillcolor=PANEL,
            line_width=0,
            layer="below",
        )
        if i < len(labels) and labels[i]:
            fig.add_annotation(
                xref="x",
                yref="paper",
                x=_span_mid(x0, x1),
                y=1,
                yanchor="top",
                yshift=-3,
                text=str(labels[i]).upper(),
                showarrow=False,
                font=_font("period"),
            )
    return fig


def _interp_y(fig: go.Figure, x) -> float:
    """Interpolate the first data trace's y value at x.

    Args:
        fig: Figure with at least one trace carrying x/y data.
        x: Axis coordinate; a number or an ISO date string.

    Returns:
        Interpolated y value.

    Raises:
        ValueError: If no usable trace is found.
    """
    for trace in fig.data:
        if trace.x is None or trace.y is None:
            continue
        xs = np.asarray(trace.x)
        ys = np.asarray(trace.y, dtype=float)
        if xs.size != ys.size or ys.size < 2:
            continue
        try:
            if xs.dtype.kind == "M":
                xs_f = xs.astype("datetime64[ns]").astype(np.int64).astype(float)
            elif xs.dtype.kind in "US":
                xs_f = (
                    np.asarray(xs.tolist(), dtype="datetime64[ns]")
                    .astype(np.int64)
                    .astype(float)
                )
            else:
                xs_f = xs.astype(float)
            if isinstance(x, (str, np.datetime64)):
                key = np.datetime64(x, "ns").astype(np.int64).astype(float)
            else:
                key = float(x)
        except (TypeError, ValueError):
            continue
        mask = np.isfinite(ys) & np.isfinite(xs_f)
        if mask.sum() < 2:
            continue
        order = np.argsort(xs_f[mask])
        return float(np.interp(key, xs_f[mask][order], ys[mask][order]))
    raise ValueError("annotate_event needs a trace with x/y data to anchor y=None")


def annotate_event(fig: go.Figure, x, text: str, y=None, ax: int = 0, ay: int = -34) -> go.Figure:
    """Annotate a point with a 0.5pt leader and a 50%-opacity arrowhead.

    Args:
        fig: Figure to annotate.
        x: x coordinate of the event.
        text: Label text.
        y: y coordinate of the anchor; interpolated from the first data trace
            when None.
        ax: Horizontal text offset from the anchor in pixels.
        ay: Vertical text offset from the anchor in pixels; negative is above.

    Returns:
        The annotated figure.
    """
    if y is None:
        y = _interp_y(fig, x)
    fig.add_annotation(
        xref="x",
        yref="y",
        x=x,
        y=y,
        text=text,
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=0.9,
        arrowcolor="rgba(12,12,12,0.5)",
        ax=ax,
        ay=ay,
        standoff=3,
        align="center",
        font=_font("annotation"),
    )
    return fig


def diverging(n: int = 9) -> list[str]:
    """Sample the blue-white-red diverging scale at n evenly spaced steps.

    Args:
        n: Number of colours; at least 2.

    Returns:
        n hex colours from dark blue through near white to dark red.

    Raises:
        ValueError: If n is below 2.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    stops = np.linspace(0.0, 1.0, len(DIVERGING))
    rgb = np.array([mpl.colors.to_rgb(c) for c in DIVERGING])
    xs = np.linspace(0.0, 1.0, n)
    out = []
    for x in xs:
        channel = [np.interp(x, stops, rgb[:, k]) for k in range(3)]
        out.append(mpl.colors.to_hex(tuple(channel)).upper())
    return out


def signed(values, positive: str | None = None, negative: str | None = None) -> list[str]:
    """Two-colour split for signed bars.

    House style colours positive and negative bars with two solid colours, not
    a magnitude ramp: depth of colour must not double-encode bar length. Use
    only where the sign carries meaning, such as hikes against cuts.

    Args:
        values: Iterable of signed numbers.
        positive: Colour for values at or above zero; MAIN["BLUE"] by default.
        negative: Colour for values below zero; MAIN["CYAN"] by default.

    Returns:
        One hex colour per value.
    """
    pos = positive or MAIN["BLUE"]
    neg = negative or MAIN["CYAN"]
    return [pos if float(v) >= 0 else neg for v in values]


def diverging_scale(values, n: int = 9) -> list[str]:
    """Map signed values onto the zero-centred diverging scale.

    The range is extended symmetrically around zero, so equal magnitudes get
    equal colour depth on both sides: hikes deepen towards red, cuts towards
    blue.

    Args:
        values: Iterable of signed numbers.
        n: Number of steps in the scale.

    Returns:
        One hex colour per value.
    """
    vals = np.asarray(list(values), dtype=float)
    span = max(float(np.nanmax(np.abs(vals))), 1e-12)
    norm = np.clip((vals + span) / (2 * span), 0.0, 1.0)
    stops = np.linspace(0.0, 1.0, len(DIVERGING))
    rgb = np.array([mpl.colors.to_rgb(c) for c in DIVERGING])
    out = []
    for x in norm:
        channel = [np.interp(x, stops, rgb[:, k]) for k in range(3)]
        out.append(mpl.colors.to_hex(tuple(channel)).upper())
    return out


def save(fig: go.Figure, path, size: str = "col2", scale: int = 2) -> Path:
    """Save a figure as PNG via Kaleido, or as self-contained HTML.

    Args:
        fig: Figure to save.
        path: Output path; ``.html`` writes self-contained interactive HTML,
            anything else writes PNG at ``scale`` times the preset size.
        size: Named preset from SIZES applied to the layout.
        scale: PNG magnification.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = SIZES[size]["px"]
    fig.update_layout(width=w, height=h)
    if path.suffix.lower() == ".html":
        fig.write_html(path, include_plotlyjs=True, full_html=True)
    else:
        fig.write_image(path, scale=scale)
    return path


def _span_mid(x0, x1):
    """Midpoint of a span, supporting dates and numbers.

    Args:
        x0: Span start.
        x1: Span end.

    Returns:
        Midpoint in the same representation as the inputs.
    """
    try:
        return (x0 + x1) / 2
    except TypeError:
        a, b = np.datetime64(x0, "D"), np.datetime64(x1, "D")
        return str(a + (b - a) // 2)
