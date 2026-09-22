"""Production chart: long-run level-model coefficients across the specification grid.

Run:
    uv run python -m fedrates.figures.specification_grid
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fedrates import dataset, register, save
from fedrates.models.level import specification_grid
from fedrates.theme import (
    ECON_RED,
    FONT_BODY,
    MAIN,
    PT_TO_PX,
    SIZES,
    TEXT,
    economist,
)

OUT = Path(__file__).resolve().parents[3] / "figures" / "production"
SIZE = "col3"

TITLE = "Signal failure"
SUBTITLE = (
    "Long-run Fed response to inflation and slack, by sample and frequency<br>"
    "Estimate and 95% confidence interval"
)
SOURCE = "Sources: Federal Reserve; FRED; our analysis"
FOOTNOTE = "Partial-adjustment Taylor rule; Newey-West standard errors"

# Panel scales. The 1994+ monthly inflation interval (upper bound 11.2) runs to
# the panel edge and is flagged with an arrowhead; the true bound sits beside it.
_XR = {"a": [-3.6, 6.0], "b": [-1.0, 3.4]}
_DTICK = {"a": 2, "b": 1}
_CLIP_ROW = "1994+ M"
_CLIP_ARROW = "rgba(12,12,12,0.5)"  # house annotation arrow, 50% black

# Rows bottom to top on the shared category axis.
_ROWS = ("1994+ Q", "1994+ M", "1983+ Q", "1983+ M", "1961+ Q", "1961+ M")
_FREQ = {"M": ("Monthly", MAIN["GREY"]), "Q": ("Quarterly", MAIN["BLUE"])}
_FREQ_NAME = {"M": "monthly", "Q": "quarterly"}
_KEY_ROW_X = 1.9  # direct colour labels sit right of the 1961+ intervals
_LEFT_PAD = 52.0  # px left margin, keeping the sample labels off the frame
_BASE_TOP = 43.0 * PT_TO_PX  # theme head band, px, before the second subtitle line
_SUB_LINE = 17.0  # px leading of one 8pt subtitle line
_TOP_BAND = _BASE_TOP + _SUB_LINE  # head band once the second subtitle line is added
_BOTTOM_BAND = 28.0 * PT_TO_PX  # theme tick-label and source band, px


def _grid() -> pd.DataFrame:
    """Fit the six {sample} x {frequency} level-model cells.

    Returns:
        One row per cell, as produced by ``level.specification_grid``.
    """
    frame_m = dataset.load_frame("analysis_monthly")
    frame_q = dataset.build_frame("q")
    return specification_grid(frame_m, frame_q)


def _cells(grid: pd.DataFrame, value: str, se: str) -> tuple[list[float], list[float]]:
    """Estimates and 95% half-widths for one coefficient, ordered on _ROWS.

    Args:
        grid: Specification grid from :func:`_grid`.
        value: Column carrying the point estimate.
        se: Column carrying the HAC standard error.

    Returns:
        (estimates, half-widths) with half-widths of 1.96 standard errors.
    """
    wide = grid.set_index(["freq", "start"])
    est, half = [], []
    for row in _ROWS:
        key = (_FREQ_NAME[row[-1]], row[:-2])
        est.append(float(wide.loc[key, value]))
        half.append(1.96 * float(wide.loc[key, se]))
    return est, half


def _trace(freq: str, est: list[float], up: list[float], lo: list[float]) -> go.Scatter:
    """Dot-and-interval trace for one frequency's three cells.

    Args:
        freq: "M" or "Q", selecting rows, colour and name.
        est: Point estimates, one per row of _ROWS.
        up: Distances from the estimate to the interval tops.
        lo: Distances from the estimate to the interval bottoms.

    Returns:
        Scatter trace with horizontal error bars on this frequency's rows.
    """
    name, color = _FREQ[freq]
    idx = [i for i, row in enumerate(_ROWS) if row.endswith(freq)]
    return go.Scatter(
        y=idx,
        x=[est[i] for i in idx],
        mode="markers",
        name=name,
        marker=dict(color=color, size=7),
        error_x=dict(
            type="data",
            symmetric=False,
            array=[up[i] for i in idx],
            arrayminus=[lo[i] for i in idx],
            color=color,
            thickness=1.4,
        ),
    )


def _panel_traces(grid: pd.DataFrame, value: str, se: str, *, clip: bool) -> list[go.Scatter]:
    """Both frequency traces for one panel, clipping where flagged.

    Args:
        grid: Specification grid.
        value: Estimate column.
        se: Standard-error column.
        clip: Cut the 1994+ monthly interval at :data:`_CLIP_END`.

    Returns:
        Monthly then quarterly traces.

    Raises:
        ValueError: If the clip bound sits below the point estimate.
    """
    traces = []
    for freq in ("M", "Q"):
        est, half = _cells(grid, value, se)
        up = list(half)
        if clip and freq == "M":
            i = _ROWS.index(_CLIP_ROW)
            up[i] = min(half[i], _XR["a"][1] - est[i])
            if up[i] <= 0:
                raise ValueError("panel edge sits below the point estimate")
        traces.append(_trace(freq, est, up, half))
    return traces


def _headroom(fig: go.Figure) -> None:
    """Add a band for the second subtitle line and keep the tag flush.

    Args:
        fig: Decorated figure; the red tag is ``shapes[0]``, the headline the
            first matching annotation.
    """
    w, h = SIZES[SIZE]["px"]
    fig.update_layout(margin=dict(t=_TOP_BAND, l=_LEFT_PAD))
    for ann in fig.layout.annotations:
        if ann.text == TITLE:
            ann.yshift += _SUB_LINE
            break
    pw, ph = w - _LEFT_PAD - 44.0, h - _TOP_BAND - _BOTTOM_BAND
    tag = fig.layout.shapes[0]
    tag.x0, tag.x1 = -_LEFT_PAD / pw, (15.0 - _LEFT_PAD) / pw
    tag.y1 = 1 + _TOP_BAND / ph
    tag.y0 = tag.y1 - 5.0 / ph


def _zero_rules(fig: go.Figure) -> None:
    """Draw the red zero rule down each panel, under the data.

    Args:
        fig: Figure with both panel x axes in place.
    """
    for xref in ("x", "x2"):
        fig.add_shape(
            type="line",
            xref=xref,
            yref="paper",
            x0=0,
            x1=0,
            y0=0,
            y1=1,
            line=dict(color=ECON_RED, width=0.72),
            layer="below",
        )


def _axes(fig: go.Figure) -> None:
    """Set panel scales and strip the category axis furniture.

    Args:
        fig: Figure after the furniture pass.
    """
    for col, coef in ((1, "a"), (2, "b")):
        fig.update_xaxes(
            showgrid=False,
            ticks="",
            showline=False,
            range=_XR[coef],
            dtick=_DTICK[coef],
            row=1,
            col=col,
        )
    fig.update_yaxes(showgrid=False, showticklabels=False, range=[-0.6, len(_ROWS) - 0.4])


def _labels(fig: go.Figure, grid: pd.DataFrame) -> None:
    """Add panel titles and the direct row and colour labels.

    Args:
        fig: Figure after the furniture pass.
        grid: Specification grid, supplying the true clipped bound.
    """
    kw = dict(showarrow=False, font=dict(family=FONT_BODY, size=12.5, color=TEXT))
    for xref, text in (
        ("x domain", "Inflation"),
        ("x2 domain", "Slack"),
    ):
        fig.add_annotation(
            x=0,
            y=1,
            xref=xref,
            yref="y domain",
            xanchor="left",
            yanchor="bottom",
            yshift=9,
            text=text,
            showarrow=False,
            font=dict(family=FONT_BODY, size=13, color=TEXT),
        )
    for y, text in ((0.5, "1994+"), (2.5, "1983+"), (4.5, "1961+")):
        fig.add_annotation(
            x=-0.035,
            y=y,
            xref="x domain",
            yref="y",
            xanchor="right",
            text=text,
            **kw,
        )
    for row, label in ((len(_ROWS) - 1, "Monthly"), (len(_ROWS) - 2, "Quarterly")):
        fig.add_annotation(
            x=_KEY_ROW_X,
            y=row,
            xref="x",
            yref="y",
            xanchor="left",
            text=label,
            **kw,
        )
    est, half = _cells(grid, "a", "a_se")
    i = _ROWS.index(_CLIP_ROW)
    fig.add_annotation(
        xref="x",
        yref="y",
        x=_XR["a"][1],
        y=i,
        ax=-13,
        ay=0,
        text="",
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=0.9,
        arrowcolor=_CLIP_ARROW,
    )
    fig.add_annotation(
        xref="x",
        yref="y",
        x=_XR["a"][1] + 0.33,
        y=i,
        text=f"{est[i] + half[i]:.1f}",
        showarrow=False,
        xanchor="left",
        font=dict(family=FONT_BODY, size=12.5, color=TEXT),
    )


def build_fig() -> go.Figure:
    """Build the specification-grid coefficient chart.

    Returns:
        Decorated Economist figure.
    """
    register()
    grid = _grid()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.10)
    for col, (value, se) in enumerate((("a", "a_se"), ("b", "b_se")), start=1):
        for trace in _panel_traces(grid, value, se, clip=col == 1):
            fig.add_trace(trace, row=1, col=col)
    economist(
        fig,
        title=TITLE,
        subtitle=SUBTITLE,
        source=SOURCE,
        footnote=FOOTNOTE,
        size=SIZE,
        legend=False,
        zeroline=False,
    )
    _headroom(fig)
    _zero_rules(fig)
    _axes(fig)
    _labels(fig, grid)
    return fig


def main() -> None:
    """Render the chart into figures/production."""
    fig = build_fig()
    for name in ("specification_grid.png", "specification_grid.svg"):
        path = save(fig, OUT / name, size=SIZE, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
