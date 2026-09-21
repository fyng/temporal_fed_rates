---
name: economist-charts
description: Build charts in The Economist's house visual style using fedrates.theme. Use before writing any plotting code, choosing colours or figure sizes, or laying out panels, legends and annotations.
---

# Economist charts with fedrates.theme

Source of truth: `src/fedrates/theme.py`. Reference PDF and extract:
`docs/reference/economist_chart_styleguide_2017.pdf` (text extract in the same
directory). For chart titles and subtitles, use the `economist-writing` skill —
this skill covers only visual matters.

Call `register()` once per process, then build every figure through
`fedrates.theme`. Never call plotly or matplotlib without the theme.

## Colours (copy-pasteable hex)

Semantic palette:

```python
ECON_RED  = "#E3120B"   # brand tag only, never a data series
BLACK     = "#0C0C0C"   # x baseline, tick marks
TEXT      = "#3F5661"   # titles, subtitles, axis labels, annotations
MUTED     = "#758D99"   # source line, footnotes, secondary labels
RULE      = "#B7C6CF"   # horizontal gridlines
PANEL     = "#E9EDF0"   # highlight panels, number boxes, period shading
BACKGROUND = "#FFFFFF"
GREY_LABEL = "#A4BDC9"  # muted series in a highlight chart
```

Main series (`theme.MAIN`):

```
RED    #DB444B   BLUE   #006BA2   CYAN  #3EBCD2   GREEN #379A8B
YELLOW #EBB434   OLIVE  #B4BA39   PURPLE #9A607F  GOLD  #D1B07C
GREY   #758D99
```

Default categorical order (`theme.CATEGORICAL`; RED is excluded and reserved):

```
#006BA2  #3EBCD2  #379A8B  #EBB434  #B4BA39  #9A607F  #D1B07C  #758D99
```

Equal-lightness scales, darkest to lightest (`theme.SCALES`):

```
RED    A81829 C7303C E64E53 FF6B6C FF8785 FFA39F
BLUE   00588D 1270A8 3D89C3 5DA4DF 7BBFFC 98DAFF
CYAN   005F73 00788D 0092A7 25ADC2 4EC8DE 6FE4FB
GREEN  005F52 00786B 2E9284 4DAD9E 69C9B9 86E5D4
YELLOW 714C00 8D6300 AA7C00 C89608 E7B030 FFCB4D
OLIVE  4C5900 667100 818A00 9DA521 BAC03F D7DB5A
PURPLE 78405F 925977 AD7291 C98CAC E6A6C7 FFC2E3
GOLD   674E1F 826636 9D7F4E B99966 D5B480 F2CF9A
GREY   3F5661 576E79 6F8793 89A2AE A4BDC9 BFD8E5
```

Derived scales:

- `theme.SEQUENTIAL` — the same ramps light-to-dark, for chronological
  categories (years, age bands). One hue per chart, never the categorical mix.
- `theme.DIVERGING` and `theme.DIVERGING_COLORSCALE` — blue through near
  white to red; the scale for rate changes (cuts vs hikes).
- `theme.diverging(n)` — n evenly spaced samples of the diverging scale.
- `theme.diverging_scale(values, n=9)` — one colour per signed value on a
  zero-centred symmetric range. For heatmaps and choropleths only.
- `theme.signed(values, positive=BLUE, negative=CYAN)` — two solid colours for
  signed bars. Use this, not a ramp: colour depth must not double-encode bar
  length.

## Sizes and type

`theme.SIZES` presets — pass the same `size` to `economist()` and `save()`:

| preset  | print pt   | web px     | use                         |
|---------|------------|------------|-----------------------------|
| col1    | 160        | 290        | narrow column               |
| col2    | 332        | 595        | default                     |
| col3    | 504        | 903        | wide / double column        |
| leader  | 117 x 83.5 | 290 x 208  | leader block, world this week |
| slide   | 893 x 502  | 1600 x 900 | 16:9 talk slides, type scaled |

Fonts: Barlow (`FONT_HEAD`) for headlines; Barlow Semi Condensed
(`FONT_BODY`) for everything else. `theme.TYPE_SCALE` in print points:
headline 9.5/11 bold, subtitle 8/9.5 regular, sub-subtitle 7.5/9 light,
tick 7/7.5 regular, annotation 7/7 light, source and period labels 6.5 light.
Web pixels = print points x 1.7922 (`theme.PT_TO_PX`). Coloured y-axis labels
on double-scale charts use Regular, never Light.

## Canonical call pattern (plotly)

```python
import plotly.graph_objects as go
from fedrates import economist, register, save
from fedrates.theme import MAIN, annotate_event, period_shading

register()  # once per process: plotly template, matplotlib rcParams, fonts

fig = go.Figure(go.Scatter(x=dates, y=rates, line=dict(color=MAIN["BLUE"], width=2.2),
                           line_shape="hv"))
economist(
    fig,
    title="Three decades of rate decisions",       # no full stop
    subtitle="United States, effective federal funds rate, %",
    source="Source: FRED; The Economist",           # semicolons, no full stop
    footnote="* Shaded areas mark recessions",
    size="col2",                                    # col1|col2|col3|leader|slide
    legend=False,                                   # True adds square swatches
)
period_shading(fig, [("2007-12-01", "2009-06-30")], labels=["Financial crisis"])
annotate_event(fig, "2008-12-16", "Zero lower bound")
save(fig, "figures/fig.png", size="col2", scale=2)  # .html gives interactive
```

API summary:

- `register(default=True)` — plotly template `"economist"`, matplotlib
  rcParams, seaborn palette, font installation. Idempotent.
- `economist(fig, *, title, subtitle=None, source=None, footnote=None,
  size="col2", legend=True, zeroline="auto", yside="right")` — all furniture.
- `highlight(fig, series, colour=None)` — focus series (default BLUE, pass
  `MAIN["RED"]` when the chart is about that series), all others `#A4BDC9`.
- `period_shading(fig, spans, labels=None)` — PANEL rects behind data,
  uppercase 6.5pt Light labels.
- `annotate_event(fig, x, text, y=None, ax=0, ay=-34)` — 0.5pt leader, 50%
  opacity arrowhead; `y=None` interpolates from the first trace.
- `save(fig, path, size="col2", scale=2)` — PNG via Kaleido, or
  self-contained HTML for `.html` paths; creates parent directories.
- `install_fonts()` — copies TTFs to `~/.local/share/fonts/fedrates` and runs
  `fc-cache -f` so Kaleido's Chromium can see Barlow.
- `mpl_style()` (context manager) and `mpl_furniture(ax, title, subtitle,
  source, footnote)` — matplotlib/seaborn escape hatch with identical
  furniture.

## Layout checklist

- White background; no outer box, no vertical gridlines, no y-axis line.
- Red tag 15x5pt flush to the top-left corner of the chart block.
- Title block above the plotting area, left-aligned: 17pt headline band, 11pt
  subtitle band, 15pt gap before the plot.
- Legend horizontal, top-right on the title band; small filled square
  swatches, no border, no legend title.
- Y tick labels on the RIGHT, outside the plot area. No y tick marks.
- Horizontal gridlines only, 0.5pt, `RULE #B7C6CF`.
- X baseline 0.4pt black; ticks 3pt outward, baseline only; tick labels in a
  10pt band below; abbreviate years after the first (1995, 2000, 05, 10).
- Scale crossing zero: draw the zero rule in ECON RED 0.4pt and drop the black
  baseline and its tick marks. Never rule both.
- Scale not crossing zero: y range starts at zero so the baseline is the zero
  line.
- Source bottom-left, footnote bottom-right, on one 10pt band; 5pt pads.
- Panel charts: 24pt spacer between panels; numbered circles as markers.
- Export PNGs at scale=2; text stays editable, never rasterise labels.

## Chart-type rules

- Max 4 categories normally, 6 is the hard ceiling; use GREY for "other".
- Chronological categories: single-hue light-to-dark ramp (`SEQUENTIAL`).
- Highlight-one charts: focus in BLUE or RED and thicker, all others
  `#A4BDC9`; use `highlight()`.
- Rate changes (cuts vs hikes): two solid colours via `signed()`, never a ramp.
- Bars and columns: never break the scale — use a thermometer chart instead.
  Colour positive/negative differently only when the difference is meaningful.
- Timelines: thin alternating-colour bars; key repeated events; break y
  gridlines so the plot area does not crowd; dotted leaders for bar labels;
  GREY timeline labels when colour already encodes data.
- Double-scale charts: never without good reason; align zero lines; one line
  per axis when possible; prefer panels or indexing. Broken scales
  (line/scatter/thermometer only): black 0.4pt symbol, 6pt wide, between the
  baseline and the first y tick.
- Index charts: 5pt black circle at the index point, 0.5pt red index line.
- Highlight panels and number boxes: PANEL fill, 12pt tall (23pt for two
  lines), 6pt horizontal padding, text at least 6pt inside the box edge.

## Never do this

- Never hand-roll hex values or guess colours — import them from
  `fedrates.theme`.
- Never use plotly or matplotlib defaults: template, palette, fonts and
  margins all come from the theme.
- Never put the y axis on the left.
- Never add vertical gridlines or a y-axis line.
- Never end a title, subtitle, axis label, source line or footnote with a
  full stop.
- Never exceed 6 series in one chart.
- Never use RED for a series that is not the point of the chart; ECON_RED
  is the brand tag only, never a data series.
- Never break a bar or column scale.
- Never add a legend title, a bordered legend box, or coloured legend text.
- Never rasterise text or export below scale=2 for print use.
