# CLAUDE.md

Analysis and visualisation of US Federal Reserve rate decisions over the last 30 years, using FRED data. Python with plotly as primary and matplotlib/seaborn as secondary. All output follows The Economist's house style, for both charts and prose.

## Delegate

The main agent brainstorms, critiques, decides and manages; it does not write implementation code.

- Delegate ALL plotting and visualisation work to `plot-agent`, invoked BEFORE writing any plotting code or choosing figure size, colours, or panel/legend layout.
- Delegate all other implementation, data-loading and testing to `code-agent`.
- Delegate all prose — READMEs, chart titles and subtitles, slide copy, analysis narrative — to a subagent, and require it to invoke the `economist-writing` skill.
- Background work (long runs, servers, SLURM) belongs to the main agent; `code-agent` hands it back rather than waiting on it.

## Style

- Every chart must use `fedrates.theme`; never hand-roll colours, fonts or layout. The `economist-charts` skill (`.claude/skills/economist-charts/`) is the source of truth for visuals.
- All prose follows the `economist-writing` skill (`.claude/skills/economist-writing/SKILL.md`).

## Commands

| Command | Purpose |
|---|---|
| `uv sync` | Install dependencies into `.venv` |
| `uv run ruff check --fix .` | Lint and autofix |
| `uv run pytest` | Run tests |
| `uv run python -m ...` | Run any module in the project environment |

## Layout

```
src/fedrates/          package (theme.py, data loading, plotting)
data/raw/              untouched FRED downloads (never committed)
data/interim/          cleaned intermediate data (never committed)
data/processed/        analysis-ready data (never committed)
figures/               output charts
notebooks/             exploratory notebooks
docs/reference/        Economist style guides (read-only)
.claude/skills/        economist-charts, economist-writing
```

## Data

- Fetch FRED series via `fredapi`.
- API key lives in `.env` as `FRED_API_KEY` (see `.env.example`); loaded with `python-dotenv`.
- Never commit raw data or the key.
