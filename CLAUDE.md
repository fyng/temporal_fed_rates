# CLAUDE.md

Analysis and visualisation of US Federal Reserve rate decisions over the last 30 years, using FRED data. Python with plotly as primary and matplotlib/seaborn as secondary. All output follows The Economist's house style, for both charts and prose.

## Analysis loop

Every analysis runs through five steps, in order. Do not jump to step 5: a production plot is only valid once the user has agreed the message in step 4.

1. **Brainstorm.** Ideate with the user. Give critical feedback, resolve forks with `AskUserQuestion`, converge on a question worth asking in 2-3 turns.
2. **Research.** Search the economics literature and reputable outlets — The Economist, the FT, NBER papers, Fed working papers and speeches. Find what is already known and what is contested. Update the plan, and tell the user what changed and why.
3. **Explore.** Implement the idea in code and plots. Show the raw data comprehensively: many series, small multiples, full date ranges, minimal editing. These plots exist to inform the next decision, not to persuade. Write to `figures/explore/`.
4. **Synthesise.** Work out what the data supports and say it in one sentence per chart. Iterate with the user over code, plots and draft explainer text until the draft and the charts stop moving.
5. **Produce.** Build the final charts in `figures/production/`. Where an exploratory chart shows more to inform, a production chart shows less to convince: one message, annotated, sourced, house style throughout. Judge on clarity, accuracy and consistent aesthetics.

Delegation within the loop: steps 1, 2 and 4 belong to the main agent. Step 3 and 5 go to `plot-agent` and `code-agent`; explainer text goes to a prose subagent under the `economist-writing` skill.

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
figures/explore/       exploratory charts, step 3
figures/production/    final charts, step 5
notebooks/             exploratory notebooks
docs/reference/        Economist style guides (read-only)
.claude/skills/        economist-charts, economist-writing
```

## Data

- Fetch FRED series via `fredapi`.
- API key lives in `.env` as `FRED_API_KEY` (see `.env.example`); loaded with `python-dotenv`.
- Never commit raw data or the key.
