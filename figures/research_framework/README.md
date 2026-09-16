# Research framework figure (`fig:research_framework`)

Publication-quality vector figure for the paper methods section.

## Files

| File | Purpose |
|------|---------|
| `research_framework.tikz` | TikZ snippet — `\input{...}` from main paper |
| `research_framework.tex` | Standalone wrapper for PDF/SVG export |
| `research_framework.pdf` | Compiled vector PDF (after build) |
| `research_framework.svg` | SVG export (after build) |

## Build

```bash
cd figures/research_framework
pdflatex -interaction=nonstopmode research_framework.tex

# SVG (requires dvisvgm)
dvisvgm --pdf research_framework.pdf -o research_framework.svg
# Alternative:
# pdf2svg research_framework.pdf research_framework.svg
```

## LaTeX inclusion (figure*)

```latex
\begin{figure*}[t]
  \centering
  \input{figures/research_framework/research_framework.tikz}
  \caption{Research framework for simulation-based household currency-allocation
    optimization. Historical KZT/USD and KZT/EUR exchange-rate data are processed
    and statistically evaluated before the selected joint stationary-bootstrap
    model generates FX scenarios within the Monte Carlo framework. The simulated
    FX paths are propagated through the household cash-flow model, producing
    cumulative expenditure shortfalls that are evaluated using downside-risk
    measures and minimized over feasible KZT--USD--EUR savings allocations.}
  \label{fig:research_framework}
\end{figure*}
```

Requires in preamble: `\usepackage{tikz}` and libraries
`arrows.meta`, `positioning`, `calc`, `fit`, `backgrounds`.
