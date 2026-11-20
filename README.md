# Black–Scholes is the heat equation in disguise

MA5030 (Partial Differential Equations, IIT Hyderabad) mini-project.
Harshil Goyal (ES23BTECH11016), Amogh Bindal (ES23BTECH11004).

The Black–Scholes PDE for a European call is reduced to the heat equation
u_τ = u_xx. The project then checks every step numerically and compares a plain
neural network with a physics-informed one.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # CPU-only PyTorch is fine
```

LaTeX needs, beyond a basic TeX Live, the `siunitx` package (`sudo tlmgr install siunitx`).
The slides use the standard Beamer `Madrid` theme (4:3), which ships with Beamer.

## Build

Every command runs from the repository root.

| command | what it does |
|---|---|
| `make test` | run the pytest suite (~5 s) |
| `make results` | run all experiments → `results/results.json` (~25 min, mostly PINN training) |
| `python scripts/run_experiments.py --skip-nn` | rerun everything except the neural networks (~1 min) |
| `make figures` | `figures/*.pdf` from `results/results.json` |
| `make macros` | `report/results.tex`, `slides/results.tex` (LaTeX macros) from the JSON |
| `make report` / `make slides` | compile with `latexmk -pdf` |
| `make all` | test + figures + report + slides |
| `make clean` | remove LaTeX build files, figures and macros (keeps `results.json`) |

No number in the report or slides is typed by hand: every one goes JSON → `export_macros.py` → `\newcommand`.

## Layout

```
src/bsheat/
  formula.py      closed-form call/put, put–call parity residual
  transform.py    S,t,V <-> x,tau,u; constants k, alpha, beta; initial data u0; closed-form u
  heat_kernel.py  u = Phi * u0 by scipy.integrate.quad -> call price
  fd.py           explicit (FTCS) and Crank–Nicolson schemes for u_tau = u_xx
  barrier.py      double-barrier knock-out call: Fourier sine series, CN reference, terms needed
  montecarlo.py   GBM Monte Carlo with 95% confidence interval
  nn.py           plain MLP vs PINN (PyTorch)
scripts/          run_experiments.py, make_figures.py, export_macros.py
tests/            pytest (parity, limits, heat-equation residual, quadrature, FD stability, barrier, MC, PINN residual)
report/           report.tex, refs.bib
slides/           slides.tex (Beamer Madrid, 28 slides, speaker notes via \note{})
```

## Conventions

Default parameters: K = 100, r = 0.05, σ = 0.25, T = 1, barriers B₁ = 80, B₂ = 130.
Heat variables: S = K eˣ, t = T − 2τ/σ², V = K e^{αx+βτ} u, with k = 2r/σ², α = −(k−1)/2, β = −(k+1)²/4.
