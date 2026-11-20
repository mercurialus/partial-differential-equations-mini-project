"""Make every figure in figures/*.pdf (vector). Experimental numbers are read
from results/results.json; smooth analytic curves are evaluated directly.

    python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bsheat import DEFAULTS                                   # noqa: E402
from bsheat.formula import bs_call                            # noqa: E402
from bsheat.transform import constants, u0, u_exact           # noqa: E402
from bsheat.barrier import barrier_call_cn, barrier_call_series  # noqa: E402

FIG = ROOT / "figures"
RES = json.loads((ROOT / "results" / "results.json").read_text())
K, r, sigma, T = (DEFAULTS[k] for k in ("K", "r", "sigma", "T"))
B1, B2 = DEFAULTS["B1"], DEFAULTS["B2"]
k, alpha, beta = constants(r, sigma)

# ---- one style for every figure --------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100")
RAMP = ["#b7d3f2", "#7fb0e8", "#3f86d9", "#1f5fae", "#0d3a73"]   # sequential blue, light -> dark
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "lines.markersize": 6,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "mathtext.fontset": "cm",
})
W2 = (7.0, 2.9)     # two panels side by side
W1 = (4.6, 3.0)     # single panel


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("  figures/" + name + ".pdf")


def panel_label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontweight="bold", va="bottom", ha="right")


# ---- 4. smoothing (the hook figure) -----------------------------------------
def fig_smoothing():
    taus_y = [0.0, 0.02, 0.1, 0.4, 1.0]                # time to expiry in years
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    S = np.linspace(60, 140, 801)
    x = np.linspace(-0.6, 0.6, 801)
    for col, ty in zip(RAMP[::-1], taus_y):
        a1.plot(S, bs_call(S, K, r, sigma, ty), color=col, label=f"{ty:g}")
        tau = 0.5 * sigma**2 * ty
        a2.plot(x, u0(x, k) if tau == 0 else u_exact(x, tau, k), color=col,
                label=f"{tau:.4f}".rstrip("0").rstrip(".") if tau else "0")
    a1.set(xlabel=r"asset price $S$ (currency units)", ylabel=r"call value $V$ (currency units)", ylim=(-1, 42))
    a1.legend(title=r"$T-t$ (years)", title_fontsize=9.5, loc="upper left")
    a2.set(xlabel=r"$x=\ln(S/K)$ (dimensionless)", ylabel=r"$u(x,\tau)$ (dimensionless)")
    a2.legend(title=r"$\tau=\sigma^2(T-t)/2$", title_fontsize=9.5, loc="upper left")
    panel_label(a1, "(a)"); panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig_smoothing")


# ---- slide 3: why a model is needed ----------------------------------------
def fig_paths():
    rng = np.random.default_rng(3)
    n_paths, n_steps = 12, 250
    t = np.linspace(0, T, n_steps + 1)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(T / n_steps)
    logS = np.log(K) + np.cumsum((r - 0.5 * sigma**2) * T / n_steps + sigma * dW, axis=1)
    S = np.exp(np.hstack([np.full((n_paths, 1), np.log(K)), logS]))
    fig, ax = plt.subplots(figsize=W1)
    for i in range(n_paths):
        ends_itm = S[i, -1] > K
        ax.plot(t, S[i], color=C["blue"] if ends_itm else "#b9b8b3", lw=1.1)
    ax.axhline(K, color=C["orange"], lw=1.5, ls="--")
    ax.text(1.02, K, r"$K$", color=INK2, fontsize=10.5, va="center", transform=ax.get_yaxis_transform())
    ax.set(xlabel=r"time $t$ (years)", ylabel=r"share price $S_t$ (currency units)")
    fig.tight_layout()
    save(fig, "fig_paths")


# ---- 1. formula vs Monte Carlo and the other methods ------------------------
def fig_methods():
    rows = RES["exp1"]["rows"]
    S = np.array([q["S"] for q in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    Sf = np.linspace(70, 130, 400)
    a1.plot(Sf, bs_call(Sf, K, r, sigma, T), color=C["blue"], label="formula")
    a1.plot(Sf, np.maximum(Sf - K, 0), color=INK2, lw=1.2, ls="--", label="payoff")
    a1.plot(S, [q["mc"] for q in rows], "o", color=C["orange"], label="Monte Carlo", zorder=3)
    a1.set(xlabel=r"asset price $S$ (currency units)", ylabel=r"$V(S,0)$ (currency units)")
    a1.legend(loc="upper left")
    off = dict(mc=-1.2, explicit=0.0, cn=1.2)
    a2.axhline(0, color=INK2, lw=1)
    d = np.array([q["mc"] - q["formula"] for q in rows])
    lo = np.array([q["mc"] - q["mc_lo"] for q in rows])
    a2.errorbar(S + off["mc"], d, yerr=lo, fmt="o", color=C["orange"], capsize=3, lw=1.5,
                label="Monte Carlo (95% CI)")
    a2.plot(S + off["explicit"], [q["explicit"] - q["formula"] for q in rows], "s", color=C["blue"],
            label="explicit FD")
    a2.plot(S + off["cn"], [q["cn"] - q["formula"] for q in rows], "^", color=C["aqua"],
            label="Crank–Nicolson")
    a2.set(xlabel=r"asset price $S$ (currency units)", ylabel="method $-$ formula (currency units)")
    a2.set_ylim(-0.03, 0.15)
    a2.legend(loc="upper left", fontsize=8.5)
    panel_label(a1, "(a)"); panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig_methods")


# ---- 2. stability -----------------------------------------------------------
def fig_stability():
    rows = RES["exp2"]["rows"]
    lam = np.array([q["lam_eff"] for q in rows])
    err = np.array([q["max_err"] for q in rows])
    fig, ax = plt.subplots(figsize=W1)
    ax.axvspan(0.5, 0.62, color="#f6dfd5", lw=0, zorder=0)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.semilogy(lam, err, "o-", color=C["blue"], label="observed max error")
    g = np.array([q["growth_bound"] for q in rows])
    m = lam > 0.5
    ax.semilogy(lam[m], g[m] * err[lam <= 0.5][-1], ":", color=C["orange"], lw=1.8,
                label=r"$|G|^{n}\times$ error at $\lambda=\frac{1}{2}$")
    ax.text(0.56, 1e-3, "unstable", color=INK2, fontsize=9.5, ha="center")
    ax.set(xlabel=r"$\lambda=\Delta\tau/\Delta x^2$ (dimensionless)",
           ylabel="max price error (currency units)", xlim=(0.28, 0.62))
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_stability")


# ---- 3. convergence -----------------------------------------------------------
def fig_convergence():
    e3 = RES["exp3"]
    dx = np.array([q["dx"] for q in e3["rows"]])
    fig, ax = plt.subplots(figsize=W1)
    ax.loglog(dx, [q["max_err"] for q in e3["rows"]], "o-", color=C["blue"],
              label=fr"max over $S\in[50,200]$ (slope {e3['order_max']:.2f})")
    ax.loglog(dx, [q["err_at_K"] for q in e3["rows"]], "s-", color=C["aqua"],
              label=fr"at $S=K$ (slope {e3['order_at_K']:.2f})")
    ref = e3["rows"][0]["max_err"] * 2.2 * (dx / dx[0]) ** 2
    ax.loglog(dx, ref, "--", color=INK2, lw=1.2, label=r"reference $\Delta x^2$")
    ax.set(xlabel=r"grid spacing $\Delta x$ (dimensionless)", ylabel="price error (currency units)")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks([0.002, 0.005, 0.01, 0.02]); ax.set_xticklabels(["0.002", "0.005", "0.01", "0.02"])
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_convergence")


# ---- 5. barrier ------------------------------------------------------------------
def fig_barrier():
    e5 = RES["exp5"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    S_fine = np.linspace(B1, B2, 400)
    for col, ty in zip([RAMP[1], RAMP[2], RAMP[4]], (0.1, 0.5, 1.0)):
        a1.plot(S_fine, barrier_call_series(S_fine, K, r, sigma, ty, B1, B2, n_terms=300),
                color=col, label=f"series, $T-t={ty:g}$")
        Sg, Vg = barrier_call_cn(K, r, sigma, ty, B1, B2, M=400, n_steps=400)
        a1.plot(Sg[::25], Vg[::25], "o", color=C["orange"], ms=4.5,
                label="Crank–Nicolson" if ty == 1.0 else None, zorder=3)
    a1.axvline(B1, color=INK2, lw=1, ls=":"); a1.axvline(B2, color=INK2, lw=1, ls=":")
    a1.set(xlabel=r"asset price $S$ (currency units)", ylabel=r"knock-out call $V$ (currency units)")
    a1.legend(loc="upper left", fontsize=8.5)
    t = [q["tau"] for q in e5["terms_needed"]]
    N = [q["N"] for q in e5["terms_needed"]]
    a2.semilogx(t, N, "o-", color=C["blue"])
    a2.set(xlabel=r"time to expiry $T-t$ (years)", ylabel=r"terms for $10^{-4}$ accuracy")
    panel_label(a1, "(a)"); panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig_barrier")


# ---- 6. neural networks -----------------------------------------------------------
def fig_nn():
    if "exp6" not in RES:
        return
    summ = RES["exp6"]["summary"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    for kind, col, mk, lab in (("plain", C["orange"], "o", "plain MLP"), ("pinn", C["blue"], "s", "PINN")):
        sel = [q for q in summ if q["kind"] == kind]
        N = [q["N"] for q in sel]
        a1.errorbar(N, [q["rmse_mean"] for q in sel], yerr=[q["rmse_std"] for q in sel],
                    fmt=mk + "-", color=col, capsize=3, label=lab)
        a2.errorbar(N, [q["res_mean"] for q in sel], yerr=[q["res_std"] for q in sel],
                    fmt=mk + "-", color=col, capsize=3, label=lab)
    for a in (a1, a2):
        a.set_xscale("log"); a.set_yscale("log")
        a.xaxis.set_minor_locator(NullLocator())
        a.set_xticks([20, 50, 200]); a.set_xticklabels(["20", "50", "200"])
        a.set_xlabel(r"labelled prices $N$")
    a1.set_ylabel("test price RMSE (currency units)")
    a2.set_ylabel(r"mean $|$PDE residual$|$ (per year)")
    a1.legend(loc="upper right")
    panel_label(a1, "(a)"); panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig_nn")

    f = RES["exp6"]["fields"]
    s, th = np.array(f["s"]), np.array(f["theta"])
    n = int(round(np.sqrt(s.size)))
    S, TH = s.reshape(n, n), th.reshape(n, n)
    fig, axs = plt.subplots(1, 2, figsize=W2, sharey=True)
    vmax = max(np.max(np.abs(f[q]["res"])) for q in ("plain", "pinn"))
    from matplotlib.colors import LogNorm
    for ax, q, lab in zip(axs, ("plain", "pinn"), ("plain MLP", "PINN")):
        im = ax.pcolormesh(S, TH, np.abs(np.array(f[q]["res"])).reshape(n, n), shading="auto",
                           cmap="Blues", norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax), rasterized=True)
        ax.set(xlabel=r"$s=S/K$ (dimensionless)")
        ax.grid(False)
        ax.text(0.03, 0.95, f"{lab}, $N={f['N']}$", transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(fc="white", ec="none", alpha=0.8))
    axs[0].set_ylabel(r"$\theta=T-t$ (years)")
    fig.colorbar(im, ax=axs, label=r"$|$PDE residual$|$ (per year)", shrink=0.9)
    save(fig, "fig_nn_residual")


# ---- 7. ill-posedness -----------------------------------------------------------
def fig_illposed():
    from scipy.fft import dst, idst
    e7 = RES["exp7"]
    n_int, eps = e7["n_modes"], e7["eps"]
    x = np.pi * np.arange(1, n_int + 1) / (n_int + 1)
    rng = np.random.default_rng(7)                      # same data as run_experiments.py
    noisy = np.sin(x) + eps * rng.standard_normal(n_int)
    n = np.arange(1, n_int + 1)
    b = dst(noisy, type=1) / (n_int + 1)                # sine coefficients
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    for col, t, z in zip([C["orange"], RAMP[2], RAMP[4]], (0.001, 0.0005, 0.0), (1, 2, 3)):
        a1.plot(x, idst(dst(noisy, type=1) * np.exp(n**2 * t), type=1), color=col,
                lw=1.0 if t == 0.001 else 2.0, label=f"$t={t:g}$", zorder=z)
    a1.set(xlabel=r"$x$ (dimensionless)", ylabel=r"backward solution $u(x,-t)$", ylim=(-2, 3.6))
    a1.legend(loc="upper center", ncol=3, fontsize=8.5, columnspacing=1.0, handlelength=1.5)
    for col, t in zip([RAMP[1], RAMP[2], RAMP[3], RAMP[4]], (0.0, 0.0005, 0.001, 0.002)):
        a2.semilogy(n, np.abs(b) * np.exp(n**2 * t), color=col, lw=1.4, label=f"$t={t:g}$")
    a2.set(xlabel=r"mode number $n$", ylabel=r"$|b_n|\,e^{n^2 t}$")
    a2.legend(loc="upper left")
    panel_label(a1, "(a)"); panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig_illposed")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (fig_smoothing, fig_paths, fig_methods, fig_stability, fig_convergence, fig_barrier, fig_nn, fig_illposed):
        f()
