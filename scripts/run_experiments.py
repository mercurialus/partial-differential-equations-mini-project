"""Run every experiment and write all numbers to results/results.json.

    python scripts/run_experiments.py            # everything (NN part takes ~20 min on a laptop CPU)
    python scripts/run_experiments.py --skip-nn  # keep the NN numbers already in results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bsheat import DEFAULTS                                       # noqa: E402
from bsheat.formula import bs_call, parity_gap                    # noqa: E402
from bsheat.heat_kernel import bs_call_heat_kernel                # noqa: E402
from bsheat.fd import price_call_fd, solve_call_heat              # noqa: E402
from bsheat.transform import constants, V_from_u                  # noqa: E402
from bsheat.barrier import (barrier_call_cn, barrier_call_series,  # noqa: E402
                            sine_coefficients, barrier_interval, terms_needed)
from bsheat.montecarlo import mc_call                             # noqa: E402

K, r, sigma, T = (DEFAULTS[k] for k in ("K", "r", "sigma", "T"))
B1, B2 = DEFAULTS["B1"], DEFAULTS["B2"]
RESULTS = ROOT / "results" / "results.json"


def exp1_comparison():
    S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    dx, lam, n_cn = 0.0125, 0.4, 400
    formula = bs_call(S, K, r, sigma, T)
    hk = bs_call_heat_kernel(S, K, r, sigma, T)
    ex = price_call_fd(S, K, r, sigma, T, dx=dx, lam=lam, method="explicit")
    cn = price_call_fd(S, K, r, sigma, T, dx=dx, method="cn", n_steps=n_cn)
    mc = [mc_call(s, K, r, sigma, T, n_paths=1_000_000, seed=0) for s in S]
    rows = []
    for i, s in enumerate(S):
        est, lo, hi, se = mc[i]
        rows.append(dict(S=s, formula=formula[i], heat_kernel=hk[i], explicit=ex[i], cn=cn[i],
                         mc=est, mc_lo=lo, mc_hi=hi, mc_se=se,
                         mc_covers=bool(lo <= formula[i] <= hi)))
    err = lambda a: float(np.max(np.abs(np.asarray(a) - formula)))
    return dict(settings=dict(dx=dx, lam=lam, cn_steps=n_cn, mc_paths=1_000_000, L=3.0),
                rows=rows,
                max_err=dict(heat_kernel=err(hk), explicit=err(ex), cn=err(cn),
                             mc=err([m[0] for m in mc])),
                parity_max_gap=float(np.max(np.abs(parity_gap(np.linspace(20, 300, 281), K, r, sigma, T)))))


def _fd_error(dx, lam, window=np.log(2.0), offset=0.0):
    """Max error over grid nodes with |x| <= window (S in [50, 200]), plus the
    error at S = K (the kink node), for the explicit scheme."""
    x, u, info = solve_call_heat(K, r, sigma, T, dx=dx, lam=lam, method="explicit", offset=offset)
    tau = 0.5 * sigma**2 * T
    with np.errstate(over="ignore", invalid="ignore"):
        V = V_from_u(u, x, tau, K, r, sigma)
        e = np.abs(V - bs_call(K * np.exp(x), K, r, sigma, T))
    m = np.abs(x) <= window + 1e-12
    emax = float(np.max(e[m])) if np.all(np.isfinite(e[m])) else float("inf")
    return emax, float(e[np.argmin(np.abs(x))]), info


def exp2_stability():
    dx = 0.0125
    out = []
    for lam in (0.3, 0.4, 0.5, 0.52, 0.55, 0.6):
        emax, eK, info = _fd_error(dx, lam)
        G = abs(1 - 4 * info["lam"])                    # |G| at the highest mode xi dx = pi
        out.append(dict(lam=lam, lam_eff=info["lam"], n_steps=info["n_steps"], max_err=emax,
                        err_at_K=eK, G_max=max(G, 1.0),
                        growth_bound=float(max(G, 1.0) ** info["n_steps"])))
    return dict(dx=dx, rows=out)


def exp3_convergence():
    dxs = [0.025 / 2**m for m in range(5)]
    rows = []
    for dx in dxs:
        emax, eK, info = _fd_error(dx, 0.4)
        rows.append(dict(dx=dx, lam_eff=info["lam"], n_steps=info["n_steps"], max_err=emax, err_at_K=eK))
    ld = np.log([r_["dx"] for r_ in rows])
    p_max = np.polyfit(ld, np.log([r_["max_err"] for r_ in rows]), 1)[0]
    p_K = np.polyfit(ld, np.log([r_["err_at_K"] for r_ in rows]), 1)[0]
    # local orders between successive refinements
    loc = [float(np.log2(rows[i]["max_err"] / rows[i + 1]["max_err"])) for i in range(len(rows) - 1)]
    # same study with the kink one third of the way between two nodes
    off = [_fd_error(dx, 0.4, offset=1 / 3)[0] for dx in dxs]
    p_off = np.polyfit(ld, np.log(off), 1)[0]
    loc_off = [float(np.log2(off[i] / off[i + 1])) for i in range(len(off) - 1)]
    return dict(lam=0.4, rows=rows, order_max=float(p_max), order_at_K=float(p_K), local_orders=loc,
                offnode=dict(offset=1 / 3, max_err=off, order=float(p_off), local_orders=loc_off))


def exp5_barrier():
    k, _, _ = constants(r, sigma)
    a, b, l = barrier_interval(K, B1, B2)
    coeffs = sine_coefficients(a, b, k, 1000)
    comp = []
    for tau in (0.1, 0.5, 1.0):
        S_grid, V_cn = barrier_call_cn(K, r, sigma, tau, B1, B2, M=2000, n_steps=2000)
        V_ser = barrier_call_series(S_grid, K, r, sigma, tau, B1, B2, coeffs=coeffs)
        comp.append(dict(tau=tau, max_diff=float(np.max(np.abs(V_ser - V_cn)))))
    S_tab = np.array([85.0, 90.0, 100.0, 110.0, 120.0, 125.0])
    prices = barrier_call_series(S_tab, K, r, sigma, T, B1, B2, coeffs=coeffs)
    times = np.geomspace(1 / 365, 1.0, 16)
    need = [dict(tau=float(t), N=terms_needed(K, r, sigma, t, B1, B2, tol=1e-4, coeffs=coeffs))
            for t in times]
    return dict(a=a, b=b, ell=l, c_first=coeffs[:6].tolist(), comparison=comp,
                prices=[dict(S=s, barrier=p, vanilla=bs_call(s, K, r, sigma, T))
                        for s, p in zip(S_tab, prices)],
                terms_needed=need, tol=1e-4)


def exp6_nn(seeds=(0, 1, 2), Ns=(20, 50, 200)):
    import torch
    from bsheat.nn import Config, train, evaluate, config_dict, test_grid, pde_residual, exact_w, _t
    torch.set_num_threads(4)
    cfg = Config(adam_iters=10000, lbfgs_iters=3000)
    runs, fields = [], {}
    for N in Ns:
        for kind in ("plain", "pinn"):
            for seed in seeds:
                model, info = train(kind, N, cfg, seed=seed)
                ev = evaluate(model, cfg)
                runs.append({**info, **ev})
                print(f"  NN {kind:5s} N={N:3d} seed={seed}: rmse={ev['rmse']:.4f} "
                      f"res={ev['mean_abs_residual']:.2e} ({info['train_seconds']:.0f}s)", flush=True)
                if seed == 0 and N == Ns[0]:
                    s, th = test_grid(cfg)
                    with torch.no_grad():
                        pred = model(_t(s), _t(th)).numpy()
                    res = pde_residual(model, _t(s), _t(th), cfg.r, cfg.sigma).detach().numpy()
                    fields[kind] = dict(err=(cfg.K * (pred - exact_w(s, th, cfg))).tolist(),
                                        res=res.tolist())
    summary = []
    for N in Ns:
        for kind in ("plain", "pinn"):
            sel = [x for x in runs if x["N"] == N and x["kind"] == kind]
            rm = np.array([x["rmse"] for x in sel])
            rs = np.array([x["mean_abs_residual"] for x in sel])
            summary.append(dict(N=N, kind=kind, rmse_mean=float(rm.mean()), rmse_std=float(rm.std(ddof=1)),
                                res_mean=float(rs.mean()), res_std=float(rs.std(ddof=1)),
                                seconds_mean=float(np.mean([x["train_seconds"] for x in sel]))))
    s, th = test_grid(cfg)
    return dict(config=config_dict(cfg), seeds=list(seeds), runs=runs, summary=summary,
                fields=dict(N=Ns[0], s=s.tolist(), theta=th.tolist(), **fields))


def exp7_illposed():
    """Backward heat equation on (0, pi) with zero boundary values.
    Data f = sin x + noise of size eps on the grid; the exact backward
    evolution multiplies sine coefficient n by e^{+n^2 t}."""
    from scipy.fft import dst, idst
    n_int, eps = 127, 1e-6
    x = np.pi * np.arange(1, n_int + 1) / (n_int + 1)
    rng = np.random.default_rng(7)
    clean = np.sin(x)
    noisy = clean + eps * rng.standard_normal(n_int)
    n = np.arange(1, n_int + 1)
    times = [0.0, 0.0005, 0.001, 0.002, 0.003]
    out = []
    for t in times:
        u_c = idst(dst(clean, type=1) * np.exp(n**2 * t), type=1)
        u_n = idst(dst(noisy, type=1) * np.exp(n**2 * t), type=1)
        out.append(dict(t=t, max_clean=float(np.max(np.abs(u_c))), max_noisy=float(np.max(np.abs(u_n))),
                        max_diff=float(np.max(np.abs(u_n - u_c))), amp_top=float(np.exp(n_int**2 * t))))
    return dict(eps=eps, n_modes=n_int, rows=out,
                hadamard=dict(n=[1, 5, 10, 20], data_size=[1 / m for m in (1, 5, 10, 20)],
                              sol_at_t=[float(np.exp(m**2 * 0.1) / m) for m in (1, 5, 10, 20)], t=0.1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nn", action="store_true", help="reuse NN results from results.json")
    args = ap.parse_args()
    old = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res = dict(params=dict(K=K, r=r, sigma=sigma, T=T, B1=B1, B2=B2))
    k, alpha, beta = constants(r, sigma)
    res["params"].update(k=k, alpha=alpha, beta=beta, tau_final=0.5 * sigma**2 * T)
    for name, fn in [("exp1", exp1_comparison), ("exp2", exp2_stability), ("exp3", exp3_convergence),
                     ("exp5", exp5_barrier), ("exp7", exp7_illposed)]:
        print(f"running {name} ...", flush=True)
        res[name] = fn()
    if args.skip_nn and "exp6" in old:
        res["exp6"] = old["exp6"]
    else:
        print("running exp6 (neural networks) ...", flush=True)
        res["exp6"] = exp6_nn()
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
