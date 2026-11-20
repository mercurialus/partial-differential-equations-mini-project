"""Plain MLP vs physics-informed neural network (PINN) for the call price.

Inputs (s, theta) = (S/K, T - t), output w = V/K. In these variables the
Black-Scholes equation reads (V_t = -K w_theta, V_S = w_s, V_SS = w_ss / K)

    R[w] := -w_theta + 0.5 sigma^2 s^2 w_ss + r s w_s - r w = 0,
    w(s, 0) = max(s - 1, 0),  w(0, theta) = 0,  w(s_max, theta) = s_max - e^{-r theta}.

Plain network: loss = mean squared error on N labelled prices.
PINN: loss = data + PDE residual on collocation points + terminal + boundary.
"""
from dataclasses import dataclass, asdict
import time

import numpy as np
import torch
from scipy.stats import qmc

from .formula import bs_call

torch.set_default_dtype(torch.float64)


@dataclass
class Config:
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.25
    T: float = 1.0          # theta in [0, T]
    s_max: float = 2.0      # s in [0, s_max]
    width: int = 32
    depth: int = 3          # hidden layers
    n_colloc: int = 4096    # power of 2 for the Sobol sequence
    n_terminal: int = 400
    n_boundary: int = 200
    adam_iters: int = 6000
    lr: float = 2e-3
    lbfgs_iters: int = 1000
    w_pde: float = 1.0
    w_tc: float = 1.0
    w_bc: float = 1.0
    theta_min_test: float = 0.02   # test grid avoids the kink at theta = 0
    n_test: int = 60               # test grid is n_test x n_test


class MLP(torch.nn.Module):
    """(s, theta) -> w. Inputs are rescaled to [-1, 1] inside the network."""

    def __init__(self, cfg: Config):
        super().__init__()
        layers, d = [], 2
        for _ in range(cfg.depth):
            layers += [torch.nn.Linear(d, cfg.width), torch.nn.Tanh()]
            d = cfg.width
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)
        self.register_buffer("scale", torch.tensor([cfg.s_max, cfg.T]))

    def forward(self, s, theta):
        z = 2.0 * torch.stack([s, theta], dim=-1) / self.scale - 1.0
        return self.net(z).squeeze(-1)


def pde_residual(model, s, theta, r, sigma):
    """R[w] at the given points, derivatives by automatic differentiation."""
    s = s.clone().requires_grad_(True)
    theta = theta.clone().requires_grad_(True)
    w = model(s, theta)
    w_s, w_th = torch.autograd.grad(w.sum(), (s, theta), create_graph=True)
    w_ss = torch.autograd.grad(w_s.sum(), s, create_graph=True)[0]
    return -w_th + 0.5 * sigma**2 * s**2 * w_ss + r * s * w_s - r * w


def exact_w(s, theta, cfg: Config):
    """Black-Scholes price divided by K (numpy)."""
    return bs_call(cfg.K * np.asarray(s), cfg.K, cfg.r, cfg.sigma, np.asarray(theta)) / cfg.K


def sample_labelled(N, cfg: Config, seed):
    """N Latin-hypercube points in [0, s_max] x [0, T] with exact labels."""
    pts = qmc.LatinHypercube(d=2, seed=seed).random(N) * [cfg.s_max, cfg.T]
    return pts[:, 0], pts[:, 1], exact_w(pts[:, 0], pts[:, 1], cfg)


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=float))


def train(kind, N, cfg: Config = Config(), seed=0, verbose=False):
    """Train a 'plain' or 'pinn' model on N labelled prices.
    Returns (model, info dict)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    model = MLP(cfg)
    s_d, th_d, w_d = map(_t, sample_labelled(N, cfg, seed))

    # physics points (only used by the PINN)
    col = qmc.Sobol(d=2, seed=seed).random(cfg.n_colloc) * [cfg.s_max, cfg.T]
    s_c, th_c = _t(col[:, 0]), _t(col[:, 1])
    s_tc = _t(rng.uniform(0, cfg.s_max, cfg.n_terminal))
    w_tc = torch.clamp(s_tc - 1.0, min=0.0)
    th_bc = _t(rng.uniform(0, cfg.T, cfg.n_boundary))

    def loss_fn():
        loss = torch.mean((model(s_d, th_d) - w_d) ** 2)
        if kind == "pinn":
            res = pde_residual(model, s_c, th_c, cfg.r, cfg.sigma)
            loss = loss + cfg.w_pde * torch.mean(res**2)
            loss = loss + cfg.w_tc * torch.mean((model(s_tc, torch.zeros_like(s_tc)) - w_tc) ** 2)
            left = model(torch.zeros_like(th_bc), th_bc)
            right = model(torch.full_like(th_bc, cfg.s_max), th_bc) - (cfg.s_max - torch.exp(-cfg.r * th_bc))
            loss = loss + cfg.w_bc * (torch.mean(left**2) + torch.mean(right**2))
        return loss

    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.adam_iters, eta_min=cfg.lr / 50)
    for it in range(cfg.adam_iters):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and it % 1000 == 0:
            print(f"  [{kind} N={N}] adam {it:5d} loss {loss.item():.3e}")

    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters,
                                  tolerance_grad=1e-12, tolerance_change=1e-15,
                                  history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            l = loss_fn()
            l.backward()
            return l
        lbfgs.step(closure)

    final_loss = loss_fn().item()
    return model, dict(kind=kind, N=N, seed=seed, final_loss=final_loss,
                       train_seconds=time.time() - t0)


def test_grid(cfg: Config):
    s = np.linspace(0.0, cfg.s_max, cfg.n_test)
    th = np.linspace(cfg.theta_min_test, cfg.T, cfg.n_test)
    S, TH = np.meshgrid(s, th, indexing="ij")
    return S.ravel(), TH.ravel()


def evaluate(model, cfg: Config = Config()):
    """Price RMSE (in currency units, i.e. times K) and mean |R[w]| on the test grid."""
    s, th = test_grid(cfg)
    st, tht = _t(s), _t(th)
    with torch.no_grad():
        pred = model(st, tht).numpy()
    rmse = cfg.K * np.sqrt(np.mean((pred - exact_w(s, th, cfg)) ** 2))
    res = pde_residual(model, st, tht, cfg.r, cfg.sigma).detach().numpy()
    return dict(rmse=float(rmse), mean_abs_residual=float(np.mean(np.abs(res))))


def config_dict(cfg: Config):
    return asdict(cfg)
