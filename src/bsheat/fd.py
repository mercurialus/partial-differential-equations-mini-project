"""Finite differences for u_tau = u_xx on a bounded interval [x_0, x_M]
with Dirichlet data u(x_0, tau) = left(tau), u(x_M, tau) = right(tau).

Explicit (FTCS) scheme, lambda = dtau / dx^2:
    u_j^{m+1} = u_j^m + lambda (u_{j+1}^m - 2 u_j^m + u_{j-1}^m).
Von Neumann: G(xi) = 1 - 4 lambda sin^2(xi dx / 2); |G| <= 1 for all xi
iff lambda <= 1/2.

Crank-Nicolson averages the explicit and implicit second differences and is
unconditionally stable. Its weak damping of the highest modes lets the payoff
kink / boundary jump cause oscillations, so the first few steps can be taken
with implicit Euler ("Rannacher start-up").
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

from .transform import constants, u0, u_right_bc, V_from_u


def n_steps_for(tau_final, lam, dx):
    """Number of time steps so that the effective lambda is as close as
    possible to the requested one. Returns (n_steps, lambda_effective)."""
    n = max(1, int(round(tau_final / (lam * dx * dx))))
    return n, tau_final / (n * dx * dx)


def explicit_heat(u_init, dx, tau_final, lam, left, right):
    """March the explicit scheme from tau = 0 to tau_final.

    u_init: values at the M+1 grid nodes; left/right: callables of tau.
    Returns (u_final, lambda_effective, n_steps). Values may blow up (that is
    the point of the stability experiment), so overflow warnings are silenced.
    """
    n, lam_eff = n_steps_for(tau_final, lam, dx)
    dtau = tau_final / n
    u = np.array(u_init, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for m in range(n):
            u[1:-1] = u[1:-1] + lam_eff * (u[2:] - 2.0 * u[1:-1] + u[:-2])
            tau = (m + 1) * dtau
            u[0], u[-1] = left(tau), right(tau)
    return u, lam_eff, n


def _theta_step(u, mu, theta, tau_new, left, right):
    """One theta-method step: theta = 1/2 is Crank-Nicolson, theta = 1 implicit Euler.
    (I - theta mu D) u^{m+1} = (I + (1 - theta) mu D) u^m with D the second difference."""
    n_int = u.size - 2
    a = theta * mu
    ab = np.zeros((3, n_int))
    ab[0, 1:] = -a          # super-diagonal
    ab[1, :] = 1.0 + 2.0 * a
    ab[2, :-1] = -a         # sub-diagonal
    b = 1.0 - theta
    rhs = u[1:-1] + b * mu * (u[2:] - 2.0 * u[1:-1] + u[:-2])
    uL, uR = left(tau_new), right(tau_new)
    rhs[0] += a * uL
    rhs[-1] += a * uR
    new = np.empty_like(u)
    new[1:-1] = solve_banded((1, 1), ab, rhs)
    new[0], new[-1] = uL, uR
    return new


def crank_nicolson_heat(u_init, dx, tau_final, n_steps, left, right, n_implicit=4):
    """Crank-Nicolson from 0 to tau_final in n_steps equal steps; the first
    n_implicit steps use implicit Euler to damp the non-smooth initial data."""
    dtau = tau_final / n_steps
    mu = dtau / (dx * dx)
    u = np.array(u_init, dtype=float)
    for m in range(n_steps):
        theta = 1.0 if m < n_implicit else 0.5
        u = _theta_step(u, mu, theta, (m + 1) * dtau, left, right)
    return u


def heat_grid(dx, L, offset=0.0):
    """Uniform grid on [-L, L] with x = 0 (the payoff kink, S = K) on a node.
    offset (a fraction of dx) shifts every node, putting the kink between
    nodes; it is only used to show why the kink should sit on a node."""
    m = int(round(L / dx))
    if not np.isclose(m * dx, L):
        raise ValueError("L must be an integer multiple of dx so x=0 is a node")
    return dx * (np.arange(-m, m + 1) + offset)


def solve_call_heat(K, r, sigma, tau_years, dx=0.0125, L=3.0, method="explicit",
                    lam=0.4, n_steps=200, offset=0.0):
    """Solve the call problem in heat coordinates on [-L, L].
    Returns (x, u, info) at tau = sigma^2 tau_years / 2."""
    k, _, _ = constants(r, sigma)
    x = heat_grid(dx, L, offset)
    tau = 0.5 * sigma**2 * tau_years
    left = lambda s: 0.0                       # V(0, t) = 0
    right = lambda s: u_right_bc(x[-1], s, k)  # V ~ S - K e^{-r(T-t)}
    init = u0(x, k)
    if method == "explicit":
        u, lam_eff, n = explicit_heat(init, dx, tau, lam, left, right)
        info = dict(lam=lam_eff, n_steps=n)
    elif method == "cn":
        u = crank_nicolson_heat(init, dx, tau, n_steps, left, right)
        info = dict(lam=tau / n_steps / dx**2, n_steps=n_steps)
    else:
        raise ValueError(method)
    return x, u, info


def price_call_fd(S, K, r, sigma, tau_years, **kw):
    """Call price from a finite-difference solution, cubic-spline interpolated
    in x = ln(S/K)."""
    x, u, _ = solve_call_heat(K, r, sigma, tau_years, **kw)
    tau = 0.5 * sigma**2 * tau_years
    V = V_from_u(u, x, tau, K, r, sigma)
    return CubicSpline(x, V)(np.log(np.asarray(S, dtype=float) / K))
