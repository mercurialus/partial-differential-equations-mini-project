import torch

from bsheat.nn import Config, pde_residual


class ExactBS(torch.nn.Module):
    """Black-Scholes w = V/K written in torch, to check the residual operator."""

    def __init__(self, r, sigma):
        super().__init__()
        self.r, self.sigma = r, sigma

    def forward(self, s, th):
        sq = self.sigma * torch.sqrt(th)
        d1 = (torch.log(s) + (self.r + 0.5 * self.sigma**2) * th) / sq
        d2 = d1 - sq
        return s * torch.special.ndtr(d1) - torch.exp(-self.r * th) * torch.special.ndtr(d2)


def test_residual_vanishes_on_exact_solution():
    cfg = Config()
    s = torch.linspace(0.3, 1.9, 50)
    th = torch.linspace(0.05, 1.0, 50)
    res = pde_residual(ExactBS(cfg.r, cfg.sigma), s, th, cfg.r, cfg.sigma)
    assert res.abs().max().item() < 1e-10
