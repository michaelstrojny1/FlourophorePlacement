import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

# ------------------------------
# Conditional PINN (u(x,y; D, alpha))
# ------------------------------

class ConditionalMLP(nn.Module):
    def __init__(self, in_dim=4, out_dim=1, width=128, depth=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xydab):
        return self.net(xydab)


def gaussian_source(xy: torch.Tensor, centers=((0.3, 0.3), (0.7, 0.7))):
    x = xy[..., 0:1]
    y = xy[..., 1:2]
    (sx1, sy1), (sx2, sy2) = centers
    s1 = torch.exp(-10 * ((x - sx1) ** 2 + (y - sy1) ** 2))
    s2 = 0.5 * torch.exp(-8 * ((x - sx2) ** 2 + (y - sy2) ** 2))
    return s1 + s2


def laplacian_wrt_xy(u: torch.Tensor, xyDa: torch.Tensor) -> torch.Tensor:
    # xyDa has columns [x, y, D, a]; we need second derivatives wrt x and y only
    grads = autograd.grad(u, xyDa, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    d2 = 0.0
    for i in [0, 1]:  # x and y
        gi = grads[..., i:i+1]
        gii = autograd.grad(gi, xyDa, grad_outputs=torch.ones_like(gi), create_graph=True, retain_graph=True)[0][..., i:i+1]
        d2 = d2 + gii
    return d2


@dataclass
class TrainConfig:
    width: int = 96
    depth: int = 6
    steps: int = 800
    lr: float = 1e-3
    n_interior: int = 1024
    n_boundary: int = 256
    device: str = "cpu"


@dataclass
class DesignConfig:
    M: int = 16
    iters: int = 250
    lr: float = 0.015
    use_fim: bool = True
    fim_weight: float = 1.0
    use_d_optimal: bool = True
    noise_std: float = 0.05  # measurement noise std used in FIM (R = sigma^2 I)
    w_cov: float = 0.1
    w_clust: float = 0.05
    normalize: bool = False
    anneal_pen: bool = False
    anneal_frac: float = 0.5


def train_conditional_pinn(cfg: TrainConfig, D_range=(0.7, 1.5), a_range=(0.5, 2.0), seed=42,
                           centers=((0.3, 0.3), (0.7, 0.7))):
    torch.manual_seed(seed)
    net = ConditionalMLP(width=cfg.width, depth=cfg.depth).to(cfg.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    nb = cfg.n_boundary // 4
    for step in range(cfg.steps):
        # Sample interior points and parameters
        xi = torch.rand(cfg.n_interior, 2, device=cfg.device, requires_grad=True)
        D = torch.rand(cfg.n_interior, 1, device=cfg.device) * (D_range[1] - D_range[0]) + D_range[0]
        a = torch.rand(cfg.n_interior, 1, device=cfg.device) * (a_range[1] - a_range[0]) + a_range[0]
        xida = torch.cat([xi, D, a], dim=1)  # [N,4]

        u = net(xida)
        lap = laplacian_wrt_xy(u, xida)
        src = gaussian_source(xi, centers=centers)
        residual = -D * lap + a * u - src

        # Boundary points (zero Dirichlet)
        t = torch.rand(nb, 1, device=cfg.device)
        b_top = torch.cat([t, torch.ones_like(t)], dim=1)
        b_bottom = torch.cat([t, torch.zeros_like(t)], dim=1)
        b_left = torch.cat([torch.zeros_like(t), t], dim=1)
        b_right = torch.cat([torch.ones_like(t), t], dim=1)
        xb = torch.cat([b_top, b_bottom, b_left, b_right], dim=0)
        Db = torch.rand(xb.shape[0], 1, device=cfg.device) * (D_range[1] - D_range[0]) + D_range[0]
        ab = torch.rand(xb.shape[0], 1, device=cfg.device) * (a_range[1] - a_range[0]) + a_range[0]
        xbda = torch.cat([xb, Db, ab], dim=1)
        ub = net(xbda)

        loss = torch.mean(residual ** 2) + torch.mean(ub ** 2)
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % 200 == 0:
            print(f"Train step {step+1}/{cfg.steps} | loss={loss.item():.6f}")

    return net.eval()


def make_grid(M: int, device: str = "cpu") -> torch.Tensor:
    n = int(np.ceil(np.sqrt(M)))
    xs = torch.linspace(0.0, 1.0, n, device=device)
    ys = torch.linspace(0.0, 1.0, n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    return grid[:M].contiguous()


def compute_variance_criterion(net: nn.Module, X: torch.Tensor, refs: List[Tuple[float, float]], device="cpu"):
    total = 0.0
    X = X.to(device)
    for (D, a) in refs:
        Dcol = torch.full((X.shape[0], 1), float(D), device=device)
        acol = torch.full((X.shape[0], 1), float(a), device=device)
        xida = torch.cat([X, Dcol, acol], dim=1)
        y = net(xida)
        total = total + torch.var(y)
    return total / len(refs)


def compute_fim_criterion(net: nn.Module, X: torch.Tensor, refs: List[Tuple[float, float]],
                          noise_std: float = 0.05, use_d_optimal: bool = True, device="cpu"):
    """
    Compute FIM using exact autograd sensitivities wrt (D, a) on a conditional PINN.
    F = (1/sigma^2) * (1/N) * sum_k (J_k^T J_k), where J_k[j,:] = dy_j/d(D,a) at ref k.
    """
    X = X.to(device)
    sigma2 = float(noise_std) ** 2
    n_params = 2
    F_total = torch.zeros(n_params, n_params, device=device)

    for (D_ref, a_ref) in refs:
        M = X.shape[0]
        J = torch.zeros(M, n_params, device=device)
        for j in range(M):
            xj = X[j]
            D_var = torch.tensor(float(D_ref), device=device, requires_grad=True)
            a_var = torch.tensor(float(a_ref), device=device, requires_grad=True)
            inp = torch.stack([xj[0], xj[1], D_var, a_var]).unsqueeze(0)
            yj = net(inp)
            dD, da = autograd.grad(yj, (D_var, a_var), retain_graph=True)
            J[j, 0] = dD
            J[j, 1] = da
        F_k = (J.t() @ J) / sigma2
        F_total = F_total + F_k

    F = F_total / len(refs)
    # Regularize slightly for numerical stability
    eps = 1e-8
    F = F + eps * torch.eye(n_params, device=device)

    if use_d_optimal:
        val = torch.logdet(F)
    else:
        # A-optimal: minimize trace(F^{-1}) => maximize -trace(F^{-1})
        val = -torch.trace(torch.linalg.inv(F))
    return val


def optimize_sensors(net: nn.Module, refs: List[Tuple[float, float]], design: DesignConfig, device="cpu"):
    X_params = torch.randn(design.M, 2, device=device, requires_grad=True)
    optX = torch.optim.Adam([X_params], lr=design.lr)
    hist = []
    # Baselines for normalization
    X_base = make_grid(design.M, device=device)
    base_fim = None
    base_var = None
    if design.normalize:
        if design.use_fim:
            base_fim = compute_fim_criterion(net, X_base, refs, noise_std=design.noise_std,
                                             use_d_optimal=design.use_d_optimal, device=device).detach()
        else:
            base_var = compute_variance_criterion(net, X_base, refs, device=device).detach()
    for it in range(design.iters):
        X = torch.sigmoid(X_params)
        # Coverage / clustering penalties
        grid_test = torch.rand(100, 2, device=device)
        min_dist = torch.min(torch.cdist(grid_test, X), dim=1)[0]
        coverage_pen = torch.mean(min_dist)
        dists = torch.pdist(X)
        clustering_pen = -torch.mean(torch.exp(-8 * dists))

        # Anneal penalties if requested
        if design.anneal_pen:
            frac = min(1.0, (it + 1) / max(1.0, design.iters * design.anneal_frac))
        else:
            frac = 1.0
        w_cov_t = design.w_cov * frac
        w_clust_t = design.w_clust * frac

        if design.use_fim:
            crit = compute_fim_criterion(net, X, refs, noise_std=design.noise_std,
                                         use_d_optimal=design.use_d_optimal, device=device)
            if design.normalize and base_fim is not None:
                crit = crit - base_fim
            total = crit - w_cov_t * coverage_pen + w_clust_t * clustering_pen
            loss = -total
        else:
            crit = compute_variance_criterion(net, X, refs, device=device)
            if design.normalize and base_var is not None:
                crit = crit - base_var
            total = crit - w_cov_t * coverage_pen + w_clust_t * clustering_pen
            loss = -total

        optX.zero_grad(); loss.backward(); optX.step()
        hist.append(float(loss.item()))
        if (it + 1) % 50 == 0:
            print(f"Iter {it+1}/{design.iters} | loss={loss.item():.6f}")
    return torch.sigmoid(X_params).detach().cpu().numpy(), np.array(hist)


def compare_methods(quick: bool = False, refs_count: int = 5, steps: int = 800, iters: int = 250,
                    width: int = 96, sigma: float = 0.05, aopt: bool = False,
                    normalize: bool = False, w_cov: float = 0.1, w_clust: float = 0.05,
                    anneal_pen: bool = False, anneal_frac: float = 0.5) -> Dict[str, Dict]:
    train = TrainConfig(steps=400 if quick else steps, width=width)
    net = train_conditional_pinn(train)

    refs = []
    rng = np.random.default_rng(42)
    for _ in range(refs_count):
        D = rng.uniform(0.7, 1.5)
        a = rng.uniform(0.5, 2.0)
        refs.append((float(D), float(a)))

    # Variance-based
    var_des = DesignConfig(use_fim=False, iters=150 if quick else iters, w_cov=w_cov, w_clust=w_clust,
                           normalize=normalize, anneal_pen=anneal_pen, anneal_frac=anneal_frac)
    X_var, hist_var = optimize_sensors(net, refs, var_des)

    # FIM-based (principled)
    fim_des = DesignConfig(use_fim=True, iters=150 if quick else iters, noise_std=sigma, use_d_optimal=(not aopt),
                           w_cov=w_cov, w_clust=w_clust, normalize=normalize, anneal_pen=anneal_pen,
                           anneal_frac=anneal_frac)
    X_fim, hist_fim = optimize_sensors(net, refs, fim_des)

    # Summaries
    out = {
        "variance": {"X_opt": X_var.tolist(), "final_loss": float(hist_var[-1]), "history": hist_var.tolist()},
        "fim": {"X_opt": X_fim.tolist(), "final_loss": float(hist_fim[-1]), "history": hist_fim.tolist()},
    }

    with open("conditional_fim_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot convergence
    plt.figure(figsize=(10, 4))
    plt.plot(hist_var, label="Variance", lw=2)
    plt.plot(hist_fim, label="FIM (principled)", lw=2)
    plt.xlabel("Iteration"); plt.ylabel("Design Loss")
    plt.title("Convergence: Variance vs Principled FIM")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig("conditional_fim_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a quicker comparison (fewer steps)")
    parser.add_argument("--refs", type=int, default=5, help="Number of reference parameter pairs")
    parser.add_argument("--steps", type=int, default=800, help="Training steps for conditional PINN")
    parser.add_argument("--iters", type=int, default=250, help="Design iterations for optimization")
    parser.add_argument("--width", type=int, default=96, help="Network width for conditional PINN")
    parser.add_argument("--sigma", type=float, default=0.05, help="Noise std used in FIM (sigma)")
    parser.add_argument("--aopt", action="store_true", help="Use A-optimal criterion (default D-optimal)")
    parser.add_argument("--normalize", action="store_true", help="Normalize criteria against a grid baseline")
    parser.add_argument("--w_cov", type=float, default=0.1, help="Coverage penalty weight")
    parser.add_argument("--w_clust", type=float, default=0.05, help="Anti-clustering penalty weight")
    parser.add_argument("--anneal_pen", action="store_true", help="Linearly anneal penalties for first fraction of iters")
    parser.add_argument("--anneal_frac", type=float, default=0.5, help="Fraction of iters to anneal penalties over")
    args = parser.parse_args()
    compare_methods(quick=args.quick, refs_count=args.refs, steps=args.steps, iters=args.iters,
                    width=args.width, sigma=args.sigma, aopt=args.aopt, normalize=args.normalize,
                    w_cov=args.w_cov, w_clust=args.w_clust, anneal_pen=args.anneal_pen,
                    anneal_frac=args.anneal_frac)
