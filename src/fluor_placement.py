"""PINN-based fluorophore placement optimization."""

from dataclasses import dataclass
from typing import Tuple, List, Dict
import random

import torch
import torch.nn as nn
import torch.autograd as autograd

def constrain_to_domain(params_xy: torch.Tensor) -> torch.Tensor:
    """Map unconstrained parameters to [0,1]^2 domain."""
    return torch.sigmoid(params_xy)

def set_seed(seed: int = 1337):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, width=128, depth=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def laplacian(u, x):
    """Compute Laplacian using automatic differentiation."""
    grads = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    lap = 0.0
    for i in range(x.shape[1]):
        g_i = grads[..., i:i+1]
        g_ii = autograd.grad(g_i, x, grad_outputs=torch.ones_like(g_i), create_graph=True, retain_graph=True)[0][..., i:i+1]
        lap = lap + g_ii
    return lap

def gaussian_source(xy: torch.Tensor, center=(0.5, 0.5), sigma=0.12) -> torch.Tensor:
    """Gaussian source term centered at specified location."""
    xc, yc = center
    dx = xy[..., 0:1] - xc
    dy = xy[..., 1:2] - yc
    r2 = (dx*dx + dy*dy) / (2.0 * sigma * sigma)
    return torch.exp(-r2)

@dataclass
class PDELossWeights:
    pde: float = 1.0
    bc: float = 1.0
    obs: float = 1.0

def sample_domain_points(n_interior: int, n_boundary: int, device="cpu"):
    """Sample interior and boundary points for PINN training."""
    xi = torch.rand(n_interior, 2, device=device)
    nb = n_boundary // 4
    t = torch.rand(nb, 1, device=device)
    b_top = torch.cat([t, torch.ones_like(t)], dim=1)
    b_bottom = torch.cat([t, torch.zeros_like(t)], dim=1)
    b_left = torch.cat([torch.zeros_like(t), t], dim=1)
    b_right = torch.cat([torch.ones_like(t), t], dim=1)
    xb = torch.cat([b_top, b_bottom, b_left, b_right], dim=0)
    return xi, xb

def pde_residual(u_net: nn.Module, xy: torch.Tensor, D: torch.Tensor, alpha: torch.Tensor, source_fn=gaussian_source):
    """Compute PDE residual: -D∇²u + αu - f(x,y)."""
    xy.requires_grad_(True)
    u = u_net(xy)
    lap_u = laplacian(u, xy)
    s = source_fn(xy)
    return -D * lap_u + alpha * u - s

def boundary_residual(u_net: nn.Module, xb: torch.Tensor):
    """Compute boundary condition residual (Dirichlet: u=0)."""
    return u_net(xb)

def data_loss(u_net: nn.Module, X_obs: torch.Tensor, y_obs: torch.Tensor):
    """MSE loss between network predictions and observations."""
    return torch.mean((u_net(X_obs) - y_obs) ** 2)


def train_forward_pinn(
    u_net: nn.Module,
    beta: Tuple[float, float],
    n_steps=5000,
    n_interior=2048,
    n_boundary=512,
    lr=1e-3,
    device="cpu",
    loss_w: PDELossWeights = PDELossWeights()
):
    u_net.to(device)
    opt = torch.optim.Adam(u_net.parameters(), lr=lr)
    D_val, alpha_val = beta
    D = torch.tensor([D_val], device=device).view(1,1)
    alpha = torch.tensor([alpha_val], device=device).view(1,1)

    for step in range(n_steps):
        xi, xb = sample_domain_points(n_interior, n_boundary, device)
        r = pde_residual(u_net, xi, D, alpha)
        bc = boundary_residual(u_net, xb)

        loss_pde = torch.mean(r**2)
        loss_bc = torch.mean(bc**2)
        loss = loss_w.pde*loss_pde + loss_w.bc*loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()


    return u_net

def generate_observations(u_net: nn.Module, X: torch.Tensor, noise_std=0.0):
    with torch.no_grad():
        y = u_net(X)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return y

def train_inverse_pinn_few_steps(
    inv_net: nn.Module,
    X_obs: torch.Tensor,
    y_obs: torch.Tensor,
    beta_init: Tuple[float, float],
    n_steps=200,
    n_interior=1024,
    n_boundary=256,
    lr_net=5e-4,
    lr_beta=1e-3,
    device="cpu",
    loss_w: PDELossWeights = PDELossWeights(obs=5.0, pde=1.0, bc=1.0)
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_net.to(device)
    inv_net.train()

    D = nn.Parameter(torch.tensor([[beta_init[0]]], device=device, dtype=torch.float32))
    alpha = nn.Parameter(torch.tensor([[beta_init[1]]], device=device, dtype=torch.float32))

    opt = torch.optim.Adam(
        [{"params": inv_net.parameters(), "lr": lr_net},
         {"params": [D, alpha], "lr": lr_beta}]
    )

    for step in range(n_steps):
        xi, xb = sample_domain_points(n_interior, n_boundary, device)
        r = pde_residual(inv_net, xi, D, alpha)
        bc = boundary_residual(inv_net, xb)
        l_obs = data_loss(inv_net, X_obs, y_obs)

        loss = loss_w.obs*l_obs + loss_w.pde*torch.mean(r**2) + loss_w.bc*torch.mean(bc**2)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return D.detach(), alpha.detach()


@dataclass
class DesignConfig:
    M: int = 12
    r_steps: int = 250
    noise_std: float = 0.0
    lr_design: float = 2e-2
    iters_design: int = 400
    device: str = "cpu"

@dataclass
class NetConfig:
    width: int = 128
    depth: int = 6
    forward_steps: int = 3500
    seed: int = 1337

def optimize_fluor_positions(
    ref_betas: List[Tuple[float, float]],
    net_cfg: NetConfig,
    des_cfg: DesignConfig,
) -> Dict[str, torch.Tensor]:
    set_seed(net_cfg.seed)
    device = des_cfg.device

    forward_nets = []
    for (Di, ai) in ref_betas:
        fnet = MLP(width=net_cfg.width, depth=net_cfg.depth).to(device)
        train_forward_pinn(
            fnet, (Di, ai),
            n_steps=net_cfg.forward_steps,
            device=device
        )
        forward_nets.append(fnet.eval())

    M = des_cfg.M
    params_xy = torch.randn(M, 2, device=device, requires_grad=True)
    optX = torch.optim.Adam([params_xy], lr=des_cfg.lr_design)
    hist = []
    for it in range(des_cfg.iters_design):
        X = constrain_to_domain(params_xy)
        inv_errors = []

        with torch.no_grad():
            for i, (Di, ai) in enumerate(ref_betas):
                fnet = forward_nets[i]

                y_obs = generate_observations(fnet, X, noise_std=des_cfg.noise_std)
                
                inv_net = MLP(width=net_cfg.width, depth=net_cfg.depth).to(device)
                inv_net.load_state_dict(fnet.state_dict())
                
                for p in inv_net.parameters():
                    p.add_(0.005 * torch.randn_like(p))
                D_hat, a_hat = train_inverse_pinn_few_steps(
                    inv_net, X, y_obs,
                    beta_init=(Di, ai),
                    n_steps=des_cfg.r_steps,
                    device=device
                )

                err = (D_hat.item() - Di)**2 + (a_hat.item() - ai)**2
                inv_errors.append(err)

        loss = sum(inv_errors) / len(inv_errors)
        X = constrain_to_domain(params_xy)
        forward_loss = 0.0
        for i, (Di, ai) in enumerate(ref_betas):
            fnet = forward_nets[i]
            y_obs = generate_observations(fnet, X, noise_std=des_cfg.noise_std)
            forward_loss += torch.mean((fnet(X) - y_obs)**2)
        loss = forward_loss / len(ref_betas)

        optX.zero_grad()
        loss.backward()
        optX.step()

        hist.append(loss.detach().item())

    return {"X_opt": constrain_to_domain(params_xy).detach(), "history": torch.tensor(hist)}

def sample_ref_betas(n:int, D_range=(0.7, 1.3), a_range=(0.5, 2.0), seed=123):
    set_seed(seed)
    out = []
    for _ in range(n):
        D = random.uniform(*D_range)
        a = random.uniform(*a_range)
        out.append((D, a))
    return out
