import os
import torch
import torch.distributed as dist

def setup_ddp(rank, world_size):
    """Initialize distributed process group (NCCL) and set device."""
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Destroy distributed process group cleanly."""
    dist.destroy_process_group()

def cosine_bar_alpha(t, s=0.008, eps=1e-5):
    # t in [0,1], returns bar_alpha in (0,1)
    theta = ((t + s) / (1.0 + s)) * (torch.pi / 2.0)
    a = torch.cos(theta) ** 2
    return a.clamp(eps, 1.0 - eps)

def d_cosine_bar_alpha_dt(t, s=0.008):
    # d/dt cos^2(theta) = -sin(2*theta) * dtheta/dt
    theta = ((t + s) / (1.0 + s)) * (torch.pi / 2.0)
    dtheta_dt = (torch.pi / 2.0) / (1.0 + s)
    return -torch.sin(2.0 * theta) * dtheta_dt

def gaussian_cosine_path(z0, z1, t_b, s=0.008, eps=1e-5):
    # t_b: [B,1,1,1] in [0,1]
    t_b = t_b.clamp(0.0 + eps, 1.0 - eps)

    a = cosine_bar_alpha(t_b, s=s, eps=eps)           # bar_alpha(t)
    da_dt = d_cosine_bar_alpha_dt(t_b, s=s)           # d/dt bar_alpha

    alpha_t = torch.sqrt(a)                           # sqrt(bar_alpha)
    sigma_t = torch.sqrt((1.0 - a).clamp(eps, 1.0))   # sqrt(1 - bar_alpha)

    # derivatives of alpha_t and sigma_t
    dalpha_dt = (0.5 / alpha_t) * da_dt
    dsigma_dt = (-0.5 / sigma_t) * da_dt

    z_t = alpha_t * z0 + sigma_t * z1
    u   = dalpha_dt * z0 + dsigma_dt * z1             # target velocity
    return z_t, u
