import torch

def _pm1_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0


@torch.no_grad()
def _rk4_generate_latent_gaussian_cosine(model, z_src, steps=50):
    device = z_src.device
    z = torch.randn_like(z_src)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    def f_scalar(t_s, z_s):
        tb = torch.full((z_s.size(0),), t_s, device=device, dtype=z_s.dtype)
        zin = torch.cat([z_s, z_src], dim=1)
        return model(zin, tb, extra={})

    for i in range(steps):
        t0, t1 = ts[i].item(), ts[i+1].item()
        h = t1 - t0
        k1 = f_scalar(t0, z)
        k2 = f_scalar(t0 + 0.5*h, z + 0.5*h*k1)
        k3 = f_scalar(t0 + 0.5*h, z + 0.5*h*k2)
        k4 = f_scalar(t1, z + h*k3)
        z  = z + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)
    return z
