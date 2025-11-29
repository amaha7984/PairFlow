import os, json, argparse, fcntl
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from datetime import timedelta


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNetModel

# SD VAE for latent encode/decode
from diffusers.models import AutoencoderKL

# ---------------------------
# Dataset: source-only, sat images with *_A
# ---------------------------
class EvalSrcDataset(Dataset):
    def __init__(self, sat_dir: str, size: int = 256):
        self.sat_dir = sat_dir
        self.to_src = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1] for VAE
        ])

        self.a_names = sorted([
            n for n in os.listdir(sat_dir)
            if n.lower().endswith((".jpg", ".jpeg", ".png")) and "_A." in n
        ])

        if len(self.a_names) == 0:
            raise RuntimeError(f"No *_A images found in sat_dir={sat_dir}")

    def __len__(self):
        return len(self.a_names)

    def __getitem__(self, i: int):
        n = self.a_names[i]
        x0 = Image.open(os.path.join(self.sat_dir, n)).convert("RGB")
        return {"sat_pm1": self.to_src(x0), "name": n}


def _pm1_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0


# =========================================================
# Robust weight loader
# =========================================================
def load_state_dict_strict(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        new_sd[k] = v
    model.load_state_dict(new_sd, strict=True)


# =========================================================
# DDP helpers
# =========================================================
def setup_ddp(rank, world_size):
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2),
    )
    return local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# =========================================================
# Global progress helpers
# =========================================================
def _add_done(done_file: str, inc: int) -> int:
    with open(done_file, "a+b") as f:
        f.seek(0)
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        cur = int(raw.decode() or "0")
        new = cur + inc
        f.seek(0)
        f.truncate(0)
        f.write(str(new).encode())
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return new


def _claim_next_chunk(cursor_file: str, chunk_size: int, n_total: int):
    with open(cursor_file, "a+b") as f:
        f.seek(0)
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        cur = int(raw.decode() or "0")
        if cur >= n_total:
            fcntl.flock(f, fcntl.LOCK_UN)
            return None
        start = cur
        end = min(cur + chunk_size, n_total)
        f.seek(0)
        f.truncate(0)
        f.write(str(end).encode())
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return (start, end)


# =========================================================
# Cosine-schedule Gaussian FM RK4 sampler
# (matches fid_eval_i2i_gaussian_cosine.py)
# =========================================================
@torch.no_grad()
def _rk4_generate_latent_gaussian_cosine(
    model: torch.nn.Module,
    z_src: torch.Tensor,
    steps: int = 50,
) -> torch.Tensor:
    """
    Cosine-schedule Gaussian FM sampling:
    integrate reverse-time ODE from t=1 (noise) -> t=0 (data).

    This matches the sampler used in fid_eval_i2i_gaussian_cosine.py.
    """
    device = z_src.device
    z = torch.randn_like(z_src)  # start from noise at t=1
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    def f_scalar(t_s: float, z_s: torch.Tensor):
        tb = torch.full((z_s.size(0),), t_s, device=device, dtype=z_s.dtype)
        zin = torch.cat([z_s, z_src], dim=1)  # [z_t, z_src]
        return model(zin, tb, extra={})      # velocity v_theta

    for i in range(steps):
        t0, t1 = ts[i].item(), ts[i + 1].item()
        h = t1 - t0  # negative step
        k1 = f_scalar(t0,         z)
        k2 = f_scalar(t0 + 0.5*h, z + 0.5*h*k1)
        k3 = f_scalar(t0 + 0.5*h, z + 0.5*h*k2)
        k4 = f_scalar(t1,         z + h*k3)
        z  = z + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return z


# =========================================================
# Dynamic latent-space generation (DDP-safe)
# =========================================================
@torch.no_grad()
def generate_all_dynamic(
    model: torch.nn.Module,
    vae: AutoencoderKL,
    scale_factor: float,
    device: torch.device,
    ds: EvalSrcDataset,
    gen_dir: str,
    steps: int,
    batch_size: int,
    cursor_file: str,
    chunk_size: int,
):
    os.makedirs(gen_dir, exist_ok=True)
    model.eval()
    vae.eval()

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world = dist.get_world_size() if is_dist else 1
    is_rank0 = (rank == 0)

    n_total = len(ds)
    done_file = cursor_file + ".done"

    # Init shared counters
    if is_rank0:
        with open(cursor_file, "wb") as f:
            f.write(b"0")
        with open(done_file, "wb") as f:
            f.write(b"0")
    if is_dist:
        dist.barrier()

    pbar = tqdm(
        total=n_total,
        desc=f"[Latent Gen GFM-Cosine] rank {rank}/{world}",
        ncols=100,
        disable=not is_rank0,
    )

    while True:
        claim = _claim_next_chunk(cursor_file, chunk_size, n_total)
        if claim is None:
            break
        start, end = claim

        idx = start
        while idx < end:
            jend = min(idx + batch_size, end)

            # Collect batch
            xs, names = [], []
            for i in range(idx, jend):
                item = ds[i]
                xs.append(item["sat_pm1"])
                names.append(item["name"])
            x_src_pm1 = torch.stack(xs, dim=0).to(device, non_blocking=True)  # [-1,1]

            # ---- Encode source to latent ----
            with torch.no_grad():
                latents = vae.encode(x_src_pm1).latent_dist.sample()
                z_src = latents * scale_factor  # (B,4,H',W')

            # ---- Integrate in latent space with Gaussian-cosine FM ----
            z_T0 = _rk4_generate_latent_gaussian_cosine(
                model=model,
                z_src=z_src,
                steps=steps,
            )

            # ---- Decode to pixel space ----
            gen_pm1 = vae.decode(z_T0 / scale_factor).sample  # [-1,1]
            gen01 = _pm1_to_01(gen_pm1)                       # [0,1] for saving

            gen01_cpu = gen01.detach().cpu()
            for j in range(gen01_cpu.size(0)):
                nm = names[j]
                base, _ = os.path.splitext(nm)
                # mirror A/B naming like non-latent script
                if "_A" in base:
                    out_name = base.replace("_A", "_B_pred") + ".png"
                else:
                    out_name = base + "_pred.png"
                save_image(gen01_cpu[j], os.path.join(gen_dir, out_name))

            # cleanup
            del x_src_pm1, z_src, z_T0, gen_pm1, gen01, gen01_cpu

            n_inc = jend - idx
            new_total = _add_done(done_file, n_inc)
            if is_rank0:
                pbar.n = new_total
                pbar.refresh()

            idx = jend

    if is_rank0:
        pbar.close()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# =========================================================
# Main
# =========================================================
def main(rank, world_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to latent UNet checkpoint (.pth) trained with Gaussian-cosine FM")
    parser.add_argument("--sat_dir", type=str, required=True,
                        help="Directory of source satellite *_A images")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory for logs/cursor files")
    parser.add_argument("--gen_dir", type=str, default=None,
                        help="Directory to save generated images "
                             "(default: out_dir/epoch_<tag>_gen_latent_gfm_cosine)")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--epoch_tag", type=int, default=0)
    parser.add_argument("--nfe", type=int, default=50,
                        help="Number of function evaluations (RK4 steps) for Gaussian-cosine FM sampler")
    parser.add_argument("--scale_factor", type=float, default=0.18215,
                        help="Scale factor used for SD-VAE latents (match training).")
    args = parser.parse_args()

    local_rank = setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # -------- Latent UNet model (must match train_latent_gaussian_cosine.py) --------
    model = UNetModel(
        in_channels=8,         # [z_t (4) + z_src (4)]
        model_channels=192,
        out_channels=4,        # latent velocity
        num_res_blocks=3,
        attention_resolutions=(2, 4, 8),
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        with_fourier_features=False,
    ).to(device)

    load_state_dict_strict(model, args.weights, device)
    model.eval()

    # -------- SD VAE (frozen) --------
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # -------- Data --------
    ds = EvalSrcDataset(args.sat_dir, size=args.size)

    # -------- Output paths --------
    os.makedirs(args.out_dir, exist_ok=True)
    if args.gen_dir is not None:
        gen_dir = args.gen_dir
    else:
        gen_dir = os.path.join(args.out_dir, f"epoch_{args.epoch_tag}_gen_latent_gfm_cosine")
    os.makedirs(gen_dir, exist_ok=True)

    cursor_file = os.path.join(args.out_dir, f".cursor_latent_gfm_cosine_e{args.epoch_tag}.txt")

    # -------- Generate --------
    generate_all_dynamic(
        model=model,
        vae=vae,
        scale_factor=args.scale_factor,
        device=device,
        ds=ds,
        gen_dir=gen_dir,
        steps=args.nfe,
        batch_size=args.batch_size,
        cursor_file=cursor_file,
        chunk_size=args.chunk_size,
    )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size)
