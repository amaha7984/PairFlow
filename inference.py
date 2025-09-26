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


# ---------------------------
# Dataset: source-only, sat images with *_A
# ---------------------------
class EvalSrcDataset(Dataset):
    def __init__(self, sat_dir: str, size: int = 256):
        self.sat_dir = sat_dir
        self.to_src = T.Compose([T.Resize((size, size)),
                                 T.ToTensor(),
                                 T.Lambda(lambda x: x*2-1)])  # [-1,1]

        self.a_names = sorted([n for n in os.listdir(sat_dir)
                               if n.endswith((".jpg", ".png")) and "_A." in n])

        if len(self.a_names) == 0:
            raise RuntimeError("No *_A images found in sat_dir.")

    def __len__(self): return len(self.a_names)

    def __getitem__(self, i: int):
        n = self.a_names[i]
        x0 = Image.open(os.path.join(self.sat_dir, n)).convert("RGB")
        return {"sat_pm1": self.to_src(x0), "name": n}


# ---------------------------
# Time grids
# ---------------------------
def make_t_grid(nfe: int, kind: str = "cosine") -> np.ndarray:
    assert nfe >= 1
    if kind == "uniform":
        return np.linspace(0.0, 1.0, nfe+1, dtype=np.float64)
    if kind == "cosine":
        i = np.arange(nfe+1, dtype=np.float64)
        return 0.5 * (1.0 - np.cos(np.pi * i / nfe))
    if kind == "front2":
        u = np.linspace(0.0, 1.0, nfe+1, dtype=np.float64)
        return u**2
    if kind == "back2":
        u = np.linspace(0.0, 1.0, nfe+1, dtype=np.float64)
        return 1.0 - (1.0 - u)**2
    raise ValueError(f"Unknown schedule kind: {kind}")


def load_t_grid_from_file(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".json":
        with open(path, "r") as f:
            arr = np.array(json.load(f), dtype=np.float64)
    else:
        with open(path, "r") as f:
            txt = f.read().replace(",", " ")
        vals = [float(tok) for tok in txt.split()]
        arr = np.array(vals, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError("t_grid file must be a 1D array/list of floats.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("t_grid must be strictly increasing.")
    if abs(arr[0]) > 1e-12 or abs(arr[-1]-1.0) > 1e-12:
        raise ValueError("t_grid must start at 0.0 and end at 1.0.")
    return arr.astype(np.float64)


# ---------------------------
# RK4 sampler
# ---------------------------
@torch.no_grad()
def rk4_generate_with_grid(model, x_src_pm1: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    device = x_src_pm1.device
    x = torch.randn_like(x_src_pm1)
    assert t_grid.device == device

    B = x_src_pm1.size(0)
    tb = torch.empty(B, device=device, dtype=x.dtype)

    def f(t_s: float, x_s: torch.Tensor):
        tb.fill_(t_s)
        xin = torch.cat([x_s, x_src_pm1], dim=1)  # cond concat
        return model(xin, tb, extra={})

    for i in range(t_grid.numel() - 1):
        t0 = float(t_grid[i].item()); t1 = float(t_grid[i+1].item()); h = t1 - t0
        k1 = f(t0, x)
        k2 = f(t0 + 0.5*h, x + 0.5*h*k1)
        k3 = f(t0 + 0.5*h, x + 0.5*h*k2)
        k4 = f(t1,         x + h*k3)
        x  = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x


def _pm1_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0


# ---------------------------
# Robust weight loader
# ---------------------------
def load_state_dict_strict(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
    new_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    model.load_state_dict(new_sd, strict=True)


# ---------------------------
# DDP helpers
# ---------------------------
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


# ---------------------------
# Global counter helpers (from multisampling)
# ---------------------------
def _add_done(done_file: str, inc: int) -> int:
    with open(done_file, "a+b") as f:
        f.seek(0)
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        cur = int(raw.decode() or "0")
        new = cur + inc
        f.seek(0); f.truncate(0); f.write(str(new).encode()); f.flush()
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
        f.seek(0); f.truncate(0); f.write(str(end).encode()); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return (start, end)


# ---------------------------
# Dynamic generation with global progress
# ---------------------------
@torch.no_grad()
def generate_all_dynamic(
    model,
    device,
    ds: EvalSrcDataset,
    gen_dir: str,
    t_grid: torch.Tensor,
    batch_size: int,
    cursor_file: str,
    chunk_size: int,
):
    os.makedirs(gen_dir, exist_ok=True)
    model.eval()

    is_dist  = dist.is_available() and dist.is_initialized()
    rank     = dist.get_rank() if is_dist else 0
    world    = dist.get_world_size() if is_dist else 1
    is_rank0 = (rank == 0)

    n_total = len(ds)
    done_file = cursor_file + ".done"

    if is_rank0:
        with open(cursor_file, "wb") as f: f.write(b"0")
        with open(done_file, "wb") as f: f.write(b"0")
    if is_dist: dist.barrier()

    pbar = tqdm(total=n_total, desc=f"Generate [rank {rank}/{world}]", ncols=100, disable=not is_rank0)

    while True:
        claim = _claim_next_chunk(cursor_file, chunk_size, n_total)
        if claim is None:
            break
        start, end = claim

        idx = start
        while idx < end:
            jend = min(idx + batch_size, end)

            xs, names = [], []
            for i in range(idx, jend):
                item = ds[i]
                xs.append(item["sat_pm1"])
                names.append(item["name"])
            x_src = torch.stack(xs, dim=0).to(device, non_blocking=True)

            gen_pm1 = rk4_generate_with_grid(model, x_src, t_grid)
            gen01   = _pm1_to_01(gen_pm1)

            gen01_cpu = gen01.detach().cpu()
            for j in range(gen01_cpu.size(0)):
                nm = names[j]
                base, _ = os.path.splitext(nm)
                out_name = (base.replace("_A", "_B_pred") if "_A" in base else base + "_pred") + ".png"
                save_image(gen01_cpu[j], os.path.join(gen_dir, out_name))

            del x_src, gen_pm1, gen01, gen01_cpu
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


# ---------------------------
# Main
# ---------------------------

#This inference logic is partially motivated from article:
#"Align Your Steps: Optimizing Sampling Schedules in Diffusion Models"
def main(rank, world_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--sat_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--gen_dir", type=str, default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--epoch_tag", type=int, default=0)
    parser.add_argument("--nfe", type=int, default=50)
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["uniform", "cosine", "front2", "back2"])
    parser.add_argument("--t_grid_file", type=str, default=None)
    args = parser.parse_args()

    local_rank = setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Model
    model = UNetModel(
        in_channels=6,
        model_channels=96,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=48,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        with_fourier_features=False,
    ).to(device)

    load_state_dict_strict(model, args.weights, device)
    model.eval()

    # Time grid
    if args.t_grid_file is not None:
        t_np = load_t_grid_from_file(args.t_grid_file)
    else:
        t_np = make_t_grid(args.nfe, args.schedule)
    t_grid = torch.from_numpy(t_np).to(device=device, dtype=torch.float32)

    # Data
    ds = EvalSrcDataset(args.sat_dir, size=args.size)

    # Output + shared cursor
    gen_dir = args.gen_dir or os.path.join(args.out_dir, f"epoch_{args.epoch_tag}_gen")
    os.makedirs(args.out_dir, exist_ok=True)
    cursor_file = os.path.join(args.out_dir, f".cursor_e{args.epoch_tag}.txt")

    # Generate
    generate_all_dynamic(
        model=model,
        device=device,
        ds=ds,
        gen_dir=gen_dir,
        t_grid=t_grid,
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
