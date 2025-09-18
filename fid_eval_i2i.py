import os, random
import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class EvalPairDataset(Dataset):
    def __init__(self, sat_dir, map_dir, size=256):
        self.sat_dir = sat_dir
        self.map_dir = map_dir
        self.to_src  = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Lambda(lambda x: x*2-1)])  # [-1,1]
        self.to_real = T.Compose([T.Resize((size, size)), T.ToTensor()])                              # [0,1]

        a_names = sorted([n for n in os.listdir(sat_dir) if n.endswith((".jpg", ".png")) and "_A." in n])
        self.pairs = []
        for n in a_names:
            # 10554_A.jpg -> 10554_B.jpg (keeps the same extension)
            base, ext = os.path.splitext(n)
            m = base.replace("_A", "_B") + ext
            if os.path.exists(os.path.join(map_dir, m)):
                self.pairs.append((n, m))

        if len(self.pairs) == 0:
            raise RuntimeError("No A→B pairs found. Expected *_A.ext in A and *_B.ext in B.")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        n, m = self.pairs[i]
        x0 = Image.open(os.path.join(self.sat_dir, n)).convert("RGB")
        x1 = Image.open(os.path.join(self.map_dir, m)).convert("RGB")
        return {"sat_pm1": self.to_src(x0), "map_01": self.to_real(x1), "name": n}


@torch.no_grad()
def _rk4_generate(model, x_src_pm1, steps=50):
    """Option-A RK4: start from noise; dx/dt = v_theta(x,t|x_src)."""
    device = x_src_pm1.device
    x = torch.randn_like(x_src_pm1)
    ts = torch.linspace(0.0, 1.0, steps+1, device=device)
    for i in range(steps):
        t0, t1 = ts[i].item(), ts[i+1].item()
        h = t1 - t0
        def f(t_s, x_s):
            tb = torch.full((x_s.size(0),), t_s, device=device, dtype=x_s.dtype)
            xin = torch.cat([x_s, x_src_pm1], dim=1)
            return model(xin, tb, extra={})
        k1 = f(t0, x)
        k2 = f(t0+0.5*h, x+0.5*h*k1)
        k3 = f(t0+0.5*h, x+0.5*h*k2)
        k4 = f(t0+h,     x+h*k3)
        x  = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x

def _pm1_to_01(x): return (x.clamp(-1,1)+1)/2

@torch.no_grad()
def eval_fid_i2i(
    model, device,
    sat_dir, map_dir,
    out_dir, epoch, steps=50, batch_size=16, num_workers=4,
    save_samples=10
):
   
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # Dist info
    is_dist  = dist.is_available() and dist.is_initialized()
    rank     = dist.get_rank() if is_dist else 0
    world    = dist.get_world_size() if is_dist else 1
    is_rank0 = (rank == 0)


    ds = EvalPairDataset(sat_dir, map_dir, size=256)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if is_dist else None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler,
                    num_workers=num_workers, pin_memory=True, drop_last=False)

    fid = FrechetInceptionDistance(normalize=True).to(device)

    # Save a few visual samples (triptychs) using the same RK4 path
   
    if is_rank0 and save_samples > 0:
        sample_dir = os.path.join(out_dir, f"epoch_{epoch}_samples")
        os.makedirs(sample_dir, exist_ok=True)
        idxs = random.sample(range(len(ds)), min(save_samples, len(ds)))
        for i, idx in enumerate(idxs):
            b  = ds[idx]
            x0 = b["sat_pm1"].unsqueeze(0).to(device)
            x1 = b["map_01"].unsqueeze(0).to(device)
            gen = _rk4_generate(model, x0, steps=steps)
            gen01 = _pm1_to_01(gen)
            vis = torch.cat([x0*0.5+0.5, x1, gen01], dim=0)
            save_image(vis, os.path.join(sample_dir, f"{i:02d}.png"), nrow=3)
            del x0, x1, gen, gen01, vis

    # Progress bar (rank 0 shows its shard progress)
    total_local = len(sampler) if sampler is not None else len(ds)
    pbar = tqdm(total=total_local,
                desc=f"FID (epoch {epoch}) [rank {rank}/{world}]",
                ncols=100, disable=not is_rank0)

    # Full FID pass on each rank’s shard
    if is_dist and sampler is not None:
        sampler.set_epoch(epoch)  # deterministic sharding

    for batch in dl:
        real01 = batch["map_01"].to(device, non_blocking=True)  # [0,1]
        x_src  = batch["sat_pm1"].to(device, non_blocking=True) # [-1,1]
        fid.update(real01, real=True)
        gen_pm1 = _rk4_generate(model, x_src, steps=steps)
        gen01   = _pm1_to_01(gen_pm1)
        fid.update(gen01, real=False)
        if is_rank0:
            pbar.update(real01.size(0))

        del real01, x_src, gen_pm1, gen01

    if is_rank0:
        pbar.close()
    
    

    # Making sure all ranks finished updates before compute
    if is_dist: dist.barrier()

    fid_val = float(fid.compute().detach().cpu())  # synced across ranks

    # Log once
    if is_rank0:
        log_path   = os.path.join(out_dir, "fid_scores.txt")
        need_header = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if need_header:
                f.write("epoch\tfid\n")
            f.write(f"{epoch}\t{fid_val:.6f}\n")

    # Keeping ranks aligned before returning
    if is_dist: dist.barrier()
    fid.reset() 

    torch.cuda.synchronize()

    del dl, ds, fid
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return fid_val