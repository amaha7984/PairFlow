import os, random, torch
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from diffusers.models import AutoencoderKL
from src.utils.sampling import (
    _rk4_generate_latent_gaussian_cosine,
    _pm1_to_01
)



class EvalPairDataset(Dataset):
    def __init__(self, sat_dir, map_dir, size=256):
        self.sat_dir = sat_dir
        self.map_dir = map_dir
        self.to_src  = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Lambda(lambda x: x*2-1)])
        self.to_real = T.Compose([T.Resize((size, size)), T.ToTensor()])
        a_names = sorted([n for n in os.listdir(sat_dir) if n.endswith((".jpg", ".png")) and "_A." in n])
        self.pairs = []
        for n in a_names:
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
def eval_fid_i2i_gaussian_cosine(
    model, device,
    sat_dir, map_dir,
    out_dir, epoch, steps=50, batch_size=16, num_workers=4,
    save_samples=10, scale_factor=0.18215
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    is_dist  = dist.is_available() and dist.is_initialized()
    rank     = dist.get_rank() if is_dist else 0
    world    = dist.get_world_size() if is_dist else 1
    is_rank0 = (rank == 0)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval(); vae.train = False
    for p in vae.parameters(): p.requires_grad = False

    ds = EvalPairDataset(sat_dir, map_dir, size=256)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if is_dist else None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler,
                    num_workers=num_workers, pin_memory=True, drop_last=False)

    fid = FrechetInceptionDistance(normalize=True).to(device)

    # Save a few samples
    if is_rank0 and save_samples > 0:
        sample_dir = os.path.join(out_dir, f"epoch_{epoch}_samples_gfm_cosine")
        os.makedirs(sample_dir, exist_ok=True)
        import random
        idxs = random.sample(range(len(ds)), min(save_samples, len(ds)))
        for i, idx in enumerate(idxs):
            b  = ds[idx]
            x0 = b["sat_pm1"].unsqueeze(0).to(device)
            x1 = b["map_01"].unsqueeze(0).to(device)
            z_src = vae.encode(x0).latent_dist.sample() * scale_factor
            z_T0  = _rk4_generate_latent_gaussian_cosine(model, z_src, steps=steps)
            gen_pm1 = vae.decode(z_T0 / scale_factor).sample
            gen01   = _pm1_to_01(gen_pm1)
            vis = torch.cat([_pm1_to_01(x0), x1, gen01], dim=0)
            save_image(vis, os.path.join(sample_dir, f"{i:02d}.png"), nrow=3)
            del x0, x1, z_src, z_T0, gen_pm1, gen01, vis

    total_local = len(sampler) if sampler is not None else len(ds)
    pbar = tqdm(total=total_local,
                desc=f"FID-GFM-COS (epoch {epoch}) [rank {rank}/{world}]",
                ncols=100, disable=not is_rank0)

    if is_dist and sampler is not None:
        sampler.set_epoch(epoch)

    for batch in dl:
        real01 = batch["map_01"].to(device, non_blocking=True)
        x_src  = batch["sat_pm1"].to(device, non_blocking=True)
        fid.update(real01, real=True)

        z_src = vae.encode(x_src).latent_dist.sample() * scale_factor
        z_T0  = _rk4_generate_latent_gaussian_cosine(model, z_src, steps=steps)
        gen_pm1 = vae.decode(z_T0 / scale_factor).sample
        gen01   = _pm1_to_01(gen_pm1)
        fid.update(gen01, real=False)

        if is_rank0: pbar.update(real01.size(0))
        del real01, x_src, z_src, z_T0, gen_pm1, gen01

    if is_rank0: pbar.close()
    if is_dist: dist.barrier()
    fid_val = float(fid.compute().detach().cpu())
    if is_rank0:
        log_path = os.path.join(out_dir, "fid_scores_gfm_cosine.txt")
        need_header = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if need_header: f.write("epoch\tfid\n")
            f.write(f"{epoch}\t{fid_val:.6f}\n")
    if is_dist: dist.barrier()
    fid.reset()
    torch.cuda.synchronize()
    return fid_val
