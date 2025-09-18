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
