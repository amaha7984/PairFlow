import os
import argparse
from src.engine import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate and save samples every N epochs")
    parser.add_argument("--no_fid", action="store_true",
                        help="Skip FID evaluation during training (still saves checkpoints).")
    parser.add_argument("--scale_factor", type=float, default=0.18215)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)
