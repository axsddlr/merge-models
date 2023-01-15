import os
from utils.pruner import prune_checkpoint
import torch
import argparse

parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('--ckpt', type=str, default=None, help='path to model ckpt')
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
args = parser.parse_args()
ckpt = args.ckpt
device = args.device


def prune_it(checkpoint_path):
    print(f"Pruning checkpoint from path: {checkpoint_path}")
    size_initial = os.path.getsize(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pruned = prune_checkpoint(checkpoint)
    fn = f"{os.path.splitext(checkpoint_path)[0]}-pruned.ckpt"
    print(f"Saving pruned checkpoint at: {fn}")
    torch.save(pruned, fn)
    newsize = os.path.getsize(fn)
    MSG = f"New ckpt size: {newsize * 1e-9:.2f} GB. " + \
          f"Saved {(size_initial - newsize) * 1e-9:.2f} GB by removing optimizer states"
    print(MSG)


if __name__ == "__main__":
    prune_it(ckpt)
