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
    """
    It loads a checkpoint, prunes it, and saves it

    :param checkpoint_path: The path to the checkpoint file
    """
    print(f"Pruning checkpoint from path: {checkpoint_path}")
    size_initial = os.path.getsize(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pruned = prune_checkpoint(checkpoint)
    base_file = os.path.basename(checkpoint_path)
    base_file_name, _ = os.path.splitext(base_file)
    fn = f"{base_file_name}-pruned.ckpt"
    print(f"Saving pruned checkpoint at: {fn}")
    torch.save(pruned, fn)
    newsize = os.path.getsize(fn)
    if newsize >= size_initial:
        os.remove(fn)  # deleting the new checkpoint because it is not smaller than the original
        print("No changes were made, original checkpoint is kept")
    else:
        MSG = f"New ckpt size: {newsize * 1e-9:.2f} GB. " + \
              f"Saved {(size_initial - newsize) * 1e-9:.2f} GB by removing optimizer states"
        print(MSG)


prune_it(ckpt)
