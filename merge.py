import argparse
import os
import time

import torch
from tqdm import tqdm

start_time = time.time()


def save_model(theta_0, output_file):
    """
    > This function takes in a parameter, theta_0, and saves it to a file

    :param theta_0: the initial value of theta
    :param output_file: the name of the file where you want to save the model
    """
    torch.save({"state_dict": theta_0}, output_file)
    print("Saving...")
    print("Done!")
    if device == "cuda":
        print(torch.cuda.memory_allocated())


parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_0", type=lambda x: x.strip("\"'"), help="Path to model 0")
parser.add_argument("model_1", type=lambda x: x.strip("\"'"), help="Path to model 1")
parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5", default=0.5, required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--without_vae", action="store_true", help="Do not merge VAE", required=False)

args = parser.parse_args()

device = args.device
# Loading the model and the state_dict.
model_0 = torch.load(args.model_0, map_location=device)
model_1 = torch.load(args.model_1, map_location=device)
# If the model is a state_dict, then it will be assigned to theta_0 or theta_1. If not, it will be assigned to
# theta_0 or theta_1.
try:
    theta_0 = model_0["state_dict"]
except KeyError:
    print("Model 0 does not have a state_dict key, assuming it is a state_dict")
    theta_0 = model_0
try:
    theta_1 = model_1["state_dict"]
except KeyError:
    print("Model 1 does not have a state_dict key, assuming it is a state_dict")
    theta_1 = model_1
alpha = args.alpha

output_file = f'{args.output}-{str(alpha)[2:] + "0"}.ckpt'

# check if output file already exists, ask to overwrite
if os.path.isfile(output_file):
    print("Output file already exists. Overwrite? (y/n)")
    while True:
        overwrite = input()
        if overwrite == "y":
            break
        elif overwrite == "n":
            print("Exiting...")
            exit()
        else:
            print("Please enter y or n")

# Merging the two models.
for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
    # clear the GPU memory
    torch.cuda.empty_cache()
    # skip VAE model parameters to get better results(tested for anime models)
    # for anime model，with merging VAE model, the result will be worse (dark and blurry)
    if args.without_vae and "first_stage_model" in key:
        continue

    if "model" in key and key in theta_1:
        theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]

for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
    # clear the GPU memory
    torch.cuda.empty_cache()
    if "model" in key and key not in theta_0:
        theta_0[key] = theta_1[key]

save_model(theta_0, output_file)
end_time = time.time()
# Checking if the device is cpu, if it is, then it will print the total time.
if device == "cpu":
    print(f"Total time: {end_time - start_time:.2f} seconds")
