import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-traindir", type=str, default="",
    help="training directory from syncache"
)
parser.add_argument(
    "-valdir", type=str, default="",
    help="validation directory from syncache"
)
args = parser.parse_args()
traindir = args.traindir
valdir = args.valdir

torch.cuda.empty_cache()