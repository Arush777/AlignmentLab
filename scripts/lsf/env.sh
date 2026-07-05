#!/bin/bash
# Keep in sync with configs/cluster.yaml
export ALAB_SCRATCH="/u/arushh/alignmentlab_scratch"
export HF_HOME="/u/arushh/.cache/huggingface"
export WANDB_PROJECT="alignmentlab"
# Remove this line after `wandb login` (then `wandb sync` any offline runs):
export WANDB_MODE="offline"
