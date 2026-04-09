#!/bin/bash
#SBATCH --job-name=llada_finetune
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=1:30:00

#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

source .venv/bin/activate
python fine_tuning/finetune.py  -c configs/gpu-100k.yml -p datasets/100k_gutenberg.train -t localShuffle5  > output.log 2>&1