#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
python3 preprocess.py