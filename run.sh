#!/bin/bash
#SBATCH --output output.err     # output log file
#SBATCH -e error.err            # error log file
#SBATCH --mem=50G 
#SBATCH --gres=gpu:RTX2080:1
#SBATCH -p scavenger-gpu    
# execute my file
python infer_catalyst.py