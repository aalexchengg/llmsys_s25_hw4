#!/bin/bash
#SBATCH --job-name=dp
#SBATCH --output=out/dp.out
#SBATCH --error=out/dp.err
#SBATCH --partition=general
#SBATCH --gpus=2

# Your job commands go here

# Load in cuda
source /etc/profile.d/modules.sh
module load cuda-12.4
nvcc --version
source ~/.bashrc

# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate hw4

# Problem 1.1
python -m pytest -l -v -k "a4_1_1"

# Problem 1.2
python project/run_data_parallel.py --pytest True --n_epochs 1
python -m pytest -l -v -k "a4_1_2"

# Problem 1.3
# single node
python project/run_data_parallel.py --world_size 1 --batch_size 64
# double nodes
python project/run_data_parallel.py --world_size 2 --batch_size 128