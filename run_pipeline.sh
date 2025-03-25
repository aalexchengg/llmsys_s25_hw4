#!/bin/bash
#SBATCH --job-name=cpu_test
#SBATCH --output=out/cpu_test.out
#SBATCH --error=out/cpu_test.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40:2

# Your job commands go here

# Load in cuda
source /etc/profile.d/modules.sh
module load cuda-12.4
nvcc --version
source ~/.bashrc

# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate hw4

nvidia-smi

# Problem 2.1
python -m pytest -l -v -k "a4_2_1"

# Problem 2.2
python -m pytest -l -v -k "a4_2_2"

# Problem 2.3
echo "Running model parallel...."
python project/run_pipeline.py --model_parallel_mode='model_parallel'
echo "Finished running model parallel."

echo "Running pipeline parallel..."
python project/run_pipeline.py --model_parallel_mode='pipeline_parallel'
echo "Finished running pipeline parallel"