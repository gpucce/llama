#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=open_clip
#SBATCH --output=distributed_test.out

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd Repos/llama/
srun --cpu_bind=v --accel-bind=gn python -u test_distributed.py
