#!/bin/bash -x
#SBATCH --nodelist=ben19
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_13b
#SBATCH --output=slurm_out/xsum_allgpt2_%j.out
#SBATCH --mem=120G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export CUDA_VISIBLE_DEVICES=0,1

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1
date="gpt2/${date}"

mkdir /home/users/giovannipuccetti/Data/xsum/${date}

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --is-hf \
    --model-dir="gpt2" \
    --batch-size=4 \
    --data-path="/home/users/giovannipuccetti/Data/xsum/gpt2/2023-05-01_11-56-28/xsum_test_synthetic_1000_gpt2.csv" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/gpt2/2023-05-01_11-56-28" \
    --max-seq-len=256