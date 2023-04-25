#!/bin/bash -x
#SBATCH --nodelist=ben[11-18]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=slurm_out/xsum_detect_%j.out
#SBATCH --mem=120G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=0,1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --model-dir="/home/users/giovannipuccetti/Models/65B_spread_32" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/65B/tokenizer.model" \
    --batch-size=16 \
    --data-path="/home/users/giovannipuccetti/Data/xsum/xsum_test_synthetic_1000_llama65b.csv" \
    --output-path="data/llama65b/" \
    --max-seq-len=196