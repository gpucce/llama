#!/bin/bash -x
#SBATCH --nodelist=ben09
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_13b
#SBATCH --output=slurm_out/xsum_hfall13b_%j.out
#SBATCH --mem=120G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
date="hf13b/2023-05-03_11-29-50/"

export CUDA_VISIBLE_DEVICES=0,1,2,3


    
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --is-hf \
    --model-dir="/home/users/giovannipuccetti/HFModels/HF13B" \
    --batch-size=2 \
    --data-path="/home/users/giovannipuccetti/Data/xsum/${date}/xsum_test_synthetic_1000_hf13b.csv" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/${date}/" \
    --max-seq-len=256 \
    --temperature=0.8