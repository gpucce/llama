#!/bin/bash -x
#SBATCH --nodelist=ben[11-18]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=slurm_out/xsum_detect_%j.out
#SBATCH --mem=240G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES="0,1,2,3"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --model-dir="/home/users/giovannipuccetti/Models/65B_spread_32" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/65B_spread_32/tokenizer.model" \
    --batch-size=2 \
    --data-path="/home/users/giovannipuccetti/Data/xsum/llama65b/2023-04-30_20-27-33/xsum_test_synthetic_1000_llama65b.csv" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/llama65b/2023-04-30_20-27-33/" \
    --max-seq-len=256 \
    --do-not-cache