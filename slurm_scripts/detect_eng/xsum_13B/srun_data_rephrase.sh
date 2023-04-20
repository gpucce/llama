#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_7b
#SBATCH --output=slurm_out/xsum_data_%j.out
#SBATCH --mem=60G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=0

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.xsum_data_prepare \
    --data-path="/home/users/giovannipuccetti/Data/xsum/xsum_test.csv" \
    --model-dir="/home/users/giovannipuccetti/Models/13B/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/13B/tokenizer.model" \
    --batch-size=8 \
    --max-seq-len=256 \
    --output-path="llama7b" \
    --col-name="document" \
    --n-samples 1000