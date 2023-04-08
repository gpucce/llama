#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=slurm_out/13B_distributed_test.out
#SBATCH --mem-per-gpu=32G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.train \
    --model-dir="/home/users/giovannipuccetti/Models/13B_spread_8/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/13B_spread_8/tokenizer.model" \
    --batch-size=12 \
    --max-seq-len=128 \
    --output-path "__test_output__" \
    --epochs=3 \
    --lr=0.00005 \
    --max-samples 1000 \
    --log-freq 1 \
    --deepspeed_config "ds_config.json"
