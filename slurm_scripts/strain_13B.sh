#!/bin/bash -x
#SBATCH --nodelist=ben03,ben04
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=distributed_test.out
#SBATCH --mem-per-gpu=32G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_LAUNCH_BLOCKING=1


export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=nccl_debug.out

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.train \
    --model-dir="/home/users/giovannipuccetti/Models/13B_spread_8/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/13B_spread_8/tokenizer.model" \
    --batch-size=12 \
    --max-seq-len=128 \
    --epochs=3 \
    --lr=0.0005 \
    --max-samples 1000 \
    --log-freq 5
