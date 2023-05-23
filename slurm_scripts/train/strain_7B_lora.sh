#!/bin/bash -x
#SBATCH --nodelist=ben[07-08]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=slurm_out/lora_13B_train_%j.out
#SBATCH --mem-per-gpu=64G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_FILE=nccl_debug.out

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.train \
    --model-dir="/home/users/giovannipuccetti/Models/7B_spread_8/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/7B_spread_8/tokenizer.model" \
    --output-path="13b_14" \
    --batch-size=4 \
    --max-seq-len=128 \
    --epochs=20 \
    --lr=0.00001 \
    --data-path "/home/users/giovannipuccetti/Data/CHANGE-it/train/tokenized_change_it_il_giornale" \
    --max-samples=1000000 \
    --log-freq=5 \
    --steps-per-ckpt=20000 \
    --accum-freq=32 \
    --wandb-project-name="llama_7b_fine_tune" \
    --optimizer-name "adamw" \
    --warmup-ratio 0.01 \
    --skip-epoch 1 \
    --do-lora \
    --resume="latest"
