#!/bin/bash -x
#SBATCH --nodelist=ben03,ben04,ben05,ben06,ben07,ben08,ben09,ben10
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=distributed_test.out
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
    --model-dir="/home/users/giovannipuccetti/Models/65B_spread_32/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/65B_spread_32/tokenizer.model" \
    --batch-size=2 \
    --max-seq-len=64 \
    --epochs=3 \
    --lr=0.0005 \
    --max-samples 100000 \
    --log-freq 5 \
    --steps-per-epoch 50 \
    --resume "latest"
    
