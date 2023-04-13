#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_generation
#SBATCH --output=slurm_out/pretrained_job_%j.out
#SBATCH --mem=120G
#SBATCH --time=00:10:00

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_LAUNCH_BLOCKING=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m scripts.scripts.llama_on_slurm \
    --ckpt-dir ~/Models/65B_spread_32/ \
    --tokenizer-path ~/Models/65B_spread_32/tokenizer.model \
    --max-batch-size=8 \
    --max-seq-len=1000 \
    --output_path="pretrained_65B"
