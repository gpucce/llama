#!/bin/bash -x
#SBATCH --nodelist=ben[11-18]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=fine_tuned_llama
#SBATCH --output=slurm_out/fine_tuned_job_%j.out
#SBATCH --mem=120G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

epoch=00003

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.dataset_rewriting_ita \
    --ckpt-dir="./news_fine_tune/run_04-06-2023-11-51-03/epoch_$epoch/model" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/65B_spread_32/tokenizer.model" \
    --max-batch-size=8 \
    --max-seq-len=1000 \
    --output_path="news_fine_tune"