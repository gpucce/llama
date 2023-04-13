#!/bin/bash -x
#SBATCH --nodelist=ben[11-18]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=fine_tuned_llama
#SBATCH --output=slurm_out/fine_tuned_job_%j.out
#SBATCH --mem=120G
#SBATCH --time=00:10:00

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

epoch=00001

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.fine_tuned_llama_on_slurm \
    --ckpt-dir ./test_output/run_03-28-2023-11-57-35/epoch_$epoch/model \
    --tokenizer-path ~/Models/65B_spread_32/tokenizer.model \
    --max-batch-size=8 \
    --max-seq-len=1000 \
    --output_path fine_tuned_epoch_$epoch
