#!/bin/bash -x
#SBATCH --nodelist=ben[09-10]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=fine_tuned_llama
#SBATCH --output=slurm_out/change-it_data_%j.out
#SBATCH --mem=120G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1
date="llama7b_fine_tuned/${date}"

mkdir /home/users/giovannipuccetti/Data/xsum/${date}

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.change-it_dataset_rewriting_ita \
    --model-dir="/home/users/giovannipuccetti/Models/7B_spread_8/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/7B/tokenizer.model" \
    --batch-size=32 \
    --max-seq-len=1000 \
    --output-path="/home/users/giovannipuccetti/Data/CHANGE-it/${date}" \
    --temperature=0.8 \
    --do-lora \
    --lora-r 8