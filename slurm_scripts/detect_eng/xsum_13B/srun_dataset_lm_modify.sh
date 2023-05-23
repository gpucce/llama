#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=slurm_out/xsum_lm_modify_%j.out
#SBATCH --mem-per-gpu=48G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=0

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.dataset_lm_modification \
    --data-path="/home/users/giovannipuccetti/Data/xsum/llama13b_rephrased.csv" \
    --col-names="true_continuations:generated_continuations" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/xsum_test_synthetic_1000_llama13b.csv" \
    --n-modifications 100 \
    --batch-size 4 \
    --modifier-model="t5-11b" \
    --n-samples 100