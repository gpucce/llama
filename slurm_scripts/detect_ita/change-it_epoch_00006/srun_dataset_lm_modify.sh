#!/bin/bash -x
#SBATCH --nodelist=ben04
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=slurm_out/lm_modify_%j.out
#SBATCH --mem-per-gpu=16G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=0,1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

epoch=00006

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.dataset_lm_modification \
    --data-path="./data/news_fine_tune/change-it.ilgiornale.test_1000_rephrased_epoch_$epoch.csv" \
    --col-names="true_continuations:generated_continuations" \
    --output-path="/home/users/giovannipuccetti/Data/CHANGE-it/test/change-it.ilgiornale.test_1000_news_epoch_${epoch}_synthetic_bis.csv" \
    --device-id 0 \
    --n-modifications 100