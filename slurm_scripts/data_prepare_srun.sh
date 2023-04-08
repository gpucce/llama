#!/bin/bash -x
#SBATCH --nodelist=ben04
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=distributed_test.out
#SBATCH --mem=500G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama

cd /home/users/giovannipuccetti/Repos/llama
srun python -m llama.data.finetune_data_prepare
