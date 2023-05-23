#!/bin/bash -x
#SBATCH --nodelist=ben10
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=slurm_out/lm_modify_%j.out
#SBATCH --mem-per-gpu=16G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=0,1,3,4

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.dataset_lm_modification \
    --data-path="/home/users/giovannipuccetti/Data/multifake/longer_chunked_df.csv" \
    --col-names="trainslations" \
    --modifier-model="t5-3b" \
    --output-path="/home/users/giovannipuccetti/Data/multifake/joint_tasks_extended_synthetic.tsv" \
    --batch-size 16 \
    --n-modifications 3 \
    --keep-empty-lines
