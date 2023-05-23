#!/bin/bash -x
#SBATCH --nodelist=ben[11-18]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_65b
#SBATCH --output=slurm_out/changeit_all65_pre_%j.out
#SBATCH --mem=60G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=0,1,2,3

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1
date="llama65b/${date}"

mkdir /home/users/giovannipuccetti/Data/xsum/${date}

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.xsum_data_prepare \
    --data-path="/home/users/giovannipuccetti/Data/CHANGE-it/test/change-it.ilgiornale.test_1000.csv" \
    --model-dir="/home/users/giovannipuccetti/Models/65B_spread_32" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/65B_spread_32/tokenizer.model" \
    --batch-size=16 \
    --max-seq-len=256 \
    --output-path="/home/users/giovannipuccetti/Data/CHANGE-it/${date}/llama65b_rephrased.csv" \
    --col-name="full_text" \
    --n-samples 100 \
    --temperature="0.8" \
    --top-p="1.0"

srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.dataset_lm_modification \
    --data-path="/home/users/giovannipuccetti/Data/CHANGE-it/${date}/llama65b_rephrased.csv" \
    --col-names="true_continuations:generated_continuations" \
    --output-path="/home/users/giovannipuccetti/Data/CHANGE-it/${date}/changeit_test_synthetic_1000_llama65b.csv" \
    --n-modifications 10 \
    --batch-size 16 \
    --modifier-model="t5-large" \
    --n-samples 100
    
    
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --model-dir="/home/users/giovannipuccetti/Models/65B_spread_32" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/65B_spread_32/tokenizer.model" \
    --batch-size=4 \
    --data-path="/home/users/giovannipuccetti/Data/CHANGE-it/${date}/changeit_test_synthetic_1000_llama65b.csv" \
    --output-path="/home/users/giovannipuccetti/Data/CHANGE-it/${date}/" \
    --max-seq-len=256