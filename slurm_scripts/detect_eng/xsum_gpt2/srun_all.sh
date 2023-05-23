#!/bin/bash -x
#SBATCH --nodelist=ben19
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_13b
#SBATCH --output=slurm_out/xsum_allgpt2_%j.out
#SBATCH --mem=60G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export CUDA_VISIBLE_DEVICES=0,1

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1
date="gpt2/${date}"

mkdir /home/users/giovannipuccetti/Data/xsum/${date}

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.xsum_data_prepare \
    --is-hf \
    --data-path="/home/users/giovannipuccetti/Data/xsum/xsum_test.csv" \
    --model-dir="gpt2" \
    --batch-size=4 \
    --max-seq-len=256 \
    --output-path="/home/users/giovannipuccetti/Data/xsum/${date}/gpt2_rephrased.csv" \
    --col-name="document" \
    --n-samples 100 \
    --temperature=1.0 \
    --top-p=1.0

srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.dataset_lm_modification \
    --data-path="/home/users/giovannipuccetti/Data/xsum/${date}/gpt2_rephrased.csv" \
    --col-names="true_continuations:generated_continuations" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/${date}/xsum_test_synthetic_1000_gpt2.csv" \
    --n-modifications 10 \
    --batch-size 16 \
    --modifier-model="t5-3b" \
    --n-samples 100
    
    
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --is-hf \
    --model-dir="gpt2" \
    --batch-size=4 \
    --data-path="/home/users/giovannipuccetti/Data/xsum/${date}/xsum_test_synthetic_1000_gpt2.csv" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/${date}/" \
    --max-seq-len=256