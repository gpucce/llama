#!/bin/bash -x
#SBATCH --nodelist=ben09
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_13b
#SBATCH --output=slurm_out/xsum_hfall13b_%j.out
#SBATCH --mem=120G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export CUDA_VISIBLE_DEVICES=0,1,2,3

printf -v date '%(%Y-%m-%d_%H-%M-%S)T' -1
date="hf13b/${date}"

mkdir /home/users/giovannipuccetti/Data/xsum/${date}

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.xsum_data_prepare \
    --is-hf \
    --data-path="/home/users/giovannipuccetti/Data/xsum/xsum_test.csv" \
    --model-dir="/home/users/giovannipuccetti/HFModels/HF13B" \
    --batch-size=8 \
    --max-seq-len=256 \
    --output-path="/home/users/giovannipuccetti/Data/xsum/${date}/hf13b_rephrased.csv" \
    --col-name="document" \
    --n-samples=100 \
    --temperature=0.8

srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.dataset_lm_modification \
    --data-path="/home/users/giovannipuccetti/Data/xsum/${date}/hf13b_rephrased.csv" \
    --col-names="true_continuations:generated_continuations" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/${date}/xsum_test_synthetic_1000_hf13b.csv" \
    --n-modifications 10 \
    --batch-size 8 \
    --modifier-model="t5-11b" \
    --n-samples 100
    
    
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --is-hf \
    --model-dir="/home/users/giovannipuccetti/HFModels/HF13B" \
    --batch-size=2 \
    --data-path="/home/users/giovannipuccetti/Data/xsum/${date}/xsum_test_synthetic_1000_hf13b.csv" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/${date}/" \
    --max-seq-len=256 \
    --temperature=0.8