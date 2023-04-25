#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_7b
#SBATCH --output=slurm_out/xsum_data_%j.out
#SBATCH --mem=60G

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_VISIBLE_DEVICES=0

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.xsum_data_prepare \
    --data-path="/home/users/giovannipuccetti/Data/xsum/xsum_test.csv" \
    --model-dir="/home/users/giovannipuccetti/Models/13B/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/13B/tokenizer.model" \
    --batch-size=8 \
    --max-seq-len=256 \
    --output-path="/home/users/giovannipuccetti/Data/xsum/llama13b_rephrased.csv" \
    --col-name="document" \
    --n-samples 100
    
srun --cpu_bind=v --accel-bind=gn python -u -m llama.data.dataset_lm_modification \
    --data-path="/home/users/giovannipuccetti/Data/xsum/llama13b_rephrased.csv" \
    --col-names="true_continuations:generated_continuations" \
    --output-path="/home/users/giovannipuccetti/Data/xsum/xsum_test_synthetic_1000_llama13b.csv" \
    --device-id 0 \
    --n-modifications 50 \
    --batch-size 16 \
    --modifier-model="t5-large" \
    --n-samples 100
    
    
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --model-dir="/home/users/giovannipuccetti/Models/13B" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/13B/tokenizer.model" \
    --batch-size=4 \
    --data-path="/home/users/giovannipuccetti/Data/xsum/xsum_test_synthetic_1000_llama13b.csv" \
    --output-path="data/llama13b/" \
    --max-seq-len=192