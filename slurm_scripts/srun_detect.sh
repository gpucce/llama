#!/bin/bash -x
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=distributed_test.out
#SBATCH --mem=120G
#SBATCH --time=00:10:00

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_LAUNCH_BLOCKING=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u -m llama.scripts.detect_llama \
    --model-dir="llama/news_fine_tune/run_04-06-2023-11-51-03/epoch_00006" \
    --tokenizer-path ~/Models/65B/tokenizer.model \
    --max-batch-size=3 \
    --data-path="~/Data/CHANGE-it/test/change-it.ilgiornale.test_1000_synthetic.csv" \
    --output_path="data/test_detection.json"