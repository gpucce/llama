#!/bin/bash -x
#SBATCH --nodelist=ben[09-10]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama_7b
#SBATCH --output=slurm_out/changeit_generate_%j.out
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
    --data-path="/home/users/giovannipuccetti/Data/CHANGE-it/test/change-it.ilgiornale.test_1000.csv" \
    --model-dir="./runs/13b_14/run_05-08-2023-07-52-54/ckpt_00008/model/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/13B_spread_8/tokenizer.model" \
    --batch-size=8 \
    --max-seq-len=256 \
    --output-path="/home/users/giovannipuccetti/Data/CHANGE-it/llam13b_finetune/test_output_00008.csv" \
    --col-name="full_text" \
    --n-samples 10 \
    --do-lora \
    --lora-r 8