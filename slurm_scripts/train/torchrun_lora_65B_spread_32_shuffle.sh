#!/bin/bash

#SBATCH --nodelist=ben[11-18]
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm_out/lora_65B_train_%j.out

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama

export MASTER_PORT=29002
export RANK=$SLURM_PROCID

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(hostname --ip-address)
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id 456 \
    --rdzv_backend="c10d" \
    --rdzv_endpoint="$head_node_ip:$MASTER_PORT" \
    -m llama.train \
    --model-dir="/home/users/giovannipuccetti/Models/7B_spread_8/" \
    --tokenizer-path="/home/users/giovannipuccetti/Models/7B_spread_8/tokenizer.model" \
    --output-path "test_custom_optim" \
    --batch-size=4 \
    --max-seq-len=128 \
    --epochs=3 \
    --lr=0.00001 \
    --data-path "/home/users/giovannipuccetti/Data/tokenized_change_it_seq_len_128.jsonl" \
    --max-samples 1000000 \
    --log-freq 5 \
    --steps-per-epoch 10000 \
    --accum-freq 64 \
    --lora-r 32 \
    --do-lora \
    --is-torchrun
