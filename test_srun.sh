#!/bin/bash -x
#SBATCH --nodelist=ben03,ben04,ben05,ben06
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llama
#SBATCH --output=distributed_test.out
#SBATCH --time=00:10:00

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate llama
export MASTER_PORT=12804
export RANK=$SLURM_PROCID
export CUDA_LAUNCH_BLOCKING=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /home/users/giovannipuccetti/Repos/llama
srun --cpu_bind=v --accel-bind=gn python -u llama_on_slurm.py --ckpt-dir ~/Models/65B_spread_16 --tokenizer-path ~/Models/65B_spread/tokenizer.model --max-batch-size=2 --max-seq-len=100
