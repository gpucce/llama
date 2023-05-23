#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_out/wiki_prepare_%j.out
#SBATCH --mem=60G


eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate transformers
cd /home/users/giovannipuccetti/Repos/llama

srun python -m llama.data.wikipedia_data_prepare