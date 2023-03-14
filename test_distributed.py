import torch.distributed as dist
import os
from time import sleep

possible_var_names = [
    # "LOCAL_RANK", 
    # "WORLD_SIZE", 
    # "PMI_RANK", 
    # "PMI_SIZE", 
    "SLURM_NTASKS", 
    # "MPI_LOCALRANKID", 
    "SLURM_LOCALID",
    "SLURM_PROCID",
    "SLURM_NODEID",
    # "OMPI_COMM_WORLD_LOCAL_RANK"
]

out = ""
nnodes = int(os.environ.get("SLURM_NNODES", 1))
for i in possible_var_names:
    out += i
    out += "  "
    _var = int(os.environ.get(i, -1))
    if i == "SLURM_PROCID":
        _var = _var % nnodes
    out += str(_var)
    out += "  "
sleep(int(os.environ.get("SLURM_LOCALID", 1)))
print(out)
