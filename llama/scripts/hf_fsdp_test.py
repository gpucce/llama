import os
import functools
import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from fairscale.nn import FullyShardedDataParallel as FSDP
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from fairscale.nn.wrap import wrap, enable_wrap, auto_wrap

# from torch.distributed.fsdp import (
#     FullyShardedDataParallel as FSDP,
#     MixedPrecision,
#     BackwardPrefetch,
#     ShardingStrategy,
#     FullStateDictConfig,
#     StateDictType,
# )

# fpSixteen = MixedPrecision(
#     param_dtype=torch.float16,
#     # Gradient communication precision.
#     reduce_dtype=torch.float16,
#     # Buffer precision.
#     buffer_dtype=torch.float16,
# )

def setup_model_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    global_rank = local_rank
    os.environ["RANK"] = os.environ["LOCAL_RANK"]

    torch.distributed.init_process_group("nccl")  # , rank=local_rank, world_size=world_size)
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, global_rank, world_size


local_rank, global_rank, world_size = setup_model_parallel()

IS_MASTER = local_rank == 0

fsdp_params = dict(wrapper_cls=FSDP, mixed_precision=True, flatten_parameters=True)

device = torch.device(local_rank)

if IS_MASTER:
    print("init")
    model = LlamaForCausalLM.from_pretrained("/home/users/giovannipuccetti/HFModels/HF13B")
    # with enable_wrap(**fsdp_params):
    #     model = LlamaForCausalLM.from_pretrained("/home/users/giovannipuccetti/HFModels/HF13B")
    #     auto_wrap(model)
    print("half")
    print("init fspd")
    model = FSDP(model)

torch.distributed.barrier()
if IS_MASTER:
    print("inference")

print(
    model(torch.tensor([1,2,3]).to(device))
)