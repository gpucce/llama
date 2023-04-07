# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import sys
sys.path.insert(0, "/home/users/giovannipuccetti/Repos/llama")
from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
)

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from llama import ModelArgs, Transformer, Tokenizer, load, setup_model_parallel


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    output_path: str = "pretrained",
):
    local_rank, global_rank, world_size = setup_model_parallel()
    node_id = int(os.environ.get("SLURM_NODEID", -1))

    generator = load(
        ckpt_dir,
        tokenizer_path,
        local_rank,
        global_rank,
        world_size,
        max_seq_len,
        max_batch_size,
    )
    torch.distributed.barrier()
    if local_rank != 0:
        time.sleep(5)
    generator.model.to(torch.device(local_rank))
    t = torch.cuda.get_device_properties(local_rank).total_memory
    r = torch.cuda.memory_reserved(local_rank)
    a = torch.cuda.memory_allocated(local_rank)
    f = r - a  # free inside reserved
    print(
        "DEVICE",
        local_rank,
        global_rank,
        torch.cuda.current_device(),
        torch.cuda.device_count(),
        node_id,
    )
    print(f"Total memory {t}, Reserved {r}, Allocated {a}, Free {f}")

    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    if global_rank == 0:
        output_path = Path("output") / output_path
        output_path.mkdir(exist_ok=True, parents=True)
        print(results)
        with open(output_path / "test_result.txt", "w") as tf:
            for result in results:
                tf.write(result)
                tf.write("\n==================================\n")


if __name__ == "__main__":
    
    ingredients = [
        "la cipolla",
        "il pollo",
        "il tacchino",
        "la carne",
        "i porri",
        "lo zucchero",
        "7 ingredienti diversi",
        "il cacciucco",
    ]

    prompts = [f"Questa è una ricetta con {i} in italiano:" for i in ingredients]
    
    fire.Fire(main)
