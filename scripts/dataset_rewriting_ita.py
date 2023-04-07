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
import pandas as pd
import random

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
    output_path: str = "fine_tuned",
):
    local_rank, global_rank, world_size = setup_model_parallel()
    if global_rank > 0:
        sys.stdout = open(os.devnull, "w")
    #     sys.stderr = open(os.devnull, "w")

    generator = load(
        ckpt_dir,
        tokenizer_path,
        local_rank,
        global_rank,
        world_size,
        max_seq_len,
        max_batch_size,
    )

    # torch.distributed.barrier()
    generator.model.to(torch.device(local_rank))

    epoch = [i for i in ckpt_dir.split("/") if "epoch" in i][0]
    file_name = "./data/complexity_ds_it.csv"
    ds = pd.read_csv(file_name) # .iloc[-20:, :]
    random.seed(42)
    idxs = random.sample(range(ds.shape[0]), 20)
    ds = ds.iloc[idxs, :].reset_index(drop=True)
    dataloader = torch.utils.data.DataLoader(ds.SENTENCE.to_list(), batch_size=8)
    rephrased = []
    start = time.time()
    print("Here we are")
    for prompts in dataloader:
        # prompts = ['"' + i + '"' + " altrimenti, equivalentemente uno potrebbe scrivere: " for i in prompts]
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p
        )
        print(results)
        rephrased += results
    ds["LM_PHRASED"] = rephrased
    data_source = Path(file_name).stem
    output_path = Path("data") / output_path
    output_path.mkdir(exist_ok=True, parents=True)
    ds.to_csv(output_path / f"{data_source}_rephrased_{epoch}.csv")
    elapsed = time.time() - start
    
    print(f"The process took: {elapsed} seconds.")


if __name__ == "__main__":
    fire.Fire(main)
