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

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int, int]:
    local_rank = int(os.environ.get("SLURM_LOCALID", -1))
    global_rank = int(os.environ.get("SLURM_PROCID", -1))
    world_size = int(os.environ.get("SLURM_NTASKS", -1))
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    torch.distributed.init_process_group(
        "nccl", world_size=world_size, rank=global_rank, init_method="env://"
    )
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, global_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    global_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[global_rank]
    time.sleep(5 * local_rank)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        do_cache=True,
        **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    
    embs = model.tok_embeddings.weight.data
    emb_height, emb_width = embs.shape
    new_embs = torch.ones(emb_height + 1, emb_width)
    model.tok_embeddings.weight.data = new_embs
    
    model.load_state_dict(checkpoint, strict=False)
    model_size = 0
    for i in model.parameters():
        model_size += i.nelement() * i.element_size()
    torch.set_default_tensor_type(torch.FloatTensor)
    print("MODEL_SIZE", model_size)
    print(
        torch.cuda.device_count(),
        os.environ.get("SLURM_NODEID", -1),
        "MP rank",
        get_model_parallel_rank(),
        # "MP src rank",
        # get_model_parallel_src_rank(),
        "rank",
        local_rank,
        "global_rank",
        global_rank,
    )

    generator = LLaMA(model, tokenizer)
    return generator


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

    torch.distributed.barrier()
    generator.model.to(torch.device(local_rank))

    epoch = [i for i in ckpt_dir.split("/") if "epoch" in i][0]
    file_name = "./data/complexity_ds_it.csv"
    ds = pd.read_csv(file_name).iloc[-20:, :]
    # idxs = random.sample(range(ds.shape[0]), 20)
    # ds = ds.iloc[idxs, :].reset_index(drop=True)
    dataloader = torch.utils.data.DataLoader(ds.SENTENCE.to_list(), batch_size=8)
    rephrased = []
    start = time.time()
    print("Here we are")
    for prompts in dataloader:
        prompts = ['"' + i + '"' + " questo passaggio può essere riscritto in questo modo: " for i in prompts]
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p
        )
        print(results)
        rephrased += results
    ds["LM_PHRASED"] = rephrased
    data_source = Path(file_name).stem
    ds.to_csv(f"data/{data_source}_rephrased_{epoch}.csv")
    elapsed = time.time() - start
    
    print(f"The process took: {elapsed} seconds.")


if __name__ == "__main__":
    fire.Fire(main)
