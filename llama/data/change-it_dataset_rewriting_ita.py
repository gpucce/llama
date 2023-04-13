# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import sys
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
    file_name = "/home/users/giovannipuccetti/Data/CHANGE-it/test/change-it.ilgiornale.test_1000.csv"
    ds = pd.read_csv(file_name, index_col=0)
    random.seed(42)
    idxs = random.sample(range(ds.shape[0]), 100)
    ds = ds.iloc[idxs, :]
    dataloader = torch.utils.data.DataLoader(
        ds.full_text.to_list(), batch_size=max_batch_size
    )
    all_prompts = []
    true_continuations = []
    generated_continuations = []
    start = time.time()
    prompt_len = 64
    max_gen_len = 128
    start = time.time()
    for idx, prompts in enumerate(dataloader):
        prompt_tokens = [
            generator.tokenizer.encode(prompt, bos=True, eos=False)
            for prompt in prompts
        ]
        batch_true_continuations = [
            i[prompt_len : prompt_len + max_gen_len] for i in prompt_tokens
        ]
        true_continuations += [
            generator.tokenizer.decode(continuation)
            for continuation in batch_true_continuations
        ]

        batch_prompt_tokens = [i[:prompt_len] for i in prompt_tokens]
        batch_prompts = [
            generator.tokenizer.decode(prompt_token)
            for prompt_token in batch_prompt_tokens
        ]
        all_prompts += [
            generator.tokenizer.decode(prompt) for prompt in batch_prompt_tokens
        ]

        batch_generated_continuations = generator.generate(
            prompts,
            prompt_tokens=batch_prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        generated_continuations += [
            cont[len(prompt) :]
            for prompt, cont in zip(all_prompts, batch_generated_continuations)
        ]
        new_start = time.time()
        print(f"Step {idx} done in {new_start - start} secs.")
        start = new_start

    ds["prompts"] = all_prompts
    ds["true_continuations"] = true_continuations
    ds["generated_continuations"] = generated_continuations
    data_source = Path(file_name).stem
    output_path = Path("data") / output_path
    output_path.mkdir(exist_ok=True, parents=True)
    ds = ds.loc[ds.notna().all(axis=1), :]
    ds.to_csv(output_path / f"{data_source}_rephrased_{epoch}.csv", sep="\t")
    elapsed = time.time() - start

    print(f"The process took: {elapsed} seconds.")


if __name__ == "__main__":
    fire.Fire(main)
