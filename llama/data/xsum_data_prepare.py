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

from ..utils import custom_parse_args
from ..data_utils import process_spaces


def main():
    args = custom_parse_args()

    ckpt_dir = args.model_dir
    tokenizer_parh = args.tokenizer_path
    temperature = args.temperature
    top_p = args.top_p
    batch_size = args.batch_size
    output_path = args.output_path
    data_path = args.data_path
    n_samples = args.n_samples
    col_name = args.col_name
    max_seq_len = args.max_seq_len
    tokenizer_path = args.tokenizer_path

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
        batch_size,
    )

    # torch.distributed.barrier()
    generator.model.to(torch.device(local_rank))

    ds = pd.read_csv(data_path, index_col=0)
    random.seed(42)
    if n_samples >= 1:
        idxs = random.sample(range(ds.shape[0]), n_samples)
        ds = ds.iloc[idxs, :]
    dataloader = torch.utils.data.DataLoader(
        ds.loc[:, col_name].to_list(), batch_size=batch_size
    )
    all_prompts = []
    true_continuations = []
    generated_continuations = []
    start = time.time()
    prompt_len = 30
    max_gen_len = 300
    actual_gen_len = 150
    start = time.time()
    for idx, prompts in enumerate(dataloader):
        prompts = [process_spaces(prompt).split(" ") for prompt in prompts]
        true_continuations += [
            " ".join(prompt[prompt_len : prompt_len + actual_gen_len])
            for prompt in prompts
        ]
        prompts = [" ".join(prompt[:prompt_len]) for prompt in prompts]

        all_prompts += prompts

        batch_generated_continuations = generator.generate(
            prompts,
            prompt_tokens=batch_prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        generated_continuations += [
            " ".join(continuation.split(" ")[:actual_gen_len])
            for continuation in batch_generated_continuations
        ]

        new_start = time.time()
        print(f"Step {idx} done in {new_start - start} secs.")
        start = new_start

    ds["prompts"] = all_prompts
    ds["true_continuations"] = true_continuations
    ds["generated_continuations"] = generated_continuations
    data_source = Path(data_path).stem
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    ds = ds.loc[ds.notna().all(axis=1), :]
    ds.to_csv(output_path, sep="\t")
    elapsed = time.time() - start

    print(f"The process took: {elapsed} seconds.")


if __name__ == "__main__":
    main()
