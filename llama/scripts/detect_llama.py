import sys
from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

import pandas as pd

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
)

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from llama import ModelArgs, Transformer, Tokenizer, load, setup_model_parallel
from llama.utils import custom_parse_args
from llama.data_utils import PandasDataset, pandas_collate


def main():
    args = custom_parse_args()
    local_rank, global_rank, world_size = setup_model_parallel()
    ckpt_dir = args.model_dir
    generator = load(
        ckpt_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
    )

    generator.model.to(torch.device(local_rank))
    col_names = ["full_text"] + [f"synthetic_{i}" for i in range(10)]
    data = pd.read_csv(args.data_path, index_col=0).loc[:, col_names]
    dataset = PandasDataset(data.iloc[:10, :])
    dl = torch.utils.data.DataLoader(
        dataset, collate_fn=pandas_collate, batch_size=args.batch_size
    )
    generated_probs = {i: [] for i in col_names}

    torch.distributed.barrier()
    for batch in dl:
        for key, val in batch.items():
            generated_probs[key] += [i.tolist() for i in generator.generate_probs(val)]

    with open(args.output_path, "w") as of:
        json.dump(generated_probs, of)


if __name__ == "__main__":
    main()
