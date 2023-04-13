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
from tqdm.auto import tqdm

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
    if global_rank > 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
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
    device = torch.device(local_rank)
    # device = "cpu"
    generator.model.to(device)
    col_names = ["true_", "generated_"]
    data = pd.read_csv(
        args.data_path, index_col=0, sep="\t", on_bad_lines="skip", encoding="utf-8"
    )
    data = data.loc[data.notna().all(axis=1), :]
    generated_probs = {}
    for col_name in col_names:
        col_data = data.filter(regex=f"{col_name}.*")
        all_colls = col_data.columns
        dataset = PandasDataset(col_data)

        torch.distributed.barrier()

        dl = torch.utils.data.DataLoader(
            dataset, collate_fn=pandas_collate, batch_size=args.batch_size
        )

        torch.distributed.barrier()

        for batch in tqdm(dl):
            for key, val in batch.items():
                print(key)
                if key in generated_probs:
                    generated_probs[key] += [
                        i.tolist() for i in generator.generate_probs(val)
                    ]
                else:
                    generated_probs[key] = [
                        i.tolist() for i in generator.generate_probs(val)
                    ]

    with open(args.output_path, "w") as of:
        json.dump(generated_probs, of)


if __name__ == "__main__":
    main()
