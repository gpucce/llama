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
    ckpt_dir = args.model_dir
    output_path = Path(args.output_path)
    if global_rank == 0:
        output_path.mkdir(exist_ok=True, parents=True)

    device = torch.device(local_rank)
    # device = "cpu"

    generator = load(
        ckpt_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
    )

    generator.model.to(device)
    generator.model.to(torch.float32)
    col_groups = ["true_", "generated_"]
    data = pd.read_csv(
        args.data_path,
        index_col=0,
        sep="\t",
        on_bad_lines="skip",
        encoding="utf-8",
        lineterminator="\n",
    )
    data = data.filter(regex="true|generated|prompts.*")
    data = data.loc[data.notna().all(axis=1), :]
    if args.n_samples >= 1:
        data = data.iloc[: args.n_samples, :]

    dataset = PandasDataset(data.reset_index())

    dl = torch.utils.data.DataLoader(
        dataset, collate_fn=pandas_collate, batch_size=args.batch_size
    )

    for idx, batch in enumerate(dl):
        generated_probs = {}
        batch_save_path = output_path / f"test_detection_batch_{idx}.csv"
        print(f"Start batch {idx}.")
        if batch_save_path.exists():
            continue
        index = batch.pop("index")
        prompts = batch.pop("prompts")
        start = time.time()
        col_start = time.time()
        torch.distributed.barrier()
        for key, val in batch.items():
            if key not in generated_probs:
                generated_probs[key] = []
            full_probs = [i.tolist() for i in generator.generate_probs(val)]
            generated_probs[key] += full_probs
            col_time = time.time()
            print(f"Col {key} done in {col_time - col_start} secs.")
            col_start = col_time

        end = time.time()
        print(f"Batch {idx} done in {end - start} secs.")
        start = end

        if global_rank == 0:
            outdf = pd.DataFrame.from_dict(generated_probs)
            outdf.index = index
            outdf.to_csv(batch_save_path)


if __name__ == "__main__":
    main()
