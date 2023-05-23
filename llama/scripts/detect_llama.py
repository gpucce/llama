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

from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # device = "cpu"
    if args.is_hf:
        generator = AutoModelForCausalLM.from_pretrained(
            args.model_dir, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path if args.tokenizer_path is not None else args.model_dir,
            model_max_length=256
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # generator.resize_token_embeddings(len(tokenizer))
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        local_rank, global_rank, world_size = 0, 0, 1
    else:
        local_rank, global_rank, world_size = setup_model_parallel()
        if global_rank > 0:
            sys.stdout = open(os.devnull, "w")
        generator = load(
            ckpt_dir=args.model_dir,
            tokenizer_path=args.tokenizer_path,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.batch_size,
            do_cache=args.do_not_cache
        )
        
        # if args.do_fp32:
        #     generator.model.to(torch.float32)

        generator.model.to(torch.bfloat16)
        generator.model.to(torch.device(local_rank))
        
    ckpt_dir = args.model_dir
    output_path = Path(args.output_path)
    if global_rank == 0:
        output_path.mkdir(exist_ok=True, parents=True)

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
        # if not args.is_hf:
            # torch.distributed.barrier()
        for key, val in batch.items():
            if key not in generated_probs:
                generated_probs[key] = []
            if args.is_hf:
                tokenized = tokenizer(
                    val,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True
                )
                labels = tokenized.input_ids.clone()
                tokenized.input_ids[tokenized.input_ids==tokenizer.pad_token_id] = 1
                labels[labels==tokenizer.pad_token_id] = -100
                logits = generator(
                    **{i:j.to("cuda") for i,j in tokenized.items() if i != "token_type_ids"}
                ).logits / args.temperature
                nlog_prob = [
                    i[i!=0].tolist()
                    for i in ce(
                        logits[:, :-1, :].permute(0, 2, 1), 
                        labels[:, 1:].to("cuda")
                    ).clone().detach().cpu()
                ]
            else:
                nlog_prob = generator.generate_probs(val, temperature=args.temperature)
            generated_probs[key] += nlog_prob
            col_time = time.time()
            print(f"Col {key} done in {col_time - col_start} secs.")
            col_start = col_time

        end = time.time()
        print(f"Batch {idx} done in {end - start} secs.")
        start = end

        if global_rank == 0:
            output_path.mkdir(exist_ok=True, parents=True)
            with open(output_path / "experiment_detect_params.json", "w") as jf:
                json.dump(vars(args), jf)
            outdf = pd.DataFrame.from_dict(generated_probs)
            outdf.index = index
            outdf.to_csv(batch_save_path)


if __name__ == "__main__":
    main()
