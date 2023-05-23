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

from transformers import AutoModelForCausalLM, AutoTokenizer

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
)

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
    is_hf = args.is_hf
    hf_sample_top_p = args.hf_sample_top_p
    do_lora = args.do_lora
    lora_r = args.lora_r

    if is_hf:
        if int(os.environ.get("SLURM_LOCALID", -1)) > 0:
            return
        generator = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else ckpt_dir, 
            max_model_length=512
        )
        local_rank, global_rank, world_size = 0, 0, 1
    else:
        local_rank, global_rank, world_size = setup_model_parallel()
        if global_rank > 0:
            sys.stdout = open(os.devnull, "w")
        #     sys.stderr = open(os.devnull, "w")
        generator = load(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            do_lora=do_lora,
            lora_r=lora_r
        )
        # generator.model = generator.model.to(torch.bfloat16)

    # torch.distributed.barrier()
    # generator.model.to(torch.device(local_rank))
    
    is_master = global_rank == 0
    
    random.seed(41)
    prompt_len = 30
    max_gen_len = 300
    actual_gen_len = 150

    ds = pd.read_csv(data_path, index_col=0)
    ds = ds.loc[ds.notna().all(axis=1), :]
    ds = ds.loc[
        ds.loc[:, col_name].apply(
            lambda x: len(x.split()) > (prompt_len + actual_gen_len)
        ),
        :,
    ]
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

    global_start = time.time()
    for idx, prompts in enumerate(dataloader):
        start = time.time()
        
        
        if is_hf:
            # prompts = [process_spaces(prompt) for prompt in prompts]
            tokenized_prompts = tokenizer(prompts, truncation=True).input_ids
            
            batch_true_continuations = tokenizer.batch_decode(
                [prompt[: actual_gen_len] for prompt in tokenized_prompts],
                skip_special_tokens=True
            )
            
            prompts = tokenizer.batch_decode(
                [prompt[:prompt_len] for prompt in tokenized_prompts], 
                skip_special_tokens=True
            )
            
            tokenized = tokenizer(prompts, return_tensors="pt", truncation=True)
            
            batch_generated_continuations = generator.generate(
                **{i:j.to("cuda") for i,j in tokenized.items() if i != "token_type_ids"},
                min_length=prompt_len,
                max_length=max_gen_len,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                temperature=temperature
            )
            
            batch_generated_continuations = tokenizer.batch_decode(
                batch_generated_continuations[: actual_gen_len], 
                skip_special_tokens=True
            )
             
                
        else:
            prompts = [process_spaces(prompt).split() for prompt in prompts]
            batch_true_continuations = [
                # " ".join(prompt[prompt_len : prompt_len + actual_gen_len])
                " ".join(prompt[: actual_gen_len])
                for prompt in prompts
            ]
            prompts = [" ".join(prompt[:prompt_len]) for prompt in prompts]

            batch_generated_continuations = generator.generate(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                ignore_eos=True,
                hf_sample_top_p=hf_sample_top_p
            )
        
        all_prompts += prompts
        true_continuations += [
            " ".join(continuation.split(" ")[: actual_gen_len])
            for continuation in batch_true_continuations
        ]
        
        generated_continuations += [
            " ".join(continuation.split(" ")[: actual_gen_len])
            for continuation in batch_generated_continuations
        ]
    
        new_start = time.time()
        print(f"Step {idx} done in {new_start - start} secs.")
        start = new_start

    ds["prompts"] = all_prompts
    ds["true_continuations"] = true_continuations
    ds["generated_continuations"] = generated_continuations
    
    if is_master:
        data_source = Path(data_path).stem
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path.parent / "experiment_rephrase_params.json", "w") as jf:
            json.dump(vars(args), jf)
        ds = ds.loc[ds.notna().all(axis=1), :]
        ds.to_csv(output_path, sep="\t")
        elapsed = time.time() - global_start

        print(f"The process took: {elapsed} seconds.")


if __name__ == "__main__":
    main()
