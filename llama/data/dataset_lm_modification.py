import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DefaultDataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    TopKLogitsWarper,
)

import torch
from tqdm.auto import tqdm
import json
import time
import os
import random
from itertools import count
from fire import Fire

from ..utils import custom_parse_args
from ..data_utils import (
    tokenize_and_mask,
    replace_masks,
    extract_fills,
    apply_extracted_fills,
    process_spaces,
)


def main():
    args = custom_parse_args()

    data_path = args.data_path
    col_names = args.col_names
    output_path = args.output_path
    device_id = args.device_id
    n_modifications = args.n_modifications
    top_k = args.top_k
    model_name = args.modifier_model
    torch.set_default_tensor_type(torch.FloatTensor)

    df = pd.read_csv(data_path, index_col=0, sep="\t")
    df = df.loc[df.notna().all(axis=1), :]
    df = df.loc[(df != "").all(axis=1), :]
    if args.n_samples > 0:
        df = df.iloc[: args.n_samples, :]
        

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # model = AutoModelForMaskedLM.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    model.half()
    hftok = AutoTokenizer.from_pretrained(model_name, fast=False, model_max_length=512)
    # collator = DefaultDataCollator()
    # collator = DataCollatorForLanguageModeling(hftok, mlm_probability=0.0)
    # collator = DataCollatorForWholeWordMask(hftok, mlm_probability=0.30)

    device = torch.device(device_id)
    model.to(device)
    all_cols = {}
    warper = TopKLogitsWarper(top_k=top_k)
    for col_name in col_names.split(":"):
        df.loc[:, col_name] = df.loc[:, col_name].str.replace("\n", " ")
        start = time.time()
        for idx in range(n_modifications):
            dl = torch.utils.data.DataLoader(
                df.loc[:, col_name].to_list(),
                collate_fn=lambda x: [process_spaces(i) for i in x],
                batch_size=args.batch_size,
            )
            new_col = []
            
            for batch in tqdm(dl):
                
                tokenized_batch = [tokenize_and_mask(i) for i in batch]
                fills = replace_masks(tokenized_batch, model, hftok, device=device)
                print(fills)
                extracted_fills = extract_fills(fills)
                
                new_texts = apply_extracted_fills(tokenized_batch, extracted_fills)
                
                count = 0
                while "" in new_texts and count < 3:
                    broken_idxs = [idx for idx, x in enumerate(new_texts) if x == ""]
                    new_tokenized_batch = [tokenize_and_mask(i) for idx, i in enumerate(batch) if idx in broken_idxs]
                    new_fills = replace_masks(new_tokenized_batch, model, hftok, device=device)
                    new_new_texts = apply_extracted_fills(new_tokenized_batch, extract_fills(new_fills))
                    for idx, x in zip(broken_idxs, new_new_texts):
                        new_texts[idx] = x
                    count += 1

                new_col += new_texts

            new_start = time.time()
            print(f"Modification {idx} done in {new_start - start}.")
            start = new_start

            new_col_name = f"{col_name}_synthetic_{idx}"
            new_col_series = pd.Series(new_col, name=new_col_name, index=df.index)

            df = pd.concat([df, new_col_series], axis=1)
            df = df.loc[new_col_series.apply(len) > 0, :]

    df.to_csv(output_path, sep="\t", encoding="utf-8", lineterminator="\n")


if __name__ == "__main__":
    main()
