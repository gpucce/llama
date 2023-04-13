import pandas as pd
from typing import List
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
)

import torch
from tqdm.auto import tqdm
import json
import time
import os
import random
from itertools import count
from fire import Fire


def main(
    data_path: str,
    col_names: str,
    output_path: str,
    device_id: int,
    n_modifications: int = 100,
):
    df = pd.read_csv(data_path, index_col=0, sep="\t").iloc[:30, :]
    model_name = "dbmdz/bert-base-italian-xxl-cased"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    hftok = AutoTokenizer.from_pretrained(model_name, fast=False)
    # collator = DataCollatorForLanguageModeling(hftok)
    collator = DataCollatorForWholeWordMask(hftok, mlm_probability=0.15)

    def tokenize(x, return_tensors=None):
        return hftok(
            x,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
            return_tensors=return_tensors,
        )

    device = torch.device(device_id)
    model.to(device)
    all_cols = {}
    for col_name in col_names.split(":"):
        for idx in range(n_modifications):
            dl = torch.utils.data.DataLoader(
                df.loc[:, col_name].to_list(),
                collate_fn=lambda x: collator(
                    [tokenize(" ".join(i.split())) for i in x]
                ),
                batch_size=4,
            )
            new_col = []
            for batch in dl:
                input_ids = batch["input_ids"].clone().detach().cpu()
                out = model(**{i: j.to(device) for i, j in batch.items()})
                mask = batch["input_ids"] == hftok.mask_token_id
                new_toks = torch.argmax(out.logits[mask], dim=-1)
                input_ids[mask] = new_toks.cpu()
                for i in hftok.batch_decode(input_ids, skip_special_tokens=True):
                    new_col.append(
                        " ".join(i.split())
                        .strip()
                        .encode("utf-8")
                        .decode("utf-8")
                        .replace("\t", "")
                        .replace("\n", "")
                    )
            new_col_name = f"{col_name}_synthetic_{idx}"
            df = pd.concat(
                [df, pd.Series(new_col, name=new_col_name, index=df.index)], axis=1
            )
            all_cols[new_col_name] = new_col
    with open("test_to_check.json", "w") as jf:
        json.dump(all_cols, jf)
    df.to_csv(output_path, sep="\t", encoding="utf-8")


if __name__ == "__main__":
    Fire(main)
