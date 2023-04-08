import pandas as pd
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


def main(data_path: str, col_name: str, output_path: str, device_id: int):
    df = pd.read_csv(data_path, index_col=0)
    model_name = "dbmdz/bert-base-italian-xxl-cased"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    hftok = AutoTokenizer.from_pretrained(model_name, fast=False)
    # collator = DataCollatorForLanguageModeling(hftok)
    collator = DataCollatorForWholeWordMask(hftok)

    def tokenize(x, return_tensors=None):
        return hftok(
            x,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_special_tokens_mask=True,
            return_tensors=return_tensors,
        )

    device = torch.device(device_id)
    model.to(device)
    output = [[] for i in range(df.shape[0])]
    for _ in tqdm(range(100)):
        dl = torch.utils.data.DataLoader(
            df.full_text.to_list(),
            collate_fn=lambda x: collator([tokenize(i) for i in x]),
            batch_size=16,
        )
        count = 0
        for batch in dl:
            input_ids = batch["input_ids"].clone().detach().cpu()
            out = model(**{i: j.to(device) for i, j in batch.items()})
            mask = batch["input_ids"] == hftok.mask_token_id
            new_toks = torch.argmax(out.logits[mask], dim=-1)
            input_ids[mask] = new_toks.cpu()
            for i in hftok.batch_decode(input_ids):
                output[count].append(i[5:-5].strip())
                count += 1

    for i in range(len(output[0])):
        df[f"synthetic_{i}"] = [j[i] for j in output]
    df.to_csv(output_path)


if __name__ == "__main__":
    Fire(main)
