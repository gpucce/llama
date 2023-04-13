import json
from pathlib import Path
from random import shuffle
import pandas as pd
from .finetune_data_prepare import data_prepare


def _prepare_input_data(data_path, tokenizer):
    data_path = Path(data_path)
    dfs = []
    for data_file in data_path.iterdir():
        dfs.append(pd.read_csv(data_path / data_file))
    all_rows = []
    for col in ["headline", "full_text"]:
        for df in dfs:
            all_rows += df.loc[:, col].to_list()
    shuffle(all_rows)
    all_tokenized_seqs = []
    for seq in all_rows:
        tokenized_seq = tokenizer.encode(seq + ".", bos=True, eos=True)
        all_tokenized_seqs.append(json.dumps(tokenized_seq))
    return all_tokenized_seqs


if __name__ == "__main__":
    tokenizer_path = "/home/users/giovannipuccetti/Models/65B/tokenizer.model"
    data_path = "/home/users/giovannipuccetti/Data/CHANGE-it/train"
    seq_len = 128
    output_path = (
        f"/home/users/giovannipuccetti/Data/tokenized_change_it_seq_len_{seq_len}.jsonl"
    )

    data_prepare(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        seq_len=seq_len,
        input_prepare_func=_prepare_input_data,
    )
