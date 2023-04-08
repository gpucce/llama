import json
from pathlib import Path
from itertools import count
import time
from random import shuffle

from tqdm.auto import tqdm
import regex as re
from .tokenizer import Tokenizer
import random


def _prepare_input_data(data_path, tokenizer):
    books_path = list((data_path / "libri-2013").iterdir()) + list(
        (data_path / "libri-2023").iterdir()
    )
    random.shuffle(books_path)
    all_tokenized_seqs = []
    for book_path in books_path:
        with open(book_path, "rb") as bf:
            data = bf.read().decode("latin1")
            data = re.sub("[ \t\r\n\f]+", " ", data)
        datas = data.split(".")
        for seq in datas:
            tokenized_seq = tokenizer.encode(seq + ".", bos=True, eos=True)
            all_tokenized_seqs.append(json.dumps(tokenized_seq))

    shuffle(all_tokenized_seqs)
    return all_tokenized_seqs


def data_prepare(
    tokenizer_path: str,
    data_path: str,
    output_path: str,
    seq_len: int,
    input_prepare_func,
):
    tokenizer = Tokenizer(tokenizer_path)
    data_path = Path(data_path)

    start = time.time()

    all_tokenized_seqs = input_prepare_func(data_path, tokenizer)

    with open(output_path, "w") as df:
        buffer = []
        for idx, tokenized_seq in enumerate(all_tokenized_seqs):
            tokenized_seq = buffer + json.loads(tokenized_seq)
            n_elem = len(tokenized_seq)
            start = 0
            for i in count(seq_len, seq_len):
                if i > n_elem:
                    buffer = tokenized_seq[start:]
                    break
                json.dump(tokenized_seq[start:i], df)
                df.write("\n")
                start = i

            elapsed = time.time() - start
            if idx % 100 == 0:
                print(f"Processed {idx} books in {elapsed}")


if __name__ == "__main__":
    tokenizer_path = "/home/users/giovannipuccetti/Models/13B/tokenizer.model"
    data_path = "/home/users/giovannipuccetti/Data/"
    output_path = f"/home/users/giovannipuccetti/Data/books_ita_squences_tokenized_{seq_len}.jsonl"
    seq_len = 128

    main(
        tokenizer_path=tokenizer_path,
        data_path=data_path,
        output_path=output_path,
        seq_len=seq_len,
        input_prepare_func=_prepare_input_data,
    )
