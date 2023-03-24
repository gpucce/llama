
import json
from pathlib import Path
from itertools import count
import time

from tqdm.auto import tqdm
import regex as re
from .tokenizer import Tokenizer
import random


tok = Tokenizer("/home/users/giovannipuccetti/Models/13B/tokenizer.model")
data_path = Path("/home/users/giovannipuccetti/Data/")
books_path = list((data_path / "libri-2013").iterdir()) + list((data_path / "libri-2023").iterdir())
random.shuffle(books_path)
seq_len = 128
start = time.time()
with open(f"/home/users/giovannipuccetti/Data/books_ita_tokenized_{seq_len}.jsonl", "w") as df:
    for idx, book_path in enumerate(books_path):
        with open(book_path, "rb") as bf:
            data = bf.read().decode("latin1")
            data = re.sub("[ \t\r\n\f]+", " ", data)
        datas = data.split(".")
        buffer = []
        for seq in datas:
            tokenized_seq = tok.encode(seq + ".", bos=True, eos=False)
            if len(tokenized_seq) < 3:
                continue
            tokenized_seq = buffer + tokenized_seq
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

