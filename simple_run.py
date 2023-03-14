from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

ckpt_dir = Path("/home/users/giovannipuccetti/Models/13B")
ckpt_path = ckpt_dir / "consolidated.00.pth"
tokenizer_path = ckpt_dir / "tokenizer.model" 

with open(ckpt_dir / "params.json", "r") as f:
    params = json.loads(f.read())

model_args: ModelArgs = ModelArgs(max_seq_len=16, max_batch_size=2, **params)
tokenizer = Tokenizer(model_path=str(tokenizer_path))
model_args.vocab_size = tokenizer.n_words
torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = Transformer(model_args)


