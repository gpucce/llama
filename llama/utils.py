
import os
import time
import json
from pathlib import Path
from typing import Tuple, Optional

import torch
from argparse import ArgumentParser


from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from .generation import LLaMA
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer

def setup_model_parallel() -> Tuple[int, int, int]:
    local_rank = int(os.environ.get("SLURM_LOCALID", -1))
    global_rank = int(os.environ.get("SLURM_PROCID", -1))
    world_size = int(os.environ.get("SLURM_NTASKS", -1))
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    torch.distributed.init_process_group(
        "nccl", world_size=world_size, rank=global_rank, init_method="env://"
    )
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, global_rank, world_size


def get_cuda_info():
    t = torch.cuda.get_device_properties(local_rank).total_memory
    r = torch.cuda.memory_reserved(local_rank)
    a = torch.cuda.memory_allocated(local_rank)
    f = r - a  # free inside reserved
    print(
        f"Memory, total {t},",
        f"reserved {r}, allocated {a},",
        f"free {f},",
        f"local rank {local_rank},",
        f"device {device},",
        f"world_size {world_size}",
    )


def custom_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-path", type=str, default="test_output")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/users/txtgiovannipuccetti/Data/books_ita_tokenized_128.jsonl",
    )
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--resume", type=Optional[str], default=None)
    parser.add_argument("--accum-freq", type=int, default=1)
    return parser.parse_args()

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    global_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:

    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[global_rank]
    time.sleep(5 * local_rank)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        do_cache=True,
        **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    
    embs = model.tok_embeddings.weight.data
    emb_height, emb_width = embs.shape
    new_embs = torch.ones(emb_height + 1, emb_width)
    model.tok_embeddings.weight.data = new_embs
    
    model.load_state_dict(checkpoint, strict=False)
    model_size = 0
    for i in model.parameters():
        model_size += i.nelement() * i.element_size()
    torch.set_default_tensor_type(torch.FloatTensor)

    generator = LLaMA(model, tokenizer)
    return generator