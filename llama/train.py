from datetime import datetime
from pathlib import Path
import json
import sys
import os
from copy import deepcopy
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR

from fire import Fire
import pandas as pd
import wandb

from .generation import LLaMA
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import setup_model_parallel

from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

PAD_ID = 32000
IGNORE_INDEX = -100

this_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")

torch.autograd.set_detect_anomaly(True)

class Collator:
    def __init__(self, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, x):
        out = torch.ones(len(x), self.max_seq_len, dtype=int) * PAD_ID
        for idx, sample in enumerate(x):
            tokenized = torch.tensor(
                self.tokenizer.encode(sample, bos=True, eos=False)
            )[: self.max_seq_len]
            out[idx, : len(tokenized)] = tokenized

        return out


def main(
    model_dir: str,
    tokenizer_path: str,
    epochs: int = 3,
    batch_size: int = 8,
    output_path: str = f"test_output_{this_time}",
    lr=1.0e-5,
    max_seq_len: int = 512,
    max_samples: Optional[int] = None
):
    print(model_dir, tokenizer_path, epochs, batch_size, output_path, lr, max_seq_len)
    args = dict(
        model_dir=model_dir,
        tokenizer_path=tokenizer_path,
        epochs=epochs,
        batch_size=batch_size,
        output_path=output_path,
        lr=lr,
        max_seq_len=max_seq_len,
    )

    

    ckpt_dir = model_dir
    local_rank, global_rank, world_size = setup_model_parallel()
    device = torch.device(local_rank)
    is_master = global_rank == 0
    output_path = Path(output_path)
    if is_master:
        output_path.mkdir(exist_ok=True, parents=True)
        wandb.init(
            project="llama_fine_tune",
            name=f"{this_time}_lama",
            id=f"{this_time}_lama",
            tags=[],
            config=args,
            mode="online"
        )

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"

    ckpt_path = checkpoints[global_rank]
    output_path = Path(output_path)
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        args_params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        do_cache=False,
        **args_params
    )
    tokenizer = Tokenizer(tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    embs = model.tok_embeddings.weight.data
    emb_height, emb_width = embs.shape
    new_embs = torch.ones(emb_height + 1, emb_width)
    new_embs[:emb_height, :] = embs
    new_embs[-1, :] = embs.mean(0)
    model.tok_embeddings.weight.data = new_embs
    with open("/home/users/giovannipuccetti/Data/capra.txt") as df:
        data = df.readlines()
        if max_samples:
            data = data[:max_samples]
        
    model = model.to(device, dtype=torch.float16)
    torch.distributed.barrier()

    torch.set_default_tensor_type(torch.FloatTensor)
    # optimizer = AdamW(params=(j for i, j in model.named_parameters() if "norm" not in i), lr=lr)
    optimizer = SGD((j for i, j in model.named_parameters() if "norm" not in i), lr=lr)
    # optimizer = OSS(
    #     params=(j for i, j in model.named_parameters() if "norm" not in i),
    #     optim=AdamW,
        # **base_optimizer_arguments
    # )
    # scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(data), epochs=int(epochs))

    collator = Collator(tokenizer, max_seq_len=max_seq_len)
    
    sampler = None # DistributedSampler(data)
    dataloader = DataLoader(data, collate_fn=collator, batch_size=batch_size, sampler=sampler)

    # uncomment to print gpu infos
    # t = torch.cuda.get_device_properties(local_rank).total_memory
    # r = torch.cuda.memory_reserved(local_rank)
    # a = torch.cuda.memory_allocated(local_rank)
    # f = r-a  # free inside reserved
    # print(f"Memory, total {t}, reserved {r}, allocated {a}, free {f}, local rank {local_rank}, device {str(device)}, world_size {world_size}")
    
    log = {}
    log_freq = 3
    for epoch in range(epochs):
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            labels = deepcopy(batch[:, 1:])
            labels[labels == PAD_ID] = IGNORE_INDEX
            out = model(batch[:, :-1].to(device), labels=labels.to(device))
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            # scheduler.step()
            step = idx + epoch * len(data) / log_freq
            _loss = deepcopy(loss.detach())
            
            if idx % log_freq == 0:
                
                torch.distributed.barrier()
                torch.distributed.reduce(loss.detach(), 0, op=torch.distributed.ReduceOp.SUM) 
                loss = loss / world_size
                if is_master:
                    print(f"STEP {step} LR {local_rank}, GR {global_rank}, LOSS {_loss}, AVERAGE LOSS {loss}")
                
            
            if is_master and (idx % log_freq == 0):
                log["epoch"] = epoch
                log["loss"] = _loss
                log["step"] = step
                log["average_loss"] = loss
                
                wandb.log(log)
                log = {}
        torch.save(
            model, output_path / f"ckpt_{epoch + 1}_shard_{global_rank}.pth",
        )


if __name__ == "__main__":
    Fire(main)
