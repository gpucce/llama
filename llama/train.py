from datetime import datetime
from pathlib import Path
import json
import sys
import os
import random
from copy import deepcopy
from typing import Optional
import gc

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR

import pandas as pd
import wandb

from .generation import LLaMA
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import setup_model_parallel, custom_parse_args, get_cuda_info
from .data_utils import CustomTrainDataLoader
from .optimizer import OffloadOptimizer

PAD_ID = 32000
IGNORE_INDEX = -100


def main():
    args = custom_parse_args()
    
    model_dir = args.model_dir
    output_path = args.output_path
    tokenizer_path = args.tokenizer_path
    epochs = args.epochs
    steps_per_ckpt = args.steps_per_ckpt
    batch_size = args.batch_size
    lr = args.lr
    data_path = args.data_path
    max_seq_len = args.max_seq_len
    max_samples = args.max_samples
    log_freq = args.log_freq
    resume = args.resume
    accum_freq = args.accum_freq
    do_lora = args.do_lora
    lora_r = args.lora_r
    is_torchrun = args.is_torchrun
    do_f16 = args.do_f16
    wandb_project_name = args.wandb_project_name
    warmup_ratio = args.warmup_ratio
    ckpt_dir = model_dir
    optimizer_name = args.optimizer_name
    skip_epoch = args.skip_epoch
    
    random.seed(42)
    local_rank, global_rank, world_size = setup_model_parallel(is_torchrun=is_torchrun)
    device = torch.device(local_rank)

    is_master = global_rank == 0
    
    if is_master:
        print(model_dir, tokenizer_path, epochs, batch_size, output_path, lr, max_seq_len)

    output_path = Path("runs") / Path(output_path)
    if resume is None:
        this_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        run_output_path = output_path / f"run_{this_time}"
        run_output_path.mkdir(exist_ok=True, parents=True)
        resume_epoch = 0 + skip_epoch
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    else:
        run_output_path = sorted(output_path.iterdir())[-1]
        this_time = str(run_output_path).split("_")[-1]
        ckpt_path = sorted(run_output_path.glob("*ckpt_*"))[-1]
        resume_epoch = int(str(ckpt_path).split("_")[-1]) + skip_epoch
        model_dir = ckpt_path / "model"
        checkpoints = sorted(model_dir.glob("consolidated.*.pth"))

        optim_dir = ckpt_path / "optimizer"
        optimizer_checkpoints = sorted(optim_dir.glob("optimizer.*.pth"))
        optim_path = optimizer_checkpoints[global_rank]

        scheduler_dir = ckpt_path / "scheduler"
        scheduler_checkpoints = sorted(scheduler_dir.glob("scheduler.*.pth"))
        scheduler_path = scheduler_checkpoints[global_rank]

    if is_master:
        wandb.init(
            project=wandb_project_name,
            dir=run_output_path,
            name=f"{this_time}_lama",
            id=f"{this_time}_lama",
            tags=[],
            config=args,
            mode="online",
            resume="auto" if resume == "latest" else None,
        )

    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    
    get_cuda_info(local_rank, global_rank, world_size)
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        args_params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        do_cache=False,
        lora_r=lora_r,
        do_lora=do_lora,
        **args_params,
    )

    tokenizer = Tokenizer(tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    ckpt_path = checkpoints[global_rank]

    torch.set_default_tensor_type(torch.cuda.HalfTensor)    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = Transformer(model_args)

    load_before_extension = checkpoint["tok_embeddings.weight"].shape[0] == 32000
    if load_before_extension:
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

    embs = model.tok_embeddings.weight.data
    emb_height, emb_width = embs.shape
    new_embs = torch.ones(emb_height + 1, emb_width)
    new_embs[:emb_height, :] = embs
    new_embs[-1, :] = embs.mean(0)
    model.tok_embeddings.weight.data = new_embs

    if not load_before_extension:
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

    model_dtype = torch.float32
    if do_f16:
        model_type = torch.float16
    model = model.to(device, dtype=model_dtype)

    
    sampler = None
    if not do_lora:
        optimizer_params = [
            {"params": (j for i, j in model.named_parameters() if "norm" in i), "wd":0.0},
            {"params": (j for i, j in model.named_parameters() if "norm" not in i), "wd":0.1},
        ]
    else:
        optimizer_params = (j for i,j in model.named_parameters() if "lora" in i)

    if optimizer_name == "sgd":
        optimizer = SGD(optimizer_params, lr=lr)
    elif optimizer_name == "adamw":
        optimizer = Adam(optimizer_params, lr=lr, eps=1.e-4)
    else:
        assert Fslse, "unknown optimizer name"

    dataloader = CustomTrainDataLoader(
        tokenizer,
        data_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        n_epochs=epochs
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        # TODO: this is horrible
        total_steps=(dataloader._n_samples * (epochs - skip_epoch)) // batch_size // accum_freq,
        pct_start=warmup_ratio
    )
    
    if resume is not None:
        optimizer.load_state_dict(torch.load(optim_path, map_location="cpu"))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

    # exhaust past samples if resume
    global_step = 0
    ckpt_n = 0
    for _ in range(0, resume_epoch):
        ckpt_n += 1
        for step in range(dataloader._n_samples // batch_size):
            next(dataloader)
            global_step += 1

    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    torch.distributed.barrier()
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        labels = deepcopy(batch[:, 1:]).to(device)
        labels[labels == PAD_ID] = IGNORE_INDEX
        input_ids = batch[:, :-1]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            out = model(input_ids, labels=labels)
        loss = out["loss"]
        assert loss.dtype == torch.float32
        scaler.scale(loss).backward()
        if (step % accum_freq == 0):
            # optimizer.step()
            scheduler.step()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            

        if is_master and (global_step % log_freq == 0):
            _loss = loss.clone().item()
            epoch = dataloader.epochs_done + 1
            print(
                f"STEP {global_step}, "
                f"EPOCH {epoch}, RESUME STEP {step}, "
                f"Learn-Rate {optimizer.param_groups[0]['lr']}, GR {global_rank}, LOSS {_loss}"
            )
            
            log = {}
            log["epoch"] = epoch
            log["loss"] = _loss
            log["resume_step"] = step
            log["global_step"] = global_step
            log["lr"] = optimizer.param_groups[0]['lr']
            wandb.log(log)

        global_step += 1

        if step % steps_per_ckpt == 0 and step > 0:

            torch.distributed.barrier()
            ckpt_path = run_output_path / f"ckpt_{ckpt_n:05}"
            ckpt_path.mkdir(exist_ok=True, parents=True)
        
            model_file_name = f"consolidated.{global_rank:03}.pth"
            model_dir = ckpt_path / "model"
            model_dir.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), model_dir / model_file_name)

            optim_dir = ckpt_path / "optimizer"
            optim_dir.mkdir(exist_ok=True, parents=True)
            optim_file_name = f"optimizer.{global_rank:03}.pth"
            torch.save(optimizer.state_dict(), optim_dir / optim_file_name)

            scheduler_dir = ckpt_path / "scheduler"
            scheduler_dir.mkdir(exist_ok=True, parents=True)
            sched_file_name = f"scheduler.{global_rank:03}.pth"
            torch.save(scheduler.state_dict(), scheduler_dir / sched_file_name)
            
            if is_master:
                with open(model_dir / "params.json", "w") as pp:
                    json.dump(args_params, pp)
            
            ckpt_n += 1

    dataloader.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
