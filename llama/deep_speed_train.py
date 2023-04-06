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
from torch.utils.checkpoint import checkpoint
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR

import deepspeed

import pandas as pd
import wandb

from .generation import LLaMA
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import setup_deep_speed_model_parallel, custom_parse_args

from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

PAD_ID = 32000
IGNORE_INDEX = -100


class CustomDataLoader:
    def __init__(
        self, tokenizer, data_path, max_samples, max_seq_len=512, batch_size=1
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.data = open(data_path)
        self.max_samples = max_samples
        self.samples_done = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.samples_done >= self.max_samples:
            self.close()
            self.data = open(data_path)
            self.samples_done = 0

        self.samples_done += self.batch_size
        new_batch = []
        while True:
            if len(new_batch) == self.batch_size:
                break
            try:
                new_sample = json.loads(next(self.data))
                assert len(new_sample) == 128
            except:
                continue
            new_batch.append(new_sample)

        return torch.tensor(new_batch, dtype=int)[:, : self.max_seq_len]

    def close(self):
        self.data.close()


def main():
    args = custom_parse_args()
    log_args = vars(args)
    to_log = "INFO |"
    for key, val in log_args.items():
        if val is not None:
            to_log += key.upper() + f" {val} | "
    to_log += "\n"

    ckpt_dir = args.model_dir
    local_rank, global_rank, world_size = setup_deep_speed_model_parallel()
    device = torch.device(local_rank)
    is_master = global_rank == 0

    if is_master:
        print(to_log)

    output_path = Path(args.output_path)
    if args.resume is None:
        this_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        run_output_path = output_path / f"run_{this_time}"
        run_output_path.mkdir(exist_ok=True, parents=True)
        resume_epoch = 0
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    else:
        run_output_path = sorted(output_path.iterdir())[-1]
        this_time = str(run_output_path).split("_")[-1]
        epoch_path = sorted(run_output_path.glob("*epoch_*"))[-1]
        resume_epoch = int(str(epoch_path).split("_")[-1])
        model_dir = epoch_path / "model"
        checkpoints = sorted(model_dir.glob("consolidated.*.pth"))

        optim_dir = epoch_path / "optimizer"
        optimizer_checkpoints = sorted(optim_dir.glob("optimizer.*.pth"))
        optim_path = optimizer_checkpoints[global_rank]

        # scheduler_dir = epoch_path / "scheduler"
        # scheduler_checkpoints = sorted(scheduler_dir.glob("scheduler.*.pth"))
        # scheduler_path = scheduler_checkpoints[global_rank]

    ckpt_path = checkpoints[global_rank]

    if is_master:
        wandb.init(
            project="llama_fine_tune",
            dir=run_output_path,
            name=f"{this_time}_lama",
            id=f"{this_time}_lama",
            tags=[],
            config=log_args,
            mode="offline",
            resume="auto" if args.resume == "latest" else None,
        )

    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        args_params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        do_cache=False,
        **args_params,
    )
    tokenizer = Tokenizer(args.tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    # torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    load_before_extension = checkpoint["tok_embeddings.weight"].shape[0] == 32000
    if load_before_extension:
        model.load_state_dict(checkpoint, strict=False)

    embs = model.tok_embeddings.weight.data
    emb_height, emb_width = embs.shape
    new_embs = torch.ones(emb_height + 1, emb_width)
    new_embs[:emb_height, :] = embs
    new_embs[-1, :] = embs.mean(0)
    model.tok_embeddings.weight.data = new_embs

    if not load_before_extension:
        model.load_state_dict(checkpoint, strict=False)

    # model = model.to(device, dtype=torch.float16)
    model_engine, _, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters()
    )
    torch.distributed.barrier()
    # torch.set_default_tensor_type(torch.FloatTensor)

    sampler = None  # DistributedSampler(data)
    # optimizer = AdamW(params=(j for i, j in model.named_parameters() if "norm" not in i), lr=lr)
    # optimizer = SGD((j for i, j in model.named_parameters() if "norm" not in i), lr=lr)
    # scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

    if args.resume is not None:
        optimizer.load_state_dict(torch.load(optim_path, map_location="cpu"))
        # scheduler.load_state_dict(scheduler_path, map_location="cpu")

    log = {}
    book_id = 0

    dataloader = CustomDataLoader(
        tokenizer,
        args.data_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # exhaust past samples if resume
    for _ in range(0, resume_epoch):
        for step in range(args.steps_per_epoch):
            next(dataloader)

    global_step = resume_epoch * args.steps_per_epoch
    for epoch in range(resume_epoch, args.epochs):
        # optimizer.zero_grad()
        for step in range(args.steps_per_epoch):
            batch = next(dataloader)
            labels = deepcopy(batch[:, 1:])
            labels[labels == PAD_ID] = IGNORE_INDEX
            out = model_engine(batch[:, :-1].to(device), labels=labels.to(device))
            loss = out["loss"]
            model_engine.backward(loss)
            model_engine.step()
            _loss = loss.clone().detach()

            if is_master and (global_step % args.log_freq == 0):
                print(
                    f"STEP {global_step}, EPOCH {epoch}, EPOCH_STEP {step}, LR {local_rank}, GR {global_rank}, LOSS {_loss}"
                )

                log["epoch"] = epoch
                log["loss"] = _loss
                log["epoch_step"] = step
                log["global_step"] = global_step

                wandb.log(log)
                log = {}
            global_step += 1

        torch.distributed.barrier()
        epoch_path = run_output_path / f"epoch_{epoch + 1:05}"
        epoch_path.mkdir(exist_ok=True, parents=True)

        model_file_name = f"consolidated.{global_rank:03}.pth"
        model_dir = epoch_path / "model"
        model_dir.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), model_dir / model_file_name)

        optim_dir = epoch_path / "optimizer"
        optim_dir.mkdir(exist_ok=True, parents=True)
        optim_file_name = f"optimizer.{global_rank:03}.pth"
        torch.save(optimizer.state_dict(), optim_dir / optim_file_name)

        # scheduler_dir = epoch_path / "scheduler"
        # scheduler_dir.mkdir(exist_ok=True, parents=True)
        # sched_file_name = f"scheduler.{global_rank:03}.pth"
        # torch.save(scheduler.state_dict(), scheduler_dir / sched_file_name)

        with open(model_dir / "params.json", "w") as pp:
            json.dump(args_params, pp)

    dataloader.close()


if __name__ == "__main__":
    main()
