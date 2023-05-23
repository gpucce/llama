# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from transformers import TopPLogitsWarper

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str] = None,
        max_gen_len: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        prompt_tokens=None,
        ignore_eos=False,
        hf_sample_top_p=False
    ) -> List[str]:
        bsz = len(prompts)
        if hasattr(self.model, "module"):
            params = self.model.module._fpw_module.params
        else:
            params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        if prompts is not None:
            prompt_tokens = [
                self.tokenizer.encode(x, bos=True, eos=False) for x in prompts
            ]
        else:
            assert (
                prompts_tokens is not None
            ), "Either prompt or prompt tokens must be provided."

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        device = next(self.model.parameters()).device
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()

        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.generation_forward(
                tokens[:, prev_pos:cur_pos], prev_pos
            ).clone()
            
            if ignore_eos:
                logits[:, self.tokenizer.eos_id] = -float("inf")

            if temperature > 0:
                if hf_sample_top_p:
                    next_token = sample_top_p_hf(logits / temperature, top_p)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def generate_probs(
        self,
        prompts: List[str],
        max_gen_len: int = 1,
        top_p: float = 0.95,
        temperature: float = 1.0,
        output_full_probs: bool = False,
        remove_pad: bool = True,
    ) -> List[str]:
        bsz = len(prompts)
        if hasattr(self.model, "module"):
            params = self.model.module._fpw_module.params
        else:
            params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [
            self.tokenizer.encode(x, bos=True, eos=False)[: params.max_seq_len]
            for x in prompts
        ]
        labels = [prompt[1:] for prompt in prompt_tokens]
        prompt_tokens = [prompt[:-1] for prompt in prompt_tokens]

        tok_sizes = [len(t) for t in prompt_tokens]
        min_prompt_size = min(tok_sizes)
        max_prompt_size = max(tok_sizes)
        all_sizes = sorted(list(set(tok_sizes)))

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        device = next(self.model.parameters()).device
        tokens = torch.full((bsz, total_len), self.tokenizer.bos_id).to(device).long()
        _labels = torch.full((bsz, total_len), -100).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
            _labels[k, : len(t)] = torch.tensor(labels[k]).long()
        tok_sizes = torch.tensor(tok_sizes).to(device)

        start_pos = min_prompt_size
        prev_pos = 0
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        logits = self.model.detect_forward(tokens, 0) / temperature
        return [i[i!=0].tolist() for i in ce(logits.permute(0, 2, 1), _labels)]

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def sample_top_p_hf(logits, top_p):
    toppwarper = TopPLogitsWarper(top_p=top_p)
    logits = toppwarper(input_ids=None, scores=logits)
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
    
    