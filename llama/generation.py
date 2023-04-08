# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        if hasattr(self.model, "module"):
            params = self.model.module._fpw_module.params
        else:
            params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

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
            )
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
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
        max_gen_len: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        output_full_probs: bool = False,
    ) -> List[str]:
        bsz = len(prompts)
        if hasattr(self.model, "module"):
            params = self.model.module._fpw_module.params
        else:
            params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [
            self.tokenizer.encode(x, bos=True, eos=False)[:max_gen_len] for x in prompts
        ]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        device = next(self.model.parameters()).device
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()

        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = 0
        logits = self.model.detect_forward(tokens, start_pos)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)

        true_probs = []
        if output_full_probs:
            full_probs = []

        for prob, tok, mask in zip(probs, tokens, input_text_mask):
            tok = tok[mask]  # ntok
            prob = prob[mask]  # ntok, vocabsize
            true_prob = prob[torch.arange(len(tok)), tok]  # ntok
            true_probs.append(true_prob)
            if output_full_probs:
                full_probs.append(prob)

        if output_full_probs:
            return full_probs, true_probs

        return true_probs


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
