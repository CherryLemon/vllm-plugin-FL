# SPDX-License-Identifier: Apache-2.0
"""Independent vLLM architecture for the FL Eager reference profile."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from .arguments import model_args
from .model import Transformer, set_dtype
from .ops import EXECUTION_PROFILE


class DeepseekV41FlashFLForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        root = Path(vllm_config.model_config.model)
        self.original_config = json.loads((root / "config.json").read_text())
        self.args = model_args(
            self.original_config, vllm_config.model_config.max_model_len
        )
        self.profile = EXECUTION_PROFILE
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        with set_dtype(torch.bfloat16):
            self.core = Transformer(self.args, tokenizer=tokenizer, enable_mtp=False)
        self.requires_grad_(False)

    def embed_input_ids(self, input_ids):
        return self.core.embed(input_ids)

    @torch.inference_mode()
    def forward(self, input_ids, positions=None, *, start_pos=None):
        if start_pos is None:
            if positions is None or positions.numel() == 0:
                raise ValueError("positions or start_pos are required")
            start_pos = int(positions.flatten()[0].item())
        ids = input_ids.reshape(1, -1)
        if start_pos and ids.shape[1] != 1:
            raise ValueError(
                "reference profile supports one prefill followed by single-token decode"
            )
        if start_pos + ids.shape[1] > self.args.max_seq_len:
            raise ValueError("request exceeds allocated context state")
        with torch.device(input_ids.device), set_dtype(torch.bfloat16):
            # This architecture's intermediate is already the final logits.
            # The custom Runner calls compute_logits through the public model API.
            _, logits, _ = self.core(ids, start_pos)
        return logits

    def compute_logits(self, hidden_states):
        return hidden_states
