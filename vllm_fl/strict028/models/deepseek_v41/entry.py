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
        self.speculative_config = vllm_config.speculative_config
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        with set_dtype(torch.bfloat16):
            self.core = Transformer(
                self.args,
                tokenizer=tokenizer,
                enable_mtp=self.speculative_config is not None,
            )
        self.requires_grad_(False)

    def embed_input_ids(self, input_ids):
        return self.core.embed(input_ids)

    @torch.inference_mode()
    def forward(self, input_ids, positions=None, *, start_pos=None):
        logits, _ = self.forward_with_aux(input_ids, positions, start_pos=start_pos)
        return logits

    @torch.inference_mode()
    def forward_with_aux(self, input_ids, positions=None, *, start_pos=None):
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
            _, logits, hidden = self.core(ids, start_pos)
        return logits, hidden

    @torch.inference_mode()
    def store_draft_context(self, hidden, start_pos):
        with torch.device(hidden.device), set_dtype(torch.bfloat16):
            self.core.store_spec_context(hidden, start_pos)

    @torch.inference_mode()
    def propose_draft(self, token, hidden, start_pos):
        with torch.device(hidden.device), set_dtype(torch.bfloat16):
            return self.core.forward_spec(token.reshape(-1), hidden, start_pos)

    def compute_logits(self, hidden_states):
        return hidden_states


class DeepseekV41DSparkFLForCausalLM(nn.Module):
    """Real DSpark draft architecture used for the host's draft-config inspection.

    WorkerFL028 executes the same layers inside the target graph so the embedding,
    vocabulary head and request-state allocation are shared, rather than loading
    a second checkpoint. This entry also supports standalone draft forwards.
    """

    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        from .model import DSparkTransformer

        root = Path(vllm_config.model_config.model)
        self.args = model_args(
            json.loads((root / "config.json").read_text()),
            vllm_config.model_config.max_model_len,
        )
        with set_dtype(torch.bfloat16):
            self.core = DSparkTransformer(self.args)
        self.requires_grad_(False)

    def embed_input_ids(self, input_ids):
        return self.core.embed(input_ids)

    @torch.inference_mode()
    def forward(self, input_ids, positions, hidden_states):
        start = int(positions.flatten()[0].item())
        with torch.device(input_ids.device), set_dtype(torch.bfloat16):
            return self.core.forward_spec(input_ids.reshape(-1), hidden_states, start)

    def compute_logits(self, hidden_states):
        return hidden_states[1]
