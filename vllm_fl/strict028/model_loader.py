# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


@register_model_loader("fl_dsv41")
class FLDeepseekV41Loader(BaseModelLoader):
    def download_model(self, model_config):
        from pathlib import Path

        if not (Path(model_config.model) / "model.safetensors.index.json").is_file():
            raise ValueError("fl_dsv41 requires an original local checkpoint")

    def load_weights(self, model, model_config):
        from .collectives import parallel_layout
        from .models.deepseek_v41.loader import load_original_checkpoint

        layout = parallel_layout()
        with torch.no_grad():
            model.load_manifest = load_original_checkpoint(
                model.core,
                model_config.model,
                layout.tensor_rank,
                layout.tensor_size,
                expert_rank=layout.global_rank,
                expert_size=layout.world_size,
            )

    def load_model(self, vllm_config, model_config, prefix=""):
        from .models.deepseek_v41.entry import DeepseekV41FlashFLForCausalLM

        self.download_model(model_config)
        with torch.device(vllm_config.device_config.device), torch.no_grad():
            model = DeepseekV41FlashFLForCausalLM(vllm_config, prefix)
        self.load_weights(model, model_config)
        return model.eval()
