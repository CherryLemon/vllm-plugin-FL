# SPDX-License-Identifier: Apache-2.0
"""Derive reference graph arguments from the original HF configuration."""

from .model import ModelArgs


def model_args(config, max_model_len):
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("expected the original DeepSeek V4.1 configuration")
    quant = config["quantization_config"]
    if (
        quant.get("quant_method") != "fp8"
        or quant.get("weight_block_size") != [32, 32]
        or quant.get("scale_fmt") != "ue8m0"
        or quant.get("expert_dtype") != "fp4"
    ):
        raise ValueError("unsupported checkpoint quantization contract")
    text = config["text_config"]
    mapping = {
        "dim": "hidden_size",
        "moe_inter_dim": "moe_intermediate_size",
        "n_layers": "num_hidden_layers",
        "n_mtp_layers": "num_nextn_predict_layers",
        "n_heads": "num_attention_heads",
        "rope_head_dim": "qk_rope_head_dim",
        "norm_eps": "rms_norm_eps",
        "n_activated_experts": "num_experts_per_tok",
        "score_func": "scoring_func",
        "route_scale": "routed_scaling_factor",
        "window_size": "sliding_window",
        "kv_source_layers": "kv_source_layer_ids",
        "index_source_layers": "index_source_layer_ids",
        "candidate_source_layer": "candidate_source_layer_id",
        "engram_pad_id": "engram_pad_token_id",
        "dspark_n_activated_experts": "dspark_num_experts_per_tok",
    }
    values = {
        key: value
        for key, value in text.items()
        if key in ModelArgs.__dataclass_fields__
    }
    values.update({dst: text[src] for dst, src in mapping.items()})
    rope = text["rope_scaling"]
    if rope["rope_type"] != "yarn":
        raise ValueError("expected YaRN compressed attention")
    values.update(
        original_seq_len=rope["original_max_position_embeddings"],
        rope_factor=rope["factor"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
    )
    vision = config["vision_config"]
    for dst, src in {
        "n_layers": "num_hidden_layers",
        "dim": "hidden_size",
        "n_heads": "num_attention_heads",
        "inter_dim": "intermediate_size",
        "patch_size": "patch_size",
        "rope_theta": "rope_theta",
        "downsample_ratio": "downsample_ratio",
        "max_n_token": "max_image_tokens",
        "min_pixels": "min_pixels",
        "max_wh_ratio": "max_wh_ratio",
    }.items():
        values["vision_" + dst] = vision[src]
    values.update(
        image_token_id=config["image_token_id"],
        dtype="fp8",
        expert_dtype="fp4",
        temperature=0,
        max_batch_size=1,
        max_seq_len=max_model_len,
    )
    if not 1 <= max_model_len <= text["max_position_embeddings"]:
        raise ValueError("context limit outside checkpoint bounds")
    return ModelArgs(**values)
