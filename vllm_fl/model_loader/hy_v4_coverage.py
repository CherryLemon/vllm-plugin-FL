# SPDX-License-Identifier: Apache-2.0
"""Track the local checkpoint components of HY4 fused parameters."""


class HY4LoadCoverage:
    def __init__(self, model, params):
        self.expected = set()
        self.loaded = set()
        modules = dict(model.named_modules())
        for name in params:
            if ".experts.routed_experts.w" in name:
                owner = modules.get(name.rsplit(".", 1)[0])
                expert_map = getattr(owner, "expert_map", None)
                if expert_map is not None:
                    expert_ids = [
                        i for i, local in enumerate(expert_map.tolist()) if local >= 0
                    ]
                else:
                    expert_ids = getattr(
                        model,
                        "_hy4_local_expert_ids",
                        range(model.config.n_routed_experts),
                    )
                shards = ("w1", "w3") if ".w13_" in name else ("w2",)
                self.expected.update(
                    (name, expert, shard) for expert in expert_ids for shard in shards
                )
            elif ".gate_up_proj." in name or ".wk_weights_proj." in name:
                self.expected.update((name, None, shard) for shard in (0, 1))

    def record(self, name, shard, expert=None):
        key = (name, expert, shard)
        if key in self.expected:
            if key in self.loaded:
                raise ValueError(f"HY4 duplicate checkpoint component: {key}")
            self.loaded.add(key)

    def finish(self):
        missing = sorted(self.expected - self.loaded, key=str)
        if missing:
            raise ValueError(
                "HY4 checkpoint component coverage failed: "
                + "; ".join(
                    f"{name} expert={expert} shard={shard}"
                    for name, expert, shard in missing[:20]
                )
            )
