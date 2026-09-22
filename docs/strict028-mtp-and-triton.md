# DeepSeek V4.1 Flash：DSpark MTP 与分支 Triton 迁移

这是 `27d43d3` 文本 Eager 基线之后的扩展。宿主仍是官方 vLLM
`v0.28.0` / `2cf0a6915ce5` 的 `VLLM_TARGET_DEVICE=empty` 安装，基础镜像仍为
`vllm/vllm-openai:v0.28.0-cu129`。插件保留 0.4.0-dev、main、#544 的合并历史。
PD 没有启用。

## MTP 的执行契约

- 使用 checkpoint 的三层 DSpark，读取主模型第 37、38、39 层的 attention 输入，
  每次生成五个 draft token；包含共享 embedding/head、Markov bias 与 confidence head。
- 配置为 `{"method":"dspark","num_speculative_tokens":5}`。只接受贪心采样，
  主模型与草稿共享 TP=8、原始权重和请求状态页。
- 每 rank 加载 13,386 个参数张量、67,366,074,352 参数字节；相比关闭 MTP 增加
  382 个参数张量。MTP 权重不再跳过，其他 rank 的专家仍按 EP 所有权跳过。
- vLLM 0.28 的 `SpeculativeConfig` 会从 checkpoint 重建草稿配置，并选择
  `DSparkDraftModel`。插件通过公开 `ModelRegistry.register_model` API 注册真实
  FL 架构及这两个配置别名；不修改宿主源码、Worker 方法或私有注册表。
- `Worker.take_draft_token_ids` 向宿主交付草稿，Runner 消费
  `scheduled_spec_decode_tokens` 并返回接受前缀及一个主模型采样 token。
  宿主继续负责 EOS、长度限制、统计、请求释放及重新调度。
- 当前按顺序验证目标 token，遇到第一个不匹配立即停止。只有确认的输入才进入
  Engram、压缩器和窗口缓存；草稿自己的 query KV 不写入目标上下文。
  DSpark 的三个窗口也属于宿主计费和隔离的请求状态页。
- Prefill 只初始化草稿上下文；第一步 decode 后开始提案。接近 256-token 上限，
  不足五个草稿位置时退回单 token 解码。

这是 MTP 正确性实现；尚未采用一次六行的并行 target verification，也没有声称
MTP 吞吐提升。后续并行验证需要同时设计 Engram/压缩状态的提交和回滚。

## 用户所指的“分支里写的 Triton”

范围为源分支 `a9e3d217cce0` 相对最后合入的官方上游父提交 `3918f3c5a3`。
直接与 vLLM 0.28 做差会混入之后的上游改动，因此不采用该口径。
工具 `branch_triton_inventory.py` 使用 AST 比较 `@triton.jit` 函数，得到
**6 个可启动 kernel、8 个设备辅助函数，共 14 个**。

| 源分支 kernel | FlagGems 对应实现 | 本次处理 | 整网是否调用该实现 |
|---|---|---|---|
| `_sm90_fp4_paged_index_logits_kernel` | `fused/DSA/mxfp4_mqa_logits.py::_mxfp4_paged_index_logits_kernel` | 前次已迁入；本次核对函数体 | 否 |
| `_sm90_fp4_grouped_paged_index_logits_kernel` | 同文件 `_mxfp4_grouped_paged_index_logits_kernel` | 前次已迁入，包括 group-6 的行归属及候选一致性检查 | 否 |
| `_sm90_fp4_workspace_index_logits_kernel` | 同文件 `_mxfp4_workspace_index_logits_kernel` | 前次已迁入 | 否 |
| `_finalize_candidate_topk_sm90_kernel` | `fused/DSA/finalize_candidate_topk.py::_finalize_candidate_topk_kernel` | 本次补入；公开 API `flag_gems.finalize_candidate_topk` | 否 |
| `_w8a8_block_fp8_matmul_hopper_static` | `runtime/backend/_nvidia/hopper/ops/w8a8_block_fp8_matmul_static.py` | 本次补入原生 FP8、K32 缩放、静态 shape 配置与 SWAP_AB | 否 |
| `_reduce_block_fp8_split_k` | 同文件同名函数 | 本次补入独立 FP32 split-K 归并，避免 atomic 累加 | 否 |

辅助函数中，`_e2m1_decode`、`_load_q_packed`、`_score_heads`、
`_sm90_fp4_group_load_keys`、`_sm90_fp4_group_score`、`_sm90_fp4_group_invalid`
已在 FlagGems 的 Indexer 中。除 E2M1 解码将 NVIDIA PTX 正零处理改为等价整数选择，
其余上述 Indexer kernel/辅助函数在去除名称、注释差异后与分支一致。

另两个辅助函数为新增的 `_fp32_to_e2m1_code_rne` 和修改的
`_fp32x2_to_fp4x2`。本次补入 `fused_indexer_q_rope_quant.py`，使 Hopper 可执行
软件 E2M1 RNE 打包，保留 Blackwell 的原有硬件转换分支。软件路径遵循源分支的有限
输入契约，舍入到零时使用正零；硬件路径保留其原有符号零编码。

**这些分支 Triton 代码现在均有 FlagGems 实现；入库和组件通过不等于整网已经接入。**
当前整网沿用显式申报的参考组合，实际调用 FlagGems 的
`block_scaled_lowp_linear`、`act_quant_triton`、`fp4_quantize_reference`、
`sparse_attention_with_sink`、`hc_split_sinkhorn_reference`。
量化权重保留 FP4/FP8 存储，Linear 仍是兼容 BF16 计算路径。

## 下一步应接入的部分

1. 将 Indexer 的 BF16 参考缓存切换到已验证的 packed MXFP4 布局，然后串联
   paged/workspace logits 与候选 top-k 映射；必须同时验证 cache writer/reader。
2. 在具备原生 FP8 的硬件上验证静态 GEMM 的整网数值，再通过能力分发选择该实现。
   非 FP8 卡继续使用兼容路径，不能无条件选择 Hopper kernel。
3. 六行并行验证与 group-6 K 重用由插件/FlagTree 接入；设备 kernel 归 FlagGems。
   当前串行验证不会触发 group-6。

分支 mHC split-H 是 **TileLang**；Marlin/底层 CUDA 修改、Python 候选选择、调度与
PD 修复也不属于上述 Triton 清单。它们保留在各自迁移清单中，不能计为这 6 个 kernel。

## 验证与复现

- `tests/strict028/test_mtp_scheduler.py`：拒绝位置 0～5、全部接受、裁剪草稿、长度
  边界、仅提交有效上下文、宿主回滚计数及一次性 draft 交付。
- `smoke_mtp.py`：TP=8 真权重，MTP 开关一致、交错请求、重复请求、EOS，三层草稿、
  logits、Markov/confidence 和窗口状态对发布者 `inference/model.py` 的逐项差分。
- `smoke_mtp_api.py`：最终镜像 HTTP 测试及正式 vLLM 开发 RPC 差分；覆盖普通输入、
  超过 128 的 prefill、解码跨 128、接近 256 上限，以及非法采样后的继续服务。
- FlagGems `test_dsv41_branch_triton.py` 与 `test_fused_indexer_q_rope_quant.py`：
  35 项通过，包含 FP4 tie/邻域、-inf/NaN/+inf、空输入、非连续行、候选映射、
  FP8 split/SWAP、真实线性 shape、确定性重跑和独立 kernel 的 graph replay。
  当前镜像未提供 Compute Sanitizer，因此未做 memcheck。

运行入口：

```bash
export FL_MODEL_PATH=/public-nvme/models/DeepSeek-V4.1-Flash
export FL_RUNTIME_IMAGE=<交付 image-build.json 中的 image_id>
bash commands/serve_mtp.sh
```

最终冻结版本、输出 token、接受数、逐 rank 差分和镜像审计见交付包 `evidence/`。
算子性能均为 `not_profiled`；没有非 NVIDIA、PD、模型 graph 或长上下文实测结论。
