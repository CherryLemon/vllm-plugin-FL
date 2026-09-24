# C80 Decode：grouped MoE、MTP 批量验证与 SGLang 对照

2026-09-24。本轮保留官方 vLLM 0.28.0+empty，修改 plugin 与 FlagGems，Triton 代码由 FlagTree 编译；FlagCX 和 P/D 拓扑保持原配置。测量口径为 80 × 131072 输入、8192 **请求输出**，DSpark5、greedy、ignore_eos、实际 CUDA Graph。诊断窗口主动结束请求，不是完成 8192 输出的吞吐评测。

真实权重两位置验证已通过全部 8 rank 的状态/hidden/ID/后续 draft 精确对照（7/8/9-token 前缀）。C80 未采样 60.107 秒窗口通过：每路中位数 5.066 TPS、均值 5.067 TPS，aggregate 405.396 TPS；0 排队、0 抢占。相对旧未完成长测的计数器均值 0.901 TPS/路，约提升 5.62 倍。C1 在同一 C80 容量部署上为 6.222 TPS；131072 长上下文的前 108 个 Decode token ID 与旧串行路径完全一致。8 rank 各十步 trace 均通过，GPU 跨度从 5523.503 降至 962.790 ms/step。

## 改动与数值约定

Plugin `a7d34dc` 增加 `VLLM_FL_BATCHED_VERIFY=1`：六个候选位置按层一起执行 dense/MoE；每层有状态 attention 仍按因果顺序推进。图内只保存被覆盖的缓存行，并逆序撤销拒绝后缀，保持 compressor、indexer、ring、Engram 与下一次 draft context 的提交语义。支持将六个位置合为一张图，或复用较小的位置图。真实 C80 启动时六位置图实例化 OOM，缓存回收后仍只有约 240 MiB 可用；`981d34b` 增加分段验证，三位置图可启动，但只剩约24MiB空间，诊断Prefill额外40MiB申请OOM；部署进一步选择两位置图复用三次，再执行一次 draft，保持 C80、MTP5 和上下文容量。原串行路径保留用于差分验证。

FlagGems `7c1e5b6` 将一个 token/expert pair 一个 CTA 改为 device 端按 expert 分组、压紧有效 tile，再调用 grouped GEMM。BN=128，BM 按 M 选择 16/32/64。保留原 K32 分块 dot、缩放和累加顺序，保留激活量化、clamp、路由权重施加位置与 expert 输出累加顺序。此轮不更改原数值约定。

测试：7 项 MoE 测试（均衡/集中/非本地路由、M80/480、Graph 内改变路由）；16 项 plugin 定向测试；FlagCX TP2×DP4/EP8 的 8 rank 小模型差分测试均通过（包含两位置与三位置分段）。覆盖完全/部分拒绝、接受六个位置、compressor/window 边界、页面重排、闲置 DP、完整状态和后续 draft。

最终运行时 plugin 为 `981d34b`，FlagGems 为 `7c1e5b6`，镜像 `local/dsv41-fl:opt-981d34b-7c1e5b6`，`VLLM_FL_BATCHED_VERIFY=1 VLLM_FL_VERIFY_WIDTH=2`。vLLM 源码 `2cf0a6915ce544dc493a0990f2ea38d81601128a` 保持 `0.28.0+empty`；FlagTree `dbf184230982e2f7cbe6b91fa3ca1ea069833d21`，FlagCX `648a6c489d2d54870d173926eda2d0ea0771b39d`。P 为 `.13` TP8/EP8，D 为 `.68` TP2×DP4/EP8、8×H100 80GB。

## 实际服务复测

| 未开启 profiler 的稳态窗口 | C80 | C1（固定 C80 容量） |
| --- | ---: | ---: |
| 观察时长 | 60.107 s | 60.427 s |
| 每路 TPS 中位数 | 5.066 | 6.222 |
| 每路 TPS 均值 | 5.067 | 6.222 |
| aggregate TPS | 405.396 | 6.222 |
| 排队 / 抢占 | 0 / 0 | 0 / 0 |

两次都使用 131072 输入、8192 请求输出，至少十个 MTP step 预热。窗口内 stream token 计数与 server counter 完全一致；C80 之后才启动 profiler，完成 8 rank receipt 后关闭流。C1 在 60 秒窗口结束后关闭流。**两次均未完成 8192 输出，不能作为完整请求评测。** C1 固定每 DP 20 个图槽位，空闲 DP 仍参与 EP，不代表专门调优的单请求部署。

真实权重短前缀准入在全部 8 rank 对照完整状态、hidden、selected ID 和下一次 draft；长上下文 C1 对照了与旧 C80 request0 相同 prompt 的前 108 个 Decode token，SHA256 为 `461da122189ffd25575509685dc086778e85e0bc4a1c56030a0946865a649fc7`，Prefill 首 token 也一致。这不是长序列完整状态差分；C80 本轮没有做输出 prefix hash 对照。此前 129-token 诊断 Prefill 在 width3 部署中因额外工作区 OOM，失败结果保留，不能算作通过。

8 rank 各十步 trace 中都有 **30 次 target + 10 次 draft** CUDA Graph replay，证实两位置验证图每步复用三次。以下均为各 rank 按 step 归一后取中位数；kernel 累计时间不是严格可加的关键路径分解。

| Trace 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| GPU 跨度，ms/step | 5523.503 | 962.790 |
| GPU kernel busy，ms/step | 5424.112 | 935.901 |
| Routed MoE GEMM，ms/step | 2598.593 | 282.954 |
| FP32 AllReduce（含等待），ms/step | 2193.109 | 67.550 |
| lowp GEMM，ms/step | 112.383 | 90.903 |
| 小 FP32 GEMV 主 kernel，ms/step | 91.836 | 93.450 |
| 主均值归约 kernel，ms/step | 87.751 | 87.754 |
| sparse attention，ms/step | 21.061 | 21.094 |
| GPU kernel 调用数/step | 147513.4 | 132555.9 |
| Graph replay 次数/step | 7 | 4 |

GPU 跨度改善 5.74 倍，与未采样 C80 的 5.62 倍吞吐改善方向一致。MoE 分组及验证分段同时落地，本轮没有生产模型单项消融，不能把收益精确拆给某一项。FlagCX 保持原版本；AllReduce 时间的大幅下降也支持旧瓶颈包含 rank 到达不均衡，并不表示网络带宽变快了。

## SGLang 实际参考算子

确认 SGLang `4cf6966f` 在 H100 上从 `Fp8Config` 分派到 `Mxfp4FlashinferCutlassMoEMethod`，使用 FlashInfer CUTLASS **W4A16**。并非 Blackwell TRTLLM，也不是 GPT-OSS 的 alpha/beta/clamp 配置。DeepSeek 路径使用 BF16 输入、[up;gate] 权重、alpha=1、beta=0、clamp=10、fused finalize。SGLang hybrid 明确保留这条 vendor MoE 路径。其 MTP 验证也将六个位置展开后做一次 target forward。

以下测试在同一张 `.68` H100 上先后执行，权重、输入、路由和路由权重的 SHA256 全部一致；只转换必要的权重布局。H=5120，I=2304，48 本地 / 384 全局 expert，top-6，EP8 rank0 的完整本地 MoE wrapper，Graph replay，预热 10 次、测量 40 次取中位数。包含路由组织、量化、两个 GEMM、激活、combine；不包含 EP 通信或 shared expert。输入和路由为合成数据。

| M / 路由 | grouped FlagGems ms | SGLang CUTLASS ms | 延迟比 |
| --- | ---: | ---: | ---: |
| 1 / 均衡 | 0.945 | 0.0527 | 17.9× |
| 1 / 集中 | 1.578 | 0.1109 | 14.2× |
| 80 / 均衡 | 3.726 | 0.3795 | 9.82× |
| 80 / 集中 | 1.947 | 0.3955 | 4.92× |
| 480 / 均衡 | 5.979 | 0.5245 | 11.40× |
| 480 / 集中 | 6.018 | 1.8010 | 3.34× |

“均衡”是从 384 个 expert 中均匀选 top-6；“集中”是所有 token 都路由到 rank0 的前六个 expert。新 FlagGems 在全部形状中与原 pair kernel 逐元素一致。前一组 BM/BN 扫描使用另一组固定随机输入：M80 集中路由从 82.744 ms 降到 1.948 ms；M480 集中路由从 495.848 ms 降到 6.019 ms。随机数据与上表不同，不能混合推导精确三方比值。

M80 对应旧单位置 target 的全 EP token 数，M480 对应六位置合批及 draft。最终 width2 部署的 target 为 M160，本轮没有单独测 M160，不能把表中某行当作最终 target 的精确算子时延。

FlagGems 保留 per-32 FP8 激活以及路由权重在 down GEMM 前施加；SGLang H100 使用 W4A16 及其原生 finalize/累加。两个原生输出在这组合成输入上的 relative L2 约 0.047，**没有将两者声明为数值可互换实现**。这张表显示剩余性能差距；它不把全部差距归因于编译器，也不证明改变精度即可保持模型输出。

运行时：FlagGems 测试为 Torch 2.13.0+cu129 / FlagTree Triton 3.7.1；SGLang 官方 0.5.18 镜像 `cb9401979086` 为 Torch 2.13.0+cu130 / FlashInfer 0.6.17，与参考部署的 FlashInfer 版本一致。FlagGems probe 通过 PYTHONPATH 载入 `7c1e5b6` 源码，已安装 distribution metadata 仍显示旧版本；实际源码路径、SHA 和此差别已写入结果。

## 为什么每路不到 1 TPS

旧 C80 记录的 server-counter 窗口约 72.09 aggregate TPS，即每路均值 0.901 TPS；不是完成请求的中位 TPS。旧 C1×16 功能测试约 0.763 TPS，也不是独立 C1 最优配置的稳态评测。

旧 profile 整轮统计中，每路每个验证 step 平均提交约 4.88–5.02 token。每步 GPU 跨度约 5.52 秒，所以 `约 5 token / 5.52 秒 ≈ 0.9 TPS` 与观测量级相符。接受率低不是这一轮低吞吐的主因；精确窗口不同，此处是量级核对。

除了 MoE 本身，还有这些执行问题：

- **串行验证**：原来每步六次完整 target，放大投影、路由和 EP collective 次数。本轮部署改为两位置合批、复用三次；每步 MoE GEMM 从 486 次降到 246 次，FP32 AllReduce 从 486 次降到 366 次。attention 的有状态操作仍按位置推进，尚未达到 SGLang 六位置一次 target forward 的组织方式。
- **逐层负载不均衡**：旧 EP AllReduce 为 2190 ms/step；逐层最慢 rank 的 MoE 合计 4778 ms，而平均 rank 仅 2614 ms。大量 collective 时间包含等待，不能都归为 FlagCX 传输成本。
- **大量逐请求小算子**：为保持先前准入的 FP32/BF16 reduction 顺序，`dense_linear`、`decode_mean/sum`、grouped output projection 按请求调用。旧每 step 约 147513 个 kernel，优化后仍有 132556 个；主均值归约 31300 次、小 FP32 GEMV 19920 次基本未变。SGLang hybrid 参考仅约 2836.2 个 kernel/step，优化后仍为它的 46.7 倍。Graph 消除了 Python 重复发射，但仍执行这些 GPU 节点，也需要实例化大量节点元数据。SGLang 将 mHC 统计按 batch 处理并融合 pre/post，本路径的小算子拆分需要继续处理。
- **固定图容量**：当前每 DP 固定 20 个槽位，C1 也填充到该容量；空槽不写状态，但仍执行不少模型计算。这是 C80 部署上的 C1 延迟，不能当成 M1 算子或独立 C1 部署性能。
- **CPU 同步与调度**：旧 rank0 的十步 CPU scope 中位数为 5383 ms，GPU kernel busy 为 5285 ms，约 98 ms 不在 GPU kernel 内。优化后分别为 961.100、933.245、27.088 ms（各指标独立取中位数）。当前 `cudaStreamSynchronize` CPU scope 约 525.212 ms、`cudaGraphLaunch` 约 427.973 ms，均与 GPU 工作重叠，不能再加到 step 上。图发射的 CPU 时间仍大，但这里没有数秒纯 CPU 空转的证据；27 ms 还包含 memcpy/memset 等，不能全归给 Python。

旧八 rank trace 的均值口径为 5523.5 ms GPU 跨度、5424.1 ms kernel busy；上述 rank0/逐 step **中位数**是另一种统计，不应直接相减混用。

## 与 SGLang 整步路径对照及剩余重点

同模型、C80、H100 数量和 D 并行布局的历史 SGLang hybrid trace 为 76.794 ms/step，vendor 为 63.602 ms；当前 plugin 为 962.790 ms，仍约为 hybrid 的 12.54 倍。SGLang 完整未采样请求中 hybrid 每路中位数 75.98/76.66 TPS、vendor 约 97.9 TPS；plugin 这里只有有限窗口，且两个实现的精度/累加约定不同，不能宣称这是严格同数值的框架胜负对照。

| 整步算子族，ms/step | 当前 plugin | SGLang hybrid | 解释 |
| --- | ---: | ---: | --- |
| Routed MoE 主 GEMM | 282.954（246 次） | 17.989（86 次） | 实现、精度约定及 verify 分组不同；隔离测量见上表 |
| lowp / W8A8 主 GEMM | 90.903（1420 次） | 20.708（230 次） | 不同位置合批与投影拆分；不能当作相同 shape 的单次倍率 |
| Sparse attention | 21.094（243 次） | 1.727（43 次） | plugin 按位置推进，SGLang 多 query 合并 |
| Indexer scores / FP4 scoring | 8.933（48 次） | 7.066（9 次） | plugin 普通打分与 SGLang 分组/普通两类合计，调度单位不同 |

因此后续优先级是：把逐请求的 dense/reduction 和 mHC 操作改成保持已验证数值行为的 batched/fused 实现，降低约 13 万图节点；再针对 H100 的 grouped MoE、lowp GEMM 调优，并研究多 query attention/完整六位置验证。在当前数值约定下，不能直接替换成 SGLang W4A16 后仅凭算子速度宣称完成。MoE 仍占约 30.2% 的 kernel 累计时间；即使把它完全消除，单项收益也不足以填平剩余整步差距。

本轮未采集 NCU 硬件计数器，不从时延推断带宽或 occupancy 上限。04:00 UTC 复查 P/D 都健康、请求归零；所有 rank 的 PD fatal_error 为空，P pending_sends 为 0。普通可恢复工作流，未声明 native Humanize RLCR 或 SOTA 达标。

## 证据

根目录：`/public-nvme/yjwu/dsv41-fl-028/campaign-steady/`。

- `kernel/moe-sglang-reference-comparison.json`、`moe-reference-*.json`：同形状同输入对照、原始 40 次时延和 SHA。
- `kernel/benchmark_sglang_moe_reference.py`：按 SGLang DeepSeek 实际 ABI 调用的重现脚本。
- `analysis/sglang-moe-verify-source-comparison.json`：源码依据和 SHA。
- `kernel/grouped-moe-*-tuning.json`：BM/BN 扫描；原始 pair kernel 保存为 `moe-reference-before-grouping.py`。
- `analysis/baseline-rank0-host-step-timing.json`：CPU step、GPU busy、runtime API 的非重复归因。
- `analysis/real-weight-verify-admission.json`：真实权重准入状态，必须以该结果为准。
- `benchmark/target-optimized-c80-profile.json`、`optimized-c1-unprofiled-window.json`：本轮未采样窗口、占用率和长前缀对照。
- `profiles/optimized-c80-10steps/`、`optimized-c80-10steps-summary.json`：8 rank 原始 trace、receipt 和汇总。
- `analysis/optimization-round1-comparison.json`、`optimized-rank0-host-step-timing.json`：优化前后及 SGLang 分类对照、CPU/GPU 时间。
- `analysis/optimized-decode-deployment.json`、`optimized-post-run-service-health.json`：部署版本与收尾健康检查。
- `kernel/width2-verify-tests.log`、`width2-parallel-verify.json`、`final-benchmark-summary-tests.log`：16 项 GPU 测试、8 rank 差分与 8 项 benchmark/trace 单测。
- `benchmark/run_admission_width2_controller.py`、`run_optimized_profile_resume.py`、`run_optimized_c1_window.py`：本轮控制脚本。它们依赖已启动部署和预填充 handoff；复跑前必须重新准备 handoff，不能重用本轮已消费的传输 ID。
