# DeepSeek V4.1 Flash：C80 Graph Decode profiling

2026-09-23，`.13` Prefill / `.68` Decode 的 8-rank 实测已完成。**主要瓶颈是当前 FlagGems 路由 MoE 实现及其引起的 EP 等待；串行 MTP 验证进一步放大了成本。** 每步 GPU 跨度为 **5523.503 ms**。本轮采集诊断窗口后主动关闭请求，没有继续跑完整 8192 输出，也不报告完成请求的 TPS。

负载与本机 SGLang 参考记录对齐：80 路并发，每路 131072 输入、8192 请求输出，greedy、`ignore_eos=True`，DSpark block5。P 为 TP8/EP8；D 为 attention TP2 × DP4 / global EP8，8 张 H100 80GB，每个 DP 组保持 20 路。完整前缀通过 FlagCX 传入，Prefill 和状态传输不在采样窗口内。所有组无等待、无抢占；至少十步稳态预热后，每个 rank 采集十步，开启 CPU/CUDA activity、stack 和 shape。

全部 8 份 trace 与 receipt 校验通过，每 rank 都有 **60 次 target + 10 次 draft CUDA Graph launch**。官方 host 的 `--enforce-eager` 只关闭其自身 graph manager；这里实际使用 FL runner 的独立 CUDA Graph。采样结束后 80 条流均以 `profile_window_complete` 关闭，P/D 健康、运行及等待请求数为零。

下面每项为八个 rank 的中位数，百分比分母是 GPU kernel 时间和 5424.116 ms/step。GPU busy union 为 5424.112 ms/step，首末 kernel 跨度为 5523.503 ms/step；这三种量不等同于未采样吞吐。

| 路径 / kernel | ms/step | kernel 时间占比 | 调用/step |
| --- | ---: | ---: | ---: |
| 路由 MoE `_routed_mxfp4_mm` | 2598.593 | 47.91% | 486 |
| FP32 AllReduce | 2193.109 | 40.43% | 486 |
| 其他低精度线性层 `_block_scaled_lowp_mm` | 112.383 | 2.07% | 1789 |
| 稀疏注意力 `_sparse_sink_kernel` | 21.061 | 0.39% | 243 |
| 索引打分 `_index_scores` | 8.336 | 0.15% | 48 |

全模型约 **147513 个 kernel/step**。还存在大量按请求拆开的 GEMV、均值归约和拷贝；其中均值归约约 31300 次/step，小型 FP32 GEMV 约 19920 次/step。它们值得后续合批，但前两项已经合计占约 88.34%，应先处理。

**MoE 的源码解释与 trace 一致。** [FlagGems 实现](/public-nvme/yjwu/dsv41-fl-028/FlagGems/src/flag_gems/fused/block_scaled_mxfp4_moe.py:21) 以一个 token/expert pair 和一个 32 列输出块为 CTA。其 BF16 dot 使用 16 行矩阵，只有第一行有效，其余 15 行补零；K 维每 32 个元素解包 FP4 权重、计算 dot、应用缩放后累加。相同 expert 的多个 token 没有组成共享权重的 GEMM tile。非本 rank 的 expert CTA 写零后返回，仍在完整 launch grid 中。

实际 target gate/up grid 为 `[480,144,1]`，down 为 `[480,160,1]`，对应全局 80 token × top-6、H=5120、I=2304，每 rank 48 个本地 expert。源码能够确认上述计算组织方式；本次没有采集 NCU 硬件计数器，因此不把带宽、缓存命中率或 occupancy 当作已测结论。

**FP32 AllReduce 的大头是 EP 路径，并包含等待较慢 rank 的时间。** 按每层顺序和 launch grid 对齐，TP RowParallelLinear reduction 只有 2.897 ms/step，路由 MoE 后的 EP reduction 为 2190.227 ms/step。调用点分别是 [RowParallelLinear](/root/wt/dsv41-strict028/vllm_fl/strict028/models/deepseek_v41/model.py:303) 和 [MoE EP reduction](/root/wt/dsv41-strict028/vllm_fl/strict028/models/deepseek_v41/model.py:1193)。kernel 名称显示 NCCL device kernel，这是本次 FlagCX 路径实际调用的设备内核。

| rank | GPU 跨度 ms/step | 路由 MoE ms/step | FP32 AllReduce ms/step |
| --- | ---: | ---: | ---: |
| 0 | 5524.530 | 2667.262 | 2124.817 |
| 1 | 5523.575 | 2562.289 | 2229.593 |
| 2 | 5523.490 | 2806.471 | 1985.935 |
| 3 | 5523.480 | 2545.402 | 2246.270 |
| 4 | 5523.504 | 2583.031 | 2208.707 |
| 5 | 5523.503 | 2338.416 | 2452.965 |
| 6 | 5523.416 | 2791.148 | 2000.146 |
| 7 | 5523.520 | 2614.156 | 2177.511 |

各 rank 的 MoE 与 AllReduce 之和均约 4792 ms/step，而两者相互补偿。进一步逐层取八个 rank 中最慢的 MoE 耗时再相加，为 **4778.357 ms/step**；逐层取 rank 均值再相加仅为 **2613.522 ms/step**。这说明逐层计算不均衡对关键路径影响显著，不能将 2190 ms 直接解释成 FlagCX 传输开销。

跨 GPU 的 CUPTI 时间戳存在异常：2430 组 EP collective 中，137 组出现某 rank 结束早于最后一个 rank 开始，故整组从到达时差统计中排除。保留的 2293 组覆盖全部 EP kernel 时间的 95.20%；其中按 kernel 时间加权，99.45% 位于最后一个 rank 开始之前。此统计支持“等待较慢 rank”的判断，但期间可能已有部分通信，且异常组未纳入，**不能据此给出纯传输/纯等待的精确分解**。上述逐层 MoE 最大耗时只使用各 GPU 内部的 duration，不依赖跨 GPU 绝对时间对齐。

**MTP 验证是第二个结构性问题。** [worker 验证循环](/root/wt/dsv41-strict028/vllm_fl/strict028/worker.py:287) 对 draft 位置依次执行 target；C80 采样窗口中每步固定六次 target replay，再执行一次 draft。拒绝后的 lane 被屏蔽，但完整模型调用仍随全局有效请求继续。应在保留 Engram、compressor、window 状态及拒绝回滚语义的前提下实现多位置批量验证。

本机 [SGLang 参考报告](/root/sglang-plugin-FL/docker/dsv41/PROFILE_STEADY_DECODE.md:1) 使用相同模型、负载及 D 拓扑，hybrid 为 76.794 ms/step，vendor 为 63.602 ms/step；MoE CUTLASS GEMM 分别为 17.989、18.010 ms/step。当前 GPU 跨度约为 hybrid 的 **71.9 倍**。两者每步内部工作组织不同：当前七次 graph launch，参考两次。因此这是完整 Decode step 的对照，不能作为单次 MoE kernel 的等形状加速比；本轮也没有新的完整未采样 TPS 可供比较。

后续应按以下顺序推进：

1. 首先在 FlagGems 中实现按 expert 组织 token 的 grouped MXFP4 MoE，继续由 FlagTree 编译，减少补零计算和重复权重读取。先用实际路由/形状做算子数值与耗时对照，再验证真实 Graph replay、持久状态及 MTP 输出。
2. 合批多个 draft 位置的 target 验证，减少串行全模型执行；用拒绝、部分接受、跨状态边界的用例检查提交和回滚语义。
3. 重采同一 C80 十步窗口，观察 MoE、EP 到达时差和完整 step 的变化；如 EP 仍高，再隔离等尺寸 collective 测传输开销。当前证据不足以支持优先更换通信后端。
4. 主热点下降后，再处理 `_block_scaled_lowp_mm` 及大量小型 GEMV/归约。当前注意力耗时占比很小。

原始证据都在 `/public-nvme/yjwu/dsv41-fl-028/campaign-steady/`：

- [八 rank 汇总](</public-nvme/yjwu/dsv41-fl-028/campaign-steady/profiles/hostpinned-c80-10steps-summary.json>)；原始 trace/receipt 在 `profiles/hostpinned-c80-10steps/`，压缩 trace 总计约 526 MB。
- [请求及部署结果](</public-nvme/yjwu/dsv41-fl-028/campaign-steady/benchmark/target-hostpinned-c80-profile.json>)，`throughput_comparable=false`；占用连续记录为同名前缀的 `.metrics.jsonl`。
- [热点摘要](</public-nvme/yjwu/dsv41-fl-028/campaign-steady/analysis/hostpinned-hotspot-summary.json>)、[跨 rank 分析](</public-nvme/yjwu/dsv41-fl-028/campaign-steady/analysis/hostpinned-rank-arrival-analysis.json>)；复算脚本为 `analysis/extract_hot_timeline.py`、`analysis/analyze_rank_arrivals.py`。
- [采样后服务健康状态](</public-nvme/yjwu/dsv41-fl-028/campaign-steady/analysis/post-profile-service-health.json>)。

通用 profiler analyzer 的原表保存在 `analysis/hostpinned-c80-rank0-skill-triage.txt`。其名称启发式将 `_routed_mxfp4_mm` 归入 quantize；本报告依据源码将其归入 MoE GEMM，未修改工具原表。Graph 内部 kernel 的 Python stack 只映射到 replay 入口，以上内部算子归因结合了源码和实际 grid/调用序列。

此次还修复了采样可靠性问题：在 CUDA context 创建前将 CUPTI activity buffer 设置为 pinned host 分配。默认 device buffer 在低剩余显存、50K-node graph 的 70 次 replay 上复现 Xid 13；提前切换后同一压力场景导出全部 350 万 kernel，真实 C80 八 rank 采样也通过。第一次失败的模型采样没有可用 trace，未计入本报告。Profiler CPU 测试 8 项、实际 CUDA Graph 单测 1 项通过；CPU/GPU 投影 annotation 区分修复后，汇总测试 3 项及实际八份 trace 校验通过。

部署基于官方 `vllm/vllm-openai:v0.28.0-cu129`，vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a` 以 `0.28.0+empty` 安装，官方源码未改动。采样 D 镜像 `local/dsv41-fl:profile-hostpinned-0f62694`、plugin `0f62694`；FlagGems `b459958`、FlagTree `dbf184230982e2f7cbe6b91fa3ca1ea069833d21`、FlagCX `648a6c489d2d54870d173926eda2d0ea0771b39d`。本轮变更集中在 profiling/诊断工具，没有修改上述 MoE 和 MTP 数学路径。
