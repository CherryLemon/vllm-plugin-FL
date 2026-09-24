# `.1/.3` Decode 吞吐差距定位

2026-09-24。结论：当前 vLLM Plugin FL 与 SGLang hybrid 的约 **14.2–14.4 倍**每路稳态 TPS 差距主要发生在 D 端 target Graph。原因不是热降频或 Prefill 等待，而是六位置验证被拆成三次 target 回放、FlagGems routed MoE 较慢，以及按请求执行的大量 cuBLAS/ATen 小算子。这三项必须一起处理；只替换 MoE 主 GEMM 不足以追平。

两边均用 `.1` P、`.3` D，各 8 张 H100 80GB；D 为 TP2×DP4/EP8、C80，每个请求 131072 输入 token、请求 8192 输出 token、greedy/ignore EOS、DSpark5、实际 CUDA Graph。prompt SHA256 同为 `e3696762465c6a4503ffe8681de376575de9ff6fa91fe55cc14b60827c2009e9`。vLLM 取预热后的约 60 秒无 profiler 窗口，并未完成 8192 输出；SGLang 用完整输出的未采样轮次。两种数值实现、服务阶段长度也不完全相同，TPS 比值是方向性对照，算子 trace 则按相同的 10 个 MTP step 归一。

| 指标 | vLLM Plugin FL | SGLang hybrid |
| --- | ---: | ---: |
| 每路未采样 Decode TPS | 5.3394 | 75.98 / 76.66 |
| 8-rank trace GPU 跨度，ms/step 中位数 | 913.227 | 76.794 |
| GPU kernel 累计时间，ms/step 中位数 | 852.672 | 75.766 |
| GPU kernel 次数/step 中位数 | 102075.7 | 2836.2 |
| target + draft Graph replay/step | 3 + 1 | 1 + 1 |

vLLM GPU busy union 是 852.591 ms/step，即 913.227 ms 跨度的 93.4%。这说明数百毫秒的差距确实在 GPU 图内；就算完全消除图间的约 61 ms 空隙，也无法消除十几倍差距。`.1/.3` 配对测量期间 D 八卡均为 1830 MHz、37–42°C，2 秒采样未见热降频或功率封顶。

vLLM 四个 DP 组的 `1 + accepted_tokens / verified_steps` 为每路每步约 4.59–4.66 个输出 token；SGLang profile 报告的 accept len 约 4.6。两端计数定义和窗口未严格对齐，但接受数量同一量级，也不能解释 14 倍吞吐差。

## 逐层工作量

该模型有 40 个 target 层和 3 个 draft 层。vLLM 的 width=2 Graph 每步回放三次；SGLang 把六个位置展平成一次 target forward。两个 MoE GEMM/层，因此 vLLM 每步 `40×3×2 + 3×2 = 246` 个 `_grouped_routed_mxfp4_mm`，SGLang 为 `40×2 + 3×2 = 86` 个 CUTLASS GEMM。稀疏 attention 的 vLLM 路径仍逐位置执行：`40×6 + 3 = 243` 次，SGLang 为 `40 + 3 = 43` 次。前者的 width=6 图在 C80 容量下实例化 OOM；width=3 图虽可启动，但可用空间不足以接纳诊断 Prefill，因此稳定部署选择 width=2。当前归约优化后尚未重新证明 width=3/6 可用。

rank0 按 Graph ID 拆开的 10-step trace，vLLM target 图（含三次回放）每步 817.91 ms/99965 个 kernel，draft 图 35.20 ms/1998 个；SGLang target 图 70.08 ms/2498 个，draft 图 4.96 ms/230 个。图外 kernel 很少。不同框架的 Graph ID 是各自 trace 中的标识，这里的分组仅用于定位 target/draft，而不是同形状算子基准。

## GPU 时间去向

vLLM 下表是互斥的 kernel 名称分类，先按每个 rank/10 step 归一，再取 8 rank 中位数；各类别中位数不能作为严格关键路径相加。具体 cuBLAS 名称分类见原始 summary，FP32 GEMV/dot/reduce 与其它 cuBLAS 分开列出。

| vLLM kernel 类别 | 次数/step | GPU ms/step |
| --- | ---: | ---: |
| Routed MoE 主 GEMM | 246 | 282.00 |
| FP32 小型 GEMV、dot、reduce | 39160 | 133.87 |
| 其它 cuBLAS/cuBLASLt 投影 | 8460 | 121.98 |
| ATen elementwise/reduce/copy 等 | 47584 | 104.01 |
| FlagGems lowp dense GEMM | 1420 | 90.71 |
| NCCL collective | 567 | 80.84 |
| sparse attention + index score | 291 | 30.07 |
| 其余 plugin/FlagGems kernel | 4348 | 9.27 |

逐请求 FP32/cuBLAS 路径累计约 256 ms、47620 个 kernel；ATen 又有约 104 ms、47584 个 kernel。`dense_linear` 对 `_decode_batch_size` 的每个元素单独调用 `torch.nn.functional.linear`，`grouped_output_projection` 也按请求拆分。`decode_mean/sum` 已在已准入的 FP32 连续末维形状下使用 FlagGems 合批归约，使同卡 TPS 提升 10.79%，但未触及这些线性投影及大量 mHC/状态 elementwise。SGLang trace 中 ATen 只有约 623 次、4.15 ms/step；它的 mHC 统计/归约/combine/post 共 3.58 ms/step，不能把 vLLM 全部 cuBLAS 时间直接归因给 mHC，但调用量说明当前组合方式仍是结构性问题。

直接可识别的跨框架锚点：

| 算子 | vLLM 次数、ms/step | SGLang 次数、ms/step | 解释 |
| --- | ---: | ---: | --- |
| Routed MoE 主 GEMM | 246、282.00 | 86、17.99 | 次数为 2.86 倍，平均单次时延约 5.5 倍；两端 M 和算术约定不同 |
| lowp/W8A8 主 GEMM | 1420、90.71 | 230、20.71 | 调用量为 6.17 倍；不同投影拆分和 M，不能说 vLLM 单次更慢 |
| sparse attention | 243、21.14 | 43、1.73 | vLLM 按六个位置推进，SGLang 多 query 合并 |
| index/FP4 score | 48、8.94 | 9、7.07 | 总时间接近，不是首要瓶颈 |
| 通信 kernel | NCCL 567、80.84 | NCCL + custom 186、10.32 | collective 数量和路由不同；时间含等待，不能单独归罪 FlagCX |

MoE 主 GEMM 时间差约 264 ms，约占两边 kernel 累计时间差 777 ms 的 **34%**。即使仅按算术把 vLLM 的 282 ms 完全换成 SGLang 的 18 ms、其它保持不变，vLLM 仍约有 589 ms/step kernel 时间，约为 SGLang 全步的 7.8 倍。这不是可直接兑现的 TPS 预测，只说明 MoE 单项不够。

## 当前 M=160 MoE 单算子对照

在 `.3` 同一张 H100 GPU0 依次运行当前 FlagGems `f5b3981` 与 SGLang FlashInfer CUTLASS 参考算子。输入、MXFP4 权重、路由及路由权重的 SHA256 一致；Graph 回放预热 10 次、测量 40 次取中位数。H=5120、I=2304、48 本地/384 全局 expert、top-6；范围为 rank-local 完整 routed MoE wrapper，不包含 EP 通信和 shared expert。M=160 对应当前 width=2 的 target 全 EP token 数。

| M=160 路由 | FlagGems ms | CUTLASS ms | 倍率 |
| --- | ---: | ---: | ---: |
| 均衡（114 个本地 token/expert pair） | 4.6518 | 0.4821 | 9.65× |
| 集中（960 个本地 pair） | 3.3535 | 0.6128 | 5.47× |

FlagGems 对自身旧参考逐元素一致；两端原生输出 relative L2 约 0.046–0.047，并非数值可互换实现。FlagGems 对 FP4 权重逐 32 值解码为 BF16，做 `tl.dot`，并按 32 值缩放/FP32 顺序累加；SGLang H100 用 CUTLASS W4A16 和原生 fused finalize。隔离测量所用 FlashInfer 为 0.6.18，SGLang 服务 trace 的镜像并非这次隔离探针镜像；该测量证明算子级数量级差距，不替代同数值端到端 A/B。服务 trace 内两端 MoE 主 GEMM 的平均单次分别约 1.146/0.209 ms，与隔离测量方向相同，但 Graph 中的路由分布、输入 M 及 wrapper 边界不同。

## 下一步应解决的约束

1. 对逐请求 FP32 投影、mHC pre/post/归约及其 ATen 临时张量做 graph-safe 合批或融合，并用已有真实权重状态、hidden、selected ID 和长前缀对照守住数值约定。它们占约 360 ms/step、9.5 万次 kernel，是 width=6 图内存及节点数的主要优化入口之一。
2. 减少 target Graph 的三次回放并复测 C80 capture 容量；六位置合批必须保留有状态 attention、回滚和 draft context 的因果语义。此项可直接削减每层 MoE、低精度投影和 collective 的调用次数。
3. 优化 H100 routed MoE：当前每个 K32 子块循环解码 FP4 并做 BF16 dot，M=160 全 wrapper 比 CUTLASS 慢 5.5–9.7 倍。若考虑直接改用 W4A16，必须另设数值准入，因为当前两端输出不等价。
4. 稀疏 attention 多 query 合并与 collective 次数放在上述工作后验证；`index_scores` 总差仅约 1.9 ms/step，不应优先投入。

证据：`/public-nvme/yjwu/dsv41-fl-028/campaign-steady/profiles/node1-node3-rowreduce-c80-10steps-summary.json`、其 `profiles/node1-node3-rowreduce/.../rank0.json.gz`、`/public-nvme/yjwu/sglang-fl-0518-pd/node3/profiles/decode-op-gap-20260923a.json` 和 `hybrid-80x128k8192-20260923a/*.trace.json.gz`。M=160 的脚本与原始 40 次样本在 `/public-nvme/yjwu/dsv41-fl-028/campaign-steady/kernel/benchmark_sglang_moe_m160.py`、`moe-m160-reference-*.json`，汇总为 `moe-m160-comparison.json`。TPS、温度与版本记录见 [配对评测](BENCHMARK_COOL_NODES.md)，SGLang trace 口径见 `/root/sglang-plugin-FL/docker/dsv41/PROFILE_STEADY_DECODE.md`。
