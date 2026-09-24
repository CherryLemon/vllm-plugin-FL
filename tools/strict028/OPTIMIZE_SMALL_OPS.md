# Decode 小算子：保持归约顺序的请求合批

2026-09-24，接续 `OPTIMIZE_STEADY_DECODE.md` 的第一轮。固定官方 vLLM 0.28.0+empty、FlagTree、FlagCX、P `.13` TP8/EP8、D `.68` TP2×DP4/EP8、8×H100 80GB、两位置验证图与 C80 容量。本轮只改变 Decode mean/sum 的独立请求合批，不修改模型精度。

## 改动

FlagGems `f5b3981` 新增 `fused/decode_row_reduce.py`；plugin `41d54f3` 在 `decode_mean/decode_sum` 的已验证形状上显式调用。原来每个请求分别运行 ATen reduction 再 cat，现在一次 kernel launch 处理全部行。

每个请求的输出行数决定参考线程分组；四路 vector accumulator 的顺序、先 block-x 后 block-y 的降序归约树，以及 mean 最后的 FP32 乘法全部保留。不能直接使用默认 `tl.sum`：第一版在 224 组中有 132 组出现差异，已拒绝并保存结果；显式归约树版本 224 组全部精确通过。适用范围是对齐、连续 FP32、最后一维 K 为 128..32768 且为 4 的倍数；其它形状保留原路径。单请求参考路径也保留，`VLLM_FL_BATCHED_REDUCTIONS=0` 可关闭新分派。kernel 在 FlagGems，编译器使用固定 FlagTree。

输入的平方、norm 后处理等仍各自物化，本轮没有改变这些舍入边界；也没有把全部归约性能差异归因于数学计算能力。

## 验证及隔离测量

- 224 组形状/分布探针逐元素精确一致；93 项算子测试通过，含尾部、特殊值和改变输入的 Graph replay。
- 18 项 plugin 小模型 Graph/状态测试通过，包含验证拒绝、页重排、compressor/window 边界与后续 draft。
- TP2×DP4/EP8 的 8 rank FlagCX 小模型状态、hidden 和空闲 DP 对照通过。
- 直接改用 `torch.bmm` 的 dense 投影探针在 21 个形状中有 18 个不满足精确对照，未纳入部署。

同一张 `.13` H100 上，以原逐请求 FP32 reduction+cat 为参考，CUDA Graph 预热 30 次、50 组各重放 10 次，取每次时延中位数；输入为合成数据，完整 wrapper 计时：

| 请求数 × 每请求行数 × K | 原路径 µs | FlagGems µs | 加速 |
| --- | ---: | ---: | ---: |
| 20 × 1 × 512 | 38.816 | 4.947 | 7.85× |
| 20 × 1 × 1280 | 43.429 | 4.934 | 8.80× |
| 20 × 1 × 5120 | 49.978 | 4.872 | 10.26× |
| 40 × 1 × 20480 | 148.613 | 4.834 | 30.75× |
| 80 × 1 × 20480 | 293.933 | 4.922 | 59.72× |
| 20 × 6 × 5120 | 77.142 | 4.874 | 15.83× |
| 20 × 6 × 20480 | 82.232 | 5.158 | 15.94× |

这是算子测量，不是服务 TPS；图中批量归约替换了原来 B 个 reduction 加 cat 的 launch，读取的数据和算术契约相同。单卡测试使用同一 Torch 2.13+cu129 和固定 FlagTree，临时 GPU 测试已结束后才开始服务 Prefill。

## 实际模型复测

真实权重短前缀准入通过全部 8 rank（25.643 秒），完整状态、hidden、selected ID、后续 draft 精确一致。C80 的 60.532 秒未采样窗口通过：每路均值 5.38100 TPS、中位数 5.35251 TPS，总计 430.480 TPS，stream/server counter 完全一致；0 排队、0 抢占。相对第一轮同配置均值 5.06745 TPS 提升 6.19%。相同 C80 容量配置的 C1 未采样 60.092 秒窗口为 6.62317 TPS，相对第一轮 6.22242 TPS 提升 6.44%；前 108 个输出 token 的 SHA256 与第一轮相同。所有请求仍设置 8192 输出，但窗口结束后关闭流，不作为完整输出评测；不混用最初的 0.9 TPS 基线。

8 rank 各取 10 个稳态 Decode step，均为每步 3 次 target、1 次 draft Graph replay。跨 rank 中位数的 kernel 数由每步 132555.9 降至 102088.0（减 22.98%），GPU kernel 时间跨度由 962.790 降至 904.743 ms（减 6.03%）。原 mean 归约由每步 31660 次、88.525 ms 降至 360 次、0.765 ms；新增合批归约每步 1073 次、2.392 ms。原有小型 FP32 GEMV 仍约 19920 次、93.44 ms，MoE 仍约 246 次、279.40 ms，说明本轮没有解决剩余主要算子耗时。rank0 采样 CPU step 中位数 903.31 ms，其中 GPU kernel busy 844.94 ms；CPU 同步调用包含等待 GPU 的时间，不能作为额外 Python 开销相加。以上 trace 只作瓶颈归因，TPS 采用未开启 profiler 的窗口。

## 证据

根目录 `/public-nvme/yjwu/dsv41-fl-028/campaign-steady/`：

- `analysis/small-ops-fusion-dossier.md`：语义、形状、物化边界及预期收益。
- `kernel/decode-row-reduce-probe.json`、`decode-row-reduce-tests.log`、`row-reduce-model-tests.log`、`row-reduce-parallel-verify.json`：数值与状态测试。
- `kernel/decode-row-reduce-plain-sum-failed.json`、`batched-dense-association.json`：被拒绝的实现。
- `kernel/decode-row-reduce-benchmark.json`、`benchmark_decode_row_reduce.py`：隔离测试和原始时延样本。
- `analysis/rowreduce-decode-deployment.json`：镜像、commit、命令和旧容器。
- `benchmark/run_rowreduce_controller.py`、`run_rowreduce_profile.py`、`run_rowreduce_c1_window.py`：有界复测脚本。
- `benchmark/target-rowreduce-c80-profile.json`、`benchmark/rowreduce-c1-unprofiled-window.json`：C80/C1 原始请求、计数器和窗口。
- `profiles/rowreduce-c80-10steps-summary.json`、`analysis/optimization-round2-comparison.json`、`analysis/rowreduce-rank0-host-step-timing.json`：八 rank trace 与第一轮对照。
