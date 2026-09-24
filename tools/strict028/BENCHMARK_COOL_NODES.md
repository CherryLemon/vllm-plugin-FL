# `.1` / `.3` H100 配对稳态 Decode 评测

2026-09-24 UTC。使用 `.1` 做 P（TP8/EP8）、`.3` 做 D（TP2×DP4/EP8），两端各 8 张 H100 80GB。两轮共用同一 P 服务和同一组 GPU，只切换 D 镜像：旧版 plugin `981d34b` / FlagGems `7c1e5b6`，候选版 plugin `41d54f3` / FlagGems `f5b3981`。基础镜像均为官方 `vllm/vllm-openai:v0.28.0-cu129`（构建提交 `2cf0a691`），容器内以 `vllm==0.28.0+empty` 覆盖安装；两端使用固定 FlagTree `dbf1842` 和 FlagCX `648a6c4`。候选版仅在已验证形状下把逐请求 FP32 mean/sum 改为 FlagGems 的合批归约。

80 路请求各有 131072 输入 token、请求 8192 输出 token，固定 greedy、DSpark 5。服务参数 `max_model_len=139264`；D 每个 DP 组容量 20。P 的 `FL_PD_HOST_SNAPSHOTS=80`，按 4096 token 调度前缀块。`--enforce-eager` 仅关闭 vLLM 自带的 host graph 管理；FL runner 的 target/draft CUDA Graph 已实际捕获和回放，8-rank trace 各每步 4 次 Graph launch。每个吞吐值取 4 个 DP 组各 20 路运行、0 等待、0 抢占，并预热至少 10 个 MTP step 后的约 60 秒**无 profiler**窗口；随后才采集 8-rank、各 10-step trace。两轮 prompt SHA256 相同，客户端 stream token 数与服务端 generation counter 完全一致。窗口结束即关闭流，**不是完整 8192 输出评测**。

| D 版本 | C80 窗口 | 每路平均 TPS | 每路中位 TPS | 总 TPS | C1 每路 TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧版 `981d34b` / `7c1e5b6` | 60.116 s | 4.8194 | 4.8157 | 385.55 | 5.9014 |
| 合批归约 `41d54f3` / `f5b3981` | 60.199 s | 5.3394 | 5.3240 | 427.15 | 6.5961 |
| 候选相对旧版 | — | **+10.79%** | — | **+10.79%** | **+11.77%** |

C1 保持同一 C80 容量配置，各取约 60 秒无 profiler 窗口；两轮输出的前 108 token SHA256 均与旧参考精确一致。8-rank 真实权重状态、hidden、selected ID 和后续 draft 准入均通过。

两轮在 D 的 60 秒 C80 窗口内，每 2 秒对 8 张卡采样温度、SM 时钟、利用率、功耗及降频原因：

| D 版本 | 温度范围 | 8 卡 SM 时钟 | 热降频采样 | 功率封顶采样 |
| --- | ---: | ---: | ---: | ---: |
| 旧版 | 36–42°C | 全部 1830 MHz | 0/216 | 0/216 |
| 合批归约 | 37–42°C | 全部 1830 MHz | 0/224 | 0/224 |

P 在两轮对应窗口均为 30–35°C、1830 MHz，无采样到的降频或功率封顶。2 秒采样不能排除更短的瞬时事件，但窗口中持续相同的 SM 频率与同卡配对结果说明此次 **10.79%** 收益不是换到更凉节点造成。候选版在此前 `.13/.68` 的同配置窗口为 5.3810 TPS/路，本次 5.3394 TPS/路，仅低 0.77%；C1 为 6.6232 对 6.5961 TPS，仅低 0.41%。

同节点 8-rank trace 中位数，旧版每步 132555.7 个 kernel、1015.428 ms GPU 时间跨度，候选版为 102075.7 个 kernel、913.227 ms，分别下降 22.99% 和 10.06%；GPU kernel busy 从 941.958 降至 852.591 ms，下降 9.49%。两轮每步均为 4 次 Graph launch。`.13/.68` 候选版 trace 为 102088.0 个 kernel、904.743 ms，和本次核数几乎相同。总体吞吐仍远低于同节点 SGLang hybrid 的约 76 TPS/路参考（`/root/sglang-plugin-FL/docker/dsv41/PROFILE_STEADY_DECODE.md`）；SGLang 是完整 8192 输出测量，这里的 vLLM 是有界稳态窗口，数值只作方向性对照。温度并不能解释这一量级差距，剩余的 MoE、小型 FP32 GEMV 和大量小 kernel 仍需单独优化。

原始数据位于 `/public-nvme/yjwu/dsv41-fl-028/campaign-steady/`：

- `benchmark/target-node1-node3-{rowreduce,baseline}-c80-profile.json`、`node1-node3-{rowreduce,baseline}-c1-unprofiled-window.json`：请求、占用率、输出、Graph 与状态检查。
- `benchmark/{candidate-retry,baseline}-node{1,3}-gpu-telemetry.csv`：两轮 2 秒 GPU 采样。
- `profiles/node1-node3-{rowreduce,baseline}/`：各 8 rank trace 与 receipt；`profiles/node1-node3-{rowreduce,baseline}-c80-10steps-summary.json`：每步汇总。
- `analysis/node1-node3-paired-benchmark.json`：窗口、温度和差值的可复算结构化摘要；`analysis/analyze_node1_node3.py`：汇总脚本。
- `analysis/node1-node3-{real-weight-admission,baseline-real-weight-admission}.json` 与 `node1-node3-post-run-health.json`：数值准入和服务闲置状态。
- `benchmark/launch_node1_p.sh`、`launch_node3_d.sh`、`launch_node3_d_baseline.sh`：固定参数。首次 P 启动误设 `FL_PD_HOST_SNAPSHOTS=0`，只产生 4 个 handoff 后队列停住；该尝试单独留档，修正为 80 后才开始上述有效测量。

测试结束已停止并移除本次新建的 P/D 容器，`.1` 和 `.3` 的 8 张卡分别回到约 4 MiB 和 0 MiB 占用；其它容器未改动。

剩余约 14 倍的 SGLang Decode 差距已进一步按 target/draft Graph、逐层调用数和 M=160 routed MoE 同卡算子对照拆解，见 [Decode 差距定位](DIAGNOSE_COOL_NODE_DECODE_GAP.md)。
