# strict028 的 FlagCX TP 接入

本次只把 DeepSeek-V4.1-Flash 的 **同机 TP 张量通信** 接入 FlagCX；MTP 沿用既有
DSpark 串行验证。PD/KV transfer、跨节点异构通信和 graph capture 不在本次验收范围。

`VLLM_FL_TP_BACKEND=flagcx` 时，worker 以 Gloo 进程组交换 FlagCX unique ID，
模型中的五处 AllReduce 和一处 AllGather 通过插件内的
`strict028.collectives` 调用 `PyFlagcxCommunicator`，再调用 `libflagcx.so`。
`fl_tp_stats` 开发 RPC 返回每 rank 的后端、控制组以及实际调用次数。
如果 FlagCX 库缺失或初始化失败，worker 启动会失败，不会悄悄降级到 NCCL。
不设置该变量时仍使用原来的 NCCL 路径。

在 NVIDIA H100 上，FlagCX 的 native adaptor 底层使用 NCCL。这次验证的是
应用调用路径确实进入 FlagCX，不能把它解读为绕过 NCCL、异构互联或性能收益。

构建 `libflagcx.so` 使用固定 FlagCX 源提交 `648a6c489d2d54870d173926eda2d0ea0771b39d`
和官方 vLLM `v0.28.0-cu129` 镜像中的 CUDA 12.9 / NCCL；从别的 CUDA
版本复制 `.so` 会导致 `libcudart` ABI 不匹配。镜像构建上下文把 FlagCX 的
`plugin/interservice/flagcx_wrapper.py` 与编好的 `build/lib/libflagcx.so`
放进 `flagcx-runtime/`，并使用 `tools/strict028/Dockerfile.flagcx`。
镜像仍安装 `vllm==0.28.0+empty`，不包含官方 vLLM CUDA 扩展。

先用 `torchrun --nproc_per_node=8 tools/strict028/smoke_flagcx_tp.py` 验证
BF16/FP32 AllReduce 与 AllGather；然后按 `serve_mtp.sh` 启动 FlagCX 镜像，
用开发 RPC `fl_tp_stats` 和 MTP 输出/参考差分确认整网路径。FlagCX
库和插件版本以交付镜像的 manifest 为准。
