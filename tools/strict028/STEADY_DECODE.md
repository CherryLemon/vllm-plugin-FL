# DeepSeek V4.1 Flash steady Decode admission

The reference workload is the SGLang-FL `PROFILE_STEADY_DECODE.md` run, not the
older 32K/512 single-node benchmark: 80 concurrent requests, exactly 131072
input tokens and 8192 generated tokens each (`ignore_eos=True`). Prefill runs
on a separate TP8/EP8 node. Decode uses eight H100 80GB cards with global
EP8, attention TP2 × DP4, DSpark block 5 and CUDA Graph. A valid steady window
has 20 running requests in each DP group, no waiting or state transfer, and
Graph enabled. The reference profiler sampled ten Decode forward steps per
rank after reaching that state; throughput used separate, unprofiled rounds.

`benchmark_pd_decode.py` now defaults to this shape. A smaller request requires
`--smoke` and is labeled functional only. Before tokenizing 80 long prompts,
the script checks both servers' admitted context and four Decode DP metric
groups. During the burst it samples running and waiting requests, and rejects
a result without at least five samples showing all four groups at 20 active
requests and zero waiting. This prevents 80 queued clients from being reported
as 80-way steady Decode.

## Current 0.28.0 plugin admission

The `.13` Prefill / `.68` Decode deployment currently has max context 33792,
`max_num_seqs=16`, and TP8/DP1 on Decode. Its Worker executes one request at a
time across all eight ranks. The platform explicitly rejects DP>1 and context
above 33792. Therefore it cannot admit the reference workload, irrespective
of client concurrency.

The request-state allocation makes the capacity limit quantitative. The
plugin's meta-device `Transformer` plus `RequestState` computes 114065408 bytes
(108.78 MiB) per rank at the deployed 33792 context, exactly matching the live
FlagCX page size. At the required 139264 total-token context it computes
452419584 bytes (431.46 MiB) per request per rank. Eighty live pages plus the
scheduler's null page and one Graph scratch page would require 34.55 GiB per
rank. The real-weight loader reports 67366074352 bytes (62.74 GiB) of local
parameters per rank; these two allocations alone total 97.29 GiB per rank,
before activations, communication, Graph storage, and safety margin. They
cannot fit an H100 80GB card in TP8/DP1.

Attention TP2 × DP4 would reduce request pages to 20 per rank, but is not a
launch-flag change in this implementation. Model weight and activation shards,
FlagCX collective groups, the request-state manager, PD transfer mapping, and
the Worker scheduler all currently assume one homogeneous TP8 group. The
Decode runner also captures one graph per absolute sequence position, which
does not scale to an 8192-token output. The 128-input/16-output functional
request initially captured 15 target and four draft graphs in 21.05 seconds;
this first-use interval is not steady performance.

## Evidence and remaining work

- Source workload: `/root/sglang-plugin-FL/docker/dsv41/PROFILE_STEADY_DECODE.md`
  and `FLAGCX_PD.md` in the same directory.
- Functional PD Graph receipt before graph-pool isolation:
  `/public-nvme/yjwu/dsv41-fl-028/evidence/decode-graph-pd-smoke-128x16.json`.
  FlagCX transferred one page on every rank, and target/draft Graph replayed.
- Identical follow-up requests produced different content hashes with both
  shared and independent graph allocator pools; see the `repeat`, `repeat3`,
  and `decode-graph-isolated-repeat-128x16-c1-r3.json` receipts beside that
  first result. Independent pools were insufficient to fix the drift.
- The model's one-entry LRU caches for window and DSpark index tensors can
  evict a GPU tensor while a captured graph still retains its raw address.
  Graph-owned references to those tensors and a cross-position GPU regression
  test have been added; online repeatability is still under validation.

The next performance milestone requires a paged request-state layout at 128K,
batched Decode for 20 requests in each of four attention TP2 groups with
global EP8, cross-topology FlagCX PD transfer, and reusable Graph buckets that
do not specialize on every absolute position. Only after the four groups are
observed at 20 active requests can the two unprofiled 8192-token rounds and
ten-step per-rank profiler window be compared with SGLang-FL.
