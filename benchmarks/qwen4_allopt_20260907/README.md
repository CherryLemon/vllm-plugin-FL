# Qwen4 integration reproduction

The retained [final review](REVIEW2_20260920.md) and
[raw results](review2_validation.json) bind the September 20 H100 acceptance to
exact wheel hashes, source manifests, dependency commits, workloads and outcomes.
The September 30 code cleanup/refactor has no new hardware/performance acceptance.
Historical 335 model checks and 92 common checks are selected suites, not all CI.

The final review retains 1k/4k performance, BF16 repeatability and cross-policy
correctness results, initial failures, environment constraints and the required
[FlagGems deterministic MoE patch](flaggems_deterministic_moe.patch). Removing
superseded intermediate reports does not remove those acceptance boundaries;
the earlier records remain available in the Git history of this PR.

Reproduction tools:

- `identity_preflight.py`: runtime versions and source/wheel identity.
- `all_on_env.sh`, `launch_server.sh`: explicit optimization flags and server setup.
- `correctness_gate.py`: repeated request correctness, including acceptance failures.
- `plan_cache_gate.py`: plan-cache behavior and correctness.
- `run_e2e.sh`, `run_benchmark.sh`, `benchmark_summary.py`: full request workloads
  and summary generation. They write new results in the chosen output directory.

Read the final review for hardware, TP8, async/concurrency and dependency setup
before running these scripts. Shared worker acceptance and stock/eager/graph,
FULL graph restrictions are documented in
[the common report](../../docs/common_worker_review2.md).
