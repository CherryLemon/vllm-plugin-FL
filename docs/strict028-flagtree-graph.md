# FlagTree compiler and Decode Graph admission

The strict vLLM 0.28.0 profile retains the official `vllm-0.28.0+empty`
installation. Its H100 PD image now selects FlagTree as the `triton` module via
`PYTHONPATH=/opt/flagtree`; the official Triton installation remains in the
base image. FlagTree is the compiler for both FlagGems and any remaining Triton
callers. The six branch-authored launch kernels and eight helpers were already
migrated into FlagGems; changing the compiler does not change that operator
ownership or connect the still-unused packed Indexer kernels to the model.

The NVIDIA cp312 wheel is from `flagos-ai/FlagTree`, branch `triton_v3.7.x`,
commit `dbf184230982e2f7cbe6b91fa3ca1ea069833d21`, version
`0.6.0+nv37.dbf18423`, Triton API `3.7.1`. Its filename is
`flagtree-0.6.0+nv37.dbf18423-cp312-cp312-linux_x86_64.whl` and SHA256 is
`14e12e5c33864b37ebad2f1b654546a1472d6abee824633c1120952e30686adf`.
It is the same locked artifact used by the local SGLang-FL image. Put it under
`flagtree/` in the Docker build context alongside `wheels/`,
`flagcx-runtime/` and `deployment-manifest.json`. Build with
`tools/strict028/Dockerfile.flagcx`; use `Dockerfile.flagcx.pip` on Docker hosts
whose imported base image lacks `uv`. Both variants check the wheel hash and
active compiler path during the build.

`tools/strict028/smoke_flagtree_graph.py` runs a real FlagGems `act_quant`
kernel through FlagTree on H100, captures it, mutates the static input buffer,
and compares two graph replays against eager results. Both replays matched
exactly. `FL_TEST_CUDA_GRAPH=1` in `smoke_flagcx_tp.py` separately captures and
replays the eight-rank FlagCX TP all-reduce and all-gather wrappers; all ranks
passed twice. These checks establish compiler and collective graph support,
not a serving Decode graph.

The current `ModelRunnerFL028` still executes each scheduled request and each
MTP target verification token sequentially. `WorkerFL028.compile_or_warm_up_model`
only runs Eager forwards, and `serve_pd.sh` forces `--enforce-eager`. The model
also uses `start_pos` as a Python integer to select cache slices, compression
branches and top-k widths; `RequestState.bind` changes GPU buffer addresses per
request page. A useful Decode graph needs device-side position/page metadata,
stable storage and a fixed kernel sequence before an end-to-end 32K/512/C1,4,16
steady benchmark can be reported. `fl_cuda_graph_probe` in the validation Worker
extension checks whether one fixed-position target step can be captured; it
does not enable serving replay.
