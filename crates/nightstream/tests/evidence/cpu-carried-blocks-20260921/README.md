# CPU carried storage

Source: uncommitted worktree based on `b141ce1e677bd7ada72388330e7cc38d2de8534b`.
Host: Apple M1 Max, 64 GiB, macOS 26.6.2. No Lean command ran.

The final implementation computes combined witness projections one ring block
at a time and reuses one full projection buffer. It also folds dense tables
in place and releases completed SumCheck tables before openings.

The allocation tests failed before the corresponding changes and now pass.
They also compare values with scalar references and exercise odd lengths,
parallel chunk movement, zero blocks, and more rows than witness coordinates.
The row prover passes. Nine supported Nightstream engine checks pass; the two
CUDA checks remain ignored. This does not establish full production parity.

Three production CPU attempts used identical source files and the saved CPU
reference. All stopped at the normal 16 GiB RSS guard. The final attempt
completed setup and SumCheck, then stopped during openings at
20,684,931,072 bytes RSS. No final proof was saved. The first direct-ring-plane
approach was replaced by block-local projection in the final source.
The three-round project fuse ended this slice; opening storage remains open.

The extra `pi_ccs_v1_1_engine_parity` run had eight failures at the unchanged
empty-running-claim guard, before the changed evaluator. Its seven other
checks passed. The fixture and guard files are unchanged from `HEAD`.
The log is retained; no workspace-wide test pass is claimed.

Every native test used a 300-second cap. No memory exception was used.
The separate request for one 45-minute CPU benchmark and a memory exception
is still pending. The earlier one-run CPU-reference exception is spent.

The private run directory is:

`/Users/nijaar/Library/Application Support/Nightstream/runs/metal-norm-20260921-bgylh03h`

It retains all inputs and records. `in-place-prefix-cpu` is the final attempt's
directory. `cpu-storage-source.patch` and `cpu-storage-new-source.tar.gz`
preserve the final source. No successful `ccs.json` exists in these stopped
candidate directories. The reference in `second-fold-cpu` remains unchanged.

The old `direct-masks-benchmark-binary` predates these CPU changes. Use matching
final builds for a CPU/Metal timing comparison; do not use that old CPU path to
claim the final 5× result.
