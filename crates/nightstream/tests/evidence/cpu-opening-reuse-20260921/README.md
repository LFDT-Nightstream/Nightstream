# CPU opening buffer reuse — 2026-09-21

This record covers the current worktree on an Apple M1 Max. It does not assert
that the full CPU lifecycle or the general memory target has passed.

The completed carried SumCheck allocation is transferred to opening scratch.
The scratch uses one interleaved `Vec<K>` instead of two new real-field planes.
Encoded witness folds also reuse their code buffers. Both allocation tests
failed before the fixes and passed afterward. Circuit, transcript, profile,
and saved Lean artifacts are unchanged.

| Production check | Result | Process seconds | Peak RSS, bytes |
| --- | --- | --- | --- |
| Opening reuse alone | Stopped in SumCheck at RSS guard | 74.83 | 17,478,352,896 |
| Opening and encoded-code reuse | Complete PiCCS proof matches saved CPU bytes | 183.03 | 17,103,896,576 |
| Decomposition openings | Seven active records match; nine zero outputs checked | 182.82 | 14,638,448,640 |

The passing PiCCS test itself took 182.43 s; the opening test took 182.02 s.
Process times above include the runner. All invocations used the normal
300-second cap and the stated 16 GiB RSS guard (17,179,869,184 bytes).
The CPU peak is 17.10 decimal GB, or 15.93 GiB. Physical footprint is diagnostic;
it is not the acceptance measure. No memory exception was used.

The opening test validates all sixteen canonical split witness matrices. Its
seven active opening files equal the saved CPU files byte for byte, including
commitments, parent, and transcript. Parent and digit inputs were reused;
this test did not produce a new complete C/R/D proof. See
[the exact comparison](opening-reuse-comparison.json).

The release checks passed: 23 cache-equivalence checks, three row/opening
checks, eight CPU unit checks, and nine engine comparisons. Two CPU tests that
need other local capture files and two CUDA comparisons remain ignored.
Two unrelated radix-four tests were filtered out. Clippy completed with
`--cap-lints warn`; inherited warnings remain, none on this slice's changed
code. Formatting and the diff whitespace check passed.

[checks.json](checks.json) records the results. The full CPU benchmark awaits
its separate timeout and memory approval. The earlier one-run CPU reference
memory exception is spent. Neither the 5× full lifecycle ratio nor the memory
bound for all supported circuits is established by these phase checks.

The final-build Metal benchmark also passed the complete three-step lifecycle:
179.514135833 s, with 16,784,228,352 bytes peak RSS and 16,580,063,808 bytes peak
physical footprint. This includes preparation (34.67 s), base proving (6.81 s),
two extends (49.75 s and 64.25 s), and terminal verification (24.04 s).
Inputs, circuit identity, and final state equal the earlier Metal measurement.
The executable is saved for the pending CPU run; it has `metal,cuda` features
and no performance-timer instrumentation or production `neo-fold-clean` edge.
This Metal timing cannot be compared with the shorter CPU PiCCS-only scope.
