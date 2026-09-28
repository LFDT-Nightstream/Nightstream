# PiDEC sparse replay performance

The existing PiDEC matrix replay computes the same values with a native-word
accumulator and a reader that the compiler can specialize. The protocol,
matrix rows, input checks, child order and output encoding are unchanged.

`PiDECNativeSparseEvaluation.nativeEvalSparse_eq_spec` proves equality with
`SparseForm.evalSparse` for arbitrary forms and reads. It includes duplicate
entries, zero coefficients and cancellation, with no additional premise.
`PiDECMatrixSparseRange.sum_value` keeps its statement and consumes this proof.
The new theorem is included in `AxiomsStage1PiDEC`.

The executable constructs the parent reader before specializing each selected
evaluator. Runtime dispatch receives a child index. Previously it received a
reader function. The generated sparse loop now calls the same parent reader
directly and has no generic closure application. Before this change, sampled
CPU time after loading included 36.68% in `apply_1_slow` and 33.68% in
`lean_dec_ref_cold`.

## Measured result

These are complete range times after shared input loading, on the same Linux
host with 16 workers and Lean runtime `a6f47234088e5b35c7a5b1b336db6372f797d998`.
All subagents were idle during the measurements.

| Range | Before | After |
| --- | ---: | ---: |
| Ordinary block 4, rows 0–1330 | 8.005 s | 4.409 s |
| Phi81 block 10, rows 842724–850068 | 21.871 s | 4.776 s |
| Poseidon block 1, one complete 94-row invocation | 0.209 s | 0.116 s |

The two sparse ranges together fell from 29.876 to 9.185 seconds, a 69.3%
reduction. All 442,604 output bytes match, including 72,576 matrix field words
and the point, bounds and other metadata. The ordinary and Phi81 outputs also
match their retained reference files.

This is not a full-loop speedup. Parent loading varied from 73.01 to 92.01
seconds and dominates these small samples. The optimized three-range command
peaked at 3,470,116 KiB RSS. The native accumulator alone had a smaller measured
effect: the two sparse ranges took 28.552 seconds before reader specialization.

The isolated equality proof checks in 2.07 seconds. It uses the existing
`List.foldl`; a draft with a new recursive equation was discarded after slow
kernel conversion. No recursion or heartbeat override was added.

Static, library build, axiom audit and package identity passed in order.
The canonical binding, structural identity, package identity and verifier-key
pins match. Independent source review found no required fix.

## Evidence and replay scope

`PIDEC_NATIVE_SPARSE_PERFORMANCE.json` records source hashes, range results,
output hashes and command/log locations. Raw evidence is outside Git in the
fresh run's `native-sparse-evaluation/` directory. Inputs are the complete
original Lean PiRLC parent and accepted C execution named by the commands in
`PIDEC_MATRIX_RANGES.json`. These older inputs are used only for this benchmark.
They do not supply intermediates to the fresh recursive loop.

The fresh-loop target remains `LeanGraph.Targets.CheckedRecursiveReplay`.
The new evaluator refines its existing PiDEC dependency. Both fresh folds,
their complete comparisons and final terminal checks remain required.
