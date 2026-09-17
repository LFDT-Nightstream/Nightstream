# PiCCS native dot product

The existing `PiCCSWeightedBasis.dotK` now compiles to a native-word
accumulator. `PiCCSNativeDot.dotK_eq_spec` proves equality for arbitrary
vectors, source functions and lengths. The `dotK_eq_native` compiler rewrite
uses that theorem. Inputs and results keep the existing `K` type; the
protocol, transcript, package and source interpretation are unchanged.

The loop uses the already proved `NativePoseidon2RoundCore` Goldilocks
operations. It keeps both accumulator coefficients in `UInt64`, converts
source coefficients to words, and converts the result back to `K` once.
This removes intermediate big-integer field values and the mapped term list.
Both equality declarations are in `tests/AxiomsPiCCSClosure.lean`.

## Production-input measurement

Measured on the same Linux Ryzen 7 7700X host with 16 workers, one command
queue, no active subagents, and no file-cache flush. Both binaries use the
existing Lean runtime fork at `14f2fed9c9782048e896c924f424624585acd772`.
The baseline executable was saved from Nightstream `dfa4589d` before editing.

| Measurement | Baseline | Native dot |
| --- | ---: | ---: |
| Complete command | 48.24 s | 30.91 s |
| Range computation | 28.24 s | 10.46 s |
| Peak RSS | 2,455,360 KiB | 2,458,456 KiB |

The complete command time is 35.9% lower; computation time is 63.0% lower.
The measured range is the previously used production matrix-prefix range
`1518288..1534672`, with the same original witnesses and Lean-derived first
challenge. All 33 output files, totaling 898,651 bytes, match exactly by
direct byte comparison. A canonical coefficient changed by one fails the
comparison and names `1518288-1518800.jsonl`.

This measures one production-input range. It does not establish a speedup
for the complete recursive loop. Source loading and other computations
remain part of the command time. Runtime, compiler and IO retain their
existing implementation boundaries.

## Reproduction and gates

Retained evidence is outside Git, under
`nightstream-stage1-evidence/recursive-loop-6c3c8c0f.d84yc9ll/piccs-native-dot/`.
The committed `PICCS_NATIVE_DOT_PERFORMANCE.json` records the commands,
times, file sizes, delivery SHA-256 values, positive and negative checks.
The SHA-256 values record custody; they do not replace the byte comparison.

From `formal/nightstream-fprime`, build `replayPiCCSFirstRound`, then run
the saved baseline and the current executable through `scripts/validate.sh
lean-executable`. Pass `carried-matrix-prefix`, the same `public.json`,
`sources.jsonl`, `round-0.json`, a new output directory, and
`1518288 1534672`. Compare the complete directories with
`diff --recursive --no-dereference`.

The committed boundary test runs with:

```sh
scripts/validate.sh lean-executable lake env lean --run tests/PiCCSNativeDot.lean
```

It checks all real/imaginary pairs of zero, one, seven, 32-bit and 63-bit
boundaries, and the two largest residues, at lengths zero, one and 54:
19,683 complete comparisons. Its reference expands the original arithmetic
so the compiler rewrite cannot replace both sides.

Validation checkpoint: static, build, axioms and identity passed in that
order. Canonical binding, structural identity, package identity and
verifier-key pins match. The two new audited declarations use only
`propext` and `Quot.sound`.

The fresh replay completed `pad-after1-1` before the optimization took the
shared queue. The next command encountered the held queue before computation
and is retained in `queue-handoff/`; no completed output was discarded.
Resume requires an explicit source transition backed by the equality theorem,
byte comparison and gates. The full fresh-loop milestone remains open.
