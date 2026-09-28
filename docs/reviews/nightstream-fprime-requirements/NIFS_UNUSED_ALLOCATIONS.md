# Removal of unused retained allocations

This change removes the three unused logical blocks `piCcsPayload`,
`runningRoundC0`, and `runningRoundC1`. The isolated source is based on
`c21f4ac8f1a5bdbd23a20e93d970670c5addb6f0`. The physical layout, physical
witness program, public inputs, and Nightstream Goldilocks profile remain
unchanged: `b = 2`, `k_rho = 16`, `B = 65536`, one fresh source, 16 running
sources, 14 matrices, 54 coefficients, and the `2^28` cube.

| Derived quantity | Before | After |
|---|---:|---:|
| Retained assignment blocks | 33 | 30 |
| Logical columns | 254260583 | 253011231 |
| Complete carrier coefficients | 254260620 | 253011276 |
| Ajtai message ring columns | 4708530 | 4685394 |
| Final alignment zeros | 37 | 45 |
| Logical rows | 6377559 | 6377559 |
| Physical rows | 29225729 | 29225729 |

The logical reduction is `(30416 + 28 + 28) * 41 = 1249352`
coordinates. The complete carrier shrinks by 1249344 coefficients after
alignment. Assignment transport schema 2 has six fields, two source domains,
30 ordered blocks, and output-digest block 23. Lean and Rust both use this
format. The decoder rejects the old schema, old source-domain code, old
digest selector, missing blocks, changed block order, and extra fields.
Logical matrix columns follow the new layout. Their entries were compared
with the independently expanded Lean reference; unchanged nonzero counts
alone were not used as equality evidence.

The fixed seed and indexed expansion do not change. The new
`Poseidon2HashChainV1Setup.productionShortKernel_to_approvedMsis` definition
in [SetupBinding.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/SetupBinding.lean)
uses the checked [Ajtai prefix reduction](../../../formal/nightstream-fprime/NightstreamFPrime/Spec/AjtaiSetupV1/Prefix.lean).
It appends zeros after the complete selected carrier, preserves the exact
commitment and nonzero integer kernel, and keeps the strict norm bound
`113246208`. Its target is the original 254260620-coordinate fixed-key
instance in [PUBLIC_SEED_MSIS_ASSUMPTION.md](PUBLIC_SEED_MSIS_ASSUMPTION.md).
It does not restore removed interior coordinates or assume hardness of a
new matrix. The approved assumption still has no numerical success bound.

The complete logical matrix comparison passed for all 14 matrices, including
the empty matrix. Their nonzero counts remain
`[33616548,4650801,139699430,117473734,315481803,1535215502,33006078,1726758,32220219,30971178,30233769,31133970,30392685,0]`.
The complete physical matrix comparison also passed, with counts
`[93701820,39358148,28868018]`. Matrix mutation checks passed separately.

The first strict nonzero assignment check reached the 300-second cap and
was killed. It is a failed attempt, even though the timing wrapper printed
`Exit status: 0`; the signal and absent completion marker determine its
status. After the existing row evaluator was divided into independent row
ranges, the second full check passed in 280.51 seconds including compilation.
It checked all 6377559 logical rows, all 30 retained-block mutations, all
13 nonempty matrix-slot mutations, the empty slot, all 256 digest bits and
four digest words, and the Phi81, First54, and output-digest recipe mutations.
The complete base assignment check also passed, including all physical rows,
all logical coordinates, all logical rows, and the 45 alignment zeros.

| Gate | Retained result |
|---|---|
| Transport decoder and rejection test | Passed, 26.00 s including compilation |
| Complete logical matrices | Passed, 56.13 s |
| Matrix mutations | Passed, 76.13 s |
| Complete physical matrices | Passed, 26.54 s |
| Strict nonzero assignment | Attempt 1 capped at 300 s; attempt 2 passed, 280.51 s |
| Complete base assignment | Passed, 50.73 s |
| Sparse fixed-key commitment | Passed; all 22 by 54 output coefficients |
| Setup, base, ownership, pilot, C, R, D, and application fixture emission | Passed; raw logs retained |
| Nonzero C, cumulative C/R, and cumulative C/R/D parity | Lean and optimized Rust passed; exact handoffs, outputs, and rejection cases |
| Post-pin static, full library, and full axiom gates | Passed; library 180 s, axioms 45 s |
| Post-pin canonical identity | Passed, 99 s; structural, package, and verifier-key pins match |
| Post-pin production loader and binding | Passed, 8 + 3 tests; 110.42 s including 25.84 s compilation |
| Post-pin pilot binding | Passed, 27.32 s including compilation; test 21.11 s |

The phase checks use comparison fixtures. The optimized paths do not execute
PaperExact. They check every C output family, all 17 R sampler and partial
combination results, and all D digits, ranges, recompositions, child fields,
and final state. Normal R/D rejection checks are retained. These phase
checks are separate from an actual complete native C/R/D proof from executed
source witnesses.

The published package is an exact copy of the checked candidate. Its SHA-256
is `043bce25083eb15c733a903f4df2958acbc31a59dbe17420cd084c8242bd4d1f`.
All 11 installed setup, binding, expanded, base, ownership, and parity files
were also checked against their retained source bytes. In particular, the
installed setup fixture SHA-256 is
`90b14fd2019d829a628635345786f6b43b54f7fe083c3c65330065d083e194ba`.
This exact-copy evidence preserves the scope of the pre-pin checks; it is
distinct from running the verifier-owned loader against the new pins.

The [evidence archive](NIFS_UNUSED_ALLOCATIONS_EVIDENCE.zip) contains all
attempt logs, the source delta, source and artifact hashes, the comparison
fixtures, and the exact-copy manifest. It records large checked artifacts
by path, size, and hash instead of duplicating their expanded bytes.

This report changes no requirements-map status or approval. Full native
C/R/D proving under the new package remains open. The main Mat commitment
adapter still needs its post-merge endpoint test; the old C-only expected
record belongs to the prior package and is not post-pin evidence. Complete
selected witness construction, recursive closure, and numerical deployed
security are not claimed by this allocation change.
