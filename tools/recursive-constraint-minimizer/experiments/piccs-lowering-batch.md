# Selected constraint-reduction batch

The selected Lean layout has 3,588,191 logical CCS rows and 149,293,044
committed coordinates (149,292,999 logical coordinates plus 45 alignment
coordinates). It preserves the existing specification, Poseidon2, and the
Nightstream Goldilocks profile `b=2`, `k_rho=16`, `B=65536`.

| Measure | Published checkpoint `c9aa75b04` | Selected Lean | Reduction |
|---|---:|---:|---:|
| Logical CCS rows | 4,147,335 | 3,588,191 | 559,144 |
| Committed coordinates | 172,217,934 | 149,293,044 | 22,924,890 (13.31%) |
| Normalized matrix nonzero entries | 2,968,490,185 | Not yet counted | Not claimed |

Packages, fixtures, and Rust still use the published checkpoint. These are
Lean layout counts, not Rust benchmark results. Rust proving benchmarks
remain paused. The extra 50% target of 92,179,782 coordinates is not achieved.

## Selected changes

The SumCheck compiler shares Horner evaluations and uses affine expressions
for evaluation at zero and one. Its chain changes from 424,657 to 2,324 rows.
The compact chain has 504 owned fields and 1,764 lowering scratch fields.
Sharing its final output also removes 5,115 rows from the final identity.

The gamma-power compiler uses 16 extension multiplications. The exponent
list is `1,2,3,6,12,24,27,54,108,216,432,864,1728,2592,5184,10368,12960`.
Lean checks each dependency and exponent equation. This changes the two
power chains from 124,402 to 144 rows. Together the PiCCS changes removed
551,706 rows and 22,619,952 committed coordinates at checkpoint `929d53686`.

The selected application uses three compact Poseidon2 permutations and four
digest pins: 262 rows instead of 7,700. It retains 258 S-box fields and four
message fields, for 10,742 coordinates. Its ten constant domain-tag blocks
are precomputed under the unchanged hash specification. This removes
another 7,438 rows and 304,938 committed coordinates, including alignment.

`Application.Program.compactHashChain` carries a proof of the exact existing
circuit. The compiler selects the compact row plan, matrix program, and
retained assignment block from that proof. Other application programs keep
the proved ordinary path. Input and output forms reuse the actual pilot
preimage coordinates.

The direct application witness computes and shares the three variable
permutation states. It places only their required S-box values in the
selected committed assignment. The unused ten-permutation source interval
is zero and is not computed. The full selected-assignment completion proof
now uses this construction. Its only caller discarded the former guarantee
that all old application R1CS rows held; that unused construction is removed.
The specified application step, full selected relation, witness values,
public input, output digest, and norm guarantees remain proved.

The physical source ABI still has 28,674,023 R1CS rows and 28,792,719 columns.
Those physical counts describe the retained source interface, not the
selected CCS row count. Rust witness execution still needs integration.

## Wide sampler merge

Branch `nico/pirlc-wide-sampler` at `919cd12b4` was reviewed and merged in
`b2c5a8cfd`. Its component proofs and 72 axiom audits pass. It is not selected
in F′ and contributes zero savings to the table above. The estimate of
about 0.13M sampler coordinates is not a proved selected-layout count.

Selecting it requires the deterministic sampler and transcript integration,
an executable witness hint, a retained layout with exact coordinate and
matrix counts, and package/Rust conformance. The merged V6 theorem compares
sampled acceptance with the existing uniform-challenge extractor; it does
not establish the complete new Fiat–Shamir lifecycle. See
[the sampler review](wide-sampler-review.md).

## Evidence and remaining work

Lean proves arbitrary-assignment soundness, constructive completeness,
source support, and witness mappings. The executable matrix program uses
the existing Poseidon and pin opcodes and has exact row correspondence.
Axiom gates are `tests/AxiomsCompactPiCCSLowering.lean` and
`tests/AxiomsApplicationSelection.lean`; only `propext`, `Classical.choice`,
and `Quot.sound` are permitted. The full batch check is recorded in
[piccs-lowering-metrics.json](piccs-lowering-metrics.json).

The five recorded cvc5 controls check the PiCCS identities and replay a
counterexample with a missing Horner link. They do not replace the Lean
proofs. Research sources include
[CLAP on expression sharing and witness correspondence](https://arxiv.org/html/2405.12115v2#S6.SS3)
and [cvc5 finite-field theory](https://cvc5.github.io/docs/latest/theories/finite_field.html).

Normalized matrix counts remain open. The earlier application counter
forced a large physical-layout computation and was stopped without a
result. A replacement must read only the changed blocks. No matrix-entry
saving or overall performance gain is claimed. After those counts and the
batch are stable, regenerate packages and fixtures and complete Rust
integration and conformance.
