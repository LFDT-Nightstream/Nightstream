# Selected constraint-reduction batch

The selected Lean layout has 3,588,191 logical CCS rows and 149,293,044
committed coordinates (149,292,999 logical coordinates plus 45 alignment
coordinates). It preserves the existing specification, Poseidon2, and the
Nightstream Goldilocks profile `b=2`, `k_rho=16`, `B=65536`.

| Measure | Published checkpoint `c9aa75b04` | Selected Lean | Reduction |
|---|---:|---:|---:|
| Logical CCS rows | 4,147,335 | 3,588,191 | 559,144 |
| Committed coordinates | 172,217,934 | 149,293,044 | 22,924,890 (13.31%) |
| Normalized matrix nonzero entries | 2,968,490,185 | 2,857,409,270 | 111,080,915 (3.74%) |

The candidate package, binding, setup, component parities, and base fixture
are installed with matching Rust dimensions and identity pins.
The complete independent 14-matrix comparison, physical matrix comparison,
and matrix mutation rejections pass. Complete assignment, derived-recipe rejection, base-step, and sparse
commitment checks pass. The actual NIFS proof and independent Lean C/R/D
comparison pass. Recursive assignment, mutation, and final consumer checks
also pass. Rust proving benchmarks remain paused. The extra 50% target of 92,179,782 coordinates is not achieved.

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

The direct Lean application witness computes and shares the three variable
permutation states. It places only their required S-box values in the
selected committed assignment. The unused ten-permutation source interval
is zero and is not computed. The full selected-assignment completion proof
now uses this construction. Its only caller discarded the former guarantee
that all old application R1CS rows held; that unused construction is removed.
The specified application step, full selected relation, witness values,
public input, output digest, and norm guarantees remain proved.

The physical source ABI has 28,674,023 R1CS rows and 28,792,719 columns.
Those physical counts describe the retained source interface, not the
selected CCS row count. Rust loads the smaller CCS relation, but its physical
application witness path still computes the thirteen source permutations.
This remaining source work is not counted as a runtime improvement.

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

The application matrix count is now 147,259 entries, down from 1,754,410
in the published package. The counter combines equal columns and removes
zero coefficients after every affine step over the exact Goldilocks field.
It reads the selected matrix operands and Poseidon2 constants from Lean.
It checks every old application row against the published application row
interval. See [application-matrix-cost.json](application-matrix-cost.json).

`ApplicationPoseidonMatrixExecution` proves that the executable metadata is
exactly the existing matrix program and registers that equality with the
Lean compiler. The full library and axiom build passes (4,262 jobs), and
static checks pass. The unsuccessful direct row counter expanded raw sparse
expression lists through partial rounds; those runs produced no usable count.
The replacement exports only the small exact operands and normalizes during
arithmetic.

Reproduce the application count by building `emitApplicationMatrixCost`
through `scripts/validate.sh`, running it through the `lean-executable` phase
with an output path, then running `application_matrix_cost.py` with that path,
the baseline package path, and a result path. The JSON records both input
hashes for provenance; they do not replace protocol verification.

The Lean batch is stable. The emitted candidate has 2,857,409,270 normalized
logical matrix entries. The independent Rust interpretation matches every
row of all 14 matrices. Physical A/B/C nonzeros are 93,238,030, 38,665,934,
and 28,343,420. Matrix block-order, in-range column, and nonzero coefficient
mutations are rejected under the selected identity. All logical assignment and recipe checks pass. The initial run exposed one
stale 31-coordinate padding expectation; the exact expectation is now 45,
and the complete rerun passes. Actual NIFS and recursive fixtures have been
regenerated. The published actual proof matches the full independent Lean
result; all 43 NIFS and 55 PiDEC mutation controls reject. All recursive
assignment and final consumer checks pass and are recorded in the metrics file. No overall proving
performance gain is claimed.

The shared-verifier manifest now distinguishes the selected compact application
suffix from the ordinary connector for other applications. Its selected matrix
and assignment metadata come from Lean. Rust preserves the selected suffix on
exact equality of the complete physical application plan and still checks the
original blueprint against the independent production pins. The separate
application assembly test also passes. The unselected generic Stage 1 decoder
fixture is unchanged historical test data; it is not part of this regeneration.

The final Lean production and test build passes (4,262 jobs), as does the
static boundary gate. Nineteen focused Rust tests pass across published NIFS,
assembly, lifecycle handoff and rejection, application parity, and Ajtai setup.
The independent recursive check covers every physical row, every logical
coordinate, all 3,588,191 logical rows, and all 45 alignment zeros. Its child
commitment, Eval_K, and Eval_A mutations are rejected by the canonical rows.
Proving benchmarks remain paused.
