# Phi81 and Poseidon constraint reduction

## Scope

This ports the Phi81 quotient and Poseidon output-row reductions from PR #121
onto `nico/f-prime-constraints-cuda-formal` at
`fe9d3b4f0809cfc88a114ad263f0b328bee84385`, including PR #124.
It keeps the total four-field PiRLC sampler and all 26 assignment blocks.
The Nightstream Goldilocks profile uses `b = 2`, `k_rho = 16`, and `B = 65536`.
Protocol binding uses Poseidon2.

The prover uses the existing full physical witness execution and then constructs
the logical assignment. The optional product shortcut, its public methods and
mode flag, and its proof support have been removed. Unselected running-transition
research and its test roots have also been removed. The earlier commit retains
that work in Git history.

Acceptance requires sound and complete Lean reductions, unchanged package bytes
after the cleanup, complete matrix and assignment checks, the restored witness
path, and the relevant CI tests.

## Layout and matrix cost

| Measure | Base with PR #124 | This PR |
| --- | ---: | ---: |
| Logical coordinates | 242,590,792 | 173,939,080 |
| Padded coordinates | 242,590,842 | 173,939,130 |
| Commitment columns | 4,492,423 | 3,221,095 |
| Active rows | 6,064,606 | 4,131,470 |
| Logical matrix nonzeros | 2,304,743,568 | 2,970,234,034 |
| Sum-check rounds | 28 | 28 |

Rows decrease by 31.9%, and logical coordinates decrease by 28.3%.
Matrix nonzeros increase by 28.9%. The base nonzero total comes from the supplied
independent review; the current total matches our complete matrix comparison.
Slots 6 and 8–12 are now empty. Slot 13 remains the required zero matrix.
The sum-check domain stays at `2^28`; the row reduction does not reduce its
round count. Fewer active rows alone do not imply the same reduction in time.

The physical reference still has 28,275,820 rows and 28,418,945 columns.
The logical assignment has 270 public coordinates and 50 alignment zeros.
The fixed maximum key has 4,708,530 columns. Its application capacity grows from
292,329 to 1,966,761 witness and local fields.

The reviewer supplied these single-run Mac measurements at `d563417cd`:

| Phase | Base | Before scope cleanup |
| --- | ---: | ---: |
| PiCCS proving | 55.2 s | 47.8 s |
| PiCCS peak memory | 12.1 GiB | 10.2 GiB |
| Fresh commitment | 29.2 s | 28.0 s |

This update did not repeat that timing comparison. These are measurements from
the review, not a whole-prover speed guarantee for all applications.

## Proof and implementation

Each Phi81 ring product uses 108 distinct points to check a quotient polynomial
identity of degree at most 107. The rows constrain arbitrary supplied quotients.
The honest quotient has degree at most 52 and a zero last coefficient.
This change removes 1,674,432 rows and 68,651,712 logical coordinates.

`Phi81Relation.QuotientProduct.equations_iff_identity`, `sound`, `complete`,
and `exists_quotient_iff` connect the rows to the ring product.
`quotientCoeff_eq_quotient` connects the executable recipe. The Rust reference
uses independent convolution and polynomial division. Tests cover every basis
product, full-field values, and a false product that passes 107 points and fails
at the final point. CI now runs the quotient witness tests.

Each retained Poseidon permutation has 86 nonlinear rows. Its outputs are
derived from the final S-box values. Removing eight tautological output rows
from 32,338 invocations saves 258,704 rows.
`PoseidonRetainedRows.rowsZero_iff` and the matrix bridges prove equivalence.
All production quotient and Poseidon proof obligations remain in the axiom audit.

The matrix operand and recipe use quotient names. The obsolete per-lane decoder
and its unused audits are removed. The existing generic retained-block geometry
still uses a one-element product index.

## Validation

The report's original artifact checks are recorded at `d563417cd` in
[VALIDATION.json](VALIDATION.json): all physical coefficients and all 14 logical
matrices; complete nonzero and base assignments; detached-application, matrix,
recipe and digest mutations; setup and binding parity.

That run also passed 22 native phases, including both folds and successors,
final acceptance, and rebuilt false-opening rejection cases. Both folds matched
all ten Lean C/R/D result fields, 945,983 proof bytes, 177,326 private and 278
public caller words, all seven caller result fields, and 227,351,560 physical
witness bytes. Its longest native phase took 147.92 seconds, with 9.21 GiB peak
RSS. The 19-file golden archive contains the checked interfaces and no private
witness matrices. See [the workflow](../../../scripts/GOLDEN_CONFORMANCE.md).

After the scope cleanup, the complete production build and axiom audit pass
with stock Lean 4.32.2: 4,202 and 4,343 jobs. The audited axioms remain
`propext`, `Classical.choice`, and `Quot.sound`.
All 109 Python CI tests pass. The norm replay now derives
`INPUT_CODES = (CARRIER + 3) // 4`, giving 43,484,783 codes per source.
Its registration checks 739,241,311 encoded values and 11,827,860,976 decoded
bytes. The new complete-profile manifest regression fails before this fix and
passes afterward; it also rejects a truncated source.

The regenerated canonical package is byte-for-byte identical to the checked
artifact. The full Nightstream Rust suite passes: 45 tests and 12 documented
ignored tests. Both quotient witness tests pass. The complete independent
assignment and mutation test passes in 228.83 seconds.

Both successors were rebuilt from the checked native NIFS messages using full
witness execution. The first new successor supplies the second statement.
Each caller, physical witness, logical carrier, commitment, and state file
matches the earlier checked output byte for byte. These two phases took 59.48
and 65.53 seconds. No fixture or identity change was needed.

Each native test uses the 300-second cap; each Lean command
uses the 1,500-second cap. No proof limit, Cargo feature, environment setting,
hash family, or profile parameter is added.

## Limits

Native code supplies the C proof messages for the Lean comparisons. These are
concrete conformance checks, not independent Lean proof generation or a universal
proof of Rust semantics. The complete independent norm-prefix producer was not
rerun for this update; its registration and input framing were tested.
Spartan was not run.

Further matrix-count and 107-point reductions are outside this change.
The September 4 external report targets the removed `neo-fold-clean` frontend;
that finding does not apply to this branch.
