# Constraint reduction with the current PiRLC sampler

## Scope and acceptance

This ports the completed reductions from PR #121 onto
`nico/f-prime-constraints-cuda-formal` at
`fe9d3b4f0809cfc88a114ad263f0b328bee84385`, including PR #124.
The four-field total sampler, its 26 assignment blocks, and its soundness and
completeness obligations remain selected. The profile is Nightstream Goldilocks:
`b = 2`, `k_rho = 16`, `B = 65536`. Protocol binding uses Poseidon2.

Acceptance requires the completed quotient, direct-assignment and Poseidon
reductions; Lean proofs and axiom checks; independent matrix and assignment
checks before identity promotion; Rust rejection tests; and two fresh native
folds with complete Lean verifier, caller and physical-witness comparisons.
The final verifier also checks fresh rows and complete commitment openings.
The unfinished running-transition research is retained at its original scope.
It is not selected by the production layout.

## Resulting layout

| Measure | Base with PR #124 | This port | Reduction |
| --- | ---: | ---: | ---: |
| Logical coordinates | 242,590,792 | 173,939,080 | 68,651,712 |
| Padded coordinates | 242,590,842 | 173,939,130 | 68,651,712 |
| Commitment columns | 4,492,423 | 3,221,095 | 1,271,328 |
| Active rows | 6,064,606 | 4,131,470 | 1,933,136 |

The physical reference keeps 28,275,820 rows and 28,418,945 columns.
The logical assignment has 270 public coordinates and 50 alignment zeros.
The fixed maximum key remains 4,708,530 columns; only the selected prefix changes.

Each Phi81 product uses 108 distinct evaluation points to check a quotient
polynomial identity. This saves 1,674,432 rows. The quotient has degree at most
52; its last stored coefficient is zero. The Lean soundness proof also covers
arbitrary supplied quotients, rather than assuming the witness recipe.

Each retained Poseidon permutation uses 86 nonlinear rows. Its eight outputs
are derived from the final S-box values. Removing the eight output pin rows
from 32,338 invocations saves another 258,704 rows.

The fixed prepared envelope has 31,971,918 JSON nodes. At the existing key
capacity, the compiler permits 1,966,761 application witness and local fields.
The exact decoder bound is 33,938,686 nodes and 712,712,406 bytes. The existing
compiler-bound test constructs that boundary; these are measured and derived
bounds, not new profile choices.

## Proof and implementation links

- `Phi81Relation.QuotientProduct.equations_iff_identity`, `sound`, `complete`,
  and `exists_quotient_iff` connect the evaluation equations to the ring product.
  `quotientCoeff_eq_quotient` connects the executable coefficient recipe.
- `CanonicalDirectPhysicalExecution.events_safe`, `execute_agree`, and
  `successful_assignment_eq` prove scratch-read safety, equal failure behavior,
  and equal retained assignment and output digest. The sampler support covers
  the current wide reduction and canonical-word witnesses.
- `PoseidonRetainedRows.rowsZero_iff` and the retained matrix bridges connect the
  86-row representation to the same Poseidon result.
- Rust has an independent convolution and polynomial-division reference. Tests
  cover every basis product, full-field inputs, and a false product that passes
  the first 107 evaluation points but fails at the last point.

Direct CCS construction is enabled only after recomputing the selected
structural identity. Prepared loading retains cached identities as metadata
and uses full product checks. Supplied native application values still pass
through their existing checks.

## Validation

The complete Lean production build, test library, static checks and axiom audit
pass with stock Lean 4.32.2. The final audit covers 4,392 build jobs and uses only
`propext`, `Classical.choice`, and `Quot.sound`. The direct proof compiled after
making its read-bound lemma generic; no proof limit was raised.

Before pin promotion, the candidate passed these independent checks:

| Gate | Evidence | Seconds |
| --- | --- | ---: |
| Physical matrices | Complete ordered coefficient comparison | 30.11 |
| Logical matrices | All 14 matrices | 39.08 |
| Matrix mutations | Block order, in-range column and canonical coefficient changes rejected | 75.24 |
| Nonzero assignment | All 4,131,470 rows, 74 terms, 50 zeros, assignment/digest/recipe mutations | 219.68 |
| Base assignment | All physical rows and every logical coordinate | 83.59 |
| Detached application | Replacement satisfies its own rows; canonical relation rejects the detached suffix | 75.69 |
| Sparse commitment | All 1,188 coefficients of the selected sparse input | 0.03 |

The detached-application test first exposed an old row offset. It now derives
the range from the decoded canonical blocks. The sampler-writer rejection test
was also updated for the quotient recipe's new field positions.

Rust validation passes: the initial 70-test F-prime unit run; targeted checks
after the cached-identity fix, including its regression; direct/full assignment
and changed-package fallback; package, pilot, binding, component and setup
checks; all seven assembly tests; and the
selected-key prefix check. The cached-identity test fails before the fix and
passes afterward. The compiler-bound test also failed before updating the
assembler from the old 15-field product record to the 11-field quotient record.

The direct/full test compares the complete assignment and invalid-input errors.
Its paired witness-plus-assignment measurement is 1.54 seconds for full physical
execution and 1.43 seconds for direct execution. This is one component run,
not a whole-prover benchmark. The canonical package emitter took 41 seconds,
the physical reference 10 seconds, and the binding emitter 94 seconds.

All 22 fresh native phases pass: base; both C/R/D folds and successors;
iteration-three acceptance; rejection after changing and recommitting the fresh
public input; and balanced `Eval_K` and `Eval_A` mutations. Each balanced case
has a rebuilt valid fresh witness and preserves PiDEC weighted recomposition.
It reaches the corresponding complete-opening rejection.

The longest native phase took 147.92 seconds. Peak RSS was 9,887,068,160 bytes
(9.21 GiB), below the existing owner-approved 16 GiB guard. Each native command
retained the 300-second cap. Each Lean command retained the 1,500-second cap;
the fold coordinators also had an outer 1,500-second cap.

Both fresh Lean comparisons pass. For each fold they check the complete
ten-field C/R/D result, all 945,983 proof bytes, all 177,326 private and 278
public caller words, all seven caller result fields, and all 227,351,560
physical witness bytes. Lean and native mutation checks pass as well.

The checked first-fold result and caller replace the formal and native fixtures.
The refreshed `golden-v1.zip` contains exactly the 19 documented interface
files; every archived byte was compared with its checked source. It contains
no private witness matrices. The public state/message request is unchanged.
The four saved application tests and the retained complete-fold mutation test
also pass after fixture promotion.
See [the workflow](../../../scripts/GOLDEN_CONFORMANCE.md) and
[validation records](VALIDATION.json).

## Scope limits and review

These executions check the selected concrete package and inputs. They are not
a universal proof of Rust semantics. Native code supplies the C proof messages
for the Lean fold checks; independent Lean proof generation is a separate task.
Spartan was not run: `FPRIME_STAGE1_GOAL.md` excludes it from this work.

The retained running-transition reduction still lacks the aggregate production
layout connection. No production count or acceptance claim uses that research.
The September 4 external review targets the removed `neo-fold-clean` frontend;
that code is absent from this branch, so its finding does not apply to this port.
