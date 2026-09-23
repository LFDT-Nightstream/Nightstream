# Compact PiCCS lowering batch

The owner authorized this implementation on 2026-09-23. Preserve checkpoint
`c9aa75b04`, the complete specification, Poseidon2, and the Nightstream profile
`b=2`, `k_rho=16`, `B=65536`. Batch Lean changes before artifact regeneration.
Rust proving benchmarks remain paused.

## Proved components

| Component | Current rows | Compact rows | Current local coordinates | Compact local coordinates |
|---|---:|---:|---:|---:|
| SumCheck chain | 424,657 | 2,324 | 17,408,641 | 92,988 |
| Gamma powers | 124,402 | 144 | 5,100,482 | 5,904 |

The SumCheck construction uses the existing causal Horner compiler. It
replaces evaluation at zero and one with affine expressions, materializes
each challenge evaluation once, and provides one shared final output. Lean
proves equivalence for arbitrary satisfying assignments, causal witness
execution, preservation of external values, and the exact local R1CS counts.
The compact block has 504 owned fields and 1,764 lowering scratch fields.

The gamma construction uses 16 extension multiplications. Its exponent list
is `1,2,3,6,12,24,27,54,108,216,432,864,1728,2592,5184,10368,12960`.
Lean verifies every dependency and exponent equation and proves arbitrary-
assignment soundness and constructive completeness. It has 32 owned fields
and 112 lowering scratch fields. The prior 4.0M-coordinate estimate omitted
the old power circuits' 27,648 owned fields.

The application prefix theorem proves that precomputing the ten full blocks
of the fixed 40-word domain tag gives the exact existing application hash.
The two variable absorption blocks and finalization remain. The compact
application matrix plan has soundness and a constructive field-witness
encoding proof. Its retained layout and selected-package mapping remain open.

Focused axiom audit: `tests/AxiomsCompactPiCCSLowering.lean`. All audited
theorems use only `propext`, `Classical.choice`, and `Quot.sound`.

## Solver controls

Run `timeout --signal=KILL 300 python3 tools/recursive-constraint-minimizer/experiments/piccs_lowering_controls.py`.
The Goldilocks quadratic-extension checks use prime-field cvc5 with the GB
solver. Boolean-sum simplification and materialized Horner equivalence are
`unsat` for a counterexample. Omitting the final Horner link admits the
recorded false assignment: zero inputs and an unlinked product `(1,0)`.
The explicit assignment is replayed. These controls do not replace Lean.

## Selected Lean layout after the PiCCS changes

`Poseidon2HashChainV1Package` and `Poseidon2HashChainV1Setup` now build with
both PiCCS changes selected. Their checked values are 3,595,629 logical CCS
rows and 149,597,982 committed coordinates (149,597,957 logical plus 25
alignment coordinates). The physical package has 28,674,023 R1CS rows and
28,792,719 columns. Artifacts and Rust still use the published checkpoint.

The PiCCS arithmetic packet has 259,963 rows, 52,734 owned logical fields,
and 207,011 R1CS scratch fields. Soundness, constructive completeness,
retained source support, and the packet witness mappings build. Sharing the
last SumCheck output removes another 5,115 rows and 209,715 coordinates from
the final identity beyond the two component reductions above.

The application candidate has a checked 262-row CCS plan. Its soundness
proof connects all three compact permutations and all four digest pins to
the unchanged `Poseidon2HashChainV1.step`. Its field-witness constructor and encoding-correctness theorem also pass.
The constructor uses the existing canonical permutation executor for only
three permutations. Retained-layout integration and package selection remain
open. Its 10,742-coordinate formula
counts four message fields and 258 S-box fields; it is not a selected-package
measurement.

## Remaining integration

Complete the application witness construction and package selection, then
repeat the affected Lean and axiom gates. The current production and test
targets pass together (4,235 jobs), and the static boundary gate passes. Count the changed normalized matrix
entries and regenerate artifacts once the complete batch is stable. Rust
integration and conformance remain open; Rust proving benchmarks remain
paused. No matrix-entry reduction or overall performance gain is claimed.

Research sources: [CLAP common-expression elimination and witness correspondence](https://arxiv.org/html/2405.12115v2#S6.SS3),
[cvc5 finite-field theory](https://cvc5.github.io/docs/latest/theories/finite_field.html).

## Application compiler connection in progress

Checkpoint `929d53686` is pushed on `nico/constraint-reduction-research`.
The following work is newer than that checkpoint and is not selected yet.

`Application.Program.compactHashChain` carries an erased proof that identifies
the exact existing hash-chain circuit. The option selects an implementation
case; a digest or a string cannot provide this proof. Separating application
interfaces from the existing hash-chain circuit avoids a dependency cycle.
The original circuit, step function, and exported names are preserved.

`ApplicationPoseidonRetainedBlock` selects the 86 S-box fields from each of
the three variable/finalization permutations. The fixed ten-block prefix is
not retained. Its 258 slots occupy 10,578 coordinates. Together with the four
message fields, `ApplicationPoseidonRetainedGeometry` proves a prospective
whole-layout logical width of 149,292,999. Its instantiated application plan
has 262 rows. These modules and their focused axiom audit pass.

`ApplicationPoseidonSoundness` connects this placed plan to the selected
program's exact four-word step relation. Its input and output forms are
equal to the actual pilot preimage forms for arbitrary assignments. The
soundness theorem needs no honest-encoding premise. The completion theorem
uses the previously constructed S-box values for every valid selected step.
The focused module build passes. The production library and all axiom tests
also pass together (4,240 jobs); the static boundary gate passes. This remains
an unselected compiler specialization, so the selected counts above do not
change.

The selected compiler still uses the ordinary `ApplicationDirectPlan` path.
Next connect the checked case to the application row plan, matrix program,
assignment blocks, and arbitrary-assignment soundness/completeness path.
Keep the generic ordinary path for programs without this exact circuit proof.
Reuse the existing Poseidon and pin matrix opcodes and the existing indexed
assignment source runs. Do not regenerate packages or fixtures before this
connection and the matrix counts are stable.
