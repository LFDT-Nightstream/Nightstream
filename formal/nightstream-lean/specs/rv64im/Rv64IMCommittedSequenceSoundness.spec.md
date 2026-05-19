# Rv64IMCommittedSequenceSoundness Spec

## Purpose

- **What it is**: The theorem-facing contract for any fixed committed lowered
  RV64IM sequence, including non-advice multi-row lowerings such as `ADDW`,
  `MULH*`, and narrow load/store sequences.
- **What it is not**: It is not an opcode catalog, and it does not by itself
  prove any particular lowering instance.
- **Protocol role**: It fixes the exact semantic obligations for one fixed
  committed row list, one fixed touched-state set, and one fixed
  sequence-boundary result row before a multi-row lowering may be accepted as
  part of the RV64IM expanded-bytecode theorem package.

## Target Formulas

Define the sequence-boundary result object:

$$
\mathrm{SequenceResult}(Output, StateEffect)
:=
(\mathrm{output}, \mathrm{stateEffect}).
$$

Fix:

- `ArchitecturalInputs`
- `AuthenticatedReads`
- `WitnessAssignment`
- `CommittedSequence`
- `TouchedStateSet`
- `PreservedStatePredicate(sequence, touchedState, inputs, reads, result)`
- `rowAssertions(sequence, inputs, reads, witness)`
- `committedResult(sequence, inputs, reads, witness)`
- `isaResult(inputs, reads)`

The correctness target is:

$$
\mathrm{CommittedSequenceCorrect}

(\mathrm{sequence},\ \mathrm{touchedState},\ \mathrm{rowAssertions},\ \mathrm{committedResult},\ \mathrm{isaResult},\ \mathrm{preservedState})
$$

meaning:

$$
\forall\ inputs,\ reads,\ witness,\ 
\mathrm{rowAssertions}(\mathrm{sequence}, inputs, reads, witness)
\Longrightarrow
\mathrm{committedResult}(\mathrm{sequence}, inputs, reads, witness)
=
\mathrm{isaResult}(inputs, reads)
\ \land\
\mathrm{preservedState}(\mathrm{sequence}, \mathrm{touchedState}, inputs, reads,\ 
\mathrm{committedResult}(\mathrm{sequence}, inputs, reads, witness)).
$$

The determinism target is:

$$
\mathrm{CommittedSequenceDeterministic}

(\mathrm{sequence},\ \mathrm{touchedState},\ \mathrm{rowAssertions},\ \mathrm{committedResult})
$$

meaning:

$$
\forall\ inputs,\ reads,\ witness_1,\ witness_2,\ 
\mathrm{rowAssertions}(\mathrm{sequence}, inputs, reads, witness_1)
\land
\mathrm{rowAssertions}(\mathrm{sequence}, inputs, reads, witness_2)
\Longrightarrow
\mathrm{committedResult}(\mathrm{sequence}, inputs, reads, witness_1)
=
\mathrm{committedResult}(\mathrm{sequence}, inputs, reads, witness_2).
$$

Define the exact theorem package:

$$
\mathrm{CommittedSequenceProofPackage}
$$

carrying the fixed committed sequence metadata, the fixed touched-state set, the
correctness theorem, and the determinism theorem.

The core structural consequence is:

$$
\mathrm{CommittedSequenceCorrect}
\Longrightarrow
\mathrm{CommittedSequenceDeterministic}
$$

because the ISA result is a functional target of `(inputs, reads)` alone for a
fixed committed sequence.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - fixed committed row list for every multi-row lowering
  - fixed touched-state set and preserved-state obligations
  - mandatory correctness and determinism theorems for all multi-row lowerings
  - opcode-class net-effect equivalence at the sequence boundary

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/CommittedSequenceSoundness.lean` | General fixed committed-sequence theorem surface |
| `Nightstream/Rv64IM/Execution/CommittedSequenceSoundnessInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Result | `SequenceResult` | structure | Definitional | Packages sequence-boundary output plus state effect |
| Metadata | `CommittedSequence` | structure | Definitional | Fixes the exact lowered row list and sequence-boundary result row |
| Metadata | `TouchedStateSet` | structure | Definitional | Fixes the exact state locations the committed sequence may modify |
| Semantics | `PreservedStatePredicate` | def | Definitional | States that locations outside the touched set are preserved |
| Semantics | `CommittedSequenceCorrect` | def | Definitional | Any satisfying witness assignment yields the ISA result and preserved-state obligation for the fixed sequence |
| Semantics | `CommittedSequenceDeterministic` | def | Definitional | Two satisfying witness assignments on the same fixed sequence cannot diverge |
| Package | `CommittedSequenceProofPackage` | structure | Definitional | Carries fixed sequence metadata plus the accepted correctness + determinism pair |
| Theorem | `committedSequenceDeterministic_of_correct` | theorem | Theorem-Target | Correctness against a functional ISA target forces determinism for the fixed sequence |
| Constructor | `committedSequenceProofPackage_of_correct` | def | Definitional | Builds the proof package from fixed sequence metadata plus correctness and derived determinism |

## Proof Obligations

- Every multi-row lowered sequence must expose a fixed committed row list, a
  fixed touched-state set, and a functional sequence-boundary result row.
- Correctness is stated against architectural inputs plus authenticated reads,
  not against prover-chosen hidden sequence shape or hidden interpreter state.
- Correctness includes preserved-state obligations for every architectural or
  virtual state location outside the fixed touched-state set.
- Determinism forbids two satisfying witness assignments from producing
  different committed outputs or state effects on the same fixed committed
  sequence, the same architectural input, and the same authenticated reads.
- The accepted theorem package for a multi-row lowering contains the fixed
  committed-sequence metadata and both semantic obligations, even if
  determinism is later derived from correctness.

## Out of Scope

- opcode-specific lowering code
- advice-specific witness modeling beyond the generic witness parameter
- SMT versus Lean implementation choice for discharging the obligations
