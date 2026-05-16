# Rv64IMAdviceSequenceSoundness Spec

## Purpose

- **What it is**: The theorem-facing contract for advice-using committed lowered
  RV64IM sequences such as `DIV*` and `REM*`.
- **What it is not**: It is not a concrete lowering catalog and it does not by
  itself prove any particular opcode implementation.
- **Protocol role**: It fixes the exact obligations for a fixed committed row
  list, a fixed touched-state set, and a fixed sequence-boundary result row
  before an advice-using lowered sequence may be accepted as part of the RV64IM
  expanded-bytecode theorem package. This is the advice-specialized instance of
  the general fixed committed-sequence soundness surface.

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
- `AdviceAssignment`
- `CommittedSequence`
- `TouchedStateSet`
- `PreservedStatePredicate(sequence, touchedState, inputs, reads, result)`
- `rowAssertions(sequence, inputs, reads, advice)`
- `committedResult(sequence, inputs, reads, advice)`
- `isaResult(inputs, reads)`

The correctness target is:

$$
\mathrm{AdviceSequenceCorrect}

(\mathrm{sequence},\ \mathrm{touchedState},\ \mathrm{rowAssertions},\ \mathrm{committedResult},\ \mathrm{isaResult},\ \mathrm{preservedState})
$$

meaning:

$$
\forall\ inputs,\ reads,\ advice,\ 
\mathrm{rowAssertions}(\mathrm{sequence}, inputs, reads, advice)
\Longrightarrow
\mathrm{committedResult}(\mathrm{sequence}, inputs, reads, advice)
=
\mathrm{isaResult}(inputs, reads)
\ \land\
\mathrm{preservedState}(\mathrm{sequence}, \mathrm{touchedState}, inputs, reads,\ 
\mathrm{committedResult}(\mathrm{sequence}, inputs, reads, advice)).
$$

The determinism target is:

$$
\mathrm{AdviceSequenceDeterministic}

(\mathrm{sequence},\ \mathrm{touchedState},\ \mathrm{rowAssertions},\ \mathrm{committedResult})
$$

meaning:

$$
\forall\ inputs,\ reads,\ advice_1,\ advice_2,\ 
\mathrm{rowAssertions}(\mathrm{sequence}, inputs, reads, advice_1)
\land
\mathrm{rowAssertions}(\mathrm{sequence}, inputs, reads, advice_2)
\Longrightarrow
\mathrm{committedResult}(\mathrm{sequence}, inputs, reads, advice_1)
=
\mathrm{committedResult}(\mathrm{sequence}, inputs, reads, advice_2).
$$

Define the exact theorem package:

$$
\mathrm{AdviceSequenceProofPackage}
$$

carrying the fixed committed sequence metadata, the fixed touched-state set, the
correctness theorem, and the determinism theorem.

The core structural consequence is:

$$
\mathrm{AdviceSequenceCorrect}
\Longrightarrow
\mathrm{AdviceSequenceDeterministic}
$$

because the ISA result is a functional target of `(inputs, reads)` alone for a
fixed committed sequence.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - expanded bytecode after lowering
  - advice-backed `DIV*` / `REM*` sequences
  - mandatory correctness and determinism theorems for `VAdvice` sequences
  - fixed committed sequence, fixed touched-state set, preserved-state
    obligations

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/CommittedSequenceSoundness.lean` | General fixed committed-sequence theorem surface specialized by this owner |
| `Nightstream/Rv64IM/Execution/AdviceSequenceSoundness.lean` | Advice-backed lowered-sequence theorem surface |
| `Nightstream/Rv64IM/Execution/AdviceSequenceSoundnessInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Result | `SequenceResult` | structure | Definitional | Packages sequence-boundary output plus state effect |
| Metadata | `CommittedSequence` | structure | Definitional | Fixes the exact lowered row list and sequence-boundary result row |
| Metadata | `TouchedStateSet` | structure | Definitional | Fixes the exact state locations the committed sequence may modify |
| Semantics | `PreservedStatePredicate` | def | Definitional | States that locations outside the touched set are preserved |
| Semantics | `AdviceSequenceCorrect` | def | Definitional | Any satisfying advice assignment yields the ISA result and preserved-state obligation for the fixed sequence |
| Semantics | `AdviceSequenceDeterministic` | def | Definitional | Two satisfying advice assignments on the same fixed sequence cannot diverge |
| Package | `AdviceSequenceProofPackage` | structure | Definitional | Carries fixed sequence metadata plus the accepted correctness + determinism pair |
| Theorem | `adviceSequenceDeterministic_of_correct` | theorem | Theorem-Target | Correctness against a functional ISA target forces determinism for the fixed sequence |
| Constructor | `adviceSequenceProofPackage_of_correct` | def | Definitional | Builds the proof package from fixed sequence metadata plus correctness and derived determinism |

## Proof Obligations

- Every lowered sequence that uses `VAdvice` must expose a fixed committed row
  list, a fixed touched-state set, and a functional sequence-boundary result
  row.
- Correctness is stated against architectural inputs plus authenticated reads,
  not against prover witness internals or prover-chosen hidden sequence shape.
- Correctness includes preserved-state obligations for every architectural or
  virtual state location outside the fixed touched-state set.
- Determinism forbids two satisfying advice assignments from producing
  different committed outputs or state effects on the same fixed committed
  sequence, the same architectural input, and the same authenticated reads.
- `VAdvice` has no standalone proof power in this kernel. Advice values are
  accepted only through the fixed committed-sequence row assertions and the
  downstream correctness/determinism theorems for that sequence.
- The accepted theorem package for an advice-backed sequence contains the fixed
  committed-sequence metadata and both semantic obligations, even if
  determinism is later derived from correctness.

## Out of Scope

- concrete `DIV*` / `REM*` lowering code
- SMT or Lean implementation choice for discharging the obligations
- non-advice lowered sequences
