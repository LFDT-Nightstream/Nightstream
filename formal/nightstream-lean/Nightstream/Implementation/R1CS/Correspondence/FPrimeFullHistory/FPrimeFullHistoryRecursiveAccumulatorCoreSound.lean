import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorCoreArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryAccumulatorClaimSerialization
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.ConstantPrefix

/-!
Contract: exact soundness and compiler completeness for the recursive direct
accumulator-digest core.

| Branch | Exact rows | Mathematical obligation | Exact owner |
|---|---:|---|---|
| `prefix` | 27 | Pin the inactive-X zero and 26 supported-profile constants | `segment0Instructions` |
| `source` | 0 | Order the checked recursive PiDEC parent as the 1,682-field accumulator-v1 preimage | `accumulatorClaimSourceColumns` |
| `digest` | 254,884 | Evaluate one complete Poseidon2 sponge trace over that preimage | `accumulatorDigestTrace` |

Owns: the 27-row checked prefix, exact accumulator-v1 source projection, and
the 254,884-row Poseidon2 trace.
Does not own: PiDEC acceptance, PiRLC parent authority, `y_zcol` validation,
or authority for the resulting digest.
Authority boundary: the digest is only Poseidon2 compression of the checked
recursive-parent projection.  It is never accepted as authority by itself.

Assurance tier: artifact-checked soundness and compiler completeness for this
exact generated owner, plus exact assignment-index refinement to the
model-level accumulator-v1 serializer.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCore

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

/-- The one compact Poseidon2 trace owned by the recursive accumulator core. -/
def digestTrace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace

theorem digestTrace_inputColumns :
    digestTrace.inputColumns = accumulatorClaimSourceColumns := by
  rfl

theorem digestTrace_outputColumns :
    digestTrace.outputColumns = accumulatorDigestColumns := by
  native_decide

private theorem digestTrace_valid :
    digestTrace.Valid
      FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.rows := by
  simpa [digestTrace] using
    FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace_valid

def accumulatorClaimConstantColumns : List Nat :=
  FPrimeFullHistoryAccumulatorClaimSerialization.constantColumnsFrom
    accumulatorClaimSourceColumns

def accumulatorClaimConstantDefinitions : List Definition :=
  AccumulatorConstantPrefix.definitions accumulatorClaimConstantColumns
    FPrimeFullHistoryAccumulatorClaimSerialization.constantValues

/-- The generated source list is exactly the supported recursive-parent
schema, rather than merely another list of the same length. -/
theorem accumulatorClaimSourceColumns_schema :
    accumulatorClaimSourceColumns =
      FPrimeFullHistoryAccumulatorClaimSerialization.expectedSourceColumns
        FPrimeFullHistoryPiDec.recursiveColumnMap
        accumulatorClaimConstantColumns := by
  native_decide

theorem accumulatorClaimSourceColumns_length :
    accumulatorClaimSourceColumns.length = 1682 := by
  native_decide

private theorem accumulatorClaimConstantColumns_length :
    accumulatorClaimConstantColumns.length =
      FPrimeFullHistoryAccumulatorClaimSerialization.constantValues.length := by
  native_decide

private theorem accumulatorClaimConstantValues_canonical :
    ∀ value ∈ FPrimeFullHistoryAccumulatorClaimSerialization.constantValues,
      value < goldilocksP := by
  native_decide

/-- Every serializer constant is pinned by the exact 27-definition prefix.
The remaining prefix definition is the inactive-X zero. -/
private theorem accumulatorClaimConstantDefinitions_member :
    ∀ definition ∈ accumulatorClaimConstantDefinitions,
      definition ∈ definitions segment0Instructions := by
  native_decide

private theorem accumulatorClaimConstantValues_sound
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (program : AssignmentHolds segment0Instructions assignment) :
    accumulatorClaimConstantColumns.map assignment =
      FPrimeFullHistoryAccumulatorClaimSerialization.constantValues := by
  exact AccumulatorConstantPrefix.values_of_assignmentHolds one
    accumulatorClaimConstantColumns_length
    accumulatorClaimConstantValues_canonical
    accumulatorClaimConstantDefinitions_member program

/-- The prefix definitions bind the generated source columns to the exact
semantic recursive-parent preimage. -/
theorem parentClaimSourceValues_sound
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (program : AssignmentHolds segment0Instructions assignment) :
    accumulatorClaimSourceColumns.map assignment =
      FPrimeFullHistoryAccumulatorClaimSerialization.recursiveParentPreimage
        assignment := by
  rw [accumulatorClaimSourceColumns_schema]
  exact
    FPrimeFullHistoryAccumulatorClaimSerialization.recursiveExpectedSourceColumns_values
      (accumulatorClaimConstantValues_sound one program)

/-- Independent conclusions reconstructed from every exact core row.  The
digest conclusion recomputes Poseidon2 over the ordered semantic source; it
does not authenticate a prover-carried digest. -/
structure Facts (assignment : Nat → Nat) : Prop where
  segment0 : AssignmentHolds segment0Instructions assignment
  parentClaimSource :
    accumulatorClaimSourceColumns.map assignment =
      FPrimeFullHistoryAccumulatorClaimSerialization.recursiveParentPreimage
        assignment
  accumulatorDigest : digestTrace.ValueAccepted assignment

/-- `CIR-SOUND` for all 254,911 exact recursive-accumulator-core rows. -/
theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Facts assignment := by
  have pieces := (satisfies_flatten_iff rowPieces assignment).mp satisfies
  have segment0Satisfies : Satisfies segment0Rows assignment :=
    pieces segment0Rows (by simp [rowPieces])
  have digestSatisfies :
      Satisfies FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.rows
        assignment :=
    pieces FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.rows
      (by simp [rowPieces])
  let segment0 := assignmentHolds_sound segment0_definitions_canonical
    canonical one segment0Satisfies
  refine {
    segment0 := segment0
    parentClaimSource := parentClaimSourceValues_sound one segment0
    accumulatorDigest := ?_
  }
  intro lane laneLt
  have laneLtFour : lane < 4 := by
    rw [← digestTrace_valid.outputLength]
    exact laneLt
  exact Poseidon2Sponge.trace_values_sound digestTrace_valid canonical one
    digestSatisfies lane laneLtFour

/-- Native/compiler data sufficient to reconstruct every exact owner row.
The checked prefix is represented by same-assignment semantics and the sponge
by explicit semantic round executions.  Neither field assumes R1CS
satisfaction. -/
structure CompilerWitness (assignment : Nat → Nat) : Prop where
  segment0 : AssignmentHolds segment0Instructions assignment
  accumulatorDigest : digestTrace.ExecutionWitness assignment

/-- `CIR-COMPLETE` for the exact compact core.  Its premises are executable
same-assignment prefix semantics and explicit Poseidon2 execution witnesses,
never owner-row satisfaction. -/
theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : CompilerWitness assignment) :
    Satisfies rows assignment := by
  have segment0Satisfies : Satisfies segment0Rows assignment :=
    assignmentHolds_complete segment0_definitions_canonical canonical one
      witness.segment0
  have digestSatisfies :
      Satisfies FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.rows
        assignment := by
    simpa [digestTrace,
      FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.rows] using
      Poseidon2Sponge.Trace.execution_complete canonical one
        witness.accumulatorDigest
  apply (satisfies_flatten_iff rowPieces assignment).mpr
  intro piece member
  change piece ∈
    [segment0Rows,
      FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.rows] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact segment0Satisfies
  · exact digestSatisfies

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSound
