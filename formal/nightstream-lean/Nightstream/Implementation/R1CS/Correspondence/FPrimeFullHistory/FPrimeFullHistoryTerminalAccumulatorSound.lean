import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryAccumulatorClaimSerialization
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.ConstantPrefix

/-!
Contract: exact soundness and compiler completeness for the compact terminal
accumulator-v1 owner.

| Branch | Exact rows | Mathematical obligation | Emits constraints? |
|---|---:|---|---|
| `prefix` | 27 | Pin the inactive-X zero and 26 supported-profile constants | yes |
| `source` | 0 | Identify the 1,682 absorbed wires with the checked PiDEC-parent projection | no |
| `digest` | 254,884 | Evaluate one Poseidon2 sponge over that projection | yes |

Owns: the terminal accumulator-v1 source schema, prefix definitions, and one
Poseidon2 compression trace.
Does not own: PiDEC acceptance, PiRLC parent authority, `y_zcol` validation,
or authority for the four digest lanes by themselves.
Authority boundary: the preimage is derived from the verifier-checked PiDEC
parent; the resulting digest is compression only.

Assurance tier: artifact-checked soundness and compiler completeness for the
exact 254,911 generated rows under the supported no-`adv` profile.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator
open Nightstream.Implementation.R1CS.FPrimeFullHistoryAccumulatorClaimSerialization

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

/-- Exact compact Poseidon2 trace owned by this terminal accumulator. -/
def accumulatorTrace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.accumulatorDigestTrace

def accumulatorClaimConstantColumns : List Nat :=
  constantColumnsFrom accumulatorClaimSourceColumns

def accumulatorClaimConstantDefinitions : List Definition :=
  AccumulatorConstantPrefix.definitions
    accumulatorClaimConstantColumns constantValues

/-- The promoted trace absorbs exactly the terminal checked-parent schema,
with the freshly allocated constants in their Rust allocation order. -/
theorem accumulatorClaimSourceColumns_schema :
    accumulatorClaimSourceColumns =
      expectedSourceColumns FPrimeFullHistoryPiDec.terminalColumnMap
        accumulatorClaimConstantColumns := by
  native_decide

theorem accumulatorClaimSourceColumns_length :
    accumulatorClaimSourceColumns.length = 1682 := by
  native_decide

private theorem accumulatorClaimConstantColumns_length :
    accumulatorClaimConstantColumns.length = constantValues.length := by
  native_decide

private theorem accumulatorClaimConstantValues_canonical :
    ∀ value ∈ constantValues, value < goldilocksP := by
  native_decide

private theorem accumulatorClaimConstantDefinitions_member :
    ∀ definition ∈ accumulatorClaimConstantDefinitions,
      definition ∈ definitions segment0Instructions := by
  native_decide

private theorem accumulatorClaimConstantValues_sound
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (segment0 : AssignmentHolds segment0Instructions assignment) :
    accumulatorClaimConstantColumns.map assignment = constantValues := by
  exact AccumulatorConstantPrefix.values_of_assignmentHolds one
    accumulatorClaimConstantColumns_length
    accumulatorClaimConstantValues_canonical
    accumulatorClaimConstantDefinitions_member segment0

/-- The exact prefix definitions turn the 1,682 trace inputs into the
supported accumulator-v1 projection of the verifier-normalized terminal
PiDEC parent. -/
theorem parentClaimSourceValues_sound
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (segment0 : AssignmentHolds segment0Instructions assignment) :
    accumulatorClaimSourceColumns.map assignment =
      terminalParentPreimage assignment := by
  rw [accumulatorClaimSourceColumns_schema]
  exact terminalExpectedSourceColumns_values
    (accumulatorClaimConstantValues_sound one segment0)

private theorem accumulatorTrace_output_length :
    accumulatorTrace.outputColumns.length = 4 := by
  native_decide

private theorem accumulatorTrace_values_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows assignment) :
    accumulatorTrace.ValueAccepted assignment := by
  intro lane laneLt
  rw [accumulatorTrace_output_length] at laneLt
  exact Poseidon2Sponge.trace_values_sound
    FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.accumulatorDigestTrace_valid
    canonical one satisfies lane laneLt

/-- Independent semantic conclusions reconstructed from the exact two-piece
owner.  No digest lane is accepted without replaying its Poseidon2 trace. -/
structure Facts (assignment : Nat → Nat) : Prop where
  segment0 : AssignmentHolds segment0Instructions assignment
  parentClaimSource :
    accumulatorClaimSourceColumns.map assignment =
      terminalParentPreimage assignment
  accumulatorDigest : accumulatorTrace.ValueAccepted assignment

/-- `CIR-SOUND` for all 254,911 exact terminal-accumulator rows. -/
theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Facts assignment := by
  have pieces := (satisfies_flatten_iff rowPieces assignment).mp satisfies
  have segment0Satisfies : Satisfies segment0Rows assignment :=
    pieces segment0Rows (by simp [rowPieces])
  have traceSatisfies : Satisfies
      FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows assignment :=
    pieces FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows
      (by simp [rowPieces])
  let segment0Facts : AssignmentHolds segment0Instructions assignment :=
    assignmentHolds_sound segment0_definitions_canonical canonical one
      segment0Satisfies
  exact {
    segment0 := segment0Facts
    parentClaimSource := parentClaimSourceValues_sound one segment0Facts
    accumulatorDigest := accumulatorTrace_values_sound canonical one
      traceSatisfies
  }

/-- Same-assignment compiler evidence for the ordinary prefix plus semantic
execution evidence for every sponge row.  Neither field contains `Satisfies`
or an accepted digest value. -/
structure CompilerWitness (assignment : Nat → Nat) : Prop where
  segment0 : AssignmentHolds segment0Instructions assignment
  accumulatorDigest : accumulatorTrace.ExecutionWitness assignment

/-- `CIR-COMPLETE` for the exact compact owner. -/
theorem complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : CompilerWitness assignment) :
    Satisfies rows assignment := by
  have segment0Satisfies : Satisfies segment0Rows assignment :=
    assignmentHolds_complete segment0_definitions_canonical canonical one
      witness.segment0
  have traceSatisfies : Satisfies
      FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows assignment := by
    simpa [accumulatorTrace,
      FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows] using
      Poseidon2Sponge.Trace.execution_complete canonical one
        witness.accumulatorDigest
  apply (satisfies_flatten_iff rowPieces assignment).mpr
  intro piece member
  change piece ∈
    [segment0Rows,
      FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact segment0Satisfies
  · exact traceSatisfies

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSound
