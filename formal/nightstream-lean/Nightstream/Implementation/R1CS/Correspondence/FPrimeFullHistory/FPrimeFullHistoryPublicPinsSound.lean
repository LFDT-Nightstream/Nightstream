import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Poseidon

/-!
Soundness and completeness of the exact full-history public-image owner.

The 4,283 generated rows are a deterministic Poseidon2 program followed by
verifier pins.  `Facts` exposes only those semantic results.  Completeness
starts from pin validity on the executable program state and constructs all
SSA columns; no row-satisfaction or accepted-bit premise is carried in the
input relation.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

namespace Artifact

open FPrimeFullHistoryPublicPins
open FPrimeFullHistoryPublicPinsPoseidonHashes

structure Facts (assignment : Nat → Nat) : Prop where
  pins : ∀ pin ∈ FPrimeFullHistoryPublicPins.pins,
    AffinePins.Pin.Holds assignment pin
  xOutHash : ∀ lane, lane < 4 →
    assignment (xOutTrace.outputColumns.getD lane 0) =
      Poseidon2Sponge.runValueRounds xOutTrace.rounds
        (xOutTrace.inputColumns.map assignment) (fun _ => 0) lane

/-- Semantic input to the exact witness compiler.  The verifier assertions
are precisely the public pins; all other columns are constructed by the
checked-program interpreter. -/
structure ValidInput (state : Nat → Nat) : Prop where
  canonical : ∀ column, state column < goldilocksP
  one : state 0 = 1
  pins : ∀ pin ∈ FPrimeFullHistoryPublicPins.pins,
    AffinePins.Pin.Holds
      (interpret state FPrimeFullHistoryPublicPins.instructions) pin

private theorem pinRows_satisfy_of_program
    {assignment : Nat → Nat}
    (satisfies : Satisfies FPrimeFullHistoryPublicPins.rows assignment) :
    Satisfies (AffinePins.rows FPrimeFullHistoryPublicPins.pins) assignment := by
  have checks := checksSatisfy_of_satisfies satisfies
  intro row member
  exact checks row
    (rowsIncluded_sound FPrimeFullHistoryPublicPins.pin_rows_in_checks row member)

theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies FPrimeFullHistoryPublicPins.rows assignment) :
    Facts assignment where
  pins := AffinePins.rows_sound FPrimeFullHistoryPublicPins.pins_canonical
    canonical one (pinRows_satisfy_of_program satisfies)
  xOutHash := Poseidon2Sponge.trace_values_sound xOutTrace_valid
    canonical one satisfies

private theorem interpreted_one
    {state : Nat → Nat} (valid : ValidInput state) :
    interpret state FPrimeFullHistoryPublicPins.instructions 0 = 1 := by
  have preserved := run_preserves_known
    FPrimeFullHistoryPublicPins.definitions_wellFormed state
  exact (preserved 0 (by native_decide)).trans valid.one

private theorem checksHold
    {state : Nat → Nat} (valid : ValidInput state) :
    ChecksHold state FPrimeFullHistoryPublicPins.instructions := by
  let assignment := interpret state FPrimeFullHistoryPublicPins.instructions
  have canonical : ∀ column, assignment column < goldilocksP :=
    run_canonical valid.canonical
  have one : assignment 0 = 1 := interpreted_one valid
  have pinRows :
      Satisfies (AffinePins.rows FPrimeFullHistoryPublicPins.pins) assignment :=
    AffinePins.rows_complete FPrimeFullHistoryPublicPins.pins_canonical
      canonical one valid.pins
  intro row member
  rcases FPrimeFullHistoryPublicPins.checks_covered row member with
    pinMember | trivialMember
  · exact pinRows row pinMember
  · exact TrivialRows.satisfy FPrimeFullHistoryPublicPins.trivial_rows_valid
      assignment row trivialMember

/-- Exact `CIR-COMPLETE` constructor for this owner. -/
theorem complete
    {state : Nat → Nat} (valid : ValidInput state) :
    Satisfies FPrimeFullHistoryPublicPins.rows
      (interpret state FPrimeFullHistoryPublicPins.instructions) := by
  apply CheckedProgram.complete
    FPrimeFullHistoryPublicPins.definitions_wellFormed
    FPrimeFullHistoryPublicPins.definitions_canonical
    valid.canonical
    (by native_decide)
    valid.one
    (checksHold valid)

end Artifact

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound
