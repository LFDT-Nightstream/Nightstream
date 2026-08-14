import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Operations

/-!
Contract: semantic composition of eager Poseidon2 transcript slices.

Owns: proof that any partition of one operation stream into Rust-style eager
slices has the same final state and digest list as one unpartitioned slice.

Does not own: generated rows, physical call placement, operation authority,
recursive lifecycle wiring, or collision resistance.

Emits constraints: no.

Assurance tier: model-level for property
`POSEIDON2-TRANSCRIPT-SLICE-COMPOSITION`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.SliceComposition

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay

private theorem absorbElem_permute_of_full
    (state : State) (value : Field)
    (full : rate ≤ state.absorbed.val) :
    absorbElem state value =
      absorbElem (TranscriptMachine.permute state) value := by
  have noRoom : ¬state.absorbed.val < rate := by omega
  have permutedAbsorbed :
      (TranscriptMachine.permute state).absorbed.val = 0 := rfl
  have permutedRoom :
      (TranscriptMachine.permute state).absorbed.val < rate := by
    rw [permutedAbsorbed]
    decide
  unfold absorbElem
  simp only [noRoom, permutedRoom, reduceDIte]
  simp only [permutedAbsorbed, Nat.zero_add]

theorem semanticNormalizeSlice_idempotent (run : SemanticRun) :
    semanticNormalizeSlice (semanticNormalizeSlice run) =
      semanticNormalizeSlice run := by
  cases run with
  | mk state digests =>
      by_cases full : rate ≤ state.absorbed.val
      · have normalizedNotFull :
            ¬rate ≤ (TranscriptMachine.permute state).absorbed.val := by
          simp [TranscriptMachine.permute, rate]
        simp [semanticNormalizeSlice, full, normalizedNotFull]
      · simp [semanticNormalizeSlice, full]

/-- Eager normalization before the next operation does not change that
operation. A full cursor would perform the same permutation inside
`absorbElem`; every other cursor is unchanged. -/
theorem semanticStep_normalized
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (operation : Operation) :
    semanticStep assignment canonical run operation =
      semanticStep assignment canonical (semanticNormalizeSlice run)
        operation := by
  cases run with
  | mk state digests =>
      by_cases full : rate ≤ state.absorbed.val
      · have absorbEq : ∀ value,
            absorbElem state value =
              absorbElem (TranscriptMachine.permute state) value :=
          fun value => absorbElem_permute_of_full state value full
        cases operation with
        | pinned value =>
            simp [semanticStep, semanticNormalizeSlice, full, absorbEq]
        | external column =>
            simp [semanticStep, semanticNormalizeSlice, full, absorbEq]
        | digest =>
            have digestEq :
                TranscriptMachine.digest state =
                  TranscriptMachine.digest
                    (TranscriptMachine.permute state) := by
              unfold TranscriptMachine.digest
              rw [absorbEq]
            simp [semanticStep, semanticNormalizeSlice, full, digestEq]
      · simp [semanticNormalizeSlice, full]

/-- Pre-normalizing the input state cannot change the final normalized
result of an operation list. -/
theorem semanticExecute_start_normalized
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (operations : List Operation) :
    semanticNormalizeSlice
        (semanticExecute assignment canonical run operations) =
      semanticNormalizeSlice
        (semanticExecute assignment canonical
          (semanticNormalizeSlice run) operations) := by
  cases operations with
  | nil => exact (semanticNormalizeSlice_idempotent run).symm
  | cons operation rest =>
      change semanticNormalizeSlice
          (semanticExecute assignment canonical
            (semanticStep assignment canonical run operation) rest) =
        semanticNormalizeSlice
          (semanticExecute assignment canonical
            (semanticStep assignment canonical
              (semanticNormalizeSlice run) operation) rest)
      rw [semanticStep_normalized assignment canonical run operation]

/-- Headline partition theorem: two consecutive eager slices are exactly one
eager slice over the concatenated operation stream. Repeated use covers any
finite chunk partition. -/
theorem semanticExecuteSlice_append
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (left right : List Operation) :
    semanticExecuteSlice assignment canonical run (left ++ right) =
      semanticExecuteSlice assignment canonical
        (semanticExecuteSlice assignment canonical run left) right := by
  unfold semanticExecuteSlice
  rw [Operations.semanticExecute_append]
  exact semanticExecute_start_normalized assignment canonical
    (semanticExecute assignment canonical run left) right

/-- Execute a finite list of eager slices. The empty partition performs the
same final normalization as one empty slice. -/
def semanticExecuteSlices
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment) :
    SemanticRun → List (List Operation) → SemanticRun
  | run, [] => semanticExecuteSlice assignment canonical run []
  | run, operations :: rest =>
      semanticExecuteSlices assignment canonical
        (semanticExecuteSlice assignment canonical run operations) rest

/-- Any finite slice partition has the same result as one eager execution of
the flattened operation stream. -/
theorem semanticExecuteSlices_eq_flatten
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (slices : List (List Operation)) :
    semanticExecuteSlices assignment canonical run slices =
      semanticExecuteSlice assignment canonical run slices.flatten := by
  induction slices generalizing run with
  | nil => rfl
  | cons operations rest induction =>
      rw [semanticExecuteSlices, induction,
        ← semanticExecuteSlice_append assignment canonical run]
      rfl

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.SliceComposition
