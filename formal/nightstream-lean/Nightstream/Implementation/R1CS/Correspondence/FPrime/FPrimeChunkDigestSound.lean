import Nightstream.Implementation.R1CS.Ownership.FPrime.FPrimeChunkDigestArtifact

/-!
Contract: artifact-level soundness and completeness of the exact production
F' chunk-shape digest gadget.

All 6,661 authoritative Rust rows are classified as a checked, well-formed SSA
program. Therefore every satisfying assignment agrees with one deterministic
execution from `(one, start_step)`, and every canonical input has a satisfying
witness constructed by that execution. The older four-row binding theorem is
retained as the small local corollary used by downstream composition proofs.

The interpreter is extracted from the exact circuit rows. Agreement between
that extracted program and the native Rust digest helper remains an M5
refinement obligation; it is not smuggled into this circuit theorem.
-/

namespace Nightstream.Implementation.R1CS.FPrimeChunkDigestSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeChunkDigest

def interpret (state : Nat → Nat) : Nat → Nat := run state definitions

theorem claimedColumns_known :
    ∀ column ∈ claimedColumns,
      column ∈ knownAfter inputColumns definitions := by
  native_decide

/-- `CIR-FPR-CHUNK`: every satisfying exact production artifact agrees with
the executable SSA semantics on all input and derived columns. -/
theorem fPrimeChunkDigest_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z) :
    AgreeOn (interpret z) z (knownAfter inputColumns definitions) := by
  apply run_agrees_of_builder_satisfies definitions_wellFormed
  · intro column _
    rfl
  · exact hcanon
  · exact hone
  · exact definitions_canonical
  · exact hsat

/-- The four public digest lanes are uniquely determined by the constant-one
and start-step input columns. -/
theorem fPrimeChunkDigest_claim_unique
    {left right : Nat → Nat}
    (leftCanonical : ∀ column, left column < goldilocksP)
    (rightCanonical : ∀ column, right column < goldilocksP)
    (leftOne : left 0 = 1) (rightOne : right 0 = 1)
    (leftSat : Satisfies rows left) (rightSat : Satisfies rows right)
    (inputsEqual : AgreeOn left right inputColumns) :
    ∀ column ∈ claimedColumns, left column = right column := by
  have leftRun := run_agrees_of_builder_satisfies definitions_wellFormed
    (z := left) (state := left) (by intro _ _; rfl)
    leftCanonical leftOne definitions_canonical leftSat
  have rightRun := run_agrees_of_builder_satisfies definitions_wellFormed
    (z := right) (state := left) inputsEqual
    rightCanonical rightOne definitions_canonical rightSat
  intro column member
  have known := claimedColumns_known column member
  exact (leftRun column known).symm.trans (rightRun column known)

/-- `CIR-COMPLETE` for the exact chunk-digest artifact: executing the checked
SSA program constructs a satisfying witness for every canonical input. -/
theorem fPrimeChunkDigest_complete {state : Nat → Nat}
    (canonical : ∀ column, state column < goldilocksP)
    (hone : state 0 = 1) : Satisfies rows (interpret state) := by
  exact run_satisfies_builder_rows definitions_wellFormed canonical
    (by decide) hone definitions_canonical

def Holds (z : Nat → Nat) : Prop :=
  ∀ pair ∈ columnPairs, z pair.1 = z pair.2

private theorem equalityRow_mem {pair : Nat × Nat} (member : pair ∈ columnPairs) :
    equalityRow pair ∈ bindingRows :=
  List.mem_map.mpr ⟨pair, member, rfl⟩

private theorem equality_of_row {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) {left right : Nat}
    (holds : RowHolds z (equalityRow (left, right))) : z left = z right := by
  have leftLt := hcanon left
  have rightLt := hcanon right
  simp only [equalityRow, RowHolds, lcEval, List.foldl, hone,
    goldilocksP] at holds leftLt rightLt
  omega

/-- `CIR-FPR-CHUNK-BIND`: the claimed state digest equals the digest gadget's
four computed output lanes for every satisfying binding assignment. -/
theorem fPrimeChunkDigest_binding_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies bindingRows z) : Holds z := by
  intro pair member
  rcases pair with ⟨left, right⟩
  exact equality_of_row hcanon hone (hsat _ (equalityRow_mem member))

end Nightstream.Implementation.R1CS.FPrimeChunkDigestSound
