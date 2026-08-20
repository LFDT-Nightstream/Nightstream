import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact output and final-readiness row leaves for the current
Rust-emitted streaming claim-replay arms.

Assurance tier: Rust-to-Lean artifact row certificate.

Owns only three eight-row blocks: the full replay output, the final replay
output, and final expected-state readiness. It does not inspect or validate
the remaining glue rows or any repeated leaf program.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayTransitionRowCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-- Rust stores the two terms of the final-readiness equality in the reverse
order from `builderLinearRow`. -/
def permutedEqualityRow (pair : Nat × Nat) : Row :=
  { a := (EqualityPins.equalityRow pair).a.reverse
    b := (EqualityPins.equalityRow pair).b
    c := (EqualityPins.equalityRow pair).c }

def permutedEqualityRows (pairs : List (Nat × Nat)) : List Row :=
  pairs.map permutedEqualityRow

theorem permutedEqualityRow_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {pair : Nat × Nat}
    (holds : RowHolds assignment (permutedEqualityRow pair)) :
    assignment pair.1 = assignment pair.2 := by
  have aPerm :
      (EqualityPins.equalityRow pair).a.Perm
        (permutedEqualityRow pair).a := by
    simp [permutedEqualityRow]
  have aEqual := Program.lcEval_eq_of_perm assignment aPerm
  have standard : RowHolds assignment (EqualityPins.equalityRow pair) := by
    unfold RowHolds at holds ⊢
    simpa [permutedEqualityRow, aEqual] using holds
  have singleton :
      Satisfies (EqualityPins.rows [pair]) assignment := by
    intro row member
    simp only [EqualityPins.rows, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at member
    subst row
    exact standard
  exact EqualityPins.rows_sound canonical one singleton pair (by simp)

theorem permutedEqualityRows_sound
    {pairs : List (Nat × Nat)} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (permutedEqualityRows pairs) assignment) :
    ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 := by
  intro pair member
  apply permutedEqualityRow_sound canonical one
  exact satisfies _ (List.mem_map.mpr ⟨pair, member, rfl⟩)

def fullOutputPairs : List (Nat × Nat) :=
  (List.range 8).map fun lane => (420 + lane, 155437 + lane)

def finalOutputPairs : List (Nat × Nat) :=
  (List.range 8).map fun lane =>
    (420 + lane, if lane < 3 then 1393 + lane else 87637 + lane)

def finalReadinessPairs : List (Nat × Nat) :=
  (List.range 8).map fun lane => (420 + lane, 411 + lane)

def fullOutputIndexed : List IndexedRow :=
  (fullArm.glueRows.drop 340).take 8

def finalOutputIndexed : List IndexedRow :=
  (finalArm.glueRows.drop 467).take 8

def finalReadinessIndexed : List IndexedRow :=
  (finalArm.glueRows.drop 475).take 8

def fullOutputRows : List Row :=
  fullOutputIndexed.map IndexedRow.row

def finalOutputRows : List Row :=
  finalOutputIndexed.map IndexedRow.row

def finalReadinessRows : List Row :=
  finalReadinessIndexed.map IndexedRow.row

theorem fullOutputRows_exact :
    fullOutputRows = EqualityPins.rows fullOutputPairs := by
  rfl

theorem finalOutputRows_exact :
    finalOutputRows = EqualityPins.rows finalOutputPairs := by
  rfl

theorem finalReadinessRows_exact :
    finalReadinessRows = permutedEqualityRows finalReadinessPairs := by
  rfl

theorem fullOutputIndexed_member
    {indexed : IndexedRow} (member : indexed ∈ fullOutputIndexed) :
    indexed ∈ fullArm.glueRows := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

theorem finalOutputIndexed_member
    {indexed : IndexedRow} (member : indexed ∈ finalOutputIndexed) :
    indexed ∈ finalArm.glueRows := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

theorem finalReadinessIndexed_member
    {indexed : IndexedRow} (member : indexed ∈ finalReadinessIndexed) :
    indexed ∈ finalArm.glueRows := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayTransitionRowCertificate
