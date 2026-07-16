import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailRows
import Nightstream.Implementation.R1CS.Core.Program

/-!
Semantic refinement of the bounded-acceptance prefix in the `Pi_RLC`
54-of-64 selection tail.

Owns: the four slack-bit leaves, exact slack recomposition, and the equation
forcing the 64-candidate accepted count to equal `54 + slack`.

Does not own: candidate accept-wire provenance, selector semantics,
first-accepted ordering, production column placement, coefficient assembly,
Rust conformance, row removal, or costs.

Emits constraints: no.

Authority boundary: the final cumulative count must already be proved from the
verifier-owned candidate decisions by the lane layer. These six rows prove only
that this authoritative count is at least 54; they cannot manufacture accepted
candidates by themselves.

| Protocol | Phase | Constraint family | Equation | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/acceptance bound | four Boolean leaves | `b_i * (1-b_i) = 0` | every slack digit is zero or one |
| `Pi_RLC` | sampler/acceptance bound | slack recomposition | `slack = sum 2^i b_i` | `0 <= slack <= 15` over the integers |
| `Pi_RLC` | sampler/acceptance bound | enough accepts | `count_63 = 54 + slack` | the bounded candidate set contains at least 54 accepts |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Acceptance

open Nightstream.Implementation.R1CS

def slackTerms : List (Nat × Nat) :=
  [(SelectionRows.slackBitCol 0, 1),
   (SelectionRows.slackBitCol 1, 2),
   (SelectionRows.slackBitCol 2, 4),
   (SelectionRows.slackBitCol 3, 8)]

def slackValue (assignment : Nat -> Nat) : Nat :=
  assignment (SelectionRows.slackBitCol 0) +
    2 * assignment (SelectionRows.slackBitCol 1) +
    4 * assignment (SelectionRows.slackBitCol 2) +
    8 * assignment (SelectionRows.slackBitCol 3)

private theorem satisfies_acceptanceBoundRows
    {assignment : Nat -> Nat}
    (satisfies : Satisfies SelectionRows.rows assignment) :
    Satisfies SelectionRows.acceptanceBoundRows assignment := by
  intro row member
  apply satisfies row
  simp [SelectionRows.rows, member]

theorem slackBitsBoolean
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    ∀ offset, offset < 4 ->
      assignment (SelectionRows.slackBitCol offset) <= 1 := by
  intro offset offsetLt
  apply bitRow_le_one prime (canonical _) one
  apply satisfies_acceptanceBoundRows satisfies
  rw [SelectionRows.acceptanceBoundRows]
  exact List.mem_append_left _
    (List.mem_map.mpr ⟨offset, List.mem_range.mpr offsetLt, rfl⟩)

theorem slackValue_le_fifteen
    {assignment : Nat -> Nat}
    (bits : ∀ offset, offset < 4 ->
      assignment (SelectionRows.slackBitCol offset) <= 1) :
    slackValue assignment <= 15 := by
  have bit0 := bits 0 (by decide)
  have bit1 := bits 1 (by decide)
  have bit2 := bits 2 (by decide)
  have bit3 := bits 3 (by decide)
  unfold slackValue
  omega

private theorem slackTerms_canonical :
    Program.CanonicalTerms slackTerms := by
  intro term member
  simp [slackTerms] at member
  rcases member with rfl | rfl | rfl | rfl <;> decide

private theorem slackRow_eq_builder :
    SelectionRows.slackRecompositionRow =
      Program.builderLinearRow SelectionRows.slackCol slackTerms := by
  decide

theorem slack_eq_value
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    assignment SelectionRows.slackCol = slackValue assignment := by
  have bits := slackBitsBoolean prime canonical one satisfies
  have valueLe := slackValue_le_fifteen bits
  have valueLtGoldilocks : slackValue assignment < goldilocksP := by
    have bound : 15 < goldilocksP := by decide
    omega
  have holds : RowHolds assignment SelectionRows.slackRecompositionRow := by
    apply satisfies_acceptanceBoundRows satisfies
    simp [SelectionRows.acceptanceBoundRows]
  rw [slackRow_eq_builder] at holds
  have decoded := Program.builderLinearRow_sound canonical one
    SelectionRows.slackCol slackTerms slackTerms_canonical holds
  have decodedValue : assignment SelectionRows.slackCol =
      slackValue assignment % goldilocksP := by
    simpa [slackTerms, slackValue, lcEval] using decoded
  rw [Nat.mod_eq_of_lt valueLtGoldilocks] at decodedValue
  exact decodedValue

def acceptedCountTerms : List (Nat × Nat) :=
  [(SelectionRows.slackCol, 1), (0, SelectionRows.outputCount)]

private theorem acceptedCountTerms_canonical :
    Program.CanonicalTerms acceptedCountTerms := by
  intro term member
  simp [acceptedCountTerms, SelectionRows.outputCount] at member
  rcases member with rfl | rfl <;> decide

private theorem acceptedCountRow_eq_builder :
    SelectionRows.acceptedCountRow =
      Program.builderLinearRow (SelectionRows.cumulativeCol 63)
        acceptedCountTerms := by
  decide

/-- The six acceptance-bound rows force an exact integer equation, not merely
a field congruence. -/
theorem finalCount_eq_outputCount_add_slack
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    assignment (SelectionRows.cumulativeCol 63) =
      SelectionRows.outputCount + assignment SelectionRows.slackCol := by
  have slackEq := slack_eq_value prime canonical one satisfies
  have bits := slackBitsBoolean prime canonical one satisfies
  have valueLe := slackValue_le_fifteen bits
  have sumLtGoldilocks :
      assignment SelectionRows.slackCol + SelectionRows.outputCount <
        goldilocksP := by
    rw [slackEq]
    simp only [SelectionRows.outputCount]
    have bound : 69 < goldilocksP := by decide
    omega
  have holds : RowHolds assignment SelectionRows.acceptedCountRow := by
    apply satisfies_acceptanceBoundRows satisfies
    simp [SelectionRows.acceptanceBoundRows]
  rw [acceptedCountRow_eq_builder] at holds
  have decoded := Program.builderLinearRow_sound canonical one
    (SelectionRows.cumulativeCol 63) acceptedCountTerms
    acceptedCountTerms_canonical holds
  have exactSum : assignment (SelectionRows.cumulativeCol 63) =
      assignment SelectionRows.slackCol + SelectionRows.outputCount := by
    simpa [acceptedCountTerms, lcEval, one,
      Nat.mod_eq_of_lt sumLtGoldilocks] using decoded
  simpa [Nat.add_comm] using exactSum

theorem enoughAccepted
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    SelectionRows.outputCount <=
      assignment (SelectionRows.cumulativeCol 63) := by
  rw [finalCount_eq_outputCount_add_slack prime canonical one satisfies]
  omega

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Acceptance
