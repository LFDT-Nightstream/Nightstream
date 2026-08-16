import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCAuthority
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: exact challenge-carry and cursor rows for one production PiRLC
family phase.

Assurance tier: generated source-row soundness.

Owns one centered-symbol decode row and one carry row for each of the 810
challenge fields, plus one linear row for the family cursor increment. The
verifier-owned cursor bound rules out Goldilocks wraparound.

Does not own either Poseidon2 replay, the 108 residual update, PiRLC
arithmetic, normalized slots, or Rust assignment conformance.

Emits constraints: 1,621 linear R1CS rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Concrete

/-- Columns for the two carried parts of one family state. -/
structure Layout where
  algebra : ProductPiRlcRingCombinationRows.Layout
  beforeChallenge : Source → Fin ringDegree → Nat
  afterChallenge : Source → Fin ringDegree → Nat
  beforeCursor : Nat
  afterCursor : Nat

/-- The carried before value is the exact centered field image of the
transcript-selected symbol. -/
def decodeRow
    (layout : Layout) (source : Source) (lane : Fin ringDegree) : Row :=
  KEquality.equalityRow
    [(layout.beforeChallenge source lane, 1)]
    (ProductPiRlcRingCombinationRows.centeredChallenge
      layout.algebra source lane)

def decodeRows (layout : Layout) : List Row :=
  (List.ofFn fun source : Source =>
    List.ofFn fun lane : Fin ringDegree => decodeRow layout source lane).flatten

theorem decodeRows_length (layout : Layout) :
    (decodeRows layout).length = 810 := by
  unfold decodeRows
  rw [List.length_flatten]
  simp [ringDegree]

/-- One exact carried challenge-field equality. -/
def challengeRow
    (layout : Layout) (source : Source) (lane : Fin ringDegree) : Row :=
  KEquality.equalityRow
    [(layout.beforeChallenge source lane, 1)]
    [(layout.afterChallenge source lane, 1)]

/-- Source-major order for all 15 by 54 challenge fields. -/
def challengeRows (layout : Layout) : List Row :=
  (List.ofFn fun source : Source =>
    List.ofFn fun lane : Fin ringDegree => challengeRow layout source lane).flatten

theorem challengeRows_length (layout : Layout) :
    (challengeRows layout).length = 810 := by
  unfold challengeRows
  rw [List.length_flatten]
  simp [ringDegree]

/-- One field equation: `afterCursor = beforeCursor + 1`. -/
def cursorRow (layout : Layout) : Row :=
  KEquality.equalityRow
    [(layout.afterCursor, 1)]
    [(layout.beforeCursor, 1), (0, 1)]

def rows (layout : Layout) : List Row :=
  decodeRows layout ++ (challengeRows layout ++ [cursorRow layout])

theorem rows_length (layout : Layout) :
    (rows layout).length = 1621 := by
  simp [rows, decodeRows_length, challengeRows_length]

/-- The source assignment places both carried challenge vectors and cursors. -/
def StateColumnsPlaced
    (layout : Layout) (assignment : Nat → Nat)
    (before after : FamilyState) : Prop :=
  (∀ source lane,
    assignment (layout.beforeChallenge source lane) =
      (before.challenges source lane).val) /\
  (∀ source lane,
    assignment (layout.afterChallenge source lane) =
      (after.challenges source lane).val) /\
  assignment layout.beforeCursor = before.familyCursor /\
  assignment layout.afterCursor = after.familyCursor

private theorem decodeRow_holds
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    (source : Source) (lane : Fin ringDegree) :
    RowHolds assignment (decodeRow layout source lane) := by
  apply satisfies
  apply List.mem_append_left
  apply List.mem_flatten.mpr
  refine ⟨List.ofFn fun lane : Fin ringDegree =>
      decodeRow layout source lane, ?_, ?_⟩
  · exact List.mem_ofFn.mpr ⟨source, rfl⟩
  · exact List.mem_ofFn.mpr ⟨lane, rfl⟩

private theorem challengeRow_holds
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    (source : Source) (lane : Fin ringDegree) :
    RowHolds assignment (challengeRow layout source lane) := by
  apply satisfies
  apply List.mem_append_right
  apply List.mem_append_left
  apply List.mem_flatten.mpr
  refine ⟨List.ofFn fun lane : Fin ringDegree =>
      challengeRow layout source lane, ?_, ?_⟩
  · exact List.mem_ofFn.mpr ⟨source, rfl⟩
  · exact List.mem_ofFn.mpr ⟨lane, rfl⟩

private theorem cursorRow_holds
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment) :
    RowHolds assignment (cursorRow layout) := by
  exact satisfies _
    (List.mem_append_right _ (List.mem_append_right _ (by simp)))

private theorem fieldValue_lt (value : F) : value.val < goldilocksP := by
  simpa [goldilocksP, goldilocksModulus] using value.isLt

/-- Accepted decode rows bind the carried challenge vector to the algebra's
centered transcript symbols. -/
theorem decoded_before_exact
    {layout : Layout} {assignment : Nat → Nat}
    {before after : FamilyState}
    (one : assignment 0 = 1)
    (range : ∀ source lane,
      assignment (layout.algebra.challengeSymbol source lane) < 5)
    (placed : StateColumnsPlaced layout assignment before after)
    (satisfies : Satisfies (rows layout) assignment) :
    decodedChallenges layout.algebra assignment range = before.challenges := by
  funext source lane
  have rowEqual :=
    (KEquality.equalityRow_iff assignment
      [(layout.beforeChallenge source lane, 1)]
      (ProductPiRlcRingCombinationRows.centeredChallenge
        layout.algebra source lane) one).mp
      (decodeRow_holds satisfies source lane)
  have beforeEqualsCentered :
      before.challenges source lane =
        ProductPiRlcRingCombinationSound.termsField assignment
          (ProductPiRlcRingCombinationRows.centeredChallenge
            layout.algebra source lane) := by
    apply Fin.ext
    simpa [ProductPiRlcRingCombinationSound.termsField, lcEval,
      placed.1 source lane,
      Nat.mod_eq_of_lt (fieldValue_lt (before.challenges source lane))] using
        rowEqual
  have centered :=
    ProductPiRlcRingCombinationSound.centeredChallenge_eq_embedCoefficient
      layout.algebra assignment one source lane (range source lane)
  change ProductPiRlcRingCombinationSound.challengeRing
      layout.algebra assignment range source lane = before.challenges source lane
  exact centered.symm.trans beforeEqualsCentered.symm

/-- Accepted challenge rows force all 810 carried fields to be unchanged. -/
theorem challenges_exact
    {layout : Layout} {assignment : Nat → Nat}
    {before after : FamilyState}
    (one : assignment 0 = 1)
    (placed : StateColumnsPlaced layout assignment before after)
    (satisfies : Satisfies (rows layout) assignment) :
    after.challenges = before.challenges := by
  funext source lane
  apply Fin.ext
  have equal :=
    (KEquality.equalityRow_iff assignment
      [(layout.beforeChallenge source lane, 1)]
      [(layout.afterChallenge source lane, 1)] one).mp
      (challengeRow_holds satisfies source lane)
  simpa [lcEval, placed.1 source lane, placed.2.1 source lane,
    Nat.mod_eq_of_lt (fieldValue_lt (before.challenges source lane)),
    Nat.mod_eq_of_lt (fieldValue_lt (after.challenges source lane))] using
      equal.symm

/-- The accepted cursor row gives the natural-number increment. The production
family schedule supplies the small before-cursor bound. -/
theorem cursor_exact
    {layout : Layout} {assignment : Nat → Nat}
    {before after : FamilyState}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeBound : before.familyCursor < 110)
    (placed : StateColumnsPlaced layout assignment before after)
    (satisfies : Satisfies (rows layout) assignment) :
    after.familyCursor = before.familyCursor + 1 := by
  have equal :=
    (KEquality.equalityRow_iff assignment
      [(layout.afterCursor, 1)]
      [(layout.beforeCursor, 1), (0, 1)] one).mp
      (cursorRow_holds satisfies)
  have afterLt : after.familyCursor < goldilocksP := by
    rw [← placed.2.2.2]
    exact canonical layout.afterCursor
  have sumLt : before.familyCursor + 1 < goldilocksP := by
    unfold goldilocksP
    omega
  simpa [lcEval, placed.2.2.1, placed.2.2.2, one,
    Nat.mod_eq_of_lt afterLt, Nat.mod_eq_of_lt sumLt] using equal

/-- Semantic result of all 1,621 challenge-and-cursor rows. -/
structure Exact
    (layout : Layout) (assignment : Nat → Nat)
    (range : ∀ source lane,
      assignment (layout.algebra.challengeSymbol source lane) < 5)
    (before after : FamilyState) : Prop where
  decoded : decodedChallenges layout.algebra assignment range = before.challenges
  challenges : after.challenges = before.challenges
  cursor : after.familyCursor = before.familyCursor + 1

/-- Main soundness theorem for the complete decode, carry, and cursor block. -/
theorem rows_sound
    {layout : Layout} {assignment : Nat → Nat}
    {before after : FamilyState}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : ∀ source lane,
      assignment (layout.algebra.challengeSymbol source lane) < 5)
    (beforeBound : before.familyCursor < 110)
    (placed : StateColumnsPlaced layout assignment before after)
    (satisfies : Satisfies (rows layout) assignment) :
    Exact layout assignment range before after :=
  ⟨decoded_before_exact one range placed satisfies,
    challenges_exact one placed satisfies,
    cursor_exact canonical one beforeBound placed satisfies⟩

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows
