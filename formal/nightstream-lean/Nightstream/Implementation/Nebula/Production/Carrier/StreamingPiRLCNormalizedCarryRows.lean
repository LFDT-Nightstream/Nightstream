import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyCompleteRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedAlgebraRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetained

/-!
Contract: exact normalized low-norm image of the production PiRLC challenge
carry and family cursor rows.

Assurance tier: model-level.

Owns the 1,621 retained equality-row images, their direct radix-seven source
decoding, and the same-assignment implication to the existing Lean
challenge-carry and cursor theorem. It also proves that a carried strong-set
challenge makes separate challenge-symbol range rows unnecessary.

Does not own state-column placement, selector authority, the Rust witness
encoder, replay rows, overlays, recursive orchestration, or cryptographic
security assumptions.

Emits constraints: no. It specifies and proves the arithmetic meaning of the
existing normalized product-row recipe.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.SuperNeo.Concrete

namespace Normalized

private abbrev sourceLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout

private abbrev carryLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
    sourceLayout

abbrev Arm :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.Arm

def sourceColumns : Nat := 146224

abbrev finalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.finalColumns

theorem sourceColumns_positive : 0 < sourceColumns := by
  decide

theorem finalColumns_positive : 0 < finalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.finalColumns_positive

def selectorColumn : Arm → Fin finalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.selectorColumn

def directSourceStart : Nat := 144276

def finalDirectStart : Nat := 1076045

/-- Direct radix-seven image of one challenge symbol. -/
def challengeSlot
    (column : Fin sourceColumns) (nonzero : column.val ≠ 0)
    (upper : column.val < 811) :
    DecodedSourceSlot sourceColumns finalColumns where
  column := column
  start := 702 + (column.val - 1) * 23
  width := 23
  widthPositive := by decide
  columnsFit := by
    have lower : 1 ≤ column.val := Nat.one_le_iff_ne_zero.mpr nonzero
    change 702 + (column.val - 1) * 23 + 23 ≤ 2484972
    omega

/-- Direct radix-seven image from the generated 1,964-column decoder run. -/
def directSlot
    (column : Fin sourceColumns) (_lower : directSourceStart ≤ column.val) :
    DecodedSourceSlot sourceColumns finalColumns where
  column := column
  start := finalDirectStart + (column.val - directSourceStart) * 23
  width := 23
  widthPositive := by decide
  columnsFit := by
    have upper := column.isLt
    unfold sourceColumns at upper
    change 1076045 + (column.val - 144276) * 23 + 23 ≤ 2484972
    omega

/-- Exact final linear image of one source column used by this block.
Unreferenced middle columns map to zero. -/
def sourceColumnForm (column : Fin sourceColumns) : Form finalColumns :=
  if zero : column.val = 0 then
    constantForm finalColumns_positive
  else if challenge : column.val < 811 then
    sourceSlotForm (challengeSlot column zero challenge)
  else if direct : directSourceStart ≤ column.val then
    sourceSlotForm (directSlot column direct)
  else
    Form.zero

/-- Sparse value of one carry source column on the final assignment. -/
def sourceColumnValue
    (column : Fin sourceColumns) (assignment : Fin finalColumns → F) : F :=
  if zero : column.val = 0 then
    assignment ⟨0, finalColumns_positive⟩
  else if challenge : column.val < 811 then
    sourceSlotValue (challengeSlot column zero challenge) assignment
  else if direct : directSourceStart ≤ column.val then
    sourceSlotValue (directSlot column direct) assignment
  else
    0

theorem evaluate_sourceColumnForm
    (column : Fin sourceColumns) (assignment : Fin finalColumns → F) :
    Form.evaluate (sourceColumnForm column) assignment =
      sourceColumnValue column assignment := by
  unfold sourceColumnForm sourceColumnValue
  split
  · exact evaluate_constantForm finalColumns_positive assignment
  · split
    · exact evaluate_sourceSlotForm _ assignment
    · split
      · exact evaluate_sourceSlotForm _ assignment
      · exact Form.evaluate_zero assignment

theorem sourceColumnValue_zero
    (assignment : Fin finalColumns → F) :
    sourceColumnValue ⟨0, sourceColumns_positive⟩ assignment =
      assignment ⟨0, finalColumns_positive⟩ := by
  unfold sourceColumnValue
  rw [dif_pos rfl]

/-- One typed source assignment decoded from the final low-norm assignment. -/
def decodedAssignment
    (assignment : Fin finalColumns → F) : ColumnId → F :=
  fun column =>
    sourceColumnValue
      (NumericBridge.finiteColumnIndex sourceColumns_positive column)
      assignment

/-- Canonical numeric view used by the existing carry-row theorem. -/
def numericAssignment
    (assignment : Fin finalColumns → F) : Nat → Nat :=
  NumericRowBridge.numericAssignment NumericBridge.sourceColumn
    (decodedAssignment assignment)

theorem numericAssignment_canonical
    (assignment : Fin finalColumns → F) :
    ∀ column, numericAssignment assignment column < goldilocksP := by
  intro column
  exact NumericRowBridge.numericAssignment_canonical
    NumericBridge.sourceColumn (decodedAssignment assignment) column

theorem numericAssignment_one
    (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1) :
    numericAssignment assignment 0 = 1 := by
  have decodedOne :
      decodedAssignment assignment (NumericBridge.sourceColumn 0) = 1 := by
    unfold decodedAssignment
    rw [NumericBridge.finiteColumnIndex_sourceColumn_of_lt
      sourceColumns_positive sourceColumns_positive]
    exact (sourceColumnValue_zero assignment).trans constantOne
  have values := congrArg Fin.val decodedOne
  simpa [numericAssignment, NumericRowBridge.numericAssignment] using values

def sourceColumnFormAt (column : Nat) : Form finalColumns :=
  sourceColumnForm
    (NumericBridge.finiteColumnIndex sourceColumns_positive
      (NumericBridge.sourceColumn column))

/-- Exact final image of one numeric source linear combination. -/
def combinationImage : LinComb → Form finalColumns
  | [] => Form.zero
  | term :: tail =>
      Form.add
        (Form.scale (NumericRowBridge.residue term.2)
          (sourceColumnFormAt term.1))
        (combinationImage tail)

theorem evaluate_combinationImage
    (source : LinComb) (assignment : Fin finalColumns → F) :
    Form.evaluate (combinationImage source) assignment =
      (NumericRowBridge.terms NumericBridge.sourceColumn source).eval
        (decodedAssignment assignment) := by
  induction source with
  | nil =>
      exact Form.evaluate_zero assignment
  | cons term tail inductionHypothesis =>
      rw [combinationImage, Form.evaluate_add, Form.evaluate_scale,
        inductionHypothesis]
      simp only [NumericRowBridge.terms, List.map_cons,
        LinearCombination.eval, NumericRowBridge.term]
      unfold sourceColumnFormAt decodedAssignment
      rw [evaluate_sourceColumnForm]

theorem evaluate_combinationImage_eq_residue
    (source : LinComb) (assignment : Fin finalColumns → F) :
    Form.evaluate (combinationImage source) assignment =
      NumericRowBridge.residue
        (R1CS.lcEval (numericAssignment assignment) source) := by
  rw [evaluate_combinationImage]
  exact NumericRowBridge.terms_eval_eq_residue_lcEval
    NumericBridge.sourceColumn (decodedAssignment assignment) source

private theorem evaluate_sub
    (left right : Form finalColumns)
    (assignment : Fin finalColumns → F) :
    Form.evaluate (Form.sub left right) assignment =
      Form.evaluate left assignment - Form.evaluate right assignment := by
  simp [Form.sub, Fin.sub_eq_add_neg]

/-- The actual Rust equality encoding is `(left - right) * 1 = 0`. -/
def equalityImage (left right : LinComb) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.RowImage where
  a := Form.sub (combinationImage left) (combinationImage right)
  b := constantForm finalColumns_positive
  c := Form.zero

theorem equalityImage_accepted_iff
    (left right : LinComb) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1) :
    (equalityImage left right).Accepted 1 assignment ↔
      R1CS.lcEval (numericAssignment assignment) left =
        R1CS.lcEval (numericAssignment assignment) right := by
  rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.rowImage_accepted_iff_holds]
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.RowImage.Holds
    equalityImage
  rw [evaluate_sub, evaluate_constantForm, Form.evaluate_zero,
    constantOne, Fin.mul_one]
  let leftValue := R1CS.lcEval (numericAssignment assignment) left
  let rightValue := R1CS.lcEval (numericAssignment assignment) right
  have leftLt : leftValue < goldilocksP := by
    unfold leftValue R1CS.lcEval
    exact Nat.mod_lt _ (by decide)
  have rightLt : rightValue < goldilocksP := by
    unfold rightValue R1CS.lcEval
    exact Nat.mod_lt _ (by decide)
  rw [evaluate_combinationImage_eq_residue,
    evaluate_combinationImage_eq_residue]
  change
    NumericRowBridge.residue leftValue -
          NumericRowBridge.residue rightValue = 0 ↔
      leftValue = rightValue
  constructor
  · intro differenceZero
    apply NumericRowBridge.residue_injective_of_lt leftLt rightLt
    exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp differenceZero
  · intro equal
    rw [equal]
    exact Lean.Grind.AddCommGroup.sub_self _

def decodeImage (source : Source) (lane : Fin ringDegree) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.RowImage :=
  equalityImage
    [(sourceLayout.beforeChallenge source lane, 1)]
    (ProductPiRlcRingCombinationRows.centeredChallenge
      sourceLayout.algebra source lane)

def challengeImage
    (source : Source) (lane : Fin ringDegree) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.RowImage :=
  equalityImage
    [(sourceLayout.afterChallenge source lane, 1)]
    [(sourceLayout.beforeChallenge source lane, 1)]

def cursorImage :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.RowImage :=
  equalityImage
    [(sourceLayout.afterCursor, 1)]
    [(sourceLayout.beforeCursor, 1), (0, 1)]

/-- Exact acceptance predicate for all retained carry-row occurrences. -/
structure ProductionAccepted
    (arm : Arm) (assignment : Fin finalColumns → F) : Prop where
  selectorOne : assignment (selectorColumn arm) = 1
  decode : ∀ source lane,
    (decodeImage source lane).Accepted
      (assignment (selectorColumn arm)) assignment
  challenge : ∀ source lane,
    (challengeImage source lane).Accepted
      (assignment (selectorColumn arm)) assignment
  cursor : cursorImage.Accepted
    (assignment (selectorColumn arm)) assignment

def productionRowCount : Nat := 15 * 54 + 15 * 54 + 1

theorem productionRowCount_exact : productionRowCount = 1621 := by
  decide

/-- Every carried ring challenge is in the exact five-symbol production
strong set. -/
def ChallengesInStrongSet (challenges : Source → RingF) : Prop :=
  ∀ source, Phi81StrongSet.ProductionMember (challenges source)

private theorem field_add_right_cancel {left right suffix : F}
    (equal : left + suffix = right + suffix) : left = right := by
  calc
    left = (left + suffix) + -suffix := by
      rw [Lean.Grind.Fin.add_assoc,
        Lean.Grind.Fin.add_comm suffix (-suffix),
        Lean.Grind.Fin.neg_add_cancel, Fin.add_zero]
    _ = (right + suffix) + -suffix :=
      congrArg (fun value => value + -suffix) equal
    _ = right := by
      rw [Lean.Grind.Fin.add_assoc,
        Lean.Grind.Fin.add_comm suffix (-suffix),
        Lean.Grind.Fin.neg_add_cancel, Fin.add_zero]

private theorem decode_rows_satisfy
    {arm : Arm} {assignment : Fin finalColumns → F}
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (accepted : ProductionAccepted arm assignment) :
    R1CS.Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.decodeRows
        carryLayout)
      (numericAssignment assignment) := by
  intro row member
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.decodeRows at member
  rcases List.mem_flatten.mp member with ⟨laneRows, laneRowsMember,
    rowMember⟩
  rcases List.mem_ofFn.mp laneRowsMember with ⟨source, rfl⟩
  rcases List.mem_ofFn.mp rowMember with ⟨lane, rfl⟩
  apply (KEquality.equalityRow_iff (numericAssignment assignment)
    [(sourceLayout.beforeChallenge source lane, 1)]
    (ProductPiRlcRingCombinationRows.centeredChallenge
      sourceLayout.algebra source lane)
    (numericAssignment_one assignment constantOne)).mpr
  exact (equalityImage_accepted_iff _ _ assignment constantOne).mp
    (by simpa [accepted.selectorOne] using accepted.decode source lane)

private theorem challenge_rows_satisfy
    {arm : Arm} {assignment : Fin finalColumns → F}
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (accepted : ProductionAccepted arm assignment) :
    R1CS.Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.challengeRows
        carryLayout)
      (numericAssignment assignment) := by
  intro row member
  unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.challengeRows at member
  rcases List.mem_flatten.mp member with ⟨laneRows, laneRowsMember,
    rowMember⟩
  rcases List.mem_ofFn.mp laneRowsMember with ⟨source, rfl⟩
  rcases List.mem_ofFn.mp rowMember with ⟨lane, rfl⟩
  apply (KEquality.equalityRow_iff (numericAssignment assignment)
    [(sourceLayout.beforeChallenge source lane, 1)]
    [(sourceLayout.afterChallenge source lane, 1)]
    (numericAssignment_one assignment constantOne)).mpr
  exact ((equalityImage_accepted_iff _ _ assignment constantOne).mp
    (by simpa [accepted.selectorOne] using
      accepted.challenge source lane)).symm

private theorem cursor_row_satisfies
    {arm : Arm} {assignment : Fin finalColumns → F}
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (accepted : ProductionAccepted arm assignment) :
    RowHolds (numericAssignment assignment)
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.cursorRow
        carryLayout) := by
  apply (KEquality.equalityRow_iff (numericAssignment assignment)
    [(sourceLayout.afterCursor, 1)]
    [(sourceLayout.beforeCursor, 1), (0, 1)]
    (numericAssignment_one assignment constantOne)).mpr
  exact (equalityImage_accepted_iff _ _ assignment constantOne).mp
    (by simpa [accepted.selectorOne] using accepted.cursor)

/-- Accepted normalized rows imply the complete source carry-row list on the
same decoded assignment. -/
theorem productionAccepted_implies_source_rows
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (accepted : ProductionAccepted arm assignment) :
    R1CS.Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows
        carryLayout)
      (numericAssignment assignment) := by
  intro row member
  rcases List.mem_append.mp member with decodeMember | tailMember
  · exact decode_rows_satisfy constantOne accepted row decodeMember
  · rcases List.mem_append.mp tailMember with challengeMember | cursorMember
    · exact challenge_rows_satisfy constantOne accepted row challengeMember
    · have equal : row =
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.cursorRow
            carryLayout := by
        simpa using cursorMember
      subst row
      exact cursor_row_satisfies constantOne accepted

/-- The decode rows recover the five-symbol range from the authoritative
strong-set challenge already carried in the family state. No separate range
row is necessary. -/
theorem productionAccepted_implies_range
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (before after : FamilyState)
    (placed :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        carryLayout
      (numericAssignment assignment) before after)
    (strongSet : ChallengesInStrongSet before.challenges)
    (accepted : ProductionAccepted arm assignment) :
    ∀ source lane,
      numericAssignment assignment
          (sourceLayout.algebra.challengeSymbol source lane) < 5 := by
  intro source lane
  obtain ⟨scalar, challengeEqual⟩ := strongSet source
  let coefficient := scalar (Phi81StrongSet.scalarPosition lane)
  have rowEqual :=
    (equalityImage_accepted_iff _ _ assignment constantOne).mp
      (by simpa [accepted.selectorOne] using accepted.decode source lane)
  have beforeEqualsCentered :
      before.challenges source lane =
        ProductPiRlcRingCombinationSound.termsField
          (numericAssignment assignment)
          (ProductPiRlcRingCombinationRows.centeredChallenge
            sourceLayout.algebra source lane) := by
    have leftReduced :
        numericAssignment assignment
              (sourceLayout.beforeChallenge source lane) % goldilocksP =
          R1CS.lcEval (numericAssignment assignment)
            (ProductPiRlcRingCombinationRows.centeredChallenge
              sourceLayout.algebra source lane) := by
      simpa [R1CS.lcEval] using rowEqual
    have beforePlaced :
        numericAssignment assignment
            (sourceLayout.beforeChallenge source lane) =
          (before.challenges source lane).val := by
      simpa [carryLayout,
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout]
        using placed.1 source lane
    rw [beforePlaced] at leftReduced
    have beforeLt : (before.challenges source lane).val < goldilocksP := by
      simpa [goldilocksP, goldilocksModulus] using
        (before.challenges source lane).isLt
    rw [Nat.mod_eq_of_lt beforeLt] at leftReduced
    apply Fin.ext
    simpa [ProductPiRlcRingCombinationSound.termsField] using leftReduced
  have beforeEqualsEmbedded :
      before.challenges source lane =
        Phi81StrongSet.embedCoefficient coefficient := by
    have atLane := congrFun challengeEqual lane
    simpa [coefficient, Phi81StrongSet.embedScalar] using atLane
  have centeredEqualsEmbedded :
      ProductPiRlcRingCombinationSound.termsField
          (numericAssignment assignment)
          (ProductPiRlcRingCombinationRows.centeredChallenge
            sourceLayout.algebra source lane) =
        Phi81StrongSet.embedCoefficient coefficient :=
    beforeEqualsCentered.symm.trans beforeEqualsEmbedded
  have centeredValue := congrArg Fin.val centeredEqualsEmbedded
  have shiftedEqual :
      (numericAssignment assignment
            (sourceLayout.algebra.challengeSymbol source lane) +
          (goldilocksP - 2)) % goldilocksP =
        (coefficient.val + (goldilocksP - 2)) % goldilocksP := by
    rw [PiRlcCanonicalSelector.embedCoefficient_val_eq_shift] at centeredValue
    simpa [ProductPiRlcRingCombinationSound.termsField,
      ProductPiRlcRingCombinationRows.centeredChallenge, R1CS.lcEval,
      numericAssignment_one assignment constantOne, Nat.add_comm] using
      centeredValue
  have symbolLtModulus :
      numericAssignment assignment
          (sourceLayout.algebra.challengeSymbol source lane) <
        goldilocksModulus := by
    simpa [goldilocksP, goldilocksModulus] using
      numericAssignment_canonical assignment
        (sourceLayout.algebra.challengeSymbol source lane)
  have coefficientLtModulus : coefficient.val < goldilocksModulus := by
    have bounded := coefficient.isLt
    change coefficient.val < 5 at bounded
    simp only [goldilocksModulus]
    omega
  let symbolField : F :=
    ⟨numericAssignment assignment
      (sourceLayout.algebra.challengeSymbol source lane), symbolLtModulus⟩
  let coefficientField : F := ⟨coefficient.val, coefficientLtModulus⟩
  let offset : F :=
    ⟨goldilocksP - 2, by
      simp only [goldilocksP, goldilocksModulus]
      omega⟩
  have shiftedFieldEqual :
      symbolField + offset = coefficientField + offset := by
    apply Fin.ext
    simpa [symbolField, coefficientField, offset, goldilocksP,
      goldilocksModulus] using shiftedEqual
  have symbolEqual :=
    congrArg Fin.val (field_add_right_cancel shiftedFieldEqual)
  have coefficientLtFive : coefficient.val < 5 := by
    simpa [coefficient,
      Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.alphabetSize]
      using coefficient.isLt
  rw [show numericAssignment assignment
      (sourceLayout.algebra.challengeSymbol source lane) = coefficient.val by
    simpa [symbolField, coefficientField] using symbolEqual]
  exact coefficientLtFive

/-- Active normalized carry rows imply exact challenge decoding, challenge
carry, and cursor increment on the same final assignment. -/
theorem productionAccepted_implies_exact
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (range : ∀ source lane,
      numericAssignment assignment
          (sourceLayout.algebra.challengeSymbol source lane) < 5)
    (before after : FamilyState)
    (beforeBound : before.familyCursor < 110)
    (placed :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        carryLayout
      (numericAssignment assignment) before after)
    (accepted : ProductionAccepted arm assignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.Exact
      carryLayout (numericAssignment assignment) range before after := by
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows_sound
    (numericAssignment_canonical assignment)
    (numericAssignment_one assignment constantOne)
    range beforeBound placed
    (productionAccepted_implies_source_rows arm assignment constantOne accepted)

/-- Active normalized carry rows and the carried strong-set invariant imply
the complete exact carry result without a separate range premise. -/
theorem productionAccepted_implies_exact_of_strong_set
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (before after : FamilyState)
    (beforeBound : before.familyCursor < 110)
    (placed :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        carryLayout
      (numericAssignment assignment) before after)
    (strongSet : ChallengesInStrongSet before.challenges)
    (accepted : ProductionAccepted arm assignment) :
    let range := productionAccepted_implies_range arm assignment constantOne
      before after placed strongSet accepted
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.Exact
      carryLayout (numericAssignment assignment) range before after := by
  dsimp only
  exact productionAccepted_implies_exact arm assignment constantOne
    (productionAccepted_implies_range arm assignment constantOne before after
      placed strongSet accepted)
    before after beforeBound placed accepted

/-- The semantic constants are the exact Rust-conformant receipt geometry. -/
theorem receipt_geometry_exact :
    sourceColumns =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.audit.localColumns /\
      finalColumns =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.audit.finalColumns /\
      productionRowCount =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.audit.sourceRows /\
      directSourceStart + 640 = 144916 /\
      finalDirectStart = 1076045 := by
  native_decide

end Normalized

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows
