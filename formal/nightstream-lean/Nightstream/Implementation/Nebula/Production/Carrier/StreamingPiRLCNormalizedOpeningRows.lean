import Nightstream.Implementation.Nebula.Commitment.Lanes.ShiftedTernaryEncodingBridge
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedFamilyRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedLinkRows
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningArtifactRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CenteredDomainPacking
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRows
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange

/-!
Contract: exact normalized meaning of the retained production PiRLC input
opening rows.

Assurance tier: model-level row semantics, security-reduced centered-pair
packing, and a separate Rust-conformant exhaustive matrix receipt.

Owns the 16,605 packed active-digit rows, both 41-row zero words, all 1,620
copies of the 21-row canonical opening, their exact normalized coordinates,
and the implication from their acceptance to the body source-digit authority
used by the normalized field links.

Does not own the outer SuperNeo norm check or the proof that its committed
assignment contains the 32,400 borrow coordinates named here. It takes that
membership consequence as `BorrowCoordinatesNormFour`. It also does not own
selector scheduling, stored Rust assignment construction, recursive
orchestration, or commitment hardness.

Emits constraints: no. This file proves the meaning of rows already present
in the normalized production relation.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open Nightstream.SuperNeo.Concrete

namespace Normalized

private abbrev Source :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.Source

private abbrev Lane := Fin ringDegree
private abbrev Digit :=
  Fin Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount
private abbrev Borrow :=
  Fin Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkBorrowCount

abbrev BodyFinalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.BodyFinalColumns
abbrev Arm :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.Arm

theorem bodyFinalColumns_positive : 0 < BodyFinalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyFinalColumns_positive

def activeDigitCount : Nat :=
  810 * Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount
def centeredRowCount : Nat := activeDigitCount / 2

theorem activeDigitCount_exact : activeDigitCount = 33210 := by
  decide

theorem centeredRowCount_exact : centeredRowCount = 16605 := by
  decide

def selectorColumn (arm : Arm) : Fin BodyFinalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.selectorColumn arm

def constantColumn : Fin BodyFinalColumns :=
  ⟨0, bodyFinalColumns_positive⟩

def openingIndex (source : Source) (lane : Lane) : Fin 810 :=
  ⟨source.val * ringDegree + lane.val, by
    have sourceUpper := source.isLt
    have laneUpper := lane.isLt
    change source.val < 15 at sourceUpper
    change lane.val < 54 at laneUpper
    change source.val * 54 + lane.val < 810
    omega⟩

/-- One coordinate in the contiguous 33,210-coordinate active digit run. -/
def flatDigitValue
    (assignment : Fin BodyFinalColumns → F)
    (index : Fin activeDigitCount) : F :=
  assignment ⟨19332 + index.val, by
    have upper := index.isLt
    have upperConcrete : index.val < 33210 := by
      calc
        index.val < activeDigitCount := upper
        _ = 33210 := activeDigitCount_exact
    change 19332 + index.val < 2484972
    omega⟩

def bodyBorrowValue
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) (borrow : Borrow) : F :=
  assignment ⟨1059845 + (source.val * ringDegree + lane.val) * 20 +
      borrow.val, by
    have sourceUpper := source.isLt
    have laneUpper := lane.isLt
    have borrowUpper := borrow.isLt
    change source.val < 15 at sourceUpper
    change lane.val < 54 at laneUpper
    change borrow.val < 20 at borrowUpper
    change 1059845 + (source.val * 54 + lane.val) * 20 + borrow.val <
      2484972
    omega⟩

theorem bodyActiveDigitValue_eq_flat
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) (digit : Digit) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
        assignment source lane digit =
      flatDigitValue assignment
        ⟨(openingIndex source lane).val *
            Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount +
            digit.val, by
          have openingUpper := (openingIndex source lane).isLt
          have digitUpper := digit.isLt
          change digit.val < 41 at digitUpper
          change (openingIndex source lane).val * 41 + digit.val < 810 * 41
          omega⟩ := by
  apply congrArg assignment
  apply Fin.ext
  simp [openingIndex,
    Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount,
    Nightstream.SuperNeo.Concrete.ringDegree]
  omega

/-- Relabel one production opening to the exact physical columns of the
generated one-opening selective artifact. -/
def openingAssignment
    (arm : Arm) (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) : Nat → Nat := fun column =>
  if column = 0 then
    (assignment constantColumn).val
  else if column = 54 then
    (assignment (selectorColumn arm)).val
  else if digitRange : 108 ≤ column ∧ column < 149 then
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
      assignment source lane
      ⟨column - 108, by omega⟩).val
  else if borrowRange : 2363 ≤ column ∧ column < 2383 then
    (bodyBorrowValue assignment source lane
      ⟨column - 2363, by
        simp only [
          Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkBorrowCount,
          Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkCount]
        omega⟩).val
  else
    0

@[simp] theorem openingAssignment_zero
    (arm : Arm) (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) :
    openingAssignment arm assignment source lane 0 =
      (assignment constantColumn).val := by
  simp [openingAssignment]

@[simp] theorem openingAssignment_selector
    (arm : Arm) (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) :
    openingAssignment arm assignment source lane 54 =
      (assignment (selectorColumn arm)).val := by
  simp [openingAssignment]

@[simp] theorem openingAssignment_digit
    (arm : Arm) (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane)
    {index : Nat}
    (indexLt :
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount) :
    openingAssignment arm assignment source lane (108 + index) =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
        assignment source lane
        ⟨index, indexLt⟩).val := by
  simp [openingAssignment, show 108 + index ≠ 54 by omega,
    show 108 ≤ 108 + index ∧ 108 + index < 149 by
      change index < 41 at indexLt
      omega]

@[simp] theorem openingAssignment_borrow
    (arm : Arm) (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane)
    {index : Nat}
    (indexLt : index <
      Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkBorrowCount) :
    openingAssignment arm assignment source lane (2363 + index) =
      (bodyBorrowValue assignment source lane ⟨index, indexLt⟩).val := by
  simp [openingAssignment, show 2363 + index ≠ 54 by omega,
    show ¬ (108 ≤ 2363 + index ∧ 2363 + index < 149) by omega,
    show 2363 ≤ 2363 + index ∧ 2363 + index < 2383 by
      change index < 20 at indexLt
      omega]

/-- Semantic acceptance of the exact row families covered by the exhaustive
Rust opening-row receipt. -/
structure ProductionAccepted
    (arm : Arm) (assignment : Fin BodyFinalColumns → F) : Prop where
  centered : ∀ row : Fin centeredRowCount,
    Semantics.evaluate
      (Rows.centeredPairPoint (assignment constantColumn)
        (flatDigitValue assignment
          ⟨2 * row.val, by
            have upper := row.isLt
            have upperConcrete : row.val < 16605 := by
              calc
                row.val < centeredRowCount := upper
                _ = 16605 := centeredRowCount_exact
            change 2 * row.val < 810 * 41
            omega⟩)
        (flatDigitValue assignment
          ⟨2 * row.val + 1, by
            have upper := row.isLt
            have upperConcrete : row.val < 16605 := by
              calc
                row.val < centeredRowCount := upper
                _ = 16605 := centeredRowCount_exact
            change 2 * row.val + 1 < 810 * 41
            omega⟩)) = 0
  zero : ∀ digit : Digit,
    Semantics.evaluate
      (Rows.productPoint (assignment (selectorColumn arm))
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyZeroDigitValue
          assignment digit)
        (assignment constantColumn) 0) = 0
  canonical : ∀ source lane,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.ArtifactRowsHold
      (openingAssignment arm assignment source lane)

/-- Exact outer-authority consequence needed for the retained borrow slots.
The later complete relation must derive this predicate from the SuperNeo
`normBounded 4` opening and a proved coordinate-membership map. -/
def BorrowCoordinatesNormFour
    (assignment : Fin BodyFinalColumns → F) : Prop :=
  ∀ source lane borrow,
    Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.NormBoundFour
      (bodyBorrowValue assignment source lane borrow).val

private theorem centeredUnitResidual_eq_cubicResidual (value : F) :
    Components.centeredUnitResidual value =
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.cubicResidual
        value := by
  have coreCube :
      Lean.Grind.Fin.npow value 3 = value * value * value := by
    rw [show Lean.Grind.Fin.npow value 3 = value ^ 3 by rfl]
    simp only [pow_succ, pow_zero, one_mul]
  unfold Components.centeredUnitResidual
  change Lean.Grind.Fin.npow value 3 - value = _
  rw [coreCube]
  let equivalence : F ≃+* ZMod goldilocksModulus :=
    ZMod.finEquiv goldilocksModulus
  apply equivalence.injective
  simp only [
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.cubicResidual,
    map_sub, map_mul, map_add, map_one]
  ring

private theorem baseFieldNoZeroDivisors :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.BaseFieldNoZeroDivisors :=
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
    Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime

private theorem centeredResidual_implies_normTwo
    (value : F)
    (zero : Components.centeredUnitResidual value = 0) :
    Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.NormBoundTwo
      value.val := by
  have cubicZero :
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.cubicResidual
        value = 0 := by
    rw [← centeredUnitResidual_eq_cubicResidual]
    exact zero
  have strictNorm : centeredMagnitude value < 2 :=
    (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.cubicResidual_eq_zero_iff_strictNormTwo
      baseFieldNoZeroDivisors value).mp cubicZero
  exact
    Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.normBoundTwo_iff_centeredResidue.mpr
      ((Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.concrete_norm_two_iff_centeredResidue
        value).mp strictNorm)

theorem flatDigit_normTwo
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (accepted : ProductionAccepted arm assignment)
    (index : Fin activeDigitCount) :
    Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.NormBoundTwo
      (flatDigitValue assignment index).val := by
  let row : Fin centeredRowCount := ⟨index.val / 2, by
    have upper := index.isLt
    have upperConcrete : index.val < 33210 := by
      calc
        index.val < activeDigitCount := upper
        _ = 33210 := activeDigitCount_exact
    have quotientUpper : index.val / 2 < 16605 := by omega
    calc
      index.val / 2 < 16605 := quotientUpper
      _ = centeredRowCount := centeredRowCount_exact.symm⟩
  have rowUpperConcrete : row.val < 16605 := by
    calc
      row.val < centeredRowCount := row.isLt
      _ = 16605 := centeredRowCount_exact
  let leftIndex : Fin activeDigitCount := ⟨2 * row.val, by
    change 2 * row.val < 810 * 41
    omega⟩
  let rightIndex : Fin activeDigitCount := ⟨2 * row.val + 1, by
    change 2 * row.val + 1 < 810 * 41
    omega⟩
  have rowAccepted := accepted.centered row
  change Semantics.evaluate
    (Rows.centeredPairPoint (assignment constantColumn)
      (flatDigitValue assignment leftIndex)
      (flatDigitValue assignment rightIndex)) = 0 at rowAccepted
  rw [constantOne] at rowAccepted
  have residuals :=
    (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking.production_centeredPair_zero_iff
      (flatDigitValue assignment leftIndex)
      (flatDigitValue assignment rightIndex)).mp rowAccepted
  have division := Nat.mod_add_div index.val 2
  have remainderLt : index.val % 2 < 2 := Nat.mod_lt _ (by decide)
  rcases Nat.eq_zero_or_pos (index.val % 2) with remainderZero |
      remainderPositive
  · have leftIndexEq : leftIndex = index := by
      apply Fin.ext
      change 2 * (index.val / 2) = index.val
      omega
    have leftZero := residuals.1
    rw [leftIndexEq] at leftZero
    exact centeredResidual_implies_normTwo _ leftZero
  · have remainderOne : index.val % 2 = 1 := by omega
    have rightIndexEq : rightIndex = index := by
      apply Fin.ext
      change 2 * (index.val / 2) + 1 = index.val
      omega
    have rightZero := residuals.2
    rw [rightIndexEq] at rightZero
    exact centeredResidual_implies_normTwo _ rightZero

theorem activeDigit_normTwo
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (accepted : ProductionAccepted arm assignment)
    (source : Source) (lane : Lane) (digit : Digit) :
    Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.NormBoundTwo
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
        assignment source lane digit).val := by
  rw [bodyActiveDigitValue_eq_flat]
  exact flatDigit_normTwo constantOne accepted _

theorem zeroDigit_exact
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (selectorOne : assignment (selectorColumn arm) = 1)
    (accepted : ProductionAccepted arm assignment)
    (digit : Digit) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyZeroDigitValue
        assignment digit = 0 := by
  have rowAccepted := accepted.zero digit
  rw [constantOne, selectorOne] at rowAccepted
  have productZero :=
    (Rows.evaluate_productPoint_one_eq_zero_iff
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyZeroDigitValue
        assignment digit) 1 0).mp rowAccepted
  simpa using productZero

private theorem artifactDigitNorm
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (accepted : ProductionAccepted arm assignment)
    (source : Source) (lane : Lane) :
    Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.DigitNormBoundTwo
      (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.localAssignment
        (openingAssignment arm assignment source lane)) := by
  intro index indexLt
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.localAssignment_digit
    _ indexLt]
  change Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.NormBoundTwo
    (openingAssignment arm assignment source lane (108 + index))
  rw [openingAssignment_digit arm assignment source lane indexLt]
  exact activeDigit_normTwo constantOne accepted source lane ⟨index, indexLt⟩

private theorem artifactBorrowNorm
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (outerNorm : BorrowCoordinatesNormFour assignment)
    (source : Source) (lane : Lane) :
    ∀ index : Fin
        Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkBorrowCount,
      Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.NormBoundFour
        (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.localAssignment
          (openingAssignment arm assignment source lane)
          (Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkBorrowColumnBase +
            index.val)) := by
  intro index
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.localAssignment_borrow
    _ index.isLt]
  change Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.NormBoundFour
    (openingAssignment arm assignment source lane (2363 + index.val))
  rw [openingAssignment_borrow arm assignment source lane index.isLt]
  exact outerNorm source lane index

theorem openingChunkSchedule
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (selectorOne : assignment (selectorColumn arm) = 1)
    (outerNorm : BorrowCoordinatesNormFour assignment)
    (accepted : ProductionAccepted arm assignment)
    (source : Source) (lane : Lane) :
    Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.ChunkScheduleHolds
      (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.localAssignment
        (openingAssignment arm assignment source lane)) := by
  apply
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.artifactRows_imply_chunkScheduleHolds
  · simpa using congrArg Fin.val constantOne
  · simpa using congrArg Fin.val selectorOne
  · exact artifactDigitNorm constantOne accepted source lane
  · exact artifactBorrowNorm outerNorm source lane
  · exact accepted.canonical source lane

/-- Semantic canonical-opening assignment for one retained production input
field. The negative coordinates are deterministic functions of the active
centered digits and are not extra witness authority. -/
def canonicalAssignment
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) : Nat → Nat := fun column =>
  if column = Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol then
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
      assignment source lane).val
  else if digitRange : 58 ≤ column ∧ column < 99 then
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
      assignment source lane ⟨column - 58, by omega⟩).val
  else if negativeRange : 99 ≤ column ∧ column < 140 then
    Nightstream.Implementation.R1CS.CenteredTernaryField.negativeIndicator
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
        assignment source lane ⟨column - 99, by omega⟩).val
  else
    0

@[simp] theorem canonicalAssignment_field
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) :
    canonicalAssignment assignment source lane
        Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment source lane).val := by
  simp [canonicalAssignment,
    Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol]

@[simp] theorem canonicalAssignment_digit
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane)
    {index : Nat}
    (indexLt :
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount) :
    canonicalAssignment assignment source lane
        (Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.getD index 0) =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
        assignment source lane ⟨index, indexLt⟩).val := by
  rw [Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative.digitColumn_formula
    indexLt]
  have notField :
      58 + index ≠ Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol := by
    simp [Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol]
    omega
  have inDigits : 58 ≤ 58 + index ∧ 58 + index < 99 := by
    change index < 41 at indexLt
    omega
  unfold canonicalAssignment
  rw [if_neg notField, dif_pos inDigits]
  congr 2
  apply Fin.ext
  simp

@[simp] theorem canonicalAssignment_negative
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane)
    {index : Nat}
    (indexLt :
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount) :
    canonicalAssignment assignment source lane
        (Nightstream.Implementation.R1CS.ShiftedTernary.negativeCols.getD index 0) =
      Nightstream.Implementation.R1CS.CenteredTernaryField.negativeIndicator
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
          assignment source lane ⟨index, indexLt⟩).val := by
  rw [Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative.negativeColumn_formula
    indexLt]
  have notField :
      99 + index ≠ Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol := by
    simp [Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol]
    omega
  have outsideDigits : ¬ (58 ≤ 99 + index ∧ 99 + index < 99) := by
    omega
  have inNegatives : 99 ≤ 99 + index ∧ 99 + index < 140 := by
    change index < 41 at indexLt
    omega
  unfold canonicalAssignment
  rw [if_neg notField, dif_neg outsideDigits, dif_pos inNegatives]
  congr 3
  apply Fin.ext
  simp

private theorem digitsHold_of_atIndex
    {localValues : Nat → Nat}
    (atIndex : ∀ index,
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount →
      Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.Digit
        (localValues
          (Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.getD index 0))
        (localValues
          (Nightstream.Implementation.R1CS.ShiftedTernary.negativeCols.getD index 0))) :
    Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.DigitsHold
      localValues := by
  intro pair member
  rcases List.mem_iff_getElem.mp member with ⟨index, indexLt, pairEq⟩
  have lengths :
      Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.length =
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount ∧
        Nightstream.Implementation.R1CS.ShiftedTernary.negativeCols.length =
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount := by
    decide
  have digitLt :
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount := by
    simpa [List.length_zip, lengths.1, lengths.2] using indexLt
  have columns :
      (Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.zip
        Nightstream.Implementation.R1CS.ShiftedTernary.negativeCols)[index] =
        (Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.getD index 0,
          Nightstream.Implementation.R1CS.ShiftedTernary.negativeCols.getD index 0) := by
    rw [List.getElem_zip]
    simp only [List.getD_eq_getElem?_getD]
    have digitColumnLt :
        index < Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.length := by
      simpa [lengths.1] using digitLt
    have negativeColumnLt :
        index < Nightstream.Implementation.R1CS.ShiftedTernary.negativeCols.length := by
      simpa [lengths.2] using digitLt
    rw [List.getElem?_eq_getElem digitColumnLt,
      List.getElem?_eq_getElem negativeColumnLt]
    simp
  rw [← pairEq]
  exact columns.symm ▸ atIndex index digitLt

theorem canonicalAssignment_digits
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (accepted : ProductionAccepted arm assignment)
    (source : Source) (lane : Lane) :
    Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.DigitsHold
      (canonicalAssignment assignment source lane) := by
  apply digitsHold_of_atIndex
  intro index indexLt
  rw [canonicalAssignment_digit assignment source lane indexLt,
    canonicalAssignment_negative assignment source lane indexLt]
  apply Nightstream.Implementation.R1CS.CenteredTernaryField.digit_of_centeredResidue
  exact
    Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.normBoundTwo_iff_centeredResidue.mp
      (activeDigit_normTwo constantOne accepted source lane ⟨index, indexLt⟩)

private theorem artifactAssignment_digit_eq_canonical
    (arm : Arm) (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane)
    {index : Nat}
    (indexLt :
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount) :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.localAssignment
        (openingAssignment arm assignment source lane)
        (Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.getD index 0) =
      canonicalAssignment assignment source lane
        (Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.getD index 0) := by
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.localAssignment_digit
    _ indexLt]
  change openingAssignment arm assignment source lane (108 + index) = _
  rw [openingAssignment_digit arm assignment source lane indexLt,
    canonicalAssignment_digit assignment source lane indexLt]

theorem canonicalAssignment_encodedLt
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (selectorOne : assignment (selectorColumn arm) = 1)
    (outerNorm : BorrowCoordinatesNormFour assignment)
    (accepted : ProductionAccepted arm assignment)
    (source : Source) (lane : Lane) :
    Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.encodedValue
        (canonicalAssignment assignment source lane) <
      Nightstream.Implementation.R1CS.goldilocksP := by
  rw [Nightstream.Implementation.R1CS.ShiftedTernarySound.encodedValue_eq_lowValue]
  have schedule := openingChunkSchedule constantOne selectorOne outerNorm
    accepted source lane
  have artifactNorm := artifactDigitNorm constantOne accepted source lane
  have artifactLt :=
    Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.chunkSchedule_encoded_lt_modulus
      artifactNorm schedule
  apply lt_of_eq_of_lt _ artifactLt
  apply Nightstream.Implementation.R1CS.ShiftedTernaryComplete.lowValue_congr
  intro index indexLt
  unfold Nightstream.Implementation.R1CS.ShiftedTernarySound.assignmentTrit
    Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk.assignmentTritMod
  rw [← artifactAssignment_digit_eq_canonical arm assignment source lane indexLt]
  have bounded := artifactNorm index indexLt
  rw [Nat.mod_eq_of_lt bounded.1]

/-- Total numeric view of one active 41-coordinate production word. -/
def activeDigitAt
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) (index : Nat) : F :=
  if indexLt :
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount
  then
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
      assignment source lane ⟨index, indexLt⟩
  else
    0

@[simp] theorem activeDigitAt_of_lt
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane)
    {index : Nat}
    (indexLt :
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount) :
    activeDigitAt assignment source lane index =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
        assignment source lane ⟨index, indexLt⟩ := by
  unfold activeDigitAt
  rw [dif_pos indexLt]

/-- Field-valued little-endian radix-three fold. -/
def fieldLowValue (digits : Nat → F) : Nat → F
  | 0 => 0
  | count + 1 =>
      fieldLowValue digits count + digits count * (3 : F) ^ count

private theorem residue_three_pow (count : Nat) :
    Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue
        (3 ^ count) =
      (3 : F) ^ count := by
  induction count with
  | zero =>
      exact
        Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue_one
  | succ count inductionHypothesis =>
      have residueThree :
          Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue
              3 =
            (3 : F) := by
        apply Fin.ext
        rfl
      have fieldPowSucc :
          Lean.Grind.Fin.npow (3 : F) (count + 1) =
            Lean.Grind.Fin.npow (3 : F) count * 3 := by
        exact Lean.Grind.Fin.pow_succ (3 : F) count
      rw [Nat.pow_succ,
        Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue_mul,
        inductionHypothesis, residueThree]
      exact fieldPowSucc

private theorem residue_lowValue (digits : Nat → F) : ∀ count,
    Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue
        (Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
          (fun index => (digits index).val) count) =
      fieldLowValue digits count := by
  intro count
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      rw [Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue,
        fieldLowValue,
        Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue_add,
        Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue_mul,
        inductionHypothesis,
        Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue_field_val,
        residue_three_pow]

private theorem algebraInput_eq_fieldLowValue
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment source lane =
      fieldLowValue (activeDigitAt assignment source lane)
        Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount := by
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.decodedInputs
    Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.inputRing
    Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.wireField
    Nightstream.Implementation.Nebula.ProductPiDecLinearCombination.fieldAt
  have inputBelow :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.input
          source lane <
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.columns := by
    rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_input]
    have sourceUpper := source.isLt
    have laneUpper := lane.isLt
    change source.val < 15 at sourceUpper
    change lane.val < 54 at laneUpper
    change 811 + source.val * 54 + lane.val < 45415
    omega
  simp only [
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraAssignment,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.numericAssignment,
    inputBelow]
  apply Fin.ext
  simp only [dif_pos True.intro]
  have inputNonzero :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.input
          source lane ≠ 0 := by
    rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_input]
    omega
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumnValue
  simp only [dif_neg inputNonzero]
  have outsideChallenges :
      ¬ Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.input
          source lane < 811 := by
    rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_input]
    omega
  have insideInputs :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.input
          source lane < 1621 := by
    rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_input]
    have sourceUpper := source.isLt
    have laneUpper := lane.isLt
    change source.val < 15 at sourceUpper
    change lane.val < 54 at laneUpper
    omega
  let inputColumn : Fin
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumns :=
    ⟨Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.input
        source lane, by simpa using inputBelow⟩
  let inputSlot :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
      inputColumn (by simp [inputColumn])
  change
    (Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage.sourceSlotValue
      inputSlot assignment).val = _
  have inputSlotWidth : inputSlot.width = 41 := by
    unfold inputSlot inputColumn
    unfold
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
    rw [dif_neg outsideChallenges, dif_pos insideInputs]
  have inputSlotStart :
      inputSlot.start =
        19332 + (source.val * 54 + lane.val) * 41 := by
    unfold inputSlot inputColumn
    unfold
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
    rw [dif_neg outsideChallenges, dif_pos insideInputs]
    change
      19332 +
            (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.input
                source lane - 811) *
              41 =
        19332 + (source.val * 54 + lane.val) * 41
    rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_input]
    omega
  have slotCoordinateEq (index : Fin inputSlot.width) :
      assignment
          ⟨inputSlot.start + index.val,
            Nat.lt_of_lt_of_le
              (Nat.add_lt_add_left index.isLt inputSlot.start)
              inputSlot.columnsFit⟩ =
        activeDigitAt assignment source lane index.val := by
    have indexLt41 : index.val < 41 := by
      calc
        index.val < inputSlot.width := index.isLt
        _ = 41 := inputSlotWidth
    have indexLt :
        index.val <
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount := by
      simpa [Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount]
        using indexLt41
    rw [activeDigitAt_of_lt assignment source lane indexLt]
    unfold
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
    apply congrArg assignment
    apply Fin.ext
    change inputSlot.start + index.val =
      19332 + (source.val * 54 + lane.val) * 41 + index.val
    rw [inputSlotStart]
  have slotRadixEq :
      Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage.slotRadix
          inputSlot.width =
        (3 : F) := by
    rw [inputSlotWidth]
    rfl
  apply congrArg Fin.val
  rw [Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage.sourceSlotValue_eq_foldr]
  simp_rw [slotCoordinateEq, slotRadixEq]
  let equivalence : F ≃+* ZMod goldilocksModulus :=
    ZMod.finEquiv goldilocksModulus
  have mappedValue (value : F) :
      equivalence value = (value.val : ZMod goldilocksModulus) := by
    symm
    exact ZMod.natCast_zmod_val (equivalence value)
  apply equivalence.injective
  simp [
    inputSlotWidth,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount,
    activeDigitAt, fieldLowValue, map_add, map_mul]
  simp only [mappedValue]
  ring

private theorem canonicalCenteredDigit_eq_activeDigitAt
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane)
    {index : Nat}
    (indexLt :
      index < Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount) :
    Nightstream.Implementation.R1CS.ShiftedTernarySound.centeredDigit
        (canonicalAssignment assignment source lane) index =
      (activeDigitAt assignment source lane index).val := by
  unfold Nightstream.Implementation.R1CS.ShiftedTernarySound.centeredDigit
  rw [canonicalAssignment_digit assignment source lane indexLt,
    activeDigitAt_of_lt assignment source lane indexLt]

private theorem algebraInput_eq_residue_centeredLowValue
    (assignment : Fin BodyFinalColumns → F)
    (source : Source) (lane : Lane) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment source lane =
      Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue
        (Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
          (Nightstream.Implementation.R1CS.ShiftedTernarySound.centeredDigit
            (canonicalAssignment assignment source lane))
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount) := by
  have lowValuesEqual :
      Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
          (Nightstream.Implementation.R1CS.ShiftedTernarySound.centeredDigit
            (canonicalAssignment assignment source lane))
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount =
        Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
          (fun index => (activeDigitAt assignment source lane index).val)
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount := by
    apply Nightstream.Implementation.R1CS.ShiftedTernaryComplete.lowValue_congr
    intro index indexLt
    exact canonicalCenteredDigit_eq_activeDigitAt assignment source lane indexLt
  calc
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment source lane =
        fieldLowValue (activeDigitAt assignment source lane)
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount :=
      algebraInput_eq_fieldLowValue assignment source lane
    _ = Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue
          (Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
            (fun index => (activeDigitAt assignment source lane index).val)
            Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount) :=
      (residue_lowValue (activeDigitAt assignment source lane) _).symm
    _ = _ := by rw [← lowValuesEqual]

/-- Accepted normalized opening rows determine one canonical shifted-ternary
opening for every production PiRLC input field. -/
theorem accepted_implies_canonicalOpening
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (selectorOne : assignment (selectorColumn arm) = 1)
    (outerNorm : BorrowCoordinatesNormFour assignment)
    (accepted : ProductionAccepted arm assignment)
    (source : Source) (lane : Lane) :
    Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.CanonicalOpening
      (canonicalAssignment assignment source lane) := by
  let localOpening := canonicalAssignment assignment source lane
  have digits :
      Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.DigitsHold
        localOpening :=
    canonicalAssignment_digits constantOne accepted source lane
  have encodedLt :
      Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.encodedValue
          localOpening <
        Nightstream.Implementation.R1CS.goldilocksP :=
    canonicalAssignment_encodedLt constantOne selectorOne outerNorm accepted
      source lane
  have sourceField :=
    algebraInput_eq_residue_centeredLowValue assignment source lane
  have sourceValue := congrArg Fin.val sourceField
  have sourceMod :
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
          assignment source lane).val =
        Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
            (Nightstream.Implementation.R1CS.ShiftedTernarySound.centeredDigit
              localOpening)
            Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount %
          Nightstream.Implementation.R1CS.goldilocksP := by
    simpa [Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue,
      Nightstream.Implementation.R1CS.goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using sourceValue
  have encodedCongruence :=
    Nightstream.Implementation.R1CS.ShiftedTernarySound.encodedValue_centered_shift_mod_of_digits
      digits
  refine {
    digits := fun index indexLt => digits.atIndex indexLt
    encodedLt := encodedLt
    fieldMatches := ?_ }
  rw [canonicalAssignment_field]
  calc
    ((Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment source lane).val +
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.shift) %
        Nightstream.Implementation.R1CS.goldilocksP =
      (Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
            (Nightstream.Implementation.R1CS.ShiftedTernarySound.centeredDigit
              localOpening)
            Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount %
          Nightstream.Implementation.R1CS.goldilocksP +
        Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.shift) %
          Nightstream.Implementation.R1CS.goldilocksP := by rw [sourceMod]
    _ = (Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
            (Nightstream.Implementation.R1CS.ShiftedTernarySound.centeredDigit
              localOpening)
            Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.digitCount +
          Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.shift) %
        Nightstream.Implementation.R1CS.goldilocksP := by
      rw [Nat.mod_add_mod]
    _ = Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.encodedValue
          localOpening % Nightstream.Implementation.R1CS.goldilocksP :=
      encodedCongruence.symm
    _ = Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.encodedValue
          localOpening := Nat.mod_eq_of_lt encodedLt

theorem accepted_implies_activeDigitExact
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (selectorOne : assignment (selectorColumn arm) = 1)
    (outerNorm : BorrowCoordinatesNormFour assignment)
    (accepted : ProductionAccepted arm assignment)
    (source : Source) (lane : Lane) (digit : Digit) :
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
        assignment source lane digit).val =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue
        (Nightstream.Protocol.Nebula.CompactCommit.signedDigit
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.canonicalInput
            (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
              assignment source lane)) digit)).val := by
  let localOpening := canonicalAssignment assignment source lane
  let value :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.canonicalInput
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment source lane)
  have opening :
      Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.CanonicalOpening
        localOpening :=
    accepted_implies_canonicalOpening constantOne selectorOne outerNorm accepted
      source lane
  have nativeExact :=
    (Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.opening_digitPair_eq_native
      opening digit.val digit.isLt).1
  calc
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.bodyActiveDigitValue
        assignment source lane digit).val =
        localOpening
          (Nightstream.Implementation.R1CS.ShiftedTernary.digitCols.getD
            digit.val 0) := by
      symm
      exact canonicalAssignment_digit assignment source lane digit.isLt
    _ = Nightstream.Implementation.R1CS.ShiftedTernaryComplete.nativeDigit
          localOpening digit.val := nativeExact
    _ = Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.canonicalDigit
          (localOpening Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol)
          digit.val :=
      Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.nativeDigit_eq_canonicalDigit
        localOpening digit.val
    _ = Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.canonicalDigit
          value.val digit.val := by
      change
        Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.canonicalDigit
            (canonicalAssignment assignment source lane
              Nightstream.Implementation.R1CS.ShiftedTernary.fieldCol)
            digit.val =
          Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.canonicalDigit
            (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
              assignment source lane).val
            digit.val
      rw [canonicalAssignment_field]
    _ = Nightstream.Protocol.Nebula.ShiftedTernary41V1.fieldDigit
          (Nightstream.Protocol.Nebula.CompactCommit.tritAt value digit) :=
      Nightstream.Implementation.Nebula.ShiftedTernaryEncodingBridge.canonicalDigit_eq_fieldDigit_tritAt
        value digit
    _ = (Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue
          (Nightstream.Protocol.Nebula.CompactCommit.signedDigit value digit)).val :=
      (Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue_signedDigit
        value digit).symm

/-- The exact opening rows discharge the source-digit premise used by the
normalized body-to-overlay link proof. -/
theorem accepted_implies_bodySourceColumnsExact
    {arm : Arm} {assignment : Fin BodyFinalColumns → F}
    (constantOne : assignment constantColumn = 1)
    (selectorOne : assignment (selectorColumn arm) = 1)
    (outerNorm : BorrowCoordinatesNormFour assignment)
    (accepted : ProductionAccepted arm assignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.BodySourceColumnsExact
      assignment
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment) := by
  constructor
  · intro source lane digit
    exact accepted_implies_activeDigitExact constantOne selectorOne outerNorm
      accepted source lane digit
  · intro digit
    have zero := zeroDigit_exact constantOne selectorOne accepted digit
    have zeroValue := congrArg Fin.val zero
    simpa using zeroValue

/-- Accepted opening rows, link rows, and overlay rows now compose without an
external source-digit exactness premise. -/
theorem accepted_implies_bodyPhaseBindingPlaced
    {setup :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.InputBindingSetup}
    {family :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.Family}
    {arm : Arm}
    {linkSelector : F}
    {bodyAssignment : Fin BodyFinalColumns → F}
    {overlayAssignment : Fin
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.OverlayFinalColumns → F}
    (bodyConstantOne : bodyAssignment constantColumn = 1)
    (bodySelectorOne : bodyAssignment (selectorColumn arm) = 1)
    (linkSelectorOne : linkSelector = 1)
    (overlayConstantOne :
      overlayAssignment
          ⟨0,
            Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.overlayFinalColumns_positive⟩ =
        1)
    (outerNorm : BorrowCoordinatesNormFour bodyAssignment)
    (openingsAccepted : ProductionAccepted arm bodyAssignment)
    (linksAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.ProductionAccepted
        linkSelector bodyAssignment overlayAssignment)
    (overlayAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized.ProductionAccepted
        setup family overlayAssignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.PhaseBindingPlaced
      setup family
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        bodyAssignment)
      bodyAssignment := by
  apply
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized.accepted_implies_bodyPhaseBindingPlaced
      linkSelectorOne overlayConstantOne
  · exact accepted_implies_bodySourceColumnsExact bodyConstantOne
      bodySelectorOne outerNorm openingsAccepted
  · exact linksAccepted
  · exact overlayAccepted

end Normalized

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows
