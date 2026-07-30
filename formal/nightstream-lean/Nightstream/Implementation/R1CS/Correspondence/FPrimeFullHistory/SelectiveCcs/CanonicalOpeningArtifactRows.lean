import Nightstream.Implementation.R1CS.Artifacts.ShiftedTernary
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.BorrowChunk

/-!
Contract: interpret the generated 21-row selective canonical-opening artifact
and refine it to the two-trit borrow schedule.

Owns: exact evaluation of the generated `rowPorts` and `polynomialTerms`, the
single-row finite transition certificate, and sequential propagation of the
Boolean borrow invariant from the fixed zero endpoint.

Does not own: Split-NC truth, placement of an opening in a production
assignment, selector activation outside these 21 rows, or the final
canonical-opening theorem.

Assurance: the structural checks use kernel reduction. The closed
`finiteRowTransition` truth table uses `native_decide`; consequently that
theorem, and the refinement theorem depending on it, have the explicit
`Lean.trustCompiler` assurance boundary.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk

set_option maxRecDepth 262144

private def physicalDigitBase : Nat := 108
private def physicalBorrowBase : Nat := 2363

/-- Relabel the generated one-opening coordinates to the compact columns used
by `BorrowChunk`. No other physical column is given semantic meaning here. -/
def localAssignment (assignment : Nat → Nat) : Nat → Nat := fun column =>
  if 58 ≤ column ∧ column < 58 + digitCount then
    assignment (physicalDigitBase + (column - 58))
  else if
      chunkBorrowColumnBase ≤ column ∧
        column < chunkBorrowColumnBase + chunkBorrowCount then
    assignment
      (physicalBorrowBase + (column - chunkBorrowColumnBase))
  else
    0

@[simp] theorem localAssignment_digit
    (assignment : Nat → Nat) {index : Nat} (indexLt : index < digitCount) :
    localAssignment assignment
        (ShiftedTernary.digitCols.getD index 0) =
      assignment (physicalDigitBase + index) := by
  rw [digitColumn_formula indexLt]
  unfold localAssignment
  have digitRange :
      58 ≤ 58 + index ∧ 58 + index < 58 + digitCount := by
    omega
  rw [if_pos digitRange]
  simp

@[simp] theorem localAssignment_borrow
    (assignment : Nat → Nat) {index : Nat}
    (indexLt : index < chunkBorrowCount) :
    localAssignment assignment (chunkBorrowColumnBase + index) =
      assignment (physicalBorrowBase + index) := by
  unfold localAssignment
  have outsideDigit :
      ¬ (58 ≤ chunkBorrowColumnBase + index ∧
        chunkBorrowColumnBase + index < 58 + digitCount) := by
    unfold chunkBorrowColumnBase digitCount
    omega
  rw [if_neg outsideDigit]
  have borrowRange :
      chunkBorrowColumnBase ≤ chunkBorrowColumnBase + index ∧
        chunkBorrowColumnBase + index <
          chunkBorrowColumnBase + chunkBorrowCount := by
    omega
  rw [if_pos borrowRange]
  simp

/-- One generated sparse matrix port applied to a physical assignment. -/
def linearAction (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) : Nat :=
  lcEval assignment terms

/-- Typed action of one of the exact 21 rows at one of the exact 13 ports. -/
def rowAction (assignment : Nat → Nat)
    (row : Fin chunkCount) (port : Fin 13) : Nat :=
  linearAction assignment
    ((ShiftedTernarySelectiveArtifact.rowPorts.getD row.val []).getD
      port.val [])

private def rowTerms (row : Nat) : List (List (Nat × Nat)) :=
  ShiftedTernarySelectiveArtifact.rowPorts.getD row []

private def rowValues (assignment : Nat → Nat) (row : Nat) : List Nat :=
  (rowTerms row).map (linearAction assignment)

private def monomialValue
    (values : List Nat) (term : Nat × List Nat) : Nat :=
  (values.zip term.2).foldl
    (fun product entry =>
      product * (entry.1 ^ entry.2 % goldilocksP) % goldilocksP)
    (term.1 % goldilocksP)

private def polynomialValue (values : List Nat) : Nat :=
  ShiftedTernarySelectiveArtifact.polynomialTerms.foldl
    (fun total term =>
      (total + monomialValue values term) % goldilocksP)
    0

/-- Exact residual obtained from the generated row ports and generated
66-term selective polynomial. -/
def artifactResidual (assignment : Nat → Nat) (row : Nat) : Nat :=
  polynomialValue (rowValues assignment row)

/-- Satisfaction of every generated canonical-opening row. -/
def ArtifactRowsHold (assignment : Nat → Nat) : Prop :=
  ∀ row : Fin chunkCount, artifactResidual assignment row.val = 0

theorem artifact_row_count_exact :
    ShiftedTernarySelectiveArtifact.rowPorts.length = chunkCount := by
  decide

theorem artifact_port_arity_exact :
    ShiftedTernarySelectiveArtifact.rowPorts.all
      (fun ports => decide (ports.length = 13)) = true := by
  decide

theorem artifact_polynomial_term_count_exact :
    ShiftedTernarySelectiveArtifact.polynomialTerms.length = 66 := by
  decide

theorem artifact_exponent_arity_exact :
    ShiftedTernarySelectiveArtifact.polynomialTerms.all
      (fun term => decide (term.2.length = 13)) = true := by
  decide

private def supportColumns (row : Nat) : List Nat :=
  (rowTerms row).flatten.map Prod.fst

private theorem foldl_linear_congr
    (left right : Nat → Nat) (terms : List (Nat × Nat))
    (agree : ∀ term ∈ terms, left term.1 = right term.1)
    (initial : Nat) :
    terms.foldl (fun total term => total + term.2 * left term.1) initial =
      terms.foldl (fun total term => total + term.2 * right term.1) initial := by
  induction terms generalizing initial with
  | nil => rfl
  | cons term tail inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [agree term (by simp)]
      apply inductionHypothesis
      intro next nextMember
      exact agree next (by simp [nextMember])

private theorem linearAction_congr_on_support
    (left right : Nat → Nat) {row : Nat}
    {terms : List (Nat × Nat)} (termsMember : terms ∈ rowTerms row)
    (agree :
      ∀ column, column ∈ supportColumns row →
        left column = right column) :
    linearAction left terms = linearAction right terms := by
  unfold linearAction lcEval
  congr 1
  apply foldl_linear_congr
  intro term termMember
  apply agree term.1
  unfold supportColumns
  apply List.mem_map.mpr
  exact ⟨term,
    List.mem_flatten.mpr ⟨terms, termsMember, termMember⟩, rfl⟩

private theorem rowValues_congr_on_support
    (left right : Nat → Nat) {row : Nat}
    (agree :
      ∀ column, column ∈ supportColumns row →
        left column = right column) :
    rowValues left row = rowValues right row := by
  unfold rowValues
  apply List.map_congr_left
  intro terms termsMember
  exact linearAction_congr_on_support left right termsMember agree

theorem artifactResidual_congr_on_support
    (left right : Nat → Nat) {row : Nat}
    (agree :
      ∀ column, column ∈ supportColumns row →
        left column = right column) :
    artifactResidual left row = artifactResidual right row := by
  unfold artifactResidual
  rw [rowValues_congr_on_support left right agree]

private def firstDigitColumn (row : Nat) : Nat :=
  physicalDigitBase + 2 * row

private def secondDigitColumn (row : Nat) : Nat :=
  physicalDigitBase + 2 * row + 1

private def inputBorrowColumn (row : Nat) : Nat :=
  physicalBorrowBase + row - 1

private def outputBorrowColumn (row : Nat) : Nat :=
  physicalBorrowBase + row

private def RelevantColumn (row column : Nat) : Prop :=
  column = 0 ∨
  column = ShiftedTernarySelectiveArtifact.selectorColumn ∨
  column = firstDigitColumn row ∨
  (2 * row + 1 < digitCount ∧ column = secondDigitColumn row) ∨
  (0 < row ∧ column = inputBorrowColumn row) ∨
  (row + 1 < chunkCount ∧ column = outputBorrowColumn row)

private instance relevantColumnDecidable (row column : Nat) :
    Decidable (RelevantColumn row column) := by
  unfold RelevantColumn
  infer_instance

private def supportCoverageCheck : Bool :=
  (List.range chunkCount).all fun row =>
    (supportColumns row).all fun column =>
      decide (RelevantColumn row column)

private theorem supportCoverageCheck_true :
    supportCoverageCheck = true := by
  decide

private theorem support_relevant
    {row column : Nat} (rowLt : row < chunkCount)
    (member : column ∈ supportColumns row) :
    RelevantColumn row column := by
  have rowCheck :=
    (List.all_eq_true.mp supportCoverageCheck_true) row
      (List.mem_range.mpr rowLt)
  have columnCheck :=
    (List.all_eq_true.mp rowCheck) column member
  exact of_decide_eq_true columnCheck

private def firstValue (assignment : Nat → Nat) (row : Nat) : Nat :=
  assignment (firstDigitColumn row)

private def secondValue (assignment : Nat → Nat) (row : Nat) : Nat :=
  if 2 * row + 1 < digitCount then
    assignment (secondDigitColumn row)
  else
    goldilocksP - 1

private def inputValue (assignment : Nat → Nat) (row : Nat) : Nat :=
  chunkInputValue (localAssignment assignment) row

private def outputValue (assignment : Nat → Nat) (row : Nat) : Nat :=
  chunkOutputValue (localAssignment assignment) row

/-- Physical assignment containing exactly the six possible inputs to one
generated row. Absent first/last endpoints may be populated, but are never
read by the generated row. -/
private def caseAssignment
    (row first second input output : Nat) : Nat → Nat := fun column =>
  if column = 0 then 1
  else if column = ShiftedTernarySelectiveArtifact.selectorColumn then 1
  else if column = firstDigitColumn row then first
  else if column = secondDigitColumn row then second
  else if column = inputBorrowColumn row then input
  else if column = outputBorrowColumn row then output
  else 0

private theorem inputValue_eq_physical
    (assignment : Nat → Nat)
    (borrowNorm : ∀ index : Fin chunkBorrowCount,
      NormBoundTwo
        (localAssignment assignment
          (chunkBorrowColumnBase + index.val)))
    {row : Nat} (rowLt : row < chunkCount) (positive : 0 < row) :
    inputValue assignment row =
      assignment (inputBorrowColumn row) := by
  have indexLt : row - 1 < chunkBorrowCount := by
    unfold chunkBorrowCount chunkCount at *
    omega
  have bounded := borrowNorm ⟨row - 1, indexLt⟩
  rw [localAssignment_borrow assignment indexLt] at bounded
  have rowNe : row ≠ 0 := by omega
  have coordinateEq :
      physicalBorrowBase + (row - 1) = inputBorrowColumn row := by
    unfold inputBorrowColumn physicalBorrowBase
    omega
  rw [coordinateEq] at bounded
  have physicalEq :
      localAssignment assignment (chunkBorrowColumnBase + row - 1) =
        assignment (inputBorrowColumn row) := by
    calc
      localAssignment assignment (chunkBorrowColumnBase + row - 1) =
          localAssignment assignment
            (chunkBorrowColumnBase + (row - 1)) := by
              congr 2
              omega
      _ = assignment (physicalBorrowBase + (row - 1)) :=
        localAssignment_borrow assignment indexLt
      _ = assignment (inputBorrowColumn row) := by rw [coordinateEq]
  simp [inputValue, chunkInputValue, chunkInput,
    CenteredTernaryDerivedBorrow.Polynomial.eval,
    rowNe, physicalEq, Nat.mod_eq_of_lt bounded.1]

private theorem outputValue_eq_physical
    (assignment : Nat → Nat)
    (borrowNorm : ∀ index : Fin chunkBorrowCount,
      NormBoundTwo
        (localAssignment assignment
          (chunkBorrowColumnBase + index.val)))
    {row : Nat} (rowLt : row < chunkCount)
    (hasOutput : row + 1 < chunkCount) :
    outputValue assignment row =
      assignment (outputBorrowColumn row) := by
  have indexLt : row < chunkBorrowCount := by
    unfold chunkBorrowCount chunkCount at *
    omega
  have bounded := borrowNorm ⟨row, indexLt⟩
  rw [localAssignment_borrow assignment indexLt] at bounded
  have notLast : row + 1 ≠ chunkCount := by omega
  change NormBoundTwo
    (assignment (outputBorrowColumn row)) at bounded
  have physicalEq :
      localAssignment assignment (chunkBorrowColumnBase + row) =
        assignment (outputBorrowColumn row) := by
    rw [localAssignment_borrow assignment indexLt]
    rfl
  simp [outputValue, chunkOutputValue, chunkOutput,
    CenteredTernaryDerivedBorrow.Polynomial.eval,
    notLast, physicalEq, Nat.mod_eq_of_lt bounded.1]

private theorem artifactResidual_eq_case
    (assignment : Nat → Nat)
    (one : assignment 0 = 1)
    (selector :
      assignment ShiftedTernarySelectiveArtifact.selectorColumn = 1)
    (borrowNorm : ∀ index : Fin chunkBorrowCount,
      NormBoundTwo
        (localAssignment assignment
          (chunkBorrowColumnBase + index.val)))
    {row : Nat} (rowLt : row < chunkCount) :
    artifactResidual assignment row =
      artifactResidual
        (caseAssignment row
          (firstValue assignment row)
          (secondValue assignment row)
          (inputValue assignment row)
          (outputValue assignment row))
        row := by
  have rowLtConcrete : row < 21 := by
    simpa [chunkCount] using rowLt
  apply artifactResidual_congr_on_support
  intro column columnMember
  rcases support_relevant rowLt columnMember with
    zero | selectorColumn | first | second | input | output
  · subst column
    unfold caseAssignment
    rw [if_pos rfl]
    exact one
  · subst column
    have notZero :
        ShiftedTernarySelectiveArtifact.selectorColumn ≠ 0 := by
      unfold ShiftedTernarySelectiveArtifact.selectorColumn
      omega
    unfold caseAssignment
    rw [if_neg notZero, if_pos rfl]
    exact selector
  · subst column
    have notZero : firstDigitColumn row ≠ 0 := by
      unfold firstDigitColumn physicalDigitBase
      omega
    have notSelector :
        firstDigitColumn row ≠
          ShiftedTernarySelectiveArtifact.selectorColumn := by
      unfold firstDigitColumn physicalDigitBase
        ShiftedTernarySelectiveArtifact.selectorColumn
      omega
    unfold caseAssignment
    rw [if_neg notZero, if_neg notSelector, if_pos rfl]
    unfold firstValue
    rfl
  · rcases second with ⟨live, rfl⟩
    have notZero : secondDigitColumn row ≠ 0 := by
      unfold secondDigitColumn physicalDigitBase
      omega
    have notSelector :
        secondDigitColumn row ≠
          ShiftedTernarySelectiveArtifact.selectorColumn := by
      unfold secondDigitColumn physicalDigitBase
        ShiftedTernarySelectiveArtifact.selectorColumn
      omega
    have notFirst :
        secondDigitColumn row ≠ firstDigitColumn row := by
      unfold secondDigitColumn firstDigitColumn
      omega
    unfold caseAssignment
    rw [if_neg notZero, if_neg notSelector, if_neg notFirst, if_pos rfl]
    simp [secondValue, live]
  · rcases input with ⟨positive, rfl⟩
    rw [inputValue_eq_physical assignment borrowNorm rowLt positive]
    have notZero : inputBorrowColumn row ≠ 0 := by
      unfold inputBorrowColumn physicalBorrowBase
      omega
    have notSelector :
        inputBorrowColumn row ≠
          ShiftedTernarySelectiveArtifact.selectorColumn := by
      change 2363 + row - 1 ≠ 54
      omega
    have notFirst :
        inputBorrowColumn row ≠ firstDigitColumn row := by
      change 2363 + row - 1 ≠ 108 + 2 * row
      omega
    have notSecond :
        inputBorrowColumn row ≠ secondDigitColumn row := by
      change 2363 + row - 1 ≠ 108 + 2 * row + 1
      omega
    unfold caseAssignment
    rw [if_neg notZero, if_neg notSelector, if_neg notFirst,
      if_neg notSecond, if_pos rfl]
  · rcases output with ⟨hasOutput, rfl⟩
    rw [outputValue_eq_physical assignment borrowNorm rowLt hasOutput]
    have notZero : outputBorrowColumn row ≠ 0 := by
      unfold outputBorrowColumn physicalBorrowBase
      omega
    have notSelector :
        outputBorrowColumn row ≠
          ShiftedTernarySelectiveArtifact.selectorColumn := by
      change 2363 + row ≠ 54
      omega
    have notFirst :
        outputBorrowColumn row ≠ firstDigitColumn row := by
      change 2363 + row ≠ 108 + 2 * row
      omega
    have notSecond :
        outputBorrowColumn row ≠ secondDigitColumn row := by
      change 2363 + row ≠ 108 + 2 * row + 1
      omega
    have notInput :
        outputBorrowColumn row ≠ inputBorrowColumn row := by
      change 2363 + row ≠ 2363 + row - 1
      omega
    unfold caseAssignment
    rw [if_neg notZero, if_neg notSelector, if_neg notFirst,
      if_neg notSecond, if_neg notInput, if_pos rfl]

private def centeredValue : Fin 3 → Nat
  | ⟨0, _⟩ => goldilocksP - 1
  | ⟨1, _⟩ => 0
  | ⟨2, _⟩ => 1

private def binaryValue (value : Fin 2) : Nat :=
  value.val

private theorem centeredValue_surjective
    {value : Nat} (centered : CenteredResidue value) :
    ∃ index : Fin 3, centeredValue index = value := by
  rcases centered with negative | zero | one
  · exact ⟨⟨0, by decide⟩, negative.symm⟩
  · exact ⟨⟨1, by decide⟩, zero.symm⟩
  · exact ⟨⟨2, by decide⟩, one.symm⟩

private theorem binaryValue_surjective
    {value : Nat} (bounded : value ≤ 1) :
    ∃ index : Fin 2, binaryValue index = value := by
  by_cases zero : value = 0
  · exact ⟨⟨0, by decide⟩, by simp [binaryValue, zero]⟩
  · have one : value = 1 := by omega
    exact ⟨⟨1, by decide⟩, by simp [binaryValue, one]⟩

private def caseInput (row input : Nat) : Nat :=
  if row = 0 then 0 else input

private def caseSecond (row second : Nat) : Nat :=
  if 2 * row + 1 < digitCount then second else goldilocksP - 1

private def caseOutput (row output : Nat) : Nat :=
  if row + 1 = chunkCount then 0 else output

private def caseDigitAssignment
    (row first second : Nat) : Nat → Nat := fun column =>
  if column = ShiftedTernary.digitCols.getD (2 * row) 0 then first
  else if column =
      ShiftedTernary.digitCols.getD (2 * row + 1) 0 then second
  else 0

private def caseScalarValue
    (row first second input : Nat) : Nat :=
  scalarCompose (caseDigitAssignment row first second)
    (chunkEntries row) (caseInput row input)

/-- Closed exact truth table for all 21 generated rows. The incoming borrow
is Boolean; the outgoing endpoint is only assumed centered. The conclusion
both identifies the intended two-trit transition and establishes the Boolean
invariant needed by the next row.

This is the sole native-code certificate in the refinement. -/
theorem finiteRowTransition :
    ∀ row : Fin chunkCount,
    ∀ first second output : Fin 3,
    ∀ input : Fin 2,
      artifactResidual
          (caseAssignment row.val
            (centeredValue first)
            (caseSecond row.val (centeredValue second))
            (caseInput row.val (binaryValue input))
            (caseOutput row.val (centeredValue output)))
          row.val = 0 →
        caseOutput row.val (centeredValue output) =
            caseScalarValue row.val
              (centeredValue first)
              (caseSecond row.val (centeredValue second))
              (caseInput row.val (binaryValue input)) ∧
          caseOutput row.val (centeredValue output) ≤ 1 := by
  native_decide

private theorem firstValue_centered
    (assignment : Nat → Nat)
    (digitNorm : DigitNormBoundTwo (localAssignment assignment))
    {row : Nat} (rowLt : row < chunkCount) :
    CenteredResidue (firstValue assignment row) := by
  have indexLt : 2 * row < digitCount := by
    unfold chunkCount digitCount at *
    omega
  have bounded := digitNorm (2 * row) indexLt
  rw [localAssignment_digit assignment indexLt] at bounded
  simpa [firstValue, firstDigitColumn, physicalDigitBase] using
    (normBoundTwo_iff_centeredResidue.mp bounded)

private theorem secondValue_centered
    (assignment : Nat → Nat)
    (digitNorm : DigitNormBoundTwo (localAssignment assignment))
    (row : Nat) :
    CenteredResidue (secondValue assignment row) := by
  by_cases live : 2 * row + 1 < digitCount
  · have bounded := digitNorm (2 * row + 1) live
    rw [localAssignment_digit assignment live] at bounded
    unfold secondValue
    rw [if_pos live]
    have centered := normBoundTwo_iff_centeredResidue.mp bounded
    unfold secondDigitColumn
    simpa [Nat.add_assoc] using centered
  · left
    simp [secondValue, live]

private theorem outputValue_centered
    (assignment : Nat → Nat)
    (borrowNorm : ∀ index : Fin chunkBorrowCount,
      NormBoundTwo
        (localAssignment assignment
          (chunkBorrowColumnBase + index.val)))
    {row : Nat} (rowLt : row < chunkCount) :
    CenteredResidue (outputValue assignment row) := by
  by_cases last : row + 1 = chunkCount
  · right
    left
    simp [outputValue, chunkOutputValue, chunkOutput,
      CenteredTernaryDerivedBorrow.Polynomial.eval, last]
  · have indexLt : row < chunkBorrowCount := by
      unfold chunkBorrowCount chunkCount at *
      omega
    have bounded := borrowNorm ⟨row, indexLt⟩
    have outputEq :=
      outputValue_eq_physical assignment borrowNorm rowLt (by omega)
    rw [outputEq]
    unfold outputBorrowColumn
    rw [← localAssignment_borrow assignment indexLt]
    exact normBoundTwo_iff_centeredResidue.mp bounded

private theorem inputValue_case
    (assignment : Nat → Nat) {row : Nat} :
    caseInput row (inputValue assignment row) =
      inputValue assignment row := by
  by_cases zero : row = 0
  · subst row
    simp [caseInput, inputValue, chunkInputValue, chunkInput,
      CenteredTernaryDerivedBorrow.Polynomial.eval]
  · simp [caseInput, zero]

private theorem secondValue_case
    (assignment : Nat → Nat) (row : Nat) :
    caseSecond row (secondValue assignment row) =
      secondValue assignment row := by
  by_cases live : 2 * row + 1 < digitCount <;>
    simp [caseSecond, secondValue, live]

private theorem outputValue_case
    (assignment : Nat → Nat) (row : Nat) :
    caseOutput row (outputValue assignment row) =
      outputValue assignment row := by
  by_cases last : row + 1 = chunkCount
  · simp [caseOutput, outputValue, chunkOutputValue, chunkOutput,
      CenteredTernaryDerivedBorrow.Polynomial.eval, last]
  · simp [caseOutput, last]

private theorem scalarCompose_congr
    (left right : Nat → Nat) (entries : List (Nat × Nat))
    (agree : ∀ entry ∈ entries, left entry.2 = right entry.2)
    (input : Nat) :
    scalarCompose left entries input =
      scalarCompose right entries input := by
  induction entries generalizing input with
  | nil => rfl
  | cons entry tail inductionHypothesis =>
      simp only [scalarCompose]
      rw [agree entry (by simp)]
      apply inductionHypothesis
      intro next nextMember
      exact agree next (by simp [nextMember])

private theorem chunkEntry_case_agrees
    (assignment : Nat → Nat) {row : Nat} (rowLt : row < chunkCount) :
    ∀ entry ∈ chunkEntries row,
      localAssignment assignment entry.2 =
        caseDigitAssignment row
          (firstValue assignment row)
          (secondValue assignment row) entry.2 := by
  intro entry entryMember
  rcases List.mem_map.mp entryMember with
    ⟨offset, offsetMember, rfl⟩
  dsimp
  have offsetLt : offset < chunkLength row :=
    List.mem_range.mp offsetMember
  have offsetCases : offset = 0 ∨ offset = 1 := by
    unfold chunkLength chunkWidth at offsetLt
    omega
  rcases offsetCases with rfl | rfl
  · have indexLt : 2 * row < digitCount := by
      unfold chunkCount digitCount at *
      omega
    have startEq : chunkStart row = 2 * row := by
      unfold chunkStart chunkWidth
      omega
    simp only [Nat.add_zero]
    rw [startEq, localAssignment_digit assignment indexLt]
    unfold caseDigitAssignment
    rw [if_pos rfl]
    unfold firstValue firstDigitColumn
    rfl
  · have live : 2 * row + 1 < digitCount := by
      unfold chunkLength chunkStart chunkWidth at offsetLt
      omega
    have firstLt : 2 * row < digitCount := by omega
    have startEq : chunkStart row + 1 = 2 * row + 1 := by
      unfold chunkStart chunkWidth
      omega
    have distinct :
        ShiftedTernary.digitCols.getD (2 * row + 1) 0 ≠
          ShiftedTernary.digitCols.getD (2 * row) 0 := by
      rw [digitColumn_formula live, digitColumn_formula firstLt]
      omega
    rw [startEq, localAssignment_digit assignment live]
    unfold caseDigitAssignment
    rw [if_neg distinct, if_pos rfl]
    unfold secondValue
    rw [if_pos live]
    unfold secondDigitColumn
    congr 1

private theorem scalarChunkValue_eq_caseScalarValue
    (assignment : Nat → Nat) {row : Nat} (rowLt : row < chunkCount) :
    scalarChunkValue (localAssignment assignment) row =
      caseScalarValue row
        (firstValue assignment row)
        (secondValue assignment row)
        (inputValue assignment row) := by
  unfold scalarChunkValue caseScalarValue
  rw [inputValue_case]
  exact scalarCompose_congr
    (localAssignment assignment)
    (caseDigitAssignment row
      (firstValue assignment row)
      (secondValue assignment row))
    (chunkEntries row)
    (chunkEntry_case_agrees assignment rowLt)
    (inputValue assignment row)

private theorem artifactChunk_sound
    (assignment : Nat → Nat)
    (one : assignment 0 = 1)
    (selector :
      assignment ShiftedTernarySelectiveArtifact.selectorColumn = 1)
    (digitNorm : DigitNormBoundTwo (localAssignment assignment))
    (borrowNorm : ∀ index : Fin chunkBorrowCount,
      NormBoundTwo
        (localAssignment assignment
          (chunkBorrowColumnBase + index.val)))
    {row : Nat} (rowLt : row < chunkCount)
    (inputLe : inputValue assignment row ≤ 1)
    (rowZero : artifactResidual assignment row = 0) :
    (chunkEquation row).Holds (localAssignment assignment) ∧
      outputValue assignment row ≤ 1 := by
  rcases centeredValue_surjective
      (firstValue_centered assignment digitNorm rowLt) with
    ⟨firstIndex, firstEq⟩
  rcases centeredValue_surjective
      (secondValue_centered assignment digitNorm row) with
    ⟨secondIndex, secondEq⟩
  rcases centeredValue_surjective
      (outputValue_centered assignment borrowNorm rowLt) with
    ⟨outputIndex, outputEq⟩
  rcases binaryValue_surjective inputLe with ⟨inputIndex, inputEq⟩
  have caseZero :
      artifactResidual
          (caseAssignment row
            (centeredValue firstIndex)
            (caseSecond row (centeredValue secondIndex))
            (caseInput row (binaryValue inputIndex))
            (caseOutput row (centeredValue outputIndex)))
          row = 0 := by
    rw [firstEq, secondEq, inputEq, outputEq,
      secondValue_case, inputValue_case, outputValue_case]
    rw [← artifactResidual_eq_case assignment one selector borrowNorm rowLt]
    exact rowZero
  have checked :=
    finiteRowTransition ⟨row, rowLt⟩ firstIndex secondIndex outputIndex
      inputIndex caseZero
  rw [firstEq, secondEq, inputEq, outputEq,
    secondValue_case, inputValue_case, outputValue_case] at checked
  have scalarEq :
      outputValue assignment row =
        scalarChunkValue (localAssignment assignment) row := by
    rw [scalarChunkValue_eq_caseScalarValue assignment rowLt]
    exact checked.1
  constructor
  · exact
      (chunkEquation_holds_iff_scalar
        (localAssignment assignment) rowLt
        (chunkDigitsCentered_of_norm digitNorm rowLt)
        inputLe).mpr scalarEq
  · exact checked.2

/-- The generated rows refine the full 21-equation borrow schedule. Borrow
Booleanity is not assumed. It is proved row-by-row, beginning with the fixed
zero input and propagated through the shared endpoint columns. -/
theorem artifactRows_imply_chunkScheduleHolds
    (assignment : Nat → Nat)
    (one : assignment 0 = 1)
    (selector :
      assignment ShiftedTernarySelectiveArtifact.selectorColumn = 1)
    (digitNorm : DigitNormBoundTwo (localAssignment assignment))
    (borrowNorm : ∀ index : Fin chunkBorrowCount,
      NormBoundTwo
        (localAssignment assignment
          (chunkBorrowColumnBase + index.val)))
    (rowsHold : ArtifactRowsHold assignment) :
    ChunkScheduleHolds (localAssignment assignment) := by
  have transitions :
      ∀ row, row < chunkCount →
        (chunkEquation row).Holds (localAssignment assignment) ∧
          outputValue assignment row ≤ 1 := by
    intro row
    induction row with
    | zero =>
        intro rowLt
        apply artifactChunk_sound assignment one selector digitNorm borrowNorm
          rowLt
        · simp [inputValue, chunkInputValue, chunkInput,
            CenteredTernaryDerivedBorrow.Polynomial.eval]
        · exact rowsHold ⟨0, rowLt⟩
    | succ previous inductionHypothesis =>
        intro rowLt
        have previousLt : previous < chunkCount := by omega
        have previousResult := inductionHypothesis previousLt
        have aliasPolynomial :=
          adjacentChunkBorrowAlias (chunk := previous) (by omega)
        have aliasValue := congrArg
          (fun polynomial => polynomial.eval (localAssignment assignment))
          aliasPolynomial
        have inputEq :
            inputValue assignment (Nat.succ previous) =
              outputValue assignment previous := by
          simpa [inputValue, outputValue, chunkInputValue,
            chunkOutputValue, Nat.succ_eq_add_one] using aliasValue.symm
        apply artifactChunk_sound assignment one selector digitNorm borrowNorm
          rowLt
        · rw [inputEq]
          exact previousResult.2
        · exact rowsHold ⟨Nat.succ previous, rowLt⟩
  intro row rowLt
  exact (transitions row rowLt).1

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows
