import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FixtureRefinement
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.PolynomialCertificate

/-!
Contract: source-row refinement for the generated grouped-product fixture.

Assurance tier: artifact-checked fixture.

Owns: exact polynomial identities between the 33 decoded source R1CS rows and
the three two-row grouped-product identities in the generated Rust fixture.

Does not own: production-family coverage, a general symbolic-normal-form
checker, final-assignment construction, the reverse existential source-witness
direction, or permission to remove production rows or coordinates.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.PolynomialCertificate
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation

def sourceRow (index : Fin 33) :
    DecodedSourceR1csRow sourceRowCount sourceColumnCount :=
  decodedSourceRows.get ⟨index.val, by
    rw [decodedSourceRows_length]
    exact index.isLt⟩

def residualAt (index : Fin 33)
    (assignment : Fin sourceColumnCount → F) : F :=
  rowResidual (sourceRow index) assignment 1

def contribution (index : Fin 6)
    (assignment : Fin sourceColumnCount → F) : F :=
  linearValue (decodedStep index).base assignment 1 +
    factorSum assignment 1 (decodedStep index).factors

/-- Residual of one complete two-row product-sum identity after its private
carry is eliminated. -/
def identityResidual (first second : Fin 6)
    (assignment : Fin sourceColumnCount → F) : F :=
  outputValue assignment 1 (fun _ => 0) (decodedStep second).output -
    (contribution first assignment + contribution second assignment)

def sourceRowPolynomial (index : Fin 33) : Polynomial sourceColumnCount :=
  rowPolynomial (sourceRow index)

def sourceRowPolynomialSum :
    List (Fin 33) → Polynomial sourceColumnCount
  | [] => []
  | index :: tail =>
      add (sourceRowPolynomial index) (sourceRowPolynomialSum tail)

def residualSum (assignment : Fin sourceColumnCount → F) :
    List (Fin 33) → F
  | [] => 0
  | index :: tail => residualAt index assignment + residualSum assignment tail

theorem evaluate_sourceRowPolynomial (index : Fin 33)
    (assignment : Fin sourceColumnCount → F) :
    evaluate assignment (sourceRowPolynomial index) =
      certificateValue (residualAt index assignment) := by
  simp only [sourceRowPolynomial, residualAt, evaluate_rowPolynomial]

theorem evaluate_sourceRowPolynomialSum (indices : List (Fin 33))
    (assignment : Fin sourceColumnCount → F) :
    evaluate assignment (sourceRowPolynomialSum indices) =
      certificateValue (residualSum assignment indices) := by
  induction indices with
  | nil => simp [sourceRowPolynomialSum, residualSum, evaluate]
  | cons head tail inductionHypothesis =>
      simp only [sourceRowPolynomialSum, residualSum, evaluate_add,
        evaluate_sourceRowPolynomial, inductionHypothesis,
        certificateValue_add]

theorem evaluate_identityResidual (first second : Fin 6)
    (assignment : Fin sourceColumnCount → F) :
    evaluate assignment
        (identityPolynomial (decodedStep first) (decodedStep second)) =
      certificateValue (identityResidual first second assignment) := by
  simp only [evaluate_identityPolynomial, identityResidual, contribution]

/-- First product-sum identity expressed only as source-row residuals. -/
def qCertificatePolynomial : Polynomial sourceColumnCount :=
  sub (sourceRowPolynomial 30)
    (sourceRowPolynomialSum [1, 6, 11, 16, 21, 26])

/-- Native normalization proves that the executable rewrite identity and the
exact 33-row source fragment have the same quadratic polynomial. -/
theorem q_polynomial_exact :
    identityPolynomial (decodedStep 0) (decodedStep 1) =
      qCertificatePolynomial := by
  native_decide

def qCertificateResidual
    (assignment : Fin sourceColumnCount → F) : F :=
  residualAt 30 assignment -
    residualSum assignment [1, 6, 11, 16, 21, 26]

theorem evaluate_qCertificatePolynomial
    (assignment : Fin sourceColumnCount → F) :
    evaluate assignment qCertificatePolynomial =
      certificateValue (qCertificateResidual assignment) := by
  simp only [qCertificatePolynomial, qCertificateResidual, evaluate_sub,
    evaluate_sourceRowPolynomial, evaluate_sourceRowPolynomialSum,
    certificateValue_sub]

theorem q_identity_exact
    (assignment : Fin sourceColumnCount → F) :
    identityResidual 0 1 assignment = qCertificateResidual assignment := by
  have evaluated := congrArg (evaluate assignment) q_polynomial_exact
  rw [evaluate_identityResidual, evaluate_qCertificatePolynomial] at evaluated
  exact (ZMod.finEquiv goldilocksModulus).injective evaluated

private def pDefinitionRows : Polynomial sourceColumnCount :=
  sourceRowPolynomialSum [0, 5, 10, 15, 20, 25]

private def weightedRows : Polynomial sourceColumnCount :=
  sourceRowPolynomialSum [3, 8, 13, 18, 23, 28]

/-- Second product-sum identity after the six private weighted values and the
first final output are eliminated. -/
def pCertificatePolynomial : Polynomial sourceColumnCount :=
  sub
    (sub (add (sourceRowPolynomial 31) weightedRows) pDefinitionRows)
    (scale 7 (sourceRowPolynomial 30))

theorem p_polynomial_exact :
    identityPolynomial (decodedStep 2) (decodedStep 3) =
      pCertificatePolynomial := by
  native_decide

def pCertificateResidual
    (assignment : Fin sourceColumnCount → F) : F :=
  (residualAt 31 assignment +
      residualSum assignment [3, 8, 13, 18, 23, 28]) -
    residualSum assignment [0, 5, 10, 15, 20, 25] -
      7 * residualAt 30 assignment

theorem evaluate_pCertificatePolynomial
    (assignment : Fin sourceColumnCount → F) :
    evaluate assignment pCertificatePolynomial =
      certificateValue (pCertificateResidual assignment) := by
  simp only [pCertificatePolynomial, pCertificateResidual, pDefinitionRows,
    weightedRows, evaluate_sub, evaluate_add, evaluate_scale,
    evaluate_sourceRowPolynomial, evaluate_sourceRowPolynomialSum,
    certificateValue_sub, certificateValue_add, certificateValue_mul]

theorem p_identity_exact
    (assignment : Fin sourceColumnCount → F) :
    identityResidual 2 3 assignment = pCertificateResidual assignment := by
  have evaluated := congrArg (evaluate assignment) p_polynomial_exact
  rw [evaluate_identityResidual, evaluate_pCertificatePolynomial] at evaluated
  exact (ZMod.finEquiv goldilocksModulus).injective evaluated

private def crossRows : Polynomial sourceColumnCount :=
  sourceRowPolynomialSum [4, 9, 14, 19, 24, 29]

private def crossDefinitionRows : Polynomial sourceColumnCount :=
  sourceRowPolynomialSum [2, 7, 12, 17, 22, 27]

/-- Third product-sum identity after all private product, weighted, and prior
output values are eliminated. -/
def rCertificatePolynomial : Polynomial sourceColumnCount :=
  sub
    (add
      (add
        (sub (add (sourceRowPolynomial 32) crossRows) crossDefinitionRows)
        (sourceRowPolynomial 31))
      weightedRows)
    (scale 6 (sourceRowPolynomial 30))

theorem r_polynomial_exact :
    identityPolynomial (decodedStep 4) (decodedStep 5) =
      rCertificatePolynomial := by
  native_decide

def rCertificateResidual
    (assignment : Fin sourceColumnCount → F) : F :=
  (((residualAt 32 assignment +
        residualSum assignment [4, 9, 14, 19, 24, 29]) -
      residualSum assignment [2, 7, 12, 17, 22, 27]) +
    residualAt 31 assignment) +
      residualSum assignment [3, 8, 13, 18, 23, 28] -
        6 * residualAt 30 assignment

theorem evaluate_rCertificatePolynomial
    (assignment : Fin sourceColumnCount → F) :
    evaluate assignment rCertificatePolynomial =
      certificateValue (rCertificateResidual assignment) := by
  simp only [rCertificatePolynomial, rCertificateResidual, crossRows,
    crossDefinitionRows, weightedRows, evaluate_sub, evaluate_add,
    evaluate_scale, evaluate_sourceRowPolynomial,
    evaluate_sourceRowPolynomialSum, certificateValue_sub,
    certificateValue_add, certificateValue_mul]

theorem r_identity_exact
    (assignment : Fin sourceColumnCount → F) :
    identityResidual 4 5 assignment = rCertificateResidual assignment := by
  have evaluated := congrArg (evaluate assignment) r_polynomial_exact
  rw [evaluate_identityResidual, evaluate_rCertificatePolynomial] at evaluated
  exact (ZMod.finEquiv goldilocksModulus).injective evaluated

theorem residualAt_zero_of_rowsHold
    (assignment : Fin sourceColumnCount → F)
    (holds : RowsHold decodedSourceRows assignment 1)
    (index : Fin 33) : residualAt index assignment = 0 := by
  let sourceIndex : Fin decodedSourceRows.length :=
    ⟨index.val, by rw [decodedSourceRows_length]; exact index.isLt⟩
  have rowHolds := rowHolds_of_rowsHold holds sourceIndex
  have residualZero :=
    (rowHolds_iff_residual_zero
      (decodedSourceRows.get sourceIndex) assignment 1).mp rowHolds
  simpa only [residualAt, sourceRow] using residualZero

theorem residualSum_zero_of_rowsHold
    (assignment : Fin sourceColumnCount → F)
    (holds : RowsHold decodedSourceRows assignment 1)
    (indices : List (Fin 33)) : residualSum assignment indices = 0 := by
  induction indices with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [residualSum, residualAt_zero_of_rowsHold assignment holds head,
        inductionHypothesis, Fin.zero_add]

theorem qCertificateResidual_zero_of_rowsHold
    (assignment : Fin sourceColumnCount → F)
    (holds : RowsHold decodedSourceRows assignment 1) :
    qCertificateResidual assignment = 0 := by
  simp only [qCertificateResidual,
    residualAt_zero_of_rowsHold assignment holds 30,
    residualSum_zero_of_rowsHold assignment holds, Fin.sub_self]

theorem pCertificateResidual_zero_of_rowsHold
    (assignment : Fin sourceColumnCount → F)
    (holds : RowsHold decodedSourceRows assignment 1) :
    pCertificateResidual assignment = 0 := by
  simp only [pCertificateResidual,
    residualAt_zero_of_rowsHold assignment holds,
    residualSum_zero_of_rowsHold assignment holds, Fin.zero_add,
    Fin.mul_zero, Fin.sub_self]

theorem rCertificateResidual_zero_of_rowsHold
    (assignment : Fin sourceColumnCount → F)
    (holds : RowsHold decodedSourceRows assignment 1) :
    rCertificateResidual assignment = 0 := by
  simp only [rCertificateResidual,
    residualAt_zero_of_rowsHold assignment holds,
    residualSum_zero_of_rowsHold assignment holds, Fin.zero_add,
    Fin.mul_zero, Fin.sub_self]

/-- Every assignment that satisfies the exact 33 decoded source rows also
satisfies all three carry-eliminated grouped-product identities. -/
theorem sourceRows_imply_identityResiduals_zero
    (assignment : Fin sourceColumnCount → F)
    (holds : RowsHold decodedSourceRows assignment 1) :
    identityResidual 0 1 assignment = 0 ∧
      identityResidual 2 3 assignment = 0 ∧
        identityResidual 4 5 assignment = 0 := by
  constructor
  · rw [q_identity_exact]
    exact qCertificateResidual_zero_of_rowsHold assignment holds
  · constructor
    · rw [p_identity_exact]
      exact pCertificateResidual_zero_of_rowsHold assignment holds
    · rw [r_identity_exact]
      exact rCertificateResidual_zero_of_rowsHold assignment holds

def derivedOutputIndex {columns : Nat} :
    DecodedOutput columns → Option Nat
  | .source _ => none
  | .derivedProductSum index => some index

def outputIsSource {columns : Nat} : DecodedOutput columns → Bool
  | .source _ => true
  | .derivedProductSum _ => false

/-- Exact carry topology of the six generated rewrite steps. -/
theorem fixture_step_flow :
    (decodedStep 0).previous = none ∧
    derivedOutputIndex (decodedStep 0).output = some 0 ∧
    (decodedStep 1).previous = some 0 ∧
    outputIsSource (decodedStep 1).output = true ∧
    (decodedStep 2).previous = none ∧
    derivedOutputIndex (decodedStep 2).output = some 1 ∧
    (decodedStep 3).previous = some 1 ∧
    outputIsSource (decodedStep 3).output = true ∧
    (decodedStep 4).previous = none ∧
    derivedOutputIndex (decodedStep 4).output = some 2 ∧
    (decodedStep 5).previous = some 2 ∧
    outputIsSource (decodedStep 5).output = true := by
  native_decide

def fixtureDerivedValues
    (assignment : Fin sourceColumnCount → F) : Nat → F
  | 0 => contribution 0 assignment
  | 1 => contribution 2 assignment
  | 2 => contribution 4 assignment
  | _ => 0

private theorem pairStepsHold
    (first second : Fin 6) (compilerIndex : Nat)
    (assignment : Fin sourceColumnCount → F) (derived : Nat → F)
    (firstPrevious : (decodedStep first).previous = none)
    (firstOutput :
      derivedOutputIndex (decodedStep first).output = some compilerIndex)
    (secondPrevious :
      (decodedStep second).previous = some compilerIndex)
    (secondSource : outputIsSource (decodedStep second).output = true)
    (derivedAt : derived compilerIndex = contribution first assignment)
    (identityZero : identityResidual first second assignment = 0) :
    StepHolds (decodedStep first) assignment 1 derived ∧
      StepHolds (decodedStep second) assignment 1 derived := by
  have outputEquation :
      outputValue assignment 1 (fun _ => 0) (decodedStep second).output =
        contribution first assignment + contribution second assignment := by
    apply (Lean.Grind.AddCommGroup.sub_eq_zero_iff).mp
    simpa only [identityResidual] using identityZero
  constructor
  · change
      outputValue assignment 1 derived (decodedStep first).output =
        previousValue derived (decodedStep first).previous +
          contribution first assignment
    rw [firstPrevious]
    simp only [previousValue]
    cases outputEq : (decodedStep first).output with
    | source value =>
        rw [outputEq] at firstOutput
        simp [derivedOutputIndex] at firstOutput
    | derivedProductSum outputIndex =>
        rw [outputEq] at firstOutput
        simp only [derivedOutputIndex, Option.some.injEq] at firstOutput
        subst outputIndex
        simp only [outputValue, derivedAt, Fin.zero_add]
  · change
      outputValue assignment 1 derived (decodedStep second).output =
        previousValue derived (decodedStep second).previous +
          contribution second assignment
    rw [secondPrevious]
    simp only [previousValue, derivedAt]
    cases outputEq : (decodedStep second).output with
    | source value =>
        simpa only [outputEq, outputValue] using outputEquation
    | derivedProductSum outputIndex =>
        rw [outputEq] at secondSource
        simp [outputIsSource] at secondSource

/-- The same source assignment extends with three explicit carry values and
satisfies every generated recurrence step. This is the forward refinement
needed before the 33 source rows can be replaced. -/
theorem sourceRows_imply_all_steps_hold
    (assignment : Fin sourceColumnCount → F)
    (holds : RowsHold decodedSourceRows assignment 1) :
    ∀ index : Fin 6,
      StepHolds (decodedStep index) assignment 1
        (fixtureDerivedValues assignment) := by
  rcases sourceRows_imply_identityResiduals_zero assignment holds with
    ⟨identity01, identity23, identity45⟩
  rcases fixture_step_flow with
    ⟨previous0, output0, previous1, source1,
      previous2, output2, previous3, source3,
      previous4, output4, previous5, source5⟩
  have pair01 := pairStepsHold 0 1 0 assignment
    (fixtureDerivedValues assignment) previous0 output0 previous1 source1
    rfl identity01
  have pair23 := pairStepsHold 2 3 1 assignment
    (fixtureDerivedValues assignment) previous2 output2 previous3 source3
    rfl identity23
  have pair45 := pairStepsHold 4 5 2 assignment
    (fixtureDerivedValues assignment) previous4 output4 previous5 source5
    rfl identity45
  exact Fin.cases pair01.1
    (Fin.cases pair01.2
      (Fin.cases pair23.1
        (Fin.cases pair23.2
          (Fin.cases pair45.1
            (Fin.cases pair45.2 (fun index => Fin.elim0 index))))))

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement
