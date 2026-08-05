import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityNifs

/-!
Focused regressions for the model-level `PaddedRowIdentity` validation.

The negative test changes the application polynomial to one with a nonzero
constant term. The first padding row then rejects. This confirms that the
proved zero-at-origin condition is necessary and that padding is not accepted
by definition.
-/

set_option autoImplicit false

namespace tests.PaddedRowIdentity

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySoundness
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityComposition
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNifs

#check identityFirstConstraintSatisfied_iff_logical
#check identityConstantOutput_eq_paddedAssignmentMLE
#check connectedSemanticTruth_iff_logicalSemanticTruth
#check statement_sumcheckDegree_exact
#check acceptedProbe_extracts_logicalSource_or_badEvent
#check sevenProjectiveNonresidue
#check extensionNoZeroDivisors
#check fixedFirstBadBound_of_rootCounting
#check fullChallengeFixedFirstBadBound
#check selectedCompatibleContext
#check selectedFiniteReductionThroughPiDec
#check selectedKey
#check strongExecutionContext_eq_selectedContext
#check compatibleContext_eq_selectedCompatibleContext
#check selectedFullOracleSoundness

example : shape.cubeVariables = 24 := shape_cubeVariables_exact
example : shape.matrixCount = 14 := shape_matrixCount_exact
example : shape.sourceCount = 15 := shape_sourceCount_exact
example : shape.carriedEvaluationCount = 10584 :=
  shape_carriedEvaluationCount_exact
example : shape.jointCoefficientCount = 10600 :=
  shape_jointCoefficientCount_exact
example : shape.sourceCount * shape.matrixCount = 210 :=
  terminalRingValueCount_exact
example : mixingNumerator = 10599 := mixingNumerator_exact
example : sumCheckNumerator = 216 := sumCheckNumerator_exact
example : algebraicNumerator = 10815 := algebraicNumerator_exact
example : fullChallengeSupport.cardinality = goldilocksP * goldilocksP :=
  fullChallengeSupport_cardinality
example (cardinality : Nat) (bindingRoot : Rat) :
    selectedInteractiveLoss cardinality bindingRoot =
      ratio 16 cardinality +
        ((ratio 10599 (goldilocksP * goldilocksP) +
          ratio 216 (goldilocksP * goldilocksP)) + bindingRoot) :=
  selectedInteractiveLoss_explicit cardinality bindingRoot
example : algebraicNumerator * 2 ^ 114 <= goldilocksP * goldilocksP :=
  oneFoldAlgebraicBits_at_least_114
example : ¬ algebraicNumerator * 2 ^ 115 <= goldilocksP * goldilocksP :=
  oneFoldAlgebraicBits_not_115
example : algebraicNumerator * 64 * 2 ^ 108 <=
    goldilocksP * goldilocksP :=
  sixtyFourFoldAlgebraicBits_at_least_108
example : ¬ algebraicNumerator * 64 * 2 ^ 109 <=
    goldilocksP * goldilocksP :=
  sixtyFourFoldAlgebraicBits_not_109

/-- The first row after the logical prefix. -/
def firstPaddingVertex : BooleanVertex rowVariables :=
  NumericBooleanDomain.vertex rowVariables
    ⟨logicalRows, logicalRows_lt_cube⟩

theorem firstPaddingVertex_index :
    rowIndex firstPaddingVertex = logicalRows := by
  exact NumericBooleanDomain.index_vertex rowVariables
    ⟨logicalRows, logicalRows_lt_cube⟩

/-- A deliberate invalid mutation: one constant monomial. -/
def constantOneMonomial : Monomial F applicationMatrixCount where
  coefficient := 1
  exponents := fun _ => 0

def constantOnePolynomial : ConstraintPolynomial F applicationMatrixCount where
  degreeBound := 1
  terms := [constantOneMonomial]
  termsBelowDegree := by
    intro term member
    simp only [List.mem_singleton] at member
    subst term
    decide

def nonzeroConstantSystem (matrices : ApplicationMatrices) :
    Structure F applicationShape assignmentColumns where
  matrices := fun matrix => RowPadding.padRows (matrices.matrixAt matrix)
  constraintPolynomial := constantOnePolynomial

theorem nonzeroConstantResidual_at_firstPadding_eq_one
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns) :
    residualAt baseOps (nonzeroConstantSystem matrices) assignment
        firstPaddingVertex = 1 := by
  unfold residualAt
  have imagesZero :
      matrixImagesAt baseOps (nonzeroConstantSystem matrices) assignment
          firstPaddingVertex = fun _ => 0 := by
    funext matrix
    unfold matrixImagesAt nonzeroConstantSystem
    exact DirectRows.matrixVectorAt_padRows_padding
      (matrices.matrixAt matrix) assignment firstPaddingVertex (by
        rw [firstPaddingVertex_index])
  rw [imagesZero]
  change evaluatePolynomial baseOps constantOnePolynomial (fun _ => 0) = 1
  decide

/-- The invalid mutation rejects even when only padding is inspected. -/
theorem nonzeroConstantSystem_not_satisfied
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns) :
    ¬ ConstraintSatisfied baseOps (nonzeroConstantSystem matrices)
      assignment := by
  intro accepted
  have atPadding := accepted firstPaddingVertex
  rw [nonzeroConstantResidual_at_firstPadding_eq_one] at atPadding
  exact (by decide : (1 : F) ≠ 0) atPadding

end tests.PaddedRowIdentity
