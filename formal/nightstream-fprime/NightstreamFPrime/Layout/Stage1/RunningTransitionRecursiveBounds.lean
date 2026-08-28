import NightstreamFPrime.Layout.Stage1.RunningTransitionData

/-! Owns bounds for the PiDEC child fields used by the running transition. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def recursiveRunningBelow
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (pointBelow : ∀ coordinate,
      ((recursiveRunningExpr logicalWidth publicFits).point coordinate
        ).VarsBelow phaseOffset) :
    RunningTransition.RunningBelow (recursiveRunningExpr logicalWidth publicFits)
      phaseOffset := by
  refine {
    point := pointBelow
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro source row coefficient
    simp only [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
      PiDECInputs.message, PiDECInputs.childCommitment, Expr.VarsBelow]
    have sourceBound := source.isLt
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    norm_num [phaseOffset, PiDECInputs.childCommitmentStart,
      PiDECInputs.commitmentInputStart, PiDECInputs.proofInputStart,
      PiDECInputs.commitmentWordsPerChild, productionShape,
      productionProfile, Phi81MatrixSource.phi81Shape, ringDegree] at sourceBound rowBound coefficientBound ⊢
    omega
  · intro source column
    simp only [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
      PiDECInputs.childPublicInput, Expr.VarsBelow]
    have sourceBound := source.isLt
    have columnBound := column.isLt
    norm_num [phaseOffset, PiDECInputs.childPublicInputStart,
      PiDECInputs.publicInputStart, PiDECInputs.evalAInputStart,
      PiDECInputs.evalKInputStart, PiDECInputs.commitmentInputStart,
      PiDECInputs.proofInputStart, PiDECInputs.childCount,
      PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
      PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
      productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
      FullShape, fullShape, Phi81Relation.Shape.publicWidth,
      publicRingColumns, ringDegree] at sourceBound columnBound ⊢
    omega
  · intro source coefficient
    simp only [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
      PiDECInputs.message, PiDECInputs.childEvalK, KExpr.VarsBelow,
      Expr.VarsBelow]
    have sourceBound := source.isLt
    have coefficientBound := coefficient.isLt
    norm_num [phaseOffset, PiDECInputs.childEvalKStart,
      PiDECInputs.evalKInputStart, PiDECInputs.commitmentInputStart,
      PiDECInputs.proofInputStart, PiDECInputs.childCount,
      PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
      productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
      ringDegree] at sourceBound coefficientBound ⊢
    omega
  · intro source matrix coefficient
    simp only [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
      PiDECInputs.message, PiDECInputs.childEvalA, KExpr.VarsBelow,
      Expr.VarsBelow]
    have sourceBound := source.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    norm_num [phaseOffset, PiDECInputs.childEvalAStart,
      PiDECInputs.evalAInputStart, PiDECInputs.evalKInputStart,
      PiDECInputs.commitmentInputStart, PiDECInputs.proofInputStart,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
      ringDegree] at sourceBound matrixBound coefficientBound ⊢
    omega

end NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
