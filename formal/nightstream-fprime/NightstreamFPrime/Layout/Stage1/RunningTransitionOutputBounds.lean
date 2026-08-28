import NightstreamFPrime.Layout.Stage1.RunningTransitionData

/-! Owns bounds for the pilot output-preimage running fields. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem outputRunningBelowOutputDigestStart
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    RunningTransition.RunningBelow (outputRunningExpr logicalWidth publicFits)
      PilotProduction.outputDigestStart := by
  refine {
    point := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro coordinate
    simp only [outputRunningExpr, outputPoint, outputPairAt, KExpr.VarsBelow,
      Expr.VarsBelow]
    have coordinateBound := coordinate.isLt
    norm_num [PilotProduction.outputDigestStart, outputBase,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth_eq, PiCCSInputs.runningPointStart,
      PiCCSInputs.priorRunningStart, productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at coordinateBound ⊢
    omega
  · intro source row coefficient
    simp only [outputRunningExpr, outputCommitment, Expr.VarsBelow]
    have sourceBound := source.isLt
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    norm_num [PilotProduction.outputDigestStart, outputBase,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth_eq, PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
      ringDegree] at sourceBound rowBound coefficientBound ⊢
    omega
  · intro source column
    simp only [outputRunningExpr, outputPublicInput, Expr.VarsBelow]
    have sourceBound := source.isLt
    have columnBound := column.isLt
    norm_num [PilotProduction.outputDigestStart, outputBase,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth_eq, PiCCSInputs.runningPublicStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
      FullShape, fullShape, Phi81Relation.Shape.publicWidth,
      publicRingColumns, ringDegree] at sourceBound columnBound ⊢
    omega
  · intro source coefficient
    simp only [outputRunningExpr, outputEval_K, outputPairAt, KExpr.VarsBelow,
      Expr.VarsBelow]
    have sourceBound := source.isLt
    have coefficientBound := coefficient.isLt
    norm_num [PilotProduction.outputDigestStart, outputBase,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth_eq, PiCCSInputs.runningEvaluationStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
      ringDegree] at sourceBound coefficientBound ⊢
    omega
  · intro source matrix coefficient
    simp only [outputRunningExpr, outputEval_A, outputPairAt, KExpr.VarsBelow,
      Expr.VarsBelow]
    have sourceBound := source.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    norm_num [PilotProduction.outputDigestStart, outputBase,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth_eq, PiCCSInputs.runningEvaluationStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
      ringDegree] at sourceBound matrixBound coefficientBound ⊢
    omega

theorem outputRunningBelow
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    RunningTransition.RunningBelow (outputRunningExpr logicalWidth publicFits)
      phaseOffset := by
  apply (outputRunningBelowOutputDigestStart logicalWidth publicFits).mono
  norm_num [PilotProduction.outputDigestStart,
    PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
    PriorStateHash.publicWidth_eq, phaseOffset]

end NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
