import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support
import NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupportData

/-!
Owns exact source support for every caller-owned production PiCCS expression.

Each typed interface family maps to one named source range. This module does
not propagate support through child-local witnesses or R1CS lowering.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem externalInputsSupported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Formal.ExternalInputsSupported
      (PiCCSInputs.interface logicalWidth publicFits)
      PiCCSInputs.phaseOffset External := by
  refine {
    priorStateFixed := ?_
    outputStateFixed := ?_
    priorStateContext := ?_
    outputStateContext := ?_
    expectedContext := ?_
    runningPoint := ?_
    runningCommitment := ?_
    runningPublicInput := ?_
    runningEval_K := ?_
    runningEval_A := ?_
    freshCommitment := ?_
    freshPublicInput := ?_
    roundCoefficient := ?_
    outputEval_K := ?_
    outputEval_A := ?_ }
  · intro word member
    simp only [PiCCSInputs.interface, PiCCSInputs.priorStateWord,
      Expr.VarsSatisfy]
    apply external_prior
    have bound := StateBinding.fixedWord_index_lt word member
    unfold InRange
    rw [PilotProduction.stateHashWords_eq]
    omega
  · intro word member
    simp only [PiCCSInputs.interface, PiCCSInputs.outputStateWord,
      Expr.VarsSatisfy]
    apply external_output
    have bound := StateBinding.fixedWord_index_lt word member
    unfold InRange
    rw [PilotProduction.stateHashWords_eq]
    omega
  · intro lane
    simp only [PiCCSInputs.interface, PiCCSInputs.priorStateWord,
      Expr.VarsSatisfy]
    apply external_prior
    have bound := lane.isLt
    unfold InRange
    norm_num [StateBinding.contextWordStart] at bound
    norm_num [StateBinding.contextWordStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq]
    omega
  · intro lane
    simp only [PiCCSInputs.interface, PiCCSInputs.outputStateWord,
      Expr.VarsSatisfy]
    apply external_output
    have bound := lane.isLt
    unfold InRange
    norm_num [StateBinding.contextWordStart] at bound
    norm_num [StateBinding.contextWordStart,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq]
    omega
  · intro lane
    simp only [PiCCSInputs.interface, PiCCSInputs.expectedContext,
      Expr.VarsSatisfy]
    apply external_context
    have bound := lane.isLt
    unfold InRange
    rw [PiCCSInputs.expectedContextWords_eq]
    omega
  · intro coordinate
    change External (PiCCSInputs.runningPointStart + coordinate.val * 2) ∧
      External (PiCCSInputs.runningPointStart + coordinate.val * 2 + 1)
    have coordinateBound := coordinate.isLt
    norm_num [productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at coordinateBound
    constructor <;> apply external_prior <;> unfold InRange <;>
      norm_num [PiCCSInputs.runningPointStart, PiCCSInputs.priorRunningStart,
        PilotProduction.priorPreimageStart,
        PilotProduction.stateHashWords_eq] <;> omega
  · intro source row coefficient
    simp only [PiCCSInputs.interface, PiCCSInputs.runningExpr,
      PiCCSInputs.runningCommitment, Expr.VarsSatisfy]
    apply external_prior
    have sourceBound := source.isLt
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound
    norm_num [productionProfile] at rowBound
    norm_num [productionShape, Phi81MatrixSource.phi81Shape,
      ringDegree] at coefficientBound
    unfold InRange
    norm_num [PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      ringDegree, PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq]
    omega
  · intro source column
    simp only [PiCCSInputs.interface, PiCCSInputs.runningExpr,
      PiCCSInputs.runningPublicInput, Expr.VarsSatisfy]
    apply external_prior
    have sourceBound := source.isLt
    have columnBound := column.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound
    norm_num [FullShape, fullShape, Phi81Relation.Shape.publicWidth,
      publicRingColumns, ringDegree] at columnBound
    unfold InRange
    norm_num [PiCCSInputs.runningPublicStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq]
    omega
  · intro source coefficient
    change External (PiCCSInputs.runningEvaluationStart source.val +
        coefficient.val * 2) ∧
      External (PiCCSInputs.runningEvaluationStart source.val +
        coefficient.val * 2 + 1)
    have sourceBound := source.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound
    norm_num [productionShape, Phi81MatrixSource.phi81Shape,
      ringDegree] at coefficientBound
    constructor <;> apply external_prior <;> unfold InRange <;>
      norm_num [PiCCSInputs.runningEvaluationStart,
        PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
        PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
        PilotProduction.priorPreimageStart,
        PilotProduction.stateHashWords_eq] <;> omega
  · intro source matrix coefficient
    change External (PiCCSInputs.runningEvaluationStart source.val + 108 +
          matrix.val * 108 + coefficient.val * 2) ∧
      External (PiCCSInputs.runningEvaluationStart source.val + 108 +
          matrix.val * 108 + coefficient.val * 2 + 1)
    have sourceBound := source.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at matrixBound
    norm_num [productionShape, Phi81MatrixSource.phi81Shape,
      ringDegree] at coefficientBound
    constructor <;> apply external_prior <;> unfold InRange <;>
      norm_num [PiCCSInputs.runningEvaluationStart,
        PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
        PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
        PilotProduction.priorPreimageStart,
        PilotProduction.stateHashWords_eq] <;> omega
  · intro source row coefficient
    simp only [PiCCSInputs.interface, PiCCSInputs.freshExpr,
      PiCCSInputs.freshCommitment, Expr.VarsSatisfy]
    apply external_proof
    have sourceBound := source.isLt
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    change source.val < 1 at sourceBound
    norm_num [productionProfile] at rowBound
    norm_num [ringDegree] at coefficientBound
    unfold InRange
    rw [PiCCSInputs.phaseOffset_eq, PiCCSInputs.proofInputStart_eq]
    norm_num [PiCCSInputs.freshCommitmentStart,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.freshCommitmentWords,
      ringDegree]
    omega
  · intro source column
    simp only [PiCCSInputs.interface, PiCCSInputs.freshExpr,
      PiCCSInputs.freshPublicInput, Expr.VarsSatisfy]
    apply external_public
    have columnBound := column.isLt
    norm_num [FullShape, fullShape, Phi81Relation.Shape.publicWidth,
      publicRingColumns, ringDegree] at columnBound
    unfold InRange
    omega
  · intro roundIndex coefficient
    change External (PiCCSInputs.roundMessageStart + roundIndex.val * 20 +
          coefficient.val * 2) ∧
      External (PiCCSInputs.roundMessageStart + roundIndex.val * 20 +
          coefficient.val * 2 + 1)
    have roundBound := roundIndex.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at roundBound
    norm_num at coefficientBound
    constructor <;> apply external_proof <;> unfold InRange <;>
      rw [PiCCSInputs.phaseOffset_eq, PiCCSInputs.proofInputStart_eq] <;>
      norm_num [PiCCSInputs.roundMessageStart,
        PiCCSInputs.freshCommitmentStart, PiCCSInputs.proofInputStart,
        PiCCSInputs.expectedContextStart, PiCCSInputs.expectedContextWords,
        PiCCSInputs.freshCommitmentWords] <;>
      omega
  · intro source coefficient
    change External (PiCCSInputs.outputEvaluationStart +
          source.val * 1620 + coefficient.val * 2) ∧
      External (PiCCSInputs.outputEvaluationStart +
          source.val * 1620 + coefficient.val * 2 + 1)
    have sourceBound := source.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape, Shape.sourceCount] at sourceBound
    norm_num [productionShape, Phi81MatrixSource.phi81Shape,
      ringDegree] at coefficientBound
    constructor <;> apply external_proof <;> unfold InRange <;>
      rw [PiCCSInputs.phaseOffset_eq, PiCCSInputs.proofInputStart_eq] <;>
      norm_num [PiCCSInputs.outputEvaluationStart,
        PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentStart,
        PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
        PiCCSInputs.expectedContextWords, PiCCSInputs.freshCommitmentWords,
        PiCCSInputs.roundMessageWords] <;>
      omega
  · intro source matrix coefficient
    change External (PiCCSInputs.outputEvaluationStart +
          source.val * 1620 + 108 + matrix.val * 108 + coefficient.val * 2) ∧
      External (PiCCSInputs.outputEvaluationStart +
          source.val * 1620 + 108 + matrix.val * 108 + coefficient.val * 2 + 1)
    have sourceBound := source.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape, Shape.sourceCount] at sourceBound
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at matrixBound
    norm_num [productionShape, Phi81MatrixSource.phi81Shape,
      ringDegree] at coefficientBound
    constructor <;> apply external_proof <;> unfold InRange <;>
      rw [PiCCSInputs.phaseOffset_eq, PiCCSInputs.proofInputStart_eq] <;>
      norm_num [PiCCSInputs.outputEvaluationStart,
        PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentStart,
        PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
        PiCCSInputs.expectedContextWords, PiCCSInputs.freshCommitmentWords,
        PiCCSInputs.roundMessageWords] <;>
      omega

end NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport
