import NightstreamFPrime.Layout.Stage1.PiDECInputs
import NightstreamFPrime.Layout.Stage1.RunningTransitionPointBoundsDirect

/-!
Owns the causal source bound for the zero-copy PiRLC-to-PiDEC bridge.

Every reused PiRLC output and every prover-supplied PiDEC message word lies
strictly before the PiDEC logical allocation. This module adds no row or
column and does not prove PiDEC acceptance.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

private theorem combinationOutput_varsBelow
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : PiRLC.v1_1.CombinationFamily.Interface blockCount cellCount)
    (offset : Nat) (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount) :
    (PiRLC.v1_1.CombinationFamily.output interface offset block lane cell
      ).VarsBelow
      (offset +
        PiRLC.v1_1.CombinationFamily.logicalPrivateCount blockCount
          cellCount) := by
  simp only [PiRLC.v1_1.CombinationFamily.output,
    PiRLC.v1_1.CombinationStep.output, Expr.VarsBelow]
  have indexBound :
      (PiRLC.v1_1.CombinationStep.indexOf block lane cell).val <
        PiRLC.v1_1.CombinationFamily.stepSize blockCount cellCount := by
    exact (PiRLC.v1_1.CombinationStep.indexOf block lane cell).isLt
  have finalSourceValue :
      PiRLC.v1_1.CombinationFamily.finalSource.val = 16 := by
    rfl
  unfold PiRLC.v1_1.CombinationFamily.stepOffset
    PiRLC.v1_1.CombinationFamily.logicalPrivateCount
  rw [finalSourceValue]
  simp only [PiRLC.v1_1.CombinationFamily.sourceCount_eq]
  omega

private theorem parentPoint_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (_relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (coordinate : Fin productionShape.cubeVariables) :
    ((interface logicalWidth publicFits).point phaseOffset coordinate
      ).VarsBelow phaseOffset := by
  have pointEq :
      (interface logicalWidth publicFits).point phaseOffset coordinate =
        RunningTransitionInputs.directRoundPoint
          PiCCSStarts.roundTranscriptWitnessStart coordinate := by
    simpa [RunningTransitionInputs.recursiveRunningExpr,
      RunningTransitionInputs.piDecInterface] using
        (RunningTransitionInputs.recursivePoint_eq_direct
          (logicalWidth := logicalWidth) (publicFits := publicFits) coordinate)
  rw [pointEq, PiCCSStarts.roundTranscriptWitnessStart_eq]
  simp only [RunningTransitionInputs.directRoundPoint,
    Quadratic.KExpr.VarsBelow, Expr.VarsBelow]
  have coordinateBound := coordinate.isLt
  change coordinate.val < 28 at coordinateBound
  norm_num [phaseOffset, proofInputStart, proofInputColumnCount, childCount,
    commitmentWordsPerChild, evalKWordsPerChild, evalAWordsPerChild,
    publicInputWordsPerChild, RunningTransitionInputs.roundStride,
    RunningTransitionInputs.roundSampleC0Offset,
    RunningTransitionInputs.roundSampleC1Offset]
  omega

/-- Every caller-owned PiDEC expression precedes the phase allocation. -/
theorem inputsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiDEC.v1_1.Formal.InputsBelow (interface logicalWidth publicFits)
      phaseOffset := by
  refine {
    point := parentPoint_varsBelow relation
    parentCommitment := ?_
    parentPublicInput := ?_
    parentEval_K := ?_
    parentEval_A := ?_
    messageCommitment := ?_
    messageEval_K := ?_
    messageEval_A := ?_
    digit := ?_ }
  · intro row lane
    apply Expr.VarsBelow.mono _
      (combinationOutput_varsBelow
        (PiRLC.v1_1.CommitmentCombination.familyInterface
          (PiRLC.v1_1.Formal.commitmentInterface
            (piRlcSharedInterface logicalWidth publicFits)))
        PiRLCStarts.commitmentLogicalStart row lane
        PiRLC.v1_1.CommitmentCombination.cell)
    rw [PiRLC.v1_1.CommitmentCombination.logicalPrivateCount_eq]
    change 19266319 + 16524 ≤ 27402496
    norm_num
  · intro column
    apply Expr.VarsBelow.mono _
      (combinationOutput_varsBelow
        (PiRLC.v1_1.PublicInputCombination.familyInterface
          (PiRLC.v1_1.Formal.publicInputInterface
            (piRlcSharedInterface logicalWidth publicFits)))
        PiRLCStarts.publicInputLogicalStart
        (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
          (FullShape logicalWidth publicFits) column)
        (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex
          column)
        PiRLC.v1_1.PublicInputCombination.cell)
    rw [PiRLC.v1_1.PublicInputCombination.logicalPrivateCount_eq]
    change 19282843 + 4590 ≤ 27402496
    norm_num
  · intro coefficient
    constructor
    · apply Expr.VarsBelow.mono _
        (combinationOutput_varsBelow
          (PiRLC.v1_1.RingKCombination.familyInterface
            (PiRLC.v1_1.EvalKCombination.ringInterface
              (PiRLC.v1_1.Formal.evalKInterface
                (piRlcSharedInterface logicalWidth publicFits))))
          PiRLCStarts.evalKLogicalStart PiRLC.v1_1.EvalKCombination.block
          (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq
            coefficient)
          PiRLC.v1_1.RingKCombination.c0Cell)
      rw [PiRLC.v1_1.EvalKCombination.logicalPrivateCount_eq]
      change 19287433 + 1836 ≤ 27402496
      norm_num
    · apply Expr.VarsBelow.mono _
        (combinationOutput_varsBelow
          (PiRLC.v1_1.RingKCombination.familyInterface
            (PiRLC.v1_1.EvalKCombination.ringInterface
              (PiRLC.v1_1.Formal.evalKInterface
                (piRlcSharedInterface logicalWidth publicFits))))
          PiRLCStarts.evalKLogicalStart PiRLC.v1_1.EvalKCombination.block
          (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq
            coefficient)
          PiRLC.v1_1.RingKCombination.c1Cell)
      rw [PiRLC.v1_1.EvalKCombination.logicalPrivateCount_eq]
      change 19287433 + 1836 ≤ 27402496
      norm_num
  · intro matrix coefficient
    constructor
    · apply Expr.VarsBelow.mono _
        (combinationOutput_varsBelow
          (PiRLC.v1_1.RingKCombination.familyInterface
            (PiRLC.v1_1.EvalACombination.ringInterface
              (PiRLC.v1_1.Formal.evalAInterface
                (piRlcSharedInterface logicalWidth publicFits))))
          PiRLCStarts.evalALogicalStart matrix
          (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq
            coefficient)
          PiRLC.v1_1.RingKCombination.c0Cell)
      rw [PiRLC.v1_1.EvalACombination.logicalPrivateCount_eq]
      change 19289269 + 25704 ≤ 27402496
      norm_num
    · apply Expr.VarsBelow.mono _
        (combinationOutput_varsBelow
          (PiRLC.v1_1.RingKCombination.familyInterface
            (PiRLC.v1_1.EvalACombination.ringInterface
              (PiRLC.v1_1.Formal.evalAInterface
                (piRlcSharedInterface logicalWidth publicFits))))
          PiRLCStarts.evalALogicalStart matrix
          (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq
            coefficient)
          PiRLC.v1_1.RingKCombination.c1Cell)
      rw [PiRLC.v1_1.EvalACombination.logicalPrivateCount_eq]
      change 19289269 + 25704 ≤ 27402496
      norm_num
  · intro child row lane
    simp only [interface, message, childCommitment, Expr.VarsBelow]
    have childBound := child.isLt
    have rowBound := row.isLt
    have laneBound := lane.isLt
    norm_num [productionGlobalParams, productionProfile] at childBound rowBound
    norm_num [ringDegree] at laneBound
    norm_num [childCommitmentStart, commitmentInputStart, phaseOffset,
      proofInputStart, proofInputColumnCount, childCount,
      commitmentWordsPerChild, evalKWordsPerChild, evalAWordsPerChild,
      publicInputWordsPerChild, ringDegree]
    omega
  · intro child coefficient
    simp only [interface, message, childEvalK, Quadratic.KExpr.VarsBelow,
      Expr.VarsBelow]
    have childBound := child.isLt
    have coefficientBound := coefficient.isLt
    change child.val < 16 at childBound
    change coefficient.val < 54 at coefficientBound
    norm_num [childEvalKStart, evalKInputStart, commitmentInputStart,
      phaseOffset, proofInputStart, proofInputColumnCount, childCount,
      commitmentWordsPerChild, evalKWordsPerChild, evalAWordsPerChild,
      publicInputWordsPerChild]
    omega
  · intro child matrix coefficient
    simp only [interface, message, childEvalA, Quadratic.KExpr.VarsBelow,
      Expr.VarsBelow]
    have childBound := child.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    change child.val < 16 at childBound
    change matrix.val < 14 at matrixBound
    change coefficient.val < 54 at coefficientBound
    norm_num [childEvalAStart, evalAInputStart, evalKInputStart,
      commitmentInputStart, phaseOffset, proofInputStart,
      proofInputColumnCount, childCount, commitmentWordsPerChild,
      evalKWordsPerChild, evalAWordsPerChild, publicInputWordsPerChild]
    omega
  · intro child coordinate
    simp only [interface, childPublicInput, Expr.VarsBelow]
    have childBound := child.isLt
    have coordinateBound := coordinate.isLt
    norm_num [productionGlobalParams] at childBound
    norm_num [PiDEC.v1_1.PublicInputSplit.coordinateCount_eq] at coordinateBound
    norm_num [childPublicInputStart, publicInputStart, evalAInputStart,
      evalKInputStart, commitmentInputStart, phaseOffset, proofInputStart,
      proofInputColumnCount, childCount, commitmentWordsPerChild,
      evalKWordsPerChild, evalAWordsPerChild, publicInputWordsPerChild]
    omega

/-- The production PiDEC assumptions are fully derived from the canonical
zero-copy input layout; no caller supplies a causal-scope hypothesis. -/
theorem assumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    PiDEC.v1_1.Formal.Assumptions relation
      (interface logicalWidth publicFits) phaseOffset env := by
  exact ⟨inputsBelow relation⟩

end NightstreamFPrime.Layout.Stage1.PiDECInputs
