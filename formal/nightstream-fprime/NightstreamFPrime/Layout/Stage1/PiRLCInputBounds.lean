import NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions
import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.PiRLCStarts

/-!
Owns the causal source bounds for the zero-copy PiCCS-to-PiRLC bridge.

The PiCCS output transcript state and all 17 source values precede the PiRLC
allocation. Sampler challenges precede every combination child. Therefore the
canonical layout, not a caller, supplies every PiRLC circuit assumption.
-/

namespace NightstreamFPrime.Layout.Stage1.PiRLCInputBounds

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem phase_le_commitment :
    PiRLCInputs.phaseOffset ≤
      PiRLC.v1_1.Formal.commitmentOffset PiRLCInputs.phaseOffset := by
  unfold PiRLC.v1_1.Formal.commitmentOffset
    PiRLC.v1_1.Formal.samplerOffset
  omega

private theorem piCcsPhase_le_piRlcPhase :
    PiCCSInputs.phaseOffset ≤ PiRLCInputs.phaseOffset := by
  norm_num [PiCCSInputs.phaseOffset_eq, PiRLCInputs.phaseOffset]

private theorem commitment_le_publicInput :
    PiRLC.v1_1.Formal.commitmentOffset PiRLCInputs.phaseOffset ≤
      PiRLC.v1_1.Formal.publicInputOffset PiRLCInputs.phaseOffset := by
  unfold PiRLC.v1_1.Formal.publicInputOffset
  omega

private theorem publicInput_le_evalK :
    PiRLC.v1_1.Formal.publicInputOffset PiRLCInputs.phaseOffset ≤
      PiRLC.v1_1.Formal.evalKOffset PiRLCInputs.phaseOffset := by
  unfold PiRLC.v1_1.Formal.evalKOffset
  omega

private theorem evalK_le_evalA :
    PiRLC.v1_1.Formal.evalKOffset PiRLCInputs.phaseOffset ≤
      PiRLC.v1_1.Formal.evalAOffset PiRLCInputs.phaseOffset := by
  unfold PiRLC.v1_1.Formal.evalAOffset
  omega

private theorem samplerInitialBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (lane : Fin Spec.Poseidon2.width) :
    (PiRLCInputs.piCcsOutputState
      (logicalWidth := logicalWidth) (publicFits := publicFits) lane).VarsBelow
        PiRLCInputs.phaseOffset := by
  let interface := PiCCSInputs.interface logicalWidth publicFits
  let outputAt := PiCCS.v1_1.Formal.outputBindingOffset relation interface
    PiCCSInputs.phaseOffset
  have outputAssumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.outputBinding relation
      interface PiCCSInputs.phaseOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits) env
  have bound := PiCCS.v1_1.OutputBinding.finalState_varsBelow
    (PiCCS.v1_1.Formal.outputBindingInterface
      (PiCCS.v1_1.Formal.atOffset interface PiCCSInputs.phaseOffset))
    outputAt env outputAssumptions lane
  rw [PiRLCInputs.piCcsOutputState_eq_parent relation]
  apply Expr.VarsBelow.mono _ bound
  unfold outputAt
  rw [← PiCCSStarts.outputBindingWitnessStart_matches relation]
  rw [PiCCS.v1_1.OutputBinding.localLength_eq]
  norm_num [PiCCSStarts.outputBindingWitnessStart_eq,
    PiRLCInputs.phaseOffset]

private theorem samplerChallengeBelow
    (source : Fin productionShape.sourceCount) (lane : Fin ringDegree) :
    (PiRLC.v1_1.SamplerChain.challengeExpr
      (PiRLC.v1_1.Formal.samplerInterface
        (PiRLC.v1_1.Formal.atOffset
          (PiRLCInputs.interface (logicalWidth := logicalWidth)
            (publicFits := publicFits)) PiRLCInputs.phaseOffset))
      PiRLCInputs.phaseOffset source lane).VarsBelow
        (PiRLC.v1_1.Formal.commitmentOffset PiRLCInputs.phaseOffset) := by
  apply Expr.VarsBelow.mono _
    (PiRLC.v1_1.SamplerChain.challengeExpr_varsBelow
      (PiRLC.v1_1.Formal.samplerInterface
        (PiRLC.v1_1.Formal.atOffset
          (PiRLCInputs.interface (logicalWidth := logicalWidth)
            (publicFits := publicFits)) PiRLCInputs.phaseOffset))
      PiRLCInputs.phaseOffset source lane)
  have sourceBound := source.isLt
  change source.val < 17 at sourceBound
  norm_num [PiRLC.v1_1.SamplerChain.sourceOffset,
    PiRLC.v1_1.Formal.commitmentOffset,
    PiRLC.v1_1.Formal.samplerOffset,
    PiRLC.v1_1.SamplerChain.logicalPrivateCount,
    PiRLC.v1_1.Sampler.logicalPrivateCount]

private theorem sourceCommitmentBelow
    (source : Fin productionShape.sourceCount)
    (row : Fin productionProfile.commitmentWidth)
    (lane : Fin ringDegree) :
    ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).commitment row lane).VarsBelow
        PiRLCInputs.phaseOffset := by
  have below := PiCCSInputs.externalInputsBelow logicalWidth publicFits
  unfold PiRLCInputs.sourceInput PiRLCInputs.piCcsInterface
  split
  · rename_i freshBound
    apply Expr.VarsBelow.mono _
      (below.freshCommitment ⟨source.val, freshBound⟩ row lane)
      piCcsPhase_le_piRlcPhase
  · rename_i notFresh
    let running : Fin productionShape.runningCount :=
      ⟨source.val - productionShape.freshCount, by
        have sourceBound := source.isLt
        change source.val < 17 at sourceBound
        change source.val - 1 < 16
        omega⟩
    apply Expr.VarsBelow.mono _
      (below.runningCommitment running row lane) piCcsPhase_le_piRlcPhase

private theorem sourcePublicInputBelow
    (source : Fin productionShape.sourceCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).publicInput column).VarsBelow
        PiRLCInputs.phaseOffset := by
  have below := PiCCSInputs.externalInputsBelow logicalWidth publicFits
  unfold PiRLCInputs.sourceInput PiRLCInputs.piCcsInterface
  split
  · rename_i freshBound
    apply Expr.VarsBelow.mono _
      (below.freshPublicInput ⟨source.val, freshBound⟩ column)
      piCcsPhase_le_piRlcPhase
  · rename_i notFresh
    let running : Fin productionShape.runningCount :=
      ⟨source.val - productionShape.freshCount, by
        have sourceBound := source.isLt
        change source.val < 17 at sourceBound
        change source.val - 1 < 16
        omega⟩
    apply Expr.VarsBelow.mono _
      (below.runningPublicInput running column) piCcsPhase_le_piRlcPhase

private theorem sourceEvalKBelow
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) :
    ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).evaluation.eval_K coefficient
      ).VarsBelow PiRLCInputs.phaseOffset := by
  have below := PiCCSInputs.externalInputsBelow logicalWidth publicFits
  unfold PiRLCInputs.sourceInput PiRLCInputs.piCcsInterface
  split <;> apply KExpr.varsBelow_mono _
      (below.outputEval_K source coefficient) piCcsPhase_le_piRlcPhase

private theorem sourceEvalABelow
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).evaluation.eval_A matrix coefficient
      ).VarsBelow PiRLCInputs.phaseOffset := by
  have below := PiCCSInputs.externalInputsBelow logicalWidth publicFits
  unfold PiRLCInputs.sourceInput PiRLCInputs.piCcsInterface
  split <;> apply KExpr.varsBelow_mono _
      (below.outputEval_A source matrix coefficient) piCcsPhase_le_piRlcPhase

private theorem inputInstance_ext
    (left right : PiRLC.v1_1.InputBinding.InputInstance logicalWidth publicFits)
    (constraintSystem : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

private theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) :
    left = right := by
  cases left
  cases right
  simp_all

/-- One fixed source claim evaluates equally across environments that agree
below the PiRLC boundary, provided its verifier point evaluates equally. -/
theorem sourceInput_eval_eq_of_point_and_agree_below
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (source : Fin productionShape.sourceCount)
    (leftPoint rightPoint : Fin productionShape.cubeVariables → KExpr)
    (left right : Env)
    (pointEq : PiRLC.v1_1.InputBinding.evalPoint leftPoint left =
      PiRLC.v1_1.InputBinding.evalPoint rightPoint right)
    (agrees : ∀ index, index < PiRLCInputs.phaseOffset →
      left index = right index) :
    PiRLC.v1_1.InputBinding.evalInput relation
        (PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits) source)
        leftPoint left =
      PiRLC.v1_1.InputBinding.evalInput relation
        (PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits) source)
        rightPoint right := by
  apply inputInstance_ext
  · rfl
  · funext row coefficient
    exact Expr.eval_eq_of_agree_below _ PiRLCInputs.phaseOffset left right
      (sourceCommitmentBelow source row coefficient) agrees
  · funext column
    exact Expr.eval_eq_of_agree_below _ PiRLCInputs.phaseOffset left right
      (sourcePublicInputBelow source column) agrees
  · exact pointEq
  · apply congrArg (fun value => #[value])
    apply evaluation_ext
    · funext coefficient
      exact KExpr.eval_eq_of_agree_below _ PiRLCInputs.phaseOffset left right
        (sourceEvalKBelow source coefficient) agrees
    · funext matrix coefficient
      exact KExpr.eval_eq_of_agree_below _ PiRLCInputs.phaseOffset left right
        (sourceEvalABelow source matrix coefficient) agrees
  · rfl

/-- The production PiRLC assumptions follow from the fixed zero-copy layout. -/
theorem assumptions
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    PiRLC.v1_1.Formal.Assumptions relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) PiRLCInputs.phaseOffset env := by
  let shared := PiRLC.v1_1.Formal.atOffset
    (PiRLCInputs.interface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) PiRLCInputs.phaseOffset
  refine {
    sampler := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · refine ⟨?_⟩
    intro lane
    simpa [shared, PiRLC.v1_1.Formal.samplerInterface,
      PiRLC.v1_1.Formal.atOffset, PiRLCInputs.interface] using
        samplerInitialBelow relation env lane
  · refine {
      challengeBelow := ?_
      inputBelow := ?_ }
    · intro source lane
      simpa [shared, PiRLC.v1_1.Formal.commitmentInterface,
        PiRLC.v1_1.CommitmentCombination.familyInterface] using
          samplerChallengeBelow (logicalWidth := logicalWidth)
            (publicFits := publicFits) source lane
    · intro source row lane cell
      apply Expr.VarsBelow.mono _
        (sourceCommitmentBelow (logicalWidth := logicalWidth)
          (publicFits := publicFits) source row lane)
      exact phase_le_commitment
  · refine {
      challengeBelow := ?_
      inputBelow := ?_ }
    · intro source lane
      apply Expr.VarsBelow.mono _
        (samplerChallengeBelow (logicalWidth := logicalWidth)
          (publicFits := publicFits) source lane)
      exact commitment_le_publicInput
    · intro source block lane cell
      apply Expr.VarsBelow.mono _
        (sourcePublicInputBelow (logicalWidth := logicalWidth)
          (publicFits := publicFits) source
          (PiRLC.v1_1.PublicInputCombination.publicColumn block lane))
      exact Nat.le_trans phase_le_commitment commitment_le_publicInput
  · refine {
      challengeBelow := ?_
      inputBelow := ?_ }
    · intro source lane
      apply Expr.VarsBelow.mono _
        (samplerChallengeBelow (logicalWidth := logicalWidth)
          (publicFits := publicFits) source lane)
      exact Nat.le_trans commitment_le_publicInput publicInput_le_evalK
    · intro source block lane cell
      change (PiRLC.v1_1.RingKCombination.expressionCell cell
        ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
          (publicFits := publicFits) source).evaluation.eval_K
            (PiRLC.v1_1.EvalKCombination.coefficient lane))).VarsBelow _
      have below := sourceEvalKBelow (logicalWidth := logicalWidth)
        (publicFits := publicFits) source
        (PiRLC.v1_1.EvalKCombination.coefficient lane)
      have bound := Nat.le_trans
        (Nat.le_trans phase_le_commitment commitment_le_publicInput)
        publicInput_le_evalK
      fin_cases cell
      · exact Expr.VarsBelow.mono _ below.1 bound
      · exact Expr.VarsBelow.mono _ below.2 bound
  · refine {
      challengeBelow := ?_
      inputBelow := ?_ }
    · intro source lane
      apply Expr.VarsBelow.mono _
        (samplerChallengeBelow (logicalWidth := logicalWidth)
          (publicFits := publicFits) source lane)
      exact Nat.le_trans
        (Nat.le_trans commitment_le_publicInput publicInput_le_evalK)
        evalK_le_evalA
    · intro source matrix lane cell
      change (PiRLC.v1_1.RingKCombination.expressionCell cell
        ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
          (publicFits := publicFits) source).evaluation.eval_A matrix
            (PiRLC.v1_1.EvalKCombination.coefficient lane))).VarsBelow _
      have below := sourceEvalABelow (logicalWidth := logicalWidth)
        (publicFits := publicFits) source matrix
        (PiRLC.v1_1.EvalKCombination.coefficient lane)
      have bound := Nat.le_trans
        (Nat.le_trans
          (Nat.le_trans phase_le_commitment commitment_le_publicInput)
          publicInput_le_evalK) evalK_le_evalA
      fin_cases cell
      · exact Expr.VarsBelow.mono _ below.1 bound
      · exact Expr.VarsBelow.mono _ below.2 bound

end NightstreamFPrime.Layout.Stage1.PiRLCInputBounds
