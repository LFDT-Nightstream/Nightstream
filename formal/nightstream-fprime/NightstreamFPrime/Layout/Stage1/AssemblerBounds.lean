import NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions
import NightstreamFPrime.Layout.Stage1.AssemblerInputs
import NightstreamFPrime.Layout.Stage1.AssemblerPilotBounds
import NightstreamFPrime.Layout.Stage1.PiDECInputBounds
import NightstreamFPrime.Layout.Stage1.RunningTransitionBounds
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.FormalRows

/-!
Owns causal input bounds for the compact Stage 1 logical assembler.

This file moves existing source-bound proofs to the compact child offsets. It
does not define a circuit, semantic predicate, row, or physical placement.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerBounds

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem piCcsPhaseOffset_le
    (program : Lifecycle.Stage1.Application.Program) :
    PiCCSInputs.phaseOffset ≤ AssemblerInputs.piCcsOffset program := by
  unfold AssemblerInputs.piCcsOffset AssemblerInputs.outputHashOffset
    AssemblerInputs.priorOffset AssemblerInputs.rootOffset
    AssemblerInputs.applicationLocalStart AssemblerInputs.applicationWitnessStart
  rw [Stage1.Spartan.sourceColumnCount_eq, PiCCSInputs.phaseOffset_eq]
  omega

/-- The unchanged caller-owned PiCCS expressions remain below the later
compact parent offset. -/
def piCcsExternalInputsLinear
    (program : Lifecycle.Stage1.Application.Program) :
    NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.ExternalInputsLinear
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program) := by
  let canonical := PiCCSInputs.externalInputsLinear logicalWidth publicFits
  have below : PiCCS.v1_1.Formal.ExternalInputsBelow
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program) := by
    have source := PiCCSInputs.externalInputsBelow logicalWidth publicFits
    have le := piCcsPhaseOffset_le program
    refine {
      priorStateFixed := fun word member => Expr.VarsBelow.mono _
        (source.priorStateFixed word member) le
      outputStateFixed := fun word member => Expr.VarsBelow.mono _
        (source.outputStateFixed word member) le
      priorStateContext := fun lane => Expr.VarsBelow.mono _
        (source.priorStateContext lane) le
      outputStateContext := fun lane => Expr.VarsBelow.mono _
        (source.outputStateContext lane) le
      expectedContext := fun lane => Expr.VarsBelow.mono _
        (source.expectedContext lane) le
      runningPoint := fun coordinate => ⟨
        Expr.VarsBelow.mono _ (source.runningPoint coordinate).1 le,
        Expr.VarsBelow.mono _ (source.runningPoint coordinate).2 le⟩
      runningCommitment := fun sourceIndex row coefficient =>
        Expr.VarsBelow.mono _
          (source.runningCommitment sourceIndex row coefficient) le
      runningPublicInput := fun sourceIndex column => Expr.VarsBelow.mono _
        (source.runningPublicInput sourceIndex column) le
      runningEval_K := fun sourceIndex coefficient => ⟨
        Expr.VarsBelow.mono _ (source.runningEval_K sourceIndex coefficient).1 le,
        Expr.VarsBelow.mono _ (source.runningEval_K sourceIndex coefficient).2 le⟩
      runningEval_A := fun sourceIndex matrix coefficient => ⟨
        Expr.VarsBelow.mono _
          (source.runningEval_A sourceIndex matrix coefficient).1 le,
        Expr.VarsBelow.mono _
          (source.runningEval_A sourceIndex matrix coefficient).2 le⟩
      freshCommitment := fun sourceIndex row coefficient => Expr.VarsBelow.mono _
        (source.freshCommitment sourceIndex row coefficient) le
      freshPublicInput := fun sourceIndex column => Expr.VarsBelow.mono _
        (source.freshPublicInput sourceIndex column) le
      roundCoefficient := fun roundIndex coefficient => ⟨
        Expr.VarsBelow.mono _
          (source.roundCoefficient roundIndex coefficient).1 le,
        Expr.VarsBelow.mono _
          (source.roundCoefficient roundIndex coefficient).2 le⟩
      outputEval_K := fun sourceIndex coefficient => ⟨
        Expr.VarsBelow.mono _ (source.outputEval_K sourceIndex coefficient).1 le,
        Expr.VarsBelow.mono _ (source.outputEval_K sourceIndex coefficient).2 le⟩
      outputEval_A := fun sourceIndex matrix coefficient => ⟨
        Expr.VarsBelow.mono _
          (source.outputEval_A sourceIndex matrix coefficient).1 le,
        Expr.VarsBelow.mono _
          (source.outputEval_A sourceIndex matrix coefficient).2 le⟩ }
  exact { canonical with below := below }

/-- The compact PiCCS child has every causal assumption required by its sole
`FormalCircuit`. -/
def piCcsAssumptions
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    PiCCS.v1_1.Formal.Assumptions relation
      (AssemblerInputs.piCcsInterface program)
      (AssemblerInputs.piCcsOffset program) env :=
  NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation
    (AssemblerInputs.piCcsInterface program)
    (AssemblerInputs.piCcsOffset program)
    (piCcsExternalInputsLinear program) env

private theorem piCcs_le_piRlc
    (program : Lifecycle.Stage1.Application.Program) :
    AssemblerInputs.piCcsOffset program ≤
      AssemblerInputs.piRlcOffset program := by
  unfold AssemblerInputs.piRlcOffset
  omega

private theorem piRlc_le_commitment
    (program : Lifecycle.Stage1.Application.Program) :
    AssemblerInputs.piRlcOffset program ≤
      PiRLC.v1_1.Formal.commitmentOffset
        (AssemblerInputs.piRlcOffset program) := by
  unfold PiRLC.v1_1.Formal.commitmentOffset
    PiRLC.v1_1.Formal.samplerOffset
  omega

private theorem commitment_le_publicInput
    (program : Lifecycle.Stage1.Application.Program) :
    PiRLC.v1_1.Formal.commitmentOffset
        (AssemblerInputs.piRlcOffset program) ≤
      PiRLC.v1_1.Formal.publicInputOffset
        (AssemblerInputs.piRlcOffset program) := by
  unfold PiRLC.v1_1.Formal.publicInputOffset
  omega

private theorem publicInput_le_evalK
    (program : Lifecycle.Stage1.Application.Program) :
    PiRLC.v1_1.Formal.publicInputOffset
        (AssemblerInputs.piRlcOffset program) ≤
      PiRLC.v1_1.Formal.evalKOffset
        (AssemblerInputs.piRlcOffset program) := by
  unfold PiRLC.v1_1.Formal.evalKOffset
  omega

private theorem evalK_le_evalA
    (program : Lifecycle.Stage1.Application.Program) :
    PiRLC.v1_1.Formal.evalKOffset
        (AssemblerInputs.piRlcOffset program) ≤
      PiRLC.v1_1.Formal.evalAOffset
        (AssemblerInputs.piRlcOffset program) := by
  unfold PiRLC.v1_1.Formal.evalAOffset
  omega

/-- Every lane of the PiCCS output transcript state is allocated before the
PiRLC phase starts. -/
theorem piCcsOutputStateBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (lane : Fin Spec.Poseidon2.width) :
    (AssemblerInputs.piCcsOutputState relation program lane).VarsBelow
      (AssemblerInputs.piRlcOffset program) := by
  let interface := AssemblerInputs.piCcsInterface
    (logicalWidth := logicalWidth) (publicFits := publicFits) program
  let outputAt := PiCCS.v1_1.Formal.outputBindingOffset relation interface
    (AssemblerInputs.piCcsOffset program)
  have outputAssumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.outputBinding relation
      interface (AssemblerInputs.piCcsOffset program)
      (piCcsExternalInputsLinear program) env
  have bound := PiCCS.v1_1.OutputBinding.finalState_varsBelow
    (PiCCS.v1_1.Formal.outputBindingInterface
      (PiCCS.v1_1.Formal.atOffset interface
        (AssemblerInputs.piCcsOffset program))) outputAt env
      outputAssumptions lane
  apply Expr.VarsBelow.mono _ bound
  have endEq : outputAt + localLength
      (Circuit.ops
        (PiCCS.v1_1.OutputBinding.circuit
          (PiCCS.v1_1.Formal.outputBindingInterface
            (PiCCS.v1_1.Formal.atOffset interface
              (AssemblerInputs.piCcsOffset program)))).main outputAt) =
      PiCCS.v1_1.Formal.finalOffset relation interface
        (AssemblerInputs.piCcsOffset program) := by
    rfl
  rw [endEq,
    PiCCS.v1_1.Formal.finalOffset_eq_finalRowOffset relation interface,
    PiCCS.v1_1.Formal.finalRowOffset_eq_add_of_degreeBound_eq_nine interface
      (AssemblerInputs.piCcsOffset program) rfl]
  unfold AssemblerInputs.piRlcOffset
  omega

private theorem samplerChallengeBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin productionShape.sourceCount) (lane : Fin ringDegree) :
    (PiRLC.v1_1.SamplerChain.challengeExpr
      (PiRLC.v1_1.Formal.samplerInterface
        (PiRLC.v1_1.Formal.atOffset
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program)))
      (AssemblerInputs.piRlcOffset program) source lane).VarsBelow
        (PiRLC.v1_1.Formal.commitmentOffset
          (AssemblerInputs.piRlcOffset program)) := by
  apply Expr.VarsBelow.mono _
    (PiRLC.v1_1.SamplerChain.challengeExpr_varsBelow
      (PiRLC.v1_1.Formal.samplerInterface
        (PiRLC.v1_1.Formal.atOffset
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program)))
      (AssemblerInputs.piRlcOffset program) source lane)
  have sourceBound := source.isLt
  change source.val < 17 at sourceBound
  norm_num [PiRLC.v1_1.SamplerChain.sourceOffset,
    PiRLC.v1_1.Formal.commitmentOffset,
    PiRLC.v1_1.Formal.samplerOffset,
    PiRLC.v1_1.SamplerChain.logicalPrivateCount,
    PiRLC.v1_1.Sampler.logicalPrivateCount]

private theorem sourceCommitmentBelow
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin productionShape.sourceCount)
    (row : Fin productionProfile.commitmentWidth)
    (lane : Fin ringDegree) :
    ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).commitment row lane).VarsBelow
        (AssemblerInputs.piRlcOffset program) := by
  have below := (piCcsExternalInputsLinear
    (logicalWidth := logicalWidth) (publicFits := publicFits) program).below
  unfold PiRLCInputs.sourceInput PiRLCInputs.piCcsInterface
  split
  · rename_i freshBound
    apply Expr.VarsBelow.mono _
      (below.freshCommitment ⟨source.val, freshBound⟩ row lane)
      (piCcs_le_piRlc program)
  · rename_i notFresh
    let running : Fin productionShape.runningCount :=
      ⟨source.val - productionShape.freshCount, by
        have sourceBound := source.isLt
        change source.val < 17 at sourceBound
        change source.val - 1 < 16
        omega⟩
    apply Expr.VarsBelow.mono _
      (below.runningCommitment running row lane) (piCcs_le_piRlc program)

private theorem sourcePublicInputBelow
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin productionShape.sourceCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).publicInput column).VarsBelow
        (AssemblerInputs.piRlcOffset program) := by
  have below := (piCcsExternalInputsLinear
    (logicalWidth := logicalWidth) (publicFits := publicFits) program).below
  unfold PiRLCInputs.sourceInput PiRLCInputs.piCcsInterface
  split
  · rename_i freshBound
    apply Expr.VarsBelow.mono _
      (below.freshPublicInput ⟨source.val, freshBound⟩ column)
      (piCcs_le_piRlc program)
  · rename_i notFresh
    let running : Fin productionShape.runningCount :=
      ⟨source.val - productionShape.freshCount, by
        have sourceBound := source.isLt
        change source.val < 17 at sourceBound
        change source.val - 1 < 16
        omega⟩
    apply Expr.VarsBelow.mono _
      (below.runningPublicInput running column) (piCcs_le_piRlc program)

private theorem sourceEvalKBelow
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) :
    ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).evaluation.eval_K coefficient
      ).VarsBelow (AssemblerInputs.piRlcOffset program) := by
  have below := (piCcsExternalInputsLinear
    (logicalWidth := logicalWidth) (publicFits := publicFits) program).below
  unfold PiRLCInputs.sourceInput PiRLCInputs.piCcsInterface
  split <;> exact ⟨
    Expr.VarsBelow.mono _ (below.outputEval_K source coefficient).1
      (piCcs_le_piRlc program),
    Expr.VarsBelow.mono _ (below.outputEval_K source coefficient).2
      (piCcs_le_piRlc program)⟩

private theorem sourceEvalABelow
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).evaluation.eval_A matrix coefficient
      ).VarsBelow (AssemblerInputs.piRlcOffset program) := by
  have below := (piCcsExternalInputsLinear
    (logicalWidth := logicalWidth) (publicFits := publicFits) program).below
  unfold PiRLCInputs.sourceInput PiRLCInputs.piCcsInterface
  split <;> exact ⟨
    Expr.VarsBelow.mono _ (below.outputEval_A source matrix coefficient).1
      (piCcs_le_piRlc program),
    Expr.VarsBelow.mono _ (below.outputEval_A source matrix coefficient).2
      (piCcs_le_piRlc program)⟩

/-- The compact PiRLC child has every causal assumption required by its sole
`FormalCircuit`. -/
def piRlcAssumptions
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    PiRLC.v1_1.Formal.Assumptions relation
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program) env := by
  let shared := PiRLC.v1_1.Formal.atOffset
    (AssemblerInputs.piRlcInterface relation program)
    (AssemblerInputs.piRlcOffset program)
  refine {
    sampler := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · refine ⟨?_⟩
    intro lane
    simpa [shared, PiRLC.v1_1.Formal.samplerInterface,
      PiRLC.v1_1.Formal.atOffset, AssemblerInputs.piRlcInterface] using
        piCcsOutputStateBelow relation program env lane
  · refine { challengeBelow := ?_, inputBelow := ?_ }
    · intro source lane
      simpa [shared, PiRLC.v1_1.Formal.commitmentInterface,
        PiRLC.v1_1.CommitmentCombination.familyInterface] using
          samplerChallengeBelow relation program source lane
    · intro source row lane cell
      apply Expr.VarsBelow.mono _
        (sourceCommitmentBelow program source row lane)
      exact piRlc_le_commitment program
  · refine { challengeBelow := ?_, inputBelow := ?_ }
    · intro source lane
      apply Expr.VarsBelow.mono _
        (samplerChallengeBelow relation program source lane)
      exact commitment_le_publicInput program
    · intro source block lane cell
      apply Expr.VarsBelow.mono _
        (sourcePublicInputBelow program source
          (PiRLC.v1_1.PublicInputCombination.publicColumn block lane))
      exact Nat.le_trans (piRlc_le_commitment program)
        (commitment_le_publicInput program)
  · refine { challengeBelow := ?_, inputBelow := ?_ }
    · intro source lane
      apply Expr.VarsBelow.mono _
        (samplerChallengeBelow relation program source lane)
      exact Nat.le_trans (commitment_le_publicInput program)
        (publicInput_le_evalK program)
    · intro source block lane cell
      change (PiRLC.v1_1.RingKCombination.expressionCell cell
        ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
          (publicFits := publicFits) source).evaluation.eval_K
            (PiRLC.v1_1.EvalKCombination.coefficient lane))).VarsBelow _
      have below := sourceEvalKBelow
        (logicalWidth := logicalWidth) (publicFits := publicFits) program source
        (PiRLC.v1_1.EvalKCombination.coefficient lane)
      have bound := Nat.le_trans
        (Nat.le_trans (piRlc_le_commitment program)
          (commitment_le_publicInput program))
        (publicInput_le_evalK program)
      fin_cases cell
      · exact Expr.VarsBelow.mono _ below.1 bound
      · exact Expr.VarsBelow.mono _ below.2 bound
  · refine { challengeBelow := ?_, inputBelow := ?_ }
    · intro source lane
      apply Expr.VarsBelow.mono _
        (samplerChallengeBelow relation program source lane)
      exact Nat.le_trans
        (Nat.le_trans (commitment_le_publicInput program)
          (publicInput_le_evalK program))
        (evalK_le_evalA program)
    · intro source matrix lane cell
      change (PiRLC.v1_1.RingKCombination.expressionCell cell
        ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
          (publicFits := publicFits) source).evaluation.eval_A matrix
            (PiRLC.v1_1.EvalKCombination.coefficient lane))).VarsBelow _
      have below := sourceEvalABelow
        (logicalWidth := logicalWidth) (publicFits := publicFits) program source
        matrix
        (PiRLC.v1_1.EvalKCombination.coefficient lane)
      have bound := Nat.le_trans
        (Nat.le_trans
          (Nat.le_trans (piRlc_le_commitment program)
            (commitment_le_publicInput program))
          (publicInput_le_evalK program))
        (evalK_le_evalA program)
      fin_cases cell
      · exact Expr.VarsBelow.mono _ below.1 bound
      · exact Expr.VarsBelow.mono _ below.2 bound

/-- The shared PiCCS round point is fully allocated before PiRLC starts. -/
theorem piCcsRoundPointBelowPiRlc
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (coordinate : Fin productionShape.cubeVariables) :
    (AssemblerInputs.piCcsRoundPoint
      (logicalWidth := logicalWidth) (publicFits := publicFits) program
      coordinate).VarsBelow (AssemblerInputs.piRlcOffset program) := by
  let interface := AssemblerInputs.piCcsInterface
    (logicalWidth := logicalWidth) (publicFits := publicFits) program
  have assumptions := (piCcsAssumptions relation program env).roundTranscript
  have bound := PiCCS.v1_1.RoundTranscript.challenge_varsBelow
    (PiCCS.v1_1.Formal.roundTranscriptInterface
      (PiCCS.v1_1.Formal.atOffset interface
        (AssemblerInputs.piCcsOffset program)))
    (PiCCS.v1_1.Formal.roundTranscriptOffset interface
      (AssemblerInputs.piCcsOffset program)) env assumptions coordinate
  change (PiCCS.v1_1.RoundTranscript.challenge
      (PiCCS.v1_1.Formal.roundTranscriptInterface
        (PiCCS.v1_1.Formal.atOffset interface
          (AssemblerInputs.piCcsOffset program)))
      (PiCCS.v1_1.Formal.roundTranscriptOffset interface
        (AssemblerInputs.piCcsOffset program)) coordinate).VarsBelow _ at bound
  have le : PiCCS.v1_1.Formal.roundTranscriptOffset interface
        (AssemblerInputs.piCcsOffset program) +
      localLength (Circuit.ops
        (PiCCS.v1_1.RoundTranscript.circuit
          (PiCCS.v1_1.Formal.roundTranscriptInterface
            (PiCCS.v1_1.Formal.atOffset interface
              (AssemblerInputs.piCcsOffset program)))).main
        (PiCCS.v1_1.Formal.roundTranscriptOffset interface
          (AssemblerInputs.piCcsOffset program))) ≤
      AssemblerInputs.piRlcOffset program := by
    rw [PiCCS.v1_1.RoundTranscript.localLength_eq]
    rw [PiCCS.v1_1.Formal.roundTranscriptOffset_eq,
      PiCCS.v1_1.Formal.challengeOffset_eq]
    norm_num [PiCCS.v1_1.RoundTranscript.perRoundRecipeCount,
      productionShape, Phi81MatrixSource.phi81Shape, cubeVariables,
      AssemblerInputs.piRlcOffset]
  rw [AssemblerInputs.piCcsRoundPoint_eq_challenge]
  exact ⟨Expr.VarsBelow.mono _ bound.1 le,
    Expr.VarsBelow.mono _ bound.2 le⟩

private theorem inputInstance_ext
    (left right : PiRLC.v1_1.InputBinding.InputInstance
      logicalWidth publicFits)
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

private theorem point_ext (left right : PaperAlgebra.Point)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- Agreement below the PiRLC boundary preserves its shared point exactly. -/
theorem piRlcPoint_eq_of_agree_below
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (left right : Env)
    (agrees : ∀ index, index < AssemblerInputs.piRlcOffset program →
      left index = right index) :
    PiRLC.v1_1.InputBinding.evalPoint
        (AssemblerInputs.piCcsRoundPoint
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        left =
      PiRLC.v1_1.InputBinding.evalPoint
        (AssemblerInputs.piCcsRoundPoint
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        right := by
  apply point_ext
  change (List.ofFn fun coordinate =>
      (AssemblerInputs.piCcsRoundPoint
        (logicalWidth := logicalWidth) (publicFits := publicFits) program
        coordinate).eval left) =
    List.ofFn fun coordinate =>
      (AssemblerInputs.piCcsRoundPoint
        (logicalWidth := logicalWidth) (publicFits := publicFits) program
        coordinate).eval right
  apply congrArg List.ofFn
  funext coordinate
  have below := piCcsRoundPointBelowPiRlc relation program left coordinate
  exact congrArg₂ K.mk
    (Expr.eval_eq_of_agree_below _ (AssemblerInputs.piRlcOffset program)
      left right below.1 agrees)
    (Expr.eval_eq_of_agree_below _ (AssemblerInputs.piRlcOffset program)
      left right below.2 agrees)

/-- Agreement below the PiRLC boundary preserves the exact eight-word
sampler seed produced by PiCCS. -/
theorem piRlcInitialState_eq_of_agree_below
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (left right : Env)
    (agrees : ∀ index, index < AssemblerInputs.piRlcOffset program →
      left index = right index) :
    PiRLC.v1_1.SamplerChain.evalInitialState
        (PiRLC.v1_1.Formal.samplerInterface
          (PiRLC.v1_1.Formal.atOffset
            (AssemblerInputs.piRlcInterface relation program)
            (AssemblerInputs.piRlcOffset program)))
        (PiRLC.v1_1.Formal.samplerOffset
          (AssemblerInputs.piRlcOffset program)) left =
      PiRLC.v1_1.SamplerChain.evalInitialState
        (PiRLC.v1_1.Formal.samplerInterface
          (PiRLC.v1_1.Formal.atOffset
            (AssemblerInputs.piRlcInterface relation program)
            (AssemblerInputs.piRlcOffset program)))
        (PiRLC.v1_1.Formal.samplerOffset
          (AssemblerInputs.piRlcOffset program)) right := by
  unfold PiRLC.v1_1.SamplerChain.evalInitialState
    PiRLC.v1_1.SamplerChain.evalStateAt PiRLC.v1_1.Sampler.evalState
    NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
  apply congrArg List.ofFn
  funext lane
  apply Expr.eval_eq_of_agree_below _
    (AssemblerInputs.piRlcOffset program) left right
  · simpa [PiRLC.v1_1.Formal.samplerInterface,
      PiRLC.v1_1.Formal.atOffset, PiRLC.v1_1.Formal.samplerOffset,
      AssemblerInputs.piRlcInterface] using
        piCcsOutputStateBelow relation program left lane
  · exact agrees

/-- Agreement below the PiRLC boundary preserves all 17 exact typed input
instances, including separate `Eval_K` and `Eval_A` families. -/
theorem piRlcInputs_eq_of_agree_below
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (left right : Env)
    (agrees : ∀ index, index < AssemblerInputs.piRlcOffset program →
      left index = right index) :
    PiRLC.v1_1.Semantics.evalInputs relation
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program) left =
      PiRLC.v1_1.Semantics.evalInputs relation
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program) right := by
  funext source
  let sourceIndex := PiRLC.v1_1.Semantics.sourceIndex source
  change PiRLC.v1_1.InputBinding.evalInput relation
      (PiRLCInputs.sourceInput sourceIndex)
      (AssemblerInputs.piCcsRoundPoint
        (logicalWidth := logicalWidth) (publicFits := publicFits) program) left =
    PiRLC.v1_1.InputBinding.evalInput relation
      (PiRLCInputs.sourceInput sourceIndex)
      (AssemblerInputs.piCcsRoundPoint
        (logicalWidth := logicalWidth) (publicFits := publicFits) program) right
  apply inputInstance_ext
  · rfl
  · funext row lane
    exact Expr.eval_eq_of_agree_below _
      (AssemblerInputs.piRlcOffset program) left right
      (sourceCommitmentBelow
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        program sourceIndex row lane) agrees
  · funext column
    exact Expr.eval_eq_of_agree_below _
      (AssemblerInputs.piRlcOffset program) left right
      (sourcePublicInputBelow
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        program sourceIndex column) agrees
  · exact piRlcPoint_eq_of_agree_below relation program left right agrees
  · apply congrArg (fun value => #[value])
    apply evaluation_ext
    · funext coefficient
      have below := sourceEvalKBelow
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        program sourceIndex coefficient
      exact congrArg₂ K.mk
        (Expr.eval_eq_of_agree_below _ (AssemblerInputs.piRlcOffset program)
          left right below.1 agrees)
        (Expr.eval_eq_of_agree_below _ (AssemblerInputs.piRlcOffset program)
          left right below.2 agrees)
    · funext matrix coefficient
      have below := sourceEvalABelow
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        program sourceIndex matrix coefficient
      exact congrArg₂ K.mk
        (Expr.eval_eq_of_agree_below _ (AssemblerInputs.piRlcOffset program)
          left right below.1 agrees)
        (Expr.eval_eq_of_agree_below _ (AssemblerInputs.piRlcOffset program)
          left right below.2 agrees)
  · rfl

private theorem piDecSourceOffset_le
    (program : Lifecycle.Stage1.Application.Program) :
    PiDECInputs.phaseOffset ≤ AssemblerInputs.piDecOffset program := by
  unfold AssemblerInputs.piDecOffset AssemblerInputs.piRlcOffset
    AssemblerInputs.piCcsOffset AssemblerInputs.outputHashOffset
    AssemblerInputs.priorOffset AssemblerInputs.rootOffset
    AssemblerInputs.applicationLocalStart AssemblerInputs.applicationWitnessStart
  rw [Stage1.Spartan.sourceColumnCount_eq]
  norm_num [PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
    PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
    PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
    PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild]
  omega

/-- Every compact PiDEC input is owned before its phase allocation. -/
def piDecInputsBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    PiDEC.v1_1.Formal.InputsBelow
      (AssemblerInputs.piDecInterface relation program)
      (AssemblerInputs.piDecOffset program) := by
  let piRlc := AssemblerInputs.piRlcInterface relation program
  let shared := PiRLC.v1_1.Formal.atOffset piRlc
    (AssemblerInputs.piRlcOffset program)
  have source := PiDECInputs.inputsBelow relation
  have sourceLe := piDecSourceOffset_le program
  refine {
    point := fun coordinate => by
      have bound := piCcsRoundPointBelowPiRlc relation program
        (fun _ => 0) coordinate
      have le : AssemblerInputs.piRlcOffset program ≤
          AssemblerInputs.piDecOffset program := by
        unfold AssemblerInputs.piDecOffset
        omega
      exact ⟨Expr.VarsBelow.mono _ bound.1 le,
        Expr.VarsBelow.mono _ bound.2 le⟩
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
      (PiDECInputs.combinationOutput_varsBelow
        (PiRLC.v1_1.CommitmentCombination.familyInterface
          (PiRLC.v1_1.Formal.commitmentInterface shared))
        (PiRLC.v1_1.Formal.commitmentOffset
          (AssemblerInputs.piRlcOffset program)) row lane
        PiRLC.v1_1.CommitmentCombination.cell)
    rw [PiRLC.v1_1.CommitmentCombination.logicalPrivateCount_eq]
    norm_num [PiRLC.v1_1.Formal.commitmentOffset,
      PiRLC.v1_1.Formal.samplerOffset,
      PiRLC.v1_1.SamplerChain.logicalPrivateCount,
      PiRLC.v1_1.SamplerChain.sourceCount_eq,
      PiRLC.v1_1.Sampler.logicalPrivateCount, AssemblerInputs.piDecOffset]
  · intro column
    apply Expr.VarsBelow.mono _
      (PiDECInputs.combinationOutput_varsBelow
        (PiRLC.v1_1.PublicInputCombination.familyInterface
          (PiRLC.v1_1.Formal.publicInputInterface shared))
        (PiRLC.v1_1.Formal.publicInputOffset
          (AssemblerInputs.piRlcOffset program))
        (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
          (FullShape logicalWidth publicFits) column)
        (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex
          column) PiRLC.v1_1.PublicInputCombination.cell)
    rw [PiRLC.v1_1.PublicInputCombination.logicalPrivateCount_eq]
    norm_num [PiRLC.v1_1.Formal.publicInputOffset,
      PiRLC.v1_1.Formal.commitmentOffset,
      PiRLC.v1_1.Formal.samplerOffset,
      PiRLC.v1_1.SamplerChain.logicalPrivateCount,
      PiRLC.v1_1.SamplerChain.sourceCount_eq,
      PiRLC.v1_1.Sampler.logicalPrivateCount, AssemblerInputs.piDecOffset]
  · intro coefficient
    constructor
    · apply Expr.VarsBelow.mono _
        (PiDECInputs.combinationOutput_varsBelow
          (PiRLC.v1_1.RingKCombination.familyInterface
            (PiRLC.v1_1.EvalKCombination.ringInterface
              (PiRLC.v1_1.Formal.evalKInterface shared)))
          (PiRLC.v1_1.Formal.evalKOffset
            (AssemblerInputs.piRlcOffset program))
          PiRLC.v1_1.EvalKCombination.block
          (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq
            coefficient) PiRLC.v1_1.RingKCombination.c0Cell)
      rw [PiRLC.v1_1.EvalKCombination.logicalPrivateCount_eq]
      norm_num [PiRLC.v1_1.Formal.evalKOffset,
        PiRLC.v1_1.Formal.publicInputOffset,
        PiRLC.v1_1.Formal.commitmentOffset,
        PiRLC.v1_1.Formal.samplerOffset,
        PiRLC.v1_1.SamplerChain.logicalPrivateCount,
        PiRLC.v1_1.SamplerChain.sourceCount_eq,
        PiRLC.v1_1.Sampler.logicalPrivateCount, AssemblerInputs.piDecOffset]
    · apply Expr.VarsBelow.mono _
        (PiDECInputs.combinationOutput_varsBelow
          (PiRLC.v1_1.RingKCombination.familyInterface
            (PiRLC.v1_1.EvalKCombination.ringInterface
              (PiRLC.v1_1.Formal.evalKInterface shared)))
          (PiRLC.v1_1.Formal.evalKOffset
            (AssemblerInputs.piRlcOffset program))
          PiRLC.v1_1.EvalKCombination.block
          (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq
            coefficient) PiRLC.v1_1.RingKCombination.c1Cell)
      rw [PiRLC.v1_1.EvalKCombination.logicalPrivateCount_eq]
      norm_num [PiRLC.v1_1.Formal.evalKOffset,
        PiRLC.v1_1.Formal.publicInputOffset,
        PiRLC.v1_1.Formal.commitmentOffset,
        PiRLC.v1_1.Formal.samplerOffset,
        PiRLC.v1_1.SamplerChain.logicalPrivateCount,
        PiRLC.v1_1.SamplerChain.sourceCount_eq,
        PiRLC.v1_1.Sampler.logicalPrivateCount, AssemblerInputs.piDecOffset]
  · intro matrix coefficient
    constructor
    · apply Expr.VarsBelow.mono _
        (PiDECInputs.combinationOutput_varsBelow
          (PiRLC.v1_1.RingKCombination.familyInterface
            (PiRLC.v1_1.EvalACombination.ringInterface
              (PiRLC.v1_1.Formal.evalAInterface shared)))
          (PiRLC.v1_1.Formal.evalAOffset
            (AssemblerInputs.piRlcOffset program)) matrix
          (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq
            coefficient) PiRLC.v1_1.RingKCombination.c0Cell)
      rw [PiRLC.v1_1.EvalACombination.logicalPrivateCount_eq]
      norm_num [PiRLC.v1_1.Formal.evalAOffset,
        PiRLC.v1_1.Formal.evalKOffset,
        PiRLC.v1_1.Formal.publicInputOffset,
        PiRLC.v1_1.Formal.commitmentOffset,
        PiRLC.v1_1.Formal.samplerOffset,
        PiRLC.v1_1.SamplerChain.logicalPrivateCount,
        PiRLC.v1_1.SamplerChain.sourceCount_eq,
        PiRLC.v1_1.Sampler.logicalPrivateCount, AssemblerInputs.piDecOffset]
    · apply Expr.VarsBelow.mono _
        (PiDECInputs.combinationOutput_varsBelow
          (PiRLC.v1_1.RingKCombination.familyInterface
            (PiRLC.v1_1.EvalACombination.ringInterface
              (PiRLC.v1_1.Formal.evalAInterface shared)))
          (PiRLC.v1_1.Formal.evalAOffset
            (AssemblerInputs.piRlcOffset program)) matrix
          (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq
            coefficient) PiRLC.v1_1.RingKCombination.c1Cell)
      rw [PiRLC.v1_1.EvalACombination.logicalPrivateCount_eq]
      norm_num [PiRLC.v1_1.Formal.evalAOffset,
        PiRLC.v1_1.Formal.evalKOffset,
        PiRLC.v1_1.Formal.publicInputOffset,
        PiRLC.v1_1.Formal.commitmentOffset,
        PiRLC.v1_1.Formal.samplerOffset,
        PiRLC.v1_1.SamplerChain.logicalPrivateCount,
        PiRLC.v1_1.SamplerChain.sourceCount_eq,
        PiRLC.v1_1.Sampler.logicalPrivateCount, AssemblerInputs.piDecOffset]
  · intro child row lane
    simpa [AssemblerInputs.piDecInterface] using
      Expr.VarsBelow.mono _ (source.messageCommitment child row lane) sourceLe
  · intro child coefficient
    exact ⟨
      Expr.VarsBelow.mono _ (source.messageEval_K child coefficient).1 sourceLe,
      Expr.VarsBelow.mono _ (source.messageEval_K child coefficient).2 sourceLe⟩
  · intro child matrix coefficient
    exact ⟨
      Expr.VarsBelow.mono _
        (source.messageEval_A child matrix coefficient).1 sourceLe,
      Expr.VarsBelow.mono _
        (source.messageEval_A child matrix coefficient).2 sourceLe⟩
  · intro child coordinate
    simpa [AssemblerInputs.piDecInterface] using
      Expr.VarsBelow.mono _ (source.digit child coordinate) sourceLe

/-- The compact PiDEC child has every causal assumption required by its sole
`FormalCircuit`. -/
def piDecAssumptions
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    PiDEC.v1_1.Formal.Assumptions relation
      (AssemblerInputs.piDecInterface relation program)
      (AssemblerInputs.piDecOffset program) env :=
  ⟨piDecInputsBelow relation program⟩

private theorem piDec_le_running
    (program : Lifecycle.Stage1.Application.Program) :
    AssemblerInputs.piDecOffset program ≤
      AssemblerInputs.runningOffset program := by
  unfold AssemblerInputs.runningOffset
  omega

private theorem sourceRunning_le_running
    (program : Lifecycle.Stage1.Application.Program) :
    RunningTransitionInputs.phaseOffset ≤
      AssemblerInputs.runningOffset program := by
  unfold AssemblerInputs.runningOffset AssemblerInputs.piDecOffset
    AssemblerInputs.piRlcOffset AssemblerInputs.piCcsOffset
    AssemblerInputs.outputHashOffset AssemblerInputs.priorOffset
    AssemblerInputs.rootOffset AssemblerInputs.applicationLocalStart
    AssemblerInputs.applicationWitnessStart
  rw [Stage1.Spartan.sourceColumnCount_eq]
  norm_num [RunningTransitionInputs.phaseOffset]
  omega

private def recursiveRunningBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.RunningTransition.RunningBelow
      (AssemblerInputs.recursiveRunningExpr relation program)
      (AssemblerInputs.runningOffset program) := by
  have inputs := piDecInputsBelow relation program
  have le := piDec_le_running program
  refine {
    point := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro coordinate
    have below := inputs.point coordinate
    exact ⟨Expr.VarsBelow.mono _ below.1 le,
      Expr.VarsBelow.mono _ below.2 le⟩
  · intro source row coefficient
    simpa [AssemblerInputs.recursiveRunningExpr] using
      Expr.VarsBelow.mono _
        (inputs.messageCommitment
          (AssemblerInputs.childOfRunning source) row coefficient) le
  · intro source column
    simpa [AssemblerInputs.recursiveRunningExpr] using
      Expr.VarsBelow.mono _
        (inputs.digit (AssemblerInputs.childOfRunning source)
          (AssemblerInputs.digitCoordinate column)) le
  · intro source coefficient
    have below := inputs.messageEval_K
      (AssemblerInputs.childOfRunning source) coefficient
    simpa [AssemblerInputs.recursiveRunningExpr] using
      (show ((AssemblerInputs.piDecInterface relation program).message
          (AssemblerInputs.piDecOffset program)
          (AssemblerInputs.childOfRunning source)).evaluation.eval_K
            coefficient |>.VarsBelow
          (AssemblerInputs.runningOffset program) from
        ⟨Expr.VarsBelow.mono _ below.1 le,
          Expr.VarsBelow.mono _ below.2 le⟩)
  · intro source matrix coefficient
    have below := inputs.messageEval_A
      (AssemblerInputs.childOfRunning source) matrix coefficient
    simpa [AssemblerInputs.recursiveRunningExpr] using
      (show ((AssemblerInputs.piDecInterface relation program).message
          (AssemblerInputs.piDecOffset program)
          (AssemblerInputs.childOfRunning source)).evaluation.eval_A
            matrix coefficient |>.VarsBelow
          (AssemblerInputs.runningOffset program) from
        ⟨Expr.VarsBelow.mono _ below.1 le,
          Expr.VarsBelow.mono _ below.2 le⟩)

/-- The compact running-transition child has every causal assumption required
by its sole `FormalCircuit`. -/
def runningAssumptions
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.Stage1.RunningTransition.Assumptions
      (AssemblerInputs.runningInterface relation program)
      (AssemblerInputs.runningOffset program) env := by
  have source := RunningTransitionInputs.assumptions logicalWidth publicFits
    relation env
  have sourceLe := sourceRunning_le_running program
  refine {
    iteration := ?_
    initialState := ?_
    currentState := ?_
    recursive := ?_
    output := ?_ }
  · simpa [AssemblerInputs.runningInterface] using
      Expr.VarsBelow.mono _ source.iteration sourceLe
  · intro index
    simpa [AssemblerInputs.runningInterface] using
      Expr.VarsBelow.mono _ (source.initialState index) sourceLe
  · intro index
    simpa [AssemblerInputs.runningInterface] using
      Expr.VarsBelow.mono _ (source.currentState index) sourceLe
  · intro index
    exact Lifecycle.Stage1.RunningTransition.runningWord_varsBelow _
      (AssemblerInputs.runningOffset program)
      (recursiveRunningBelow relation program) index
  · intro index
    exact Lifecycle.Stage1.RunningTransition.runningWord_varsBelow _
      (AssemblerInputs.runningOffset program)
      ((RunningTransitionInputs.outputRunningBelow logicalWidth publicFits).mono
        sourceLe) index

/-- Every verifier-owned application input precedes all compact logical
children. -/
def applicationInputsBelowRoot
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.Application.InputsBelow
      (AssemblerInputs.applicationInterface program)
      (AssemblerInputs.rootOffset program) := by
  refine {
    input := ?_
    witness := ?_
    output := ?_ }
  · intro index
    simp only [AssemblerInputs.applicationInterface, Expr.VarsBelow]
    have bound := index.isLt
    unfold AssemblerInputs.rootOffset
      AssemblerInputs.applicationLocalStart
      AssemblerInputs.applicationWitnessStart
    rw [Stage1.Spartan.sourceColumnCount_eq]
    norm_num [ApplicationInputs.inputSourceColumn,
      ApplicationInputs.currentWordStart,
      PilotProduction.priorPreimageStart,
      Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
    omega
  · intro index
    simp only [AssemblerInputs.applicationInterface,
      AssemblerInputs.applicationWitnessColumn, Expr.VarsBelow]
    have bound := index.isLt
    unfold AssemblerInputs.rootOffset
      AssemblerInputs.applicationLocalStart
      AssemblerInputs.applicationWitnessStart
    omega
  · intro index
    simp only [AssemblerInputs.applicationInterface, Expr.VarsBelow]
    have bound := index.isLt
    unfold AssemblerInputs.rootOffset
      AssemblerInputs.applicationLocalStart
      AssemblerInputs.applicationWitnessStart
    rw [Stage1.Spartan.sourceColumnCount_eq]
    norm_num [ApplicationInputs.outputSourceColumn,
      ApplicationInputs.currentWordStart,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth_eq,
      Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
    omega

/-- Every verifier-owned application input precedes the exact compact
application child start. -/
def applicationInputsBelow
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.Application.InputsBelow
      (AssemblerInputs.applicationInterface program)
      (AssemblerInputs.applicationOffset program) := by
  have source := applicationInputsBelowRoot
    (logicalWidth := logicalWidth) program
  have le : AssemblerInputs.rootOffset program ≤
      AssemblerInputs.applicationOffset program := by
    unfold AssemblerInputs.applicationOffset AssemblerInputs.runningOffset
      AssemblerInputs.piDecOffset AssemblerInputs.piRlcOffset
      AssemblerInputs.piCcsOffset AssemblerInputs.outputHashOffset
      AssemblerInputs.priorOffset
    omega
  exact {
    input := fun index => Expr.VarsBelow.mono _ (source.input index) le
    witness := fun index => Expr.VarsBelow.mono _ (source.witness index) le
    output := fun index => Expr.VarsBelow.mono _ (source.output index) le }

/-- The compact application child receives the exact assumptions of the
verifier-selected Lean program. -/
def applicationAssumptions
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (program.circuit (AssemblerInputs.applicationInterface program)
      ).assumptions (AssemblerInputs.applicationOffset program) env :=
  program.assumptions (AssemblerInputs.applicationInterface program)
    (AssemblerInputs.applicationOffset program) env
    (applicationInputsBelow
      (logicalWidth := logicalWidth) program)

/-- The exact seven child assumptions for the compact Stage 1 parent at its
production root. -/
def stage1Assumptions
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) :
    Lifecycle.Stage1.Assumptions relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program) env := by
  refine {
    prior := ?_
    outputHash := ?_
    piCcs := ?_
    piRlc := ?_
    piDec := ?_
    running := ?_
    application := ?_ }
  · rw [AssemblerInputs.parent_priorOffset_eq relation program]
    exact AssemblerPilotBounds.priorAssumptions program env
  · rw [AssemblerInputs.parent_outputHashOffset_eq relation program]
    exact AssemblerPilotBounds.outputAssumptions program env
  · rw [AssemblerInputs.parent_piCcsOffset_eq relation program]
    exact piCcsAssumptions relation program env
  · rw [AssemblerInputs.parent_piRlcOffset_eq relation ajtai program template]
    exact piRlcAssumptions relation program env
  · rw [AssemblerInputs.parent_piDecOffset_eq relation ajtai program template]
    exact piDecAssumptions relation program env
  · rw [AssemblerInputs.parent_runningOffset_eq relation ajtai program template]
    exact runningAssumptions relation program env
  · rw [AssemblerInputs.parent_applicationOffset_eq relation ajtai program
      template]
    exact applicationAssumptions
      (logicalWidth := logicalWidth) program env

end NightstreamFPrime.Layout.Stage1.AssemblerBounds
