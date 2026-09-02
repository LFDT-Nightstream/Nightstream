import NightstreamFPrime.Layout.Stage1.AssemblerPiRLCCompleteness
import NightstreamFPrime.Layout.Stage1.PiDECSourceSupportData
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.Completeness

/-!
Owns the exact PiRLC-to-PiDEC semantic handoff and the fifth opaque-child
append for the compact Stage 1 assembler. It adds no row, copy value, or
alternate verifier predicate.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerPiDECCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem sourceColumn_agrees
    (program : Lifecycle.Stage1.Application.Program)
    {env : Env}
    (completed : Sequence.Prefix env (AssemblerInputs.rootOffset program))
    {index : Nat} (below : index < Spartan.SourceColumnCount) :
    env index = completed.current index := by
  symm
  apply completed.agrees index
  apply Or.inl
  rw [AssemblerPilotBounds.rootOffset_eq]
  rw [Spartan.sourceColumnCount_eq] at below
  omega

private theorem sourceInput_agrees
    (program : Lifecycle.Stage1.Application.Program)
    {env : Env}
    (completed : Sequence.Prefix env (AssemblerInputs.rootOffset program))
    {index : Nat} (below : index < PiDECInputs.phaseOffset) :
    env index = completed.current index := by
  apply sourceColumn_agrees program completed
  rw [Spartan.sourceColumnCount_eq]
  norm_num [PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
    PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
    PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
    PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild]
    at below ⊢
  omega

private theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem running_ext
    (left right : Spec.Folding.Nifs.PaperNonInteractive.Running K
      PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape)
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem recursiveRunning_eval_eq_of_piDecOutput_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (left right : Env)
    (outputsEq : PiDEC.v1_1.Semantics.output relation
        (AssemblerInputs.piDecInterface relation program)
        (AssemblerInputs.piDecOffset program) left =
      PiDEC.v1_1.Semantics.output relation
        (AssemblerInputs.piDecInterface relation program)
        (AssemblerInputs.piDecOffset program) right) :
    PiCCS.v1_1.StatementAbsorption.evalRunning
        (AssemblerInputs.recursiveRunningExpr relation program) left =
      PiCCS.v1_1.StatementAbsorption.evalRunning
        (AssemblerInputs.recursiveRunningExpr relation program) right := by
  apply running_ext
  · let child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex :=
      ⟨0, by decide⟩
    have childEq := congrFun outputsEq child
    have pointEq := congrArg (fun value => value.point) childEq
    simpa [PiCCS.v1_1.StatementAbsorption.evalRunning,
      AssemblerInputs.recursiveRunningExpr,
      PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface,
      PiDEC.v1_1.Formal.atOffset] using pointEq
  · funext source row coefficient
    have childEq := congrFun outputsEq
      (AssemblerInputs.childOfRunning source)
    have commitmentEq := congrArg (fun value => value.commitment) childEq
    have coordinateEq := congrFun (congrFun commitmentEq row) coefficient
    simpa [PiCCS.v1_1.StatementAbsorption.evalRunning,
      AssemblerInputs.recursiveRunningExpr,
      PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface,
      PiDEC.v1_1.Formal.atOffset] using coordinateEq
  · funext source column
    have childEq := congrFun outputsEq
      (AssemblerInputs.childOfRunning source)
    have publicEq := congrArg (fun value => value.publicInput) childEq
    have coordinateEq := congrFun publicEq
      (AssemblerInputs.digitCoordinate column)
    simpa [PiCCS.v1_1.StatementAbsorption.evalRunning,
      AssemblerInputs.recursiveRunningExpr,
      PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface,
      PiDEC.v1_1.Formal.atOffset] using coordinateEq
  · funext source
    have childEq := congrFun outputsEq
      (AssemblerInputs.childOfRunning source)
    have evaluationEq := congrArg
      (fun value => value.evaluations.getD 0 PaperAlgebra.evaluationZero)
      childEq
    simpa [PiCCS.v1_1.StatementAbsorption.evalRunning,
      AssemblerInputs.recursiveRunningExpr,
      PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface,
      PiDEC.v1_1.Formal.atOffset] using evaluationEq

private theorem piDecOutput_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    {env : Env}
    (completed : Sequence.Prefix env (AssemblerInputs.rootOffset program))
    (piRlcOutputEq :
      PiRLC.v1_1.Semantics.evalOutput relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) env =
        PiRLC.v1_1.Semantics.evalOutput relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) completed.current) :
    PiDEC.v1_1.Semantics.output relation
        (AssemblerInputs.piDecInterface relation program)
        (AssemblerInputs.piDecOffset program) env =
      PiDEC.v1_1.Semantics.output relation
        (AssemblerInputs.piDecInterface relation program)
        (AssemblerInputs.piDecOffset program) completed.current := by
  apply PiDEC.v1_1.Semantics.output_eq_of_components
  · have pointEq := congrArg (fun value => value.point) piRlcOutputEq
    simpa [AssemblerInputs.piDecInterface,
      AssemblerInputs.piRlcOutputInterface,
      PiRLC.v1_1.Semantics.evalOutput,
      PiRLC.v1_1.OutputBinding.evalOutput,
      PiRLC.v1_1.Formal.outputBindingInterface,
      PiRLC.v1_1.Formal.atOffset] using pointEq
  · intro child row lane
    apply sourceInput_agrees program completed
    have childBound := child.isLt
    have rowBound := row.isLt
    have laneBound := lane.isLt
    norm_num [AssemblerInputs.piDecInterface, PiDECInputs.message,
      PiDECInputs.childCommitment, PiDECInputs.childCommitmentStart,
      PiDECInputs.commitmentInputStart, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild, productionProfile, ringDegree]
      at childBound rowBound laneBound ⊢
    omega
  · intro child coordinate
    apply sourceInput_agrees program completed
    have childBound := child.isLt
    have coordinateBound : coordinate.val < 270 := by
      simpa only [PiDEC.v1_1.PublicInputSplit.coordinateCount_eq] using
        coordinate.isLt
    norm_num [AssemblerInputs.piDecInterface,
      PiDECInputs.childPublicInput, PiDECInputs.childPublicInputStart,
      PiDECInputs.publicInputStart, PiDECInputs.evalAInputStart,
      PiDECInputs.evalKInputStart, PiDECInputs.commitmentInputStart,
      PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
      PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
      PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
      PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild]
      at childBound coordinateBound ⊢
    omega
  · intro child
    apply evaluation_ext
    · funext coefficient
      apply congrArg₂ K.mk
      · apply sourceInput_agrees program completed
        have childBound := child.isLt
        have coefficientBound := coefficient.isLt
        norm_num [AssemblerInputs.piDecInterface, PiDECInputs.message,
          PiDECInputs.childEvalK, PiDECInputs.childEvalKStart,
          PiDECInputs.evalKInputStart, PiDECInputs.commitmentInputStart,
          PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
          PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
          PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
          PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
          productionShape, Phi81MatrixSource.phi81Shape, ringDegree]
          at childBound coefficientBound ⊢
        omega
      · apply sourceInput_agrees program completed
        have childBound := child.isLt
        have coefficientBound := coefficient.isLt
        norm_num [AssemblerInputs.piDecInterface, PiDECInputs.message,
          PiDECInputs.childEvalK, PiDECInputs.childEvalKStart,
          PiDECInputs.evalKInputStart, PiDECInputs.commitmentInputStart,
          PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
          PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
          PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
          PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
          productionShape, Phi81MatrixSource.phi81Shape, ringDegree]
          at childBound coefficientBound ⊢
        omega
    · funext matrix coefficient
      apply congrArg₂ K.mk
      · apply sourceInput_agrees program completed
        have childBound := child.isLt
        have matrixBound := matrix.isLt
        have coefficientBound := coefficient.isLt
        norm_num [AssemblerInputs.piDecInterface, PiDECInputs.message,
          PiDECInputs.childEvalA, PiDECInputs.childEvalAStart,
          PiDECInputs.evalAInputStart, PiDECInputs.evalKInputStart,
          PiDECInputs.commitmentInputStart, PiDECInputs.phaseOffset,
          PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
          PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
          PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
          PiDECInputs.publicInputWordsPerChild, productionShape,
          productionProfile, Phi81MatrixSource.phi81Shape, ringDegree]
          at childBound matrixBound coefficientBound ⊢
        omega
      · apply sourceInput_agrees program completed
        have childBound := child.isLt
        have matrixBound := matrix.isLt
        have coefficientBound := coefficient.isLt
        norm_num [AssemblerInputs.piDecInterface, PiDECInputs.message,
          PiDECInputs.childEvalA, PiDECInputs.childEvalAStart,
          PiDECInputs.evalAInputStart, PiDECInputs.evalKInputStart,
          PiDECInputs.commitmentInputStart, PiDECInputs.phaseOffset,
          PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
          PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
          PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
          PiDECInputs.publicInputWordsPerChild, productionShape,
          productionProfile, Phi81MatrixSource.phi81Shape, ringDegree]
          at childBound matrixBound coefficientBound ⊢
        omega

/-- Honest completion through the PiDEC parent child. The canonical PiDEC
builder owns its six internal children, while Stage 1 retains one opaque
PiDEC operation. -/
theorem completePiDecPrefix
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program) env) :
    ∃ completed : Sequence.Prefix env (AssemblerInputs.rootOffset program),
      completed.operations =
        [Lifecycle.Stage1.childOp "stage1.prior_state_hash"
          (Lifecycle.Stage1.priorChild relation program
            (AssemblerInputs.interface relation program))
          (AssemblerInputs.priorOffset program),
        Lifecycle.Stage1.childOp "stage1.output_hash"
          (Lifecycle.Stage1.outputHashChild relation program
            (AssemblerInputs.interface relation program))
          (AssemblerInputs.outputHashOffset program),
        Lifecycle.Stage1.childOp "stage1.piccs.v1_1"
          (Lifecycle.Stage1.piCcsChild relation ajtai program
            (AssemblerInputs.interface relation program) template)
          (AssemblerInputs.piCcsOffset program),
        Lifecycle.Stage1.childOp "stage1.pirlc.v1_1"
          (Lifecycle.Stage1.piRlcChild relation ajtai program
            (AssemblerInputs.interface relation program))
          (AssemblerInputs.piRlcOffset program),
        Lifecycle.Stage1.childOp "stage1.pidec.v1_1"
          (Lifecycle.Stage1.piDecChild relation ajtai program
            (AssemblerInputs.interface relation program))
          (AssemblerInputs.piDecOffset program)] ∧
      AssemblerInputs.rootOffset program +
          localLength completed.operations =
        AssemblerInputs.runningOffset program ∧
      Lifecycle.Stage1.RunningTransition.SpecHolds
        (AssemblerInputs.runningInterface relation program)
        (AssemblerInputs.runningOffset program) completed.current := by
  rcases AssemblerPiRLCCompleteness.completePiRlcPrefix relation ajtai program
      template env specification with
    ⟨p4, p4Operations, p4End, piRlcAttemptEq⟩
  have piRlcOutputEq := congrArg (fun attempt => attempt.output) piRlcAttemptEq
  have parentEq :
      (PiDEC.v1_1.Semantics.inputAttempt relation
        (AssemblerInputs.piDecInterface relation program)
        (AssemblerInputs.piDecOffset program) env).parent =
      (PiDEC.v1_1.Semantics.inputAttempt relation
        (AssemblerInputs.piDecInterface relation program)
        (AssemblerInputs.piDecOffset program) p4.current).parent := by
    calc
      _ = PiRLC.v1_1.Semantics.evalOutput relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) env :=
        AssemblerInputs.piDecParent_eval_eq_piRlcOutput relation program env
      _ = PiRLC.v1_1.Semantics.evalOutput relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) p4.current := piRlcOutputEq
      _ = _ :=
        (AssemblerInputs.piDecParent_eval_eq_piRlcOutput relation program
          p4.current).symm
  have outputEq := piDecOutput_eq relation program p4 piRlcOutputEq
  have piDecInitial : PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piDecInterface relation program)
      (AssemblerInputs.piDecOffset program) env := by
    have phase := specification.piDec
    change PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piDecInterface relation program)
      (Lifecycle.Stage1.piDecOffset relation ajtai program
        (AssemblerInputs.interface relation program) template
        (AssemblerInputs.rootOffset program)) env at phase
    rw [AssemblerInputs.parent_piDecOffset_eq relation ajtai program template]
      at phase
    exact phase
  have phase := PiDEC.v1_1.Semantics.phaseHolds_of_parent_output_eq relation
    ajtai (AssemblerInputs.piDecInterface relation program)
    (AssemblerInputs.piDecOffset program) env p4.current parentEq outputEq
    piDecInitial
  have assumptions := AssemblerBounds.piDecAssumptions relation program p4.current
  rcases PiDEC.v1_1.Formal.completePrefix relation ajtai
      (AssemblerInputs.piDecInterface relation program) p4.current
      (AssemblerInputs.piDecOffset program) assumptions phase with
    ⟨built, builtOperations⟩
  let child := Lifecycle.Stage1.piDecChild relation ajtai program
    (AssemblerInputs.interface relation program)
  have childMain : child.main = PiDEC.v1_1.Formal.main relation
      (AssemblerInputs.piDecInterface relation program) := by
    rfl
  have childOperations : built.operations = Circuit.ops child.main
      (AssemblerInputs.piDecOffset program) := by
    rw [childMain, PiDEC.v1_1.Formal.main_ops]
    exact builtOperations
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main (AssemblerInputs.piDecOffset program)),
      expression.VarsBelow
        (AssemblerInputs.piDecOffset program + localLength
          (Circuit.ops child.main (AssemblerInputs.piDecOffset program))) := by
    rw [← childOperations]
    exact built.scope
  have childAgrees : AgreesOutside p4.current built.current
      (AssemblerInputs.piDecOffset program)
      (localLength
        (Circuit.ops child.main (AssemblerInputs.piDecOffset program))) := by
    rw [← childOperations]
    exact built.agrees
  have childRows : holdsFlat built.current
      (Circuit.ops child.main (AssemblerInputs.piDecOffset program)) := by
    rw [← childOperations]
    exact built.rows
  rcases Sequence.appendBuiltAt p4 "stage1.pidec.v1_1" child
      (AssemblerInputs.piDecOffset program) p4End childScope built.current
      childAgrees childRows with
    ⟨p5, p5Operations, p5End, p4to5, _piDecRows⟩
  have belowP4P5 : ∀ index, index < AssemblerInputs.piDecOffset program →
      p4.current index = p5.current index := by
    intro index below
    symm
    apply p4to5.values index
    rw [p4End]
    exact below
  have piDecOutputP4P5 := PiDEC.v1_1.Semantics.output_eq_of_agree relation
    (AssemblerInputs.piDecInterface relation program)
    (AssemblerInputs.piDecOffset program) p4.current p5.current
    (AssemblerBounds.piDecAssumptions relation program p4.current) belowP4P5
  have piDecOutputEnvP5 := outputEq.trans piDecOutputP4P5
  have recursiveValueEq := recursiveRunning_eval_eq_of_piDecOutput_eq
    relation program env p5.current piDecOutputEnvP5
  have sourceAgrees : ∀ index, index < RunningTransitionInputs.phaseOffset →
      env index = p5.current index := by
    intro index below
    apply sourceColumn_agrees program p5
    rw [Spartan.sourceColumnCount_eq]
    norm_num [RunningTransitionInputs.phaseOffset] at below ⊢
    omega
  have sourceBounds := RunningTransitionInputs.assumptions logicalWidth
    publicFits relation env
  have iterationEq : Lifecycle.Stage1.RunningTransition.iterationValue
      (AssemblerInputs.runningInterface relation program)
      (AssemblerInputs.runningOffset program) env =
    Lifecycle.Stage1.RunningTransition.iterationValue
      (AssemblerInputs.runningInterface relation program)
      (AssemblerInputs.runningOffset program) p5.current := by
    change RunningTransitionInputs.iterationExpr.eval env =
      RunningTransitionInputs.iterationExpr.eval p5.current
    exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
      env p5.current sourceBounds.iteration sourceAgrees
  have initialStateEq : ∀ index,
      ((AssemblerInputs.runningInterface relation program).initialState
        (AssemblerInputs.runningOffset program) index).eval env =
      ((AssemblerInputs.runningInterface relation program).initialState
        (AssemblerInputs.runningOffset program) index).eval p5.current := by
    intro index
    change (RunningTransitionInputs.initialStateExpr index).eval env =
      (RunningTransitionInputs.initialStateExpr index).eval p5.current
    exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
      env p5.current (sourceBounds.initialState index) sourceAgrees
  have currentStateEq : ∀ index,
      ((AssemblerInputs.runningInterface relation program).currentState
        (AssemblerInputs.runningOffset program) index).eval env =
      ((AssemblerInputs.runningInterface relation program).currentState
        (AssemblerInputs.runningOffset program) index).eval p5.current := by
    intro index
    change (RunningTransitionInputs.currentStateExpr index).eval env =
      (RunningTransitionInputs.currentStateExpr index).eval p5.current
    exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
      env p5.current (sourceBounds.currentState index) sourceAgrees
  have recursiveWordEq : ∀ index,
      (Lifecycle.Stage1.RunningTransition.runningWord
        ((AssemblerInputs.runningInterface relation program).recursive
          (AssemblerInputs.runningOffset program)) index).eval env =
      (Lifecycle.Stage1.RunningTransition.runningWord
        ((AssemblerInputs.runningInterface relation program).recursive
          (AssemblerInputs.runningOffset program)) index).eval p5.current := by
    intro index
    change (Lifecycle.Stage1.RunningTransition.runningWord
        (AssemblerInputs.recursiveRunningExpr relation program) index).eval
          env =
      (Lifecycle.Stage1.RunningTransition.runningWord
        (AssemblerInputs.recursiveRunningExpr relation program) index).eval
          p5.current
    rw [Lifecycle.Stage1.RunningTransition.runningWord_eval,
      Lifecycle.Stage1.RunningTransition.runningWord_eval, recursiveValueEq]
  have outputWordEq : ∀ index,
      (Lifecycle.Stage1.RunningTransition.runningWord
        ((AssemblerInputs.runningInterface relation program).output
          (AssemblerInputs.runningOffset program)) index).eval env =
      (Lifecycle.Stage1.RunningTransition.runningWord
        ((AssemblerInputs.runningInterface relation program).output
          (AssemblerInputs.runningOffset program)) index).eval p5.current := by
    intro index
    change (Lifecycle.Stage1.RunningTransition.runningWord
        (RunningTransitionInputs.outputRunningExpr logicalWidth publicFits)
        index).eval env =
      (Lifecycle.Stage1.RunningTransition.runningWord
        (RunningTransitionInputs.outputRunningExpr logicalWidth publicFits)
        index).eval p5.current
    exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
      env p5.current
      (Lifecycle.Stage1.RunningTransition.runningWord_varsBelow _
        RunningTransitionInputs.phaseOffset
        (RunningTransitionInputs.outputRunningBelow logicalWidth publicFits)
        index)
      sourceAgrees
  have runningInitial := specification.running
  change Lifecycle.Stage1.RunningTransition.SpecHolds
    (AssemblerInputs.runningInterface relation program)
    (Lifecycle.Stage1.runningOffset relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program)) env at runningInitial
  rw [AssemblerInputs.parent_runningOffset_eq relation ajtai program template]
    at runningInitial
  have runningCurrent :=
    Lifecycle.Stage1.RunningTransition.specHolds_of_values_eq
      (AssemblerInputs.runningInterface relation program)
      (AssemblerInputs.runningOffset program) env p5.current iterationEq
      initialStateEq currentStateEq recursiveWordEq outputWordEq runningInitial
  refine ⟨p5, ?_, ?_, runningCurrent⟩
  · rw [p5Operations, p4Operations]
    rfl
  · calc
      _ = AssemblerInputs.piDecOffset program + localLength
          (Circuit.ops child.main (AssemblerInputs.piDecOffset program)) :=
        p5End
      _ = Lifecycle.Stage1.runningOffset relation ajtai program
          (AssemblerInputs.interface relation program) template
          (AssemblerInputs.rootOffset program) := by
        unfold Lifecycle.Stage1.runningOffset
        rw [AssemblerInputs.parent_piDecOffset_eq relation ajtai program template]
        rw [child.privateCount_eq]
      _ = AssemblerInputs.runningOffset program :=
        AssemblerInputs.parent_runningOffset_eq relation ajtai program template

end NightstreamFPrime.Layout.Stage1.AssemblerPiDECCompleteness
