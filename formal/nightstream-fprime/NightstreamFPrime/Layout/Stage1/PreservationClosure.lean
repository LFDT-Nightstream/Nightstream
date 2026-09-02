import NightstreamFPrime.Layout.Stage1.AssemblerSoundness
import NightstreamFPrime.Layout.Stage1.Preservation

/-!
Closes physical preservation for the sole compact Stage 1 logical parent.

The phase-local preservation theorems remain in `Preservation`. This module
owns only the final running-transition relocation, seven-child parent
assembly, and composition with the exact HyperNova step relation. It does not
define rows, another circuit, or another semantic predicate.
-/

namespace NightstreamFPrime.Layout.Stage1.PreservationClosure

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem sourceEnv_eq_compactEnv_belowRunning
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    ∀ index, index < RunningTransitionInputs.phaseOffset →
      CompactPullback.sourceEnv program env index =
        CompactPullback.compactEnv program env index := by
  intro index bounded
  symm
  apply CompactPullback.compactEnv_source
  rw [Spartan.sourceColumnCount_eq]
  norm_num [RunningTransitionInputs.phaseOffset] at bounded ⊢
  omega

private theorem compactRunningOutput_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    PiCCS.v1_1.StatementAbsorption.evalRunning
        (RunningTransitionInputs.outputRunningExpr logicalWidth publicFits)
        (CompactPullback.sourceEnv program env) =
      PiCCS.v1_1.StatementAbsorption.evalRunning
        (RunningTransitionInputs.outputRunningExpr logicalWidth publicFits)
        (CompactPullback.compactEnv program env) := by
  let running := RunningTransitionInputs.outputRunningExpr logicalWidth publicFits
  let before := CompactPullback.sourceEnv program env
  let after := CompactPullback.compactEnv program env
  have below := RunningTransitionInputs.outputRunningBelow
    logicalWidth publicFits
  have agrees : ∀ index, index < RunningTransitionInputs.phaseOffset →
      before index = after index :=
    sourceEnv_eq_compactEnv_belowRunning program env
  unfold PiCCS.v1_1.StatementAbsorption.evalRunning
  congr 1
  · apply cubePoint_ext
    change List.ofFn (fun coordinate =>
        (running.point coordinate).eval before) =
      List.ofFn (fun coordinate =>
        (running.point coordinate).eval after)
    apply congrArg List.ofFn
    funext coordinate
    exact (running.point coordinate).eval_eq_of_agree_below
      RunningTransitionInputs.phaseOffset before after
      (below.point coordinate) agrees
  · funext source row coefficient
    exact (running.commitment source row coefficient).eval_eq_of_agree_below
      RunningTransitionInputs.phaseOffset before after
      (below.commitment source row coefficient) agrees
  · funext source column
    exact (running.publicInput source column).eval_eq_of_agree_below
      RunningTransitionInputs.phaseOffset before after
      (below.publicInput source column) agrees
  · funext source
    unfold PiCCS.v1_1.StatementAbsorption.evalEvaluation
    congr 1
    · funext coefficient
      exact ((running.evaluation source).eval_K coefficient
        ).eval_eq_of_agree_below RunningTransitionInputs.phaseOffset
          before after (below.eval_K source coefficient) agrees
    · funext matrix coefficient
      exact ((running.evaluation source).eval_A matrix coefficient
        ).eval_eq_of_agree_below RunningTransitionInputs.phaseOffset
          before after (below.eval_A source matrix coefficient) agrees

private theorem compactRecursiveRunning_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    PiCCS.v1_1.StatementAbsorption.evalRunning
        (RunningTransitionInputs.recursiveRunningExpr logicalWidth publicFits)
        (CompactPullback.sourceEnv program env) =
      PiCCS.v1_1.StatementAbsorption.evalRunning
        (AssemblerInputs.recursiveRunningExpr relation program)
        (CompactPullback.compactEnv program env) := by
  have outputsEq := Preservation.compactPiDecOutput_eq relation program env
  apply Lifecycle.PiCCS.v1_1.Formal.PhaseTransport.running_ext
  · let child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex :=
      ⟨0, by decide⟩
    have childEq := congrFun outputsEq child
    have pointEq := congrArg (fun value => value.point) childEq
    simpa [PiCCS.v1_1.StatementAbsorption.evalRunning,
      RunningTransitionInputs.recursiveRunningExpr,
      AssemblerInputs.recursiveRunningExpr,
      PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface,
      PiDEC.v1_1.Formal.atOffset] using pointEq
  · funext source row coefficient
    have childEq := congrFun outputsEq
      (RunningTransitionInputs.childOfRunning source)
    have commitmentEq := congrArg (fun value => value.commitment) childEq
    have coordinateEq := congrFun (congrFun commitmentEq row) coefficient
    simpa [PiCCS.v1_1.StatementAbsorption.evalRunning,
      RunningTransitionInputs.recursiveRunningExpr,
      AssemblerInputs.recursiveRunningExpr,
      PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface,
      PiDEC.v1_1.Formal.atOffset,
      AssemblerInputs.childOfRunning] using coordinateEq
  · funext source column
    have childEq := congrFun outputsEq
      (RunningTransitionInputs.childOfRunning source)
    have publicEq := congrArg (fun value => value.publicInput) childEq
    have coordinateEq := congrFun publicEq
      (RunningTransitionInputs.digitCoordinate column)
    simpa [PiCCS.v1_1.StatementAbsorption.evalRunning,
      RunningTransitionInputs.recursiveRunningExpr,
      AssemblerInputs.recursiveRunningExpr,
      PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface,
      PiDEC.v1_1.Formal.atOffset,
      AssemblerInputs.childOfRunning,
      AssemblerInputs.digitCoordinate] using coordinateEq
  · funext source
    have childEq := congrFun outputsEq
      (RunningTransitionInputs.childOfRunning source)
    have evaluationEq := congrArg
      (fun value => value.evaluations.getD 0 PaperAlgebra.evaluationZero)
      childEq
    simpa [PiCCS.v1_1.StatementAbsorption.evalRunning,
      RunningTransitionInputs.recursiveRunningExpr,
      AssemblerInputs.recursiveRunningExpr,
      PiDEC.v1_1.Semantics.output, PiDEC.v1_1.OutputBinding.evalOutput,
      PiDEC.v1_1.Formal.outputBindingInterface,
      PiDEC.v1_1.Formal.atOffset,
      AssemblerInputs.childOfRunning] using evaluationEq

/-- The source running-transition specification relocates to the exact
running field of the compact Stage 1 parent. -/
theorem physical_implies_compactRunning
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) (physical : Preservation.PhysicalHolds relation program env) :
    Lifecycle.Stage1.RunningTransition.SpecHolds
      (AssemblerInputs.runningInterface relation program)
      (AssemblerInputs.runningOffset program)
      (CompactPullback.compactEnv program env) := by
  have children := Preservation.physical_implies_childSpecs relation ajtai
    program template env physical
  let before := CompactPullback.sourceEnv program env
  let after := CompactPullback.compactEnv program env
  have assumptions := RunningTransitionInputs.assumptions
    logicalWidth publicFits relation before
  have agrees : ∀ index, index < RunningTransitionInputs.phaseOffset →
      before index = after index :=
    sourceEnv_eq_compactEnv_belowRunning program env
  apply Lifecycle.Stage1.RunningTransition.specHolds_of_cross_values_eq
    (RunningTransitionInputs.interface logicalWidth publicFits)
    (AssemblerInputs.runningInterface relation program)
    RunningTransitionInputs.phaseOffset (AssemblerInputs.runningOffset program)
    before after
  · exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
      before after assumptions.iteration agrees
  · intro index
    exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
      before after (assumptions.initialState index) agrees
  · intro index
    exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
      before after (assumptions.currentState index) agrees
  · exact compactRecursiveRunning_eq relation program env
  · exact compactRunningOutput_eq program env
  · exact children.running

/-- Every physical Stage 1 row family implies the sole compact seven-child
logical parent specification. -/
theorem physical_implies_compactSpec
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) (physical : Preservation.PhysicalHolds relation program env) :
    Lifecycle.Stage1.SpecHolds relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program)
      (CompactPullback.compactEnv program env) := by
  have children := Preservation.physical_implies_childSpecs relation ajtai
    program template env physical
  refine {
    prior := ?_
    outputHash := ?_
    piCcs := ?_
    piRlc := ?_
    piDec := ?_
    running := ?_
    application := ?_ }
  · rw [AssemblerInputs.parent_priorOffset_eq relation program]
    exact Preservation.compactPilotPrior program env children.pilot.1
  · rw [AssemblerInputs.parent_outputHashOffset_eq relation program]
    exact Preservation.compactPilotOutput program env children.pilot.2
  · rw [AssemblerInputs.parent_piCcsOffset_eq relation program]
    exact Preservation.compactPiCcsPhaseHolds relation ajtai program template
      env children.piCcs
  · rw [AssemblerInputs.parent_piRlcOffset_eq relation ajtai program template]
    exact Preservation.compactPiRlcPhaseHolds relation ajtai program env
      children.piRlc
  · rw [AssemblerInputs.parent_piDecOffset_eq relation ajtai program template]
    exact Preservation.compactPiDecPhaseHolds relation ajtai program env
      children.piDec
  · rw [AssemblerInputs.parent_runningOffset_eq relation ajtai program template]
    exact physical_implies_compactRunning relation ajtai program template env
      physical
  · rw [AssemblerInputs.parent_applicationOffset_eq relation ajtai program
      template]
    exact Preservation.physical_implies_compactApplication relation ajtai
      program template env physical

/-- Physical Stage 1 rows, together with the canonical representation
boundary, imply the exact fixed HyperNova step relation. -/
theorem physical_implies_stepHoldsFor
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (physical : Preservation.PhysicalHolds relation program env)
    (represents : AssemblerSoundness.Represents relation ajtai vk program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program)
      (CompactPullback.compactEnv program env) input output) :
    StepHoldsFor relation ajtai vk program input output := by
  exact AssemblerSoundness.compactSpec_implies_stepHoldsFor relation ajtai vk
    program template (CompactPullback.compactEnv program env) input output
    (physical_implies_compactSpec relation ajtai program template env physical)
    represents

end NightstreamFPrime.Layout.Stage1.PreservationClosure
