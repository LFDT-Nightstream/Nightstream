import NightstreamFPrime.Layout.Stage1.AssemblerRunningCompleteness

/-!
Owns honest completion of the application and next-preimage Stage 1 children.
The selected application is fixed by the Lean `Program`; neither the prover nor this
assembler selects a circuit at runtime.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerApplicationCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Honest completion of the exact eight-child Stage 1 operation list. -/
theorem completeStage1
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
      completed.operations = Circuit.ops
        (Lifecycle.Stage1.main relation ajtai program
          (AssemblerInputs.interface relation program) template)
        (AssemblerInputs.rootOffset program) ∧
      AssemblerInputs.rootOffset program +
          localLength completed.operations =
        Lifecycle.Stage1.finalOffset relation ajtai program
          (AssemblerInputs.interface relation program) template
          (AssemblerInputs.rootOffset program) := by
  rcases AssemblerRunningCompleteness.completeRunningPrefix relation ajtai
      program template env specification with
    ⟨p6, p6Operations, p6End⟩
  have rootAgrees : ∀ index, index < AssemblerInputs.rootOffset program →
      env index = p6.current index := by
    intro index below
    exact (p6.agrees index (Or.inl below)).symm
  have inputsRoot := AssemblerBounds.applicationInputsBelowRoot
    (logicalWidth := logicalWidth) program
  let appInterface := AssemblerInputs.applicationInterface program
  let appOffset := AssemblerInputs.applicationOffset program
  have inputEq : Lifecycle.Stage1.Application.inputState appInterface appOffset
      env = Lifecycle.Stage1.Application.inputState appInterface appOffset
        p6.current := by
    unfold Lifecycle.Stage1.Application.inputState
    apply congrArg List.ofFn
    funext index
    exact Expr.eval_eq_of_agree_below _ (AssemblerInputs.rootOffset program)
      env p6.current (by simpa [appInterface, appOffset] using
        (inputsRoot.input index)) rootAgrees
  have witnessEq : Lifecycle.Stage1.Application.witnessValue appInterface
      appOffset env = Lifecycle.Stage1.Application.witnessValue appInterface
        appOffset p6.current := by
    unfold Lifecycle.Stage1.Application.witnessValue
    apply congrArg List.ofFn
    funext index
    exact Expr.eval_eq_of_agree_below _ (AssemblerInputs.rootOffset program)
      env p6.current (by simpa [appInterface, appOffset] using
        (inputsRoot.witness index)) rootAgrees
  have outputEq : Lifecycle.Stage1.Application.outputState appInterface
      appOffset env = Lifecycle.Stage1.Application.outputState appInterface
        appOffset p6.current := by
    unfold Lifecycle.Stage1.Application.outputState
    apply congrArg List.ofFn
    funext index
    exact Expr.eval_eq_of_agree_below _ (AssemblerInputs.rootOffset program)
      env p6.current (by simpa [appInterface, appOffset] using
        (inputsRoot.output index)) rootAgrees
  have applicationInitialSpec := specification.application
  change (program.circuit appInterface).spec
    (Lifecycle.Stage1.applicationOffset relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program)) env at applicationInitialSpec
  rw [AssemblerInputs.parent_applicationOffset_eq relation ajtai program
    template] at applicationInitialSpec
  have applicationInitial : Lifecycle.Stage1.Application.Holds program.step
      appInterface appOffset env :=
    (program.spec_iff appInterface appOffset env).mp applicationInitialSpec
  have applicationCurrent := program.holds_of_values_eq appInterface appOffset
    env p6.current inputEq witnessEq outputEq applicationInitial
  let child := Lifecycle.Stage1.applicationChild relation program
    (AssemblerInputs.interface relation program)
  have assumptions := AssemblerBounds.applicationAssumptions
    (logicalWidth := logicalWidth) program p6.current
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main (AssemblerInputs.applicationOffset program)),
      expression.VarsBelow
        (AssemblerInputs.applicationOffset program + localLength
          (Circuit.ops child.main
            (AssemblerInputs.applicationOffset program))) := by
    change ∀ expression ∈ flatConstraints
        (Circuit.ops (program.circuit appInterface).main appOffset),
      expression.VarsBelow
        (appOffset + localLength
          (Circuit.ops (program.circuit appInterface).main appOffset))
    exact program.scope appInterface appOffset p6.current
      (AssemblerBounds.applicationInputsBelow
        (logicalWidth := logicalWidth) program)
      assumptions
  have applicationSpec : child.spec
      (AssemblerInputs.applicationOffset program) p6.current := by
    change (program.circuit appInterface).spec appOffset p6.current
    exact (program.spec_iff appInterface appOffset p6.current).mpr
      applicationCurrent
  rcases Sequence.appendAt p6 "stage1.application" child
      (AssemblerInputs.applicationOffset program) p6End childScope assumptions
      applicationSpec with
    ⟨p7, p7Operations, p7End, _p6to7, _applicationRows⟩
  let finalOffset := Lifecycle.Stage1.finalOffset relation ajtai program
    (AssemblerInputs.interface relation program) template
    (AssemblerInputs.rootOffset program)
  have p7EndFinal : AssemblerInputs.rootOffset program +
      localLength p7.operations = finalOffset := by
    calc
      _ = AssemblerInputs.applicationOffset program + localLength
          (Circuit.ops child.main
            (AssemblerInputs.applicationOffset program)) := p7End
      _ = finalOffset := by
        unfold finalOffset Lifecycle.Stage1.finalOffset
        rw [AssemblerInputs.parent_applicationOffset_eq relation ajtai program
          template]
        rw [child.privateCount_eq]
  let nextChild := Lifecycle.Stage1.nextPreimageChild relation program
    (AssemblerInputs.interface relation program)
  have nextAssumptions := AssemblerBounds.nextPreimageAssumptions relation
    ajtai program template p7.current
  have sourceBounds := NextPreimageInputs.sourceAssumptions env
  have sourceAgrees : ∀ index,
      index < RunningTransitionInputs.phaseOffset →
      env index = p7.current index := by
    intro index below
    exact (p7.agrees index (Or.inl
      (lt_of_lt_of_le below
        (AssemblerBounds.nextPreimageSourceOffset_le_root program)))).symm
  have nextSpec : nextChild.spec finalOffset p7.current := by
    apply Lifecycle.Stage1.NextPreimage.SpecHolds.of_cross_values_eq
      NextPreimageInputs.sourceInterface NextPreimageInputs.sourceInterface
      finalOffset finalOffset env p7.current
    · exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
        env p7.current sourceBounds.priorIteration sourceAgrees
    · exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
        env p7.current sourceBounds.outputIteration sourceAgrees
    · intro index
      exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
        env p7.current (sourceBounds.priorInitialState index) sourceAgrees
    · intro index
      exact Expr.eval_eq_of_agree_below _ RunningTransitionInputs.phaseOffset
        env p7.current (sourceBounds.outputInitialState index) sourceAgrees
    · exact specification.nextPreimage
  have nextScope : ∀ expression ∈ flatConstraints
      (Circuit.ops nextChild.main finalOffset),
      expression.VarsBelow
        (finalOffset + localLength
          (Circuit.ops nextChild.main finalOffset)) := by
    intro expression member
    apply Expr.VarsBelow.mono _
      (Lifecycle.Stage1.NextPreimage.flatConstraints_varsBelow
        NextPreimageInputs.sourceInterface finalOffset p7.current
        nextAssumptions expression member)
    omega
  rcases Sequence.appendAt p7 "stage1.next_preimage" nextChild finalOffset
      p7EndFinal nextScope nextAssumptions nextSpec with
    ⟨p8, p8Operations, p8End, _p7to8, _nextRows⟩
  refine ⟨p8, ?_, ?_⟩
  · have operationsEq : p8.operations = Lifecycle.Stage1.opsAt relation ajtai
        program (AssemblerInputs.interface relation program) template
        (AssemblerInputs.rootOffset program) := by
      rw [p8Operations, p7Operations, p6Operations]
      unfold Lifecycle.Stage1.opsAt
      rw [AssemblerInputs.parent_priorOffset_eq relation program,
        AssemblerInputs.parent_outputHashOffset_eq relation program,
        AssemblerInputs.parent_piCcsOffset_eq relation program,
        AssemblerInputs.parent_piRlcOffset_eq relation ajtai program template,
        AssemblerInputs.parent_piDecOffset_eq relation ajtai program template,
        AssemblerInputs.parent_runningOffset_eq relation ajtai program template,
        AssemblerInputs.parent_applicationOffset_eq relation ajtai program
          template]
      simp [child, nextChild, finalOffset, Lifecycle.Stage1.childOp]
    exact operationsEq.trans
      (Lifecycle.Stage1.main_ops relation ajtai program
        (AssemblerInputs.interface relation program) template
        (AssemblerInputs.rootOffset program)).symm
  · rw [p8End]
    change finalOffset + localLength (Circuit.ops
      (Lifecycle.Stage1.NextPreimage.main NextPreimageInputs.sourceInterface)
      finalOffset) = finalOffset
    rw [Lifecycle.Stage1.NextPreimage.localLength_eq, Nat.add_zero]

/-- The canonical compact layout supplies the proof-only root completion used
by the sole logical Stage 1 circuit. -/
def rootCompleteness
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    Lifecycle.Stage1.RootCompleteness relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program) where
  complete := by
    intro env _assumptions specification
    exact completeStage1 relation ajtai program template env specification

end NightstreamFPrime.Layout.Stage1.AssemblerApplicationCompleteness
