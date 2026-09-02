import NightstreamFPrime.Layout.Stage1.AssemblerRunningCompleteness

/-!
Owns honest completion of the seventh and final Stage 1 child. The selected
application is fixed by the Lean `Program`; neither the prover nor this
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

/-- Honest completion of the exact seven-child Stage 1 operation list. -/
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
  refine ⟨p7, ?_, ?_⟩
  · have operationsEq : p7.operations = Lifecycle.Stage1.opsAt relation ajtai
        program (AssemblerInputs.interface relation program) template
        (AssemblerInputs.rootOffset program) := by
      rw [p7Operations, p6Operations]
      unfold Lifecycle.Stage1.opsAt
      rw [AssemblerInputs.parent_priorOffset_eq relation program,
        AssemblerInputs.parent_outputHashOffset_eq relation program,
        AssemblerInputs.parent_piCcsOffset_eq relation program,
        AssemblerInputs.parent_piRlcOffset_eq relation ajtai program template,
        AssemblerInputs.parent_piDecOffset_eq relation ajtai program template,
        AssemblerInputs.parent_runningOffset_eq relation ajtai program template,
        AssemblerInputs.parent_applicationOffset_eq relation ajtai program
          template]
      simp [child, Lifecycle.Stage1.childOp]
    exact operationsEq.trans
      (Lifecycle.Stage1.main_ops relation ajtai program
        (AssemblerInputs.interface relation program) template
        (AssemblerInputs.rootOffset program)).symm
  · calc
      _ = AssemblerInputs.applicationOffset program + localLength
          (Circuit.ops child.main
            (AssemblerInputs.applicationOffset program)) := p7End
      _ = Lifecycle.Stage1.finalOffset relation ajtai program
          (AssemblerInputs.interface relation program) template
          (AssemblerInputs.rootOffset program) := by
        unfold Lifecycle.Stage1.finalOffset
        rw [AssemblerInputs.parent_applicationOffset_eq relation ajtai program
          template]
        rw [child.privateCount_eq]

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
