import NightstreamFPrime.Layout.Stage1.AssemblerPiDECCompleteness

/-!
Owns the sixth opaque-child append for the compact Stage 1 assembler. The
running-transition child keeps its existing base/recursive semantics and
canonical witness builder. This file adds no branch rule or row.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerRunningCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Honest completion through the verifier-owned running-instance branch. -/
theorem completeRunningPrefix
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
          (AssemblerInputs.piDecOffset program),
        Lifecycle.Stage1.childOp "stage1.running_transition"
          (Lifecycle.Stage1.runningChild relation program
            (AssemblerInputs.interface relation program))
          (AssemblerInputs.runningOffset program)] ∧
      AssemblerInputs.rootOffset program +
          localLength completed.operations =
        AssemblerInputs.applicationOffset program := by
  rcases AssemblerPiDECCompleteness.completePiDecPrefix relation ajtai program
      template env specification with
    ⟨p5, p5Operations, p5End, runningSpec⟩
  let child := Lifecycle.Stage1.runningChild relation program
    (AssemblerInputs.interface relation program)
  have assumptions := AssemblerBounds.runningAssumptions relation program
    p5.current
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main (AssemblerInputs.runningOffset program)),
      expression.VarsBelow
        (AssemblerInputs.runningOffset program + localLength
          (Circuit.ops child.main (AssemblerInputs.runningOffset program))) := by
    change ∀ expression ∈ flatConstraints
        (Lifecycle.Stage1.RunningTransition.operations
          (AssemblerInputs.runningInterface relation program)
          (AssemblerInputs.runningOffset program)),
      expression.VarsBelow
        (AssemblerInputs.runningOffset program + localLength
          (Lifecycle.Stage1.RunningTransition.operations
            (AssemblerInputs.runningInterface relation program)
            (AssemblerInputs.runningOffset program)))
    rw [Lifecycle.Stage1.RunningTransition.localLength_eq]
    exact Lifecycle.Stage1.RunningTransition.flatConstraints_varsBelow
      (AssemblerInputs.runningInterface relation program)
      (AssemblerInputs.runningOffset program) p5.current assumptions
  rcases Sequence.appendAt p5 "stage1.running_transition" child
      (AssemblerInputs.runningOffset program) p5End childScope assumptions
      runningSpec with
    ⟨p6, p6Operations, p6End, _p5to6, _runningRows⟩
  refine ⟨p6, ?_, ?_⟩
  · rw [p6Operations, p5Operations]
    rfl
  · calc
      _ = AssemblerInputs.runningOffset program + localLength
          (Circuit.ops child.main (AssemblerInputs.runningOffset program)) :=
        p6End
      _ = Lifecycle.Stage1.applicationOffset relation ajtai program
          (AssemblerInputs.interface relation program) template
          (AssemblerInputs.rootOffset program) := by
        unfold Lifecycle.Stage1.applicationOffset
        rw [AssemblerInputs.parent_runningOffset_eq relation ajtai program
          template]
        rw [child.privateCount_eq]
      _ = AssemblerInputs.applicationOffset program :=
        AssemblerInputs.parent_applicationOffset_eq relation ajtai program
          template

end NightstreamFPrime.Layout.Stage1.AssemblerRunningCompleteness
