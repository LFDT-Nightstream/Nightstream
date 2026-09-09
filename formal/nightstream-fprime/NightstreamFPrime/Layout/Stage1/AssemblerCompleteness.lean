import NightstreamFPrime.Layout.Stage1.AssemblerBounds
import NightstreamFPrime.Layout.Stage1.AssemblerPilotBounds
import NightstreamFPrime.Layout.Stage1.PiCCSInputSupport
import NightstreamFPrime.Layout.Stage1.PiCCSTranscriptSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBindingSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.PhaseTransport

/-!
Owns ordered witness composition for the compact eight-child Stage 1 logical
assembler. It uses only opaque child completeness and exact interface wiring.
It adds no row, verifier predicate, or physical placement.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def piCcsExternalSupport
    (program : Lifecycle.Stage1.Application.Program) :
    PiCCS.v1_1.Formal.ExternalInputsSupported
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      PiCCSOrdinarySourceSupport.External := by
  let source := PiCCSOrdinarySourceSupport.externalInputsSupported
    logicalWidth publicFits
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
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.priorStateFixed word member
  · intro word member
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.outputStateFixed word member
  · intro lane
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.priorStateContext lane
  · intro lane
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.outputStateContext lane
  · intro lane
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.expectedContext lane
  · intro coordinate
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.runningPoint coordinate
  · intro sourceIndex row coefficient
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.runningCommitment sourceIndex row coefficient
  · intro sourceIndex column
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.runningPublicInput sourceIndex column
  · intro sourceIndex coefficient
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.runningEval_K sourceIndex coefficient
  · intro sourceIndex matrix coefficient
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.runningEval_A sourceIndex matrix coefficient
  · intro sourceIndex row coefficient
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.freshCommitment sourceIndex row coefficient
  · intro sourceIndex column
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.freshPublicInput sourceIndex column
  · intro roundIndex coefficient
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.roundCoefficient roundIndex coefficient
  · intro sourceIndex coefficient
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.outputEval_K sourceIndex coefficient
  · intro sourceIndex matrix coefficient
    simpa [AssemblerInputs.piCcsInterface, PiCCSInputs.interface] using
      source.outputEval_A sourceIndex matrix coefficient

private def PiCcsSupport
    (program : Lifecycle.Stage1.Application.Program) (index : Nat) : Prop :=
  PiCCSOrdinarySourceSupport.External index ∨
    AssemblerInputs.piCcsOffset program ≤ index

private theorem piCcsSupport_agrees
    (program : Lifecycle.Stage1.Application.Program) {env : Env}
    (completed : Sequence.Prefix env (AssemblerInputs.rootOffset program))
    (endEq : AssemblerInputs.rootOffset program +
        localLength completed.operations = AssemblerInputs.piCcsOffset program) :
    ∀ index, PiCcsSupport program index →
      env index = completed.current index := by
  intro index support
  symm
  apply completed.agrees index
  rcases support with external | generated
  · apply Or.inl
    have sourceBound :=
      PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
        (PiCCSOrdinarySourceSupport.external_source index external)
    rw [Spartan.sourceColumnCount_eq] at sourceBound
    rw [AssemblerPilotBounds.rootOffset_eq]
    omega
  · apply Or.inr
    rw [endEq]
    exact generated

private theorem roundFinalState_supported
    (program : Lifecycle.Stage1.Application.Program) :
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.StateSupported
      (PiCCS.v1_1.Formal.roundTranscriptFinalState
        (PiCCS.v1_1.Formal.atOffset
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program))
        (AssemblerInputs.piCcsOffset program))
      (PiCcsSupport program) := by
  have precise :=
    PiCCSOrdinarySourceSupport.roundFinalState_outputPrefix_supported
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
  apply precise.mono
  intro index support
  rcases support with external | ⟨invocation, lane, source⟩
  · exact Or.inl external
  · subst index
    exact Or.inr (by omega)

private theorem piCcsOffset_le_outputBindingOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    AssemblerInputs.piCcsOffset program ≤
      PiCCS.v1_1.Formal.outputBindingOffset relation
        (AssemblerInputs.piCcsInterface program)
        (AssemblerInputs.piCcsOffset program) := by
  unfold PiCCS.v1_1.Formal.outputBindingOffset
    PiCCS.v1_1.Formal.finalIdentityOffset PiCCS.v1_1.Formal.normOffset
    PiCCS.v1_1.Formal.ccsOffset PiCCS.v1_1.Formal.evalAOffset
    PiCCS.v1_1.Formal.evalKOffset PiCCS.v1_1.Formal.sumcheckOffset
    PiCCS.v1_1.Formal.initialClaimOffset
    PiCCS.v1_1.Formal.roundTranscriptOffset
    PiCCS.v1_1.Formal.challengeOffset
    PiCCS.v1_1.Formal.statementAbsorptionOffset
    PiCCS.v1_1.Formal.nextOffset
  omega

private theorem piCcsOutputState_supported
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.StateSupported
      (AssemblerInputs.piCcsOutputState relation program)
      (PiCcsSupport program) := by
  let interface := AssemblerInputs.piCcsInterface
    (logicalWidth := logicalWidth) (publicFits := publicFits) program
  let outputOffset := PiCCS.v1_1.Formal.outputBindingOffset relation interface
    (AssemblerInputs.piCcsOffset program)
  have support := PiCCS.v1_1.OutputBinding.finalState_supported_from_offset
    (PiCCS.v1_1.Formal.outputBindingInterface
      (PiCCS.v1_1.Formal.atOffset interface
        (AssemblerInputs.piCcsOffset program))) outputOffset
  change NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.StateSupported
    (PiCCS.v1_1.OutputBinding.finalState
      (PiCCS.v1_1.Formal.outputBindingInterface
        (PiCCS.v1_1.Formal.atOffset interface
          (AssemblerInputs.piCcsOffset program))) outputOffset)
    (PiCcsSupport program)
  apply support.mono
  intro index generated
  exact Or.inr (Nat.le_trans
    (piCcsOffset_le_outputBindingOffset relation program) generated)

private theorem piCcsRoundPoint_eval_eq
    (program : Lifecycle.Stage1.Application.Program) {env : Env}
    (completed : Sequence.Prefix env (AssemblerInputs.rootOffset program))
    (endEq : AssemblerInputs.rootOffset program +
        localLength completed.operations = AssemblerInputs.piCcsOffset program) :
    PiCCS.v1_1.RoundTranscript.evalRoundPoint
        (PiCCS.v1_1.Formal.roundTranscriptInterface
          (PiCCS.v1_1.Formal.atOffset
            (AssemblerInputs.piCcsInterface
              (logicalWidth := logicalWidth) (publicFits := publicFits) program)
            (AssemblerInputs.piCcsOffset program)))
        (PiCCS.v1_1.Formal.roundTranscriptOffset
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program)) env =
      PiCCS.v1_1.RoundTranscript.evalRoundPoint
        (PiCCS.v1_1.Formal.roundTranscriptInterface
          (PiCCS.v1_1.Formal.atOffset
            (AssemblerInputs.piCcsInterface
              (logicalWidth := logicalWidth) (publicFits := publicFits) program)
            (AssemblerInputs.piCcsOffset program)))
        (PiCCS.v1_1.Formal.roundTranscriptOffset
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program)) completed.current := by
  apply PiCCSOrdinarySourceSupport.evalRoundPoint_eq_of_agree_outputPrefix
    (AssemblerInputs.piCcsInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits) program)
    (AssemblerInputs.piCcsOffset program) env completed.current
  intro index selected
  apply piCcsSupport_agrees program completed endEq index
  rcases selected with external | ⟨invocation, lane, source⟩
  · exact Or.inl external
  · subst index
    exact Or.inr (by omega)

private theorem piCcsOutputState_eval_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) {env : Env}
    (completed : Sequence.Prefix env (AssemblerInputs.rootOffset program))
    (endEq : AssemblerInputs.rootOffset program +
        localLength completed.operations = AssemblerInputs.piCcsOffset program) :
    PiCCS.v1_1.StatementAbsorption.evalState env
        (AssemblerInputs.piCcsOutputState relation program) =
      PiCCS.v1_1.StatementAbsorption.evalState completed.current
        (AssemblerInputs.piCcsOutputState relation program) := by
  unfold PiCCS.v1_1.StatementAbsorption.evalState
    NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState
  apply congrArg List.ofFn
  funext lane
  exact Expr.eval_eq_of_agree_satisfy
    (AssemblerInputs.piCcsOutputState relation program lane)
    (PiCcsSupport program) env completed.current
      (piCcsOutputState_supported relation program lane)
      (piCcsSupport_agrees program completed endEq)

private theorem piCcsPhase_after_pilot
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program) env)
    (completed : Sequence.Prefix env (AssemblerInputs.rootOffset program))
    (endEq : AssemblerInputs.rootOffset program +
        localLength completed.operations = AssemblerInputs.piCcsOffset program) :
    PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface program)
      (AssemblerInputs.piCcsOffset program) completed.current template := by
  have phase := specification.piCcs
  change PiCCS.v1_1.Formal.PhaseHolds relation ajtai
    (AssemblerInputs.piCcsInterface program)
    (Lifecycle.Stage1.piCcsOffset relation program
      (AssemblerInputs.interface relation program)
      (AssemblerInputs.rootOffset program)) env template at phase
  rw [AssemblerInputs.parent_piCcsOffset_eq relation program] at phase
  apply PiCCS.v1_1.Formal.PhaseTransport.phaseHolds_of_agree_satisfy
    relation ajtai (AssemblerInputs.piCcsInterface program)
    (AssemblerInputs.piCcsOffset program) (PiCcsSupport program)
    env completed.current template
    ((piCcsExternalSupport program).mono (fun _ external => Or.inl external))
    (piCcsSupport_agrees program completed endEq)
    (piCcsRoundPoint_eval_eq program completed endEq)
  · simpa [AssemblerInputs.piCcsOutputState] using
      piCcsOutputState_eval_eq relation program completed endEq
  · exact phase

private theorem completePilotPrefix
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
          (AssemblerInputs.outputHashOffset program)] ∧
      AssemblerInputs.rootOffset program +
          localLength completed.operations =
        AssemblerInputs.piCcsOffset program := by
  let p0 := Sequence.empty env (AssemblerInputs.rootOffset program)
  have priorSpec := specification.prior
  rw [AssemblerInputs.parent_priorOffset_eq relation program] at priorSpec
  change PriorStateHash.SpecHolds PilotProduction.priorInterface
    (AssemblerInputs.priorOffset program) env at priorSpec
  have priorStart : AssemblerInputs.rootOffset program +
      localLength p0.operations = AssemblerInputs.priorOffset program := by
    change AssemblerInputs.rootOffset program + 0 =
      AssemblerInputs.rootOffset program
    omega
  rcases Sequence.appendAt p0 "stage1.prior_state_hash"
      (Lifecycle.Stage1.priorChild relation program
        (AssemblerInputs.interface relation program))
      (AssemblerInputs.priorOffset program) priorStart
      (PriorStateHash.flatConstraints_varsBelow
        PilotProduction.priorInterface (AssemblerInputs.priorOffset program)
        (AssemblerPilotBounds.priorAssumptions program env))
      (AssemblerPilotBounds.priorAssumptions program env) priorSpec with
    ⟨p1, p1Operations, p1End, _p0to1, _priorRows⟩
  have outputStart : AssemblerInputs.rootOffset program +
      localLength p1.operations =
        AssemblerInputs.outputHashOffset program := by
    calc
      _ = AssemblerInputs.priorOffset program + localLength
          (Circuit.ops
            (Lifecycle.Stage1.priorChild relation program
              (AssemblerInputs.interface relation program)).main
            (AssemblerInputs.priorOffset program)) := p1End
      _ = Lifecycle.Stage1.outputHashOffset relation program
          (AssemblerInputs.interface relation program)
          (AssemblerInputs.rootOffset program) := by
        unfold Lifecycle.Stage1.outputHashOffset
        rw [AssemblerInputs.parent_priorOffset_eq relation program]
        rw [(Lifecycle.Stage1.priorChild relation program
          (AssemblerInputs.interface relation program)).privateCount_eq]
      _ = AssemblerInputs.outputHashOffset program :=
        AssemblerInputs.parent_outputHashOffset_eq relation program
  have outputSpecInitial := specification.outputHash
  rw [AssemblerInputs.parent_outputHashOffset_eq relation program]
    at outputSpecInitial
  change OutputHash.SpecHolds PilotProduction.outputInterface
    (AssemblerInputs.outputHashOffset program) env at outputSpecInitial
  have outputSpecCurrent : OutputHash.SpecHolds
      PilotProduction.outputInterface
      (AssemblerInputs.outputHashOffset program) p1.current := by
    apply OutputHash.specHolds_of_agree_satisfy
      PilotProduction.outputInterface
      (AssemblerInputs.outputHashOffset program)
      (fun index => index < AssemblerInputs.rootOffset program)
      env p1.current
      (AssemblerPilotBounds.outputSupport program).1
      (AssemblerPilotBounds.outputSupport program).2
    · intro index below
      exact p1.agrees index (Or.inl below)
    · exact outputSpecInitial
  let outputAssumptions := AssemblerPilotBounds.outputAssumptions program
    p1.current
  rcases Sequence.appendAt p1 "stage1.output_hash"
      (Lifecycle.Stage1.outputHashChild relation program
        (AssemblerInputs.interface relation program))
      (AssemblerInputs.outputHashOffset program) outputStart
      (OutputHash.flatConstraints_varsBelow PilotProduction.outputInterface
        (AssemblerInputs.outputHashOffset program) outputAssumptions)
      outputAssumptions outputSpecCurrent with
    ⟨p2, p2Operations, p2End, _p1to2, _outputRows⟩
  refine ⟨p2, ?_, ?_⟩
  · rw [p2Operations, p1Operations]
    rfl
  · calc
      _ = AssemblerInputs.outputHashOffset program + localLength
          (Circuit.ops
            (Lifecycle.Stage1.outputHashChild relation program
              (AssemblerInputs.interface relation program)).main
            (AssemblerInputs.outputHashOffset program)) := p2End
      _ = Lifecycle.Stage1.piCcsOffset relation program
          (AssemblerInputs.interface relation program)
          (AssemblerInputs.rootOffset program) := by
        unfold Lifecycle.Stage1.piCcsOffset
        rw [AssemblerInputs.parent_outputHashOffset_eq relation program]
        rw [(Lifecycle.Stage1.outputHashChild relation program
          (AssemblerInputs.interface relation program)).privateCount_eq]
      _ = AssemblerInputs.piCcsOffset program :=
        AssemblerInputs.parent_piCcsOffset_eq relation program

/-- Honest completion of the pilot and PiCCS parent prefix. PiCCS remains one
opaque Stage 1 child even though its canonical builder internally completes
all 12 PiCCS leaves. -/
theorem completePiCcsPrefix
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
          (AssemblerInputs.piCcsOffset program)] ∧
      AssemblerInputs.rootOffset program +
          localLength completed.operations =
        AssemblerInputs.piRlcOffset program := by
  rcases completePilotPrefix relation ajtai program template env specification with
    ⟨p2, p2Operations, p2End⟩
  have phase := piCcsPhase_after_pilot relation ajtai program template env
    specification p2 p2End
  have assumptions := AssemblerBounds.piCcsAssumptions relation program p2.current
  rcases PiCCS.v1_1.Formal.completePrefix relation ajtai
      (AssemblerInputs.piCcsInterface program) template p2.current
      (AssemblerInputs.piCcsOffset program) assumptions phase with
    ⟨built, builtOperations⟩
  let child := Lifecycle.Stage1.piCcsChild relation ajtai program
    (AssemblerInputs.interface relation program) template
  have childMain : child.main = PiCCS.v1_1.Formal.main relation
      (AssemblerInputs.piCcsInterface program) := by
    rfl
  have childOperations : built.operations = Circuit.ops child.main
      (AssemblerInputs.piCcsOffset program) := by
    rw [childMain, PiCCS.v1_1.Formal.main_ops]
    exact builtOperations
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main (AssemblerInputs.piCcsOffset program)),
      expression.VarsBelow
        (AssemblerInputs.piCcsOffset program + localLength
          (Circuit.ops child.main (AssemblerInputs.piCcsOffset program))) := by
    rw [← childOperations]
    exact built.scope
  have childAgrees : AgreesOutside p2.current built.current
      (AssemblerInputs.piCcsOffset program)
      (localLength
        (Circuit.ops child.main (AssemblerInputs.piCcsOffset program))) := by
    rw [← childOperations]
    exact built.agrees
  have childRows : holdsFlat built.current
      (Circuit.ops child.main (AssemblerInputs.piCcsOffset program)) := by
    rw [← childOperations]
    exact built.rows
  rcases Sequence.appendBuiltAt p2 "stage1.piccs.v1_1" child
      (AssemblerInputs.piCcsOffset program) p2End childScope built.current
      childAgrees childRows with
    ⟨p3, p3Operations, p3End, _p2to3, _piCcsRows⟩
  refine ⟨p3, ?_, ?_⟩
  · rw [p3Operations, p2Operations]
    rfl
  · calc
      _ = AssemblerInputs.piCcsOffset program + localLength
          (Circuit.ops child.main (AssemblerInputs.piCcsOffset program)) :=
        p3End
      _ = Lifecycle.Stage1.piRlcOffset relation ajtai program
          (AssemblerInputs.interface relation program) template
          (AssemblerInputs.rootOffset program) := by
        unfold Lifecycle.Stage1.piRlcOffset
        rw [AssemblerInputs.parent_piCcsOffset_eq relation program]
        rw [child.privateCount_eq]
      _ = AssemblerInputs.piRlcOffset program :=
        AssemblerInputs.parent_piRlcOffset_eq relation ajtai program template

end NightstreamFPrime.Layout.Stage1.AssemblerCompleteness
