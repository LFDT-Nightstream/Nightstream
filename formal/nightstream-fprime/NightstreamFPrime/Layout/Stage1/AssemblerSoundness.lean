import NightstreamFPrime.Layout.Stage1.AssemblerInputs
import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics
import NightstreamFPrime.Layout.Stage1.PiCCSRepresentation
import NightstreamFPrime.Lifecycle.Stage1.Accumulator

/-!
Owns deterministic semantic composition for the seven-child Stage 1 parent.

The representation record names the exact typed HyperNova input and output
carried by the symbolic wires. The canonical theorem composes the compact
PiCCS, PiRLC, and PiDEC phase results into the recursive SuperNeo accumulator
graph and derives the complete fixed augmented-step relation.

This file does not emit rows, select an application, close the recursive fixed
point, or include the outer terminal verifier.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerSoundness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def initialStateValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env) : AppState :=
  List.ofFn fun index =>
    (interface.running.initialState offset index).eval env

def currentStateValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env) : AppState :=
  List.ofFn fun index =>
    (interface.running.currentState offset index).eval env

def recursiveRunningValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  PiCCS.v1_1.StatementAbsorption.evalRunning
    (interface.running.recursive offset) env

def outputRunningValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  PiCCS.v1_1.StatementAbsorption.evalRunning
    (interface.running.output offset) env

/-- The one complete NIFS proof value carried by the Stage 1 parent. PiCCS
owns the round and output fields; PiDEC owns the child message fields. -/
def nifsProofValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (piCcsOffset piDecOffset : Nat) (env : Env) :
    Proof (ProductionKey.degreeBound relation) :=
  let piCcs := PiCCS.v1_1.Formal.evalProof relation interface.piCcs
    piCcsOffset env template
  let attempt := PiDEC.v1_1.Semantics.inputAttempt relation interface.piDec
    piDecOffset env
  { piCcsRounds := piCcs.piCcsRounds
    piCcsOutput := piCcs.piCcsOutput
    piDecCommitments := fun running =>
      (attempt.messages (AssemblerInputs.childOfRunning running)).commitment
    piDecEvaluations := fun running =>
      (attempt.messages
        (AssemblerInputs.childOfRunning running)).evaluations.getD 0
          PaperAlgebra.evaluationZero }

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

private theorem evaluationFamily_ext
    (left right : StrongReduction.EvaluationFamily K productionShape)
    (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem piDecChildMessage_ext
    (left right : PiDEC.PaperVerifier.ChildMessage
      PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (commitment : left.commitment = right.commitment)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem piDecAttempt_ext
    (left right : PiDEC.v1_1.InputBinding.Attempt logicalWidth publicFits)
    (parent : left.parent = right.parent)
    (messages : left.messages = right.messages) : left = right := by
  cases left
  cases right
  simp_all

private theorem running_ext
    (left right : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

theorem compactPoint_eq_roundTranscript
    (program : Application.Program) (env : Env) :
    PiCCS.v1_1.StatementAbsorption.evalPoint
        (AssemblerInputs.piCcsRoundPoint
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        env =
      PiCCS.v1_1.RoundTranscript.evalRoundPoint
        (PiCCS.v1_1.Formal.roundTranscriptInterface
          (PiCCS.v1_1.Formal.atOffset
            (AssemblerInputs.piCcsInterface
              (logicalWidth := logicalWidth) (publicFits := publicFits) program)
            (AssemblerInputs.piCcsOffset program)))
        (PiCCS.v1_1.Formal.roundTranscriptOffset
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program)) env := by
  have interfaceEq :
      PiCCS.v1_1.Formal.atOffset
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) =
        AssemblerInputs.piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program := by
    rfl
  have startEq :
      PiCCS.v1_1.Formal.roundTranscriptOffset
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) =
        PiCCS.v1_1.Formal.roundTranscriptStart
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits)
            program) := by
    rw [← PiCCS.v1_1.Formal.roundTranscriptStart_atOffset
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program), interfaceEq]
  apply cubePoint_ext
  unfold PiCCS.v1_1.StatementAbsorption.evalPoint
    PiCCS.v1_1.RoundTranscript.evalRoundPoint
    AssemblerInputs.piCcsRoundPoint PiCCS.v1_1.Formal.roundPoint
  rw [interfaceEq, startEq]
  simp [canonicalFinIndices]

theorem compactPiRlcInputs_eq_keyOutputs
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (piCcsPhase : PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program) env template) :
    PiRLC.v1_1.Semantics.evalInputs relation
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program) env =
      (ProductionKey.key relation ajtai).piCcsOutputs
        (PiCCS.v1_1.Formal.evalRunning
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) env)
        (PiCCS.v1_1.Formal.evalFresh
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) env)
        (nifsProofValue (AssemblerInputs.interface relation program) template
          (AssemblerInputs.piCcsOffset program)
          (AssemblerInputs.piDecOffset program) env) := by
  have phasePoint :
      PiCCS.v1_1.RoundTranscript.evalRoundPoint
          (PiCCS.v1_1.Formal.roundTranscriptInterface
            (PiCCS.v1_1.Formal.atOffset
              (AssemblerInputs.piCcsInterface
                (logicalWidth := logicalWidth) (publicFits := publicFits)
                program)
              (AssemblerInputs.piCcsOffset program)))
          (PiCCS.v1_1.Formal.roundTranscriptOffset
            (AssemblerInputs.piCcsInterface
              (logicalWidth := logicalWidth) (publicFits := publicFits) program)
            (AssemblerInputs.piCcsOffset program)) env =
        ((ProductionKey.key relation ajtai).piCcsExecution
          (PiCCS.v1_1.Formal.evalRunning
            (AssemblerInputs.piCcsInterface
              (logicalWidth := logicalWidth) (publicFits := publicFits) program)
            (AssemblerInputs.piCcsOffset program) env)
          (PiCCS.v1_1.Formal.evalFresh
            (AssemblerInputs.piCcsInterface
              (logicalWidth := logicalWidth) (publicFits := publicFits) program)
            (AssemblerInputs.piCcsOffset program) env)
          (nifsProofValue (AssemblerInputs.interface relation program) template
            (AssemblerInputs.piCcsOffset program)
            (AssemblerInputs.piDecOffset program) env)).coins.roundPoint := by
    simpa [nifsProofValue] using piCcsPhase.roundPoint
  have pointEq := (compactPoint_eq_roundTranscript program env).trans phasePoint
  change (fun source =>
      PiRLC.v1_1.InputBinding.evalInput relation
        (PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          (PiRLC.v1_1.Semantics.sourceIndex source))
        (AssemblerInputs.piCcsRoundPoint
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        env) = _
  funext source
  let joint : Fin productionShape.sourceCount :=
    Fin.cast (ProductionKey.key relation ajtai).total_eq_sourceCount source
  have sourceEq : PiRLC.v1_1.Semantics.sourceIndex source = joint := by
    apply Fin.ext
    rfl
  apply inputInstance_ext
  · rfl
  · change
      (fun row coefficient =>
        ((PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          (PiRLC.v1_1.Semantics.sourceIndex source)).commitment row coefficient
          ).eval env) =
      Fin.addCases
        (PiCCS.v1_1.Formal.evalFresh
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) env).commitments
        (PiCCS.v1_1.Formal.evalRunning
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) env).commitments joint
    rw [PiRLCInputs.sourceInput_eq_canonical, sourceEq]
    refine Fin.addCases (fun fresh => ?_) (fun running => ?_) joint
    · funext row coefficient
      simp [PiRLCInputs.canonicalSourceInput,
        PiCCS.v1_1.Formal.evalFresh,
        PiCCS.v1_1.StatementAbsorption.evalFresh,
        AssemblerInputs.piCcsInterface, PiRLCInputs.piCcsInterface,
        PiCCSInputs.interface]
    · funext row coefficient
      simp [PiRLCInputs.canonicalSourceInput,
        PiCCS.v1_1.Formal.evalRunning,
        PiCCS.v1_1.StatementAbsorption.evalRunning,
        AssemblerInputs.piCcsInterface, PiRLCInputs.piCcsInterface,
        PiCCSInputs.interface]
  · change
      (fun column =>
        ((PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          (PiRLC.v1_1.Semantics.sourceIndex source)).publicInput column).eval
          env) =
      Fin.addCases
        (PiCCS.v1_1.Formal.evalFresh
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) env).publicInputs
        (PiCCS.v1_1.Formal.evalRunning
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) env).publicInputs joint
    rw [PiRLCInputs.sourceInput_eq_canonical, sourceEq]
    refine Fin.addCases (fun fresh => ?_) (fun running => ?_) joint
    · funext column
      simp [PiRLCInputs.canonicalSourceInput,
        PiCCS.v1_1.Formal.evalFresh,
        PiCCS.v1_1.StatementAbsorption.evalFresh,
        AssemblerInputs.piCcsInterface, PiRLCInputs.piCcsInterface,
        PiCCSInputs.interface]
    · funext column
      simp [PiRLCInputs.canonicalSourceInput,
        PiCCS.v1_1.Formal.evalRunning,
        PiCCS.v1_1.StatementAbsorption.evalRunning,
        AssemblerInputs.piCcsInterface, PiRLCInputs.piCcsInterface,
        PiCCSInputs.interface]
  · change
      PiCCS.v1_1.StatementAbsorption.evalPoint
          (AssemblerInputs.piCcsRoundPoint
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          env = _
    exact pointEq
  · change
      #[PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          (PiRLC.v1_1.Semantics.sourceIndex source)).evaluation env] =
      #[{
        pad := (nifsProofValue (AssemblerInputs.interface relation program)
          template (AssemblerInputs.piCcsOffset program)
          (AssemblerInputs.piDecOffset program) env
          ).piCcsOutput.padCoordinate joint
        matrix := (nifsProofValue (AssemblerInputs.interface relation program)
          template (AssemblerInputs.piCcsOffset program)
          (AssemblerInputs.piDecOffset program) env
          ).piCcsOutput.matrixCoordinate joint }]
    rw [PiRLCInputs.sourceInput_eq_canonical, sourceEq]
    refine Fin.addCases (fun fresh => ?_) (fun running => ?_) joint
    · have injectionEq : UnifiedSources.freshSourceIndex fresh =
          Fin.castAdd productionShape.runningCount fresh := by
        apply Fin.ext
        rfl
      apply congrArg (fun value : StrongReduction.EvaluationFamily K
        productionShape =>
          (#[value] : Array (StrongReduction.EvaluationFamily K
            productionShape)))
      apply evaluationFamily_ext
      · funext coefficient
        simp [PiRLCInputs.canonicalSourceInput,
          PiCCS.v1_1.StatementAbsorption.evalEvaluation,
          PiCCS.v1_1.Formal.evalProof, PiCCS.v1_1.Formal.evalOutput,
          nifsProofValue,
          AssemblerInputs.interface, AssemblerInputs.piCcsInterface,
          PiRLCInputs.piCcsInterface,
          PiCCSInputs.interface, injectionEq]
      · funext matrix coefficient
        simp [PiRLCInputs.canonicalSourceInput,
          PiCCS.v1_1.StatementAbsorption.evalEvaluation,
          PiCCS.v1_1.Formal.evalProof, PiCCS.v1_1.Formal.evalOutput,
          nifsProofValue,
          AssemblerInputs.interface, AssemblerInputs.piCcsInterface,
          PiRLCInputs.piCcsInterface,
          PiCCSInputs.interface, injectionEq]
    · have injectionEq : UnifiedSources.runningSourceIndex running =
          Fin.natAdd productionShape.freshCount running := by
        apply Fin.ext
        rfl
      apply congrArg (fun value : StrongReduction.EvaluationFamily K
        productionShape =>
          (#[value] : Array (StrongReduction.EvaluationFamily K
            productionShape)))
      apply evaluationFamily_ext
      · funext coefficient
        simp [PiRLCInputs.canonicalSourceInput,
          PiCCS.v1_1.StatementAbsorption.evalEvaluation,
          PiCCS.v1_1.Formal.evalProof, PiCCS.v1_1.Formal.evalOutput,
          nifsProofValue,
          AssemblerInputs.interface, AssemblerInputs.piCcsInterface,
          PiRLCInputs.piCcsInterface,
          PiCCSInputs.interface, injectionEq]
      · funext matrix coefficient
        simp [PiRLCInputs.canonicalSourceInput,
          PiCCS.v1_1.StatementAbsorption.evalEvaluation,
          PiCCS.v1_1.Formal.evalProof, PiCCS.v1_1.Formal.evalOutput,
          nifsProofValue,
          AssemblerInputs.interface, AssemblerInputs.piCcsInterface,
          PiRLCInputs.piCcsInterface,
          PiCCSInputs.interface, injectionEq]
  · rfl

theorem compactPiRlcInitialState_eq_key
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (piCcsPhase : PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program) env template) :
    PiRLC.v1_1.SamplerChain.evalInitialState
        (PiRLC.v1_1.Formal.samplerInterface
          (PiRLC.v1_1.Formal.atOffset
            (AssemblerInputs.piRlcInterface relation program)
            (AssemblerInputs.piRlcOffset program)))
        (PiRLC.v1_1.Formal.samplerOffset
          (AssemblerInputs.piRlcOffset program)) env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        (PiCCS.v1_1.Formal.evalRunning
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) env)
        (PiCCS.v1_1.Formal.evalFresh
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program) env)
        (nifsProofValue (AssemblerInputs.interface relation program) template
          (AssemblerInputs.piCcsOffset program)
          (AssemblerInputs.piDecOffset program) env)).outgoingState := by
  have outgoing := piCcsPhase.outgoingState
  change PiCCS.v1_1.StatementAbsorption.evalState env
      (PiCCS.v1_1.Formal.outputBindingFinalState relation
        (AssemblerInputs.piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        (AssemblerInputs.piCcsOffset program)) = _ at outgoing
  simpa [PiRLC.v1_1.SamplerChain.evalInitialState,
    PiRLC.v1_1.SamplerChain.evalStateAt, PiRLC.v1_1.Sampler.evalState,
    PiRLC.v1_1.Formal.samplerInterface, PiRLC.v1_1.Formal.atOffset,
    PiRLC.v1_1.Formal.samplerOffset, AssemblerInputs.piRlcInterface,
    AssemblerInputs.piCcsOutputState, nifsProofValue,
    PiCCS.v1_1.StatementAbsorption.evalState] using outgoing

private theorem compactPiDecAttempt_eq_key
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation)) :
    PiDEC.v1_1.Semantics.inputAttempt relation
        (AssemblerInputs.piDecInterface relation program)
        (AssemblerInputs.piDecOffset program) env =
      (ProductionKey.key relation ajtai).piDecAttemptForParent
        (nifsProofValue (AssemblerInputs.interface relation program) template
          (AssemblerInputs.piCcsOffset program)
          (AssemblerInputs.piDecOffset program) env)
        (PiRLC.v1_1.Semantics.evalOutput relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) env) := by
  apply piDecAttempt_ext
  · rfl
  · funext child
    let running : Fin productionShape.runningCount :=
      Fin.cast (ProductionKey.key relation ajtai).outputCount_eq child
    have childEq : AssemblerInputs.childOfRunning running = child := by
      apply Fin.ext
      rfl
    apply piDecChildMessage_ext
    · funext row coefficient
      simp [PiDEC.v1_1.Semantics.inputAttempt,
        PiDEC.v1_1.InputBinding.evalAttempt,
        PiDEC.v1_1.InputBinding.evalMessage,
        Nifs.PaperNonInteractive.Key.piDecAttemptForParent,
        nifsProofValue, AssemblerInputs.interface,
        AssemblerInputs.piDecInterface, childEq, running]
    · simp [PiDEC.v1_1.Semantics.inputAttempt,
        PiDEC.v1_1.InputBinding.evalAttempt,
        PiDEC.v1_1.InputBinding.evalMessage,
        Nifs.PaperNonInteractive.Key.piDecAttemptForParent,
        nifsProofValue, AssemblerInputs.interface,
        AssemblerInputs.piDecInterface, childEq, running]

private theorem compactOutputForAttempt_eq_recursive
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (piDecPhase : PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piDecInterface relation program)
      (AssemblerInputs.piDecOffset program) env) :
    (ProductionKey.key relation ajtai).outputForAttempt
        (nifsProofValue (AssemblerInputs.interface relation program) template
          (AssemblerInputs.piCcsOffset program)
          (AssemblerInputs.piDecOffset program) env)
        (PiDEC.v1_1.Semantics.inputAttempt relation
          (AssemblerInputs.piDecInterface relation program)
          (AssemblerInputs.piDecOffset program) env)
        ((ProductionKey.key relation ajtai).piDecPublicInputSplit.split
          (PiDEC.v1_1.Semantics.inputAttempt relation
            (AssemblerInputs.piDecInterface relation program)
            (AssemblerInputs.piDecOffset program) env).parent.publicInput) =
      recursiveRunningValue (AssemblerInputs.interface relation program)
        (AssemblerInputs.runningOffset program) env := by
  apply running_ext
  · rfl
  · funext running row coefficient
    rfl
  · funext running column
    let child := AssemblerInputs.childOfRunning running
    have publicEq := PiDEC.PaperVerifier.OutputAccepted.childPublicInput_eq
      piDecPhase child
    change
      (PaperAlgebra.publicInputSplit ajtai).split
          (PiDEC.v1_1.Semantics.inputAttempt relation
            (AssemblerInputs.piDecInterface relation program)
            (AssemblerInputs.piDecOffset program) env).parent.publicInput
          child column =
        (PiDEC.v1_1.Semantics.output relation
          (AssemblerInputs.piDecInterface relation program)
          (AssemblerInputs.piDecOffset program) env child).publicInput column
    exact (congrFun publicEq column).symm
  · funext running
    rfl

/-- Typed meaning of the external wires of one fixed Stage 1 parent. -/
structure Represents
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (program : Application.Program)
    (interface : Lifecycle.Stage1.Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount) : Prop where
  priorPreimage : PriorStateHash.RepresentsPreimage interface.pilot.prior
    (Lifecycle.Stage1.priorOffset offset) env
    (priorHashPreimage (setup relation ajtai vk) input)
  priorPublicInput : PriorStateHash.RepresentsPublicInput
    interface.pilot.prior (Lifecycle.Stage1.priorOffset offset) env
    ((machineFor publicFits program).freshPublic input.fresh)
  nextPreimage : OutputHash.RepresentsPreimage interface.pilot.output
    (Lifecycle.Stage1.outputHashOffset relation program interface offset) env
    (nextHashPreimage (setup relation ajtai vk) input output)
  nextDigest : OutputHash.RepresentsDigest interface.pilot.output
    (Lifecycle.Stage1.outputHashOffset relation program interface offset) env
    output.x
  applicationInput : Application.inputState interface.application
    (Lifecycle.Stage1.applicationOffset relation ajtai program interface
      template offset) env = input.zi
  applicationWitness : Application.witnessValue interface.application
    (Lifecycle.Stage1.applicationOffset relation ajtai program interface
      template offset) env = input.witness
  applicationOutput : Application.outputState interface.application
    (Lifecycle.Stage1.applicationOffset relation ajtai program interface
      template offset) env = output.zNext
  runningInput : PiCCS.v1_1.Formal.evalRunning interface.piCcs
    (Lifecycle.Stage1.piCcsOffset relation program interface offset) env =
      input.running functionIndex
  freshInput : PiCCS.v1_1.Formal.evalFresh interface.piCcs
    (Lifecycle.Stage1.piCcsOffset relation program interface offset) env =
      input.fresh
  proofInput : nifsProofValue interface template
    (Lifecycle.Stage1.piCcsOffset relation program interface offset)
    (Lifecycle.Stage1.piDecOffset relation ajtai program interface template
      offset) env = input.nifsProof
  iterationZero :
    RunningTransition.iterationValue interface.running
        (Lifecycle.Stage1.runningOffset relation ajtai program interface
          template offset) env = 0 ↔
      input.iteration = 0
  initialState : initialStateValue interface
    (Lifecycle.Stage1.runningOffset relation ajtai program interface
      template offset) env = input.z0
  currentState : currentStateValue interface
    (Lifecycle.Stage1.runningOffset relation ajtai program interface
      template offset) env = input.zi
  runningOutput : outputRunningValue interface
    (Lifecycle.Stage1.runningOffset relation ajtai program interface
      template offset) env = output.runningNext functionIndex
  priorPc : input.priorPc = 1
  pcNext : output.pcNext = functionIndex

/-- The compact seven-child Stage 1 specification implies the complete
deterministic SuperNeo accumulator update. -/
theorem spec_implies_compactAccumulator
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (program : Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program) env) :
    Accumulator.Holds relation ajtai vk
      (PiCCS.v1_1.Formal.evalRunning
        (AssemblerInputs.piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        (AssemblerInputs.piCcsOffset program) env)
      (PiCCS.v1_1.Formal.evalFresh
        (AssemblerInputs.piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        (AssemblerInputs.piCcsOffset program) env)
      (nifsProofValue (AssemblerInputs.interface relation program) template
        (AssemblerInputs.piCcsOffset program)
        (AssemblerInputs.piDecOffset program) env)
      (recursiveRunningValue (AssemblerInputs.interface relation program)
        (AssemblerInputs.runningOffset program) env) := by
  have piCcsPhase : PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program) env template := by
    have phase := specification.piCcs
    change PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (Lifecycle.Stage1.piCcsOffset relation program
        (AssemblerInputs.interface relation program)
        (AssemblerInputs.rootOffset program)) env template at phase
    rw [AssemblerInputs.parent_piCcsOffset_eq relation program] at phase
    exact phase
  have piRlcPhase : PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program) env := by
    have phase := specification.piRlc
    change PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piRlcInterface relation program)
      (Lifecycle.Stage1.piRlcOffset relation ajtai program
        (AssemblerInputs.interface relation program) template
        (AssemblerInputs.rootOffset program)) env at phase
    rw [AssemblerInputs.parent_piRlcOffset_eq relation ajtai program template]
      at phase
    exact phase
  have piDecPhase : PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
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
  let proof := nifsProofValue (AssemblerInputs.interface relation program)
    template (AssemblerInputs.piCcsOffset program)
      (AssemblerInputs.piDecOffset program) env
  let computedOutput := recursiveRunningValue
    (AssemblerInputs.interface relation program)
      (AssemblerInputs.runningOffset program) env
  have wiring : AccumulatorSemantics.PhaseWiring relation ajtai env
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program) template proof
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program)
      (AssemblerInputs.piDecInterface relation program)
      (AssemblerInputs.piDecOffset program) computedOutput := by
    refine {
      proofView := ?_
      inputs := ?_
      initialState := ?_
      attempt := ?_
      output := ?_ }
    · exact ⟨rfl, rfl⟩
    · exact compactPiRlcInputs_eq_keyOutputs relation ajtai program env
        template piCcsPhase
    · exact compactPiRlcInitialState_eq_key relation ajtai program env
        template piCcsPhase
    · exact compactPiDecAttempt_eq_key relation ajtai program env template
    · exact compactOutputForAttempt_eq_recursive relation ajtai program env
        template piDecPhase
  exact AccumulatorSemantics.phases_imply_holds_of_wiring relation ajtai vk env
    (AssemblerInputs.piCcsInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits) program)
    (AssemblerInputs.piCcsOffset program) template proof
    (AssemblerInputs.piRlcInterface relation program)
    (AssemblerInputs.piRlcOffset program)
    (AssemblerInputs.piDecInterface relation program)
    (AssemblerInputs.piDecOffset program) computedOutput piCcsPhase piRlcPhase
    piDecPhase wiring

private theorem stateValues_eq
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env)
    (equal : ∀ index,
      (interface.running.initialState offset index).eval env =
        (interface.running.currentState offset index).eval env) :
    initialStateValue interface offset env =
      currentStateValue interface offset env := by
  apply List.ext_get
  · simp [initialStateValue, currentStateValue]
  · intro index leftBound rightBound
    have bounded : index < RunningTransition.stateWordCount := by
      simpa [initialStateValue] using leftBound
    simpa [initialStateValue, currentStateValue] using
      equal ⟨index, bounded⟩

private theorem slot_eq_functionIndex (slot : Fin slotCount) :
    slot = functionIndex := by
  apply Fin.ext
  have bound := slot.isLt
  change slot.val < 1 at bound
  change slot.val = 0
  omega

private theorem application_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest)
    (program : Application.Program)
    (interface : Lifecycle.Stage1.Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      interface template offset env)
    (represents : Represents relation ajtai vk program interface template
      offset env input output) :
    output.zNext = program.step input.zi input.witness := by
  let applicationOffsetValue := Lifecycle.Stage1.applicationOffset relation
    ajtai program interface template offset
  have applicationSpec :
      (program.circuit interface.application).spec applicationOffsetValue env := by
    simpa [Lifecycle.Stage1.applicationChild, applicationOffsetValue] using
      specification.application
  have applicationHolds :
      Application.Holds program.step interface.application
        applicationOffsetValue env :=
    (program.spec_iff interface.application applicationOffsetValue env).mp
      applicationSpec
  unfold Application.Holds at applicationHolds
  calc
    output.zNext = Application.outputState interface.application
        applicationOffsetValue env :=
      represents.applicationOutput.symm
    _ = program.step
        (Application.inputState interface.application applicationOffsetValue env)
        (Application.witnessValue interface.application applicationOffsetValue
          env) :=
      applicationHolds
    _ = program.step input.zi input.witness := by
      rw [represents.applicationInput, represents.applicationWitness]

/-- Arbitrary satisfying child assignments imply the exact fixed HyperNova
step once their external wires are identified and the recursive SuperNeo
verifier graph is supplied. The accumulator premise is unused on the base
branch. -/
theorem spec_implies_stepHoldsFor
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (program : Application.Program)
    (interface : Lifecycle.Stage1.Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      interface template offset env)
    (represents : Represents relation ajtai vk program interface template
      offset env input output)
    (accumulator : input.iteration ≠ 0 →
      Accumulator.Holds relation ajtai vk (input.running functionIndex)
        input.fresh input.nifsProof
        (recursiveRunningValue interface
          (Lifecycle.Stage1.runningOffset relation ajtai program interface
            template offset) env)) :
    StepHoldsFor relation ajtai vk program input output := by
  let runningAt := Lifecycle.Stage1.runningOffset relation ajtai program
    interface template offset
  have runningSpec : RunningTransition.SpecHolds interface.running
      runningAt env := by
    simpa [Lifecycle.Stage1.runningChild, runningAt] using
      specification.running
  have application : output.zNext = program.step input.zi input.witness :=
    application_eq relation ajtai vk program interface template offset env
      input output specification represents
  have outputHash : OutputHolds (setup relation ajtai vk)
      (machineFor publicFits program) input output := by
    have childSpec : OutputHash.SpecHolds interface.pilot.output
        (Lifecycle.Stage1.outputHashOffset relation program interface offset)
        env := by
      simpa [Lifecycle.Stage1.outputHashChild, Pilot.outputCircuit] using
        specification.outputHash
    simpa [machineFor] using
      OutputHash.builder_implies_output_slot interface.pilot.output
        (Lifecycle.Stage1.outputHashOffset relation program interface offset)
        env relation ajtai vk program.step input output childSpec
        represents.nextPreimage represents.nextDigest
  have priorPublicInput :
      (machineFor publicFits program).freshPublic input.fresh =
        (machineFor publicFits program).encodeInstance
          ((machineFor publicFits program).hash
            (priorHashPreimage (setup relation ajtai vk) input)) := by
    have childSpec : PriorStateHash.SpecHolds interface.pilot.prior
        (Lifecycle.Stage1.priorOffset offset) env := by
      simpa [Lifecycle.Stage1.priorChild, Pilot.priorCircuit] using
        specification.prior
    simpa [machineFor] using
      PriorStateHash.builder_implies_recursive_slot interface.pilot.prior
        (Lifecycle.Stage1.priorOffset offset) env relation ajtai vk
        program.step input childSpec represents.priorPreimage
        represents.priorPublicInput
  change FixedAugmentedTransition (setup relation ajtai vk)
    (machineFor publicFits program) functionIndex input output
  refine ⟨represents.pcNext, ?_, outputHash, ?_⟩
  · simpa [machineFor, machine] using application
  · rcases Nat.eq_zero_or_pos input.iteration with iterationZero |
      iterationPositive
    · have fieldZero : RunningTransition.iterationValue interface.running
          runningAt env = 0 :=
        represents.iterationZero.mpr iterationZero
      have initialState : input.z0 = input.zi := by
        calc
          input.z0 = initialStateValue interface runningAt env :=
            represents.initialState.symm
          _ = currentStateValue interface runningAt env :=
            stateValues_eq interface runningAt env
              (runningSpec.initialState fieldZero)
          _ = input.zi := represents.currentState
      have runningBase : outputRunningValue interface runningAt env =
          defaultRunning (logicalWidth := logicalWidth)
            (publicFits := publicFits) := by
        apply PiCCSRepresentation.serializeRunning_injective
        exact RunningTransition.spec_serialized_base runningSpec fieldZero
      have defaultOutput : output.runningNext =
          fun _ => (setup relation ajtai vk).defaultRunning := by
        funext slot
        have slotEq : slot = functionIndex := slot_eq_functionIndex slot
        subst slot
        calc
          output.runningNext functionIndex =
              outputRunningValue interface runningAt env :=
            represents.runningOutput.symm
          _ = defaultRunning (logicalWidth := logicalWidth)
              (publicFits := publicFits) := runningBase
          _ = (setup relation ajtai vk).defaultRunning := rfl
      exact Or.inl ⟨iterationZero, initialState, defaultOutput⟩
    · have iterationNonzero : input.iteration ≠ 0 :=
        Nat.ne_of_gt iterationPositive
      have fieldNonzero : RunningTransition.iterationValue interface.running
          runningAt env ≠ 0 := by
        intro fieldZero
        exact iterationNonzero (represents.iterationZero.mp fieldZero)
      have runningRecursive : outputRunningValue interface runningAt env =
          recursiveRunningValue interface runningAt env := by
        apply PiCCSRepresentation.serializeRunning_injective
        exact RunningTransition.spec_serialized_recursive runningSpec
          fieldNonzero
      have priorPcValid : InRange slotCount input.priorPc := by
        rw [represents.priorPc]
        norm_num [InRange, slotCount]
      have selectedEq : selectedIndex priorPcValid = functionIndex :=
        slot_eq_functionIndex _
      have selectedNifs : Accepts (setup relation ajtai vk).nifs
          ((setup relation ajtai vk).verifierKeys
            (selectedIndex priorPcValid))
          (input.running (selectedIndex priorPcValid)) input.fresh
          input.nifsProof
          (output.runningNext (selectedIndex priorPcValid)) := by
        have accepted := accumulator iterationNonzero
        rw [selectedEq]
        rw [← represents.runningOutput, runningRecursive]
        simpa only [Accepts, setup, nifsVerifier, Accumulator.Holds] using
          accepted
      have unchanged : ∀ slot, slot ≠ selectedIndex priorPcValid →
          output.runningNext slot = input.running slot := by
        intro slot notSelected
        exact False.elim (notSelected
          ((slot_eq_functionIndex slot).trans selectedEq.symm))
      exact Or.inr ⟨priorPcValid, iterationPositive, priorPublicInput,
        selectedNifs, unchanged⟩

/-- The canonical compact seven-child parent implies the exact fixed
HyperNova step. Its recursive accumulator premise is derived from the three
phase-local specifications and their Lean-owned wiring. -/
theorem compactSpec_implies_stepHoldsFor
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (program : Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program) env)
    (represents : Represents relation ajtai vk program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program) env input output) :
    StepHoldsFor relation ajtai vk program input output := by
  have accumulator := spec_implies_compactAccumulator relation ajtai vk
    program template env specification
  have runningInput := represents.runningInput
  change PiCCS.v1_1.Formal.evalRunning
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (Lifecycle.Stage1.piCcsOffset relation program
        (AssemblerInputs.interface relation program)
        (AssemblerInputs.rootOffset program)) env =
    input.running functionIndex at runningInput
  rw [AssemblerInputs.parent_piCcsOffset_eq relation program] at runningInput
  have freshInput := represents.freshInput
  change PiCCS.v1_1.Formal.evalFresh
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (Lifecycle.Stage1.piCcsOffset relation program
        (AssemblerInputs.interface relation program)
        (AssemblerInputs.rootOffset program)) env =
    input.fresh at freshInput
  rw [AssemblerInputs.parent_piCcsOffset_eq relation program] at freshInput
  have proofInput := represents.proofInput
  change nifsProofValue (AssemblerInputs.interface relation program) template
      (Lifecycle.Stage1.piCcsOffset relation program
        (AssemblerInputs.interface relation program)
        (AssemblerInputs.rootOffset program))
      (Lifecycle.Stage1.piDecOffset relation ajtai program
        (AssemblerInputs.interface relation program) template
        (AssemblerInputs.rootOffset program)) env =
    input.nifsProof at proofInput
  rw [AssemblerInputs.parent_piCcsOffset_eq relation program,
    AssemblerInputs.parent_piDecOffset_eq relation ajtai program template]
      at proofInput
  rw [runningInput, freshInput, proofInput] at accumulator
  have accumulatorAtParent : Accumulator.Holds relation ajtai vk
      (input.running functionIndex) input.fresh input.nifsProof
      (recursiveRunningValue (AssemblerInputs.interface relation program)
        (Lifecycle.Stage1.runningOffset relation ajtai program
          (AssemblerInputs.interface relation program) template
          (AssemblerInputs.rootOffset program)) env) := by
    rw [AssemblerInputs.parent_runningOffset_eq relation ajtai program template]
    exact accumulator
  exact spec_implies_stepHoldsFor relation ajtai vk program
    (AssemblerInputs.interface relation program) template
    (AssemblerInputs.rootOffset program) env input output specification
    represents (fun _ => accumulatorAtParent)

end NightstreamFPrime.Layout.Stage1.AssemblerSoundness
