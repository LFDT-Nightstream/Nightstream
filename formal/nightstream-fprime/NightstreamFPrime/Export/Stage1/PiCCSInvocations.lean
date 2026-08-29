import NightstreamFPrime.Export.Stage1.Invocations
import NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions
import NightstreamFPrime.Layout.Stage1.PiCCSStarts
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.FormalRows

/-!
Owns the four Poseidon2-only PiCCS row packets in the current Stage 1 package.

The action lists come from the lifecycle leaves. This file owns only their
physical row starts, source witness starts, phase tags, and compact assembly.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSInvocations

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1.Invocations
open NightstreamFPrime.Layout.Stage1.PiCCSInputs

def statementPhase : Nat := 3
def challengePhase : Nat := 4
def roundPhase : Nat := 5
def outputPhase : Nat := 6

def statementRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionRowStart
def challengeRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeRowStart
def roundRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptRowStart
def outputRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingRowStart

def statementWitnessStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart
def challengeWitnessStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart
def roundWitnessStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart
def outputWitnessStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart

def parentInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.Interface
      logicalWidth 9 publicFits :=
  NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
    logicalWidth publicFits

def sharedInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.Interface
      logicalWidth 9 publicFits :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.atOffset
    (parentInterface logicalWidth publicFits) phaseOffset

def inputShapes
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiCCS.v1_1.InputShapes relation
      (parentInterface logicalWidth publicFits) phaseOffset :=
  NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.inputShapes relation
    (parentInterface logicalWidth publicFits) phaseOffset
    (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
      logicalWidth publicFits)

def statementInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementAbsorptionInterface
    (sharedInterface logicalWidth publicFits)

def challengeInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.challengeInterface
    (parentInterface logicalWidth publicFits) phaseOffset

def roundInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptInterface
    (sharedInterface logicalWidth publicFits)

def outputInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingInterface
    (sharedInterface logicalWidth publicFits)

/-! The following named equalities are the transcript-to-parent wiring
boundary. They prevent the package assembler from relying on repeated numeric
normalization. -/

theorem statementWitnessStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    statementWitnessStart =
      Formal.statementAbsorptionOffset
        (parentInterface logicalWidth publicFits) phaseOffset := by
  rw [Formal.statementAbsorptionOffset_eq]
  exact (NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq).trans
    rfl

theorem challengeWitnessStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    challengeWitnessStart =
      Formal.challengeOffset (parentInterface logicalWidth publicFits)
        phaseOffset := by
  rw [Formal.challengeOffset_eq,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  rfl

theorem roundWitnessStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    roundWitnessStart =
      Formal.roundTranscriptOffset (parentInterface logicalWidth publicFits)
        phaseOffset := by
  rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  rfl

theorem outputWitnessStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    outputWitnessStart =
      Formal.outputBindingOffset relation
        (parentInterface logicalWidth publicFits) phaseOffset := by
  unfold Formal.outputBindingOffset Formal.nextOffset Formal.childLength
  rw [Formal.finalIdentityOffset_eq_finalIdentityRowOffset relation,
    Formal.finalIdentityCircuit,
    FormalCircuit.withConstantFootprint_main,
    FinalIdentity.localLength_eq]
  rfl

def statementActions
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  StatementAbsorption.actions (statementInterface logicalWidth publicFits)
    statementWitnessStart

def challengeActions
    (_logicalWidth : Nat)
    (_publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth _logicalWidth) : List Formal.Action :=
  ChallengeDerivation.layoutActions

def roundActions
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  RoundTranscript.layoutActions (roundInterface logicalWidth publicFits)
    roundWitnessStart

def outputActions
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  OutputBinding.actions (outputInterface logicalWidth publicFits)
    outputWitnessStart

def statementTrace
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Trace :=
  compileActions statementPhase statementRowStart statementWitnessStart
    Hash.zeroE (statementActions logicalWidth publicFits)

def challengeTrace
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Trace :=
  compileActions challengePhase challengeRowStart challengeWitnessStart
    (statementTrace logicalWidth publicFits).state
    (challengeActions logicalWidth publicFits)

def challengeSemanticTrace
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Trace :=
  compileActions challengePhase challengeRowStart challengeWitnessStart
    ((challengeInterface logicalWidth publicFits).initialState
      challengeWitnessStart)
    (ChallengeDerivation.actions
      (challengeInterface logicalWidth publicFits) challengeWitnessStart)

def roundTrace
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Trace :=
  compileActions roundPhase roundRowStart roundWitnessStart
    (challengeTrace logicalWidth publicFits).state
    (roundActions logicalWidth publicFits)

def roundSemanticTrace
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Trace :=
  compileActions roundPhase roundRowStart roundWitnessStart
    ((roundInterface logicalWidth publicFits).initialState roundWitnessStart)
    (RoundTranscript.actions (roundInterface logicalWidth publicFits)
      roundWitnessStart)

def outputTrace
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Trace :=
  compileActions outputPhase outputRowStart outputWitnessStart
    (roundTrace logicalWidth publicFits).state
    (outputActions logicalWidth publicFits)

def outputSemanticTrace
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Trace :=
  compileActions outputPhase outputRowStart outputWitnessStart
    ((outputInterface logicalWidth publicFits).initialState outputWitnessStart)
    (OutputBinding.actions (outputInterface logicalWidth publicFits)
      outputWitnessStart)

theorem challengeActions_shape_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (challengeActions logicalWidth publicFits).map Formal.Action.shape =
      (ChallengeDerivation.actions
        (challengeInterface logicalWidth publicFits)
        challengeWitnessStart).map Formal.Action.shape := by
  exact (ChallengeDerivation.actions_shape_eq_layout
    (challengeInterface logicalWidth publicFits)
    challengeWitnessStart).symm

theorem roundActions_shape_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (roundActions logicalWidth publicFits).map Formal.Action.shape =
      (RoundTranscript.actions (roundInterface logicalWidth publicFits)
        roundWitnessStart).map Formal.Action.shape := by
  exact (RoundTranscript.actions_shape_eq_layout
    (roundInterface logicalWidth publicFits) roundWitnessStart).symm

theorem statementTrace_state_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (statementTrace logicalWidth publicFits).state =
      StatementAbsorption.finalState
        (statementInterface logicalWidth publicFits)
        statementWitnessStart := by
  unfold statementTrace StatementAbsorption.finalState
    StatementAbsorption.program
  exact compileActions_state_eq statementPhase statementRowStart
    statementWitnessStart Hash.zeroE (statementActions logicalWidth publicFits)

/-- Held compact statement invocations imply the exact statement-absorption
leaf predicate under the proved Spartan pullback. -/
theorem statementTrace_implies_spec
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (holds : ∀ current ∈
      (statementTrace logicalWidth publicFits).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env) :
    StatementAbsorption.SpecHolds
      (statementInterface logicalWidth publicFits) statementWitnessStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have trace := compileActions_traceHolds statementPhase statementRowStart
    statementWitnessStart Hash.zeroE
    (statementActions logicalWidth publicFits) env (by
      change NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
    NightstreamFPrime.Layout.Poseidon2.zeroE_affine
    (NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption.actions_affine
      (statementInterface logicalWidth publicFits) statementWitnessStart
      ((inputShapes logicalWidth publicFits relation).statementAbsorption
        statementWitnessStart))
    (expectedSamples_eq_samples_of_assertionCount_zero statementWitnessStart
      Hash.zeroE (statementActions logicalWidth publicFits)
      (StatementAbsorption.assertionCount_eq
        (statementInterface logicalWidth publicFits) statementWitnessStart))
    holds
  change Formal.TraceHolds
      (List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) Hash.zeroE))
      ((statementActions logicalWidth publicFits).map
        (Formal.Action.eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)))
      (List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        (statementTrace logicalWidth publicFits).state)) at trace
  rw [statementTrace_state_matches logicalWidth publicFits] at trace
  exact trace

theorem challengeInitialState_eq_statementFinalState
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (challengeInterface logicalWidth publicFits).initialState
        challengeWitnessStart =
      StatementAbsorption.finalState
        (statementInterface logicalWidth publicFits)
        statementWitnessStart := by
  let parent := parentInterface logicalWidth publicFits
  let targetInterface :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementAbsorptionInterface
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.atOffset parent phaseOffset)
  have interfaceEq : targetInterface =
      statementInterface logicalWidth publicFits := by
    rfl
  have offsetEq : phaseOffset = statementWitnessStart := by
    simpa [statementWitnessStart] using
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq
  have finalStateEq :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementFinalState
          parent phaseOffset =
        StatementAbsorption.finalState
          (statementInterface logicalWidth publicFits)
          statementWitnessStart := by
    unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementFinalState
    exact (congrArg
      (fun current => StatementAbsorption.finalState current phaseOffset)
      interfaceEq).trans (congrArg
        (StatementAbsorption.finalState
          (statementInterface logicalWidth publicFits)) offsetEq)
  calc
    (challengeInterface logicalWidth publicFits).initialState
        challengeWitnessStart =
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementFinalState
        parent phaseOffset := by
      exact
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.challengeInterface_initialState
          parent phaseOffset challengeWitnessStart
    _ = _ := finalStateEq

theorem challengeTrace_state_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (challengeTrace logicalWidth publicFits).state =
      ChallengeDerivation.finalState
        (challengeInterface logicalWidth publicFits)
        challengeWitnessStart := by
  let interface := challengeInterface logicalWidth publicFits
  have initialState :
      (statementTrace logicalWidth publicFits).state =
        interface.initialState challengeWitnessStart := by
    exact (statementTrace_state_matches logicalWidth publicFits).trans
      (challengeInitialState_eq_statementFinalState
        logicalWidth publicFits).symm
  calc
    (challengeTrace logicalWidth publicFits).state =
        (Formal.compile challengeWitnessStart
          (statementTrace logicalWidth publicFits).state
          (challengeActions logicalWidth publicFits)).output :=
      compileActions_state_eq challengePhase challengeRowStart
        challengeWitnessStart (statementTrace logicalWidth publicFits).state
        (challengeActions logicalWidth publicFits)
    _ = (Formal.compile challengeWitnessStart
          (interface.initialState challengeWitnessStart)
          (challengeActions logicalWidth publicFits)).output :=
      congrArg (fun initial =>
        (Formal.compile challengeWitnessStart initial
          (challengeActions logicalWidth publicFits)).output) initialState
    _ = (ChallengeDerivation.layoutProgram interface
          challengeWitnessStart).output := by
      rfl
    _ = (ChallengeDerivation.program interface
          challengeWitnessStart).output :=
      (ChallengeDerivation.program_shape_eq_layout interface
        challengeWitnessStart).2.2.symm
    _ = ChallengeDerivation.finalState interface challengeWitnessStart := by
      rfl

theorem challengeTrace_eq_semantic
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    challengeTrace logicalWidth publicFits =
      challengeSemanticTrace logicalWidth publicFits := by
  have initialEq : (statementTrace logicalWidth publicFits).state =
      (challengeInterface logicalWidth publicFits).initialState
        challengeWitnessStart :=
    (statementTrace_state_matches logicalWidth publicFits).trans
      (challengeInitialState_eq_statementFinalState
        logicalWidth publicFits).symm
  unfold challengeTrace challengeSemanticTrace
  rw [initialEq]
  exact compileActions_eq_of_shapes challengePhase challengeRowStart
    challengeWitnessStart
    ((challengeInterface logicalWidth publicFits).initialState
      challengeWitnessStart)
    (challengeActions logicalWidth publicFits)
    (ChallengeDerivation.actions
      (challengeInterface logicalWidth publicFits) challengeWitnessStart)
    (challengeActions_shape_matches logicalWidth publicFits)

/-- Held compact challenge invocations imply the exact pre-SumCheck
Fiat--Shamir predicate under the proved Spartan pullback. -/
theorem challengeTrace_implies_spec
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (holds : ∀ current ∈
      (challengeTrace logicalWidth publicFits).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env) :
    ChallengeDerivation.SpecHolds
      (challengeInterface logicalWidth publicFits) challengeWitnessStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  let interface := challengeInterface logicalWidth publicFits
  let semantic := challengeSemanticTrace logicalWidth publicFits
  have semanticEq : challengeTrace logicalWidth publicFits = semantic :=
    challengeTrace_eq_semantic logicalWidth publicFits
  have semanticHolds : ∀ current ∈ semantic.invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
    intro current member
    apply holds current
    rw [semanticEq]
    exact member
  have trace := compileActions_traceHolds challengePhase challengeRowStart
    challengeWitnessStart (interface.initialState challengeWitnessStart)
    (ChallengeDerivation.actions interface challengeWitnessStart) env (by
      change NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
    ((inputShapes logicalWidth publicFits relation).challengeDerivation
      challengeWitnessStart).initialState
    (NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation.actions_affine
      interface challengeWitnessStart
      ((inputShapes logicalWidth publicFits relation).challengeDerivation
        challengeWitnessStart))
    (ChallengeDerivation.expectedSamples_eq_samples interface
      challengeWitnessStart)
    semanticHolds
  have stateEq : semantic.state =
      ChallengeDerivation.finalState interface challengeWitnessStart := by
    exact (congrArg Trace.state semanticEq).symm.trans
      (challengeTrace_state_matches logicalWidth publicFits)
  change Formal.TraceHolds
      (List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        (interface.initialState challengeWitnessStart)))
      ((ChallengeDerivation.actions interface challengeWitnessStart).map
        (Formal.Action.eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)))
      (List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        semantic.state)) at trace
  rw [stateEq] at trace
  exact ChallengeDerivation.trace_implies_specHolds interface
    challengeWitnessStart
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) trace

theorem roundTrace_eq_semantic
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    roundTrace logicalWidth publicFits =
      roundSemanticTrace logicalWidth publicFits := by
  have initialEq : (challengeTrace logicalWidth publicFits).state =
      (roundInterface logicalWidth publicFits).initialState
        roundWitnessStart := by
    simpa [roundInterface, Formal.roundTranscriptInterface,
      Formal.challengeFinalState, sharedInterface] using
      challengeTrace_state_matches logicalWidth publicFits
  unfold roundTrace roundSemanticTrace
  rw [initialEq]
  exact compileActions_eq_of_shapes roundPhase roundRowStart
    roundWitnessStart
    ((roundInterface logicalWidth publicFits).initialState roundWitnessStart)
    (roundActions logicalWidth publicFits)
    (RoundTranscript.actions (roundInterface logicalWidth publicFits)
      roundWitnessStart)
    (roundActions_shape_matches logicalWidth publicFits)

/-- Held compact round invocations imply the exact indexed 28-round
Fiat--Shamir predicate under the proved Spartan pullback. -/
theorem roundTrace_implies_spec
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (holds : ∀ current ∈ (roundTrace logicalWidth publicFits).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env) :
    RoundTranscript.SpecHolds (roundInterface logicalWidth publicFits)
      roundWitnessStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  let interface := roundInterface logicalWidth publicFits
  let semantic := roundSemanticTrace logicalWidth publicFits
  have semanticEq : roundTrace logicalWidth publicFits = semantic :=
    roundTrace_eq_semantic logicalWidth publicFits
  have semanticHolds : ∀ current ∈ semantic.invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
    intro current member
    apply holds current
    rw [semanticEq]
    exact member
  have trace := compileActions_traceHolds roundPhase roundRowStart
    roundWitnessStart (interface.initialState roundWitnessStart)
    (RoundTranscript.actions interface roundWitnessStart) env (by
      change NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
    ((inputShapes logicalWidth publicFits relation).roundTranscript
      roundWitnessStart).initialState
    (NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript.actions_affine
      interface roundWitnessStart
      ((inputShapes logicalWidth publicFits relation).roundTranscript
        roundWitnessStart))
    (RoundTranscript.expectedSamples_eq_samples interface roundWitnessStart)
    semanticHolds
  have semanticState : semantic.state =
      RoundTranscript.finalState interface roundWitnessStart := by
    unfold semantic roundSemanticTrace RoundTranscript.finalState
      RoundTranscript.program
    exact compileActions_state_eq roundPhase roundRowStart roundWitnessStart
      (interface.initialState roundWitnessStart)
      (RoundTranscript.actions interface roundWitnessStart)
  change Formal.TraceHolds
      (List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        (interface.initialState roundWitnessStart)))
      ((RoundTranscript.actions interface roundWitnessStart).map
        (Formal.Action.eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)))
      (List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        semantic.state)) at trace
  rw [semanticState] at trace
  exact (RoundTranscript.trace_iff_specHolds interface roundWitnessStart
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)).mp trace

theorem roundTrace_state_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (roundTrace logicalWidth publicFits).state =
      RoundTranscript.finalState (roundInterface logicalWidth publicFits)
        roundWitnessStart := by
  calc
    (roundTrace logicalWidth publicFits).state =
        (roundSemanticTrace logicalWidth publicFits).state :=
      congrArg Trace.state
        (roundTrace_eq_semantic logicalWidth publicFits)
    _ = _ := by
      unfold roundSemanticTrace RoundTranscript.finalState
        RoundTranscript.program
      exact compileActions_state_eq roundPhase roundRowStart roundWitnessStart
        ((roundInterface logicalWidth publicFits).initialState
          roundWitnessStart)
        (RoundTranscript.actions (roundInterface logicalWidth publicFits)
          roundWitnessStart)

theorem outputTrace_eq_semantic
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    outputTrace logicalWidth publicFits =
      outputSemanticTrace logicalWidth publicFits := by
  have initialEq : (roundTrace logicalWidth publicFits).state =
      (outputInterface logicalWidth publicFits).initialState
        outputWitnessStart := by
    simpa [outputInterface, Formal.outputBindingInterface,
      Formal.roundTranscriptFinalState, sharedInterface] using
      roundTrace_state_matches logicalWidth publicFits
  unfold outputTrace outputSemanticTrace
  rw [initialEq]
  rfl

theorem outputSemanticTrace_state_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (outputSemanticTrace logicalWidth publicFits).state =
      OutputBinding.finalState (outputInterface logicalWidth publicFits)
        outputWitnessStart := by
  have traced := compileActions_state_eq outputPhase outputRowStart
    outputWitnessStart
    ((outputInterface logicalWidth publicFits).initialState outputWitnessStart)
    (OutputBinding.actions (outputInterface logicalWidth publicFits)
      outputWitnessStart)
  exact traced.trans
    (OutputBinding.finalState_eq_compile
      (outputInterface logicalWidth publicFits) outputWitnessStart).symm

/-- Held compact output invocations imply the exact post-PiCCS transcript
predicate under the proved Spartan pullback. -/
theorem outputTrace_implies_spec
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (holds : ∀ current ∈ (outputTrace logicalWidth publicFits).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env) :
    OutputBinding.SpecHolds (outputInterface logicalWidth publicFits)
      outputWitnessStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  let interface := outputInterface logicalWidth publicFits
  let semantic := outputSemanticTrace logicalWidth publicFits
  have semanticEq : outputTrace logicalWidth publicFits = semantic :=
    outputTrace_eq_semantic logicalWidth publicFits
  have semanticHolds : ∀ current ∈ semantic.invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
    intro current member
    apply holds current
    rw [semanticEq]
    exact member
  have trace := compileActions_traceHolds outputPhase outputRowStart
    outputWitnessStart (interface.initialState outputWitnessStart)
    (outputActions logicalWidth publicFits) env (by
      change NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
    ((inputShapes logicalWidth publicFits relation).outputBinding
      outputWitnessStart).initialState
    (NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding.actions_affine
      interface outputWitnessStart
      ((inputShapes logicalWidth publicFits relation).outputBinding
        outputWitnessStart))
    (expectedSamples_eq_samples_of_assertionCount_zero outputWitnessStart
      (interface.initialState outputWitnessStart)
      (outputActions logicalWidth publicFits) rfl)
    semanticHolds
  have semanticState : semantic.state =
      OutputBinding.finalState interface outputWitnessStart :=
    outputSemanticTrace_state_matches logicalWidth publicFits
  change Formal.TraceHolds
      (List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        (interface.initialState outputWitnessStart)))
      ((outputActions logicalWidth publicFits).map
        (Formal.Action.eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)))
      (List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        semantic.state)) at trace
  rw [semanticState] at trace
  exact OutputBinding.trace_implies_specHolds interface outputWitnessStart
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) trace

def invocations
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List PermutationInvocation :=
  (statementTrace logicalWidth publicFits).invocations ++
    (challengeTrace logicalWidth publicFits).invocations ++
    (roundTrace logicalWidth publicFits).invocations ++
    (outputTrace logicalWidth publicFits).invocations

/-! ## Transcript assembly

This is the compact physical assembly boundary for the four transcript leaves.
It maps the one exported invocation list back to the four child predicates and
does not unfold a child's circuit operations.
-/

/-- The four transcript conjuncts supplied to the canonical PiCCS parent. -/
structure TranscriptSpecs
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Env) : Prop where
  statementAbsorption :
    StatementAbsorption.SpecHolds
      (statementInterface logicalWidth publicFits) statementWitnessStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  challengeDerivation :
    ChallengeDerivation.SpecHolds
      (challengeInterface logicalWidth publicFits) challengeWitnessStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  roundTranscript :
    RoundTranscript.SpecHolds
      (roundInterface logicalWidth publicFits) roundWitnessStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  outputBinding :
    OutputBinding.SpecHolds
      (outputInterface logicalWidth publicFits) outputWitnessStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)

theorem TranscriptSpecs.statementAbsorption_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env} (specs : TranscriptSpecs logicalWidth publicFits env) :
    (Formal.statementAbsorptionCircuit
      (sharedInterface logicalWidth publicFits)).spec
      (Formal.statementAbsorptionOffset
        (parentInterface logicalWidth publicFits) phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  change StatementAbsorption.SpecHolds
    (statementInterface logicalWidth publicFits)
    (Formal.statementAbsorptionOffset
      (parentInterface logicalWidth publicFits) phaseOffset)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  rw [← statementWitnessStart_matches]
  exact specs.statementAbsorption

theorem TranscriptSpecs.challengeDerivation_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env} (specs : TranscriptSpecs logicalWidth publicFits env) :
    (Formal.challengeCircuit (parentInterface logicalWidth publicFits)
      phaseOffset).spec
      (Formal.challengeOffset (parentInterface logicalWidth publicFits)
        phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  change ChallengeDerivation.SpecHolds
    (challengeInterface logicalWidth publicFits)
    (Formal.challengeOffset (parentInterface logicalWidth publicFits)
      phaseOffset)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  rw [← challengeWitnessStart_matches]
  exact specs.challengeDerivation

theorem TranscriptSpecs.roundTranscript_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env} (specs : TranscriptSpecs logicalWidth publicFits env) :
    (Formal.roundTranscriptCircuit
      (sharedInterface logicalWidth publicFits)).spec
      (Formal.roundTranscriptOffset
        (parentInterface logicalWidth publicFits) phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  change RoundTranscript.SpecHolds
    (roundInterface logicalWidth publicFits)
    (Formal.roundTranscriptOffset
      (parentInterface logicalWidth publicFits) phaseOffset)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  rw [← roundWitnessStart_matches]
  exact specs.roundTranscript

theorem TranscriptSpecs.outputBinding_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    {env : Env} (specs : TranscriptSpecs logicalWidth publicFits env) :
    (Formal.outputBindingCircuit
      (sharedInterface logicalWidth publicFits)).spec
      (Formal.outputBindingOffset relation
        (parentInterface logicalWidth publicFits) phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  change OutputBinding.SpecHolds
    (outputInterface logicalWidth publicFits)
    (Formal.outputBindingOffset relation
      (parentInterface logicalWidth publicFits) phaseOffset)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  rw [← outputWitnessStart_matches logicalWidth publicFits relation]
  exact specs.outputBinding

/-- Satisfaction of the one canonical appended invocation list covers every
transcript child exactly once. -/
theorem invocations_imply_transcriptSpecs
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (holds : ∀ current ∈ invocations logicalWidth publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env) :
    TranscriptSpecs logicalWidth publicFits env := by
  refine {
    statementAbsorption := statementTrace_implies_spec logicalWidth publicFits
      relation env ?_
    challengeDerivation := challengeTrace_implies_spec logicalWidth publicFits
      relation env ?_
    roundTranscript := roundTrace_implies_spec logicalWidth publicFits relation
      env ?_
    outputBinding := outputTrace_implies_spec logicalWidth publicFits relation
      env ?_ }
  · intro current member
    apply holds current
    unfold invocations
    simp only [List.mem_append]
    exact Or.inl (Or.inl (Or.inl member))
  · intro current member
    apply holds current
    unfold invocations
    simp only [List.mem_append]
    exact Or.inl (Or.inl (Or.inr member))
  · intro current member
    apply holds current
    unfold invocations
    simp only [List.mem_append]
    exact Or.inl (Or.inr member)
  · intro current member
    apply holds current
    unfold invocations
    simp only [List.mem_append]
    exact Or.inr member

theorem statementInvocations_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (statementTrace logicalWidth publicFits).invocations.length = 325 := by
  rw [statementTrace, compileActions_invocations_length]
  have compiled := recipeCount_eq_invocationCount_mul
    (statementActions logicalWidth publicFits)
  have fixed := StatementAbsorption.recipeCount_eq
    (statementInterface logicalWidth publicFits) statementWitnessStart
  change Formal.recipeCount (statementActions logicalWidth publicFits) =
    192400 at fixed
  rw [fixed] at compiled
  omega

theorem challengeInvocations_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (challengeTrace logicalWidth publicFits).invocations.length = 87 := by
  rw [challengeTrace, compileActions_invocations_length]
  have sameCount := invocationCount_eq_of_shapes
    (ChallengeDerivation.actions (challengeInterface logicalWidth publicFits)
      challengeWitnessStart)
    (challengeActions logicalWidth publicFits)
    (challengeActions_shape_matches logicalWidth publicFits).symm
  have compiled := recipeCount_eq_invocationCount_mul
    (ChallengeDerivation.actions (challengeInterface logicalWidth publicFits)
      challengeWitnessStart)
  have fixed := ChallengeDerivation.recipeCount_eq
    (challengeInterface logicalWidth publicFits) challengeWitnessStart
  change Formal.recipeCount
    (ChallengeDerivation.actions (challengeInterface logicalWidth publicFits)
      challengeWitnessStart) = 51504 at fixed
  rw [fixed] at compiled
  rw [← sameCount]
  omega

theorem roundInvocations_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (roundTrace logicalWidth publicFits).invocations.length = 252 := by
  rw [roundTrace, compileActions_invocations_length]
  have sameCount := invocationCount_eq_of_shapes
    (RoundTranscript.actions (roundInterface logicalWidth publicFits)
      roundWitnessStart)
    (roundActions logicalWidth publicFits)
    (roundActions_shape_matches logicalWidth publicFits).symm
  have compiled := recipeCount_eq_invocationCount_mul
    (RoundTranscript.actions (roundInterface logicalWidth publicFits)
      roundWitnessStart)
  have fixed := RoundTranscript.recipeCount_eq
    (roundInterface logicalWidth publicFits) roundWitnessStart
  change Formal.recipeCount
    (RoundTranscript.actions (roundInterface logicalWidth publicFits)
      roundWitnessStart) =
      productionShape.cubeVariables *
        RoundTranscript.perRoundRecipeCount 9 at fixed
  norm_num [RoundTranscript.perRoundRecipeCount, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables] at fixed
  rw [fixed] at compiled
  rw [← sameCount]
  omega

theorem outputInvocations_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (outputTrace logicalWidth publicFits).invocations.length = 6886 := by
  rw [outputTrace, compileActions_invocations_length]
  have compiled := recipeCount_eq_invocationCount_mul
    (outputActions logicalWidth publicFits)
  have fixed := OutputBinding.recipeCount_eq
    (outputInterface logicalWidth publicFits) outputWitnessStart
  change Formal.recipeCount (outputActions logicalWidth publicFits) =
    4076512 at fixed
  rw [fixed] at compiled
  omega

theorem invocations_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (invocations logicalWidth publicFits).length = 7550 := by
  unfold invocations
  rw [List.length_append, List.length_append, List.length_append,
    statementInvocations_length, challengeInvocations_length,
    roundInvocations_length, outputInvocations_length]

/-! ## Constructive invocation schedule -/

/-- Final private boundary of the four compact transcript packets. It is the
mapped start of the generic R1CS-fresh region, not a new layout owner. -/
def invocationCeiling : Nat :=
  NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase
theorem invocationCeiling_eq : invocationCeiling = 18270868 := by
  norm_num [invocationCeiling,
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
    NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
    NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
    NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
    NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
theorem invocationCeiling_le_private :
    invocationCeiling ≤
      NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount := by
  rw [invocationCeiling_eq]
  norm_num [NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount]
theorem statementInvocationCount_eq (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    invocationCount (statementActions logicalWidth publicFits) = 325 := by
  have count := statementInvocations_length logicalWidth publicFits
  rw [statementTrace, compileActions_invocations_length] at count
  exact count
theorem challengeInvocationCount_eq (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    invocationCount (challengeActions logicalWidth publicFits) = 87 := by
  have count := challengeInvocations_length logicalWidth publicFits
  rw [challengeTrace, compileActions_invocations_length] at count
  exact count
theorem roundInvocationCount_eq (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    invocationCount (roundActions logicalWidth publicFits) = 252 := by
  have count := roundInvocations_length logicalWidth publicFits
  rw [roundTrace, compileActions_invocations_length] at count
  exact count
theorem outputInvocationCount_eq (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    invocationCount (outputActions logicalWidth publicFits) = 6886 := by
  have count := outputInvocations_length logicalWidth publicFits
  rw [outputTrace, compileActions_invocations_length] at count
  exact count
theorem statementEnd_eq_challengeStart (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    statementWitnessStart +
        invocationCount (statementActions logicalWidth publicFits) * 592 =
      challengeWitnessStart := by
  rw [statementInvocationCount_eq]
  unfold statementWitnessStart challengeWitnessStart
  rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
theorem challengeEnd_eq_roundStart (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    challengeWitnessStart +
        invocationCount (challengeActions logicalWidth publicFits) * 592 =
      roundWitnessStart := by
  rw [challengeInvocationCount_eq]
  unfold challengeWitnessStart roundWitnessStart
  rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
theorem roundEnd_lt_outputStart (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    roundWitnessStart +
        invocationCount (roundActions logicalWidth publicFits) * 592 <
      outputWitnessStart := by
  rw [roundInvocationCount_eq]
  unfold roundWitnessStart outputWitnessStart
  rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num
theorem outputEnd_eq_logicalFreshBase (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    outputWitnessStart +
        invocationCount (outputActions logicalWidth publicFits) * 592 =
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase := by
  rw [outputInvocationCount_eq]
  unfold outputWitnessStart
  rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num [NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
theorem statementTrace_scheduleWithin (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ScheduleWithin
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          statementWitnessStart)
        invocationCeiling
        (statementTrace logicalWidth publicFits).invocations ∧
      InvocationsBefore
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          challengeWitnessStart)
        (statementTrace logicalWidth publicFits).invocations := by
  let external :=
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
      logicalWidth publicFits
  let transcript :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.transcript relation
      (parentInterface logicalWidth publicFits) phaseOffset external
      (fun _ => 0)
  have startMatch := statementWitnessStart_matches logicalWidth publicFits
  have strongAffine :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption.actions_affine
      (statementInterface logicalWidth publicFits) statementWitnessStart
      ((inputShapes logicalWidth publicFits relation).statementAbsorption
        statementWitnessStart)
  have actionsAffine := actionsInvocationInputsAffine_of_actionsAffine
    (statementActions logicalWidth publicFits) strongAffine
  have boundedAtFormal := actionsInvocationInputsBelow_of_actionsBelow
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementAbsorptionOffset
      (parentInterface logicalWidth publicFits) phaseOffset)
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.actions
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementAbsorptionInterface
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.atOffset
          (parentInterface logicalWidth publicFits) phaseOffset))
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementAbsorptionOffset
        (parentInterface logicalWidth publicFits) phaseOffset))
    transcript.statementAbsorption
  have actionsBelow : ActionsInvocationInputsBelow statementWitnessStart
      (statementActions logicalWidth publicFits) := by
    simpa [statementActions, statementInterface, sharedInterface,
      startMatch] using boundedAtFormal
  have endLocal : NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
      challengeWitnessStart := by
    unfold challengeWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have endStrict : challengeWitnessStart <
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase := by
    unfold challengeWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  have endWithin : NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (statementWitnessStart +
        invocationCount (statementActions logicalWidth publicFits) * 592) ≤
      invocationCeiling := by
    rw [statementEnd_eq_challengeStart]
    exact (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      challengeWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase endLocal
      endStrict).le
  have scheduled := compileActions_scheduleWithin statementPhase
    statementRowStart statementWitnessStart invocationCeiling Hash.zeroE
    (statementActions logicalWidth publicFits) (by
      change NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
    invocationCeiling_le_private endWithin
    NightstreamFPrime.Layout.Poseidon2.zeroE_affine (by
      intro lane
      simp [Hash.zeroE, Expr.VarsBelow]) actionsAffine actionsBelow
  rw [statementEnd_eq_challengeStart] at scheduled
  exact scheduled
theorem challengeTrace_scheduleWithin (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ScheduleWithin
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          challengeWitnessStart)
        invocationCeiling
        (challengeTrace logicalWidth publicFits).invocations ∧
      InvocationsBefore
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          roundWitnessStart)
        (challengeTrace logicalWidth publicFits).invocations := by
  let external :=
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
      logicalWidth publicFits
  let transcript :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.transcript relation
      (parentInterface logicalWidth publicFits) phaseOffset external
      (fun _ => 0)
  let interface := challengeInterface logicalWidth publicFits
  have initialEq : (statementTrace logicalWidth publicFits).state =
      interface.initialState challengeWitnessStart := by
    exact (statementTrace_state_matches logicalWidth publicFits).trans
      (challengeInitialState_eq_statementFinalState logicalWidth publicFits).symm
  have stateAffine : NightstreamFPrime.Layout.Poseidon2.StateAffine
      (statementTrace logicalWidth publicFits).state := by
    rw [initialEq]
    exact ((inputShapes logicalWidth publicFits relation).challengeDerivation
      challengeWitnessStart).initialState
  have stateBelowAtFormal := transcript.challenge
  have stateBelowSemantic : ∀ lane,
      (interface.initialState challengeWitnessStart lane).VarsBelow
        challengeWitnessStart := by
    have startMatch := challengeWitnessStart_matches logicalWidth publicFits
    simpa [interface, challengeInterface, startMatch] using stateBelowAtFormal
  have stateBelow : ∀ lane,
      ((statementTrace logicalWidth publicFits).state lane).VarsBelow
        challengeWitnessStart := by
    rw [initialEq]
    exact stateBelowSemantic
  have semanticStrongAffine :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation.actions_affine
      interface challengeWitnessStart
      ((inputShapes logicalWidth publicFits relation).challengeDerivation
        challengeWitnessStart)
  have semanticAffine := actionsInvocationInputsAffine_of_actionsAffine
    (ChallengeDerivation.actions interface challengeWitnessStart)
    semanticStrongAffine
  have actionsAffine := actionsInvocationInputsAffine_of_shapes
    (ChallengeDerivation.actions interface challengeWitnessStart)
    (challengeActions logicalWidth publicFits)
    (challengeActions_shape_matches logicalWidth publicFits).symm
    semanticAffine
  have actionsBelow := actionsInvocationInputsBelow_of_actionsBelow
    challengeWitnessStart (challengeActions logicalWidth publicFits)
    (ChallengeDerivation.layoutActions_below challengeWitnessStart)
  have endLocal : NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
      roundWitnessStart := by
    unfold roundWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have endStrict : roundWitnessStart <
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase := by
    unfold roundWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  have endWithin : NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (challengeWitnessStart +
        invocationCount (challengeActions logicalWidth publicFits) * 592) ≤
      invocationCeiling := by
    rw [challengeEnd_eq_roundStart]
    exact (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      roundWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase endLocal
      endStrict).le
  have scheduled := compileActions_scheduleWithin challengePhase
    challengeRowStart challengeWitnessStart invocationCeiling
    (statementTrace logicalWidth publicFits).state
    (challengeActions logicalWidth publicFits) (by
      change NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
    invocationCeiling_le_private endWithin stateAffine stateBelow actionsAffine
    actionsBelow
  rw [challengeEnd_eq_roundStart] at scheduled
  exact scheduled
theorem roundTrace_scheduleWithin (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ScheduleWithin
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          roundWitnessStart)
        invocationCeiling
        (roundTrace logicalWidth publicFits).invocations ∧
      InvocationsBefore
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          (roundWitnessStart +
            invocationCount (roundActions logicalWidth publicFits) * 592))
        (roundTrace logicalWidth publicFits).invocations := by
  let external :=
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
      logicalWidth publicFits
  let transcript :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.transcript relation
      (parentInterface logicalWidth publicFits) phaseOffset external
      (fun _ => 0)
  let interface := roundInterface logicalWidth publicFits
  have initialEq : (challengeTrace logicalWidth publicFits).state =
      interface.initialState roundWitnessStart := by
    simpa [interface, roundInterface, Formal.roundTranscriptInterface,
      Formal.challengeFinalState, sharedInterface] using
      challengeTrace_state_matches logicalWidth publicFits
  have stateAffine : NightstreamFPrime.Layout.Poseidon2.StateAffine
      (challengeTrace logicalWidth publicFits).state := by
    rw [initialEq]
    exact ((inputShapes logicalWidth publicFits relation).roundTranscript
      roundWitnessStart).initialState
  have stateBelowSemantic : ∀ lane,
      (interface.initialState roundWitnessStart lane).VarsBelow
        roundWitnessStart := by
    have startMatch := roundWitnessStart_matches logicalWidth publicFits
    have below := transcript.roundTranscript.1
    simpa [interface, roundInterface, startMatch] using below
  have stateBelow : ∀ lane,
      ((challengeTrace logicalWidth publicFits).state lane).VarsBelow
        roundWitnessStart := by
    rw [initialEq]
    exact stateBelowSemantic
  have semanticStrongAffine :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript.actions_affine
      interface roundWitnessStart
      ((inputShapes logicalWidth publicFits relation).roundTranscript
        roundWitnessStart)
  have semanticAffine := actionsInvocationInputsAffine_of_actionsAffine
    (RoundTranscript.actions interface roundWitnessStart) semanticStrongAffine
  have actionsAffine := actionsInvocationInputsAffine_of_shapes
    (RoundTranscript.actions interface roundWitnessStart)
    (roundActions logicalWidth publicFits)
    (roundActions_shape_matches logicalWidth publicFits).symm semanticAffine
  have roundAssumptions : RoundTranscript.Assumptions interface
      roundWitnessStart (fun _ => 0) := by
    have startMatch := roundWitnessStart_matches logicalWidth publicFits
    simpa [interface, roundInterface, startMatch] using
      transcript.roundTranscript
  have strongBelow := RoundTranscript.layoutActions_below interface
    roundWitnessStart roundAssumptions
  have actionsBelow := actionsInvocationInputsBelow_of_actionsBelow
    roundWitnessStart (roundActions logicalWidth publicFits) strongBelow
  have endSourceLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        roundWitnessStart +
          invocationCount (roundActions logicalWidth publicFits) * 592 := by
    have roundLocal :
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
          roundWitnessStart := by
      unfold roundWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
    omega
  have outputStrict : outputWitnessStart <
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase := by
    unfold outputWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  have endStrict : roundWitnessStart +
      invocationCount (roundActions logicalWidth publicFits) * 592 <
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase :=
    lt_trans (roundEnd_lt_outputStart logicalWidth publicFits) outputStrict
  have endWithin : NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (roundWitnessStart +
        invocationCount (roundActions logicalWidth publicFits) * 592) ≤
      invocationCeiling :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      _ NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase
      endSourceLocal endStrict).le
  have scheduled := compileActions_scheduleWithin roundPhase roundRowStart
    roundWitnessStart invocationCeiling
    (challengeTrace logicalWidth publicFits).state
    (roundActions logicalWidth publicFits) (by
      change NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
    invocationCeiling_le_private endWithin stateAffine stateBelow actionsAffine
    actionsBelow
  exact scheduled
theorem outputTrace_scheduleWithin (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ScheduleWithin
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          outputWitnessStart)
        invocationCeiling
        (outputTrace logicalWidth publicFits).invocations ∧
      InvocationsBefore invocationCeiling
        (outputTrace logicalWidth publicFits).invocations := by
  let external :=
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
      logicalWidth publicFits
  let interface := outputInterface logicalWidth publicFits
  have initialEq : (roundTrace logicalWidth publicFits).state =
      interface.initialState outputWitnessStart := by
    simpa [interface, outputInterface, Formal.outputBindingInterface,
      Formal.roundTranscriptFinalState, sharedInterface] using
      roundTrace_state_matches logicalWidth publicFits
  have stateAffine : NightstreamFPrime.Layout.Poseidon2.StateAffine
      (roundTrace logicalWidth publicFits).state := by
    rw [initialEq]
    exact ((inputShapes logicalWidth publicFits relation).outputBinding
      outputWitnessStart).initialState
  have outputAssumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.outputBinding relation
      (parentInterface logicalWidth publicFits) phaseOffset external
      (fun _ => 0)
  have outputInputsBelow :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.outputBindingInputsBelow
      relation (parentInterface logicalWidth publicFits) phaseOffset external
      (fun _ => 0)
  have stateBelowSemantic : ∀ lane,
      (interface.initialState outputWitnessStart lane).VarsBelow
        outputWitnessStart := by
    have startMatch := outputWitnessStart_matches logicalWidth publicFits
      relation
    have interfaceMatch : interface =
        Formal.outputBindingInterface
          (Formal.atOffset (parentInterface logicalWidth publicFits)
            phaseOffset) := by
      rfl
    rw [interfaceMatch, startMatch]
    exact outputInputsBelow.initialState
  have stateBelow : ∀ lane,
      ((roundTrace logicalWidth publicFits).state lane).VarsBelow
        outputWitnessStart := by
    rw [initialEq]
    exact stateBelowSemantic
  have strongAffine :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding.actions_affine
      interface outputWitnessStart
      ((inputShapes logicalWidth publicFits relation).outputBinding
        outputWitnessStart)
  have actionsAffine := actionsInvocationInputsAffine_of_actionsAffine
    (outputActions logicalWidth publicFits) strongAffine
  have strongBelowAtFormal := outputAssumptions.2
  have strongBelow : Formal.ActionsBelow outputWitnessStart
      (outputActions logicalWidth publicFits) := by
    have startMatch := outputWitnessStart_matches logicalWidth publicFits
      relation
    simpa [interface, outputInterface, outputActions, startMatch] using
      strongBelowAtFormal
  have actionsBelow := actionsInvocationInputsBelow_of_actionsBelow
    outputWitnessStart (outputActions logicalWidth publicFits) strongBelow
  have endWithin : NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (outputWitnessStart +
        invocationCount (outputActions logicalWidth publicFits) * 592) ≤
      invocationCeiling := by
    rw [outputEnd_eq_logicalFreshBase]
    exact le_rfl
  have scheduled := compileActions_scheduleWithin outputPhase outputRowStart
    outputWitnessStart invocationCeiling
    (roundTrace logicalWidth publicFits).state
    (outputActions logicalWidth publicFits) (by
      change NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart
      rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
    invocationCeiling_le_private endWithin stateAffine stateBelow actionsAffine
    actionsBelow
  rw [outputEnd_eq_logicalFreshBase] at scheduled
  simpa [invocationCeiling] using scheduled
/-- The one production PiCCS invocation list is the ordered composition of
the four transcript packets. The proof uses only child schedule and footprint
theorems; it does not inspect any child invocation. -/
theorem invocations_scheduleWithin (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ScheduleWithin
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          statementWitnessStart)
        invocationCeiling (invocations logicalWidth publicFits) ∧
      InvocationsBefore invocationCeiling
        (invocations logicalWidth publicFits) := by
  have statement := statementTrace_scheduleWithin logicalWidth publicFits
    relation
  have challenge := challengeTrace_scheduleWithin logicalWidth publicFits
    relation
  have rounds := roundTrace_scheduleWithin logicalWidth publicFits relation
  have output := outputTrace_scheduleWithin logicalWidth publicFits relation
  have statementLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        statementWitnessStart := by
    unfold statementWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have challengeLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        challengeWitnessStart := by
    unfold challengeWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have roundLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        roundWitnessStart := by
    unfold roundWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have outputLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        outputWitnessStart := by
    unfold outputWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have statementStrict : statementWitnessStart < challengeWitnessStart := by
    calc
      statementWitnessStart < statementWitnessStart +
          invocationCount (statementActions logicalWidth publicFits) * 592 := by
        rw [statementInvocationCount_eq]
        omega
      _ = challengeWitnessStart :=
        statementEnd_eq_challengeStart logicalWidth publicFits
  have challengeStrict : challengeWitnessStart < roundWitnessStart := by
    calc
      challengeWitnessStart < challengeWitnessStart +
          invocationCount (challengeActions logicalWidth publicFits) * 592 := by
        rw [challengeInvocationCount_eq]
        omega
      _ = roundWitnessStart :=
        challengeEnd_eq_roundStart logicalWidth publicFits
  have roundStrict : roundWitnessStart < outputWitnessStart := by
    calc
      roundWitnessStart < roundWitnessStart +
          invocationCount (roundActions logicalWidth publicFits) * 592 := by
        rw [roundInvocationCount_eq]
        omega
      _ < outputWitnessStart :=
        roundEnd_lt_outputStart logicalWidth publicFits
  have outputStrict : outputWitnessStart <
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase := by
    calc
      outputWitnessStart < outputWitnessStart +
          invocationCount (outputActions logicalWidth publicFits) * 592 := by
        rw [outputInvocationCount_eq]
        omega
      _ = NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase :=
        outputEnd_eq_logicalFreshBase logicalWidth publicFits
  have statementToChallenge :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          statementWitnessStart ≤
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          challengeWitnessStart :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      statementWitnessStart challengeWitnessStart statementLocal
      statementStrict).le
  have challengeToRound :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          challengeWitnessStart ≤
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          roundWitnessStart :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      challengeWitnessStart roundWitnessStart challengeLocal
      challengeStrict).le
  have roundToOutput :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          roundWitnessStart ≤
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          outputWitnessStart :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      roundWitnessStart outputWitnessStart roundLocal roundStrict).le
  have outputToCeiling :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          outputWitnessStart ≤ invocationCeiling := by
    exact (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      outputWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase outputLocal
      outputStrict).le
  have statementChallengeSchedule := ScheduleWithin.append statement.1
    statement.2 statementToChallenge challenge.1
  have statementBeforeRound := InvocationsBefore.mono statement.2
    challengeToRound
  have statementChallengeBeforeRound := InvocationsBefore.append
    statementBeforeRound challenge.2
  have statementToRound := Nat.le_trans statementToChallenge challengeToRound
  have prefixRoundSchedule := ScheduleWithin.append
    statementChallengeSchedule statementChallengeBeforeRound statementToRound
    rounds.1
  have statementChallengeBeforeOutput := InvocationsBefore.mono
    statementChallengeBeforeRound roundToOutput
  have roundsBeforeOutput := InvocationsBefore.mono rounds.2
    ((NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      _ outputWitnessStart (by omega)
      (roundEnd_lt_outputStart logicalWidth publicFits)).le)
  have prefixRoundBeforeOutput := InvocationsBefore.append
    statementChallengeBeforeOutput roundsBeforeOutput
  have statementToOutput := Nat.le_trans statementToRound roundToOutput
  have prefixOutputSchedule := ScheduleWithin.append prefixRoundSchedule
    prefixRoundBeforeOutput statementToOutput output.1
  have prefixRoundBeforeCeiling := InvocationsBefore.mono
    prefixRoundBeforeOutput outputToCeiling
  have allBefore := InvocationsBefore.append prefixRoundBeforeCeiling output.2
  constructor
  · simpa [invocations, List.append_assoc] using prefixOutputSchedule
  · simpa [invocations, List.append_assoc] using allBefore
end NightstreamFPrime.Export.Stage1.PiCCSInvocations
