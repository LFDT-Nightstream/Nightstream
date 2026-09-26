import NightstreamFPrime.Layout.Stage1.PiCCSProtocolCompleteness
import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics

/-! PiCCS statement, proof and outgoing-transcript views used by the next phase. -/

namespace NightstreamFPrime.Layout.Stage1.PiCCSBoundary

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiCCSProofInputs (relationInterface relationProof)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

private theorem interface_eq : relationInterface relation =
    AccumulatorInputs.piCcsInterface logicalWidth publicFits := by
  rfl

theorem cViews_eq_of_fields
    (key : ProductionKey.KeyType relation)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (left right : Proof (ProductionKey.degreeBound relation))
    (rounds : left.piCcsRounds = right.piCcsRounds)
    (output : left.piCcsOutput = right.piCcsOutput) :
    key.piCcsExecution running fresh left = key.piCcsExecution running fresh right ∧
    key.piCcsOutputs running fresh left = key.piCcsOutputs running fresh right := by
  cases left
  cases right
  cases rounds
  cases output
  exact ⟨rfl, rfl⟩

private theorem input_readback
    (left right : Env) (template : Proof (ProductionKey.degreeBound relation))
    (agrees : ∀ index, index < PiCCSInputs.phaseOffset → left index = right index) :
    Formal.evalRunning (relationInterface relation) PiCCSInputs.phaseOffset left =
      Formal.evalRunning (relationInterface relation) PiCCSInputs.phaseOffset right ∧
    Formal.evalFresh (relationInterface relation) PiCCSInputs.phaseOffset left =
      Formal.evalFresh (relationInterface relation) PiCCSInputs.phaseOffset right ∧
    Formal.evalProof relation (relationInterface relation) PiCCSInputs.phaseOffset left template =
      Formal.evalProof relation (relationInterface relation) PiCCSInputs.phaseOffset right template := by
  have below : Formal.ExternalInputsBelow (relationInterface relation) PiCCSInputs.phaseOffset := by
    rw [interface_eq relation]
    exact PiCCSInputs.externalInputsBelow logicalWidth publicFits
  exact ⟨Formal.CompletenessSupport.evalRunning_eq_of_agree_below _ _ _ _ below agrees,
    Formal.CompletenessSupport.evalFresh_eq_of_agree_below _ _ _ _ below agrees,
    Formal.CompletenessSupport.evalProof_eq_of_agree_below relation _ _ _ _ template below agrees⟩

theorem initialState_eq_of_phase
    (env : Env) (template : Proof (ProductionKey.degreeBound relation))
    (phase : Formal.PhaseHolds relation ajtai (relationInterface relation)
      PiCCSInputs.phaseOffset env template) :
    PiRLC.v1_1.SamplerChain.evalInitialState
      (PiRLC.v1_1.Formal.samplerInterface (PiRLC.v1_1.Formal.atOffset
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset)) (PiRLC.v1_1.Formal.samplerOffset PiRLCInputs.phaseOffset) env =
    ((ProductionKey.key relation ajtai).piCcsExecution
      (Formal.evalRunning (relationInterface relation) PiCCSInputs.phaseOffset env)
      (Formal.evalFresh (relationInterface relation) PiCCSInputs.phaseOffset env)
      (Formal.evalProof relation (relationInterface relation) PiCCSInputs.phaseOffset env template)).outgoingState := by
  calc
    _ = StatementAbsorption.evalState env
        (PiRLCInputs.piCcsOutputState (logicalWidth := logicalWidth) (publicFits := publicFits)) := rfl
    _ = StatementAbsorption.evalState env
        (Formal.outputBindingFinalState relation (relationInterface relation) PiCCSInputs.phaseOffset) := by
      rw [PiRLCInputs.piCcsOutputState_eq_parent relation, interface_eq relation]
      rfl
    _ = _ := phase.outgoingState

theorem accumulator_phase
    (env : Env) (template : Proof (ProductionKey.degreeBound relation))
    (phase : Formal.PhaseHolds relation ajtai (relationInterface relation)
      PiCCSInputs.phaseOffset env template) :
    Formal.PhaseHolds relation ajtai (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env (AccumulatorInputs.proof relation env) := by
  rw [interface_eq relation] at phase
  let running := AccumulatorInputs.running logicalWidth publicFits env
  let fresh := AccumulatorInputs.fresh logicalWidth publicFits env
  let evaluated := Formal.evalProof relation
    (AccumulatorInputs.piCcsInterface logicalWidth publicFits) PiCCSInputs.phaseOffset env template
  have views := cViews_eq_of_fields relation (ProductionKey.key relation ajtai) running fresh
    evaluated (AccumulatorInputs.proof relation env) rfl rfl
  refine ⟨phase.stateBinding, ?_, ?_, ?_⟩
  · change Folding.Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai)
      running fresh (AccumulatorInputs.proof relation env) = true
    rw [← AccumulatorSemantics.piCcsCheck_eq_of_proof_fields relation ajtai
      running fresh evaluated (AccumulatorInputs.proof relation env) rfl rfl]
    exact phase.accepted
  · exact phase.roundPoint.trans (congrArg (fun execution => execution.coins.roundPoint) views.1)
  · exact phase.outgoingState.trans (congrArg (fun execution => execution.outgoingState) views.1)

variable
  (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (priorPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
  (output : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (digest : Digest)
  (priorFixed : PilotProduction.FixedPreimage prior)
  (outputFixed : PilotProduction.FixedPreimage output)
  (digestFixed : digest.length = PilotProduction.digestWords)
  (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4)
  (template : Proof 9)

theorem protocol_readback
    (initial : Env)
    (source : ∀ index, PiCCSOrdinarySourceSupport.External index → initial index =
      PiCCSProtocolCompleteness.environment prior priorPublic output digest
        priorFixed outputFixed digestFixed values context index)
    (env : Env)
    (agrees : ∀ index, index < PiCCSInputs.phaseOffset → env index = initial index) :
    Formal.evalRunning (relationInterface relation) PiCCSInputs.phaseOffset env =
      prior.running functionIndex ∧
    Formal.evalFresh (relationInterface relation) PiCCSInputs.phaseOffset env =
      PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values ∧
    Formal.evalProof relation (relationInterface relation) PiCCSInputs.phaseOffset env
      (relationProof relation values template) = relationProof relation values template := by
  have initialRead := PiCCSProtocolCompleteness.inputs_eq_of_external
    prior priorPublic output digest priorFixed outputFixed digestFixed values context
    relation template initial source
  have preserved := input_readback relation env initial (relationProof relation values template) agrees
  exact ⟨preserved.1.trans initialRead.1, preserved.2.1.trans initialRead.2.1,
    preserved.2.2.trans initialRead.2.2⟩

end NightstreamFPrime.Layout.Stage1.PiCCSBoundary
