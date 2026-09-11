import NightstreamFPrime.Layout.Stage1.PiCCSProtocolCompleteness
import NightstreamFPrime.Layout.Stage1.PiRLCInputBounds
import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics

/-!
Owns construction of the canonical local C/R witnesses from protocol inputs.
The C constructor supplies its transcript state; the R constructor consumes
actual bounded sampler availability at that state. Existing source support
preserves C through R, and the resulting R output is the production parent.
This module adds no rows and does not construct the pilot or physical lowering.
-/

namespace NightstreamFPrime.Layout.Stage1.PiRLCProtocolCompleteness

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

private theorem cViews_eq_of_fields
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

private theorem initialState_eq_of_phase
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

private theorem accumulator_phase
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

private theorem protocol_readback
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

/-- Continue from the already constructed C prefix after completing its
physical fresh interval. Agreement is needed only below the logical C end;
the constructor derives C semantics, the actual sampler state, and R outputs.
No equality on the physical fresh cells is required. -/
theorem completePrefix_after_c
    (available : Folding.Nifs.NonInteractive.PiRlcSampler.Available
      Transcript.PiRlcSampler.specification PiRLC.v1_1.SamplerChain.sourceCount
      Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound
      ((ProductionKey.key relation ajtai).piCcsExecution (prior.running functionIndex)
        (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
        (relationProof relation values template)).outgoingState)
    (initial : Env)
    (source : ∀ index, PiCCSOrdinarySourceSupport.External index → initial index =
      PiCCSProtocolCompleteness.environment prior priorPublic output digest
        priorFixed outputFixed digestFixed values context index)
    (c : Sequence.Prefix initial PiCCSInputs.phaseOffset)
    (cOperations : c.operations = Formal.opsAt relation (relationInterface relation) PiCCSInputs.phaseOffset)
    (afterC : Env)
    (preserved : ∀ index, index < PiCCSInputs.phaseOffset + localLength c.operations →
      afterC index = c.current index) :
    ∃ r : Sequence.Prefix afterC PiRLCInputs.phaseOffset,
        r.operations = PiRLC.v1_1.Formal.opsAt relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset ∧
        holdsFlat r.current c.operations ∧
        PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset r.current ∧
        (ProductionKey.key relation ajtai).piRlcChallenges (prior.running functionIndex)
          (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
          (relationProof relation values template) =
          some (PiRLC.v1_1.Semantics.evalChallenges
            (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
            PiRLCInputs.phaseOffset r.current) ∧
        PiRLC.v1_1.Semantics.evalOutput relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset r.current =
          (ProductionKey.key relation ajtai).parentForChallenges (prior.running functionIndex)
            (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
            (relationProof relation values template)
            (PiRLC.v1_1.Semantics.evalChallenges
              (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
              PiRLCInputs.phaseOffset r.current) := by
  have cRows : holds afterC
      (Circuit.ops (Formal.main relation (relationInterface relation)) PiCCSInputs.phaseOffset) := by
    rw [Formal.main_ops, ← cOperations]
    apply holdsFlat_implies_holds
    intro expression member
    exact (expression.eval_eq_of_agree_below
      (PiCCSInputs.phaseOffset + localLength c.operations) afterC c.current
      (c.scope expression member) preserved).trans (c.rows expression member)
  have cAssumptions := NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation
    (relationInterface relation) PiCCSInputs.phaseOffset
    (PiCCSInputs.externalInputsLinear logicalWidth publicFits) afterC
  have cPhase := Formal.spec_implies_phaseHolds relation ajtai (relationInterface relation)
    PiCCSInputs.phaseOffset afterC (relationProof relation values template)
    (Formal.soundness relation (relationInterface relation) afterC PiCCSInputs.phaseOffset cAssumptions cRows)
  have cRead := protocol_readback relation prior priorPublic output digest priorFixed outputFixed
    digestFixed values context template initial source afterC (fun index below =>
      (preserved index (Nat.lt_of_lt_of_le below (Nat.le_add_right _ _))).trans
        (c.agrees index (Or.inl below)))
  have stateEq := initialState_eq_of_phase relation ajtai afterC
    (relationProof relation values template) cPhase
  rw [cRead.1, cRead.2.1, cRead.2.2] at stateEq
  have actualAvailable : Folding.Nifs.NonInteractive.PiRlcSampler.Available
      Transcript.PiRlcSampler.specification PiRLC.v1_1.SamplerChain.sourceCount
      Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound
      (PiRLC.v1_1.SamplerChain.evalInitialState
        (PiRLC.v1_1.Formal.samplerInterface (PiRLC.v1_1.Formal.atOffset
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset)) (PiRLC.v1_1.Formal.samplerOffset PiRLCInputs.phaseOffset)
        afterC) := by
    rw [stateEq]
    exact available
  obtain ⟨r, rOperations, rPhase⟩ := PiRLC.v1_1.Formal.completePrefix_of_available relation ajtai
    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) afterC
    PiRLCInputs.phaseOffset (PiRLCInputBounds.assumptions relation afterC) actualAvailable
  have cLimit : PiCCSInputs.phaseOffset + localLength c.operations ≤ PiRLCInputs.phaseOffset := by
    rw [cOperations, ← Formal.main_ops, Formal.localLength_eq]
    change NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset ≤ _
    rw [← PiCCSStarts.logicalFreshBase_eq_layout relation]
    exact PiRLCInputs.piCcsLogicalFreshBase_le_phaseOffset
  have rowsPreserved : holdsFlat r.current c.operations := by
    intro expression member
    have equal := expression.eval_eq_of_agree_below
      (PiCCSInputs.phaseOffset + localLength c.operations) r.current c.current
      (c.scope expression member) (fun index below =>
        (r.agrees index (Or.inl (Nat.lt_of_lt_of_le below cLimit))).trans
          (preserved index below))
    exact equal.trans (c.rows expression member)
  have cRows : holds r.current
      (Circuit.ops (Formal.main relation (relationInterface relation)) PiCCSInputs.phaseOffset) := by
    rw [Formal.main_ops, ← cOperations]
    exact holdsFlat_implies_holds r.current c.operations rowsPreserved
  have cAssumptions := NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation
    (relationInterface relation) PiCCSInputs.phaseOffset
    (PiCCSInputs.externalInputsLinear logicalWidth publicFits) r.current
  have finalC := Formal.spec_implies_phaseHolds relation ajtai (relationInterface relation)
    PiCCSInputs.phaseOffset r.current (relationProof relation values template)
    (Formal.soundness relation (relationInterface relation) r.current PiCCSInputs.phaseOffset
      cAssumptions cRows)
  have finalRead := protocol_readback relation prior priorPublic output digest priorFixed outputFixed
    digestFixed values context template initial source r.current (fun index below =>
      (r.agrees index (Or.inl (by omega))).trans
        ((preserved index (Nat.lt_of_lt_of_le below (Nat.le_add_right _ _))).trans
          (c.agrees index (Or.inl below))))
  have finalState := initialState_eq_of_phase relation ajtai r.current
    (relationProof relation values template) finalC
  rw [finalRead.1, finalRead.2.1, finalRead.2.2] at finalState
  have challenges := AccumulatorSemantics.piRlcChallenges_eq_key_of_initialState relation ajtai
    r.current (prior.running functionIndex)
    (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
    (relationProof relation values template)
    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCInputs.phaseOffset rPhase finalState
  have canonicalC := accumulator_phase relation ajtai r.current
    (relationProof relation values template) finalC
  have inputs := AccumulatorSemantics.piRlcInputs_eq_keyOutputs relation ajtai r.current canonicalC
  have runningEq : AccumulatorInputs.running logicalWidth publicFits r.current =
      prior.running functionIndex := finalRead.1
  have freshEq : AccumulatorInputs.fresh logicalWidth publicFits r.current =
      PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values := finalRead.2.1
  rw [runningEq, freshEq] at inputs
  have roundsEq : (AccumulatorInputs.proof relation r.current).piCcsRounds =
      (relationProof relation values template).piCcsRounds :=
    congrArg (fun proof => proof.piCcsRounds) finalRead.2.2
  have outputEq : (AccumulatorInputs.proof relation r.current).piCcsOutput =
      (relationProof relation values template).piCcsOutput :=
    congrArg (fun proof => proof.piCcsOutput) finalRead.2.2
  have views := cViews_eq_of_fields relation (ProductionKey.key relation ajtai)
    (prior.running functionIndex) (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
    (AccumulatorInputs.proof relation r.current) (relationProof relation values template) roundsEq outputEq
  have parent := AccumulatorSemantics.piRlcOutput_eq_keyParentForChallenges_of_inputs relation ajtai
    r.current (prior.running functionIndex)
    (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
    (relationProof relation values template)
    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCInputs.phaseOffset rPhase (inputs.trans views.2)
  exact ⟨r, rOperations, rowsPreserved, rPhase, challenges, parent⟩


/-- Accepted typed protocol inputs and actual bounded sampler availability
construct the canonical local C/R witnesses. R preserves the constructed C
rows and its output is the exact production-key parent for the same statement,
proof, and transcript-derived challenges. No generated phase output is assumed. -/
theorem completePrefix_from
    (priorPc : prior.pc = 1) (outputPc : output.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (outputContext : output.verifierKeys functionIndex = context.toList)
    (accepted : Folding.PiCCS.Accepted (ProductionKey.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
      (relationProof relation values template))
    (available : Folding.Nifs.NonInteractive.PiRlcSampler.Available
      Transcript.PiRlcSampler.specification PiRLC.v1_1.SamplerChain.sourceCount
      Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound
      ((ProductionKey.key relation ajtai).piCcsExecution (prior.running functionIndex)
        (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
        (relationProof relation values template)).outgoingState)
    (initial : Env)
    (source : ∀ index, PiCCSOrdinarySourceSupport.External index → initial index =
      PiCCSProtocolCompleteness.environment prior priorPublic output digest
        priorFixed outputFixed digestFixed values context index) :
    ∃ c : Sequence.Prefix initial PiCCSInputs.phaseOffset,
      ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
        c.operations = Formal.opsAt relation (relationInterface relation) PiCCSInputs.phaseOffset ∧
        r.operations = PiRLC.v1_1.Formal.opsAt relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset ∧
        holdsFlat r.current c.operations ∧
        PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset r.current ∧
        (ProductionKey.key relation ajtai).piRlcChallenges (prior.running functionIndex)
          (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
          (relationProof relation values template) =
          some (PiRLC.v1_1.Semantics.evalChallenges
            (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
            PiRLCInputs.phaseOffset r.current) ∧
        PiRLC.v1_1.Semantics.evalOutput relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset r.current =
          (ProductionKey.key relation ajtai).parentForChallenges (prior.running functionIndex)
            (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
            (relationProof relation values template)
            (PiRLC.v1_1.Semantics.evalChallenges
              (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
              PiRLCInputs.phaseOffset r.current) := by
  obtain ⟨c, cOperations, _⟩ := PiCCSProtocolCompleteness.completePrefix_from
    prior priorPublic output digest priorFixed outputFixed digestFixed values context relation
    ajtai template priorPc outputPc priorContext outputContext accepted initial source
  obtain ⟨r, completed⟩ := completePrefix_after_c relation ajtai prior priorPublic output digest
    priorFixed outputFixed digestFixed values context template available initial source
    c cOperations c.current (fun _ _ => rfl)
  exact ⟨c, r, cOperations, completed⟩


/-- Accepted typed protocol inputs and actual bounded sampler availability
construct the canonical local C/R witnesses. R preserves the constructed C
rows and its output is the exact production-key parent for the same statement,
proof, and transcript-derived challenges. No generated phase output is assumed. -/
theorem completePrefix
    (priorPc : prior.pc = 1) (outputPc : output.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (outputContext : output.verifierKeys functionIndex = context.toList)
    (accepted : Folding.PiCCS.Accepted (ProductionKey.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
      (relationProof relation values template))
    (available : Folding.Nifs.NonInteractive.PiRlcSampler.Available
      Transcript.PiRlcSampler.specification PiRLC.v1_1.SamplerChain.sourceCount
      Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound
      ((ProductionKey.key relation ajtai).piCcsExecution (prior.running functionIndex)
        (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
        (relationProof relation values template)).outgoingState) :
    ∃ c : Sequence.Prefix
        (PiCCSProtocolCompleteness.environment prior priorPublic output digest
          priorFixed outputFixed digestFixed values context) PiCCSInputs.phaseOffset,
      ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
        c.operations = Formal.opsAt relation (relationInterface relation) PiCCSInputs.phaseOffset ∧
        r.operations = PiRLC.v1_1.Formal.opsAt relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset ∧
        holdsFlat r.current c.operations ∧
        PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset r.current ∧
        (ProductionKey.key relation ajtai).piRlcChallenges (prior.running functionIndex)
          (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
          (relationProof relation values template) =
          some (PiRLC.v1_1.Semantics.evalChallenges
            (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
            PiRLCInputs.phaseOffset r.current) ∧
        PiRLC.v1_1.Semantics.evalOutput relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset r.current =
          (ProductionKey.key relation ajtai).parentForChallenges (prior.running functionIndex)
            (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
            (relationProof relation values template)
            (PiRLC.v1_1.Semantics.evalChallenges
              (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
              PiRLCInputs.phaseOffset r.current) := by
  exact completePrefix_from relation ajtai prior priorPublic output digest priorFixed outputFixed
    digestFixed values context template priorPc outputPc priorContext outputContext accepted available
    (PiCCSProtocolCompleteness.environment prior priorPublic output digest
      priorFixed outputFixed digestFixed values context) (fun _ _ => rfl)

end NightstreamFPrime.Layout.Stage1.PiRLCProtocolCompleteness
