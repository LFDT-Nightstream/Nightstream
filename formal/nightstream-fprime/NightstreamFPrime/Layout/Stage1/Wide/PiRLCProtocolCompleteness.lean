import NightstreamFPrime.Layout.Stage1.PiRLCProtocolCompleteness
import NightstreamFPrime.Layout.Stage1.Wide.PiRLCInputBounds
import NightstreamFPrime.Layout.Stage1.Wide.AccumulatorSemantics

/-! Continue the actual PiCCS witness with the total wide PiRLC constructor.
Its challenges and output match the candidate key on the same protocol inputs.
There is no sampler-availability or generated-output premise. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.PiRLCProtocolCompleteness

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout.Stage1.PiCCSProofInputs (relationInterface relationProof)
open NightstreamFPrime.Layout.Stage1.PiRLCProtocolCompleteness
  (cViews_eq_of_fields initialState_eq_of_phase accumulator_phase protocol_readback)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (priorPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
  (output : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (digest : Digest)
  (priorFixed : PilotProduction.FixedPreimage prior)
  (outputFixed : PilotProduction.FixedPreimage output)
  (digestFixed : digest.length = PilotProduction.digestWords)
  (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4) (template : Proof 9)

/-- Complete PiRLC after the physical PiCCS scratch has been filled. Only
agreement below the logical PiCCS endpoint is needed from that lowering. -/
theorem completePrefix_after_c
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
      r.operations = PiRLC.Wide.Formal.opsAt relation
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset ∧
      holdsFlat r.current c.operations ∧
      PiRLC.Wide.Semantics.PhaseHolds relation ajtai
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current ∧
      (PiRLC.Wide.Key.key relation ajtai).piRlcChallenges (prior.running functionIndex)
        (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values) (relationProof relation values template) =
        some (PiRLC.Wide.Semantics.evalChallenges
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current) ∧
      PiRLC.Wide.Semantics.evalOutput relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current =
        (PiRLC.Wide.Key.key relation ajtai).parentForChallenges (prior.running functionIndex)
          (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values) (relationProof relation values template)
          (PiRLC.Wide.Semantics.evalChallenges
            (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current) := by
  obtain ⟨r, operations, phase⟩ := PiRLC.Wide.Formal.completePrefix_constructive relation ajtai
    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) afterC PiRLCInputs.phaseOffset
    (PiRLCInputBounds.assumptions relation afterC)
  have cLimit : PiCCSInputs.phaseOffset + localLength c.operations ≤ PiRLCInputs.phaseOffset := by
    rw [cOperations, ← Formal.main_ops, Formal.localLength_eq]
    change NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset ≤ _
    rw [← PiCCSStarts.logicalFreshBase_eq_layout relation]
    exact NightstreamFPrime.Layout.Stage1.PiRLCInputs.piCcsLogicalFreshBase_le_phaseOffset
  have rowsPreserved : holdsFlat r.current c.operations := by
    intro expression member
    have same := expression.eval_eq_of_agree_below
      (PiCCSInputs.phaseOffset + localLength c.operations) r.current c.current
      (c.scope expression member) (fun index below =>
        (r.agrees index (Or.inl (Nat.lt_of_lt_of_le below cLimit))).trans (preserved index below))
    exact same.trans (c.rows expression member)
  have cRows : holds r.current (Circuit.ops (Formal.main relation (relationInterface relation)) PiCCSInputs.phaseOffset) := by
    rw [Formal.main_ops, ← cOperations]
    exact holdsFlat_implies_holds r.current c.operations rowsPreserved
  have cAssumptions := NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation
    (relationInterface relation) PiCCSInputs.phaseOffset
    (PiCCSInputs.externalInputsLinear logicalWidth publicFits) r.current
  have finalC := Formal.spec_implies_phaseHolds relation ajtai (relationInterface relation)
    PiCCSInputs.phaseOffset r.current (relationProof relation values template)
    (Formal.soundness relation (relationInterface relation) r.current PiCCSInputs.phaseOffset cAssumptions cRows)
  have finalRead := protocol_readback relation prior priorPublic output digest priorFixed outputFixed
    digestFixed values context template initial source r.current (fun index below =>
      (r.agrees index (Or.inl (by omega))).trans
        ((preserved index (Nat.lt_of_lt_of_le below (Nat.le_add_right _ _))).trans (c.agrees index (Or.inl below))))
  have finalState := initialState_eq_of_phase relation ajtai r.current (relationProof relation values template) finalC
  rw [finalRead.1, finalRead.2.1, finalRead.2.2] at finalState
  have sampled := AccumulatorSemantics.challenges_eq_key relation ajtai r.current
    (prior.running functionIndex) (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
    (relationProof relation values template) _ _ phase (by
      rw [PiRLC.Wide.Key.piCcsExecution_unchanged]
      exact finalState)
  have canonicalC := accumulator_phase relation ajtai r.current (relationProof relation values template) finalC
  have inputs := AccumulatorSemantics.inputs_eq_keyOutputs relation ajtai r.current canonicalC
  have runningEq : Stage1.AccumulatorInputs.running logicalWidth publicFits r.current = prior.running functionIndex := finalRead.1
  have freshEq : Stage1.AccumulatorInputs.fresh logicalWidth publicFits r.current =
      PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values := finalRead.2.1
  rw [runningEq, freshEq] at inputs
  have roundsEq : (Stage1.AccumulatorInputs.proof relation r.current).piCcsRounds =
      (relationProof relation values template).piCcsRounds := congrArg (fun proof => proof.piCcsRounds) finalRead.2.2
  have outputEq : (Stage1.AccumulatorInputs.proof relation r.current).piCcsOutput =
      (relationProof relation values template).piCcsOutput := congrArg (fun proof => proof.piCcsOutput) finalRead.2.2
  have views := cViews_eq_of_fields relation (PiRLC.Wide.Key.key relation ajtai)
    (prior.running functionIndex) (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
    (Stage1.AccumulatorInputs.proof relation r.current) (relationProof relation values template) roundsEq outputEq
  have parent := AccumulatorSemantics.output_eq_keyParent relation ajtai r.current
    (prior.running functionIndex) (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
    (relationProof relation values template) _ _ phase (inputs.trans views.2)
  exact ⟨r, operations, rowsPreserved, phase, sampled, parent⟩

/-- Accepted PiCCS protocol inputs construct both phases with no sampler premise. -/
theorem completePrefix_from
    (priorPc : prior.pc = 1) (outputPc : output.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (outputContext : output.verifierKeys functionIndex = context.toList)
    (accepted : Folding.PiCCS.Accepted (PiRLC.Wide.Key.key relation ajtai)
      (prior.running functionIndex) (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
      (relationProof relation values template))
    (initial : Env)
    (source : ∀ index, PiCCSOrdinarySourceSupport.External index → initial index =
      PiCCSProtocolCompleteness.environment prior priorPublic output digest
        priorFixed outputFixed digestFixed values context index) :
    ∃ c : Sequence.Prefix initial PiCCSInputs.phaseOffset,
      ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
        c.operations = Formal.opsAt relation (relationInterface relation) PiCCSInputs.phaseOffset ∧
        r.operations = PiRLC.Wide.Formal.opsAt relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset ∧
        holdsFlat r.current c.operations ∧
        PiRLC.Wide.Semantics.PhaseHolds relation ajtai
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current ∧
        (PiRLC.Wide.Key.key relation ajtai).piRlcChallenges (prior.running functionIndex)
          (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values) (relationProof relation values template) =
          some (PiRLC.Wide.Semantics.evalChallenges
            (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current) ∧
        PiRLC.Wide.Semantics.evalOutput relation
            (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current =
          (PiRLC.Wide.Key.key relation ajtai).parentForChallenges (prior.running functionIndex)
            (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values) (relationProof relation values template)
            (PiRLC.Wide.Semantics.evalChallenges
              (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current) := by
  obtain ⟨c, operations, _⟩ := PiCCSProtocolCompleteness.completePrefix_from prior priorPublic output digest
    priorFixed outputFixed digestFixed values context relation ajtai template priorPc outputPc
    priorContext outputContext (by
      unfold Folding.PiCCS.Accepted at accepted ⊢
      rw [PiRLC.Wide.Key.piCcsCheck_unchanged] at accepted
      exact accepted) initial source
  obtain ⟨r, completed⟩ := completePrefix_after_c relation ajtai prior priorPublic output digest
    priorFixed outputFixed digestFixed values context template initial source c operations c.current (fun _ _ => rfl)
  exact ⟨c, r, operations, completed⟩

end NightstreamFPrime.Layout.Stage1.Wide.PiRLCProtocolCompleteness
