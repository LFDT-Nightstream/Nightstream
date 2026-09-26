import NightstreamFPrime.Layout.Stage1.Wide.PhysicalPrefixCompleteness
import NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionCompleteness
import NightstreamFPrime.Layout.Stage1.Wide.StepSourceSpecs
import NightstreamFPrime.Layout.Stage1.Wide.PilotPiCCSPiRLCPiDECRunningTransition

/-! Construct the complete wide physical source from the candidate semantic
step and its accepted NIFS advice. The source retains exact prior/next state
words and the actual verifier result for the compact CCS assignment. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.StepPhysicalCompleteness

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint
open Spec.HyperNova.Construction2.Paper
open Stage1.StepPhysicalCompleteness (external_before_c prior_word_below next_word_below)

private theorem c_before_running : PiCCSInputs.phaseOffset ≤ RunningTransitionInputs.phaseOffset := by decide
private theorem pilot_before_running : PilotProduction.witnessOffset ≤ RunningTransitionInputs.phaseOffset := by decide

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- An accepted wide-key step constructs every physical prefix row and both
state readbacks. No physical rows or environment agreements are premises. -/
theorem complete_with_values
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : VerifierContext.Digest4)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (step : Lifecycle.Stage1.Wide.Relation.StepHoldsFor relation ajtai context.toList Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (priorWellFormed : StateEncoding.WellFormed (priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed (nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input output))
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input)))
    (accepted : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
      (input.running functionIndex) input.fresh input.nifsProof = some result)
    (recursiveResult : 0 < input.iteration → result = output.runningNext functionIndex) :
    let prior := priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input
    let next := nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input output
    let values := PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
    ∃ (digestFixed : output.x.length = PilotProduction.digestWords), ∃ env : Env,
      PilotPiCCSPiRLCPiDECRunningTransition.PhysicalHolds relation env ∧
      Lifecycle.Stage1.NextPreimage.SpecHolds
        NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset env ∧
      RunningTransitionInputs.piDecRunningOutput relation env = result ∧
      (∀ index, index < PilotProduction.witnessOffset ∨ PiCCSOrdinarySourceSupport.External index →
        env index = PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior))
          next output.x priorWellFormed.1 nextWellFormed.1 digestFixed values context index) ∧
      (∀ index : Fin PilotProduction.stateHashWords,
        env (PilotProduction.priorPreimageStart + index.val) =
          (serializePreimage (publicFits := publicFits) prior).getD index.val 0) ∧
      (∀ index : Fin PilotProduction.stateHashWords,
        env (PilotProduction.outputPreimageStart + index.val) =
          (serializePreimage (publicFits := publicFits) next).getD index.val 0) ∧
      PiRLC.Wide.Formal.RangesCompleted
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset env := by
  let prior := priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input
  let next := nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input output
  let values := PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
  have outputHash : output.x = stateHash next := step.2.2.1
  have digestFixed : output.x.length = PilotProduction.digestWords := by
    rw [outputHash]
    exact StateEncoding.stateHash_length next
  let initial := PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) next output.x
    priorWellFormed.1 nextWellFormed.1 digestFixed values context
  have fresh : PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values = input.fresh := by
    have readback := PiCCSProofReadback.protocolFresh_ofProof relation input.fresh input.nifsProof
    rw [freshPublic] at readback
    exact readback
  have proofReadback : PiCCSProofInputs.relationProof relation values input.nifsProof = input.nifsProof :=
    PiCCSProofReadback.relationProof_ofProof relation _ _
  have acceptedSource : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values)
      (PiCCSProofInputs.relationProof relation values input.nifsProof) = some result := by
    rw [fresh, proofReadback]
    exact accepted
  obtain ⟨nifs, nifsRows, nifsScope, nifsOutput, sourceReadback, rangeValues⟩ := PhysicalPrefixCompleteness.complete_with_values relation ajtai prior next output.x
    priorWellFormed.1 nextWellFormed.1 digestFixed values context input.nifsProof result
    priorWellFormed.2.2 nextWellFormed.2.2 rfl rfl outputHash acceptedSource
  have priorWords : ∀ index : Fin PilotProduction.stateHashWords,
      nifs (PilotProduction.priorPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) prior).getD index.val 0 := by
    intro index
    exact (sourceReadback _ (Or.inl (prior_word_below index))).trans
      (PiCCSProtocolCompleteness.prior_word prior (encHash (stateHash prior)) next output.x
        priorWellFormed.1 nextWellFormed.1 digestFixed values context index)
  have nextWords : ∀ index : Fin PilotProduction.stateHashWords,
      nifs (PilotProduction.outputPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) next).getD index.val 0 := by
    intro index
    exact (sourceReadback _ (Or.inl (next_word_below index))).trans
      (PiCCSProtocolCompleteness.output_word prior (encHash (stateHash prior)) next output.x
        priorWellFormed.1 nextWellFormed.1 digestFixed values context index)
  have specs := StepSourceSpecs.specs_of_step relation ajtai context input output nifs step priorWellFormed
    nextWellFormed priorWords nextWords (fun positive => nifsOutput.trans (recursiveResult positive))
  obtain ⟨completed, runningAgrees, runningRows⟩ := RunningTransitionLayout.physical_complete relation nifs specs.1
  have prefixRows : PilotPiCCSPiRLCPiDEC.PhysicalHolds relation completed :=
    R1CS.rowsHold_of_agree_below _ _ nifs completed nifsScope
      (fun index below => runningAgrees index (Or.inl below)) nifsRows
  have physical : PilotPiCCSPiRLCPiDECRunningTransition.PhysicalHolds relation completed :=
    (PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation completed).2 ⟨prefixRows, runningRows⟩
  have finalOutput : RunningTransitionInputs.piDecRunningOutput relation completed = result :=
    (RunningTransitionInputs.piDecOutput_eq_of_agree relation completed nifs (fun index below =>
      runningAgrees index (Or.inl (below.trans_le RunningTransitionInputs.piDecPhaseOffset_le)))).trans nifsOutput
  have finalReadback : ∀ index, index < PilotProduction.witnessOffset ∨ PiCCSOrdinarySourceSupport.External index →
      completed index = initial index := by
    intro index support
    have below : index < RunningTransitionInputs.phaseOffset := support.elim
      (fun h => h.trans_le pilot_before_running) (fun h => (external_before_c index h).trans_le c_before_running)
    exact (runningAgrees index (Or.inl below)).trans (sourceReadback index support)
  have finalPrior : ∀ index : Fin PilotProduction.stateHashWords,
      completed (PilotProduction.priorPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) prior).getD index.val 0 := by
    intro index
    exact (runningAgrees _ (Or.inl ((prior_word_below index).trans_le pilot_before_running))).trans (priorWords index)
  have finalNext : ∀ index : Fin PilotProduction.stateHashWords,
      completed (PilotProduction.outputPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) next).getD index.val 0 := by
    intro index
    exact (runningAgrees _ (Or.inl ((next_word_below index).trans_le pilot_before_running))).trans (nextWords index)
  have finalSpecs := StepSourceSpecs.specs_of_step relation ajtai context input output completed step priorWellFormed
    nextWellFormed finalPrior finalNext (fun positive => finalOutput.trans (recursiveResult positive))
  have finalRanges : PiRLC.Wide.Formal.RangesCompleted
      (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset completed := by
    apply PiRLC.Wide.Formal.rangesCompleted_of_agree relation PiRLCInputs.interface PiRLCInputs.phaseOffset
      nifs completed (PiRLCInputBounds.assumptions relation nifs) rangeValues
    intro index below
    exact runningAgrees index (Or.inl (below.trans_le (by decide)))
  exact ⟨digestFixed, completed, physical, finalSpecs.2, finalOutput, finalReadback, finalPrior, finalNext, finalRanges⟩

theorem complete
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : VerifierContext.Digest4)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (step : Lifecycle.Stage1.Wide.Relation.StepHoldsFor relation ajtai context.toList Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (priorWellFormed : StateEncoding.WellFormed (priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed (nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input output))
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input)))
    (accepted : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
      (input.running functionIndex) input.fresh input.nifsProof = some result)
    (recursiveResult : 0 < input.iteration → result = output.runningNext functionIndex) :
    let prior := priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input
    let next := nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input output
    let values := PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
    ∃ (digestFixed : output.x.length = PilotProduction.digestWords), ∃ env : Env,
      PilotPiCCSPiRLCPiDECRunningTransition.PhysicalHolds relation env ∧
      Lifecycle.Stage1.NextPreimage.SpecHolds
        NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset env ∧
      RunningTransitionInputs.piDecRunningOutput relation env = result ∧
      (∀ index, index < PilotProduction.witnessOffset ∨ PiCCSOrdinarySourceSupport.External index →
        env index = PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior))
          next output.x priorWellFormed.1 nextWellFormed.1 digestFixed values context index) ∧
      (∀ index : Fin PilotProduction.stateHashWords,
        env (PilotProduction.priorPreimageStart + index.val) =
          (serializePreimage (publicFits := publicFits) prior).getD index.val 0) ∧
      (∀ index : Fin PilotProduction.stateHashWords,
        env (PilotProduction.outputPreimageStart + index.val) =
          (serializePreimage (publicFits := publicFits) next).getD index.val 0) := by
  obtain ⟨digestFixed, env, rows, next, resultValue, sources, priorWords, nextWords, _⟩ :=
    complete_with_values relation ajtai context input output result step priorWellFormed nextWellFormed
      freshPublic accepted recursiveResult
  exact ⟨digestFixed, env, rows, next, resultValue, sources, priorWords, nextWords⟩

end NightstreamFPrime.Layout.Stage1.Wide.StepPhysicalCompleteness
