import NightstreamFPrime.Layout.Stage1.NifsSourceReadback
import NightstreamFPrime.Layout.Stage1.StepSourceSpecs
import NightstreamFPrime.Layout.Stage1.PiCCSProofReadback
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns constructive sequencing of the existing pilot, C/R/D, running-transition,
and next-preimage rows from a selected semantic step and an actual NIFS run.
The source words and phase specifications are derived after construction.
Application witness construction and physical lowering remain separate.
-/

namespace NightstreamFPrime.Layout.Stage1.StepWitnessPrefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

private theorem pilot_before_c :
    Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤ PiCCSInputs.phaseOffset := by
  have bound : Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤
      PiCCSInputs.expectedContextStart := by
    rw [PiCCSInputs.expectedContextStart_matches_pilot, Pilot.physicalColumnCount_eq]
    exact Nat.le_add_right _ _
  unfold PiCCSInputs.phaseOffset PiCCSInputs.proofInputStart
  exact Nat.le_trans bound (Nat.le_trans (Nat.le_add_right _ _) (Nat.le_add_right _ _))

private theorem c_before_r : PiCCSInputs.phaseOffset ≤ PiRLCInputs.phaseOffset := by
  have bound := PiRLCInputs.piCcsLogicalFreshBase_le_phaseOffset
  unfold PiCCSStarts.logicalFreshBase at bound
  exact Nat.le_trans (Nat.le_add_right _ _) bound

private theorem r_before_d : PiRLCInputs.phaseOffset ≤ PiDECInputs.phaseOffset :=
  Nat.le_trans (Nat.le_trans (Nat.le_add_right _ _) PiDECProtocolCompleteness.rEnd_before_dInputs)
    (Nat.le_add_right _ _)

private theorem d_end_before_running :
    PiDECInputs.phaseOffset + PiDEC.v1_1.Formal.logicalPrivateCount ≤ RunningTransitionInputs.phaseOffset := by
  change PiDECStarts.phaseFreshStart ≤ PiDECStarts.outputFreshStart
  unfold PiDECStarts.outputFreshStart PiDECStarts.evalAFreshStart PiDECStarts.evalKFreshStart
    PiDECStarts.commitmentFreshStart PiDECStarts.publicInputFreshStart PiDECStarts.inputFreshStart
  omega

private theorem pilot_sources_before_running : PilotProduction.witnessOffset ≤ RunningTransitionInputs.phaseOffset := by
  have beforeC : PilotProduction.witnessOffset ≤ PiCCSInputs.phaseOffset := by
    rw [PilotProduction.witnessOffset_eq, PiCCSInputs.phaseOffset_eq]
    decide
  exact Nat.le_trans beforeC (Nat.le_trans c_before_r
    (Nat.le_trans r_before_d RunningTransitionInputs.piDecPhaseOffset_le))

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The exact PiDEC running result is preserved when every input source
below the PiDEC phase offset is preserved. No child opening is asserted. -/
theorem piDecOutput_eq_of_agree
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (before after : Env)
    (agrees : ∀ index, index < PiDECInputs.phaseOffset → before index = after index) :
    RunningTransitionInputs.piDecRunningOutput relation before =
      RunningTransitionInputs.piDecRunningOutput relation after := by
  have outputs := PiDEC.v1_1.Semantics.output_eq_of_agree relation
    (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset before after
    (PiDECInputs.assumptions relation before) agrees
  unfold RunningTransitionInputs.piDecRunningOutput
  dsimp only
  congr 1
  · exact congrArg (fun family => (family ⟨0, by decide⟩).point) outputs
  · funext source
    exact congrArg (fun family => (family (RunningTransitionInputs.childOfRunning source)).commitment) outputs
  · funext source
    exact congrArg (fun family => (family (RunningTransitionInputs.childOfRunning source)).publicInput) outputs
  · funext source
    exact congrArg (fun family => (family (RunningTransitionInputs.childOfRunning source)).evaluations.getD
      0 PaperAlgebra.evaluationZero) outputs

private theorem prior_word_below (index : Fin PilotProduction.stateHashWords) :
    PilotProduction.priorPreimageStart + index.val < PilotProduction.witnessOffset := by
  have bound := index.isLt
  unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
    PilotProduction.outputPreimageStart PilotProduction.priorPublicInputStart
  omega

private theorem next_word_below (index : Fin PilotProduction.stateHashWords) :
    PilotProduction.outputPreimageStart + index.val < PilotProduction.witnessOffset := by
  have bound := index.isLt
  unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
  omega

/-- A selected semantic step and its actual accepted NIFS run construct the
pilot, C/R/D, running-transition, and next-preimage rows in one environment.
The input conditions are semantic data conditions; no source-word agreement,
generated phase specification, or row is assumed. WellFormed keeps both
natural counters in range. Base dummy results remain separate from the
semantic default running output. -/
theorem completePrefix
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
    (step : StepHoldsFor relation ajtai context.toList Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (priorWellFormed : StateEncoding.WellFormed (priorHashPreimage (setup relation ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed (nextHashPreimage (setup relation ajtai context.toList) input output))
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation ajtai context.toList) input)))
    (accepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (input.running functionIndex) input.fresh input.nifsProof = some result)
    (recursiveResult : 0 < input.iteration → result = output.runningNext functionIndex) :
    let prior := priorHashPreimage (setup relation ajtai context.toList) input
    let next := nextHashPreimage (setup relation ajtai context.toList) input output
    let values := PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
    ∃ (digestFixed : output.x.length = PilotProduction.digestWords),
      ∃ p : Sequence.Prefix
          (PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) next output.x
            priorWellFormed.1 nextWellFormed.1 digestFixed values context) PilotProduction.witnessOffset,
        ∃ c : Sequence.Prefix p.current PiCCSInputs.phaseOffset,
          ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
            ∃ d : Sequence.Prefix
                (PiDECProofInputs.load r.current input.nifsProof
                  (PiRLC.v1_1.Semantics.evalOutput relation
                    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
                    PiRLCInputs.phaseOffset r.current).publicInput) PiDECInputs.phaseOffset,
              ∃ t : Sequence.Prefix d.current RunningTransitionInputs.phaseOffset,
                flatConstraints p.operations = Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset ∧
                c.operations = PiCCS.v1_1.Formal.opsAt relation (PiCCSProofInputs.relationInterface relation) PiCCSInputs.phaseOffset ∧
                r.operations = PiRLC.v1_1.Formal.opsAt relation PiRLCInputs.interface PiRLCInputs.phaseOffset ∧
                d.operations = PiDEC.v1_1.Formal.opsAt relation (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset ∧
                t.operations = Lifecycle.Stage1.RunningTransition.operations
                  (RunningTransitionInputs.interface logicalWidth publicFits) RunningTransitionInputs.phaseOffset ∧
                holdsFlat t.current p.operations ∧ holdsFlat t.current c.operations ∧
                holdsFlat t.current r.operations ∧ holdsFlat t.current d.operations ∧
                holdsFlat t.current (Lifecycle.Stage1.NextPreimage.opsAt NextPreimageInputs.sourceInterface
                  RunningTransitionInputs.phaseOffset) ∧
                Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset t.current ∧
                PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface logicalWidth publicFits)
                  PiDECInputs.phaseOffset t.current ∧
                Lifecycle.Stage1.RunningTransition.SpecHolds (RunningTransitionInputs.interface logicalWidth publicFits)
                  RunningTransitionInputs.phaseOffset t.current ∧
                Lifecycle.Stage1.NextPreimage.SpecHolds NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset t.current ∧
                RunningTransitionInputs.piDecRunningOutput relation t.current = result ∧
                (∀ index : Fin PilotProduction.stateHashWords,
                  t.current (PilotProduction.priorPreimageStart + index.val) =
                    (serializePreimage (publicFits := publicFits) prior).getD index.val 0) ∧
                (∀ index : Fin PilotProduction.stateHashWords,
                  t.current (PilotProduction.outputPreimageStart + index.val) =
                    (serializePreimage (publicFits := publicFits) next).getD index.val 0) := by
  let prior := priorHashPreimage (setup relation ajtai context.toList) input
  let next := nextHashPreimage (setup relation ajtai context.toList) input output
  let values := PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
  have outputHash : output.x = stateHash next := step.2.2.1
  have digestFixed : output.x.length = PilotProduction.digestWords := by
    rw [outputHash]
    exact StateEncoding.stateHash_length next
  have fresh : PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values = input.fresh := by
    have readback := PiCCSProofReadback.protocolFresh_ofProof relation input.fresh input.nifsProof
    rw [freshPublic] at readback
    exact readback
  have proofReadback : PiCCSProofInputs.relationProof relation values input.nifsProof = input.nifsProof :=
    PiCCSProofReadback.relationProof_ofProof relation _ _
  have acceptedSource : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values)
      (PiCCSProofInputs.relationProof relation values input.nifsProof) = some result := by
    rw [fresh, proofReadback]
    exact accepted
  have constructed := PilotNifsCompleteness.completePrefix prior next output.x priorWellFormed.1
    nextWellFormed.1 digestFixed values context relation ajtai input.nifsProof result
    priorWellFormed.2.2 nextWellFormed.2.2 rfl rfl outputHash acceptedSource
  rw [proofReadback] at constructed
  obtain ⟨p, c, r, d, pEnd, pConstraints, cOperations, rOperations, dOperations,
    pRows, cRows, rRows, _, _, dOutput⟩ := constructed
  have words := NifsSourceReadback.words_of_prefixes prior next (encHash (stateHash prior)) output.x
    priorWellFormed.1 nextWellFormed.1 digestFixed values context input.nifsProof _ p c r d
  have sourceSpecs := StepSourceSpecs.specs_of_step relation ajtai context input output d.current step
    priorWellFormed nextWellFormed words.1 words.2 (fun positive => dOutput.trans (recursiveResult positive))
  have transitionAssumptions := RunningTransitionInputs.assumptions logicalWidth publicFits relation d.current
  obtain ⟨completed, completionAgrees, completionRows⟩ := Lifecycle.Stage1.RunningTransition.completeness
    (RunningTransitionInputs.interface logicalWidth publicFits) d.current RunningTransitionInputs.phaseOffset
    transitionAssumptions sourceSpecs.1
  let t : Sequence.Prefix d.current RunningTransitionInputs.phaseOffset := {
    current := completed
    operations := Lifecycle.Stage1.RunningTransition.operations
      (RunningTransitionInputs.interface logicalWidth publicFits) RunningTransitionInputs.phaseOffset
    agrees := completionAgrees
    rows := completionRows
    scope := by
      rw [Lifecycle.Stage1.RunningTransition.localLength_eq]
      exact Lifecycle.Stage1.RunningTransition.flatConstraints_varsBelow _ _ _ transitionAssumptions }
  have pBound : PilotProduction.witnessOffset + localLength p.operations ≤ RunningTransitionInputs.phaseOffset := by
    rw [pEnd]
    exact Nat.le_trans pilot_before_c (Nat.le_trans c_before_r
      (Nat.le_trans r_before_d RunningTransitionInputs.piDecPhaseOffset_le))
  have cBound : PiCCSInputs.phaseOffset + localLength c.operations ≤ RunningTransitionInputs.phaseOffset := by
    rw [cOperations, ← PiCCS.v1_1.Formal.main_ops, PiCCS.v1_1.Formal.localLength_eq]
    change NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset ≤ _
    rw [← PiCCSStarts.logicalFreshBase_eq_layout relation]
    exact Nat.le_trans PiRLCInputs.piCcsLogicalFreshBase_le_phaseOffset
      (Nat.le_trans r_before_d RunningTransitionInputs.piDecPhaseOffset_le)
  have rBound : PiRLCInputs.phaseOffset + localLength r.operations ≤ RunningTransitionInputs.phaseOffset := by
    rw [rOperations, ← PiRLC.v1_1.Formal.main_ops, PiRLC.v1_1.Formal.localLength_eq]
    exact Nat.le_trans PiDECProtocolCompleteness.rEnd_before_dInputs
      (Nat.le_trans (Nat.le_add_right _ _) RunningTransitionInputs.piDecPhaseOffset_le)
  have dBound : PiDECInputs.phaseOffset + localLength d.operations ≤ RunningTransitionInputs.phaseOffset := by
    rw [dOperations, ← PiDEC.v1_1.Formal.main_ops, PiDEC.v1_1.Formal.localLength_eq]
    exact d_end_before_running
  have preserve (operations : List Op) (bound : Nat)
      (scope : ∀ expression ∈ flatConstraints operations, expression.VarsBelow bound)
      (before : bound ≤ RunningTransitionInputs.phaseOffset) (rows : holdsFlat d.current operations) :
      holdsFlat t.current operations := by
    intro expression member
    exact (expression.eval_eq_of_agree_below bound t.current d.current (scope expression member)
      (fun index below => t.agrees index (Or.inl (Nat.lt_of_lt_of_le below before)))).trans (rows expression member)
  have pFinal := preserve p.operations _ p.scope pBound pRows
  have cFinal := preserve c.operations _ c.scope cBound cRows
  have rFinal := preserve r.operations _ r.scope rBound rRows
  have dFinal := preserve d.operations _ d.scope dBound d.rows
  have finalOutput : RunningTransitionInputs.piDecRunningOutput relation t.current = result :=
    (piDecOutput_eq_of_agree relation t.current d.current (fun index below =>
      t.agrees index (Or.inl (Nat.lt_of_lt_of_le below RunningTransitionInputs.piDecPhaseOffset_le)))).trans dOutput
  have priorWords : ∀ index : Fin PilotProduction.stateHashWords,
      t.current (PilotProduction.priorPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) prior).getD index.val 0 := by
    intro index
    exact (t.agrees _ (Or.inl (Nat.lt_of_lt_of_le (prior_word_below index) pilot_sources_before_running))).trans (words.1 index)
  have nextWords : ∀ index : Fin PilotProduction.stateHashWords,
      t.current (PilotProduction.outputPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) next).getD index.val 0 := by
    intro index
    exact (t.agrees _ (Or.inl (Nat.lt_of_lt_of_le (next_word_below index) pilot_sources_before_running))).trans (words.2 index)
  have finalSpecs := StepSourceSpecs.specs_of_step relation ajtai context input output t.current step
    priorWellFormed nextWellFormed priorWords nextWords (fun positive => finalOutput.trans (recursiveResult positive))
  obtain ⟨nextEnv, nextAgrees, nextRows⟩ := Lifecycle.Stage1.NextPreimage.completeness
    NextPreimageInputs.sourceInterface t.current RunningTransitionInputs.phaseOffset finalSpecs.2
  have nextEnvEq : nextEnv = t.current := by
    funext index
    apply nextAgrees index
    rw [Lifecycle.Stage1.NextPreimage.localLength_eq]
    omega
  rw [nextEnvEq, Lifecycle.Stage1.NextPreimage.main_ops] at nextRows
  have selectedPilotRows : ∀ expression ∈ Pilot.logicalConstraints PilotProduction.interface
      PilotProduction.witnessOffset, expression.eval t.current = 0 := by
    rw [← pConstraints]
    exact pFinal
  have priorRows : holdsFlat t.current (Circuit.ops
      (Lifecycle.Pilot.priorCircuit PilotProduction.interface).main PilotProduction.witnessOffset) := by
    intro expression member
    exact selectedPilotRows expression (List.mem_append_left _ member)
  have outputRows : holdsFlat t.current (Circuit.ops
      (Lifecycle.Pilot.outputCircuit PilotProduction.interface).main
      (Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset)) := by
    intro expression member
    exact selectedPilotRows expression (List.mem_append_right _ member)
  have pilotPhase := Lifecycle.Pilot.phase_soundness PilotProduction.interface PilotProduction.witnessOffset
    t.current (PilotProduction.assumptions t.current)
    (holdsFlat_implies_holds _ _ priorRows) (holdsFlat_implies_holds _ _ outputRows)
  have dPhase := PiDEC.v1_1.Semantics.spec_implies_phaseHolds relation ajtai
    (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset t.current
    (PiDEC.v1_1.Formal.soundness relation (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset t.current (PiDECInputs.assumptions relation t.current) (by
        change holds t.current (PiDEC.v1_1.Formal.opsAt relation
          (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset)
        rw [← dOperations]
        exact holdsFlat_implies_holds _ _ dFinal))
  exact ⟨digestFixed, p, c, r, d, t, pConstraints, cOperations, rOperations, dOperations, rfl,
    pFinal, cFinal, rFinal, dFinal, nextRows, pilotPhase, dPhase, finalSpecs.1, finalSpecs.2,
    finalOutput, priorWords, nextWords⟩

end NightstreamFPrime.Layout.Stage1.StepWitnessPrefix
