import NightstreamFPrime.Layout.Stage1.StepPhysicalCompleteness
import NightstreamFPrime.Layout.Stage1.Wide.PiDECProtocolCompleteness
import NightstreamFPrime.Layout.Stage1.Wide.PilotPiCCSPiRLCPiDEC

/-! Construct the pilot and all three physical NIFS phases from an accepted
wide-key run. Reuse the existing lowering and source-preservation lemmas;
no physical witness or sampler-availability premise is supplied by the caller. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.PhysicalPrefixCompleteness

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint
open Stage1.StepPhysicalCompleteness
  (lower_prefix pilot_end_before_c pilot_start_le_end pilot_logical_le_physical
   external_outside_pilot_physical external_before_c c_end_before_r firstFresh_le_next)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem r_end_before_d (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (NightstreamFPrime.Layout.PiRLC.Wide.plan relation PiRLCInputs.interface PiRLCInputs.phaseOffset).next ≤
      PiDECInputs.proofInputStart := by
  have bound := Nat.le_max_right (PilotPiCCS.physicalColumnCount relation)
    (NightstreamFPrime.Layout.PiRLC.Wide.physicalColumnCount relation PiRLCInputs.interface PilotPiCCSPiRLC.piRlcOffset)
  change _ ≤ PilotPiCCSPiRLC.physicalColumnCount relation at bound
  rw [PilotPiCCSPiRLC.physicalColumnCount_eq relation] at bound
  exact bound

private theorem d_end_before_running (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (NightstreamFPrime.Layout.PiDEC.v1_1.plan relation (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset).next ≤ RunningTransitionInputs.phaseOffset := by
  have bound := Nat.le_max_right PiDECInputs.phaseOffset
    (NightstreamFPrime.Layout.PiDEC.v1_1.physicalColumnCount relation
      (PiDECInputs.interface logicalWidth publicFits) PilotPiCCSPiRLCPiDEC.piDecOffset)
  change _ ≤ PilotPiCCSPiRLCPiDEC.physicalColumnCount relation at bound
  rw [PilotPiCCSPiRLCPiDEC.physicalColumnCount_eq relation] at bound
  exact bound

theorem complete_with_values
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (prior advertised : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (advertisedFixed : PilotProduction.FixedPreimage advertised)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4)
    (template : Proof 9)
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPc : prior.pc = 1) (advertisedPc : advertised.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (advertisedContext : advertised.verifierKeys functionIndex = context.toList)
    (outputHash : digest = stateHash advertised)
    (accepted : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values)
      (PiCCSProofInputs.relationProof relation values template) = some result) :
    ∃ completed,
      PilotPiCCSPiRLCPiDEC.PhysicalHolds relation completed ∧
      (∀ row ∈ PilotPiCCSPiRLCPiDEC.physicalRows relation,
        row.VarsBelow RunningTransitionInputs.phaseOffset) ∧
      RunningTransitionInputs.piDecRunningOutput relation completed = result ∧
      (∀ index, index < PilotProduction.witnessOffset ∨ PiCCSOrdinarySourceSupport.External index →
        completed index = PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
          priorFixed advertisedFixed digestFixed values context index) ∧
      PiRLC.Wide.Formal.RangesCompleted
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset completed := by
  let initial := PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
    priorFixed advertisedFixed digestFixed values context
  let proof := PiCCSProofInputs.relationProof relation values template
  let fresh := PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values
  obtain ⟨cAccepted, _⟩ := PiDECProtocolCompleteness.verifierInputs relation ajtai
    (prior.running functionIndex) fresh proof result accepted
  obtain ⟨p, pEnd, pConstraints⟩ := PilotNifsCompleteness.pilot_prefix prior advertised digest
    priorFixed advertisedFixed digestFixed values context outputHash
  have pLogical : ConstraintsHold p.current (Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset) := by
    rw [← pConstraints]
    exact p.rows
  obtain ⟨pPhysical, pLower, pRows⟩ := PilotProduction.physical_complete p.current pLogical
  have pScope : ∀ row ∈ Pilot.physicalRows PilotProduction.interface PilotProduction.witnessOffset,
      row.VarsBelow (Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset) := by
    rw [Pilot.physicalColumnCount_eq]
    exact R1CS.lowerConstraints_rows_varsBelow _ _
      (Pilot.logicalConstraints_varsBelow _ _ (PilotProduction.layoutAssumptions p.current))
  have pAgrees : ∀ index, index < PilotProduction.witnessOffset ∨ PiCCSOrdinarySourceSupport.External index →
      pPhysical index = initial index := by
    intro index support
    have outside := support.elim Or.inl (external_outside_pilot_physical index)
    have physicalEnd : Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset + 788 =
        Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset := by
      rw [Pilot.physicalColumnCount_eq, PilotProduction.logicalConstraints_freshCount]
    exact (pLower index (outside.imp (fun h => h.trans_le pilot_start_le_end) (by
      intro h
      simpa only [physicalEnd] using h))).trans (p.agrees index (outside.imp id (by
        intro h
        rw [pEnd]
        exact pilot_logical_le_physical.trans h)))
  obtain ⟨c, cOperations, _⟩ := PiCCSProtocolCompleteness.completePrefix_from prior (encHash (stateHash prior))
    advertised digest priorFixed advertisedFixed digestFixed values context relation ajtai template
    priorPc advertisedPc priorContext advertisedContext (by
      unfold PiCCS.Accepted at cAccepted ⊢
      rw [PiRLC.Wide.Key.piCcsCheck_unchanged] at cAccepted
      exact cAccepted) pPhysical (fun index support => pAgrees index (Or.inr support))
  let cPlan := NightstreamFPrime.Layout.PiCCS.v1_1.plan relation
    (PiCCSProofInputs.relationInterface relation) PiCCSInputs.phaseOffset
  have cConstraints : cPlan.constraints = flatConstraints c.operations := by
    rw [cOperations]
    rfl
  have cFirst : cPlan.firstFresh = PiCCSInputs.phaseOffset + localLength c.operations := by
    rw [cOperations, ← PiCCS.v1_1.Formal.main_ops]
    exact NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount_eq relation _ _
  obtain ⟨cPhysical, cLower, cRows, cScope⟩ := lower_prefix c cPlan cConstraints cFirst
  have cAfter : ∀ index, index < cPlan.firstFresh → cPhysical index = c.current index :=
    fun index below => cLower index (Or.inl below)
  have cBefore : ∀ index, index < PiCCSInputs.phaseOffset → cPhysical index = pPhysical index := by
    intro index below
    exact (cAfter index (by rw [cFirst]; omega)).trans (c.agrees index (Or.inl below))
  obtain ⟨r, rOperations, _, _, rSampled, rParent, rangeValues⟩ := PiRLCProtocolCompleteness.completePrefix_after_c_with_values
    relation ajtai prior (encHash (stateHash prior)) advertised digest priorFixed advertisedFixed digestFixed
    values context template pPhysical (fun index support => pAgrees index (Or.inr support))
    c cOperations cPhysical (by simpa only [cFirst] using cAfter)
  let rPlan := NightstreamFPrime.Layout.PiRLC.Wide.plan relation PiRLCInputs.interface PiRLCInputs.phaseOffset
  have rConstraints : rPlan.constraints = flatConstraints r.operations := by
    rw [rOperations]
    rfl
  have rFirst : rPlan.firstFresh = PiRLCInputs.phaseOffset + localLength r.operations := by
    rw [rOperations, ← PiRLC.Wide.Formal.main_ops]
    exact NightstreamFPrime.Layout.PiRLC.Wide.logicalColumnCount_eq relation _ _
  obtain ⟨rPhysical, rLower, rRows, rScope⟩ := lower_prefix r rPlan rConstraints rFirst
  have rAfter : ∀ index, index < rPlan.firstFresh → rPhysical index = r.current index :=
    fun index below => rLower index (Or.inl below)
  have rBefore : ∀ index, index < PiRLCInputs.phaseOffset → rPhysical index = cPhysical index := by
    intro index below
    exact (rAfter index (by rw [rFirst]; omega)).trans (r.agrees index (Or.inl below))
  obtain ⟨d, dOperations, _, _, dOutput⟩ := PiDECProtocolCompleteness.completePrefix_after_r
    relation ajtai (prior.running functionIndex) fresh proof result accepted cPhysical r rOperations
    rSampled rParent rPhysical (by simpa only [rFirst] using rAfter)
  let dPlan := NightstreamFPrime.Layout.PiDEC.v1_1.plan relation (PiDECInputs.interface logicalWidth publicFits)
    PiDECInputs.phaseOffset
  have dConstraints : dPlan.constraints = flatConstraints d.operations := by
    rw [dOperations]
    rfl
  have dFirst : dPlan.firstFresh = PiDECInputs.phaseOffset + localLength d.operations := by
    rw [dOperations, ← PiDEC.v1_1.Formal.main_ops]
    exact NightstreamFPrime.Layout.PiDEC.v1_1.logicalColumnCount_eq relation _ _
  obtain ⟨dPhysical, dLower, dRows, dScope⟩ := lower_prefix d dPlan dConstraints dFirst
  have dAfter : ∀ index, index < PiDECInputs.phaseOffset → dPhysical index = d.current index := by
    intro index below
    exact dLower index (Or.inl (by rw [dFirst]; omega))
  have dBefore : ∀ index, index < PiDECInputs.proofInputStart → dPhysical index = rPhysical index := by
    intro index below
    have beforeD : index < PiDECInputs.phaseOffset := below.trans_le (Nat.le_add_right _ _)
    exact (dAfter index beforeD).trans ((d.agrees index (Or.inl beforeD)).trans
      (PiDECProofInputs.load_agreesOutside rPhysical proof _ index (Or.inl below)))
  have cLeR : PiCCSInputs.phaseOffset ≤ PiRLCInputs.phaseOffset := by
    exact (by rw [cFirst]; omega : PiCCSInputs.phaseOffset ≤ cPlan.firstFresh).trans
      ((firstFresh_le_next cPlan).trans (c_end_before_r relation))
  have rLeD : PiRLCInputs.phaseOffset ≤ PiDECInputs.proofInputStart := by
    exact (by rw [rFirst]; omega : PiRLCInputs.phaseOffset ≤ rPlan.firstFresh).trans
      ((firstFresh_le_next rPlan).trans (r_end_before_d relation))
  have pAtFinal := R1CS.rowsHold_of_agree_below _ _ pPhysical dPhysical pScope (by
    intro index below
    have cBound := below.trans_le pilot_end_before_c
    exact (dBefore index (cBound.trans_le (cLeR.trans rLeD))).trans
      ((rBefore index (cBound.trans_le cLeR)).trans (cBefore index cBound))) pRows
  have cAtFinal := R1CS.rowsHold_of_agree_below _ _ cPhysical dPhysical cScope (by
    intro index below
    have rBound := below.trans_le (c_end_before_r relation)
    exact (dBefore index (rBound.trans_le rLeD)).trans (rBefore index rBound)) cRows
  have rAtFinal := R1CS.rowsHold_of_agree_below _ _ rPhysical dPhysical rScope (by
    intro index below
    exact dBefore index (below.trans_le (r_end_before_d relation))) rRows
  refine ⟨dPhysical, ?_, ?_, ?_, ?_, ?_⟩
  · apply (PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation dPhysical).2
    refine ⟨(PilotPiCCSPiRLC.physicalHolds_iff relation dPhysical).2
      ⟨(PilotPiCCS.physicalHolds_iff relation dPhysical).2 ⟨pAtFinal, cAtFinal⟩, rAtFinal⟩, dRows⟩
  · intro row member
    have dLeT : PiDECInputs.proofInputStart ≤ RunningTransitionInputs.phaseOffset :=
      (Nat.le_add_right _ _).trans RunningTransitionInputs.piDecPhaseOffset_le
    change row ∈ ((Pilot.physicalRows PilotProduction.interface PilotProduction.witnessOffset ++ cPlan.rows) ++
      rPlan.rows) ++ dPlan.rows at member
    rcases List.mem_append.mp member with earlier | last
    · rcases List.mem_append.mp earlier with first | middle
      · rcases List.mem_append.mp first with pilot | ccs
        · exact (pScope row pilot).mono row (pilot_end_before_c.trans (cLeR.trans (rLeD.trans dLeT)))
        · exact (cScope row ccs).mono row ((c_end_before_r relation).trans (rLeD.trans dLeT))
      · exact (rScope row middle).mono row ((r_end_before_d relation).trans dLeT)
    · exact (dScope row last).mono row (d_end_before_running relation)
  · exact (RunningTransitionInputs.piDecOutput_eq_of_agree relation dPhysical d.current dAfter).trans dOutput
  · intro index support
    have beforeC : index < PiCCSInputs.phaseOffset := by
      rcases support with early | external
      · exact early.trans_le (pilot_start_le_end.trans (pilot_logical_le_physical.trans pilot_end_before_c))
      · exact external_before_c index external
    exact (dBefore index (beforeC.trans_le (cLeR.trans rLeD))).trans
      ((rBefore index (beforeC.trans_le cLeR)).trans ((cBefore index beforeC).trans (pAgrees index support)))
  · apply PiRLC.Wide.Formal.rangesCompleted_of_agree relation PiRLCInputs.interface PiRLCInputs.phaseOffset
      r.current dPhysical (PiRLCInputBounds.assumptions relation r.current) rangeValues
    intro index below
    have beforeD : index < PiDECInputs.proofInputStart := below.trans_le (by decide)
    have beforeFresh : index < rPlan.firstFresh := by
      rw [rFirst, rOperations, ← PiRLC.Wide.Formal.main_ops, PiRLC.Wide.Formal.localLength_eq]
      change index < 19513117 + 107729
      change index < 19568520 at below
      omega
    exact (dBefore index beforeD).trans (rAfter index beforeFresh)

theorem complete
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (prior advertised : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (advertisedFixed : PilotProduction.FixedPreimage advertised)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4)
    (template : Proof 9)
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPc : prior.pc = 1) (advertisedPc : advertised.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (advertisedContext : advertised.verifierKeys functionIndex = context.toList)
    (outputHash : digest = stateHash advertised)
    (accepted : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values)
      (PiCCSProofInputs.relationProof relation values template) = some result) :
    ∃ completed,
      PilotPiCCSPiRLCPiDEC.PhysicalHolds relation completed ∧
      (∀ row ∈ PilotPiCCSPiRLCPiDEC.physicalRows relation,
        row.VarsBelow RunningTransitionInputs.phaseOffset) ∧
      RunningTransitionInputs.piDecRunningOutput relation completed = result ∧
      (∀ index, index < PilotProduction.witnessOffset ∨ PiCCSOrdinarySourceSupport.External index →
        completed index = PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
          priorFixed advertisedFixed digestFixed values context index) := by
  obtain ⟨completed, rows, scope, resultValue, sources, _⟩ := complete_with_values relation ajtai prior advertised digest
    priorFixed advertisedFixed digestFixed values context template result priorPc advertisedPc priorContext
    advertisedContext outputHash accepted
  exact ⟨completed, rows, scope, resultValue, sources⟩

end NightstreamFPrime.Layout.Stage1.Wide.PhysicalPrefixCompleteness
