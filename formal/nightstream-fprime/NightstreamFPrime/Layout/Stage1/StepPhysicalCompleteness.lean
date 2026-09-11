import NightstreamFPrime.Layout.Stage1.StepWitnessPrefix
import NightstreamFPrime.Layout.Stage1.SpartanRows
import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation

/-!
Owns sequential physical witness construction for the selected semantic step.
Each phase is lowered before the next logical phase is constructed. Earlier
physical rows are preserved below their certified endpoints. The final source
is copied through the existing Spartan inverse; no row representation or
caller-supplied physical witness is introduced.
-/

namespace NightstreamFPrime.Layout.Stage1.StepPhysicalCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

private theorem lower_prefix {initial : Env} {offset : Nat}
    (builtPrefix : Sequence.Prefix initial offset) (plan : R1CS.LoweringPlan)
    (constraints : plan.constraints = flatConstraints builtPrefix.operations)
    (firstFresh : plan.firstFresh = offset + localLength builtPrefix.operations) :
    ∃ completed,
      AgreesOutside builtPrefix.current completed plan.firstFresh plan.freshColumnCount ∧
      R1CS.RowsHold completed plan.rows ∧
      (∀ row ∈ plan.rows, row.VarsBelow plan.next) := by
  have scope : ∀ expression ∈ plan.constraints, expression.VarsBelow plan.firstFresh := by
    rw [constraints, firstFresh]
    exact builtPrefix.scope
  obtain ⟨completed, agrees, rows⟩ := R1CS.LoweringPlan.complete plan builtPrefix.current scope (by
    rw [constraints]
    exact builtPrefix.rows)
  refine ⟨completed, agrees, rows, ?_⟩
  change ∀ row ∈ (R1CS.lowerConstraints plan.constraints plan.firstFresh).rows, row.VarsBelow plan.next
  rw [R1CS.LoweringPlan.next_eq]
  exact R1CS.lowerConstraints_rows_varsBelow plan.constraints plan.firstFresh scope

private theorem pilot_end_before_c :
    Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤ PiCCSInputs.phaseOffset := by
  rw [← PiCCSInputs.expectedContextStart_matches_pilot]
  unfold PiCCSInputs.phaseOffset PiCCSInputs.proofInputStart
  omega

private theorem pilot_start_le_end :
    PilotProduction.witnessOffset ≤ Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset := by
  rw [Pilot.logicalColumnCount_eq_add, Pilot.outputOffset_eq_add]
  omega

private theorem pilot_logical_le_physical :
    Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤
      Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset := by
  rw [Pilot.physicalColumnCount_eq]
  exact Nat.le_add_right _ _

private theorem external_outside_pilot_physical (index : Nat)
    (external : PiCCSOrdinarySourceSupport.External index) :
    index < PilotProduction.witnessOffset ∨
      Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤ index := by
  rcases external with priorRange | publicRange | outputRange | contextRange | proofRange
  · apply Or.inl
    rcases priorRange with ⟨_, upper⟩
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
      PilotProduction.outputPreimageStart PilotProduction.priorPublicInputStart
    omega
  · apply Or.inl
    rcases publicRange with ⟨_, upper⟩
    change index < PilotProduction.priorPublicInputStart + PriorStateHash.publicWidth at upper
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
      PilotProduction.outputPreimageStart
    omega
  · apply Or.inl
    rcases outputRange with ⟨_, upper⟩
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
    omega
  · exact Or.inr (by rw [← PiCCSInputs.expectedContextStart_matches_pilot]; exact contextRange.1)
  · apply Or.inr
    rw [← PiCCSInputs.expectedContextStart_matches_pilot]
    have lower := proofRange.1
    unfold PiCCSInputs.proofInputStart at lower
    omega


private theorem external_before_c (index : Nat)
    (external : PiCCSOrdinarySourceSupport.External index) : index < PiCCSInputs.phaseOffset := by
  have early : PilotProduction.witnessOffset ≤ PiCCSInputs.phaseOffset :=
    pilot_start_le_end.trans (pilot_logical_le_physical.trans pilot_end_before_c)
  rcases external with priorRange | publicRange | outputRange | contextRange | proofRange
  · apply Nat.lt_of_lt_of_le _ early
    have upper := priorRange.2
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
      PilotProduction.outputPreimageStart PilotProduction.priorPublicInputStart
    omega
  · apply Nat.lt_of_lt_of_le _ early
    have upper := publicRange.2
    change index < PilotProduction.priorPublicInputStart + PriorStateHash.publicWidth at upper
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
      PilotProduction.outputPreimageStart
    omega
  · apply Nat.lt_of_lt_of_le _ early
    have upper := outputRange.2
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
    omega
  · have upper := contextRange.2
    unfold PiCCSInputs.phaseOffset PiCCSInputs.proofInputStart
    omega
  · have upper := proofRange.2
    have before : PiCCSInputs.proofInputStart ≤ PiCCSInputs.phaseOffset := Nat.le_add_right _ _
    simpa only [Nat.add_sub_of_le before] using upper

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem c_end_before_r (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (NightstreamFPrime.Layout.PiCCS.v1_1.plan relation
      (PiCCSProofInputs.relationInterface relation) PiCCSInputs.phaseOffset).next ≤ PiRLCInputs.phaseOffset := by
  have bound := Nat.le_max_right
    (Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset)
    (NightstreamFPrime.Layout.PiCCS.v1_1.physicalColumnCount relation
      (PilotPiCCS.interface (publicFits := publicFits)) PilotPiCCS.piCcsOffset)
  change _ ≤ PilotPiCCS.physicalColumnCount relation at bound
  rw [PilotPiCCS.physicalColumnCount_eq relation] at bound
  exact bound

private theorem r_end_before_d (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (NightstreamFPrime.Layout.PiRLC.v1_1.plan relation PiRLCInputs.interface PiRLCInputs.phaseOffset).next ≤
      PiDECInputs.proofInputStart := by
  have bound := Nat.le_max_right (PilotPiCCS.physicalColumnCount relation)
    (NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnCount relation PiRLCInputs.interface PilotPiCCSPiRLC.piRlcOffset)
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

private theorem firstFresh_le_next (plan : R1CS.LoweringPlan) : plan.firstFresh ≤ plan.next := by
  rw [R1CS.LoweringPlan.next_eq]
  exact Nat.le_add_right _ _

private theorem nifs_complete
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
    (accepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
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
  let initial := PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
    priorFixed advertisedFixed digestFixed values context
  let proof := PiCCSProofInputs.relationProof relation values template
  let fresh := PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values
  obtain ⟨cAccepted, available, _, _, _⟩ := PiDECProtocolCompleteness.verifierInputs relation ajtai
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
    priorPc advertisedPc priorContext advertisedContext cAccepted pPhysical (fun index support => pAgrees index (Or.inr support))
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
  obtain ⟨r, rOperations, _, _, rSampled, rParent⟩ := PiRLCProtocolCompleteness.completePrefix_after_c
    relation ajtai prior (encHash (stateHash prior)) advertised digest priorFixed advertisedFixed digestFixed
    values context template available pPhysical (fun index support => pAgrees index (Or.inr support))
    c cOperations cPhysical (by simpa only [cFirst] using cAfter)
  let rPlan := NightstreamFPrime.Layout.PiRLC.v1_1.plan relation PiRLCInputs.interface PiRLCInputs.phaseOffset
  have rConstraints : rPlan.constraints = flatConstraints r.operations := by
    rw [rOperations]
    rfl
  have rFirst : rPlan.firstFresh = PiRLCInputs.phaseOffset + localLength r.operations := by
    rw [rOperations, ← PiRLC.v1_1.Formal.main_ops]
    exact NightstreamFPrime.Layout.PiRLC.v1_1.logicalColumnCount_eq relation _ _
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
  refine ⟨dPhysical, ?_, ?_, ?_, ?_⟩
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
  · exact (StepWitnessPrefix.piDecOutput_eq_of_agree relation dPhysical d.current dAfter).trans dOutput
  · intro index support
    have beforeC : index < PiCCSInputs.phaseOffset := by
      rcases support with early | external
      · exact early.trans_le (pilot_start_le_end.trans (pilot_logical_le_physical.trans pilot_end_before_c))
      · exact external_before_c index external
    exact (dBefore index (beforeC.trans_le (cLeR.trans rLeD))).trans
      ((rBefore index (beforeC.trans_le cLeR)).trans ((cBefore index beforeC).trans (pAgrees index support)))

private theorem c_before_running : PiCCSInputs.phaseOffset ≤ RunningTransitionInputs.phaseOffset := by
  have cR : PiCCSInputs.phaseOffset ≤ PiRLCInputs.phaseOffset := by
    have bound := PiRLCInputs.piCcsLogicalFreshBase_le_phaseOffset
    unfold PiCCSStarts.logicalFreshBase at bound
    exact (Nat.le_add_right _ _).trans bound
  exact cR.trans ((Nat.le_add_right _ _).trans (PiDECProtocolCompleteness.rEnd_before_dInputs.trans
    ((Nat.le_add_right _ _).trans RunningTransitionInputs.piDecPhaseOffset_le)))

private theorem pilot_before_running : PilotProduction.witnessOffset ≤ RunningTransitionInputs.phaseOffset :=
  pilot_start_le_end.trans (pilot_logical_le_physical.trans (pilot_end_before_c.trans c_before_running))

private theorem running_before_source_end : RunningTransitionInputs.phaseOffset ≤ Spartan.SourceColumnCount := by
  rw [Spartan.sourceColumnCount_eq_physicalEnd]
  unfold RunningTransitionLayout.physicalEnd RunningTransitionLayout.logicalColumnCount
  omega

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

private def spartan_copy (source : Env) : Env := fun column =>
  if column = Spartan.constantColumn then 1
  else ((Spartan.spartanToSource column).map source).getD 0

private theorem spartan_copy_source (source : Env) (column : Nat) (bound : column < Spartan.SourceColumnCount) :
    Spartan.pullback (spartan_copy source) column = source column := by
  simp only [Spartan.pullback, spartan_copy, if_neg (Spartan.sourceToSpartan_ne_constant column bound),
    Spartan.spartanToSource_sourceToSpartan column bound, Option.map_some, Option.getD_some]

/-- The selected semantic step and its accepted NIFS run construct one
Spartan witness through the running transition. The theorem derives the
physical rows, next-preimage rows, actual NIFS result, and exact protocol
source readback. It assumes no physical rows, generated phase specification,
or environment agreement. The application suffix is appended separately from
its actual witness. -/
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
    ∃ (digestFixed : output.x.length = PilotProduction.digestWords), ∃ target : Env,
      target Spartan.constantColumn = 1 ∧
      R1CS.RowsHold target (Spartan.remappedRows relation) ∧
      holdsFlat (Spartan.pullback target) (Lifecycle.Stage1.NextPreimage.opsAt
        NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset) ∧
      RunningTransitionInputs.piDecRunningOutput relation (Spartan.pullback target) = result ∧
      (∀ index, index < PilotProduction.witnessOffset ∨ PiCCSOrdinarySourceSupport.External index →
        Spartan.pullback target index = PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior))
          next output.x priorWellFormed.1 nextWellFormed.1 digestFixed values context index) ∧
      (∀ index : Fin PilotProduction.stateHashWords,
        Spartan.pullback target (PilotProduction.priorPreimageStart + index.val) =
          (serializePreimage (publicFits := publicFits) prior).getD index.val 0) ∧
      (∀ index : Fin PilotProduction.stateHashWords,
        Spartan.pullback target (PilotProduction.outputPreimageStart + index.val) =
          (serializePreimage (publicFits := publicFits) next).getD index.val 0) := by
  let prior := priorHashPreimage (setup relation ajtai context.toList) input
  let next := nextHashPreimage (setup relation ajtai context.toList) input output
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
  have acceptedSource : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values)
      (PiCCSProofInputs.relationProof relation values input.nifsProof) = some result := by
    rw [fresh, proofReadback]
    exact accepted
  obtain ⟨nifs, nifsRows, nifsScope, nifsOutput, sourceReadback⟩ := nifs_complete relation ajtai prior next output.x
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
  have fullScope : ∀ row ∈ PilotPiCCSPiRLCPiDECRunningTransition.physicalRows relation,
      row.VarsBelow Spartan.SourceColumnCount := by
    intro row member
    rcases List.mem_append.mp member with earlier | last
    · exact (nifsScope row earlier).mono row running_before_source_end
    · have scope := RunningTransitionLayout.physicalRows_varsBelow relation row last
      rwa [RunningTransitionLayout.physicalColumnCount_eq relation, ← Spartan.sourceColumnCount_eq] at scope
  have finalOutput : RunningTransitionInputs.piDecRunningOutput relation completed = result :=
    (StepWitnessPrefix.piDecOutput_eq_of_agree relation completed nifs (fun index below =>
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
  obtain ⟨nextEnv, nextAgrees, nextRows⟩ := Lifecycle.Stage1.NextPreimage.completeness
    NextPreimageInputs.sourceInterface completed RunningTransitionInputs.phaseOffset finalSpecs.2
  have nextEnvEq : nextEnv = completed := by
    funext index
    apply nextAgrees index
    rw [Lifecycle.Stage1.NextPreimage.localLength_eq]
    omega
  rw [nextEnvEq, Lifecycle.Stage1.NextPreimage.main_ops] at nextRows
  let target := spartan_copy completed
  have sourceCopied := spartan_copy_source completed
  have targetRows : R1CS.RowsHold target (Spartan.remappedRows relation) := by
    apply (Spartan.remappedRows_hold relation target).2
    exact R1CS.rowsHold_of_agree_below _ _ completed (Spartan.pullback target) fullScope sourceCopied physical
  have targetNext : holdsFlat (Spartan.pullback target) (Lifecycle.Stage1.NextPreimage.opsAt
      NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset) := by
    have scope := Lifecycle.Stage1.NextPreimage.flatConstraints_varsBelow NextPreimageInputs.sourceInterface
      RunningTransitionInputs.phaseOffset completed (NextPreimageInputs.sourceAssumptions completed)
    rw [Lifecycle.Stage1.NextPreimage.main_ops] at scope
    intro expression member
    exact (expression.eval_eq_of_agree_below RunningTransitionInputs.phaseOffset (Spartan.pullback target) completed
      (scope expression member) (fun index below => sourceCopied index (below.trans_le running_before_source_end))).trans
      (nextRows expression member)
  have targetOutput : RunningTransitionInputs.piDecRunningOutput relation (Spartan.pullback target) = result :=
    (StepWitnessPrefix.piDecOutput_eq_of_agree relation (Spartan.pullback target) completed (fun index below =>
      sourceCopied index (below.trans_le Spartan.sourceColumnCount_ge_piDecPhaseOffset))).trans finalOutput
  refine ⟨digestFixed, target, ?_, targetRows, targetNext, targetOutput, ?_, ?_, ?_⟩
  · simp [target, spartan_copy]
  · intro index support
    have below : index < Spartan.SourceColumnCount := support.elim
      (fun h => h.trans_le (pilot_before_running.trans running_before_source_end))
      (fun h => (external_before_c index h).trans_le (c_before_running.trans running_before_source_end))
    exact (sourceCopied index below).trans (finalReadback index support)
  · intro index
    exact (sourceCopied _ ((prior_word_below index).trans_le (pilot_before_running.trans running_before_source_end))).trans
      (finalPrior index)
  · intro index
    exact (sourceCopied _ ((next_word_below index).trans_le (pilot_before_running.trans running_before_source_end))).trans
      (finalNext index)

end NightstreamFPrime.Layout.Stage1.StepPhysicalCompleteness
