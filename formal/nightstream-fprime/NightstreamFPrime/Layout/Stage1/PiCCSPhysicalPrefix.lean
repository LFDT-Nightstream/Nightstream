import NightstreamFPrime.Layout.Stage1.PiCCSBoundary

/-! Pilot and PiCCS physical lowering and source bounds shared by the selected assembly. -/

namespace NightstreamFPrime.Layout.Stage1.PiCCSPhysicalPrefix

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

theorem lower_prefix {initial : Env} {offset : Nat}
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

theorem pilot_end_before_c :
    Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤ PiCCSInputs.phaseOffset := by
  rw [← PiCCSInputs.expectedContextStart_matches_pilot]
  unfold PiCCSInputs.phaseOffset PiCCSInputs.proofInputStart
  omega

theorem pilot_start_le_end :
    PilotProduction.witnessOffset ≤ Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset := by
  rw [Pilot.logicalColumnCount_eq_add, Pilot.outputOffset_eq_add]
  omega

theorem pilot_logical_le_physical :
    Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤
      Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset := by
  rw [Pilot.physicalColumnCount_eq]
  exact Nat.le_add_right _ _

theorem external_outside_pilot_physical (index : Nat)
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

theorem external_before_c (index : Nat)
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

theorem c_end_before_r (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (NightstreamFPrime.Layout.PiCCS.v1_1.plan relation
      (PiCCSProofInputs.relationInterface relation) PiCCSInputs.phaseOffset).next ≤ PiRLCInputs.phaseOffset := by
  have bound := Nat.le_max_right
    (Pilot.physicalColumnCount PilotProduction.interface PilotProduction.witnessOffset)
    (NightstreamFPrime.Layout.PiCCS.v1_1.physicalColumnCount relation
      (PilotPiCCS.interface (publicFits := publicFits)) PilotPiCCS.piCcsOffset)
  change _ ≤ PilotPiCCS.physicalColumnCount relation at bound
  rw [PilotPiCCS.physicalColumnCount_eq relation] at bound
  exact bound

theorem firstFresh_le_next (plan : R1CS.LoweringPlan) : plan.firstFresh ≤ plan.next := by
  rw [R1CS.LoweringPlan.next_eq]
  exact Nat.le_add_right _ _

theorem prior_word_below (index : Fin PilotProduction.stateHashWords) :
    PilotProduction.priorPreimageStart + index.val < PilotProduction.witnessOffset := by
  have bound := index.isLt
  unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
    PilotProduction.outputPreimageStart PilotProduction.priorPublicInputStart
  omega

theorem next_word_below (index : Fin PilotProduction.stateHashWords) :
    PilotProduction.outputPreimageStart + index.val < PilotProduction.witnessOffset := by
  have bound := index.isLt
  unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
  omega

end NightstreamFPrime.Layout.Stage1.PiCCSPhysicalPrefix
