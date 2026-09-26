import NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceInputs
import NightstreamFPrime.Export.Stage1.PiCCSCompletedAssignment
import NightstreamFPrime.Export.Stage1.PilotCompletedAssignment

/-! Complete the compact pilot/PiCCS prefix on the same wide source assignment.
Only accepted physical prefix rows are required; no old PiRLC rows are copied. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PrefixCompletedAssignment

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint

private theorem pilot_scope (interface : Lifecycle.Pilot.Interface) (offset : Nat) (env : Env)
    (assumptions : Lifecycle.Pilot.Assumptions interface offset env) :
    ∀ row ∈ Layout.Pilot.physicalRows interface offset,
      row.VarsBelow (Layout.Pilot.physicalColumnCount interface offset) := by
  change ∀ row ∈ (R1CS.lowerConstraints (Layout.Pilot.logicalConstraints interface offset)
    (Layout.Pilot.logicalColumnCount interface offset)).rows,
    row.VarsBelow (R1CS.lowerConstraints (Layout.Pilot.logicalConstraints interface offset)
      (Layout.Pilot.logicalColumnCount interface offset)).next
  rw [R1CS.lowerConstraints_next]
  exact R1CS.lowerConstraints_rows_varsBelow _ _ (Layout.Pilot.logicalConstraints_varsBelow interface offset assumptions)

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

/-- The wide source view preserves both physical prefix phases. -/
theorem physical_prefix (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9) (env : Env)
    (pilotAssumptions : Lifecycle.Pilot.Assumptions Layout.PilotProduction.interface Layout.PilotProduction.witnessOffset env)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (physical : Layout.Stage1.PilotPiCCS.PhysicalHolds relation env) :
    Layout.Stage1.PilotPiCCS.PhysicalHolds relation
      (Layout.Stage1.Spartan.pullback (SourceAssignment.targetEnv env)) := by
  obtain ⟨pilotRows, cRows⟩ := (Layout.Stage1.PilotPiCCS.physicalHolds_iff relation env).mp physical
  apply (Layout.Stage1.PilotPiCCS.physicalHolds_iff relation _).mpr
  refine ⟨?_, PiRLCSourceInputs.piCcs_physical env relation ajtai template cAssumptions cRows⟩
  have scope := pilot_scope Layout.PilotProduction.interface Layout.PilotProduction.witnessOffset env pilotAssumptions
  have endpoint : Layout.Pilot.physicalColumnCount Layout.PilotProduction.interface Layout.PilotProduction.witnessOffset ≤
      SourceAssignment.prefixEnd := by
    rw [SourceAssignment.prefixEnd, Layout.Stage1.PiRLCInputs.phaseOffset_matches_piCcs relation]
    exact Nat.le_max_left _ _
  apply R1CS.rowsHold_of_agree_below _ _ env _ scope _ pilotRows
  intro source below
  have prefixBound := lt_of_lt_of_le below endpoint
  change SourceAssignment.targetEnv env (Layout.Stage1.Spartan.sourceToSpartan source) = env source
  rw [SourceAssignment.targetEnv_source env source (by
    change source < 19513117 at prefixBound
    rw [Layout.Stage1.Spartan.sourceColumnCount_eq]; omega)]
  exact SourceAssignment.sourceEnv_prefix env source prefixBound

/-- The six reference components accept the copied-base packet. -/
theorem reference_rowsZero (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (pilotAssumptions : Lifecycle.Pilot.Assumptions Layout.PilotProduction.interface Layout.PilotProduction.witnessOffset env)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (physical : Layout.Stage1.PilotPiCCS.PhysicalHolds relation env) :
    (DirectPiDECPrefixPlan.piCcsCompletePlan relation (Stage1Plan.piDecGeometry program)).RowsZero
      (SourceAssignment.raw program env application).assignment := by
  let raw := SourceAssignment.raw program env application
  let target := SourceAssignment.targetEnv env
  have prefixRows := physical_prefix relation ajtai template env pilotAssumptions cAssumptions physical
  have parts := (Layout.Stage1.PilotPiCCS.physicalHolds_iff relation _).mp prefixRows
  have hashes := PilotHashRowsCompleteness.hashChains_of_pilotRows target parts.1
  have hashRows := PilotPoseidonCompleteness.rowsZero_of_hashRows program target application raw rfl hashes.1 hashes.2
  have pilot := PilotCompletedAssignment.rowsZero_of_base program relation target application prefixRows raw rfl
  have c := PiCCSCompletedAssignment.rowsZero_of_base program relation target application parts.2 raw rfl
  rw [DirectPiDECPrefixPlan.piCcsCompletePlan, Plan.append_rowsZero_iff,
    DirectPiDECPrefixPlan.pilotBindingPrefixPlan, Plan.append_rowsZero_iff,
    DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan, Plan.append_rowsZero_iff,
    DirectPiDECPrefixPlan.piCcsCorePlan, Plan.append_rowsZero_iff,
    DirectPiDECPrefixPlan.piCcsPoseidonPrefix, Plan.append_rowsZero_iff]
  exact ⟨⟨⟨⟨⟨hashRows, c.2.1⟩, c.1⟩, pilot.1⟩, pilot.2⟩, c.2.2⟩

/-- Filling the direct PiRLC allocation preserves all 3,054,685 prefix rows. -/
theorem rowsZero (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (pilotAssumptions : Lifecycle.Pilot.Assumptions Layout.PilotProduction.interface Layout.PilotProduction.witnessOffset env)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (physical : Layout.Stage1.PilotPiCCS.PhysicalHolds relation env) :
    (Stage1Plan.prefixPlan program relation).RowsZero (SourceAssignment.assignment program env application) := by
  exact (AssignmentProjection.common_rowsZero_iff program (SourceAssignment.raw program env application).assignment
    _ (ReadSupport.prefixPlan program relation (Stage1Plan.piDecGeometry program)) _).mpr
      (reference_rowsZero program env application relation ajtai template pilotAssumptions cAssumptions physical)

end NightstreamFPrime.Export.Stage1.Wide.PrefixCompletedAssignment
