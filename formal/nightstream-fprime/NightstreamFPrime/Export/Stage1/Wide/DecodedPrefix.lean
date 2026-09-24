import NightstreamFPrime.Export.Stage1.Wide.AssignmentPullback
import NightstreamFPrime.Export.Stage1.PilotDecodedPhase
import NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase

/-! Decode the reused pilot and PiCCS prefix from any accepted candidate
assignment. Checked support makes the reference-address view exact; no
canonical witness or encoding premise is required. -/

namespace NightstreamFPrime.Export.Stage1.Wide.DecodedPrefix

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ProductionRelation

def pilotEnv (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program)) : Env :=
  Layout.PilotSpartan.pullback (PilotDecodedEnvironment.env
    (DirectPiDECPrefixPlan.pilotOrdinaryGeometry (Stage1Plan.piDecGeometry program))
    (AssignmentPullback.assignment program assignment))

def piCcsEnv (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program)) : Env :=
  Layout.Stage1.Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
    (Stage1Plan.piCcsGeometry program) (AssignmentPullback.assignment program assignment))

theorem reference_one (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1) :
    AssignmentPullback.assignment program assignment
      (ApplicationRetainedGeometry.oneColumn (Stage1Plan.referenceGeometry program)) = 1 := by
  rw [AssignmentPullback.at_live program assignment _ (ReadSupport.one program _ rfl)]
  exact one

variable {width : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
  (program : RetainedLayout.Program) (relation : ProductionKey.LogicalRelation width publicFits)
  (assignment : Assignment F (RetainedLayout.logicalWidth program))

private theorem prefix_rows
    (rows : (Stage1Plan.prefixPlan program relation).RowsZero assignment) :
    (DirectPiDECPrefixPlan.piCcsCompletePlan relation (Stage1Plan.piDecGeometry program)).RowsZero
      (AssignmentPullback.assignment program assignment) :=
  (AssignmentPullback.rowsZero_iff program assignment _ _).mp rows

theorem pilot
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.prefixPlan program relation).RowsZero assignment) :
    Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset
      (pilotEnv program assignment) := by
  have parts := prefix_rows program relation assignment rows
  simp only [DirectPiDECPrefixPlan.piCcsCompletePlan,
    DirectPiDECPrefixPlan.pilotBindingPrefixPlan, DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan,
    DirectPiDECPrefixPlan.piCcsCorePlan, DirectPiDECPrefixPlan.piCcsPoseidonPrefix,
    Plan.append_rowsZero_iff] at parts
  obtain ⟨⟨⟨⟨⟨hashes, _⟩, _⟩, ordinary⟩, binding⟩, _⟩ := parts
  exact PilotDecodedPhase.rowsZero_implies_specHolds _ _
    (reference_one program assignment one) hashes ordinary binding

theorem piCcsSpec
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.prefixPlan program relation).RowsZero assignment) :
    PiCCS.v1_1.Formal.SpecHolds relation (PiCCSInvocations.parentInterface width publicFits)
      Layout.Stage1.PiCCSInputs.phaseOffset (piCcsEnv program assignment) := by
  have parts := prefix_rows program relation assignment rows
  simp only [DirectPiDECPrefixPlan.piCcsCompletePlan,
    DirectPiDECPrefixPlan.pilotBindingPrefixPlan, DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan,
    DirectPiDECPrefixPlan.piCcsCorePlan, DirectPiDECPrefixPlan.piCcsPoseidonPrefix,
    Plan.append_rowsZero_iff] at parts
  obtain ⟨⟨⟨⟨⟨_, transcript⟩, ordinary⟩, _⟩, _⟩, endpoints⟩ := parts
  exact PiCCSDecodedPhase.rowsZero_implies_specHolds relation
    (Stage1Plan.piCcsGeometry program) (Stage1Plan.poseidonGeometry program)
    (AssignmentPullback.assignment program assignment) (reference_one program assignment one)
    ordinary transcript endpoints


theorem piCcs (ajtai : AjtaiKey (logicalWidth := width) (publicFits := publicFits))
    (template : Proof (ProductionKey.degreeBound relation))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.prefixPlan program relation).RowsZero assignment) :
    PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (PiCCSInvocations.parentInterface width publicFits) Layout.Stage1.PiCCSInputs.phaseOffset
      (piCcsEnv program assignment) template := by
  apply PiCCS.v1_1.Formal.spec_implies_phaseHolds
  exact piCcsSpec program relation assignment one rows

end NightstreamFPrime.Export.Stage1.Wide.DecodedPrefix
