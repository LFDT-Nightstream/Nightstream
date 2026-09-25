import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportSchedule
import NightstreamFPrime.Export.Stage1.Wide.OutputDigest

/-! Execution of the exact emitted schema-4 plan gives the existing direct
wide assignment and its padded carrier. The physical values are supplied
by the constructive completion, including its canonical range readback. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCorrectness

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open ProductionRelation CanonicalBlockAssignment
open Spec.Folding.PiCCS.PaperJoint
open AssignmentTransportExecution AssignmentTransportSemantics

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

variable (program : RetainedLayout.Program) (env : Env)
  (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
  (plan : AssignmentTransport.Plan)
  (emitted : AssignmentTransport.plan program (CommonSchedule.physicalWidth program) = .ok plan)
  (relation : ProductionKey.LogicalRelation width fits)
  (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
  (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
    (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
  (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
    (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
  (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
    (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env)
  (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
    (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env)
  (completed : PiRLC.Wide.Formal.RangesCompleted
    (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env)

include emitted relation ajtai template cAssumptions cRows rAssumptions rRows completed

theorem canonical_executeUnchecked_eq_assignment :
    executeUnchecked program plan (CommonSchedule.physicalWidth program) (physicalValues program env application) =
      SourceAssignment.assignment program env application := by
  have schedules := AssignmentTransportSchedule.schedule_correct program env application plan emitted relation ajtai template
    cAssumptions cRows rAssumptions rRows completed
  have digest := OutputDigest.value program env application plan emitted
  rw [executeUnchecked, digest, SourceAssignment.assignment,
    ← AssignmentTransportTail.canonical_assignment program (SourceAssignment.raw program env application)]
  funext column
  unfold CanonicalBlockAssignment.assignment
  split
  · rfl
  · exact schedules.2 _

/-- The fail-closed executable accepts the constructed physical witness and
returns the already proved direct logical assignment at every coordinate. -/
theorem canonical_execute_eq_assignment :
    execute program plan (CommonSchedule.physicalWidth program) (physicalValues program env application) =
      some (SourceAssignment.assignment program env application) := by
  obtain ⟨_, _, _, _, _, _, _, challenges, _⟩ :=
    emitted_parts program (CommonSchedule.physicalWidth program) plan emitted
  rw [execute_emitted program plan (CommonSchedule.physicalWidth program) (physicalValues program env application) emitted
    (SamplerSourceValues.digits_valid program env application plan challenges relation rRows completed)]
  rw [canonical_executeUnchecked_eq_assignment program env application plan emitted relation ajtai template
    cAssumptions cRows rAssumptions rRows completed]

/-- The same equality includes all committed padding coordinates. -/
theorem canonical_carrier_eq :
    Phi81CarrierLayout.extendAssignment (0 : F)
        (executeUnchecked program plan (CommonSchedule.physicalWidth program) (physicalValues program env application)) =
      CarrierAssignment.values program env application := by
  rw [canonical_executeUnchecked_eq_assignment program env application plan emitted relation ajtai template
    cAssumptions cRows rAssumptions rRows completed]
  rfl

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCorrectness
