import NightstreamFPrime.Export.Stage1.PiCCSCompletedReadout
import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Layout.Stage1.SpartanRows

/-!
Owns the running-transition plan on the completed canonical assignment.
Actual PiCCS rows determine the transcript readout; all other physical values
are copied by the existing source constructor.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionCompletedAssignment

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PerApplicationAssignmentTransportExecution

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The actual cumulative physical rows imply the running-transition rows
of the same canonical low-norm assignment, including the computed transcript
point. No transcript-coherence or retained-row premise is supplied. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    let raw := canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (RunningTransitionDirectPlan.plan relation
      (PerApplicationCanonicalEncodes.runningGeometry application)).RowsZero raw.assignment := by
  intro raw
  have allRows := (Spartan.remappedRows_hold relation target).mp physical
  have throughRunning := (PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation
    (Spartan.pullback target)).mp allRows
  have throughD := (PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation
    (Spartan.pullback target)).mp throughRunning.1
  have throughR := (PilotPiCCSPiRLC.physicalHolds_iff relation
    (Spartan.pullback target)).mp throughD.1
  have throughC := (PilotPiCCS.physicalHolds_iff relation
    (Spartan.pullback target)).mp throughR.1
  apply (RunningTransitionDirectPlan.rowsZero_iff_physical relation
    (PerApplicationCanonicalEncodes.runningGeometry application) raw.assignment
    raw.base raw.groupValue raw.products
    (PerApplicationCanonicalAssignment.assignment_one raw)
    (PerApplicationCanonicalEncodes.runningPrefixEncodes raw).transition).mpr
  apply R1CS.rowsHold_of_agree_below _ Spartan.SourceColumnCount
    (Spartan.pullback target) _ _ _ throughRunning.2
  · intro row member
    have bounded := RunningTransitionLayout.physicalRows_varsBelow relation row member
    rw [RunningTransitionLayout.physicalColumnCount_eq_physicalEnd relation,
      ← Spartan.sourceColumnCount_eq_physicalEnd] at bounded
    exact bounded
  · exact PiCCSCompletedReadout.transitionEnv_of_completed application relation target suffix throughC.2

end NightstreamFPrime.Export.Stage1.RunningTransitionCompletedAssignment
