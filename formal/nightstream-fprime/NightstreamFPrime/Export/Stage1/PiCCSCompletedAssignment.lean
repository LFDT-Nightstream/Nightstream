import NightstreamFPrime.Export.Stage1.PiCCSCompletedReadout
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryPhysicalCompleteness
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectPlan
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonCompleteness
import NightstreamFPrime.Export.Stage1.PiCCSEndpointCompleteness
import NightstreamFPrime.Layout.Stage1.SpartanRows

/-!
Owns the canonical C-plan consumer for the completed Spartan assignment.
The source rows and computed transcript readout both come from the same
actual physical C phase; retained encodings come from the canonical carrier.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCompletedAssignment

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PerApplicationAssignmentTransportExecution

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem c_rows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env) (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target) := by
  have allRows := (Spartan.remappedRows_hold relation target).mp physical
  have throughD := (PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation _).mp allRows
  have throughR := (PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation _).mp throughD.1
  have throughC := (PilotPiCCSPiRLC.physicalHolds_iff relation _).mp throughR.1
  exact ((PilotPiCCS.physicalHolds_iff relation _).mp throughC.1).2

/-- Actual completed Spartan rows make the canonical direct C arithmetic
plan zero. Source agreement, transcript readout and encodings are derived;
the caller supplies no compact-row or retained-value assertion. -/
private theorem ordinaryRowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (PiCCSOrdinaryDirectPlan.plan relation
      (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)).RowsZero raw.assignment := by
  intro raw
  have cRows := c_rows relation target physical
  apply (PiCCSOrdinaryDirectPlan.rowsZero_iff_rowsHold relation
    (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)
    raw.assignment raw.base raw.groupValue raw.products
    (PerApplicationCanonicalAssignment.assignment_one raw)
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).prior.pilotOrdinary.prior).mpr
  apply R1CS.rowsHold_of_agree _ PiCCSOrdinarySourceSupport.Target target _
    (PiCCSOrdinaryDirectSupport.sourceRows_varsSatisfy relation)
  · intro column supported
    rcases supported with ⟨source, sourceSupport, rfl⟩
    exact PiCCSCompletedReadout.transitionEnv_of_completed application relation target suffix
      cRows source (PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount sourceSupport)
  · exact PiCCSOrdinaryPhysicalCompleteness.ordinaryRows_of_physical relation target cRows

/-- The canonical assignment satisfies every direct C component: ordinary
arithmetic, retained Poseidon permutations and all transcript endpoints.
The sole row premise is the actual cumulative Spartan physical relation. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (PiCCSOrdinaryDirectPlan.plan relation
      (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)).RowsZero raw.assignment ∧
    (PiCCSPoseidonPlan.plan
      (DirectPiDECPrefixPlan.piCcsPayload (PerApplicationCanonicalEncodes.piDecGeometry application))
      (PerApplicationCanonicalEncodes.poseidonGeometry application)).RowsZero raw.assignment ∧
    (PiCCSTranscriptEndpointPlan.plan
      (PerApplicationCanonicalEncodes.poseidonGeometry application)
      (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)).RowsZero raw.assignment := by
  intro raw
  have cRows := c_rows relation target physical
  exact ⟨ordinaryRowsZero_of_completed application relation target suffix physical,
    PiCCSPoseidonCompleteness.rowsZero_of_completed application relation target suffix cRows,
    PiCCSEndpointCompleteness.rowsZero_of_completed application relation target suffix cRows⟩

end NightstreamFPrime.Export.Stage1.PiCCSCompletedAssignment
