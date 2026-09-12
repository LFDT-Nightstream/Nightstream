import NightstreamFPrime.Export.Stage1.PilotHashRowsCompleteness
import NightstreamFPrime.Export.Stage1.PilotOrdinaryPhysicalCompleteness
import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan
import NightstreamFPrime.Export.Stage1.PilotDigestBindingPlan
import NightstreamFPrime.Export.Stage1.PiCCSCompletedReadout
import NightstreamFPrime.Layout.Stage1.SpartanRows

/-!
Owns the pilot ordinary and digest-binding plans on the canonical completed
assignment. Source copies and both final hash outputs come from the same
physical prefix. No compact-row or output-coherence premise is supplied.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PilotCompletedAssignment

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Package
open PerApplicationAssignmentTransportExecution

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem phases
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env) (rows : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    NightstreamFPrime.Layout.Pilot.PhysicalHolds PilotProduction.interface
      PilotProduction.witnessOffset (Spartan.pullback target) ∧
    NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target) := by
  have allRows := (Spartan.remappedRows_hold relation target).mp rows
  have throughD := (PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation _).mp allRows
  have throughR := (PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation _).mp throughD.1
  have throughC := (PilotPiCCSPiRLC.physicalHolds_iff relation _).mp throughR.1
  exact (PilotPiCCS.physicalHolds_iff relation _).mp throughC.1

private theorem copied_pilotEnv
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (rows : R1CS.RowsHold target (Spartan.remappedRows relation))
    (column : Nat) (support : PilotOrdinaryDirectSource.Target column) :
    PilotOrdinaryDirectPlan.pilotEnv application
      (PerApplicationSourceAssignment.ofCompleted application target suffix) column =
      target (Spartan.liftPilotColumn column) := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  have pilotBound : source < PilotSpartan.SourceColumnCount := by
    change source < PilotValues.sourceColumnCount
    exact PilotOrdinaryDirectSource.physicalSource_lt source sourceSupport
  have bound : source < Spartan.SourceColumnCount := by
    rw [PilotSpartan.sourceColumnCount_eq] at pilotBound
    rw [Spartan.sourceColumnCount_eq]
    omega
  have mapped : Spartan.sourceToSpartan source =
      Spartan.liftPilotColumn (PilotSpartan.sourceToSpartan source) := by
    unfold Spartan.sourceToSpartan
    rw [if_pos (by simpa only [Spartan.pilotSourceColumnCount_matches] using pilotBound)]
  have same := PiCCSCompletedReadout.transitionEnv_of_completed application relation target suffix
    (phases relation target rows).2 source bound
  change RunningTransitionDirectPlan.transitionEnv application
    (PerApplicationSourceAssignment.ofCompleted application target suffix) (Spartan.sourceToSpartan source) =
      target (Spartan.sourceToSpartan source) at same
  change RunningTransitionDirectPlan.transitionEnv application
    (PerApplicationSourceAssignment.ofCompleted application target suffix)
      (Spartan.liftPilotColumn (PilotSpartan.sourceToSpartan source)) = _
  exact (congrArg (RunningTransitionDirectPlan.transitionEnv application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)) mapped.symm).trans
    (same.trans (congrArg target mapped))

private theorem form_value
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (rows : R1CS.RowsHold target (Spartan.remappedRows relation))
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (baseEq : raw.base = PerApplicationSourceAssignment.ofCompleted application target suffix)
    (location : PilotOrdinaryDirectPlan.Location) :
    (location.form (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)).eval raw.assignment =
      target (Spartan.liftPilotColumn (PilotSpartan.sourceToSpartan location.sourceColumn)) := by
  have value := PilotOrdinaryDirectPlan.Location.form_eval
    (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application) raw.assignment
    raw.base raw.groupValue raw.products
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).prior.pilotOrdinary location
  refine value.trans ?_
  rw [baseEq]
  exact copied_pilotEnv application relation target suffix rows _
    ⟨location.sourceColumn, location.physicalSupport, rfl⟩

private theorem lifted_chainOutput (chain : HashChain) (target : Env)
    (privateStart : Spartan.pilotInputPrivateColumnCount ≤ chain.witnessStart)
    (privateEnd : chain.witnessStart + (chain.absorbCount + 1) * 592 ≤ Spartan.pilotPrivateColumnCount)
    (lane : Fin 8) :
    Pilot.chainOutputState chain chain.absorbCount
      (fun column => target (Spartan.liftPilotColumn column)) lane =
      target (invocationLocalStart (PilotData.circuitPackage ()) (Data.liftPilotChain chain)
        chain.absorbCount + 584 + lane.val) := by
  change target (Spartan.liftPilotColumn
      (chain.witnessStart + chain.absorbCount * 592 + 584 + lane.val)) =
    target (Spartan.liftPilotColumn chain.witnessStart + chain.absorbCount * 592 + 584 + lane.val)
  apply congrArg target
  have laneBound := lane.isLt
  have mapped := Spartan.liftPilotColumn_add_of_private chain.witnessStart
    (chain.absorbCount * 592 + 584 + lane.val) privateStart (by omega)
  simpa only [Nat.add_assoc] using mapped

private theorem ordinary_rows
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (rows : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (PilotOrdinaryDirectPlan.plan (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)).RowsZero
      raw.assignment := by
  intro raw
  apply (PilotOrdinaryDirectPlan.rowsZero_iff_rowsHold
    (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application) raw.assignment
    raw.base raw.groupValue raw.products (PerApplicationCanonicalAssignment.assignment_one raw)
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).prior.pilotOrdinary).mpr
  apply R1CS.rowsHold_of_agree _ PilotOrdinaryDirectSource.Target
    (fun column => target (Spartan.liftPilotColumn column)) _ PilotOrdinaryDirectSource.sourceRows_varsSatisfy
  · exact copied_pilotEnv application relation target suffix rows
  · exact PilotOrdinaryPhysicalCompleteness.rows_of_physical target (phases relation target rows).1

private theorem prior_binding
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (rows : R1CS.RowsHold target (Spartan.remappedRows relation))
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (baseEq : raw.base = PerApplicationSourceAssignment.ofCompleted application target suffix)
    (lane : Fin PilotDigestBindingPlan.laneCount) :
    (PilotDigestBindingPlan.legacyForm (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)
      (PilotDigestBindingPlan.priorRow lane)).eval raw.assignment =
    (PilotDigestBindingPlan.derivedForm (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)
      (PilotDigestBindingPlan.priorRow lane)).eval raw.assignment := by
  let geometry := PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application
  have hashes := PilotHashRowsCompleteness.hashChains_of_pilotRows target (phases relation target rows).1
  have legacy := PilotOrdinaryDirectPlan.priorDigest_form_eval_chainOutput geometry raw.assignment
    (fun column => target (Spartan.liftPilotColumn column))
    (fun selected => form_value application relation target suffix rows raw baseEq (.priorDigest selected)) lane
  have lifted := lifted_chainOutput PilotData.priorChain target
    (by decide : Spartan.pilotInputPrivateColumnCount ≤ PilotData.priorChain.witnessStart)
    (by decide : PilotData.priorChain.witnessStart + (PilotData.priorChain.absorbCount + 1) * 592 ≤
      Spartan.pilotPrivateColumnCount) (PilotDigestBindingPlan.digestLane lane)
  have retained := congrFun
    (PilotPoseidonCompleteness.prior_output application target suffix raw baseEq hashes.1
      PilotDigestBindingPlan.lastInvocation) (PilotDigestBindingPlan.digestLane lane)
  have last : PilotDigestBindingPlan.lastInvocation.val = PilotData.priorChain.absorbCount := rfl
  conv at retained =>
    rhs
    rw [last]
  rw [PilotDigestBindingPlan.legacyForm_priorRow, PilotDigestBindingPlan.derivedForm_priorRow]
  change ((PilotOrdinaryDirectPlan.Location.priorDigest lane).form geometry).eval raw.assignment = _
  exact legacy.trans (lifted.trans retained.symm)

private theorem output_binding
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (rows : R1CS.RowsHold target (Spartan.remappedRows relation))
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (baseEq : raw.base = PerApplicationSourceAssignment.ofCompleted application target suffix)
    (lane : Fin PilotDigestBindingPlan.laneCount) :
    (PilotDigestBindingPlan.legacyForm (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)
      (PilotDigestBindingPlan.outputRow lane)).eval raw.assignment =
    (PilotDigestBindingPlan.derivedForm (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)
      (PilotDigestBindingPlan.outputRow lane)).eval raw.assignment := by
  let geometry := PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application
  have hashes := PilotHashRowsCompleteness.hashChains_of_pilotRows target (phases relation target rows).1
  have legacy := PilotOrdinaryDirectPlan.outputState_form_eval_chainOutput geometry raw.assignment
    (fun column => target (Spartan.liftPilotColumn column))
    (fun selected => form_value application relation target suffix rows raw baseEq (.outputState selected)) lane
  have lifted := lifted_chainOutput PilotData.outputChain target
    (by decide : Spartan.pilotInputPrivateColumnCount ≤ PilotData.outputChain.witnessStart)
    (by decide : PilotData.outputChain.witnessStart + (PilotData.outputChain.absorbCount + 1) * 592 ≤
      Spartan.pilotPrivateColumnCount) (PilotDigestBindingPlan.digestLane lane)
  have retained := congrFun
    (PilotPoseidonCompleteness.output_output application target suffix raw baseEq hashes.2
      PilotDigestBindingPlan.lastInvocation) (PilotDigestBindingPlan.digestLane lane)
  have last : PilotDigestBindingPlan.lastInvocation.val = PilotData.outputChain.absorbCount := rfl
  conv at retained =>
    rhs
    rw [last]
  rw [PilotDigestBindingPlan.legacyForm_outputRow, PilotDigestBindingPlan.derivedForm_outputRow]
  change ((PilotOrdinaryDirectPlan.Location.outputState lane).form geometry).eval raw.assignment = _
  exact legacy.trans (lifted.trans retained.symm)

private theorem binding_rows
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (rows : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (PilotDigestBindingPlan.plan (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)).RowsZero
      raw.assignment := by
  intro raw
  apply (PilotDigestBindingPlan.rowsZero_iff_matches
    (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application) raw.assignment
    (PerApplicationCanonicalAssignment.assignment_one raw)).mpr
  intro row
  let decoded := PilotDigestBindingPlan.descriptor row
  have inverse : Fin.encodeProd decoded = row := Fin.encodeProd_decodeProd row
  by_cases prior : decoded.1.val = 0
  · have chain : decoded.1 = PilotDigestBindingPlan.priorChain := by
      apply Fin.ext
      exact prior
    have position : row = PilotDigestBindingPlan.priorRow decoded.2 := by
      rw [← inverse]
      exact congrArg Fin.encodeProd (Prod.ext chain rfl)
    rw [position]
    exact prior_binding application relation target suffix rows raw rfl decoded.2
  · have chain : decoded.1 = PilotDigestBindingPlan.outputChain := by
      apply Fin.ext
      have bound := decoded.1.isLt
      change decoded.1.val < 2 at bound
      change decoded.1.val = 1
      omega
    have position : row = PilotDigestBindingPlan.outputRow decoded.2 := by
      rw [← inverse]
      exact congrArg Fin.encodeProd (Prod.ext chain rfl)
    rw [position]
    exact output_binding application relation target suffix rows raw rfl decoded.2

/-- The canonical completed assignment satisfies the pilot ordinary plan
and all eight digest-binding rows. Both results derive their source values
from the same cumulative physical prefix. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (rows : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (PilotOrdinaryDirectPlan.plan (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)).RowsZero
      raw.assignment ∧
    (PilotDigestBindingPlan.plan (PerApplicationCanonicalEncodes.pilotOrdinaryGeometry application)).RowsZero
      raw.assignment := by
  intro raw
  exact ⟨ordinary_rows application relation target suffix rows,
    binding_rows application relation target suffix rows⟩

end NightstreamFPrime.Export.Stage1.PilotCompletedAssignment
