import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Export.Stage1.PiDECDirectPlan
import NightstreamFPrime.Layout.Stage1.SpartanRows

/-!
Owns the PiDEC part of the completed canonical assignment. The actual physical
prefix supplies the PiDEC rows. The existing source copy preserves their
declared support, which lies after the PiCCS transcript readout interval.
No encoded-row or source-agreement premise is supplied by the caller.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECCompletedAssignment

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

private theorem sourceRows_support
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits,
      row.VarsSatisfy PiDECSourceSupport.Target := by
  intro row member
  change row ∈
    ((PiDECOrdinaryDirectSource.publicRows logicalWidth publicFits ++
      PiDECOrdinaryDirectSource.commitmentRows logicalWidth publicFits) ++
      PiDECOrdinaryDirectSource.evalKRows logicalWidth publicFits) ++
      PiDECOrdinaryDirectSource.evalARows logicalWidth publicFits at member
  rcases List.mem_append.mp member with beforeA | evalARow
  · rcases List.mem_append.mp beforeA with beforeK | evalKRow
    · rcases List.mem_append.mp beforeK with publicRow | commitmentRow
      · exact PiDECOrdinaryDirectSource.publicRows_varsSatisfy relation row publicRow
      · exact PiDECOrdinaryDirectSource.commitmentRows_varsSatisfy relation row commitmentRow
    · exact PiDECOrdinaryDirectSource.evalKRows_varsSatisfy relation row evalKRow
  · exact PiDECOrdinaryDirectSource.evalARows_varsSatisfy relation row evalARow

private theorem copied_source
    (application : Lifecycle.Stage1.Application.Program) (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (column : Nat) (supported : PiDECSourceSupport.Target column) :
    RunningTransitionDirectPlan.transitionEnv application
      (PerApplicationSourceAssignment.ofCompleted application target suffix) column = target column := by
  obtain ⟨source, support, rfl⟩ := supported
  have bounded := PiDECSourceSupport.source_lt_sourceColumnCount support
  have afterTranscript : PiCCSInputs.phaseOffset +
      PiCCSOrdinarySourceSupport.transcriptInvocationCount * 592 ≤ source := by
    apply Nat.le_trans _ (PiDECSourceSupport.parentStart_le_source support)
    rw [PiCCSInputs.phaseOffset_eq, PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq,
      PiDECSourceSupport.parentCommitmentStart_eq]
    decide
  rw [RunningTransitionDirectPlan.transitionEnv_of_outside application _ source bounded
    (Or.inr afterTranscript)]
  apply PerApplicationSourceAssignment.packageEnv_ofCompleted
  have mapped := Spartan.sourceToSpartan_lt source bounded
  change Spartan.sourceToSpartan source < PerApplicationPackage.basePackage.layout.totalColumnCount
  rw [PerApplicationPackage.basePackage_totalColumnCount_eq, Spartan.spartanColumnCount_eq] at *
  exact mapped

private theorem sourceRows_of_physical
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env) (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    R1CS.RowsHold target (PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits) := by
  have allRows := (Spartan.remappedRows_hold relation target).mp physical
  have throughD := (PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation
    (Spartan.pullback target)).mp allRows
  have dRows := ((PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation
    (Spartan.pullback target)).mp throughD.1).2
  rw [PiDECOrdinaryDirectSource.sourceRows_eq_canonical,
    PiDECArithmetic.Plan.rows_to_layout _ _ (PiDECArithmetic.canonicalPlan_matches relation)]
  exact (Spartan.remapRows_hold target _).mpr dRows

/-- The completed physical prefix makes the PiDEC rows of the exact canonical
low-norm assignment zero. Source transport and retained encodings are derived
from the existing constructor; they are not additional hypotheses. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    let raw := canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (PiDECDirectPlan.plan relation
      (PerApplicationCanonicalEncodes.piDecGeometry application)).RowsZero raw.assignment := by
  intro raw
  apply (PiDECDirectPlan.rowsZero_iff_rowsHold relation
    (PerApplicationCanonicalEncodes.piDecGeometry application) raw.assignment
    raw.base raw.groupValue raw.products
    (PerApplicationCanonicalAssignment.assignment_one raw)
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).prior.piDec).mpr
  apply R1CS.rowsHold_of_agree _ PiDECSourceSupport.Target target _ (sourceRows_support relation)
  · exact copied_source application target suffix
  · exact sourceRows_of_physical relation target physical

end NightstreamFPrime.Export.Stage1.PiDECCompletedAssignment
