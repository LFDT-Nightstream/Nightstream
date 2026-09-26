import NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceOutput
import NightstreamFPrime.Export.Stage1.Wide.PiDECSourceWitness
import NightstreamFPrime.Export.Stage1.Wide.AssignmentPullback
import NightstreamFPrime.Export.Stage1.PiDECFormSemantics

/-! Compact PiDEC acceptance from the completed wide physical prefix.
The constructor's local fields and direct parent outputs all read that prefix.
The old sampler and its success conditions are absent from this proof. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiDECCompletedAssignment

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable (program : RetainedLayout.Program) (env : Env)
  (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
  {width : Nat} {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
  (relation : ProductionKey.LogicalRelation width fits)
  (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
  (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
    (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
  (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
    (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
  (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
    (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits))
    Layout.Stage1.Wide.PiRLCInputs.phaseOffset env)
  (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
    (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits))
    Layout.Stage1.Wide.PiRLCInputs.phaseOffset env)

include relation ajtai template cAssumptions cRows rAssumptions rRows

/-- Every decoded PiDEC source form reads the same physical field value. -/
theorem location_value (location : PiDECDirectPlan.Location) :
    (location.form (Stage1Plan.piDecGeometry program)).eval
      (AssignmentPullback.assignment program (SourceAssignment.assignment program env application)) =
      PiDECSource.value env location := by
  rw [← AssignmentPullback.form_eval program _ _ (ReadSupport.piDec_location program _ location)]
  have parent := PiRLCSourceOutput.retained_parent program env application relation ajtai template
    cAssumptions cRows rAssumptions rRows
  cases location with
  | parentCommitment index =>
    let block : Fin 22 := ⟨index.val / 54, by have h : index.val < 1188 := index.isLt; omega⟩
    let lane : Fin ringDegree := ⟨index.val % 54, Nat.mod_lt _ (by decide)⟩
    have same : PiDECOutput.parentView .commitment block ⟨0, by decide⟩ lane = .parentCommitment index := by
      apply congrArg PiDECDirectPlan.Location.parentCommitment
      apply Fin.ext
      change index.val / 54 * 54 + index.val % 54 = index.val
      omega
    simpa only [same] using parent .commitment block ⟨0, by decide⟩ lane
  | parentPublicInput index =>
    let block : Fin 5 := ⟨index.val / 54, by have h : index.val < 270 := index.isLt; omega⟩
    let lane : Fin ringDegree := ⟨index.val % 54, Nat.mod_lt _ (by decide)⟩
    have same : PiDECOutput.parentView .publicInput block ⟨0, by decide⟩ lane = .parentPublicInput index := by
      apply congrArg PiDECDirectPlan.Location.parentPublicInput
      apply Fin.ext
      change index.val / 54 * 54 + index.val % 54 = index.val
      omega
    simpa only [same] using parent .publicInput block ⟨0, by decide⟩ lane
  | parentEvalK index =>
    let lane : Fin ringDegree := ⟨index.val / 2, by have h : index.val < 108 := index.isLt; change index.val / 2 < 54; omega⟩
    let cell : Fin 2 := ⟨index.val % 2, Nat.mod_lt _ (by decide)⟩
    have same : PiDECOutput.parentView .evalK ⟨0, by decide⟩ cell lane = .parentEvalK index := by
      apply congrArg PiDECDirectPlan.Location.parentEvalK
      apply Fin.ext
      change index.val / 2 * 2 + index.val % 2 = index.val
      omega
    simpa only [same] using parent .evalK ⟨0, by decide⟩ cell lane
  | parentEvalA index =>
    let block : Fin 14 := ⟨index.val / 108, by have h : index.val < 1512 := index.isLt; omega⟩
    let lane : Fin ringDegree := ⟨index.val % 108 / 2, by change _ < 54; omega⟩
    let cell : Fin 2 := ⟨index.val % 108 % 2, Nat.mod_lt _ (by decide)⟩
    have same : PiDECOutput.parentView .evalA block cell lane = .parentEvalA index := by
      apply congrArg PiDECDirectPlan.Location.parentEvalA
      apply Fin.ext
      change index.val / 108 * 108 + index.val % 108 / 2 * 2 + index.val % 108 % 2 = index.val
      omega
    simpa only [same] using parent .evalA block cell lane
  | proof index => exact PiDECWitnessInputs.local_value program env application (.proof index) trivial
  | logical index => exact PiDECWitnessInputs.local_value program env application (.logical index) trivial
  | fresh index => exact PiDECWitnessInputs.local_value program env application (.fresh index) trivial

private theorem source_reads (column : Fin Layout.Stage1.Spartan.spartanColumnCount)
    (supported : Layout.Stage1.PiDECSourceSupport.Target column.val) :
    ((PiDECDirectPlan.sourceMap (Stage1Plan.piDecGeometry program)).form column).eval
      (AssignmentPullback.assignment program (SourceAssignment.assignment program env application)) =
      SourceAssignment.targetEnv env column.val := by
  rcases PiDECDirectPlan.classifyTarget_complete supported with ⟨decoded, found, mapped⟩
  change (match PiDECDirectPlan.classifyTarget column.val with
    | none => SparseForm.empty
    | some value => value.location.form (Stage1Plan.piDecGeometry program)).eval _ = _
  rw [found, location_value program env application relation ajtai template cAssumptions cRows rAssumptions rRows]
  have located : Layout.Stage1.Spartan.sourceToSpartan decoded.location.sourceColumn = column.val := by
    rw [decoded.owns, mapped]
  rw [← located, SourceAssignment.targetEnv_source env _ decoded.location.sourceColumn_lt,
    SourceAssignment.sourceEnv_piDec]

/-- Physical C, wide R and D acceptance construct all 25,488 compact PiDEC rows. -/
theorem rowsZero
    (dAssumptions : PiDEC.v1_1.Formal.Assumptions relation
      (Layout.Stage1.Wide.PiDECInputs.interface width fits) Layout.Stage1.Wide.PiDECInputs.phaseOffset env)
    (dRows : Layout.PiDEC.v1_1.PhysicalHolds relation
      (Layout.Stage1.Wide.PiDECInputs.interface width fits) Layout.Stage1.Wide.PiDECInputs.phaseOffset env) :
    (Stage1Plan.piDec program relation).RowsZero (SourceAssignment.assignment program env application) := by
  let after := SourceAssignment.assignment program env application
  let before := AssignmentPullback.assignment program after
  let geometry := Stage1Plan.piDecGeometry program
  have one : before (PiDECRetainedGeometry.oneColumn geometry) = 1 := by
    have live : RetainedLayout.Live program (PiDECRetainedGeometry.oneColumn geometry).val := by
      left
      rw [(RetainedLayout.boundaries program).1]
      change 0 < 113904174
      decide
    exact (AssignmentPullback.at_live program after _ live).trans
      (SourceAssignment.assignment_one program env application)
  have oldRows := PiDECSourceWitness.rowsHold relation ajtai env dAssumptions dRows
  have targetRows : Layout.PiDEC.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiDECInputs.interface width fits) Layout.Stage1.PiDECInputs.phaseOffset
      (Layout.Stage1.Spartan.pullback (SourceAssignment.targetEnv env)) := by
    intro row member
    apply row.holds_of_agree Layout.Stage1.PiDECSourceSupport.Source
      (SourceAssignment.sourceEnv env) (Layout.Stage1.Spartan.pullback (SourceAssignment.targetEnv env))
      (PiDECSourceWitness.old_rows_supported relation row member)
    · intro source supported
      exact SourceAssignment.targetEnv_source env source
        (Layout.Stage1.PiDECSourceSupport.source_lt_sourceColumnCount supported)
    · exact oldRows row member
  have sourceRows : R1CS.RowsHold (SourceAssignment.targetEnv env)
      (PiDECOrdinaryDirectSource.sourceRows width fits) := by
    rw [PiDECOrdinaryDirectSource.sourceRows_eq_canonical,
      PiDECArithmetic.Plan.rows_to_layout _ _ (PiDECArithmetic.canonicalPlan_matches relation)]
    exact (Layout.Stage1.Spartan.remapRows_hold _ _).mpr targetRows
  change (Stage1Plan.rename program (PiDECDirectPlan.plan relation geometry) _).RowsZero after
  rw [AssignmentPullback.rowsZero_iff]
  exact (PiDECFormSemantics.rowsZero_iff relation geometry before (SourceAssignment.targetEnv env) one
    (source_reads program env application relation ajtai template cAssumptions cRows rAssumptions rRows)).mpr sourceRows

end NightstreamFPrime.Export.Stage1.Wide.PiDECCompletedAssignment
