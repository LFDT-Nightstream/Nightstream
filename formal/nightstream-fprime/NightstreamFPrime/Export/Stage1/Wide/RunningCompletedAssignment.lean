import NightstreamFPrime.Export.Stage1.Wide.RunningSourceRows
import NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceInputs

/-! The compact running transition accepts the wide source assignment. The
physical PiCCS rows supply its transcript point; the wide transition supplies
the checked branch, state and child values. -/

namespace NightstreamFPrime.Export.Stage1.Wide.RunningCompletedAssignment

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint ProductionRelation

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

private theorem reduced_rows_agree (relation : ProductionKey.LogicalRelation width fits)
    (before after : Env)
    (agrees : ∀ column, column < Layout.Stage1.Spartan.SourceColumnCount → after column = before column)
    (rows : R1CS.RowsHold before (Layout.Stage1.RunningTransitionReducedRows.rows width fits)) :
    R1CS.RowsHold after (Layout.Stage1.RunningTransitionReducedRows.rows width fits) := by
  obtain ⟨flag, logical⟩ := (Layout.Stage1.RunningTransitionReducedRows.rows_iff_logical relation before).mp rows
  apply (Layout.Stage1.RunningTransitionReducedRows.rows_iff_logical relation after).mpr
  constructor
  · change after 28488881 = after 28 * after 28488880
    rw [agrees 28488881 (by decide), agrees 28 (by decide), agrees 28488880 (by decide)]
    exact flag
  · intro expression member
    rw [Expr.eval_eq_of_agree_satisfy expression Layout.Stage1.RunningTransitionSourceSupport.Logical
      after before (Layout.Stage1.RunningTransitionSourceSupport.logicalConstraints_varsSatisfy width fits expression member)]
    · exact logical expression member
    · intro source supported
      apply agrees
      exact lt_of_lt_of_le (Layout.Stage1.RunningTransitionSourceSupport.logical_lt_columnCount source supported)
        (by decide)

/-- All reduced transition rows hold on the packet with the copied base. -/
theorem reference_rowsZero (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (assumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (runningRows : R1CS.RowsHold env (Layout.Stage1.Wide.RunningTransitionLayout.physicalRows width fits)) :
    (RunningTransitionReducedPlan.plan
      (PerApplicationCanonicalEncodes.runningGeometry program)).RowsZero
      (SourceAssignment.raw program env application).assignment := by
  let raw := SourceAssignment.raw program env application
  let geometry := PerApplicationCanonicalEncodes.runningGeometry program
  have cView := PiRLCSourceInputs.piCcs_physical env relation ajtai template assumptions cRows
  apply (RunningTransitionReducedPlan.rowsZero_iff_accepts relation geometry (fun _ => none) raw.assignment).mpr
  apply (RunningTransitionReducedMatrixComplete.accepts_iff_rows relation geometry
    (RunningTransitionRetainedGeometry.oneColumn geometry) (fun _ => none) raw.assignment
    (PerApplicationCanonicalAssignment.assignment_one raw)).mpr
  apply (RunningTransitionReducedEncoding.rows_iff_source geometry raw.assignment raw.base raw.groupValue raw.products
    (PerApplicationCanonicalEncodes.runningPrefixEncodes raw).transition relation).mpr
  apply reduced_rows_agree relation (SourceAssignment.sourceEnv env) _ _
    (RunningSourceRows.reduced_rows relation env runningRows)
  intro source bounded
  exact (PiCCSCompletedReadout.transitionEnv_of_completed program relation
    (SourceAssignment.targetEnv env) application cView source bounded).trans
    (SourceAssignment.targetEnv_source env source bounded)

/-- Filling direct PiRLC values preserves all 49,359 compact transition rows. -/
theorem rowsZero (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (assumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (runningRows : R1CS.RowsHold env (Layout.Stage1.Wide.RunningTransitionLayout.physicalRows width fits)) :
    (Stage1Plan.running program).RowsZero (SourceAssignment.assignment program env application) := by
  exact (AssignmentProjection.common_rowsZero_iff program (SourceAssignment.raw program env application).assignment
    _ (ReadSupport.running program _) _).mpr
      (reference_rowsZero program env application relation ajtai template assumptions cRows runningRows)

end NightstreamFPrime.Export.Stage1.Wide.RunningCompletedAssignment
