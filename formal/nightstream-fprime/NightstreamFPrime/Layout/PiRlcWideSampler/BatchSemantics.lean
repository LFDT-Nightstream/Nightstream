import NightstreamFPrime.Layout.PiRlcWideSampler.BatchPlan
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Batch

/-! Relate the direct CCS rows to their checked field relation. This proof
uses arbitrary accepted assignments; witness-generation hints are not
assumptions of soundness. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.BatchSemantics

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open ProductionRelation BatchPlan
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Gadgets.Sampling

private theorem range_residual {columns : Nat} (compiled : RangePlan.Compiled)
    (interface : Interface columns) (assignment : Assignment F columns) (source : Fin 17)
    (row : Fin compiled.rows.length) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      ((rangeFamily compiled interface).rowImage assignment
        ((rangeFamily compiled interface).rowLayout.toVertex (Fin.encodeProd (source, row)))) =
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      ((rangePlan compiled interface source).rowImage assignment
        ((rangePlan compiled interface source).rowLayout.toVertex row)) := by
  rw [Plan.rowImage_toVertex, Plan.rowImage_toVertex]
  apply congrArg (evaluatePolynomial baseOps Spec.ProductionRelation.polynomial)
  funext port
  unfold ProductionRelation.Plan.portForm
  split
  · exact congrArg (fun form => form.eval assignment) (Plan.indexed_forms _ _ source row _)
  · rfl

theorem rangeFamily_zero_iff {columns : Nat} (compiled : RangePlan.Compiled)
    (interface : Interface columns) (assignment : Assignment F columns) :
    (rangeFamily compiled interface).RowsZero assignment ↔
      ∀ source, (rangePlan compiled interface source).RowsZero assignment := by
  constructor
  · intro all source row
    rw [← range_residual compiled interface assignment source row]
    exact all (Fin.encodeProd (source, row))
  · intro all global
    let decoded : Fin 17 × Fin compiled.rows.length := Fin.decodeProd global
    have same : Fin.encodeProd decoded = global := Fin.encodeProd_decodeProd global
    rw [← same, range_residual]
    exact all decoded.1 decoded.2

def rangeSource {columns : Nat} (interface : Interface columns) (source : Fin 17) :
    SourceCompiler.SourceMap 2025 columns :=
  Retained.sourceMap (rangeStart interface source) (rangeFits interface source)
    (fun lane => outputState interface ⟨source.val * 2, by omega⟩ ⟨lane.val, by omega⟩)

def rangeEnv {columns : Nat} (interface : Interface columns) (assignment : Assignment F columns)
    (source : Fin 17) : Env :=
  SourceCompiler.sourceEnv (fun column => ((rangeSource interface source).form column).eval assignment)

theorem rangeSource_preserves {columns : Nat} (interface : Interface columns)
    (assignment : Assignment F columns) (source : Fin 17) :
    (rangeSource interface source).Preserves assignment (rangeEnv interface assignment source) := by
  intro column
  exact (SourceCompiler.sourceEnv_at
    (fun column => ((rangeSource interface source).form column).eval assignment) column).symm

theorem range_rows_iff {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1) (source : Fin 17) :
    (rangePlan compiled interface source).RowsZero assignment ↔
      ConstraintsHold (rangeEnv interface assignment source) RangePlan.constraints := by
  exact compiled.plan_rows_iff _ _ _ one (fun _ => rangeSource_preserves interface assignment source)

theorem range_sound {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1)
    (source : Fin 17) (rows : (rangePlan compiled interface source).RowsZero assignment) :
    WideReduction.SpecHolds RangePlan.interface 1408 (rangeEnv interface assignment source) := by
  have checked := (range_rows_iff compiled interface assignment one source).mp rows
  exact (WideReduction.Program.circuit RangePlan.interface).soundness _ 4
    (fun lane => lane.isLt) (holdsFlat_implies_holds _ _ checked)

end NightstreamFPrime.Layout.PiRlcWideSampler.BatchSemantics
