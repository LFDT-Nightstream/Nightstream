import NightstreamFPrime.Export.Stage1.PiDECDirectPlan

/-! PiDEC acceptance depends on decoded source-form values. This interface
allows the compact candidate to reuse the same four ordinary row packets. -/

namespace NightstreamFPrime.Export.Stage1.PiDECFormSemantics

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiDECDirectPlan

variable {program : Lifecycle.Stage1.Application.Program} {columns : Nat}

private theorem packet (geometry : PiDECRetainedGeometry.Geometry program columns)
    (assignment : Assignment F columns) (env : Env)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (reads : ∀ column : Fin Layout.Stage1.Spartan.spartanColumnCount,
      Layout.Stage1.PiDECSourceSupport.Target column.val →
      ((sourceMap geometry).form column).eval assignment = env column.val)
    {rows : List R1CS.Row} (source : SupportedProgram rows) :
    (source.toProgram.compile (inputs source.toProgram geometry)).toPlan.RowsZero assignment ↔
      R1CS.RowsHold env rows := by
  have compiled := OrdinarySourcePlan.Program.rowsZero_iff source.toProgram
    (inputs source.toProgram geometry) assignment env one (by
      intro index
      have scope := source.supported index
      refine ⟨?_, ?_, ?_⟩
      · intro term member; exact reads ⟨term.1, (source.toProgram.bounded index).1 term member⟩ (scope.1 term member)
      · intro term member; exact reads ⟨term.1, (source.toProgram.bounded index).2.1 term member⟩ (scope.2.1 term member)
      · intro term member; exact reads ⟨term.1, (source.toProgram.bounded index).2.2 term member⟩ (scope.2.2 term member))
  refine compiled.trans ?_
  constructor
  · intro holds row member
    rw [← source.exactRows] at member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    exact holds index
  · intro holds index
    apply holds
    exact Eq.mp (congrArg (fun rows' => source.row index ∈ rows') source.exactRows)
      (List.mem_ofFn.mpr ⟨index, rfl⟩)

theorem rowsZero_iff {width : Nat}
    {fits : ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (relation : Lifecycle.ProductionKey.LogicalRelation width fits)
    (geometry : PiDECRetainedGeometry.Geometry program columns)
    (assignment : Assignment F columns) (env : Env)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (reads : ∀ column : Fin Layout.Stage1.Spartan.spartanColumnCount,
      Layout.Stage1.PiDECSourceSupport.Target column.val →
      ((sourceMap geometry).form column).eval assignment = env column.val) :
    (plan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold env (PiDECOrdinaryDirectSource.sourceRows width fits) := by
  rw [plan, Plan.append_rowsZero_iff, recompositionPlan, Plan.append_rowsZero_iff,
    evaluationPlan, Plan.append_rowsZero_iff]
  rw [publicPlan, packet geometry assignment env one reads,
    commitmentPlan, packet geometry assignment env one reads,
    evalKPlan, packet geometry assignment env one reads,
    evalAPlan, packet geometry assignment env one reads]
  simp only [PiDECOrdinaryDirectSource.sourceRows, R1CS.rowsHold_append, and_assoc]

end NightstreamFPrime.Export.Stage1.PiDECFormSemantics
