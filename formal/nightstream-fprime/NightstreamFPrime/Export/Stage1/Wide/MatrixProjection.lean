import NightstreamFPrime.Layout.MatrixProgram.Exact
import NightstreamFPrime.Export.Stage1.Wide.Stage1Plan

/-! Wire data for the candidate's checked retained-column map. Reused
matrix programs use this exact map, including its rejection of removed columns. -/

namespace NightstreamFPrime.Export.Stage1.Wide.MatrixProjection

open NightstreamFPrime.Layout NightstreamFPrime.Spec
open MatrixProgram ProductionRelation

abbrev Program := RetainedLayout.Program

def projection (program : Program) : SourceProjection := .mapped [
  ⟨0, 0, RetainedLayout.hashEnd program⟩,
  ⟨RetainedLayout.sharedStart program, RetainedLayout.hashEnd program,
    RetainedLayout.sharedEnd program - RetainedLayout.sharedStart program⟩,
  ⟨RetainedLayout.applicationStart program,
    RetainedLayout.hashEnd program + (RetainedLayout.sharedEnd program - RetainedLayout.sharedStart program),
    RetainedLayout.applicationCount program⟩,
  ⟨119147994, RetainedLayout.outputStart program, 2145366⟩,
  ⟨114443652, RetainedLayout.quotientStart program, 2145366⟩]

private theorem range_column (range : SourceProjectionRange) (source : Nat) :
    range.column? source =
      if range.packageStart ≤ source ∧ source < range.packageStart + range.count then
        some (range.sourceStart + (source - range.packageStart)) else none := by
  simp only [SourceProjectionRange.column?]
  split_ifs <;> first | rfl | omega

theorem column_eq (program : Program) (source : Nat) :
    (projection program).column? source = RetainedLayout.column? program source := by
  obtain ⟨hash, first, last, app⟩ := RetainedLayout.boundaries program
  simp only [projection, SourceProjection.column?, List.filterMap_cons, List.filterMap_nil,
    range_column, RetainedLayout.column?, hash, first, last, app]
  norm_num only [Nat.zero_le, Nat.zero_add, Nat.sub_zero, Nat.reduceAdd, Nat.reduceSub]
  by_cases h : source < 113904174
  · simp [h, show ¬(121293360 ≤ source ∧ source < 140293745) by omega,
      show ¬(149282257 ≤ source ∧ source < 149282257 + RetainedLayout.applicationCount program) by omega,
      show ¬(119147994 ≤ source ∧ source < 121293360) by omega,
      show ¬(114443652 ≤ source ∧ source < 116589018) by omega]
  · by_cases shared : 121293360 ≤ source ∧ source < 140293745
    · simp [h, shared,
        show ¬(149282257 ≤ source ∧ source < 149282257 + RetainedLayout.applicationCount program) by omega,
        show ¬(119147994 ≤ source ∧ source < 121293360) by omega,
        show ¬(114443652 ≤ source ∧ source < 116589018) by omega]
    · by_cases application : 149282257 ≤ source ∧ source < 149282257 + RetainedLayout.applicationCount program
      · simp [h, shared, application,
          show ¬(119147994 ≤ source ∧ source < 121293360) by omega,
          show ¬(114443652 ≤ source ∧ source < 116589018) by omega]
      · by_cases output : 119147994 ≤ source ∧ source < 121293360
        · simp [h, shared, application, output,
            show ¬(114443652 ≤ source ∧ source < 116589018) by omega]
        · by_cases quotient : 114443652 ≤ source ∧ source < 116589018
          · simp [h, shared, application, output, quotient]
          · simp [h, shared, application, output, quotient]

def program (application : Program) (matrix : MatrixProgram.Program) : MatrixProgram.Program :=
  matrix.mapColumns (PerApplicationFixedPoint.logicalWidth application) (projection application)

/-- Every reused phase keeps its exact rows after the proved coordinate map.
The encoded interpreter rejects unmapped entries rather than substituting zero. -/
theorem exact (application : Program)
    {matrix : MatrixProgram.Program}
    {plan : ProductionRelation.Plan (PerApplicationFixedPoint.logicalWidth application)}
    {sourceRow : Nat → Option R1CS.Row}
    (before : MatrixProgram.Exact matrix plan sourceRow)
    (support : ReadSupport.Plans application plan) :
    MatrixProgram.Exact (program application matrix) (Stage1Plan.rename application plan support) sourceRow := by
  apply MatrixProgram.Exact.mapColumns before (projection application)
    (RetainedLayout.column application) support
  intro index live
  rw [column_eq]
  exact RetainedLayout.column_mapped application index live

end NightstreamFPrime.Export.Stage1.Wide.MatrixProjection
