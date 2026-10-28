import NightstreamFPrime.Layout.MatrixProgram

namespace NightstreamFPrime.Layout.MatrixProgram.SourceGrid

theorem source_of_form?_some (grid : SourceGrid) {logicalWidth source : Nat}
    {form : NightstreamFPrime.Layout.ProductionRelation.SparseForm logicalWidth}
    (found : grid.form? logicalWidth source = some form) :
    ∃ major : Fin grid.majorCount, ∃ minor : Fin grid.minorCount,
      ∃ offset : Fin grid.runCount,
        source = grid.sourceStart + major.val * grid.majorSourceStride +
          minor.val * grid.minorSourceStride + offset.val := by
  have after : grid.sourceStart ≤ source := by
    by_contra rejected
    simp only [SourceGrid.form?, ite_eq_right rejected] at found
    cases found
  have majorPositive : 0 < grid.majorSourceStride := by
    by_contra rejected
    simp only [SourceGrid.form?, ite_eq_left after, ite_eq_right rejected] at found
    cases found
  have majorBound : (source - grid.sourceStart) / grid.majorSourceStride < grid.majorCount := by
    by_contra rejected
    simp only [SourceGrid.form?, ite_eq_left after, ite_eq_left majorPositive,
      ite_eq_right rejected] at found
    cases found
  have minorPositive : 0 < grid.minorSourceStride := by
    by_contra rejected
    simp only [SourceGrid.form?, ite_eq_left after, ite_eq_left majorPositive,
      ite_eq_left majorBound, ite_eq_right rejected] at found
    cases found
  have minorBound : (source - grid.sourceStart) % grid.majorSourceStride /
      grid.minorSourceStride < grid.minorCount := by
    by_contra rejected
    simp only [SourceGrid.form?, ite_eq_left after, ite_eq_left majorPositive,
      ite_eq_left majorBound, ite_eq_left minorPositive, ite_eq_right rejected] at found
    cases found
  have offsetBound : (source - grid.sourceStart) % grid.majorSourceStride %
      grid.minorSourceStride < grid.runCount := by
    by_contra rejected
    simp only [SourceGrid.form?, ite_eq_left after, ite_eq_left majorPositive,
      ite_eq_left majorBound, ite_eq_left minorPositive, ite_eq_left minorBound, ite_eq_right rejected] at found
    cases found
  refine ⟨⟨_, majorBound⟩, ⟨_, minorBound⟩, ⟨_, offsetBound⟩, ?_⟩
  have major := Nat.mod_add_div (source - grid.sourceStart) grid.majorSourceStride
  have minor := Nat.mod_add_div
    ((source - grid.sourceStart) % grid.majorSourceStride) grid.minorSourceStride
  nlinarith [Nat.sub_add_cancel after]

end NightstreamFPrime.Layout.MatrixProgram.SourceGrid
