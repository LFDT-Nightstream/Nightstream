import NightstreamFPrime.Layout.MatrixProgram

/-! Move retained blocks without changing source-key selection. A decoder
proof supplies the exact partial transformation for every retained form. -/

namespace NightstreamFPrime.Layout.MatrixProgram

open ProductionRelation

def RetainedBlock.shift (distance : Nat) (block : RetainedBlock) : RetainedBlock :=
  ⟨block.kind, block.slotCount, block.start - distance⟩

theorem RetainedBlock.shift_slotCount (distance : Nat) (block : RetainedBlock) :
    (RetainedBlock.shift distance block).slotCount = block.slotCount := rfl

theorem RetainedBlock.form?_of_not_lt (block : RetainedBlock) (columns slot : Nat)
    (outside : ¬slot < block.slotCount) : block.form? columns slot = none := by
  simp only [RetainedBlock.form?, dif_neg outside]

def SourceRange.mapRetained (range : SourceRange) (move : RetainedBlock → RetainedBlock) : SourceRange :=
  {range with retained := move range.retained}

def SourceGrid.mapRetained (grid : SourceGrid) (move : RetainedBlock → RetainedBlock) : SourceGrid :=
  {grid with retained := move grid.retained}

def SourceSubstitution.mapRetained (substitution : SourceSubstitution) (move : RetainedBlock → RetainedBlock) : SourceSubstitution :=
  ⟨substitution.ranges.map (fun range => range.mapRetained move), substitution.grids.map (fun grid => grid.mapRetained move)⟩

theorem SourceRange.mapRetained_form {before after : Nat} (range : SourceRange)
    (move : RetainedBlock → RetainedBlock) (convert : SparseForm before → Option (SparseForm after))
    (loads : ∀ slot, (move range.retained).form? after slot = (range.retained.form? before slot).bind convert)
    (source : Nat) :
    (range.mapRetained move).form? after source = (range.form? before source).bind convert := by
  simp only [mapRetained, form?]
  split_ifs <;> simp_all

theorem SourceGrid.mapRetained_form {before after : Nat} (grid : SourceGrid)
    (direct : grid.mode = .direct) (move : RetainedBlock → RetainedBlock)
    (convert : SparseForm before → Option (SparseForm after))
    (loads : ∀ slot, (move grid.retained).form? after slot = (grid.retained.form? before slot).bind convert)
    (source : Nat) :
    (grid.mapRetained move).form? after source = (grid.form? before source).bind convert := by
  simp only [mapRetained, form?, direct]
  split_ifs <;> simp_all

private theorem mapped_candidates {Alpha Beta Gamma : Type}
    (items : List Alpha) (before : Alpha → Option Beta) (after : Alpha → Option Gamma)
    (convert : Beta → Option Gamma)
    (same : ∀ item ∈ items, after item = (before item).bind convert) :
    items.filterMap after = (items.filterMap before).filterMap convert := by
  induction items with
  | nil => rfl
  | cons item rest ih =>
    have head := same item (by simp)
    have tail := ih (fun value member => same value (List.mem_cons_of_mem item member))
    simp only [List.filterMap_cons, head]
    cases found : before item <;> simp [tail, List.filterMap_cons]

theorem SourceSubstitution.mapRetained_form {before after : Nat} (substitution : SourceSubstitution)
    (move : RetainedBlock → RetainedBlock) (convert : SparseForm before → Option (SparseForm after))
    (ranges : ∀ range ∈ substitution.ranges, ∀ slot,
      (move range.retained).form? after slot = (range.retained.form? before slot).bind convert)
    (grids : ∀ grid ∈ substitution.grids, grid.mode = .direct ∧ ∀ slot,
      (move grid.retained).form? after slot = (grid.retained.form? before slot).bind convert)
    (source : Nat) (old : SparseForm before) (new : SparseForm after)
    (loaded : substitution.form? before source = some old) (converted : convert old = some new) :
    (substitution.mapRetained move).form? after source = some new := by
  have rangeMap := mapped_candidates substitution.ranges
    (fun range => range.form? before source) (fun range => (range.mapRetained move).form? after source)
    convert (fun range member => SourceRange.mapRetained_form range move convert (ranges range member) source)
  have gridMap := mapped_candidates substitution.grids
    (fun grid => grid.form? before source) (fun grid => (grid.mapRetained move).form? after source)
    convert (fun grid member => SourceGrid.mapRetained_form grid (grids grid member).1 move convert (grids grid member).2 source)
  have selected : substitution.ranges.filterMap (fun range => range.form? before source) ++
      substitution.grids.filterMap (fun grid => grid.form? before source) = [old] := by
    unfold SourceSubstitution.form? at loaded
    split at loaded <;> simp_all
  unfold SourceSubstitution.form? mapRetained
  simp only [List.filterMap_map, Function.comp_def, rangeMap, gridMap, ← List.filterMap_append, selected]
  simp [converted]

end NightstreamFPrime.Layout.MatrixProgram
