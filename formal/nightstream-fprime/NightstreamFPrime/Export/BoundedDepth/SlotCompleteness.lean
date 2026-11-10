import NightstreamFPrime.Export.BoundedDepth.SlotRuns
import NightstreamFPrime.Export.Generic.RelationCompleteness
import NightstreamFPrime.Export.Generic.WideLayout
import NightstreamFPrime.Export.Generic.NativeBlock

/-! Completeness of a step package's relation at its slot layout (CW-10.2, SPEC.md section 4.4).
The honest assignment reads the public input and every slot from the completed environment
(`RelationCompleteness.completeEnv`, `honestAssignment`). The package gives a slot to each column
that a row of its relation reads (`StepPackage.runsOf`), so the slot layout reads that environment
on each column that the relation reads (`represents_of_runs`), and the relation's plan holds
(`rowsZero_of_holds`).

Each coordinate is small when each slot's value is valid for its kind (`norm_of_runs`). A field
slot is always valid. The package gives a slot the kind bit only on a column that a row of a
piece forces to 0 or 1 (`SlotLayout.forcedColumn?`), so a bit slot holds 0 or 1 wherever its row
holds (`NativeBlock.forced_valid`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.SlotCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Export.BoundedDepth.StepPackage
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ### The honest assignment -/

/-- The honest assignment of a slot layout: the public input and every slot read from `env`. -/
def honestAssignment {width : Nat} (slots : List (LowNormAssignment.Slot width)) (env : Env) :
    Assignment F (ProductionAssignment.logicalWidth slots) :=
  ProductionAssignment.logicalAssignment (fun column => env column.val) slots
    (fun column => env column.val)

/-- The honest assignment reads `env` on each public column and on each column with a slot. -/
theorem sourceEnv_honest {width : Nat} (slots : List (LowNormAssignment.Slot width)) (env : Env)
    {column : Nat} (inside : column < width)
    (covered : column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column) :
    LogicalPlan.sourceEnv (SlotLayout.layout slots) (honestAssignment slots env) column =
      env column := by
  unfold LogicalPlan.sourceEnv
  rw [dite_eq_left inside]
  show (if bound : column < LogicalPlan.publicCount then
      SparseForm.singleton ⟨column, Nat.lt_of_lt_of_le bound (SlotLayout.publicFits slots)⟩ 1
    else match SlotLayout.slotIndex slots ⟨column, inside⟩ with
      | some index => RetainedSlot.form slots index
      | none => .empty).eval (honestAssignment slots env) = _
  by_cases publicRegion : column < LogicalPlan.publicCount
  · rw [dite_eq_left publicRegion, SparseForm.singleton_eval, one_mul]
    exact ProductionAssignment.logicalAssignment_publicColumn _ slots _ ⟨column, publicRegion⟩
  · rw [dite_eq_right publicRegion]
    have slotted : ∃ slot ∈ slots, slot.source = ⟨column, inside⟩ := by
      rcases covered with isPublic | ⟨slot, member, source⟩
      · exact absurd isPublic publicRegion
      · exact ⟨slot, member, Fin.ext source⟩
    obtain ⟨index, found, source⟩ := SlotRuns.slotIndex_of_mem slotted
    rw [found]
    change (RetainedSlot.form slots index).eval (ProductionAssignment.logicalAssignment
      (fun column => env column.val) slots (fun column => env column.val)) = _
    rw [RetainedSlot.form_eval, source]

variable {pieces : List Piece} {next width : Nat}

theorem slotEnd_le (fits : Fits publicColumns pieces next width)
    (packed : packable publicColumns pieces = true) : slotEnd pieces next ≤ width := by
  have := (fits.blocks packed).sboxColumns
  unfold slotEnd
  omega

/-- At the slots of `runsOf` over package pieces that include the package pieces of `pieces`
and have their blocks and no other output local, the honest assignment reads `env` on each column
that the relation of `pieces` reads. -/
theorem represents_of_runs (room : slotEnd pieces next ≤ width)
    (below : ∀ piece ∈ pieces, pieceBelow next piece = true) (packageList : List PackagePiece)
    (bits : List Nat) (includes : ∀ piece ∈ packagePieces pieces next, piece ∈ packageList)
    (blocks : (packageList.filterMap PackagePiece.block?).length = (blocksOf pieces).length)
    (outputs : ∀ piece ∈ packageList, ∀ column ∈ piece.outputs,
      outputIn (blocksOf pieces) column = true)
    (env : Env) {column : Nat} (read : Reads pieces next column) :
    LogicalPlan.sourceEnv (SlotLayout.layout (SlotLayout.runSlots width
        (runsOf publicColumns (sboxStart pieces next) packageList bits)))
      (honestAssignment (SlotLayout.runSlots width
        (runsOf publicColumns (sboxStart pieces next) packageList bits)) env) column =
      env column := by
  have publicEq : publicColumns = 270 := rfl
  have countEq : LogicalPlan.publicCount = 270 := rfl
  have sizeEq : SlotRuns.runsSize (sboxStart pieces next) packageList = slotEnd pieces next := by
    unfold SlotRuns.runsSize slotEnd
    rw [blocks, Nat.mul_comm]
  have startLe : sboxStart pieces next ≤ slotEnd pieces next := Nat.le_add_right _ _
  have scope : column < slotEnd pieces next ∧
      (sboxStart pieces next ≤ column ∨ ((∃ piece ∈ packageList, column ∈ piece.reads) ∧
        ∀ piece ∈ packageList, column ∉ piece.outputs)) := by
    rcases read with ⟨outside, choice⟩ | sbox
    · have noOutput : ∀ piece ∈ packageList, column ∉ piece.outputs := by
        intro piece member inPiece
        rw [outputs piece member column inPiece] at outside
        exact Bool.noConfusion outside
      rcases choice with ⟨row, rowMember, columnMember⟩ | ⟨block, blockMember, lane, laneMember,
        columnMember⟩
      · rw [← SlotRuns.packagePieces_rows] at rowMember
        obtain ⟨piece, pieceMember, inPiece⟩ := List.mem_flatMap.mp rowMember
        have reads := SlotRuns.relationRows_reads piece row inPiece column columnMember
        have low := SlotRuns.reads_below below piece pieceMember column reads
        exact ⟨by omega, Or.inr ⟨⟨piece, includes piece pieceMember, reads⟩, noOutput⟩⟩
      · have member := SlotRuns.block_mem (next := next) blockMember
        have reads : column ∈ (PackagePiece.block block.start block.lanes).reads :=
          List.mem_flatMap.mpr ⟨lane, laneMember, columnMember⟩
        have low := SlotRuns.reads_below below _ member column reads
        exact ⟨by omega, Or.inr ⟨⟨_, includes _ member, reads⟩, noOutput⟩⟩
    · exact ⟨sbox.2, Or.inl sbox.1⟩
  have inside : column < width := by omega
  apply sourceEnv_honest _ env inside
  by_cases isPublic : column < LogicalPlan.publicCount
  · exact Or.inl isPublic
  · exact Or.inr (SlotRuns.runsOf_covers (by omega) (by omega) inside scope.2)

/-- At the slots of `runsOf`, the honest assignment lies in the norm bound when the public columns
do and each column that a piece forces to 0 or 1 holds 0 or 1. -/
theorem norm_of_runs (publicStart sboxStart : Nat) (packageList : List PackagePiece)
    (bits : List Nat) (env : Env)
    (publicNorm : ∀ column, column < publicColumns → centeredMagnitude (env column) < 2)
    (forced : ∀ piece ∈ packageList, ∀ column ∈ piece.forced,
      LowNormSlot.Valid .bit (env column)) :
    ∀ column, centeredMagnitude (honestAssignment (SlotLayout.runSlots width
      (runsOf publicStart sboxStart packageList bits)) env column) < 2 := by
  apply ProductionAssignment.logicalAssignment_norm
  · intro column
    exact publicNorm column.val column.isLt
  · intro slot member
    rcases SlotRuns.runsOf_kind member with field | ⟨bit, piece, pieceMember, inPiece⟩
    · show LowNormSlot.Valid slot.kind (env slot.source.val)
      rw [field]
      trivial
    · show LowNormSlot.Valid slot.kind (env slot.source.val)
      rw [bit]
      exact forced piece pieceMember _ inPiece

/-- An honest step satisfies the package's relation at its slot layout widened to `wide` logical
columns, and every coordinate of the honest assignment is small. -/
theorem rowsZero_of_holds (fits : Fits publicColumns pieces next width)
    (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid) (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (publicBelow : publicColumns ≤ next) (batches : List WitnessBatch) {wide : Nat}
    (wideFits : ProductionAssignment.logicalWidth
      (SlotLayout.runSlots width (slotRuns publicColumns pieces next batches)) ≤ wide)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    (oneValue : env 0 = 1)
    (publicNorm : ∀ column, column < publicColumns → centeredMagnitude (env column) < 2) :
    let slots := SlotLayout.runSlots width (slotRuns publicColumns pieces next batches)
    let assignment := WideLayout.extend (wide := wide)
      (honestAssignment slots (completeEnv pieces next env))
    (plan publicColumns pieces next width fits
        (WideLayout.widen (SlotLayout.layout slots) wideFits)).RowsZero assignment ∧
      ∀ column, centeredMagnitude (assignment column) < 2 := by
  intro slots
  have publicValue : ∀ column, column < publicColumns →
      completeEnv pieces next env column = env column :=
    fun column inside => completeEnv_public packed publicBelow env inside
  have completeOne : completeEnv pieces next env 0 = 1 :=
    (publicValue 0 (by decide)).trans oneValue
  have one : honestAssignment slots (completeEnv pieces next env) (SlotLayout.layout slots).one =
      1 :=
    (ProductionAssignment.logicalAssignment_publicColumn _ slots _ ⟨0, by decide⟩).trans
      completeOne
  have rowsHold := completeEnv_rows packed valid below env holds
  rw [← SlotRuns.packagePieces_rows] at rowsHold
  refine ⟨RelationCompleteness.rowsZero_of_holds fits packed valid below env holds
    (WideLayout.widen (SlotLayout.layout slots) wideFits) (WideLayout.extend _)
    ((WideLayout.one_widen _ wideFits _).trans one) (fun column read => ?_),
    WideLayout.extend_norm _ (norm_of_runs _ _ _ _ _ (fun column inside => ?_) ?_)⟩
  · rw [WideLayout.sourceEnv_widen]
    exact represents_of_runs (slotEnd_le fits packed) below _ _ (fun _ member => member)
      (by rw [SlotRuns.packagePieces_blocks])
      (by
        have outputs := SlotRuns.outputs_outputIn (packagePieces pieces next)
        rwa [SlotRuns.packagePieces_blocks] at outputs) _ read
  · rw [publicValue column inside]
    exact publicNorm column inside
  · intro piece member column inPiece
    obtain ⟨row, rowMember, found⟩ := SlotRuns.forced_row piece inPiece
    exact NativeBlock.forced_valid found completeOne
      (rowsHold row (List.mem_flatMap.mpr ⟨piece, member, rowMember⟩))

end NightstreamFPrime.Export.BoundedDepth.SlotCompleteness
