import NightstreamFPrime.Export.BoundedDepth.NativeCompleteness
import NightstreamFPrime.Export.Generic.BooleanInference

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.NativeBooleanCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Export.BoundedDepth.StepPackage
open NightstreamFPrime.Export.BoundedDepth.SlotCompleteness
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

open NightstreamFPrime.Export.BoundedDepth.NativeCompleteness

theorem sourceEnv_honest_absent {width : Nat}
    (slots : List (LowNormAssignment.Slot width)) (env : Env) (column : Nat)
    (inside : column < width) (notPublic : ¬ column < LogicalPlan.publicCount)
    (absent : ¬ ∃ slot ∈ slots, slot.source.val = column) :
    LogicalPlan.sourceEnv (SlotLayout.layout slots) (honestAssignment slots env) column = 0 := by
  have noIndex : SlotLayout.slotIndex slots ⟨column, inside⟩ = none := by
    apply List.find?_eq_none.mpr
    intro index _
    have different : (slots.get index).source ≠ ⟨column, inside⟩ := by
      intro same
      exact absent ⟨slots.get index, List.get_mem slots index, congrArg Fin.val same⟩
    simpa using different
  simp only [LogicalPlan.sourceEnv, inside, dite_true, SlotLayout.layout,
    notPublic, dite_false, noIndex, SparseForm.empty_eval]

theorem sourceEnv_honest_eq_of_sources {width : Nat}
    (slots before : List (LowNormAssignment.Slot width))
    (sameSources : ∀ column, (∃ slot ∈ slots, slot.source.val = column) ↔
      ∃ slot ∈ before, slot.source.val = column) (env : Env) (column : Nat) :
    LogicalPlan.sourceEnv (SlotLayout.layout slots) (honestAssignment slots env) column =
      LogicalPlan.sourceEnv (SlotLayout.layout before) (honestAssignment before env) column := by
  by_cases inside : column < width
  · by_cases covered : column < LogicalPlan.publicCount ∨
        ∃ slot ∈ before, slot.source.val = column
    · have newCovered : column < LogicalPlan.publicCount ∨
          ∃ slot ∈ slots, slot.source.val = column :=
        covered.elim Or.inl (fun found => Or.inr ((sameSources column).mpr found))
      rw [sourceEnv_honest slots env inside newCovered,
        sourceEnv_honest before env inside covered]
    · have notPublic := (not_or.mp covered).1
      have oldAbsent := (not_or.mp covered).2
      have newAbsent : ¬ ∃ slot ∈ slots, slot.source.val = column :=
        fun found => oldAbsent ((sameSources column).mp found)
      rw [sourceEnv_honest_absent slots env column inside notPublic newAbsent,
        sourceEnv_honest_absent before env column inside notPublic oldAbsent]
  · simp only [LogicalPlan.sourceEnv, inside, dite_false]

variable {package : Package} {nativeRows : List R1CS.Row} {pieces : List Piece}

theorem inferred_native_complete (bits : List Nat)
    (piecesEq : package.pieces = .rows nativeRows :: packagePieces pieces package.firstFresh)
    (sboxEq : package.sboxStart = sboxStart pieces package.firstFresh)
    (sameSources : ∀ column,
      (∃ slot ∈ SlotLayout.runSlots package.width package.slots, slot.source.val = column) ↔
        ∃ slot ∈ SlotLayout.runSlots package.width
          (runsOf publicColumns package.sboxStart package.pieces bits), slot.source.val = column)
    (kinds : ∀ slot ∈ SlotLayout.runSlots package.width package.slots,
      slot.kind = .field ∨ slot.kind = .bit ∧
        (BooleanInference.Derived nativeRows slot.source.val ∨
          ∃ piece ∈ package.pieces, slot.source.val ∈ piece.forced))
    (fits : Fits publicColumns pieces package.firstFresh package.width)
    (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid)
    (below : ∀ piece ∈ pieces, pieceBelow package.firstFresh piece = true)
    (publicBelow : publicColumns ≤ package.firstFresh)
    (nativeBelow : ∀ row ∈ nativeRows, row.VarsBelow package.firstFresh)
    (nativeLive : ∀ row ∈ nativeRows, NativeBlock.rowAvoids (deadIn (blocksOf pieces)) row = true)
    (rowsFit : LogicalPlan.Fits package.relationRows package.width)
    (rowCount : package.blocks.length * 150 ≤ 2 ^ cubeVariables) {wide : Nat}
    (wideFits : ProductionAssignment.logicalWidth
      (SlotLayout.runSlots package.width package.slots) ≤ wide)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    (nativeHold : R1CS.RowsHold env nativeRows) (oneValue : env 0 = 1)
    (publicNorm : ∀ column, column < publicColumns → centeredMagnitude (env column) < 2) :
    let layout := WideLayout.widen (SlotLayout.layout (SlotLayout.runSlots package.width
      package.slots)) wideFits
    let assignment := WideLayout.extend (wide := wide) (honestAssignment
      (SlotLayout.runSlots package.width package.slots) (completeEnv pieces package.firstFresh env))
    (LogicalPlan.plan package.relationRows package.width rowsFit layout).RowsZero assignment ∧
      (PoseidonSboxFamilyPlan.plan (sboxInterface layout package.blocks package.sboxStart)
        rowCount).RowsZero assignment ∧
      (∀ column, column < publicColumns →
        LogicalPlan.sourceEnv layout assignment column = env column) ∧
      ∀ column, centeredMagnitude (assignment column) < 2 := by
  intro layout assignment
  let slots := SlotLayout.runSlots package.width package.slots
  let completed := completeEnv pieces package.firstFresh env
  have publicValue : ∀ column, column < publicColumns → completed column = env column :=
    fun column inside => completeEnv_public packed publicBelow env inside
  have completeOne : completed 0 = 1 := (publicValue 0 (by decide)).trans oneValue
  have room := slotEnd_le fits packed
  have nextLe := next_le_sboxStart pieces package.firstFresh
  have startLe : sboxStart pieces package.firstFresh ≤ slotEnd pieces package.firstFresh :=
    Nat.le_add_right _ _
  have represents : ∀ column, Reads pieces package.firstFresh column →
      LogicalPlan.sourceEnv (SlotLayout.layout slots) (honestAssignment slots completed) column =
        completed column := by
    intro column read
    show LogicalPlan.sourceEnv (SlotLayout.layout (SlotLayout.runSlots package.width
      package.slots)) (honestAssignment (SlotLayout.runSlots package.width package.slots)
        completed) column = completed column
    rw [sourceEnv_honest_eq_of_sources _ _ sameSources completed column, sboxEq]
    refine represents_of_runs room below _ bits (fun piece member => ?_) ?_ ?_ completed read
    · rw [piecesEq]
      exact List.mem_cons_of_mem _ member
    · rw [← Package.blocks, blocks_eq piecesEq]
    · have outputs := SlotRuns.outputs_outputIn package.pieces
      rwa [← Package.blocks, blocks_eq piecesEq] at outputs
  have one : honestAssignment slots completed (SlotLayout.layout slots).one = 1 :=
    (ProductionAssignment.logicalAssignment_publicColumn _ slots _ ⟨0, by decide⟩).trans
      completeOne
  have oneWide : assignment layout.one = 1 := (WideLayout.one_widen _ wideFits _).trans one
  have lean := RelationCompleteness.rowsZero_of_holds fits packed valid below env holds layout
    assignment oneWide (fun column read => by
      rw [WideLayout.sourceEnv_widen]
      exact represents column read)
  unfold plan at lean
  rw [dite_eq_left packed, Plan.append_rowsZero_iff] at lean
  obtain ⟨leanRows, sboxRows⟩ := lean
  rw [LogicalPlan.rowsZero_iff_rowsHold _ _ _ layout assignment oneWide] at leanRows
  have liveHold : R1CS.RowsHold (liveEnv pieces env) nativeRows :=
    NativeBlock.rowsHold_congr (dead := deadIn (blocksOf pieces))
      (fun column live => by simp [liveEnv, live]) (List.all_eq_true.mpr nativeLive) nativeHold
  have completeHold : R1CS.RowsHold completed nativeRows :=
    R1CS.rowsHold_of_agree_below nativeRows package.firstFresh (liveEnv pieces env) completed
      nativeBelow (fun column columnBelow => (completeEnv_low (by omega)).trans
        (executePieces_below _ _ _ _ _ columnBelow)) liveHold
  have nativeAt : R1CS.RowsHold (LogicalPlan.sourceEnv layout assignment)
      (nativeRows.map (Substitute.row (outputForms package.width (blocksOf pieces)
        (sboxStart pieces package.firstFresh)))) := by
    rw [Substitute.rowsHold_substitute]
    refine R1CS.rowsHold_of_agree nativeRows
      (fun column => column ∈ (PackagePiece.rows nativeRows).reads ∧
        column < package.firstFresh) completed _ (fun row member => ?_) (fun column read => ?_)
      completeHold
    · exact (RowReads.rowColumns_varsSatisfy row).mono row fun column columnMember =>
        ⟨List.mem_flatMap.mpr ⟨row, member, columnMember⟩,
          RowReads.rowColumns_of_varsSatisfy (allowed := (· < package.firstFresh))
            (nativeBelow row member) column columnMember⟩
    · by_cases output : outputIn (blocksOf pieces) column = true
      · obtain ⟨index, lane, rfl⟩ := outputIn_get output
        obtain ⟨atForms, atCompleted⟩ := outputEnv_complete packed below env holds layout
          assignment (fun column read => by
            rw [WideLayout.sourceEnv_widen]
            exact represents column read) index lane
        exact atForms.trans atCompleted.symm
      · have outside : outputIn (blocksOf pieces) column = false := by simpa using output
        have noOutput : ∀ piece ∈ package.pieces, column ∉ piece.outputs := by
          intro piece member inPiece
          have marked := SlotRuns.outputs_outputIn package.pieces piece member column inPiece
          rw [← Package.blocks, blocks_eq piecesEq, outside] at marked
          exact Bool.noConfusion marked
        show Substitute.refill _ _ column = _
        unfold Substitute.refill
        rw [outputForms_of_not_outputIn _ _ outside, WideLayout.sourceEnv_widen]
        show LogicalPlan.sourceEnv (SlotLayout.layout (SlotLayout.runSlots package.width
          package.slots)) (honestAssignment (SlotLayout.runSlots package.width package.slots)
            completed) column = completed column
        have inside : column < package.width := by omega
        apply sourceEnv_honest _ completed inside
        by_cases isPublic : column < LogicalPlan.publicCount
        · exact Or.inl isPublic
        · have publicEq : publicColumns = 320 := rfl
          have countEq : LogicalPlan.publicCount = 320 := rfl
          have sizeEq : SlotRuns.runsSize package.sboxStart package.pieces =
              slotEnd pieces package.firstFresh := by
            unfold SlotRuns.runsSize slotEnd
            rw [← Package.blocks, blocks_eq piecesEq, sboxEq, Nat.mul_comm]
          apply Or.inr
          apply (sameSources column).mpr
          refine SlotRuns.runsOf_covers (by omega) (by omega) inside
            (Or.inr ⟨⟨.rows nativeRows, ?_, read.1⟩, noOutput⟩)
          rw [piecesEq]
          exact List.mem_cons_self
  have sboxAt : ∀ (blocks : List Block) (start : Nat), blocks = blocksOf pieces →
      start = sboxStart pieces package.firstFresh →
      ∀ count : blocks.length * 150 ≤ 2 ^ cubeVariables,
        (PoseidonSboxFamilyPlan.plan (sboxInterface layout blocks start) count).RowsZero
          assignment := by
    intro blocks start blocksSame startSame count
    subst blocksSame
    subst startSame
    exact sboxRows
  have bounded : ∀ column, centeredMagnitude (honestAssignment slots completed column) < 2 := by
    apply ProductionAssignment.logicalAssignment_norm
    · intro column
      rw [publicValue column.val column.isLt]
      exact publicNorm column.val column.isLt
    · intro slot member
      show LowNormSlot.Valid slot.kind (completed slot.source.val)
      rcases kinds slot member with field | ⟨bit, derived | ⟨piece, pieceMember, forced⟩⟩
      · rw [field]
        trivial
      · rw [bit]
        exact BooleanInference.derived_valid derived completeOne completeHold
      · rw [bit]
        obtain ⟨row, rowMember, found⟩ := SlotRuns.forced_row piece forced
        rw [piecesEq] at pieceMember
        rcases List.mem_cons.mp pieceMember with rfl | pieceMember
        · exact NativeBlock.forced_valid found completeOne (completeHold row rowMember)
        · have rowsHold := completeEnv_rows packed valid below env holds
          rw [← SlotRuns.packagePieces_rows] at rowsHold
          exact NativeBlock.forced_valid found completeOne
            (rowsHold row (List.mem_flatMap.mpr ⟨piece, pieceMember, rowMember⟩))
  refine ⟨(LogicalPlan.rowsZero_iff_rowsHold _ _ _ layout assignment oneWide).mpr ?_,
    sboxAt _ _ (blocks_eq piecesEq) sboxEq rowCount, fun column inside => ?_,
    WideLayout.extend_norm _ bounded⟩
  · rw [relationRows_eq piecesEq sboxEq packed, R1CS.rowsHold_append]
    exact ⟨nativeAt, leanRows⟩
  · rw [WideLayout.sourceEnv_widen]
    have publicEq : publicColumns = 320 := rfl
    have countEq : LogicalPlan.publicCount = 320 := rfl
    exact (sourceEnv_honest slots completed (by omega) (Or.inl (by omega))).trans
      (publicValue column inside)

end NightstreamFPrime.Export.BoundedDepth.NativeBooleanCompleteness
