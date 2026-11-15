import NightstreamFPrime.Export.BoundedDepth.NativeCompleteness
import NightstreamFPrime.Export.Generic.PackedBoolean

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Export.BoundedDepth.StepPackage
open NightstreamFPrime.Export.BoundedDepth.SlotCompleteness
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Gadgets.Poseidon2

variable {pieces : List Piece} {next width : Nat}

theorem completeEnv_output (packed : packable publicColumns pieces = true)
    (below : ∀ piece ∈ pieces, pieceBelow next piece = true) (env : Env)
    (index : Fin (blocksOf pieces).length) (lane : Fin 16) :
    completeEnv pieces next env (((blocksOf pieces).get index).start + 1080 + lane.val) =
      env (((blocksOf pieces).get index).start + 1080 + lane.val) := by
  have nextLe := next_le_sboxStart pieces next
  have member := permutation_mem_of_block (List.get_mem (blocksOf pieces) index)
  have bound := below _ member
  simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq] at bound
  have live := NativeBlock.output_live (packable_framed packed)
    (List.get_mem (blocksOf pieces) index) lane
  simp only [List.get_eq_getElem] at live
  have laneBound := lane.isLt
  rw [completeEnv_low (by omega), executePieces_below _ _ _ _ _ (by omega)]
  simp [liveEnv, live]

theorem completeEnv_refill (fits : Fits publicColumns pieces next width)
    (packed : packable publicColumns pieces = true)
    (below : ∀ piece ∈ pieces, pieceBelow next piece = true) (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ()))) :
    Substitute.refill (outputForms width (blocksOf pieces) (sboxStart pieces next))
      (completeEnv pieces next env) = completeEnv pieces next env := by
  funext column
  by_cases output : outputIn (blocksOf pieces) column = true
  · obtain ⟨index, lane, rfl⟩ := outputIn_get output
    have frame := packable_framed packed
    have member := List.get_mem (blocksOf pieces) index
    have templates := holds _ (permutation_mem_of_block member)
    have invocation := (NativeBlock.constraints_iff_invocation 0 0 _ _
      (frame.linear _ member) env).mp templates
    have final := congrFun (PermutationOutput.invocation_finalLayer _ env invocation) lane
    have finals : SparseLayer.evalState
        (fun coordinate : Fin width => completeEnv pieces next env coordinate.val)
        (fun target : Fin 16 =>
        formOf (width := width) (R1CS.LinearCombination.ofVar
          (sboxColumn (sboxStart pieces next) index.val
            (PoseidonRetainedSlots.finalRow target).val))) =
        fun target => env (NativeBlock.sboxLocal ((blocksOf pieces).get index)
          (PoseidonRetainedSlots.finalRow target)) := by
      funext target
      change (formOf (width := width) (R1CS.LinearCombination.ofVar
        (sboxColumn (sboxStart pieces next) index.val (PoseidonRetainedSlots.finalRow target).val))).eval
          (fun coordinate => completeEnv pieces next env coordinate.val) = _
      have room := (fits.blocks packed).sboxColumns
      have rowBound : (PoseidonRetainedSlots.finalRow target).val < 150 := by
        simpa using (PoseidonRetainedSlots.finalRow target).isLt
      have indexRoom : index.val * 150 + 150 ≤ (blocksOf pieces).length * 150 := by
        have bound := Nat.mul_le_mul_right 150 (Nat.succ_le_of_lt index.isLt)
        simpa [Nat.succ_mul] using bound
      rw [formOf_ofVar_eval _ _ (by unfold sboxColumn; omega)]
      exact completeEnv_sbox index (PoseidonRetainedSlots.finalRow target)
    unfold Substitute.refill
    rw [outputForms_output width _ _ frame.separated index lane]
    change (toCombination (outputStateAt (width := width) (sboxStart pieces next) index.val lane)).eval
      (completeEnv pieces next env) = _
    rw [toCombination_eval]
    simp only [outputStateAt, SparseLayer.eval_external]
    rw [finals, completeEnv_output packed below]
    exact final.symm
  · have outside : outputIn (blocksOf pieces) column = false := by simpa using output
    simp only [Substitute.refill, outputForms_of_not_outputIn width _ outside]

def originalRows (nativeRows : List R1CS.Row) (pieces : List Piece) (next width : Nat) : List R1CS.Row :=
  (nativeRows ++ (PieceRows.lowerPieces (PilotData.canonicalConstraints ())
    (otherPieces pieces) next).1).map
      (Substitute.row (outputForms width (blocksOf pieces) (sboxStart pieces next)))

theorem originalRows_eq (nativeRows : List R1CS.Row)
    (packed : packable publicColumns pieces = true) :
    originalRows nativeRows pieces next width =
      nativeRows.map (Substitute.row (outputForms width (blocksOf pieces) (sboxStart pieces next))) ++
        CircuitRelation.relationRows publicColumns pieces next width := by
  simp only [originalRows, List.map_append, CircuitRelation.relationRows, packed, ite_true]

theorem reads_before (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    {column : Nat} (read : Reads pieces next column) : column < slotEnd pieces next := by
  have startLe : sboxStart pieces next ≤ slotEnd pieces next := Nat.le_add_right _ _
  rcases read with ⟨_, choice⟩ | sbox
  · rcases choice with ⟨r, rowMember, columnMember⟩ |
        ⟨block, blockMember, lane, laneMember, columnMember⟩
    · rw [← SlotRuns.packagePieces_rows] at rowMember
      obtain ⟨piece, pieceMember, inPiece⟩ := List.mem_flatMap.mp rowMember
      have reads := SlotRuns.relationRows_reads piece r inPiece column columnMember
      exact lt_of_lt_of_le (SlotRuns.reads_below below piece pieceMember column reads) startLe
    · have member := SlotRuns.block_mem (next := next) blockMember
      have reads : column ∈ (PackagePiece.block block.start block.lanes).reads :=
        List.mem_flatMap.mpr ⟨lane, laneMember, columnMember⟩
      exact lt_of_lt_of_le (SlotRuns.reads_below below _ member column reads) startLe
  · exact sbox.2

theorem native_rows_hold (nativeRows : List R1CS.Row)
    (nativeBelow : ∀ r ∈ nativeRows, r.VarsBelow next)
    (nativeLive : ∀ r ∈ nativeRows, NativeBlock.rowAvoids (deadIn (blocksOf pieces)) r = true)
    (env : Env) (nativeHold : R1CS.RowsHold env nativeRows) :
    R1CS.RowsHold (completeEnv pieces next env) nativeRows := by
  have liveHold : R1CS.RowsHold (liveEnv pieces env) nativeRows :=
    NativeBlock.rowsHold_congr (dead := deadIn (blocksOf pieces))
      (fun column live => by simp [liveEnv, live]) (List.all_eq_true.mpr nativeLive) nativeHold
  exact R1CS.rowsHold_of_agree_below nativeRows next (liveEnv pieces env) (completeEnv pieces next env)
    nativeBelow (fun column before => (completeEnv_low
      (lt_of_lt_of_le before (next_le_sboxStart pieces next))).trans
        (executePieces_below _ _ _ _ _ before)) liveHold

theorem native_derived_valid (nativeRows : List R1CS.Row)
    (packed : packable publicColumns pieces = true) (publicBelow : publicColumns ≤ next)
    (nativeBelow : ∀ r ∈ nativeRows, r.VarsBelow next)
    (nativeLive : ∀ r ∈ nativeRows, NativeBlock.rowAvoids (deadIn (blocksOf pieces)) r = true)
    (env : Env) (nativeHold : R1CS.RowsHold env nativeRows) (one : env 0 = 1)
    {column : Nat} (derived : BooleanInference.Derived nativeRows column) :
    LowNormSlot.Valid .bit (completeEnv pieces next env column) := by
  have completeOne : completeEnv pieces next env 0 = 1 :=
    (completeEnv_public packed publicBelow env (column := 0) (by decide)).trans one
  exact BooleanInference.derived_valid derived completeOne
    (native_rows_hold nativeRows nativeBelow nativeLive env nativeHold)

theorem original_rows_hold (nativeRows : List R1CS.Row)
    (fits : Fits publicColumns pieces next width)
    (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid)
    (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (nativeBelow : ∀ r ∈ nativeRows, r.VarsBelow next)
    (nativeLive : ∀ r ∈ nativeRows, NativeBlock.rowAvoids (deadIn (blocksOf pieces)) r = true)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    (nativeHold : R1CS.RowsHold env nativeRows) :
    R1CS.RowsHold (completeEnv pieces next env) (originalRows nativeRows pieces next width) := by
  unfold originalRows
  apply (Substitute.rowsHold_substitute _ _ _).mpr
  rw [completeEnv_refill fits packed below env holds, R1CS.rowsHold_append]
  exact ⟨native_rows_hold nativeRows nativeBelow nativeLive env nativeHold,
    completeEnv_rows packed valid below env holds⟩

def NativeSlotKind (nativeRows : List R1CS.Row) (pieces : List Piece) (next : Nat)
    (sources : List Nat) (slot : LowNormAssignment.Slot width) : Prop :=
  slot.kind = .field ∨ slot.kind = .bit ∧
    ((slot.source.val < slotEnd pieces next ∧
      (BooleanInference.Derived nativeRows slot.source.val ∨
        ∃ piece ∈ packagePieces pieces next, slot.source.val ∈ piece.forced)) ∨
      ∃ index : Fin sources.length, slot.source.val = slotEnd pieces next + index.val)

theorem extended_slots_valid (nativeRows : List R1CS.Row) (sources : List Nat)
    (slots : List (LowNormAssignment.Slot width))
    (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid)
    (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (publicBelow : publicColumns ≤ next)
    (nativeBelow : ∀ r ∈ nativeRows, r.VarsBelow next)
    (nativeLive : ∀ r ∈ nativeRows, NativeBlock.rowAvoids (deadIn (blocksOf pieces)) r = true)
    (forced : ∀ source ∈ sources, PackedBoolean.SourceForced nativeRows source)
    (kinds : ∀ slot ∈ slots, NativeSlotKind nativeRows pieces next sources slot)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    (nativeHold : R1CS.RowsHold env nativeRows) (one : env 0 = 1) :
    ∀ slot ∈ slots, slot.Valid (fun column => PackedBoolean.extendSource (slotEnd pieces next)
      sources (completeEnv pieces next env) column.val) := by
  have completeHold := native_rows_hold nativeRows nativeBelow nativeLive env nativeHold
  have completeOne : completeEnv pieces next env 0 = 1 :=
    (completeEnv_public packed publicBelow env (column := 0) (by decide)).trans one
  have baseHold := completeEnv_rows packed valid below env holds
  rw [← SlotRuns.packagePieces_rows] at baseHold
  intro slot member
  rcases kinds slot member with field | ⟨bit, ⟨before, derived | base⟩ | companion⟩
  · simp only [LowNormAssignment.Slot.Valid, field, LowNormSlot.Valid]
  · have sourceValid := BooleanInference.derived_valid derived completeOne completeHold
    simpa only [LowNormAssignment.Slot.Valid, bit,
      PackedBoolean.extendSource_before _ _ _ before] using sourceValid
  · obtain ⟨piece, pieceMember, forced⟩ := base
    obtain ⟨r, rowMember, found⟩ := SlotRuns.forced_row piece forced
    have sourceValid := NativeBlock.forced_valid found completeOne
      (baseHold r (List.mem_flatMap.mpr ⟨piece, pieceMember, rowMember⟩))
    simpa only [LowNormAssignment.Slot.Valid, bit,
      PackedBoolean.extendSource_before _ _ _ before] using sourceValid
  · obtain ⟨index, indexEq⟩ := companion
    have sourceValid := PackedBoolean.complement_valid
      (PackedBoolean.sourceForced_valid (forced _ (List.get_mem sources index)) completeOne completeHold)
    simpa only [LowNormAssignment.Slot.Valid, bit, indexEq,
      PackedBoolean.extendSource_companion] using sourceValid

def loweredRows (kept : List R1CS.Row) (groups : List (List PackedBoolean.Pair))
    (pieces : List Piece) (next width : Nat) : List R1CS.Row :=
  kept ++ groups.map PackedBoolean.row ++ CircuitRelation.relationRows publicColumns pieces next width

theorem native_complete (nativeRows kept : List R1CS.Row)
    (groups : List (List PackedBoolean.Pair)) (sources : List Nat)
    (slots : List (LowNormAssignment.Slot width))
    (fits : Fits publicColumns pieces next width)
    (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid)
    (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (publicBelow : publicColumns ≤ next)
    (nativeBelow : ∀ r ∈ nativeRows, r.VarsBelow next)
    (nativeLive : ∀ r ∈ nativeRows, NativeBlock.rowAvoids (deadIn (blocksOf pieces)) r = true)
    (before : ∀ source ∈ sources, source < slotEnd pieces next)
    (keptRows : ∀ r ∈ kept, r ∈ originalRows nativeRows pieces next width)
    (scope : ∀ r ∈ kept ++ CircuitRelation.relationRows publicColumns pieces next width,
      r.VarsBelow (slotEnd pieces next))
    (paired : ∀ group ∈ groups, ∀ pair ∈ group,
      pair ∈ PackedBoolean.companions (slotEnd pieces next) sources)
    (forced : ∀ source ∈ sources, PackedBoolean.SourceForced nativeRows source)
    (kinds : ∀ slot ∈ slots, NativeSlotKind nativeRows pieces next sources slot)
    (rowFits : LogicalPlan.Fits (loweredRows kept groups pieces next width) width)
    (covered : ∀ r ∈ loweredRows kept groups pieces next width,
      r.VarsSatisfy (fun column => column < width ∧
        (column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column)))
    (readCovered : ∀ column, Reads pieces next column →
      column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column)
    {wide : Nat} (wideFits : ProductionAssignment.logicalWidth slots ≤ wide)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    (nativeHold : R1CS.RowsHold env nativeRows) (oneValue : env 0 = 1)
    (publicNorm : ∀ column, column < publicColumns → centeredMagnitude (env column) < 2) :
    let layout := WideLayout.widen (SlotLayout.layout slots) wideFits
    let assignment := WideLayout.extend (wide := wide)
      (honestAssignment slots (PackedBoolean.extendSource (slotEnd pieces next) sources
        (completeEnv pieces next env)))
    (LogicalPlan.plan (loweredRows kept groups pieces next width) width rowFits layout).RowsZero assignment ∧
      (PoseidonSboxFamilyPlan.plan (sboxInterface layout (blocksOf pieces) (sboxStart pieces next))
        (by have bound := (fits.blocks packed).rowCount; omega)).RowsZero assignment ∧
      (∀ column, column < publicColumns → LogicalPlan.sourceEnv layout assignment column = env column) ∧
      ∀ column, centeredMagnitude (assignment column) < 2 := by
  intro layout assignment
  let completed := completeEnv pieces next env
  let extended := PackedBoolean.extendSource (slotEnd pieces next) sources completed
  let narrow := honestAssignment slots extended
  have room := slotEnd_le fits packed
  have publicStart : LogicalPlan.publicCount ≤ slotEnd pieces next := by
    have nextLe := next_le_sboxStart pieces next
    have startLe : sboxStart pieces next ≤ slotEnd pieces next := Nat.le_add_right _ _
    exact le_trans publicBelow (le_trans nextLe startLe)
  have publicValue : ∀ column, column < LogicalPlan.publicCount → completed column = env column :=
    fun column inside => completeEnv_public packed publicBelow env inside
  have allOld := original_rows_hold nativeRows fits packed valid below nativeBelow nativeLive env holds nativeHold
  have allKept : ∀ r ∈ kept ++ CircuitRelation.relationRows publicColumns pieces next width,
      r ∈ originalRows nativeRows pieces next width := by
    intro r member
    rcases List.mem_append.mp member with left | right
    · exact keptRows r left
    · rw [originalRows_eq nativeRows packed]
      exact List.mem_append_right _ right
  have extendedHold := PackedBoolean.extended_ordered_rows
    (originalRows nativeRows pieces next width) kept
    (CircuitRelation.relationRows publicColumns pieces next width) groups
    (slotEnd pieces next) sources completed before allKept scope paired allOld
  have extendedPublic : ∀ column, column < LogicalPlan.publicCount → extended column = env column := by
    intro column inside
    exact (PackedBoolean.extendSource_before _ _ _ (lt_of_lt_of_le inside publicStart)).trans
      (publicValue column inside)
  have one : narrow (SlotLayout.layout slots).one = 1 :=
    (ProductionAssignment.logicalAssignment_publicColumn _ slots _ ⟨0, by decide⟩).trans
      ((extendedPublic 0 (by decide)).trans oneValue)
  have ordinary : (LogicalPlan.plan (loweredRows kept groups pieces next width) width rowFits
      (SlotLayout.layout slots)).RowsZero narrow := by
    apply (LogicalPlan.rowsZero_iff_rowsHold _ _ _ _ _ one).mpr
    apply R1CS.rowsHold_of_agree (loweredRows kept groups pieces next width) _ extended _ covered
    · intro column coverage
      exact sourceEnv_honest slots extended coverage.1 coverage.2
    · exact extendedHold
  have norm : ∀ column, centeredMagnitude (narrow column) < 2 := by
    apply ProductionAssignment.logicalAssignment_norm
    · intro column
      rw [extendedPublic column.val column.isLt]
      exact publicNorm column.val column.isLt
    · exact extended_slots_valid nativeRows sources slots packed valid below publicBelow
        nativeBelow nativeLive forced kinds env holds nativeHold oneValue
  have oneWide : assignment layout.one = 1 := (WideLayout.one_widen _ wideFits _).trans one
  have represents : ∀ column, Reads pieces next column →
      LogicalPlan.sourceEnv layout assignment column = completed column := by
    intro column read
    rw [WideLayout.sourceEnv_widen]
    have beforeEnd := reads_before below read
    rw [sourceEnv_honest slots extended (lt_of_lt_of_le beforeEnd room) (readCovered column read)]
    exact PackedBoolean.extendSource_before _ _ _ beforeEnd
  have lean := RelationCompleteness.rowsZero_of_holds fits packed valid below env holds layout
    assignment oneWide represents
  unfold CircuitRelation.plan at lean
  rw [dite_eq_left packed, Plan.append_rowsZero_iff] at lean
  refine ⟨?_, lean.2, ?_, WideLayout.extend_norm _ norm⟩
  · apply (LogicalPlan.rowsZero_iff_rowsHold _ _ _ _ _ oneWide).mpr
    rw [WideLayout.sourceEnv_widen]
    exact (LogicalPlan.rowsZero_iff_rowsHold _ _ _ _ _ one).mp ordinary
  · intro column inside
    rw [WideLayout.sourceEnv_widen]
    exact (sourceEnv_honest slots extended (by
      have low := lt_of_lt_of_le inside publicStart
      exact lt_of_lt_of_le low room) (Or.inl inside)).trans (extendedPublic column inside)

end NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness
