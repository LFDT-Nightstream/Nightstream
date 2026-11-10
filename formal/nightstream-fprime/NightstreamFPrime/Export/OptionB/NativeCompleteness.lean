import NightstreamFPrime.Export.OptionB.SlotCompleteness

/-! Completeness of a package whose application is native (CW-8.3, SPEC.md section 7.7). Rust's
`StepPackage::with_application` gives the package one rows piece with the application's rows, and
then the package pieces of Lean pieces from the first fresh column: the application's Poseidon2
blocks, as permutation pieces, and the rest of the step. So the package's rows are the native
rows and the relation rows of the Lean pieces (`relationRows_eq`). The completed environment of
the Lean pieces (`RelationCompleteness.completeEnv`) satisfies the Lean part
(`RelationCompleteness.rowsZero_of_holds`). The native rows hold in it because they read only
columns below the first fresh column and no dead local (`NativeBlock.rowAvoids`), as
`NativeBlock.blocks_sound` also requires. The slots follow Lean's rule on the package's pieces
(`StepPackage.runsOf`), so the slot layout reads the completed environment on each column that a
row reads, and each bit slot holds 0 or 1 (`native_complete`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.NativeCompleteness

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
open NightstreamFPrime.Export.OptionB.StepPackage
open NightstreamFPrime.Export.OptionB.SlotCompleteness
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {package : Package} {nativeRows : List R1CS.Row} {pieces : List Piece}

theorem blocks_eq
    (piecesEq : package.pieces = .rows nativeRows :: packagePieces pieces package.firstFresh) :
    package.blocks = blocksOf pieces := by
  simp only [Package.blocks, piecesEq, List.filterMap_cons, PackagePiece.block?,
    SlotRuns.packagePieces_blocks]

/-- The rows of a package with a native application: the native rows with the output forms, then
the relation rows of the Lean pieces. -/
theorem relationRows_eq
    (piecesEq : package.pieces = .rows nativeRows :: packagePieces pieces package.firstFresh)
    (sboxEq : package.sboxStart = sboxStart pieces package.firstFresh)
    (packed : packable publicColumns pieces = true) :
    package.relationRows =
      nativeRows.map (Substitute.row (outputForms package.width (blocksOf pieces)
          (sboxStart pieces package.firstFresh))) ++
        CircuitRelation.relationRows publicColumns pieces package.firstFresh package.width := by
  rw [Package.relationRows, blocks_eq piecesEq, sboxEq, piecesEq, List.flatMap_cons,
    SlotRuns.packagePieces_rows, CircuitRelation.relationRows, ite_eq_left packed]
  simp only [PackagePiece.relationRows, List.map_append]

/-- A package with a native application is complete. Let the constraints of the Lean pieces and
the native rows hold in `env`, and let the native rows read only columns below the first fresh
column and no dead local. Then the honest assignment of the package's slots, read from the
completed environment of the Lean pieces and widened to `wide` logical columns, makes the
package's rows and S-box rows zero, carries `env` on the public columns, and lies in the norm
bound. -/
theorem native_complete (bits : List Nat)
    (piecesEq : package.pieces = .rows nativeRows :: packagePieces pieces package.firstFresh)
    (sboxEq : package.sboxStart = sboxStart pieces package.firstFresh)
    (slotsEq : package.slots = runsOf publicColumns package.sboxStart package.pieces bits)
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
    rw [slotsEq, sboxEq]
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
        · have publicEq : publicColumns = 270 := rfl
          have countEq : LogicalPlan.publicCount = 270 := rfl
          have sizeEq : SlotRuns.runsSize package.sboxStart package.pieces =
              slotEnd pieces package.firstFresh := by
            unfold SlotRuns.runsSize slotEnd
            rw [← Package.blocks, blocks_eq piecesEq, sboxEq, Nat.mul_comm]
          rw [slotsEq]
          refine Or.inr (SlotRuns.runsOf_covers (by omega) (by omega) inside
            (Or.inr ⟨⟨.rows nativeRows, ?_, read.1⟩, noOutput⟩))
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
    show ∀ column, centeredMagnitude (honestAssignment (SlotLayout.runSlots package.width
      package.slots) completed column) < 2
    rw [slotsEq]
    apply norm_of_runs _ _ _ _ _ (fun column inside => by
      rw [publicValue column inside]
      exact publicNorm column inside)
    intro piece member column inPiece
    obtain ⟨row, rowMember, found⟩ := SlotRuns.forced_row piece inPiece
    rw [piecesEq] at member
    rcases List.mem_cons.mp member with rfl | member
    · exact NativeBlock.forced_valid found completeOne (completeHold row rowMember)
    · have rowsHold := completeEnv_rows packed valid below env holds
      rw [← SlotRuns.packagePieces_rows] at rowsHold
      exact NativeBlock.forced_valid found completeOne
        (rowsHold row (List.mem_flatMap.mpr ⟨piece, member, rowMember⟩))
  refine ⟨(LogicalPlan.rowsZero_iff_rowsHold _ _ _ layout assignment oneWide).mpr ?_,
    sboxAt _ _ (blocks_eq piecesEq) sboxEq rowCount, fun column inside => ?_,
    WideLayout.extend_norm _ bounded⟩
  · rw [relationRows_eq piecesEq sboxEq packed, R1CS.rowsHold_append]
    exact ⟨nativeAt, leanRows⟩
  · rw [WideLayout.sourceEnv_widen]
    have publicEq : publicColumns = 270 := rfl
    have countEq : LogicalPlan.publicCount = 270 := rfl
    exact (sourceEnv_honest slots completed (by omega) (Or.inl (by omega))).trans
      (publicValue column inside)

end NightstreamFPrime.Export.OptionB.NativeCompleteness
