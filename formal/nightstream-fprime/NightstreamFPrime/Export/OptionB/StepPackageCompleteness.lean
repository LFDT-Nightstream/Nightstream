import NightstreamFPrime.Export.OptionB.StepPackage
import NightstreamFPrime.Export.Generic.CircuitRelationNorm

/-! Completeness at the package level (CW-5.9). A package's slots (`StepPackage.runsOf`) give a
slot to each column that its relation reads (`runsOf_covers`, `readsCovered_runsOf`), so a
circuit's slots (`slotRuns`) cover its relation when the blocks pack (`covers_packageOf`). Each
slot is a field slot, or a bit slot on a bit candidate that a row of the package forces to 0 or 1
(`runsOf_kind`, `slotRuns_kind`, `slotRuns_forced`), so each bit slot holds 0 or 1 wherever the rows
hold. For a package that Rust builds, such as one whose application region holds one native rows
piece and native Poseidon2 blocks, `package_complete` is the companion of
`CircuitRelationCompleteness.rowsZero_of_pieces`: when the pieces' rows hold in the honest
environment (a ring piece's rows with its honest quotients, `PieceRowsCompleteness.ringRows_complete`)
and every block's template constraints hold, the prover's assignment (`blockAssignment`) makes the
package's rows and S-box rows zero, and its public input encodes the honest digest. When the
application is native, its rows and its blocks' template constraints, with the rest of the step,
suffice (`native_complete`); at the slots of the package's rule the assignment is also in the norm
bound (`native_normComplete`). -/

namespace NightstreamFPrime.Export.OptionB.StepPackageCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.NativeBlock
open NightstreamFPrime.Export.Generic.NativeBlockCompleteness
open NightstreamFPrime.Export.Generic.SlotLayout
open NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
open NightstreamFPrime.Export.OptionB.StepPackage
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ### The runs of `runsOf` -/

/-- The column count of `slotRuns`: up to the last S-box column. -/
def size (pieces : List Piece) (next : Nat) : Nat :=
  CircuitRelation.sboxStart pieces next + 150 * (CircuitRelation.blocksOf pieces).length

/-- The column count of `runsOf`: up to the last S-box column. -/
def runsSize (sboxStart : Nat) (pieces : List PackagePiece) : Nat :=
  sboxStart + 150 * (pieces.filterMap PackagePiece.block?).length

/-- The columns that the pieces read, as `runsOf` marks them. -/
def readMask (sboxStart : Nat) (pieces : List PackagePiece) : Array Bool :=
  pieces.foldl (fun mask piece => markColumns mask piece.reads)
    (Array.replicate (runsSize sboxStart pieces) false)

/-- The bit candidates, as `runsOf` marks them. -/
def bitMask (sboxStart : Nat) (pieces : List PackagePiece) (bits : List Nat) : Array Bool :=
  markColumns (Array.replicate (runsSize sboxStart pieces) false) bits

/-- The columns that a row forces to 0 or 1, as `runsOf` marks them. -/
def forcedMask (sboxStart : Nat) (pieces : List PackagePiece) : Array Bool :=
  pieces.foldl (fun mask piece => markColumns mask piece.forced)
    (Array.replicate (runsSize sboxStart pieces) false)

/-- The kind code of each column, as `runsOf` computes it. -/
def codeOf (sboxStart : Nat) (pieces : List PackagePiece) (bits : List Nat) : Nat → Nat :=
  slotCode sboxStart (readMask sboxStart pieces) (bitMask sboxStart pieces bits)
    (forcedMask sboxStart pieces)

theorem runsOf_eq (publicColumns sboxStart : Nat) (pieces : List PackagePiece) (bits : List Nat) :
    runsOf publicColumns sboxStart pieces bits =
      (((List.range (runsSize sboxStart pieces)).drop publicColumns).foldl
        (runStep (codeOf sboxStart pieces bits)) []).reverse :=
  rfl

theorem slotRuns_eq (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) :
    slotRuns publicColumns pieces next batches =
      runsOf publicColumns (CircuitRelation.sboxStart pieces next) (packagePieces pieces next)
        (bitColumns batches) :=
  rfl

theorem foldl_invariant {α β : Type} (invariant : β → Prop) (step : β → α → β) :
    ∀ (items : List α) (initial : β), (∀ item ∈ items, ∀ value, invariant value →
      invariant (step value item)) → invariant initial → invariant (items.foldl step initial)
  | [], _, _, holds => holds
  | item :: rest, _, preserves, holds =>
      foldl_invariant invariant step rest _
        (fun other member => preserves other (List.mem_cons_of_mem _ member))
        (preserves item List.mem_cons_self _ holds)

/-- A read column and an S-box column have a slot code. -/
theorem slotCode_ne {sboxStart : Nat} {reads bits forced : Array Bool} {column : Nat}
    (read : reads.getD column false = true ∨ sboxStart ≤ column) :
    slotCode sboxStart reads bits forced column ≠ 3 := by
  unfold slotCode
  split
  · split <;> decide
  · rename_i outside
    exfalso
    apply outside
    rcases read with marked | high
    · simp [marked]
    · simp [high]

/-- A code is 2 or 3, or 0 on a column that both the bit mask and the forced mask mark. -/
theorem slotCode_cases (sboxStart : Nat) (reads bits forced : Array Bool) (column : Nat) :
    slotCode sboxStart reads bits forced column = 2 ∨
      slotCode sboxStart reads bits forced column = 3 ∨
      (slotCode sboxStart reads bits forced column = 0 ∧ bits.getD column false = true ∧
        forced.getD column false = true) := by
  unfold slotCode
  split
  · split
    · rename_i both
      simp only [Bool.and_eq_true] at both
      exact Or.inr (Or.inr ⟨rfl, both.1, both.2⟩)
    · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)

/-- A code 0 marks a bit candidate that a piece forces to 0 or 1. -/
theorem codeOf_cases (sboxStart : Nat) (pieces : List PackagePiece) (bits : List Nat)
    (column : Nat) :
    codeOf sboxStart pieces bits column = 2 ∨ codeOf sboxStart pieces bits column = 3 ∨
      (codeOf sboxStart pieces bits column = 0 ∧ column ∈ bits ∧
        ∃ piece ∈ pieces, column ∈ piece.forced) := by
  rcases slotCode_cases sboxStart (readMask sboxStart pieces) (bitMask sboxStart pieces bits)
      (forcedMask sboxStart pieces) column with two | three | ⟨zero, bit, forced⟩
  · exact Or.inl two
  · exact Or.inr (Or.inl three)
  · refine Or.inr (Or.inr ⟨zero, ?_, ?_⟩)
    · rcases (markColumns_getD _ bits column).mp bit with marked | ⟨_, member⟩
      · rw [replicate_getD] at marked
        exact absurd marked (by decide)
      · exact member
    · rcases (markAll_getD PackagePiece.forced pieces _ column).mp forced with
        marked | ⟨_, piece, pieceMember, member⟩
      · rw [replicate_getD] at marked
        exact absurd marked (by decide)
      · exact ⟨piece, pieceMember, member⟩

def Covered (runs : List (Nat × Nat × Nat)) (column : Nat) : Prop :=
  ∃ run ∈ runs, run.1 ≤ column ∧ column < run.1 + run.2.1

theorem runStep_mono (code : Nat → Nat) (runs : List (Nat × Nat × Nat)) (column other : Nat)
    (covered : Covered runs other) : Covered (runStep code runs column) other := by
  obtain ⟨run, member, low, high⟩ := covered
  unfold runStep
  split
  · exact ⟨run, member, low, high⟩
  · match runs, member with
    | (start, count, kind) :: rest, member =>
        dsimp only
        split
        · rcases List.mem_cons.mp member with rfl | member
          · exact ⟨(start, count + 1, kind), List.mem_cons_self, low, by simp only at high ⊢; omega⟩
          · exact ⟨run, List.mem_cons_of_mem _ member, low, high⟩
        · exact ⟨run, List.mem_cons_of_mem _ member, low, high⟩

theorem runStep_covers (code : Nat → Nat) (runs : List (Nat × Nat × Nat)) (column : Nat)
    (marked : code column ≠ 3) : Covered (runStep code runs column) column := by
  unfold runStep
  rw [ite_eq_right marked]
  match runs with
  | (start, count, kind) :: rest =>
      dsimp only
      split
      · exact ⟨(start, count + 1, kind), List.mem_cons_self, by omega, by simp only; omega⟩
      · exact ⟨(column, 1, _), List.mem_cons_self, le_refl _, by simp⟩
  | [] => exact ⟨(column, 1, _), List.mem_cons_self, le_refl _, by simp⟩

theorem runFold_mono (code : Nat → Nat) :
    ∀ (columns : List Nat) (runs : List (Nat × Nat × Nat)) (other : Nat),
      Covered runs other → Covered (columns.foldl (runStep code) runs) other
  | [], _, _, covered => covered
  | column :: rest, runs, other, covered =>
      runFold_mono code rest _ other (runStep_mono code runs column other covered)

theorem runFold_covers (code : Nat → Nat) :
    ∀ (columns : List Nat) (runs : List (Nat × Nat × Nat)) (column : Nat),
      column ∈ columns → code column ≠ 3 → Covered (columns.foldl (runStep code) runs) column
  | [], _, _, member, _ => by simp at member
  | first :: rest, runs, column, member, marked => by
      rw [List.foldl_cons]
      rcases List.mem_cons.mp member with rfl | member
      · exact runFold_mono code rest _ column (runStep_covers code runs column marked)
      · exact runFold_covers code rest _ column member marked

/-- `runsOf` puts each S-box column and each column that a piece reads, from the public columns to
the last S-box column, in a run. -/
theorem runsOf_covers (publicColumns sboxStart : Nat) (pieces : List PackagePiece)
    (bits : List Nat) {column : Nat} (low : publicColumns ≤ column)
    (high : column < runsSize sboxStart pieces)
    (read : sboxStart ≤ column ∨ ∃ piece ∈ pieces, column ∈ piece.reads) :
    ∃ run ∈ runsOf publicColumns sboxStart pieces bits, run.1 ≤ column ∧
      column < run.1 + run.2.1 := by
  rw [runsOf_eq]
  have member : column ∈ (List.range (runsSize sboxStart pieces)).drop publicColumns := by
    rw [List.range_eq_range', List.drop_range', List.mem_range'_1]
    omega
  have marked : codeOf sboxStart pieces bits column ≠ 3 := by
    apply slotCode_ne
    rcases read with sbox | ⟨piece, pieceMember, inPiece⟩
    · exact Or.inr sbox
    · refine Or.inl ((markAll_getD PackagePiece.reads pieces _ column).mpr (Or.inr ⟨?_, piece,
        pieceMember, inPiece⟩))
      simpa using high
  obtain ⟨run, runMember, runLow, runHigh⟩ := runFold_covers _ _ [] column member marked
  exact ⟨run, List.mem_reverse.mpr runMember, runLow, runHigh⟩

/-- Each run's columns carry its kind code, which is not 3. -/
def RunsKinds (code : Nat → Nat) (runs : List (Nat × Nat × Nat)) : Prop :=
  ∀ run ∈ runs, run.2.2 ≠ 3 ∧
    ∀ column, run.1 ≤ column → column < run.1 + run.2.1 → code column = run.2.2

theorem runsKinds_cons {code : Nat → Nat} {run : Nat × Nat × Nat}
    {runs : List (Nat × Nat × Nat)} (kind : run.2.2 ≠ 3)
    (columns : ∀ column, run.1 ≤ column → column < run.1 + run.2.1 → code column = run.2.2)
    (rest : RunsKinds code runs) : RunsKinds code (run :: runs) := by
  intro other member
  rcases List.mem_cons.mp member with rfl | member
  · exact ⟨kind, columns⟩
  · exact rest other member

theorem runStep_kinds (code : Nat → Nat) (runs : List (Nat × Nat × Nat)) (column : Nat)
    (kinds : RunsKinds code runs) : RunsKinds code (runStep code runs column) := by
  have fresh : ∀ others : List (Nat × Nat × Nat), RunsKinds code others →
      code column ≠ 3 → RunsKinds code ((column, 1, code column) :: others) :=
    fun others rest live => runsKinds_cons live (fun other low high => by
      simp only at low high
      rw [show other = column by omega]) rest
  unfold runStep
  split
  · exact kinds
  · rename_i live
    match runs, kinds with
    | (start, count, kind) :: rest, kinds =>
        dsimp only
        split
        · rename_i extend
          obtain ⟨kindNe, columns⟩ := kinds _ List.mem_cons_self
          refine runsKinds_cons kindNe (fun other low high => ?_)
            (fun run member => kinds run (List.mem_cons_of_mem _ member))
          simp only at low high columns ⊢
          by_cases last : other < start + count
          · exact columns other low last
          · rw [show other = column by omega]
            exact extend.1.symm
        · exact fresh _ kinds live
    | [], _ => exact fresh [] (fun _ member => by simp at member) live

/-- A slot of `runsOf` is a field slot, or a bit slot on a bit candidate that a piece forces to 0
or 1. -/
theorem runsOf_kind (publicColumns sboxStart : Nat) (pieces : List PackagePiece)
    (bits : List Nat) (width : Nat) {slot : LowNormAssignment.Slot width}
    (member : slot ∈ runSlots width (runsOf publicColumns sboxStart pieces bits)) :
    slot.kind = .field ∨ (slot.kind = .bit ∧ slot.source.val ∈ bits ∧
      ∃ piece ∈ pieces, slot.source.val ∈ piece.forced) := by
  rw [runsOf_eq] at member
  unfold runSlots at member
  obtain ⟨⟨start, count, kind⟩, runMember, slotMember⟩ := List.mem_flatMap.mp member
  obtain ⟨offset, offsetMember, found⟩ := List.mem_filterMap.mp slotMember
  have offsetLt := List.mem_range.mp offsetMember
  split at found
  · obtain rfl := Option.some.inj found
    have kinds := foldl_invariant (RunsKinds (codeOf sboxStart pieces bits))
      (runStep (codeOf sboxStart pieces bits))
      ((List.range (runsSize sboxStart pieces)).drop publicColumns) []
      (fun column _ runs kinds => runStep_kinds _ runs column kinds)
      (fun _ member => by simp at member)
    obtain ⟨kindNe, columns⟩ := kinds _ (List.mem_reverse.mp runMember)
    have codeEq := columns (start + offset) (by simp only; omega) (by simp only; omega)
    simp only at kindNe codeEq ⊢
    rcases codeOf_cases sboxStart pieces bits (start + offset) with
      two | three | ⟨zero, candidate, forced⟩
    · rw [← codeEq, two]
      exact Or.inl rfl
    · exact absurd (codeEq.symm.trans three) kindNe
    · rw [← codeEq, zero]
      exact Or.inr ⟨rfl, candidate, forced⟩
  · exact absurd found (by simp)

/-! ### What a package's rows read -/

/-- A lane of `lanes`, or the zero lane past them, reads only the lanes' columns. -/
theorem getD_varsSatisfy (lanes : List Expr) (index : Nat) :
    (lanes.getD index 0).VarsSatisfy (· ∈ lanes.flatMap exprColumns) := by
  by_cases inside : index < lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact (exprColumns_varsSatisfy _).mono _ fun column member =>
      List.mem_flatMap.mpr ⟨_, List.getElem_mem inside, member⟩
  · rw [List.getD_eq_default _ _ (by omega)]
    trivial

/-- The rows of a piece read only the piece's columns (`PackagePiece.reads`). -/
theorem relationRows_reads :
    ∀ (piece : PackagePiece), ∀ row ∈ piece.relationRows, ∀ column ∈ rowColumns row,
      column ∈ piece.reads
  | .rows chunk, row, member, column, columnMember =>
      List.mem_flatMap.mpr ⟨row, member, columnMember⟩
  | .ring cells start quotient lanes, row, member, column, columnMember => by
      have laneReads : ∀ index, (lanes.getD index 0).VarsSatisfy
          (· ∈ (PackagePiece.ring cells start quotient lanes).reads) := fun index =>
        (getD_varsSatisfy lanes index).mono _ fun _ inLanes => List.mem_append_right _ inLanes
      have scope := PieceRowsCompleteness.ringRows_varsSatisfy cells start quotient lanes
        (fun lane => laneReads lane.val) (fun _ _ lane => laneReads _) (fun _ _ lane => laneReads _)
        (fun cell inside lane => List.mem_append_left _ (List.mem_append_left _
          (List.mem_range'_1.mpr ⟨by simp [PieceRows.outputColumn],
            by have := PieceRows.slot_lt lane inside; simp only [PieceRows.outputColumn]; omega⟩)))
        (fun cell inside lane => List.mem_append_left _ (List.mem_append_right _
          (List.mem_range'_1.mpr ⟨by simp [PieceRows.quotientColumn],
            PieceRowsCompleteness.quotientColumn_lt inside lane⟩)))
        row member
      exact rowColumns_of_varsSatisfy scope column columnMember
  | .block .., _, member, _, _ => by simp [PackagePiece.relationRows] at member

/-- A column that a piece forces to 0 or 1 has a row of the piece that forces it. -/
theorem forced_row :
    ∀ (piece : PackagePiece) {column : Nat}, column ∈ piece.forced →
      ∃ row ∈ piece.relationRows, SlotLayout.forcedColumn? row = some column
  | .rows _, _, member => List.mem_filterMap.mp member
  | .ring .., _, member => by simp [PackagePiece.forced] at member
  | .block .., _, member => by simp [PackagePiece.forced] at member

/-- The slots of `runsOf` cover every column that the package's relation reads. -/
theorem readsCovered_runsOf (publicColumns sboxStart : Nat) (pieces : List PackagePiece)
    (bits : List Nat) (width : Nat) (publicLe : publicColumns ≤ LogicalPlan.publicCount)
    (fits : sboxStart + (pieces.filterMap PackagePiece.block?).length * 150 ≤ width) :
    ReadsCovered (runSlots width (runsOf publicColumns sboxStart pieces bits))
      (pieces.flatMap PackagePiece.relationRows) (pieces.filterMap PackagePiece.block?)
      sboxStart := by
  intro column high reads
  by_cases isPublic : column < LogicalPlan.publicCount
  · exact Or.inl isPublic
  · have read : sboxStart ≤ column ∨ ∃ piece ∈ pieces, column ∈ piece.reads := by
      rcases reads with ⟨row, rowMember, columnMember⟩ | ⟨block, blockMember, choice⟩ | sbox
      · obtain ⟨piece, pieceMember, inPiece⟩ := List.mem_flatMap.mp rowMember
        exact Or.inr ⟨piece, pieceMember, relationRows_reads piece row inPiece column columnMember⟩
      · obtain ⟨piece, pieceMember, found⟩ := List.mem_filterMap.mp blockMember
        cases piece with
        | rows => simp [PackagePiece.block?] at found
        | ring => simp [PackagePiece.block?] at found
        | block start lanes =>
            simp only [PackagePiece.block?, Option.some.injEq] at found
            subst found
            refine Or.inr ⟨_, pieceMember, ?_⟩
            rcases choice with ⟨lane, laneMember, columnMember⟩ | ⟨outputLow, outputHigh⟩
            · exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨lane, laneMember, columnMember⟩)
            · exact List.mem_append_left _ (List.mem_range'_1.mpr ⟨outputLow, by omega⟩)
      · exact Or.inl sbox.1
    obtain ⟨⟨start, count, kind⟩, member, low, runHigh⟩ := runsOf_covers publicColumns sboxStart
      pieces bits (by omega) (by unfold runsSize; omega) read
    exact Or.inr ⟨_, runSlots_mem member low runHigh (by omega), rfl⟩

/-! ### The slots of a circuit's package -/

theorem packagePieces_rows (pieces : List Piece) (next : Nat) :
    (packagePieces pieces next).flatMap PackagePiece.relationRows =
      (PieceRows.lowerPieces (PilotData.canonicalConstraints ())
        (CircuitRelation.otherPieces pieces) next).1 := by
  simp only [packagePieces, packageAux_eq, List.reverse_nil, List.nil_append,
    flatMap_packageSpec, PieceRows.lowerPieces, lowerPiecesAux_eq _ rfl]

theorem packagePieces_blocks (pieces : List Piece) (next : Nat) :
    (packagePieces pieces next).filterMap PackagePiece.block? = CircuitRelation.blocksOf pieces := by
  simp only [packagePieces, packageAux_eq, List.reverse_nil, List.nil_append, blocks_packageSpec]

/-- `slotRuns` puts each S-box column and each column that a package piece reads, from the public
columns to the last S-box column, in a run. -/
theorem slotRuns_covers (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) {column : Nat} (low : publicColumns ≤ column)
    (high : column < size pieces next)
    (read : CircuitRelation.sboxStart pieces next ≤ column ∨
      ∃ piece ∈ packagePieces pieces next, column ∈ piece.reads) :
    ∃ run ∈ slotRuns publicColumns pieces next batches, run.1 ≤ column ∧
      column < run.1 + run.2.1 :=
  runsOf_covers _ _ _ _ low (by rw [runsSize, packagePieces_blocks]; exact high) read

/-- When the blocks pack, the relation fits the width up to the last S-box column. -/
theorem size_le {publicColumns : Nat} {pieces : List Piece} {next width : Nat}
    (fits : CircuitRelation.Fits publicColumns pieces next width)
    (packed : CircuitRelation.packable publicColumns pieces = true) : size pieces next ≤ width := by
  have sboxColumns := (fits.blocks packed).sboxColumns
  unfold size
  rw [Nat.mul_comm]
  exact sboxColumns

/-- The slots of `slotRuns` cover the relation of the circuit's pieces when the blocks pack:
each column that the relation reads has a slot. -/
theorem covers_packageOf (pieces : List Piece) (next width : Nat) (batches : List WitnessBatch)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns pieces = true)
    (fits : size pieces next ≤ width) :
    CircuitRelationCompleteness.Covers
      (runSlots width (slotRuns CircuitRelation.publicColumns pieces next batches))
      CircuitRelation.publicColumns pieces next := by
  refine ⟨fun _ => ?_, fun notPacked => absurd packed (by simp [notPacked])⟩
  have cover := readsCovered_runsOf CircuitRelation.publicColumns
    (CircuitRelation.sboxStart pieces next) (packagePieces pieces next) (bitColumns batches) width
    (by decide) (by rw [packagePieces_blocks]; unfold size at fits; omega)
  rw [packagePieces_rows, packagePieces_blocks] at cover
  exact cover

/-! ### The kinds of `slotRuns` -/

/-- `column` is the column of a `bit` hint of `batches`. -/
def BitColumn (batches : List WitnessBatch) (column : Nat) : Prop :=
  ∃ batch ∈ batches, ∃ index source bit, batch.hints[index]? = some (Hint.bit source bit) ∧
    column = batch.start + batch.recipes.length + index

theorem bitColumns_mem {batches : List WitnessBatch} {column : Nat}
    (member : column ∈ bitColumns batches) : BitColumn batches column := by
  obtain ⟨batch, batchMember, inBatch⟩ := List.mem_flatMap.mp member
  obtain ⟨⟨hint, index⟩, entryMember, found⟩ := List.mem_filterMap.mp inBatch
  cases hint with
  | bit source position =>
      obtain rfl := Option.some.inj found
      exact ⟨batch, batchMember, index, source, position,
        List.mem_zipIdx_iff_getElem?.mp entryMember, rfl⟩
  | inverseOrZero source => simp at found
  | quotientFive source => simp at found
  | remainderFive source => simp at found

/-- A slot of `slotRuns` is a field slot, or a bit slot on the column of a `bit` hint. -/
theorem slotRuns_kind (publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (batches : List WitnessBatch) {slot : LowNormAssignment.Slot width}
    (member : slot ∈ runSlots width (slotRuns publicColumns pieces next batches)) :
    slot.kind = .field ∨ (slot.kind = .bit ∧ BitColumn batches slot.source.val) := by
  rcases runsOf_kind publicColumns (CircuitRelation.sboxStart pieces next)
      (packagePieces pieces next) (bitColumns batches) width member with
    field | ⟨bit, candidate, _⟩
  · exact Or.inl field
  · exact Or.inr ⟨bit, bitColumns_mem candidate⟩

/-- A slot of `slotRuns` other than a field slot is a bit slot whose column a lowered row of the
pieces outside the blocks forces to 0 or 1. -/
theorem slotRuns_forced (publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (batches : List WitnessBatch) {slot : LowNormAssignment.Slot width}
    (member : slot ∈ runSlots width (slotRuns publicColumns pieces next batches))
    (nonField : slot.kind ≠ .field) :
    slot.kind = .bit ∧ ∃ row ∈ (PieceRows.lowerPieces (PilotData.canonicalConstraints ())
      (CircuitRelation.otherPieces pieces) next).1,
      SlotLayout.forcedColumn? row = some slot.source.val := by
  rcases runsOf_kind publicColumns (CircuitRelation.sboxStart pieces next)
      (packagePieces pieces next) (bitColumns batches) width member with
    field | ⟨bit, _, piece, pieceMember, forced⟩
  · exact absurd field nonField
  · obtain ⟨row, rowMember, found⟩ := forced_row piece forced
    refine ⟨bit, row, ?_, found⟩
    rw [← packagePieces_rows]
    exact List.mem_flatMap.mpr ⟨piece, pieceMember, rowMember⟩

/-- The package's slots are valid for an environment that holds each `bit` hint's value at its
column, below `next` (`NativeBlock.bitHint_valid`). -/
theorem slotRuns_valid (publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (batches : List WitnessBatch) (env : Env)
    (hints : ∀ batch ∈ batches, ∀ index source bit,
      batch.hints[index]? = some (Hint.bit source bit) →
        batch.start + batch.recipes.length + index < next ∧
          env (batch.start + batch.recipes.length + index) = Hint.eval env (.bit source bit)) :
    CircuitRelationNorm.SlotsValid (runSlots width (slotRuns publicColumns pieces next batches))
      next env := by
  intro slot member nonField
  rcases slotRuns_kind publicColumns pieces next width batches member with
    field | ⟨bit, batch, batchMember, index, source, position, found, columnEq⟩
  · exact absurd field nonField
  · obtain ⟨below, value⟩ := hints batch batchMember index source position found
    rw [columnEq, bit, value]
    exact ⟨below, bitHint_valid env source position⟩

/-! ### Rust-built packages -/

/-- The prover's assignment of a package: the S-box columns filled from the honest
environment, then encoded in the package's slots. -/
def packageAssignment (package : Package) (honest : Env) :
    Assignment F (ProductionAssignment.logicalWidth (runSlots package.width package.slots)) :=
  blockAssignment (runSlots package.width package.slots) package.blocks package.sboxStart honest

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Package completeness, the companion of `CircuitRelationCompleteness.rowsZero_of_pieces` for a
package that Rust builds: when the pieces' rows hold in the honest environment, every block's
template constraints hold there, the rows and the blocks read below the S-box columns, and every
column that the relation reads has a slot, the prover's assignment makes the package's rows and
S-box rows zero. Its public input is the honest one. -/
theorem package_complete (package : Package) (frame : Framed package.blocks)
    (rowsBelow : ∀ row ∈ package.pieces.flatMap PackagePiece.relationRows,
      row.VarsBelow package.sboxStart)
    (blocksBelow : ∀ block ∈ package.blocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()),
      expression.VarsBelow package.sboxStart)
    (publicBelow : LogicalPlan.publicCount ≤ package.sboxStart)
    (sboxFits : package.sboxStart + package.blocks.length * 150 ≤ package.width)
    (covered : ReadsCovered (runSlots package.width package.slots)
      (package.pieces.flatMap PackagePiece.relationRows) package.blocks package.sboxStart)
    (fits : LogicalPlan.Fits package.relationRows package.width)
    (rowCount_le : package.blocks.length * 150 ≤ 2 ^ cubeVariables)
    (honest : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) honest = encHash digest)
    (rowsHold : R1CS.RowsHold honest (package.pieces.flatMap PackagePiece.relationRows))
    (templates : ∀ block ∈ package.blocks, ConstraintsHold honest
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()))) :
    (LogicalPlan.plan package.relationRows package.width fits
        (layout (runSlots package.width package.slots))).RowsZero
        (packageAssignment package honest) ∧
      (PoseidonSboxFamilyPlan.plan (CircuitRelation.sboxInterface
          (layout (runSlots package.width package.slots)) package.blocks package.sboxStart)
        rowCount_le).RowsZero (packageAssignment package honest) ∧
      LogicalPlan.publicInput (publicFits := publicFits) (LogicalPlan.sourceEnv
        (layout (runSlots package.width package.slots)) (packageAssignment package honest)) =
        encHash digest := by
  have one := CircuitRelationCompleteness.one_of_encoded (publicFits := publicFits) honest digest
    encoded
  obtain ⟨sboxRows, rows, publicEq⟩ := blocks_rows_complete
    (runSlots package.width package.slots) frame package.sboxStart rowsBelow blocksBelow
    publicBelow sboxFits covered rowCount_le honest one rowsHold templates
  have oneEq : packageAssignment package honest
      (layout (runSlots package.width package.slots)).one = 1 := by
    unfold packageAssignment blockAssignment
    rw [encode_one, sboxFill_outside (Or.inl (lt_of_lt_of_le (by decide) publicBelow))]
    exact one
  refine ⟨(LogicalPlan.rowsZero_iff_rowsHold _ _ fits _ _ oneEq).mpr rows, sboxRows, ?_⟩
  rw [← encoded]
  funext column
  exact publicEq column.val column.isLt

/-! ### Packages with a native application -/

theorem native_blocks {package : Package} {nativeRows : List R1CS.Row}
    {nativeBlocks : List Block} {rest : List Piece}
    (piecesEq : package.pieces = PackagePiece.rows nativeRows ::
      ((nativeBlocks.map fun block => PackagePiece.block block.start block.lanes) ++
        packagePieces rest package.firstFresh)) :
    package.blocks = nativeBlocks ++ CircuitRelation.blocksOf rest := by
  have nativeEq : ∀ blocks : List Block, (blocks.map fun block =>
      PackagePiece.block block.start block.lanes).filterMap PackagePiece.block? = blocks := by
    intro blocks
    induction blocks with
    | nil => rfl
    | cons block others inductionHypothesis =>
        simp only [List.map_cons, List.filterMap_cons, PackagePiece.block?, inductionHypothesis]
  simp only [Package.blocks, piecesEq, List.filterMap_cons, PackagePiece.block?,
    List.filterMap_append, nativeEq, packagePieces, packageAux_eq, List.reverse_nil,
    List.nil_append, blocks_packageSpec]

theorem native_rows {package : Package} {nativeRows : List R1CS.Row}
    {nativeBlocks : List Block} {rest : List Piece}
    (piecesEq : package.pieces = PackagePiece.rows nativeRows ::
      ((nativeBlocks.map fun block => PackagePiece.block block.start block.lanes) ++
        packagePieces rest package.firstFresh)) :
    package.pieces.flatMap PackagePiece.relationRows = nativeRows ++
      (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (CircuitRelation.otherPieces rest)
        package.firstFresh).1 := by
  have nativeEq : (nativeBlocks.map fun block =>
      PackagePiece.block block.start block.lanes).flatMap PackagePiece.relationRows = [] := by
    simp [List.flatMap_map, PackagePiece.relationRows]
  simp only [piecesEq, List.flatMap_cons, List.flatMap_append, nativeEq, PackagePiece.relationRows,
    List.nil_append, packagePieces, packageAux_eq, List.reverse_nil, flatMap_packageSpec,
    PieceRows.lowerPieces, lowerPiecesAux_eq _ rfl]

/-- Completeness of a package whose application is native (`with_application` in Rust): one rows
piece and the native Poseidon2 blocks lead the pieces, and the rest are the package pieces of Lean
pieces with fresh columns from the first fresh column. It needs only that the native rows and the
native blocks' template constraints hold in the honest environment, and that the rest of the step
holds there: each Lean piece's constraints hold and read below the first fresh column. The prover
fills the rest's fresh columns (`PieceRowsCompleteness.fillPieces`): the quotient of each ring
product (`ringRows_complete`) and the products of each lowering (`lowerPieces_complete`). Then
`package_complete` gives zero rows and S-box rows and the honest public input. Every column that
the relation reads needs a slot (`ReadsCovered`). -/
theorem native_complete (package : Package) (nativeRows : List R1CS.Row)
    (nativeBlocks : List Block) (rest : List Piece)
    (piecesEq : package.pieces = PackagePiece.rows nativeRows ::
      ((nativeBlocks.map fun block => PackagePiece.block block.start block.lanes) ++
        packagePieces rest package.firstFresh))
    (frame : Framed package.blocks)
    (nativeBelow : ∀ row ∈ nativeRows, row.VarsBelow package.firstFresh)
    (nativeBlocksBelow : ∀ block ∈ nativeBlocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()),
      expression.VarsBelow package.firstFresh)
    (restValid : ∀ piece ∈ rest, piece.Valid)
    (restBelow : ∀ piece ∈ rest,
      ∀ expression ∈ piece.constraints (PilotData.canonicalConstraints ()),
        expression.VarsBelow package.firstFresh)
    (restEnd : CircuitRelation.sboxStart rest package.firstFresh ≤ package.sboxStart)
    (publicBelow : LogicalPlan.publicCount ≤ package.firstFresh)
    (sboxFits : package.sboxStart + package.blocks.length * 150 ≤ package.width)
    (covered : ReadsCovered (runSlots package.width package.slots)
      (package.pieces.flatMap PackagePiece.relationRows) package.blocks package.sboxStart)
    (fits : LogicalPlan.Fits package.relationRows package.width)
    (rowCount_le : package.blocks.length * 150 ≤ 2 ^ cubeVariables)
    (honest : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) honest = encHash digest)
    (nativeHold : R1CS.RowsHold honest nativeRows)
    (nativeTemplates : ∀ block ∈ nativeBlocks, ConstraintsHold honest
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ())))
    (restHolds : ∀ piece ∈ rest,
      ConstraintsHold honest (piece.constraints (PilotData.canonicalConstraints ()))) :
    (LogicalPlan.plan package.relationRows package.width fits
        (layout (runSlots package.width package.slots))).RowsZero
        (packageAssignment package (PieceRowsCompleteness.fillPieces
          (PilotData.canonicalConstraints ()) honest (CircuitRelation.otherPieces rest)
          package.firstFresh)) ∧
      (PoseidonSboxFamilyPlan.plan (CircuitRelation.sboxInterface
          (layout (runSlots package.width package.slots)) package.blocks package.sboxStart)
        rowCount_le).RowsZero (packageAssignment package (PieceRowsCompleteness.fillPieces
          (PilotData.canonicalConstraints ()) honest (CircuitRelation.otherPieces rest)
          package.firstFresh)) ∧
      LogicalPlan.publicInput (publicFits := publicFits) (LogicalPlan.sourceEnv
        (layout (runSlots package.width package.slots)) (packageAssignment package
          (PieceRowsCompleteness.fillPieces (PilotData.canonicalConstraints ()) honest
            (CircuitRelation.otherPieces rest) package.firstFresh))) = encHash digest := by
  let canonical := PilotData.canonicalConstraints ()
  let filled := PieceRowsCompleteness.fillPieces canonical honest (CircuitRelation.otherPieces rest)
    package.firstFresh
  have blocksEq := native_blocks piecesEq
  have rowsEq := native_rows piecesEq
  have nextLe : package.firstFresh ≤ package.sboxStart :=
    le_trans (PieceRowsCompleteness.lowerPiecesAux_le canonical (CircuitRelation.otherPieces rest)
      package.firstFresh []) restEnd
  have othersValid : ∀ piece ∈ CircuitRelation.otherPieces rest, piece.Valid :=
    fun piece member => restValid piece (CircuitRelationCompleteness.otherPieces_mem member).1
  have othersBelow : ∀ piece ∈ CircuitRelation.otherPieces rest,
      ∀ expression ∈ piece.constraints canonical, expression.VarsBelow package.firstFresh :=
    fun piece member => restBelow piece (CircuitRelationCompleteness.otherPieces_mem member).1
  have othersHold : ∀ piece ∈ CircuitRelation.otherPieces rest,
      ConstraintsHold honest (piece.constraints canonical) :=
    fun piece member => restHolds piece (CircuitRelationCompleteness.otherPieces_mem member).1
  obtain ⟨loweredRows, agree⟩ := PieceRowsCompleteness.lowerPieces_complete canonical
    (CircuitRelation.otherPieces rest) package.firstFresh honest othersValid othersBelow othersHold
  have low : ∀ column, column < package.firstFresh → filled column = honest column :=
    fun column below => agree column (Or.inl below)
  have allBelow : ∀ block ∈ package.blocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints canonical,
      expression.VarsBelow package.firstFresh := by
    intro block member
    rw [blocksEq] at member
    rcases List.mem_append.mp member with native | other
    · exact nativeBlocksBelow block native
    · exact restBelow _ (CircuitRelationCompleteness.blocksOf_mem other)
  refine package_complete package frame ?_
    (fun block member expression inside =>
      Expr.VarsBelow.mono _ (allBelow block member expression inside) nextLe)
    (le_trans publicBelow nextLe) sboxFits covered fits rowCount_le filled digest ?_ ?_ ?_
  · intro row member
    rw [rowsEq] at member
    rcases List.mem_append.mp member with native | lowered
    · exact R1CS.Row.VarsBelow.mono row (nativeBelow row native) nextLe
    · exact R1CS.Row.VarsBelow.mono row (PieceRowsCompleteness.lowerPieces_rows_below canonical _
        package.firstFresh othersBelow row lowered) restEnd
  · rw [← encoded]
    funext column
    exact low column.val (lt_of_lt_of_le column.isLt publicBelow)
  · rw [rowsEq]
    intro row member
    rcases List.mem_append.mp member with native | lowered
    · exact R1CS.rowsHold_of_agree_below nativeRows package.firstFresh honest filled nativeBelow
        low nativeHold row native
    · exact loweredRows row lowered
  · intro block member
    apply constraintsHold_of_agree_below honest filled _ package.firstFresh (allBelow block member)
      low
    rw [blocksEq] at member
    rcases List.mem_append.mp member with native | other
    · exact nativeTemplates block native
    · exact restHolds _ (CircuitRelationCompleteness.blocksOf_mem other)

/-- `native_complete` at the slots of the package's rule (`StepPackage.runsOf`), as Rust builds
them: every column that the relation reads has a slot, and the slot is a bit only on a candidate
that a row of the package forces to 0 or 1. Then the assignment is also in the norm bound
(`CircuitRelationNorm.encode_norm_of_forced`), whatever the candidates are. -/
theorem native_normComplete (package : Package) (bits : List Nat)
    (slotsEq : package.slots = runsOf package.publicColumns package.sboxStart package.pieces bits)
    (publicColumnsLe : package.publicColumns ≤ LogicalPlan.publicCount)
    (nativeRows : List R1CS.Row) (nativeBlocks : List Block) (rest : List Piece)
    (piecesEq : package.pieces = PackagePiece.rows nativeRows ::
      ((nativeBlocks.map fun block => PackagePiece.block block.start block.lanes) ++
        packagePieces rest package.firstFresh))
    (frame : Framed package.blocks)
    (nativeBelow : ∀ row ∈ nativeRows, row.VarsBelow package.firstFresh)
    (nativeBlocksBelow : ∀ block ∈ nativeBlocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()),
      expression.VarsBelow package.firstFresh)
    (restValid : ∀ piece ∈ rest, piece.Valid)
    (restBelow : ∀ piece ∈ rest,
      ∀ expression ∈ piece.constraints (PilotData.canonicalConstraints ()),
        expression.VarsBelow package.firstFresh)
    (restEnd : CircuitRelation.sboxStart rest package.firstFresh ≤ package.sboxStart)
    (publicBelow : LogicalPlan.publicCount ≤ package.firstFresh)
    (sboxFits : package.sboxStart + package.blocks.length * 150 ≤ package.width)
    (fits : LogicalPlan.Fits package.relationRows package.width)
    (rowCount_le : package.blocks.length * 150 ≤ 2 ^ cubeVariables)
    (honest : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) honest = encHash digest)
    (nativeHold : R1CS.RowsHold honest nativeRows)
    (nativeTemplates : ∀ block ∈ nativeBlocks, ConstraintsHold honest
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ())))
    (restHolds : ∀ piece ∈ rest,
      ConstraintsHold honest (piece.constraints (PilotData.canonicalConstraints ()))) :
    (LogicalPlan.plan package.relationRows package.width fits
        (layout (runSlots package.width package.slots))).RowsZero
        (packageAssignment package (PieceRowsCompleteness.fillPieces
          (PilotData.canonicalConstraints ()) honest (CircuitRelation.otherPieces rest)
          package.firstFresh)) ∧
      (PoseidonSboxFamilyPlan.plan (CircuitRelation.sboxInterface
          (layout (runSlots package.width package.slots)) package.blocks package.sboxStart)
        rowCount_le).RowsZero (packageAssignment package (PieceRowsCompleteness.fillPieces
          (PilotData.canonicalConstraints ()) honest (CircuitRelation.otherPieces rest)
          package.firstFresh)) ∧
      LogicalPlan.publicInput (publicFits := publicFits) (LogicalPlan.sourceEnv
        (layout (runSlots package.width package.slots)) (packageAssignment package
          (PieceRowsCompleteness.fillPieces (PilotData.canonicalConstraints ()) honest
            (CircuitRelation.otherPieces rest) package.firstFresh))) = encHash digest ∧
      ∀ column, centeredMagnitude (packageAssignment package (PieceRowsCompleteness.fillPieces
        (PilotData.canonicalConstraints ()) honest (CircuitRelation.otherPieces rest)
        package.firstFresh) column) < 2 := by
  let canonical := PilotData.canonicalConstraints ()
  let filled := PieceRowsCompleteness.fillPieces canonical honest (CircuitRelation.otherPieces rest)
    package.firstFresh
  have covered : ReadsCovered (runSlots package.width package.slots)
      (package.pieces.flatMap PackagePiece.relationRows) package.blocks package.sboxStart := by
    rw [slotsEq]
    exact readsCovered_runsOf _ _ _ bits _ publicColumnsLe sboxFits
  obtain ⟨rowsZero, sboxRows, publicEq⟩ := native_complete package nativeRows nativeBlocks rest
    piecesEq frame nativeBelow nativeBlocksBelow restValid restBelow restEnd publicBelow sboxFits
    covered fits rowCount_le honest digest encoded nativeHold nativeTemplates restHolds
  refine ⟨rowsZero, sboxRows, publicEq, ?_⟩
  have nextLe : package.firstFresh ≤ package.sboxStart :=
    le_trans (PieceRowsCompleteness.lowerPiecesAux_le canonical (CircuitRelation.otherPieces rest)
      package.firstFresh []) restEnd
  have low : ∀ column, column < package.firstFresh →
      sboxFill package.blocks package.sboxStart filled column = honest column :=
    fun column below => (sboxFill_outside (Or.inl (by omega))).trans
      (PieceRowsCompleteness.fillPieces_outside canonical _ honest package.firstFresh
        (Or.inl below))
  have positive : 0 < LogicalPlan.publicCount := by decide
  intro column
  apply CircuitRelationNorm.encode_norm_of_forced package.relationRows fits
    (runSlots package.width package.slots) (sboxFill package.blocks package.sboxStart filled)
    rowsZero
  · rw [low 0 (by omega)]
    exact CircuitRelationCompleteness.one_of_encoded (publicFits := publicFits) honest digest
      encoded
  · intro column isPublic kind
    rw [low column (by omega)]
    exact CircuitRelationNorm.encoded_valid (publicFits := publicFits) encoded isPublic kind
  · intro slot member nonField
    rw [slotsEq] at member
    rcases runsOf_kind _ _ _ bits _ member with field | ⟨bit, _, piece, pieceMember, forced⟩
    · exact absurd field nonField
    · obtain ⟨row, rowMember, found⟩ := forced_row piece forced
      exact Or.inr ⟨bit, row,
        List.mem_append_left _ (List.mem_flatMap.mpr ⟨piece, pieceMember, rowMember⟩), found⟩

end NightstreamFPrime.Export.OptionB.StepPackageCompleteness
