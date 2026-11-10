import NightstreamFPrime.Export.OptionB.StepPackage
import NightstreamFPrime.Export.Generic.PieceRowsCompleteness
import NightstreamFPrime.Export.Generic.RowReads

/-! The low-norm slots that a step package lists (CW-10.2, SPEC.md section 4.4).
`StepPackage.runsOf` gives each column a code (`StepPackage.slotCode`): 3, no slot, when no piece
reads it and it is no S-box column; 0, a bit, when it is a bit candidate that a row forces to 0
or 1; 2, a field element, otherwise (`codeOf`). It then folds the columns from the public columns
to the last S-box column into runs. A column has a slot exactly when its code is not 3, and the
slot has the kind `SlotLayout.kindOf` of its code (`mem_runSlots_fold`). The proof expands the runs
into their columns (`expand`): each fold step adds its column at the end. So each column that a
piece reads and each S-box column has a slot (`runsOf_covers`), and each slot is a field slot or a
bit slot on a column that a row of a piece forces (`runsOf_kind`). The rows of a package piece read
only the columns that the piece lists (`relationRows_reads`). For a circuit's package pieces these
lie below the first S-box column when every piece reads below the first fresh column
(`reads_below`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.SlotRuns

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.OptionB.StepPackage

/-! ### Marks -/

theorem markColumns_size (mask : Array Bool) (columns : List Nat) :
    (SlotLayout.markColumns mask columns).size = mask.size := by
  induction columns generalizing mask with
  | nil => rfl
  | cons column rest inductionHypothesis =>
      show (SlotLayout.markColumns (mask.set! column true) rest).size = mask.size
      rw [inductionHypothesis]
      simp [Array.set!]

theorem set!_getD (mask : Array Bool) (index column : Nat) :
    (mask.set! index true).getD column false = true ↔
      mask.getD column false = true ∨ (column < mask.size ∧ column = index) := by
  simp only [Array.getD_eq_getD_getElem?, Array.set!, Array.getElem?_setIfInBounds]
  by_cases same : index = column
  · subst same
    by_cases inside : index < mask.size
    · simp [inside]
    · simp [inside]
  · simp only [same, ↓reduceIte]
    constructor
    · exact Or.inl
    · rintro (marked | ⟨_, equal⟩)
      · exact marked
      · exact absurd equal.symm same

/-- A column is marked when the mask had it or `columns` names it below the mask's size. -/
theorem markColumns_getD (mask : Array Bool) (columns : List Nat) (column : Nat) :
    (SlotLayout.markColumns mask columns).getD column false = true ↔
      mask.getD column false = true ∨ (column < mask.size ∧ column ∈ columns) := by
  induction columns generalizing mask with
  | nil => simp [SlotLayout.markColumns]
  | cons first rest inductionHypothesis =>
      show (SlotLayout.markColumns (mask.set! first true) rest).getD column false = true ↔ _
      rw [inductionHypothesis, set!_getD]
      have sizeEq : (mask.set! first true).size = mask.size := by simp [Array.set!]
      rw [sizeEq]
      simp only [List.mem_cons]
      constructor
      · rintro ((marked | ⟨inside, rfl⟩) | ⟨inside, member⟩)
        · exact Or.inl marked
        · exact Or.inr ⟨inside, Or.inl rfl⟩
        · exact Or.inr ⟨inside, Or.inr member⟩
      · rintro (marked | ⟨inside, rfl | member⟩)
        · exact Or.inl (Or.inl marked)
        · exact Or.inl (Or.inr ⟨inside, rfl⟩)
        · exact Or.inr ⟨inside, member⟩

/-- The marks of a fold over `items`, each marking the columns `columnsOf item`. -/
theorem markAll_getD {α : Type} (columnsOf : α → List Nat) (items : List α) (mask : Array Bool)
    (column : Nat) :
    (items.foldl (fun mask item => SlotLayout.markColumns mask (columnsOf item)) mask).getD column
        false = true ↔
      mask.getD column false = true ∨
        (column < mask.size ∧ ∃ item ∈ items, column ∈ columnsOf item) := by
  induction items generalizing mask with
  | nil => simp
  | cons item rest inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis, markColumns_getD, markColumns_size]
      constructor
      · rintro ((marked | ⟨inside, member⟩) | ⟨inside, other, otherMember, member⟩)
        · exact Or.inl marked
        · exact Or.inr ⟨inside, item, List.mem_cons_self, member⟩
        · exact Or.inr ⟨inside, other, List.mem_cons_of_mem _ otherMember, member⟩
      · rintro (marked | ⟨inside, other, otherMember, member⟩)
        · exact Or.inl (Or.inl marked)
        · rcases List.mem_cons.mp otherMember with rfl | otherMember
          · exact Or.inl (Or.inr ⟨inside, member⟩)
          · exact Or.inr ⟨inside, other, otherMember, member⟩

theorem replicate_getD (size column : Nat) :
    (Array.replicate size false).getD column false = false := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  split <;> rfl

/-! ### Expansion -/

/-- Each column of the runs with its code, in order. -/
def expand (runs : List (Nat × Nat × Nat)) : List (Nat × Nat) :=
  runs.flatMap fun run => (List.range run.2.1).map fun offset => (run.1 + offset, run.2.2)

/-- The column with its code, if the column has a slot. -/
def entryOf (code : Nat → Nat) (column : Nat) : Option (Nat × Nat) :=
  if code column = 3 then none else some (column, code column)

/-- The slot of a column with its code, if the column is below `width`. -/
def slotOf (width : Nat) (entry : Nat × Nat) : Option (LowNormAssignment.Slot width) :=
  if bound : entry.1 < width then some ⟨⟨entry.1, bound⟩, SlotLayout.kindOf entry.2⟩
  else none

theorem expand_nil : expand [] = [] := rfl

theorem expand_append (front back : List (Nat × Nat × Nat)) :
    expand (front ++ back) = expand front ++ expand back := by
  simp only [expand, List.flatMap_append]

theorem expand_single (start count kind : Nat) :
    expand [(start, count, kind)] =
      (List.range count).map fun offset => (start + offset, kind) := by
  simp only [expand, List.flatMap_cons, List.flatMap_nil, List.append_nil]

theorem expand_step (code : Nat → Nat) (runs : List (Nat × Nat × Nat)) (column : Nat)
    (live : code column ≠ 3) :
    expand (runStep code runs column).reverse =
      expand runs.reverse ++ [(column, code column)] := by
  cases runs with
  | nil =>
      simp only [runStep, live, ↓reduceIte, List.reverse_cons, List.reverse_nil, List.nil_append,
        expand_single, List.range_one, List.map_cons, List.map_nil, Nat.add_zero, expand_nil]
  | cons run rest =>
      obtain ⟨start, count, kind⟩ := run
      by_cases joins : kind = code column ∧ start + count = column
      · obtain ⟨sameKind, adjacent⟩ := joins
        simp only [runStep, live, ↓reduceIte, sameKind, adjacent, and_self,
          List.reverse_cons, expand_append, expand_single, List.range_succ, List.map_append,
          List.map_cons, List.map_nil, List.append_assoc]
      · have pushed : runStep code ((start, count, kind) :: rest) column =
            (column, 1, code column) :: (start, count, kind) :: rest := by
          simp only [runStep, live, ↓reduceIte, joins]
        rw [pushed, List.reverse_cons (a := (column, 1, code column)), expand_append,
          expand_single, List.range_one, List.map_cons, List.map_nil, Nat.add_zero]

theorem expand_foldl (code : Nat → Nat) (columns : List Nat) (runs : List (Nat × Nat × Nat)) :
    expand (columns.foldl (runStep code) runs).reverse =
      expand runs.reverse ++ columns.filterMap (entryOf code) := by
  induction columns generalizing runs with
  | nil => simp only [List.foldl_nil, List.filterMap_nil, List.append_nil]
  | cons column rest earlier =>
      rw [List.foldl_cons, earlier]
      by_cases dead : code column = 3
      · simp only [runStep, dead, ↓reduceIte, List.filterMap_cons, entryOf]
      · simp only [expand_step code runs column dead, List.filterMap_cons, entryOf, dead,
          ↓reduceIte, List.append_assoc, List.singleton_append]

theorem runSlots_eq (width : Nat) (runs : List (Nat × Nat × Nat)) :
    SlotLayout.runSlots width runs = (expand runs).filterMap (slotOf width) := by
  induction runs with
  | nil => rfl
  | cons run rest earlier =>
      obtain ⟨start, count, kind⟩ := run
      rw [← List.singleton_append, expand_append, List.filterMap_append, ← earlier,
        expand_single, List.filterMap_map]
      rfl

theorem entryOf_eq_some (code : Nat → Nat) (column : Nat) (entry : Nat × Nat) :
    entryOf code column = some entry ↔ code column ≠ 3 ∧ (column, code column) = entry := by
  by_cases dead : code column = 3
  · simp only [entryOf, dead, ↓reduceIte, reduceCtorEq, ne_eq, not_true_eq_false, false_and]
  · simp only [entryOf, dead, ↓reduceIte, Option.some.injEq, ne_eq, not_false_eq_true, true_and]

theorem slotOf_eq_some {width : Nat} (entry : Nat × Nat) (slot : LowNormAssignment.Slot width) :
    slotOf width entry = some slot ↔
      entry.1 = slot.source.val ∧ slot.kind = SlotLayout.kindOf entry.2 := by
  obtain ⟨⟨column, bound⟩, kind⟩ := slot
  by_cases below : entry.1 < width
  · simp only [slotOf, below, ↓reduceDIte, Option.some.injEq, LowNormAssignment.Slot.mk.injEq,
      Fin.mk.injEq]
    exact and_congr Iff.rfl eq_comm
  · simp only [slotOf, below, ↓reduceDIte, reduceCtorEq, false_iff, not_and]
    intro same
    exact absurd (same ▸ bound) below

theorem mem_drop_range (publicColumns size column : Nat) :
    column ∈ (List.range size).drop publicColumns ↔
      publicColumns ≤ column ∧ column < size := by
  rw [List.range_eq_range', List.drop_range', List.mem_range'_1]
  omega

/-- A slot of the runs of the columns from `publicColumns` to `size` is a column there whose code
is not 3, with the kind of its code. -/
theorem mem_runSlots_fold (code : Nat → Nat) (publicColumns size width : Nat)
    (slot : LowNormAssignment.Slot width) :
    slot ∈ SlotLayout.runSlots width
        (((List.range size).drop publicColumns).foldl (runStep code) []).reverse ↔
      publicColumns ≤ slot.source.val ∧ slot.source.val < size ∧ code slot.source.val ≠ 3 ∧
        slot.kind = SlotLayout.kindOf (code slot.source.val) := by
  rw [runSlots_eq, expand_foldl, List.reverse_nil, expand_nil, List.nil_append,
    List.mem_filterMap]
  constructor
  · rintro ⟨entry, inside, found⟩
    obtain ⟨column, member, listed⟩ := List.mem_filterMap.mp inside
    obtain ⟨live, rfl⟩ := (entryOf_eq_some code column entry).mp listed
    obtain ⟨same, sameKind⟩ := (slotOf_eq_some _ slot).mp found
    subst same
    obtain ⟨low, high⟩ := (mem_drop_range publicColumns size _).mp member
    exact ⟨low, high, live, sameKind⟩
  · rintro ⟨low, high, live, sameKind⟩
    refine ⟨(slot.source.val, code slot.source.val), ?_, ?_⟩
    · exact List.mem_filterMap.mpr ⟨slot.source.val, (mem_drop_range _ _ _).mpr ⟨low, high⟩,
        (entryOf_eq_some _ _ _).mpr ⟨live, rfl⟩⟩
    · exact (slotOf_eq_some _ slot).mpr ⟨rfl, sameKind⟩

theorem slotIndex_of_mem {width : Nat} {slots : List (LowNormAssignment.Slot width)}
    {column : Fin width} (member : ∃ slot ∈ slots, slot.source = column) :
    ∃ index, SlotLayout.slotIndex slots column = some index ∧
      (slots.get index).source = column := by
  obtain ⟨slot, inside, source⟩ := member
  obtain ⟨position, found⟩ := List.mem_iff_get.mp inside
  have selected : (slots.get position).source = column := by rw [found]; exact source
  cases search :
      (List.finRange slots.length).find? (fun index => (slots.get index).source = column) with
  | none =>
      rw [List.find?_eq_none] at search
      exact absurd selected (by simpa using search position (List.mem_finRange position))
  | some first =>
      exact ⟨first, search, by simpa using List.find?_some search⟩

/-! ### The slots of `runsOf` -/

/-- The number of columns that `StepPackage.runsOf` scans: up to the last S-box column. -/
def runsSize (sboxStart : Nat) (pieces : List PackagePiece) : Nat :=
  sboxStart + 150 * (pieces.filterMap PackagePiece.block?).length

/-- The code of each column, as `StepPackage.runsOf` computes it. -/
def codeOf (sboxStart : Nat) (pieces : List PackagePiece) (bits : List Nat) : Nat → Nat :=
  let empty := Array.replicate (runsSize sboxStart pieces) false
  slotCode sboxStart (pieces.foldl (fun mask piece => SlotLayout.markColumns mask piece.reads) empty)
    (pieces.foldl (fun mask piece => SlotLayout.markColumns mask piece.outputs) empty)
    (SlotLayout.markColumns empty bits)
    (pieces.foldl (fun mask piece => SlotLayout.markColumns mask piece.forced) empty)

theorem runsOf_eq (publicColumns sboxStart : Nat) (pieces : List PackagePiece) (bits : List Nat) :
    runsOf publicColumns sboxStart pieces bits =
      (((List.range (runsSize sboxStart pieces)).drop publicColumns).foldl
        (runStep (codeOf sboxStart pieces bits)) []).reverse :=
  rfl

/-- A column below `runsSize` that a piece reads and that is no output local of a block, and each
S-box column, has a code other than 3. -/
theorem codeOf_ne {sboxStart : Nat} {pieces : List PackagePiece} {bits : List Nat} {column : Nat}
    (high : column < runsSize sboxStart pieces)
    (read : sboxStart ≤ column ∨ ((∃ piece ∈ pieces, column ∈ piece.reads) ∧
      ∀ piece ∈ pieces, column ∉ piece.outputs)) :
    codeOf sboxStart pieces bits column ≠ 3 := by
  have marked : ((pieces.foldl (fun mask piece => SlotLayout.markColumns mask piece.reads)
      (Array.replicate (runsSize sboxStart pieces) false)).getD column false = true ∧
      (pieces.foldl (fun mask piece => SlotLayout.markColumns mask piece.outputs)
        (Array.replicate (runsSize sboxStart pieces) false)).getD column false = false) ∨
      sboxStart ≤ column := by
    rcases read with sbox | ⟨⟨piece, member, inPiece⟩, noOutput⟩
    · exact Or.inr sbox
    · refine Or.inl ⟨(markAll_getD PackagePiece.reads pieces _ column).mpr
        (Or.inr ⟨by simpa using high, piece, member, inPiece⟩), ?_⟩
      apply Bool.eq_false_iff.mpr
      intro output
      rcases (markAll_getD PackagePiece.outputs pieces _ column).mp output with
        start | ⟨_, other, otherMember, inOther⟩
      · rw [replicate_getD] at start
        exact absurd start (by decide)
      · exact noOutput other otherMember inOther
  simp only [codeOf, slotCode]
  split
  · split <;> decide
  · rename_i outside
    exfalso
    apply outside
    rcases marked with marked | high
    · simp [marked.1, marked.2]
    · simp [high]

/-- A code is 2 or 3, or 0 on a column that a piece forces to 0 or 1. -/
theorem codeOf_cases (sboxStart : Nat) (pieces : List PackagePiece) (bits : List Nat)
    (column : Nat) :
    codeOf sboxStart pieces bits column = 2 ∨ codeOf sboxStart pieces bits column = 3 ∨
      (codeOf sboxStart pieces bits column = 0 ∧ ∃ piece ∈ pieces, column ∈ piece.forced) := by
  simp only [codeOf, slotCode]
  split
  · split
    · rename_i both
      simp only [Bool.and_eq_true] at both
      refine Or.inr (Or.inr ⟨rfl, ?_⟩)
      rcases (markAll_getD PackagePiece.forced pieces _ column).mp both.2 with
        marked | ⟨_, piece, member, forced⟩
      · rw [replicate_getD] at marked
        exact absurd marked (by decide)
      · exact ⟨piece, member, forced⟩
    · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)

/-- Each column from the public columns to the last S-box column that a piece reads and that is
no output local of a block, and each S-box column there, has a slot. -/
theorem runsOf_covers {publicColumns sboxStart width column : Nat} {pieces : List PackagePiece}
    {bits : List Nat} (low : publicColumns ≤ column) (high : column < runsSize sboxStart pieces)
    (inside : column < width)
    (read : sboxStart ≤ column ∨ ((∃ piece ∈ pieces, column ∈ piece.reads) ∧
      ∀ piece ∈ pieces, column ∉ piece.outputs)) :
    ∃ slot ∈ SlotLayout.runSlots width (runsOf publicColumns sboxStart pieces bits),
      slot.source.val = column := by
  rw [runsOf_eq]
  exact ⟨⟨⟨column, inside⟩, SlotLayout.kindOf (codeOf sboxStart pieces bits column)⟩,
    (mem_runSlots_fold _ _ _ _ _).mpr ⟨low, high, codeOf_ne high read, rfl⟩, rfl⟩

/-- Each slot is a field slot, or a bit slot on a column that a piece forces to 0 or 1. -/
theorem runsOf_kind {publicColumns sboxStart width : Nat} {pieces : List PackagePiece}
    {bits : List Nat} {slot : LowNormAssignment.Slot width}
    (member : slot ∈ SlotLayout.runSlots width (runsOf publicColumns sboxStart pieces bits)) :
    slot.kind = .field ∨
      (slot.kind = .bit ∧ ∃ piece ∈ pieces, slot.source.val ∈ piece.forced) := by
  rw [runsOf_eq] at member
  obtain ⟨_, _, live, kindEq⟩ := (mem_runSlots_fold _ _ _ _ _).mp member
  rcases codeOf_cases sboxStart pieces bits slot.source.val with
    two | three | ⟨zero, forced⟩
  · rw [kindEq, two]
    exact Or.inl rfl
  · exact absurd three live
  · rw [kindEq, zero]
    exact Or.inr ⟨rfl, forced⟩

/-! ### What the rows of a package piece read -/

/-- The rows of a piece read only the piece's columns (`PackagePiece.reads`). -/
theorem relationRows_reads :
    ∀ (piece : PackagePiece), ∀ row ∈ piece.relationRows,
      ∀ column ∈ SlotLayout.rowColumns row, column ∈ piece.reads
  | .rows chunk, row, member, column, columnMember =>
      List.mem_flatMap.mpr ⟨row, member, columnMember⟩
  | .ring cells start quotient lanes, row, member, column, columnMember => by
      have laneReads : ∀ index, (lanes.getD index 0).VarsSatisfy
          (· ∈ (PackagePiece.ring cells start quotient lanes).reads) := fun index =>
        (RowReads.getD_varsSatisfy lanes index).mono _ fun _ inLanes =>
          List.mem_append_right _ inLanes
      have scope := RowReads.ringRows_varsSatisfy cells start quotient lanes
        (fun lane => laneReads lane.val) (fun _ _ lane => laneReads _) (fun _ _ lane => laneReads _)
        (fun cell inside lane => List.mem_append_left _ (List.mem_append_left _
          (List.mem_range'_1.mpr ⟨by simp [PieceRows.outputColumn],
            by have := PieceRows.slot_lt lane inside; simp only [PieceRows.outputColumn]; omega⟩)))
        (fun cell inside lane => List.mem_append_left _ (List.mem_append_right _
          (List.mem_range'_1.mpr ⟨by simp [PieceRows.quotientColumn],
            RowReads.quotientColumn_lt inside lane⟩)))
        row member
      exact RowReads.rowColumns_of_varsSatisfy scope column columnMember
  | .block .., _, member, _, _ => by simp [PackagePiece.relationRows] at member

/-- A column that a piece forces to 0 or 1 has a row of the piece that forces it. -/
theorem forced_row :
    ∀ (piece : PackagePiece) {column : Nat}, column ∈ piece.forced →
      ∃ row ∈ piece.relationRows, SlotLayout.forcedColumn? row = some column
  | .rows _, _, member => List.mem_filterMap.mp member
  | .ring .., _, member => by simp [PackagePiece.forced] at member
  | .block .., _, member => by simp [PackagePiece.forced] at member

/-! ### The package pieces of a circuit -/

theorem packagePieces_rows (pieces : List Piece) (next : Nat) :
    (packagePieces pieces next).flatMap PackagePiece.relationRows =
      (PieceRows.lowerPieces (PilotData.canonicalConstraints ())
        (CircuitRelation.otherPieces pieces) next).1 := by
  simp only [packagePieces, packageAux_eq, List.reverse_nil, List.nil_append,
    flatMap_packageSpec, PieceRows.lowerPieces, lowerPiecesAux_eq _ rfl]

theorem packagePieces_blocks (pieces : List Piece) (next : Nat) :
    (packagePieces pieces next).filterMap PackagePiece.block? = CircuitRelation.blocksOf pieces := by
  simp only [packagePieces, packageAux_eq, List.reverse_nil, List.nil_append, blocks_packageSpec]

/-- A block of the pieces is a block piece of their package pieces. -/
theorem block_mem {pieces : List Piece} {next : Nat} {block : Frame.Block}
    (member : block ∈ CircuitRelation.blocksOf pieces) :
    PackagePiece.block block.start block.lanes ∈ packagePieces pieces next := by
  rw [← packagePieces_blocks pieces next] at member
  obtain ⟨piece, inside, found⟩ := List.mem_filterMap.mp member
  cases piece with
  | rows => simp [PackagePiece.block?] at found
  | ring => simp [PackagePiece.block?] at found
  | block start lanes =>
      simp only [PackagePiece.block?, Option.some.injEq] at found
      subst found
      exact inside

theorem pieceOf_next_le (piece : Piece) (next : Nat) : next ≤ (pieceOf piece next).2 := by
  cases piece with
  | permutation start lanes => exact le_refl _
  | ringProduct cells start lanes => exact Nat.le_add_right _ _
  | constraint expression =>
      exact PieceRowsCompleteness.lowerPiece_next_le (PilotData.canonicalConstraints ())
        (.constraint expression) next

theorem next_le_freshAfter : ∀ (pieces : List Piece) (next : Nat), next ≤ freshAfter pieces next
  | [], _ => le_refl _
  | piece :: rest, next =>
      le_trans (pieceOf_next_le piece next) (next_le_freshAfter rest _)

theorem lowerSpec_others : ∀ (pieces : List Piece) (next : Nat),
    (lowerSpec (CircuitRelation.otherPieces pieces) next).2 = freshAfter pieces next
  | [], _ => rfl
  | .permutation start lanes :: rest, next => by
      simpa [CircuitRelation.otherPieces, CircuitRelation.isBlock, freshAfter, pieceOf] using
        lowerSpec_others rest next
  | .ringProduct cells start lanes :: rest, next => by
      simpa [CircuitRelation.otherPieces, CircuitRelation.isBlock, lowerSpec, freshAfter, pieceOf,
        PieceRows.lowerPiece] using lowerSpec_others rest (next + ringDegree * cells)
  | .constraint expression :: rest, next => by
      simpa [CircuitRelation.otherPieces, CircuitRelation.isBlock, lowerSpec, freshAfter,
        pieceOf] using lowerSpec_others rest _

/-- The first S-box column is the first column after the fresh columns of the package pieces. -/
theorem sboxStart_eq (pieces : List Piece) (next : Nat) :
    CircuitRelation.sboxStart pieces next = freshAfter pieces next := by
  rw [← lowerSpec_others]
  simp only [CircuitRelation.sboxStart, PieceRows.lowerPieces, lowerPiecesAux_eq _ rfl]

theorem exprColumns_below {bound : Nat} {expression : Expr}
    (below : PieceRowsCompleteness.exprBelow bound expression = true) :
    ∀ column ∈ SlotLayout.exprColumns expression, column < bound :=
  RowReads.exprColumns_of_varsSatisfy expression
    ((Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr
      ((PieceRowsCompleteness.exprBelow_iff bound expression).mp below))

theorem packageSpec_reads (bound : Nat) :
    ∀ (pieces : List Piece) (next : Nat), bound ≤ next →
      (∀ piece ∈ pieces, PieceRowsCompleteness.pieceBelow bound piece = true) →
      ∀ piece ∈ packageSpec pieces next, ∀ column ∈ piece.reads, column < freshAfter pieces next
  | [], _, _, _ => by simp [packageSpec]
  | piece :: rest, next, boundLe, below => by
      have nextLe := pieceOf_next_le piece next
      have restLe := next_le_freshAfter rest (pieceOf piece next).2
      intro current member column read
      simp only [packageSpec, List.mem_cons] at member
      rcases member with rfl | member
      · have headBelow := below piece List.mem_cons_self
        cases piece with
        | permutation start lanes =>
            simp only [PieceRowsCompleteness.pieceBelow, Bool.and_eq_true,
              decide_eq_true_eq] at headBelow
            simp only [pieceOf, PackagePiece.reads, List.mem_flatMap] at read
            simp only [freshAfter, pieceOf] at restLe ⊢
            obtain ⟨lane, laneMember, columnMember⟩ := read
            have := exprColumns_below (List.all_eq_true.mp headBelow.1 lane laneMember) column
              columnMember
            omega
        | ringProduct cells start lanes =>
            simp only [PieceRowsCompleteness.pieceBelow, Bool.and_eq_true,
              decide_eq_true_eq] at headBelow
            simp only [pieceOf, PackagePiece.reads, List.mem_append, List.mem_range'_1,
              List.mem_flatMap] at read
            simp only [freshAfter, pieceOf] at restLe ⊢
            rcases read with (⟨_, high⟩ | ⟨_, high⟩) | ⟨lane, laneMember, columnMember⟩
            · omega
            · omega
            · have := exprColumns_below (List.all_eq_true.mp headBelow.1 lane laneMember) column
                columnMember
              omega
        | constraint expression =>
            have scope : ∀ other ∈ [expression], other.VarsBelow next := by
              intro other inside
              rw [List.mem_singleton.mp inside]
              exact Expr.VarsBelow.mono _
                ((PieceRowsCompleteness.exprBelow_iff bound expression).mp headBelow) boundLe
            simp only [pieceOf, PackagePiece.reads, List.mem_flatMap] at read
            obtain ⟨row, rowMember, columnMember⟩ := read
            have rowBelow := R1CS.lowerConstraints_rows_varsBelow [expression] next scope row
              rowMember
            have := RowReads.rowColumns_of_varsSatisfy (allowed := (· < next +
              R1CS.totalFreshCount [expression])) rowBelow column columnMember
            simp only [freshAfter, pieceOf, PieceRows.lowerPiece, Piece.constraints,
              R1CS.lowerConstraints_next] at restLe ⊢
            omega
      · exact packageSpec_reads bound rest (pieceOf piece next).2 (le_trans boundLe nextLe)
          (fun other inside => below other (List.mem_cons_of_mem _ inside)) current member column
          read

/-- The columns that the package pieces of `pieces` read lie below the first S-box column, when
every piece reads only columns below the first fresh column. -/
theorem reads_below {pieces : List Piece} {next : Nat}
    (below : ∀ piece ∈ pieces, PieceRowsCompleteness.pieceBelow next piece = true) :
    ∀ piece ∈ packagePieces pieces next, ∀ column ∈ piece.reads,
      column < CircuitRelation.sboxStart pieces next := by
  intro piece member column read
  rw [sboxStart_eq]
  simp only [packagePieces, packageAux_eq, List.reverse_nil, List.nil_append] at member
  exact packageSpec_reads next pieces next (le_refl _) below piece member column read

end NightstreamFPrime.Export.OptionB.SlotRuns
