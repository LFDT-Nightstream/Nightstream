import NightstreamFPrime.Export.OptionB.StepPackage

/-! The low-norm slots that a step package lists (CW-10.2). `StepPackage.slotRuns` gives each
source column a code: 2 by default, 3 at each block's dead locals, and 0 at each column that a
`bit` hint fills (`slotCodes`). It then folds the columns from the public columns to `slotSize`
into runs (`runsOf`). `slotRuns_eq` splits `slotRuns` into these two parts. A column with code 3
has no slot, and every other column has one slot of kind `SlotLayout.kindOf` of its code.
`mem_runSlots_runsOf` states this for any codes. The proof expands the runs into their columns
(`expand`): each fold step adds its column at the end. `slotIndex_of_mem` finds the slot of a
column that has one. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.SlotRuns

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces

/-- The code of each source column, as `StepPackage.slotRuns` computes it. -/
def slotCodes (pieces : List Piece) (next : Nat) (batches : List WitnessBatch) : Array Nat :=
  let blocks := CircuitRelation.blocksOf pieces
  let size := CircuitRelation.sboxStart pieces next + 150 * blocks.length
  let codes := blocks.foldl (fun codes block =>
      (List.range 1080).foldl (fun codes offset => codes.set! (block.start + offset) 3) codes)
    (Array.replicate size (2 : Nat))
  batches.foldl (fun codes batch =>
      (batch.hints.zipIdx).foldl (fun codes (hint, index) => match hint with
        | .bit .. => codes.set! (batch.start + batch.recipes.length + index) 0
        | _ => codes) codes) codes

/-- The number of source columns that `StepPackage.slotRuns` scans. -/
def slotSize (pieces : List Piece) (next : Nat) : Nat :=
  CircuitRelation.sboxStart pieces next + 150 * (CircuitRelation.blocksOf pieces).length

/-- One fold step of `StepPackage.slotRuns`. The runs are in reverse order. -/
def step (codes : Array Nat) (runs : List (Nat × Nat × Nat)) (column : Nat) :
    List (Nat × Nat × Nat) :=
  let code : Nat := codes.getD column 3
  if code = 3 then runs else
    match runs with
    | (start, count, kind) :: rest =>
        if kind = code ∧ start + count = column then (start, count + 1, kind) :: rest
        else (column, 1, code) :: runs
    | [] => [(column, 1, code)]

/-- The runs `(start, count, kind)` of the columns from `publicColumns` to `size`. -/
def runsOf (codes : Array Nat) (publicColumns size : Nat) : List (Nat × Nat × Nat) :=
  (((List.range size).drop publicColumns).foldl (step codes) []).reverse

theorem slotRuns_eq (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) :
    StepPackage.slotRuns publicColumns pieces next batches =
      runsOf (slotCodes pieces next batches) publicColumns (slotSize pieces next) :=
  rfl

/-! ### Expansion -/

/-- Each column of the runs with its code, in order. -/
def expand (runs : List (Nat × Nat × Nat)) : List (Nat × Nat) :=
  runs.flatMap fun run => (List.range run.2.1).map fun offset => (run.1 + offset, run.2.2)

/-- The column with its code, if the column has a slot. -/
def entryOf (codes : Array Nat) (column : Nat) : Option (Nat × Nat) :=
  if codes.getD column 3 = 3 then none else some (column, codes.getD column 3)

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

theorem expand_step (codes : Array Nat) (runs : List (Nat × Nat × Nat)) (column : Nat)
    (live : codes.getD column 3 ≠ 3) :
    expand (step codes runs column).reverse =
      expand runs.reverse ++ [(column, codes.getD column 3)] := by
  cases runs with
  | nil =>
      simp only [step, live, ↓reduceIte, List.reverse_cons, List.reverse_nil, List.nil_append,
        expand_single, List.range_one, List.map_cons, List.map_nil, Nat.add_zero, expand_nil]
  | cons run rest =>
      obtain ⟨start, count, kind⟩ := run
      by_cases joins : kind = codes.getD column 3 ∧ start + count = column
      · obtain ⟨sameKind, adjacent⟩ := joins
        simp only [step, live, ↓reduceIte, sameKind, adjacent, and_self,
          List.reverse_cons, expand_append, expand_single, List.range_succ, List.map_append,
          List.map_cons, List.map_nil, List.append_assoc]
      · have pushed : step codes ((start, count, kind) :: rest) column =
            (column, 1, codes.getD column 3) :: (start, count, kind) :: rest := by
          simp only [step, live, ↓reduceIte, joins]
        rw [pushed, List.reverse_cons (a := (column, 1, codes.getD column 3)), expand_append,
          expand_single, List.range_one, List.map_cons, List.map_nil, Nat.add_zero]

theorem expand_foldl (codes : Array Nat) (columns : List Nat) (runs : List (Nat × Nat × Nat)) :
    expand (columns.foldl (step codes) runs).reverse =
      expand runs.reverse ++ columns.filterMap (entryOf codes) := by
  induction columns generalizing runs with
  | nil => simp only [List.foldl_nil, List.filterMap_nil, List.append_nil]
  | cons column rest earlier =>
      rw [List.foldl_cons, earlier]
      by_cases dead : codes.getD column 3 = 3
      · simp only [step, dead, ↓reduceIte, List.filterMap_cons, entryOf]
      · simp only [expand_step codes runs column dead, List.filterMap_cons, entryOf, dead,
          ↓reduceIte, List.append_assoc, List.singleton_append]

theorem expand_runsOf (codes : Array Nat) (publicColumns size : Nat) :
    expand (runsOf codes publicColumns size) =
      ((List.range size).drop publicColumns).filterMap (entryOf codes) := by
  rw [runsOf, expand_foldl, List.reverse_nil, expand_nil, List.nil_append]

theorem runSlots_eq (width : Nat) (runs : List (Nat × Nat × Nat)) :
    SlotLayout.runSlots width runs = (expand runs).filterMap (slotOf width) := by
  induction runs with
  | nil => rfl
  | cons run rest earlier =>
      obtain ⟨start, count, kind⟩ := run
      rw [← List.singleton_append, expand_append, List.filterMap_append, ← earlier,
        expand_single, List.filterMap_map]
      rfl

theorem entryOf_eq_some (codes : Array Nat) (column : Nat) (entry : Nat × Nat) :
    entryOf codes column = some entry ↔
      codes.getD column 3 ≠ 3 ∧ (column, codes.getD column 3) = entry := by
  by_cases dead : codes.getD column 3 = 3
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

/-! ### Slots -/

theorem mem_runSlots_runsOf (codes : Array Nat) (publicColumns size width : Nat)
    (slot : LowNormAssignment.Slot width) :
    slot ∈ SlotLayout.runSlots width (runsOf codes publicColumns size) ↔
      publicColumns ≤ slot.source.val ∧ slot.source.val < size ∧
        codes.getD slot.source.val 3 ≠ 3 ∧
          slot.kind = SlotLayout.kindOf (codes.getD slot.source.val 3) := by
  rw [runSlots_eq, expand_runsOf, List.mem_filterMap]
  constructor
  · rintro ⟨entry, inside, found⟩
    obtain ⟨column, member, listed⟩ := List.mem_filterMap.mp inside
    obtain ⟨live, rfl⟩ := (entryOf_eq_some codes column entry).mp listed
    obtain ⟨same, sameKind⟩ := (slotOf_eq_some _ slot).mp found
    subst same
    obtain ⟨low, high⟩ := (mem_drop_range publicColumns size _).mp member
    exact ⟨low, high, live, sameKind⟩
  · rintro ⟨low, high, live, sameKind⟩
    refine ⟨(slot.source.val, codes.getD slot.source.val 3), ?_, ?_⟩
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

end NightstreamFPrime.Export.OptionB.SlotRuns
