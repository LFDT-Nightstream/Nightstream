import NightstreamFPrime.Export.Stage1.Wide.PhysicalRelabel

/-! Successful compact-invocation relocation supplies exact input lookups.
The affine checks are read from the emitter's existing checked loop. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalCompactRelabel

open NightstreamFPrime.Export.Package
open PhysicalRelabel

private theorem checked_indices (mapping : Map) (before : CompactInputRange)
    (first : Nat) (indices : List Nat)
    (checked : (forIn indices () fun index _ => do
      let mapped ← mapping.column (before.columnStart + index * before.columnStride)
      unless mapped = first + index * before.columnStride do
        throw "non-affine compact input relocation"
      pure (ForInStep.yield ()) : Except String Unit) = .ok ()) :
    ∀ index ∈ indices, mapping.column (before.columnStart + index * before.columnStride) =
      .ok (first + index * before.columnStride) := by
  induction indices with
  | nil => intro index member; cases member
  | cons index rest ih =>
    rw [List.forIn_cons] at checked
    cases read : mapping.column (before.columnStart + index * before.columnStride) with
    | error message => simp [read, Bind.bind, Except.bind] at checked
    | ok target =>
      by_cases same : target = first + index * before.columnStride
      · simp [read, same, Bind.bind, Except.bind] at checked
        intro current member
        rcases List.mem_cons.mp member with rfl | member
        · simpa only [same] using read
        · exact ih checked current member
      · simp [read, same, Bind.bind, Except.bind] at checked

private theorem range_parts (before after : CompactInputRange)
    (start : Except String Nat) (check : Nat → Except String Unit)
    (emitted : (do
      let first ← start
      let _ ← check first
      return { before with columnStart := first } : Except String CompactInputRange) = .ok after) :
    ∃ first, start = .ok first ∧ check first = .ok () ∧ after = { before with columnStart := first } := by
  cases start with
  | error message => simp [Bind.bind, Except.bind] at emitted
  | ok first =>
    cases tested : check first with
    | error message => simp [tested, Bind.bind, Except.bind] at emitted
    | ok unit =>
      cases unit
      simp only [tested, Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq] at emitted
      exact ⟨first, rfl, tested, emitted.symm⟩

theorem inputRange_facts (mapping : Map) (before after : CompactInputRange)
    (emitted : mapping.inputRange before = .ok after) :
    after.inputStart = before.inputStart ∧ after.inputCount = before.inputCount ∧
      after.columnStride = before.columnStride ∧
      mapping.column before.columnStart = .ok after.columnStart ∧
      ∀ index, index < before.inputCount →
        mapping.column (before.columnStart + index * before.columnStride) =
          .ok (after.columnStart + index * after.columnStride) := by
  obtain ⟨first, start, checked, rfl⟩ := range_parts before after (mapping.column before.columnStart)
    (fun first => forIn (List.range before.inputCount) () fun index _ => do
      let mapped ← mapping.column (before.columnStart + index * before.columnStride)
      unless mapped = first + index * before.columnStride do
        throw "non-affine compact input relocation"
      pure (ForInStep.yield ())) emitted
  refine ⟨rfl, rfl, rfl, start, ?_⟩
  intro index bounded
  exact checked_indices mapping before first _ checked index (List.mem_range.mpr bounded)

private theorem lookup_cons (head : CompactInputRange) (rest : List CompactInputRange) (input : Nat) :
    compactInputColumn (head :: rest) input =
      if head.inputStart ≤ input ∧ input < head.inputStart + head.inputCount then
        head.columnStart + (input - head.inputStart) * head.columnStride
      else compactInputColumn rest input := by
  by_cases inside : head.inputStart ≤ input ∧ input < head.inputStart + head.inputCount
  · simp [compactInputColumn, List.find?_cons, inside, inside.1, inside.2]
  · have rejected : ¬head.inputStart ≤ input ∨ ¬input < head.inputStart + head.inputCount :=
      not_and_or.mp inside
    rcases rejected with earlier | later
    · simp [compactInputColumn, List.find?_cons, inside, earlier]
    · simp [compactInputColumn, List.find?_cons, inside, later]

theorem inputRanges_lookup (mapping : Map) (zero : mapping.column 0 = .ok 0)
    (before after : List CompactInputRange)
    (pairs : List.Forall₂ (fun original moved => mapping.inputRange original = .ok moved) before after)
    (input : Nat) : mapping.column (compactInputColumn before input) = .ok (compactInputColumn after input) := by
  induction pairs with
  | nil => exact zero
  | @cons original moved before after emitted pairs ih =>
    obtain ⟨start, count, stride, _, reads⟩ := inputRange_facts mapping original moved emitted
    rw [lookup_cons, lookup_cons, start, count]
    split
    · rename_i inside
      rw [stride]
      simpa only [stride] using reads (input - original.inputStart) (by omega)
    · exact ih

theorem compact_facts (mapping : Map) (zero : mapping.column 0 = .ok 0)
    (before after : CompactRowInvocation) (emitted : mapping.compact before = .ok after) :
    after.templateIndex = before.templateIndex ∧
      mapping.column before.localStart = .ok after.localStart ∧
      ∀ input, mapping.column (compactInputColumn before.inputRanges input) =
        .ok (compactInputColumn after.inputRanges input) := by
  unfold Map.compact at emitted
  cases row : mapping.row before.rowStart with
  | error message => simp [row, Bind.bind, Except.bind] at emitted
  | ok rowStart =>
    cases localResult : mapping.column before.localStart with
    | error message => simp [row, localResult, Bind.bind, Except.bind] at emitted
    | ok localStart =>
      cases ranges : before.inputRanges.mapM mapping.inputRange with
      | error message => simp [row, localResult, ranges, Bind.bind, Except.bind] at emitted
      | ok inputRanges =>
        simp only [row, localResult, ranges, Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq] at emitted
        subst after
        refine ⟨rfl, rfl, ?_⟩
        exact inputRanges_lookup mapping zero before.inputRanges inputRanges
          (mapM_pairs _ _ _ ranges)

end NightstreamFPrime.Export.Stage1.Wide.PhysicalCompactRelabel
