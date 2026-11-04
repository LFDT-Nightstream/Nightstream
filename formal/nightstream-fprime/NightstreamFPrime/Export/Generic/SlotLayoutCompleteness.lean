import NightstreamFPrime.Export.Generic.SlotLayout

/-! Completeness of the slot layout (CW-5.11). The low-norm encoding of an environment (`encode`,
today's `ProductionAssignment.logicalAssignment`) decodes through `SlotLayout.layout` to that
environment on every public column and on every column that has a slot, and to 0 on every other
column (`sourceEnv_encode`, `sourceEnv_unslotted`). When the public columns carry norm-bounded
values and each slotted value is `LowNormSlot.Valid` for its slot's kind, every coordinate lies in
`{-1, 0, 1}` (`encode_norm`). `runSlots` gives a slot to every column of its runs below the width
(`runSlots_mem`). -/

namespace NightstreamFPrime.Export.Generic.SlotLayoutCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Export.Generic.SlotLayout

variable {width : Nat}

theorem slotIndex_source {slots : List (LowNormAssignment.Slot width)} {column : Fin width}
    {index : Fin slots.length} (found : slotIndex slots column = some index) :
    (slots.get index).source = column := by
  simpa using List.find?_some found

theorem slotIndex_isSome {slots : List (LowNormAssignment.Slot width)} {column : Fin width}
    {slot : LowNormAssignment.Slot width} (member : slot ∈ slots) (source : slot.source = column) :
    (slotIndex slots column).isSome := by
  obtain ⟨index, rfl⟩ := List.mem_iff_get.mp member
  exact List.find?_isSome.mpr ⟨index, List.mem_finRange index, by simpa using source⟩

theorem slotIndex_none {slots : List (LowNormAssignment.Slot width)} {column : Fin width}
    (unslotted : ∀ slot ∈ slots, slot.source ≠ column) : slotIndex slots column = none := by
  apply List.find?_eq_none.mpr
  intro index _
  simpa using unslotted _ (List.get_mem slots index)

theorem layout_form_slot (slots : List (LowNormAssignment.Slot width)) {column : Fin width}
    (notPublic : ¬column.val < LogicalPlan.publicCount) {index : Fin slots.length}
    (found : slotIndex slots column = some index) :
    (layout slots).map.form column = RetainedSlot.form slots index := by
  simp only [layout, notPublic, ↓reduceDIte, found]

theorem layout_form_none (slots : List (LowNormAssignment.Slot width)) {column : Fin width}
    (notPublic : ¬column.val < LogicalPlan.publicCount) (found : slotIndex slots column = none) :
    (layout slots).map.form column = .empty := by
  simp only [layout, notPublic, ↓reduceDIte, found]

/-- The low-norm encoding of `env`: the public columns carry `env`'s first columns and each slot
the digits of its source column's value. -/
def encode (slots : List (LowNormAssignment.Slot width)) (env : Env) :
    Assignment F (ProductionAssignment.logicalWidth slots) :=
  ProductionAssignment.logicalAssignment (fun column => env column.val) slots
    (fun column => env column.val)

theorem encode_public (slots : List (LowNormAssignment.Slot width)) (env : Env) {column : Nat}
    (isPublic : column < LogicalPlan.publicCount)
    (bound : column < ProductionAssignment.logicalWidth slots) :
    encode slots env ⟨column, bound⟩ = env column := by
  have publicRegion : column < ProductionAssignment.publicWidth := isPublic
  unfold encode ProductionAssignment.logicalAssignment
  rw [dite_eq_left publicRegion]

/-- The encoding's constant column is `env`'s column 0. -/
theorem encode_one (slots : List (LowNormAssignment.Slot width)) (env : Env) :
    encode slots env (layout slots).one = env 0 :=
  encode_public slots env (by decide) _

theorem sourceEnv_encode_public (slots : List (LowNormAssignment.Slot width)) (env : Env)
    {column : Nat} (inside : column < width) (isPublic : column < LogicalPlan.publicCount) :
    LogicalPlan.sourceEnv (layout slots) (encode slots env) column = env column := by
  rw [LogicalPlan.sourceEnv_public (layout slots) _ column isPublic inside,
    encode_public slots env isPublic]

theorem sourceEnv_encode_slot (slots : List (LowNormAssignment.Slot width)) (env : Env)
    {slot : LowNormAssignment.Slot width} (member : slot ∈ slots)
    (notPublic : LogicalPlan.publicCount ≤ slot.source.val) :
    LogicalPlan.sourceEnv (layout slots) (encode slots env) slot.source.val =
      env slot.source.val := by
  obtain ⟨index, found⟩ := Option.isSome_iff_exists.mp (slotIndex_isSome member rfl)
  unfold LogicalPlan.sourceEnv
  rw [dite_eq_left slot.source.isLt]
  rw [layout_form_slot slots (Nat.not_lt.mpr notPublic) found, encode, RetainedSlot.form_eval,
    slotIndex_source found]

/-- The encoding decodes to `env` on every public column and every column with a slot. -/
theorem sourceEnv_encode (slots : List (LowNormAssignment.Slot width)) (env : Env)
    {column : Nat} (inside : column < width)
    (covered : column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column) :
    LogicalPlan.sourceEnv (layout slots) (encode slots env) column = env column := by
  by_cases isPublic : column < LogicalPlan.publicCount
  · exact sourceEnv_encode_public slots env inside isPublic
  · rcases covered with isPublic' | ⟨slot, member, rfl⟩
    · exact absurd isPublic' isPublic
    · exact sourceEnv_encode_slot slots env member (Nat.le_of_not_gt isPublic)

/-- A column at or beyond the public columns without a slot decodes to 0. -/
theorem sourceEnv_unslotted (slots : List (LowNormAssignment.Slot width))
    (assignment : Assignment F (ProductionAssignment.logicalWidth slots)) {column : Nat}
    (notPublic : LogicalPlan.publicCount ≤ column)
    (unslotted : ∀ slot ∈ slots, slot.source.val ≠ column) :
    LogicalPlan.sourceEnv (layout slots) assignment column = 0 := by
  unfold LogicalPlan.sourceEnv
  split
  · rename_i inside
    have none : slotIndex slots ⟨column, inside⟩ = none :=
      slotIndex_none fun slot member equal => unslotted slot member (by rw [equal])
    rw [layout_form_none slots (Nat.not_lt.mpr notPublic) none]
    exact SparseForm.empty_eval _
  · rfl

/-- The encoding lies within the norm bound when the public columns carry norm-bounded values and
each slotted value is valid for its slot's kind. -/
theorem encode_norm (slots : List (LowNormAssignment.Slot width)) (env : Env)
    (publicNorm : ∀ column : Fin ProductionAssignment.publicWidth,
      centeredMagnitude (env column.val) < 2)
    (valid : ∀ slot ∈ slots, LowNormSlot.Valid slot.kind (env slot.source.val))
    (column : Fin (ProductionAssignment.logicalWidth slots)) :
    centeredMagnitude (encode slots env column) < 2 :=
  ProductionAssignment.logicalAssignment_norm _ slots (fun column => env column.val) publicNorm
    valid column

/-- Slot-layout completeness: the encoding of an environment whose slotted values are valid lies
within the norm bound and decodes to the environment on every public column and every column that
has a slot. -/
theorem encode_complete (slots : List (LowNormAssignment.Slot width)) (env : Env)
    (publicNorm : ∀ column : Fin ProductionAssignment.publicWidth,
      centeredMagnitude (env column.val) < 2)
    (valid : ∀ slot ∈ slots, LowNormSlot.Valid slot.kind (env slot.source.val)) :
    (∀ column, centeredMagnitude (encode slots env column) < 2) ∧
      ∀ column, column < width →
        (column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column) →
          LogicalPlan.sourceEnv (layout slots) (encode slots env) column = env column :=
  ⟨encode_norm slots env publicNorm valid, fun _ inside covered =>
    sourceEnv_encode slots env inside covered⟩

/-- Each column of a run below the width has a slot of the run's kind. -/
theorem runSlots_mem {runs : List (Nat × Nat × Nat)} {start count kind column : Nat}
    (member : (start, count, kind) ∈ runs) (low : start ≤ column) (high : column < start + count)
    (inside : column < width) :
    (⟨⟨column, inside⟩, kindOf kind⟩ : LowNormAssignment.Slot width) ∈ runSlots width runs := by
  unfold runSlots
  refine List.mem_flatMap.mpr ⟨(start, count, kind), member, ?_⟩
  refine List.mem_filterMap.mpr ⟨column - start, List.mem_range.mpr (by omega), ?_⟩
  have equal : start + (column - start) = column := by omega
  simp only [equal, inside, ↓reduceDIte]

end NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
