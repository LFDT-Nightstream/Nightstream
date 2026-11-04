import NightstreamFPrime.Export.Generic.SlotLayout
import NightstreamFPrime.Layout.R1CS.Support

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

/-! ### Slotted columns -/

/-- `column` is a public column or has a slot, so the encoding decodes to the environment there
(`sourceEnv_encode`). -/
def Slotted (slots : List (LowNormAssignment.Slot width)) (column : Nat) : Prop :=
  column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column

theorem Slotted.lt_width {slots : List (LowNormAssignment.Slot width)} {column : Nat}
    (inside : LogicalPlan.publicCount ≤ width) (slotted : Slotted slots column) :
    column < width := by
  rcases slotted with isPublic | ⟨slot, _, rfl⟩
  · omega
  · exact slot.source.isLt

theorem Slotted.mono {slots others : List (LowNormAssignment.Slot width)}
    (included : ∀ slot ∈ slots, slot ∈ others) {column : Nat} (slotted : Slotted slots column) :
    Slotted others column := by
  rcases slotted with isPublic | ⟨slot, member, source⟩
  · exact Or.inl isPublic
  · exact Or.inr ⟨slot, included slot member, source⟩

/-! ### The columns that the rows read -/

theorem exprColumns_varsSatisfy :
    ∀ expression : Expr, expression.VarsSatisfy (· ∈ exprColumns expression)
  | .var column => List.mem_singleton_self column
  | .const _ => trivial
  | .add left right =>
      ⟨(exprColumns_varsSatisfy left).mono _ fun _ member => List.mem_append_left _ member,
        (exprColumns_varsSatisfy right).mono _ fun _ member => List.mem_append_right _ member⟩
  | .mul left right =>
      ⟨(exprColumns_varsSatisfy left).mono _ fun _ member => List.mem_append_left _ member,
        (exprColumns_varsSatisfy right).mono _ fun _ member => List.mem_append_right _ member⟩

theorem rowColumns_varsSatisfy (row : R1CS.Row) : row.VarsSatisfy (· ∈ rowColumns row) := by
  refine ⟨fun term member => ?_, fun term member => ?_, fun term member => ?_⟩ <;>
    exact List.mem_map_of_mem (by simp [member])

theorem rowColumns_of_varsSatisfy {row : R1CS.Row} {allowed : Nat → Prop}
    (scope : row.VarsSatisfy allowed) : ∀ column ∈ rowColumns row, allowed column := by
  intro column member
  obtain ⟨term, termMember, rfl⟩ := List.mem_map.mp member
  rcases List.mem_append.mp termMember with inFirst | inC
  · rcases List.mem_append.mp inFirst with inA | inB
    · exact scope.1 term inA
    · exact scope.2.1 term inB
  · exact scope.2.2 term inC

/-! ### Marked columns -/

theorem markColumns_size (mask : Array Bool) (columns : List Nat) :
    (markColumns mask columns).size = mask.size := by
  induction columns generalizing mask with
  | nil => rfl
  | cons column rest inductionHypothesis =>
      show (markColumns (mask.set! column true) rest).size = mask.size
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
    (markColumns mask columns).getD column false = true ↔
      mask.getD column false = true ∨ (column < mask.size ∧ column ∈ columns) := by
  induction columns generalizing mask with
  | nil => simp [markColumns]
  | cons first rest inductionHypothesis =>
      show (markColumns (mask.set! first true) rest).getD column false = true ↔ _
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
    (items.foldl (fun mask item => markColumns mask (columnsOf item)) mask).getD column false =
        true ↔
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

theorem replicate_getD (size column : Nat) : (Array.replicate size false).getD column false = false := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  split <;> rfl

end NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
