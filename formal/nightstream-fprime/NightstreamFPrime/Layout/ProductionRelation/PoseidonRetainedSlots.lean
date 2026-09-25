import NightstreamFPrime.Layout.ProductionRelation.PoseidonTemplatePlan
import NightstreamFPrime.Layout.ProductionRelation.RetainedSlot

/-!
Owns the exact retained low-norm slots for one canonical Poseidon2 template.
The compiler keeps one general-field slot for each of the 86 S-box outputs,
in the same order as the direct template rows. Linear trace values are not
retained by this module.

This is a fixed-size template object. It does not expand package invocations
or select the remaining Stage 1 source slots.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev SourceWidth := PoseidonScheduleTrace.sourceColumnCount

/-- Canonical S-box rows in schedule-step order and then lane order. -/
def rows : List (PoseidonSourceRows.SboxSource SourceWidth) :=
  PoseidonTemplatePlan.plan.flatMap fun step => step.sboxes

@[simp] theorem rows_length : rows.length = 86 := by
  rfl

/-- One exact balanced-ternary field slot for each retained S-box output. -/
def slots : List (LowNormAssignment.Slot SourceWidth) :=
  rows.map fun row =>
    { source := row.step.output
      kind := .field }

@[simp] theorem slots_length : slots.length = 86 := by
  rfl

/-- One template uses exactly 86 times 41 low-norm coordinates. -/
@[simp] theorem slots_logicalWidth :
    LowNormAssignment.logicalWidth slots = 3526 := by
  rfl

/-- The final terminal full round owns retained rows 78 through 85. -/
def finalRow (lane : Fin 8) : Fin rows.length :=
  ⟨78 + lane.val, by
    rw [rows_length]
    omega⟩

@[simp] theorem finalRow_val (lane : Fin 8) :
    (finalRow lane).val = 78 + lane.val := by
  rfl

/-- The retained slot at one direct-row index has the same canonical index. -/
def slotIndex (row : Fin rows.length) : Fin slots.length :=
  ⟨row.val, by simpa [slots] using row.isLt⟩

@[simp] theorem slot_source (row : Fin rows.length) :
    (slots.get (slotIndex row)).source = (rows.get row).step.output := by
  simp [slots, slotIndex]

@[simp] theorem slot_kind (row : Fin rows.length) :
    (slots.get (slotIndex row)).kind = .field := by
  simp [slots, slotIndex]

/-- Every retained S-box output is a local permutation column, after the
eight caller-input columns. -/
theorem output_local_bounds (row : Fin rows.length) :
    PoseidonScheduleTrace.inputCount ≤ (rows.get row).step.output.val ∧
      (rows.get row).step.output.val <
        PoseidonScheduleTrace.sourceColumnCount := by
  constructor
  · fin_cases row <;> decide
  · exact (rows.get row).step.output.isLt

/-- Exact local-column index of one retained S-box output. -/
def localOutput (row : Fin rows.length) :
    Fin PoseidonScheduleTrace.localColumnCount :=
  ⟨(rows.get row).step.output.val - PoseidonScheduleTrace.inputCount, by
    have bounds := output_local_bounds row
    norm_num [PoseidonScheduleTrace.inputCount,
      PoseidonScheduleTrace.localColumnCount,
      PoseidonScheduleTrace.sourceColumnCount] at bounds ⊢
    omega⟩

@[simp] theorem output_eq_input_add_local (row : Fin rows.length) :
    (rows.get row).step.output.val =
      PoseidonScheduleTrace.inputCount + (localOutput row).val := by
  have bounds := (output_local_bounds row).1
  unfold localOutput
  change
    (rows.get row).step.output.val =
      PoseidonScheduleTrace.inputCount +
        ((rows.get row).step.output.val -
          PoseidonScheduleTrace.inputCount)
  exact (Nat.add_sub_of_le bounds).symm

/-- Sparse reconstruction form of one retained S-box output. -/
def form (row : Fin rows.length) :
    SparseForm (ProductionAssignment.logicalWidth slots) :=
  RetainedSlot.form slots (slotIndex row)

/-- Each retained form reconstructs the exact source column selected by the
corresponding direct S-box row. -/
theorem form_eval
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (source : Fin SourceWidth → F) (row : Fin rows.length) :
    (form row).eval
        (ProductionAssignment.logicalAssignment publicInput slots source) =
      source (rows.get row).step.output := by
  rw [form, RetainedSlot.form_eval, slot_source]

/-- General-field slots are valid for every source assignment because their
balanced-ternary encoding is total and canonical. -/
theorem slots_valid (source : Fin SourceWidth → F) :
    ∀ slot ∈ slots, slot.Valid source := by
  intro slot member
  rcases List.mem_map.mp member with ⟨row, _, rfl⟩
  trivial

end NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots
