import NightstreamFPrime.Layout.ProductionRelation.PoseidonTemplatePlan
import NightstreamFPrime.Layout.ProductionRelation.RetainedSlot

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev SourceWidth := PoseidonScheduleTrace.sourceColumnCount

def rows : List (PoseidonSourceRows.SboxSource SourceWidth) :=
  PoseidonTemplatePlan.plan.flatMap fun step => step.sboxes

@[simp] theorem rows_length : rows.length = 150 := by
  rfl

def slots : List (LowNormAssignment.Slot SourceWidth) :=
  rows.map fun row =>
    { source := row.step.output
      kind := .field }

@[simp] theorem slots_length : slots.length = 150 := by
  rfl

@[simp] theorem slots_logicalWidth :
    LowNormAssignment.logicalWidth slots = 6150 := by
  have widths : slots.map LowNormAssignment.Slot.width =
      List.replicate rows.length 41 := by
    simp [slots, Function.comp_def, LowNormAssignment.Slot.width,
      List.map_const']
    rfl
  simp [LowNormAssignment.logicalWidth, widths, rows_length]

def finalRow (lane : Fin 16) : Fin rows.length :=
  ⟨134 + lane.val, by
    rw [rows_length]
    omega⟩

@[simp] theorem finalRow_val (lane : Fin 16) :
    (finalRow lane).val = 134 + lane.val := by
  rfl

def slotIndex (row : Fin rows.length) : Fin slots.length :=
  ⟨row.val, by simpa [slots] using row.isLt⟩

@[simp] theorem slot_source (row : Fin rows.length) :
    (slots.get (slotIndex row)).source = (rows.get row).step.output := by
  simp [slots, slotIndex]

@[simp] theorem slot_kind (row : Fin rows.length) :
    (slots.get (slotIndex row)).kind = .field := by
  simp [slots, slotIndex]

private theorem outputs_after_inputs :
    rows.all (fun row =>
      decide (PoseidonScheduleTrace.inputCount ≤ row.step.output.val)) =
        true := by
  decide +kernel

theorem output_local_bounds (row : Fin rows.length) :
    PoseidonScheduleTrace.inputCount ≤ (rows.get row).step.output.val ∧
      (rows.get row).step.output.val <
        PoseidonScheduleTrace.sourceColumnCount := by
  constructor
  · have all := outputs_after_inputs
    rw [List.all_eq_true] at all
    simpa using all (rows.get row) (List.get_mem rows row)
  · exact (rows.get row).step.output.isLt

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

def form (row : Fin rows.length) :
    SparseForm (ProductionAssignment.logicalWidth slots) :=
  RetainedSlot.form slots (slotIndex row)

theorem form_eval
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (source : Fin SourceWidth → F) (row : Fin rows.length) :
    (form row).eval
        (ProductionAssignment.logicalAssignment publicInput slots source) =
      source (rows.get row).step.output := by
  rw [form, RetainedSlot.form_eval, slot_source]

theorem slots_valid (source : Fin SourceWidth → F) :
    ∀ slot ∈ slots, slot.Valid source := by
  intro slot member
  rcases List.mem_map.mp member with ⟨row, _, rfl⟩
  trivial

end NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots
