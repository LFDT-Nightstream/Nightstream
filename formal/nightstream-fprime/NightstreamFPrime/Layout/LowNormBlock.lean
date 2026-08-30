import NightstreamFPrime.Layout.LowNormAssignment
import NightstreamFPrime.Layout.ProductionRelation.RetainedSlot

/-!
Owns a compact homogeneous block of retained low-norm slots. A block stores
only its kind, count, and source-index function. Its direct coordinate and
sparse-form operations do not construct the expanded slot list.

`expanded` is the proof-oriented reference view. The structural theorems
connect the direct block geometry to that view without evaluating a
production-sized block.
-/

namespace NightstreamFPrime.Layout.LowNormBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout.ProductionRelation

/-- One homogeneous retained-slot block in canonical compiler order. -/
structure Block (sourceWidth : Nat) where
  kind : LowNormSlot.Kind
  slotCount : Nat
  source : Fin slotCount → Fin sourceWidth

namespace Block

/-- Lift a block into a larger source domain without changing its slot or
coordinate order. -/
def lift {sourceWidth largerWidth : Nat} (block : Block sourceWidth)
    (fits : sourceWidth ≤ largerWidth) : Block largerWidth where
  kind := block.kind
  slotCount := block.slotCount
  source := fun slot =>
    ⟨(block.source slot).val,
      Nat.lt_of_lt_of_le (block.source slot).isLt fits⟩

@[simp] theorem lift_kind {sourceWidth largerWidth : Nat}
    (block : Block sourceWidth) (fits : sourceWidth ≤ largerWidth) :
    (block.lift fits).kind = block.kind := by
  rfl

@[simp] theorem lift_slotCount {sourceWidth largerWidth : Nat}
    (block : Block sourceWidth) (fits : sourceWidth ≤ largerWidth) :
    (block.lift fits).slotCount = block.slotCount := by
  rfl

@[simp] theorem lift_source_val {sourceWidth largerWidth : Nat}
    (block : Block sourceWidth) (fits : sourceWidth ≤ largerWidth)
    (slot : Fin block.slotCount) :
    ((block.lift fits).source ⟨slot.val, by simpa using slot.isLt⟩).val =
      (block.source slot).val := by
  rfl

/-- Zero-copy contiguous view of a parent block. The source function remains
opaque and is selected only at the proved parent slot. -/
def slice {sourceWidth : Nat} (block : Block sourceWidth)
    (offset count : Nat) (fits : offset + count ≤ block.slotCount) :
    Block sourceWidth where
  kind := block.kind
  slotCount := count
  source := fun slot =>
    block.source ⟨offset + slot.val, by
      exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left slot.isLt offset) fits⟩

@[simp] theorem slice_kind {sourceWidth : Nat} (block : Block sourceWidth)
    (offset count : Nat) (fits : offset + count ≤ block.slotCount) :
    (block.slice offset count fits).kind = block.kind := by
  rfl

@[simp] theorem slice_slotCount {sourceWidth : Nat}
    (block : Block sourceWidth) (offset count : Nat)
    (fits : offset + count ≤ block.slotCount) :
    (block.slice offset count fits).slotCount = count := by
  rfl

theorem slice_source {sourceWidth : Nat} (block : Block sourceWidth)
    (offset count : Nat) (fits : offset + count ≤ block.slotCount)
    (slot : Fin count) :
    (block.slice offset count fits).source slot =
      block.source ⟨offset + slot.val, by
        exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left slot.isLt offset) fits⟩ := by
  rfl

private theorem sum_ofFn_const (count value : Nat) :
    (List.ofFn fun _ : Fin count => value).sum = count * value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      simp [List.ofFn_succ, inductionHypothesis, Nat.succ_mul, Nat.add_comm]

/-- Exact number of low-norm coordinates owned by the block. -/
def coordinateCount {sourceWidth : Nat} (block : Block sourceWidth) : Nat :=
  block.slotCount * block.kind.width

@[simp] theorem lift_coordinateCount {sourceWidth largerWidth : Nat}
    (block : Block sourceWidth) (fits : sourceWidth ≤ largerWidth) :
    (block.lift fits).coordinateCount = block.coordinateCount := by
  rfl

@[simp] theorem slice_coordinateCount {sourceWidth : Nat}
    (block : Block sourceWidth) (offset count : Nat)
    (fits : offset + count ≤ block.slotCount) :
    (block.slice offset count fits).coordinateCount = count * block.kind.width := by
  rfl

/-- Proof-oriented reference expansion. Production code uses the indexed
block directly. -/
def expanded {sourceWidth : Nat} (block : Block sourceWidth) :
    List (LowNormAssignment.Slot sourceWidth) :=
  List.ofFn fun slot =>
    { source := block.source slot
      kind := block.kind }

@[simp] theorem expanded_length {sourceWidth : Nat}
    (block : Block sourceWidth) :
    block.expanded.length = block.slotCount := by
  simp [expanded]

/-- Expanded reference geometry has exactly the direct block width. -/
theorem expanded_logicalWidth {sourceWidth : Nat}
    (block : Block sourceWidth) :
    LowNormAssignment.logicalWidth block.expanded = block.coordinateCount := by
  unfold LowNormAssignment.logicalWidth expanded
  rw [List.map_ofFn]
  change
    (List.ofFn fun _ : Fin block.slotCount => block.kind.width).sum =
      block.slotCount * block.kind.width
  exact sum_ofFn_const block.slotCount block.kind.width

/-- Canonical offset of one coordinate inside the block. -/
def coordinateOffset {sourceWidth : Nat} (block : Block sourceWidth)
    (slot : Fin block.slotCount) (coordinate : Fin block.kind.width) :
    Fin block.coordinateCount :=
  ⟨slot.val * block.kind.width + coordinate.val, by
    unfold coordinateCount
    have slotSucc : slot.val + 1 ≤ block.slotCount := by omega
    calc
      slot.val * block.kind.width + coordinate.val <
          slot.val * block.kind.width + block.kind.width :=
        Nat.add_lt_add_left coordinate.isLt _
      _ = (slot.val + 1) * block.kind.width := by ring
      _ ≤ block.slotCount * block.kind.width :=
        Nat.mul_le_mul_right block.kind.width slotSucc⟩

/-- Canonical assignment column of one block coordinate. -/
def column {sourceWidth logicalWidth : Nat} (block : Block sourceWidth)
    (start : Nat) (fits : start + block.coordinateCount ≤ logicalWidth)
    (slot : Fin block.slotCount) (coordinate : Fin block.kind.width) :
    Fin logicalWidth :=
  ⟨start + (block.coordinateOffset slot coordinate).val, by
    have offsetBound := (block.coordinateOffset slot coordinate).isLt
    omega⟩

/-- The final assignment contains the canonical encoding at every coordinate
of this block. -/
def EncodesAt {sourceWidth logicalWidth : Nat} (block : Block sourceWidth)
    (start : Nat) (fits : start + block.coordinateCount ≤ logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin sourceWidth → F) : Prop :=
  ∀ slot coordinate,
    assignment (block.column start fits slot coordinate) =
      LowNormSlot.coordinate block.kind (source (block.source slot)) coordinate

/-- Extending the source domain preserves an encoded block when the extended
source agrees on every selected slot. -/
theorem encodesAt_lift
    {sourceWidth largerWidth logicalWidth : Nat}
    (block : Block sourceWidth) (sourceFits : sourceWidth ≤ largerWidth)
    (start : Nat) (fits : start + block.coordinateCount ≤ logicalWidth)
    (liftedFits : start + (block.lift sourceFits).coordinateCount ≤
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin sourceWidth → F) (largerSource : Fin largerWidth → F)
    (encodes : block.EncodesAt start fits assignment source)
    (agrees : ∀ slot : Fin block.slotCount,
      largerSource ((block.lift sourceFits).source slot) =
        source (block.source slot)) :
    (block.lift sourceFits).EncodesAt start liftedFits assignment
      largerSource := by
  intro slot coordinate
  change assignment (block.column start fits slot coordinate) =
    LowNormSlot.coordinate block.kind
      (largerSource ((block.lift sourceFits).source slot)) coordinate
  rw [agrees slot]
  exact encodes slot coordinate

/-- A contiguous slot view reuses the parent coordinates at the exact shifted
start. No copy row or duplicate assignment value is introduced. -/
theorem encodesAt_slice
    {sourceWidth logicalWidth : Nat}
    (block : Block sourceWidth) (offset count : Nat)
    (slotFits : offset + count ≤ block.slotCount)
    (start : Nat) (fits : start + block.coordinateCount ≤ logicalWidth)
    (sliceFits : start + offset * block.kind.width +
        (block.slice offset count slotFits).coordinateCount ≤ logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin sourceWidth → F)
    (encodes : block.EncodesAt start fits assignment source) :
    (block.slice offset count slotFits).EncodesAt
      (start + offset * block.kind.width) sliceFits assignment source := by
  intro slot coordinate
  let parentSlot : Fin block.slotCount :=
    ⟨offset + slot.val, by
      exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left slot.isLt offset)
        slotFits⟩
  have columnEq :
      (block.slice offset count slotFits).column
          (start + offset * block.kind.width) sliceFits slot coordinate =
        block.column start fits parentSlot coordinate := by
    apply Fin.ext
    unfold column coordinateOffset parentSlot
    simp only [slice_kind]
    ring
  rw [columnEq, slice_source]
  exact encodes parentSlot coordinate

/-- Equal starts and proof-irrelevant fit witnesses preserve an encoding. -/
theorem encodesAt_start_eq
    {sourceWidth logicalWidth : Nat} (block : Block sourceWidth)
    (leftStart rightStart : Nat) (startEq : leftStart = rightStart)
    (leftFits : leftStart + block.coordinateCount ≤ logicalWidth)
    (rightFits : rightStart + block.coordinateCount ≤ logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin sourceWidth → F)
    (encodes : block.EncodesAt leftStart leftFits assignment source) :
    block.EncodesAt rightStart rightFits assignment source := by
  cases startEq
  exact encodes

/-- Direct sparse reconstruction form for one retained source value. -/
def form {sourceWidth logicalWidth : Nat} (block : Block sourceWidth)
    (start : Nat) (fits : start + block.coordinateCount ≤ logicalWidth)
    (slot : Fin block.slotCount) : SparseForm logicalWidth :=
  RetainedSlot.recomposeForms <| List.ofFn fun coordinate =>
    SparseForm.singleton (block.column start fits slot coordinate) 1

/-- Direct block forms reconstruct the exact retained source value. -/
theorem form_eval {sourceWidth logicalWidth : Nat} (block : Block sourceWidth)
    (start : Nat) (fits : start + block.coordinateCount ≤ logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin sourceWidth → F) (encodes : block.EncodesAt start fits assignment source)
    (slot : Fin block.slotCount) :
    (block.form start fits slot).eval assignment = source (block.source slot) := by
  rw [form, RetainedSlot.recomposeForms_eval]
  have coordinates :
      (List.ofFn fun coordinate : Fin block.kind.width =>
        SparseForm.singleton (block.column start fits slot coordinate) 1).map
          (fun value => value.eval assignment) =
        LowNormSlot.encode block.kind (source (block.source slot)) := by
    rw [List.map_ofFn]
    calc
      List.ofFn (fun coordinate : Fin block.kind.width =>
          (SparseForm.singleton
            (block.column start fits slot coordinate) 1).eval assignment) =
          List.ofFn (LowNormSlot.coordinate block.kind
            (source (block.source slot))) := by
        apply congrArg List.ofFn
        funext coordinate
        simp [encodes slot coordinate]
      _ = LowNormSlot.encode block.kind (source (block.source slot)) :=
        LowNormSlot.coordinateList_eq_encode _ _
  rw [coordinates]
  exact LowNormSlot.recompose_encode _ _

/-- Every expanded reference slot has the block's one fixed kind. -/
theorem expanded_kind {sourceWidth : Nat} (block : Block sourceWidth)
    (slot : Fin block.expanded.length) :
    (block.expanded.get slot).kind = block.kind := by
  simp [expanded]

/-- Every expanded reference slot has the exact indexed source selected by
the compact block. -/
theorem expanded_source {sourceWidth : Nat} (block : Block sourceWidth)
    (slot : Fin block.slotCount) :
    (block.expanded.get ⟨slot.val, by simpa using slot.isLt⟩).source =
      block.source slot := by
  simp [expanded]

end Block

end NightstreamFPrime.Layout.LowNormBlock
