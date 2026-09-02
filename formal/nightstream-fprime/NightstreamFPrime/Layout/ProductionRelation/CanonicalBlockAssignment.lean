import NightstreamFPrime.Layout.LowNormBlock
import NightstreamFPrime.Layout.ProductionAssignment

/-!
Owns a compact canonical assignment over an ordered list of retained blocks.
It evaluates one coordinate by selecting one block and one slot directly. It
does not materialize an expanded slot list or a complete assignment vector.

The public prefix is verifier-owned. Every retained block begins at the sum of
the preceding block widths and is encoded from its declared source function.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignment

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- One retained block together with the source values it encodes. -/
structure BlockValue where
  sourceWidth : Nat
  block : LowNormBlock.Block sourceWidth
  source : Fin sourceWidth → F

namespace BlockValue

def coordinateCount (entry : BlockValue) : Nat :=
  entry.block.coordinateCount

private theorem kindWidth_pos (kind : LowNormSlot.Kind) : 0 < kind.width := by
  cases kind <;> norm_num [LowNormSlot.Kind.width, BalancedTernary.width]

/-- Constant-time coordinate lookup inside one homogeneous block. -/
def coordinateAt (entry : BlockValue) (index : Nat) : F :=
  if _inside : index < entry.coordinateCount then
    let widthPositive := kindWidth_pos entry.block.kind
    let slot : Fin entry.block.slotCount :=
      ⟨index / entry.block.kind.width, by
        apply (Nat.div_lt_iff_lt_mul widthPositive).2
        simpa [coordinateCount, LowNormBlock.Block.coordinateCount] using _inside⟩
    let coordinate : Fin entry.block.kind.width :=
      ⟨index % entry.block.kind.width,
        Nat.mod_lt index widthPositive⟩
    LowNormSlot.coordinate entry.block.kind
      (entry.source (entry.block.source slot)) coordinate
  else
    0

theorem coordinateAt_coordinateOffset (entry : BlockValue)
    (slot : Fin entry.block.slotCount)
    (coordinate : Fin entry.block.kind.width) :
    entry.coordinateAt (entry.block.coordinateOffset slot coordinate).val =
      LowNormSlot.coordinate entry.block.kind
        (entry.source (entry.block.source slot)) coordinate := by
  unfold coordinateAt
  rw [dif_pos (by
    change (entry.block.coordinateOffset slot coordinate).val <
      entry.block.coordinateCount
    exact (entry.block.coordinateOffset slot coordinate).isLt)]
  dsimp only
  have widthPositive := kindWidth_pos entry.block.kind
  have slotEq :
      (⟨(entry.block.coordinateOffset slot coordinate).val /
          entry.block.kind.width, by
        apply (Nat.div_lt_iff_lt_mul widthPositive).2
        exact (entry.block.coordinateOffset slot coordinate).isLt⟩ :
          Fin entry.block.slotCount) = slot := by
    apply Fin.ext
    change
      (slot.val * entry.block.kind.width + coordinate.val) /
          entry.block.kind.width = slot.val
    rw [Nat.mul_comm slot.val entry.block.kind.width,
      Nat.mul_add_div widthPositive, Nat.div_eq_of_lt coordinate.isLt,
      Nat.add_zero]
  have coordinateEq :
      (⟨(entry.block.coordinateOffset slot coordinate).val %
          entry.block.kind.width,
        Nat.mod_lt _ widthPositive⟩ : Fin entry.block.kind.width) =
        coordinate := by
    apply Fin.ext
    change
      (slot.val * entry.block.kind.width + coordinate.val) %
          entry.block.kind.width = coordinate.val
    exact Nat.mul_add_mod_of_lt coordinate.isLt
  rw [slotEq, coordinateEq]

end BlockValue

abbrev Schedule := List BlockValue

def coordinateCount : Schedule → Nat
  | [] => 0
  | entry :: rest => entry.coordinateCount + coordinateCount rest

/-- Direct coordinate lookup across a small ordered block schedule. -/
def coordinateAt : Schedule → Nat → F
  | [], _ => 0
  | entry :: rest, index =>
      if _inside : index < entry.coordinateCount then
        entry.coordinateAt index
      else
        coordinateAt rest (index - entry.coordinateCount)

@[simp] theorem coordinateCount_append (left right : Schedule) :
    coordinateCount (left ++ right) =
      coordinateCount left + coordinateCount right := by
  induction left with
  | nil => simp [coordinateCount]
  | cons entry rest inductionHypothesis =>
      simp only [List.cons_append, coordinateCount, inductionHypothesis]
      omega

theorem coordinateAt_append_offset (left right : Schedule) (index : Nat) :
    coordinateAt (left ++ right) (coordinateCount left + index) =
      coordinateAt right index := by
  induction left with
  | nil => simp [coordinateCount]
  | cons entry rest inductionHypothesis =>
      simp only [List.cons_append, coordinateCount]
      rw [coordinateAt]
      rw [dif_neg (by omega)]
      have subtract :
          entry.coordinateCount + coordinateCount rest + index -
              entry.coordinateCount =
            coordinateCount rest + index := by
        omega
      rw [subtract]
      exact inductionHypothesis

def ofBlock {sourceWidth : Nat} (block : LowNormBlock.Block sourceWidth)
    (source : Fin sourceWidth → F) : BlockValue :=
  ⟨sourceWidth, block, source⟩

@[simp] theorem ofBlock_eta (entry : BlockValue) :
    ofBlock entry.block entry.source = entry := by
  cases entry
  rfl

theorem coordinateAt_block {sourceWidth : Nat}
    (before after : Schedule) (block : LowNormBlock.Block sourceWidth)
    (source : Fin sourceWidth → F) (slot : Fin block.slotCount)
    (coordinate : Fin block.kind.width) :
    coordinateAt (before ++ ofBlock block source :: after)
        (coordinateCount before + (block.coordinateOffset slot coordinate).val) =
      LowNormSlot.coordinate block.kind (source (block.source slot))
        coordinate := by
  rw [coordinateAt_append_offset]
  unfold coordinateAt
  rw [dif_pos (by
    change (block.coordinateOffset slot coordinate).val <
      block.coordinateCount
    exact (block.coordinateOffset slot coordinate).isLt)]
  exact BlockValue.coordinateAt_coordinateOffset (ofBlock block source)
    slot coordinate

/-- Canonical assignment with a verifier-owned public prefix and direct
retained-block coordinates. Coordinates after the schedule are zero. -/
def assignment {logicalWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (schedule : Schedule) : Assignment F logicalWidth :=
  fun column =>
    if publicRegion : column.val < ProductionAssignment.publicWidth then
      publicInput ⟨column.val, publicRegion⟩
    else
      coordinateAt schedule
        (column.val - ProductionAssignment.publicWidth)

def publicColumn {logicalWidth : Nat}
    (fits : ProductionAssignment.publicWidth ≤ logicalWidth)
    (column : Fin ProductionAssignment.publicWidth) : Fin logicalWidth :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt fits⟩

@[simp] theorem assignment_publicColumn {logicalWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (schedule : Schedule)
    (fits : ProductionAssignment.publicWidth ≤ logicalWidth)
    (column : Fin ProductionAssignment.publicWidth) :
    assignment (logicalWidth := logicalWidth) publicInput schedule
        (publicColumn fits column) =
      publicInput column := by
  unfold assignment publicColumn
  rw [dif_pos column.isLt]

/-- A block at its exact schedule offset has the canonical encoding required
by the direct relation semantics. -/
theorem assignment_encodesAt {sourceWidth logicalWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (before after : Schedule) (block : LowNormBlock.Block sourceWidth)
    (source : Fin sourceWidth → F) (start : Nat)
    (fits : start + block.coordinateCount ≤ logicalWidth)
    (startEq : start =
      ProductionAssignment.publicWidth + coordinateCount before) :
    block.EncodesAt start fits
      (assignment publicInput (before ++ ofBlock block source :: after))
      source := by
  intro slot coordinate
  unfold assignment
  have notPublic : ¬
      (block.column start fits slot coordinate).val <
        ProductionAssignment.publicWidth := by
    unfold LowNormBlock.Block.column
    change ¬start + (block.coordinateOffset slot coordinate).val <
      ProductionAssignment.publicWidth
    omega
  rw [dif_neg notPublic]
  have indexEq :
      (block.column start fits slot coordinate).val -
          ProductionAssignment.publicWidth =
        coordinateCount before +
          (block.coordinateOffset slot coordinate).val := by
    unfold LowNormBlock.Block.column
    change
      start + (block.coordinateOffset slot coordinate).val -
          ProductionAssignment.publicWidth =
        coordinateCount before +
          (block.coordinateOffset slot coordinate).val
    omega
  rw [indexEq]
  exact coordinateAt_block before after block source slot coordinate

theorem schedule_split (schedule : Schedule)
    (index : Fin schedule.length) :
    schedule = schedule.take index.val ++
      schedule.get index :: schedule.drop (index.val + 1) := by
  calc
    schedule = schedule.take index.val ++ schedule.drop index.val :=
      (List.take_append_drop index.val schedule).symm
    _ = schedule.take index.val ++
        schedule.get index :: schedule.drop (index.val + 1) := by
      rw [List.cons_get_drop_succ]

theorem coordinateCount_split (schedule : Schedule)
    (index : Fin schedule.length) :
    coordinateCount schedule =
      coordinateCount (schedule.take index.val) +
        (schedule.get index).coordinateCount +
          coordinateCount (schedule.drop (index.val + 1)) := by
  have counts := congrArg coordinateCount (schedule_split schedule index)
  simpa only [coordinateCount_append, coordinateCount, Nat.add_assoc] using counts

def entryStart (schedule : Schedule) (index : Fin schedule.length) : Nat :=
  ProductionAssignment.publicWidth +
    coordinateCount (schedule.take index.val)

def entryFits {logicalWidth : Nat} (schedule : Schedule)
    (scheduleFits : ProductionAssignment.publicWidth +
      coordinateCount schedule ≤ logicalWidth)
    (index : Fin schedule.length) :
    entryStart schedule index +
        (schedule.get index).block.coordinateCount ≤ logicalWidth := by
  have split := coordinateCount_split schedule index
  unfold entryStart BlockValue.coordinateCount at *
  omega

/-- Every entry of a fitting canonical schedule is encoded at its exact
computed start. -/
theorem assignment_encodesEntry {logicalWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (schedule : Schedule)
    (scheduleFits : ProductionAssignment.publicWidth +
      coordinateCount schedule ≤ logicalWidth)
    (index : Fin schedule.length) :
    let entry := schedule.get index
    entry.block.EncodesAt (entryStart schedule index)
      (entryFits schedule scheduleFits index)
      (assignment publicInput schedule) entry.source := by
  dsimp only
  let before := schedule.take index.val
  let after := schedule.drop (index.val + 1)
  let entry := schedule.get index
  have split : schedule = before ++ entry :: after :=
    schedule_split schedule index
  have encoded := assignment_encodesAt publicInput before after entry.block
    entry.source (entryStart schedule index)
    (entryFits schedule scheduleFits index) (by
      rfl)
  rw [ofBlock_eta] at encoded
  rw [← split] at encoded
  exact encoded

/-- The fixed public marker is one for every canonical encoded-hash public
input. -/
theorem assignment_encHashMarker {logicalWidth : Nat} (digest : Digest)
    (schedule : Schedule)
    (fits : ProductionAssignment.publicWidth ≤ logicalWidth) :
    assignment (logicalWidth := logicalWidth)
        (encodedHashCells digest)
        schedule
        (publicColumn fits encHashMarkerIndex) = 1 := by
  rw [assignment_publicColumn]
  exact encodedHashCells_marker digest

end NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignment
