import Mathlib.Tactic.FinCases
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Layout.LowNormBlock
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan

/-!
Owns the low-level wire operands used by a compact sparse 14-matrix program.
The first operand is one retained low-norm block. It carries only the data
that determines final sparse forms: slot kind, slot count, and final-column
start. Semantic source functions remain in Lean and are not package data.

This module does not select Stage 1 blocks or their order.
-/

namespace NightstreamFPrime.Export.MatrixProgram

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- The exact stored ports for one live production row. -/
abbrev RowForms (logicalWidth : Nat) :=
  Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm logicalWidth

/-- One explicit final-matrix entry on the package wire. -/
structure WireEntry where
  column : Nat
  coefficient : Nat
deriving Repr, DecidableEq

def WireEntry.format : Format WireEntry where
  encode := fun entry => .array [.atom entry.column, .atom entry.coefficient]
  decode
    | .array [.atom column, .atom coefficient] =>
        .ok ⟨column, coefficient⟩
    | _ => .error "invalid matrix sparse entry"
  decode_encode := by
    intro entry
    cases entry
    rfl

/-- Decode one entry only when both its column and field word are canonical. -/
def WireEntry.semantic? (entry : WireEntry) (logicalWidth : Nat) :
    Option (ProductionRelation.SparseEntry logicalWidth) :=
  if columnBound : entry.column < logicalWidth then
    if coefficientBound : entry.coefficient < Spec.goldilocksModulus then
      some ⟨⟨entry.column, columnBound⟩,
        ⟨entry.coefficient, coefficientBound⟩⟩
    else
      none
  else
    none

def WireEntry.ofSemantic {logicalWidth : Nat}
    (entry : ProductionRelation.SparseEntry logicalWidth) : WireEntry where
  column := entry.column.val
  coefficient := entry.coefficient.val

@[simp] theorem WireEntry.semantic?_ofSemantic {logicalWidth : Nat}
    (entry : ProductionRelation.SparseEntry logicalWidth) :
    (WireEntry.ofSemantic entry).semantic? logicalWidth = some entry := by
  simp [semantic?, ofSemantic, entry.column.isLt, entry.coefficient.isLt]

/-- Canonical explicit wire encoding of one final sparse form. -/
structure WireForm where
  entries : List WireEntry
deriving Repr, DecidableEq

def WireForm.format : Format WireForm where
  encode := fun form => (list WireEntry.format).encode form.entries
  decode := fun value => do
    pure ⟨← (list WireEntry.format).decode value⟩
  decode_encode := by
    intro form
    cases form
    simp [Format.decode_encode]

private def decodeWireEntries? (logicalWidth : Nat) :
    List WireEntry → Option (List (ProductionRelation.SparseEntry logicalWidth))
  | [] => some []
  | entry :: rest => do
      let head ← entry.semantic? logicalWidth
      let tail ← decodeWireEntries? logicalWidth rest
      pure (head :: tail)

/-- Reject a form if any encoded entry is noncanonical. -/
def WireForm.semantic? (form : WireForm) (logicalWidth : Nat) :
    Option (ProductionRelation.SparseForm logicalWidth) := do
  pure ⟨← decodeWireEntries? logicalWidth form.entries⟩

def WireForm.ofSemantic {logicalWidth : Nat}
    (form : ProductionRelation.SparseForm logicalWidth) : WireForm where
  entries := form.entries.map WireEntry.ofSemantic

private theorem decodeWireEntries?_ofSemantic {logicalWidth : Nat}
    (entries : List (ProductionRelation.SparseEntry logicalWidth)) :
    decodeWireEntries? logicalWidth (entries.map WireEntry.ofSemantic) =
      some entries := by
  induction entries with
  | nil => rfl
  | cons entry rest inductionHypothesis =>
      simp [decodeWireEntries?, inductionHypothesis]

/-- Explicit wire forms round-trip to the exact semantic sparse form. -/
@[simp] theorem WireForm.semantic?_ofSemantic {logicalWidth : Nat}
    (form : ProductionRelation.SparseForm logicalWidth) :
    (WireForm.ofSemantic form).semantic? logicalWidth = some form := by
  cases form with
  | mk entries =>
      simp [semantic?, ofSemantic, decodeWireEntries?_ofSemantic]

def retainedKindFormat : Format LowNormSlot.Kind where
  encode
    | .bit => .atom 0
    | .centered => .atom 1
    | .field => .atom 2
  decode
    | .atom 0 => .ok .bit
    | .atom 1 => .ok .centered
    | .atom 2 => .ok .field
    | _ => .error "invalid retained slot kind"
  decode_encode := by
    intro kind
    cases kind <;> rfl

/-- Complete wire data for one homogeneous retained block. -/
structure RetainedBlock where
  kind : LowNormSlot.Kind
  slotCount : Nat
  start : Nat
deriving Repr, DecidableEq

def RetainedBlock.format : Format RetainedBlock where
  encode := fun block => .array [
    retainedKindFormat.encode block.kind,
    .atom block.slotCount,
    .atom block.start]
  decode
    | .array [kind, .atom slotCount, .atom start] => do
      pure ⟨← retainedKindFormat.decode kind, slotCount, start⟩
    | _ => .error "invalid retained matrix block"
  decode_encode := by
    intro block
    cases block
    simp [retainedKindFormat.decode_encode]

/-- A proof-erased semantic block with the same sparse-form geometry. -/
def RetainedBlock.semantic (block : RetainedBlock) :
    LowNormBlock.Block block.slotCount where
  kind := block.kind
  slotCount := block.slotCount
  source := id

def RetainedBlock.coordinateCount (block : RetainedBlock) : Nat :=
  block.slotCount * block.kind.width

@[simp] theorem RetainedBlock.semantic_coordinateCount
    (block : RetainedBlock) :
    block.semantic.coordinateCount = block.coordinateCount := by
  rfl

/-- Fail-closed reconstruction of one retained sparse form. -/
def RetainedBlock.form? (block : RetainedBlock) (logicalWidth slot : Nat) :
    Option (SparseForm logicalWidth) :=
  if slotBound : slot < block.slotCount then
    if fits : block.start + block.coordinateCount ≤ logicalWidth then
      some (block.semantic.form block.start
        (by simpa only [semantic_coordinateCount] using fits) ⟨slot, slotBound⟩)
    else
      none
  else
    none

/-- Erase the semantic source function while retaining exact sparse-form
geometry. -/
def RetainedBlock.ofSemantic {sourceWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (start : Nat) : RetainedBlock where
  kind := block.kind
  slotCount := block.slotCount
  start := start

/-- Wire reconstruction is exactly the semantic retained form. -/
theorem RetainedBlock.form?_ofSemantic {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + block.coordinateCount ≤ logicalWidth)
    (slot : Fin block.slotCount) :
    (ofSemantic block start).form? logicalWidth slot.val =
      some (block.form start fits slot) := by
  unfold form? ofSemantic
  rw [dif_pos slot.isLt]
  rw [dif_pos (by simpa only [coordinateCount,
    LowNormBlock.Block.coordinateCount] using fits)]
  rfl

private def fixedState8 {Alpha : Type}
    (lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7 : Alpha) : Fin 8 → Alpha :=
  fun lane =>
    [lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7].get
      ⟨lane.val, by simpa using lane.isLt⟩

/-- Reconstruct one Poseidon2 external-layer output from eight consecutive
retained final-round S-box slots. -/
def RetainedBlock.externalForm? (block : RetainedBlock)
    (logicalWidth slotBase lane : Nat) : Option (SparseForm logicalWidth) := do
  if laneBound : lane < 8 then do
    let lane0 ← block.form? logicalWidth (slotBase + 0)
    let lane1 ← block.form? logicalWidth (slotBase + 1)
    let lane2 ← block.form? logicalWidth (slotBase + 2)
    let lane3 ← block.form? logicalWidth (slotBase + 3)
    let lane4 ← block.form? logicalWidth (slotBase + 4)
    let lane5 ← block.form? logicalWidth (slotBase + 5)
    let lane6 ← block.form? logicalWidth (slotBase + 6)
    let lane7 ← block.form? logicalWidth (slotBase + 7)
    pure (SparseLayer.external
      (fixedState8 lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7)
      ⟨lane, laneBound⟩)
  else
    none

/-- The wire external-layer reconstruction is the exact semantic sparse
external layer over the selected retained slots. -/
theorem RetainedBlock.externalForm?_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (slotBase : Nat)
    (slotBound : ∀ lane : Fin 8, slotBase + lane.val < block.slotCount)
    (lane : Fin 8) :
    (RetainedBlock.ofSemantic block retainedStart).externalForm?
        logicalWidth slotBase lane.val =
      some (SparseLayer.external (fun selected : Fin 8 =>
        block.form retainedStart fits
          ⟨slotBase + selected.val, slotBound selected⟩) lane) := by
  let sourceSlot (selected : Fin 8) : Fin block.slotCount :=
    ⟨slotBase + selected.val, slotBound selected⟩
  unfold externalForm?
  rw [dif_pos lane.isLt]
  have lane0_eq :
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + 0) =
        some (block.form retainedStart fits (sourceSlot 0)) := by
    simpa [sourceSlot] using
      RetainedBlock.form?_ofSemantic block retainedStart fits (sourceSlot 0)
  have lane1_eq :
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + 1) =
        some (block.form retainedStart fits (sourceSlot 1)) := by
    simpa [sourceSlot] using
      RetainedBlock.form?_ofSemantic block retainedStart fits (sourceSlot 1)
  have lane2_eq :
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + 2) =
        some (block.form retainedStart fits (sourceSlot 2)) := by
    simpa [sourceSlot] using
      RetainedBlock.form?_ofSemantic block retainedStart fits (sourceSlot 2)
  have lane3_eq :
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + 3) =
        some (block.form retainedStart fits (sourceSlot 3)) := by
    simpa [sourceSlot] using
      RetainedBlock.form?_ofSemantic block retainedStart fits (sourceSlot 3)
  have lane4_eq :
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + 4) =
        some (block.form retainedStart fits (sourceSlot 4)) := by
    simpa [sourceSlot] using
      RetainedBlock.form?_ofSemantic block retainedStart fits (sourceSlot 4)
  have lane5_eq :
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + 5) =
        some (block.form retainedStart fits (sourceSlot 5)) := by
    simpa [sourceSlot] using
      RetainedBlock.form?_ofSemantic block retainedStart fits (sourceSlot 5)
  have lane6_eq :
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + 6) =
        some (block.form retainedStart fits (sourceSlot 6)) := by
    simpa [sourceSlot] using
      RetainedBlock.form?_ofSemantic block retainedStart fits (sourceSlot 6)
  have lane7_eq :
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + 7) =
        some (block.form retainedStart fits (sourceSlot 7)) := by
    simpa [sourceSlot] using
      RetainedBlock.form?_ofSemantic block retainedStart fits (sourceSlot 7)
  rw [lane0_eq, lane1_eq, lane2_eq, lane3_eq, lane4_eq, lane5_eq,
    lane6_eq, lane7_eq]
  apply congrArg some
  apply congrArg (fun state => SparseLayer.external state lane)
  funext selected
  fin_cases selected <;> rfl

/-- One explicit contiguous source interval mapped to a contiguous retained
slot interval. -/
structure SourceRange where
  sourceStart : Nat
  sourceCount : Nat
  retained : RetainedBlock
  slotStart : Nat
deriving Repr, DecidableEq

def SourceRange.format : Format SourceRange where
  encode := fun range => .array [
    .atom range.sourceStart,
    .atom range.sourceCount,
    RetainedBlock.format.encode range.retained,
    .atom range.slotStart]
  decode
    | .array [.atom sourceStart, .atom sourceCount, retained,
        .atom slotStart] => do
      pure ⟨sourceStart, sourceCount,
        ← RetainedBlock.format.decode retained, slotStart⟩
    | _ => .error "invalid matrix source range"
  decode_encode := by
    intro range
    cases range
    simp [RetainedBlock.format.decode_encode]

/-- Fail-closed source-column substitution for one contiguous range. -/
def SourceRange.form? (range : SourceRange) (logicalWidth source : Nat) :
    Option (SparseForm logicalWidth) :=
  if range.sourceStart ≤ source then
    let offset := source - range.sourceStart
    if offset < range.sourceCount then
      range.retained.form? logicalWidth (range.slotStart + offset)
    else
      none
  else
    none

theorem SourceRange.form?_eq_none_of_before (range : SourceRange)
    (logicalWidth source : Nat) (before : source < range.sourceStart) :
    range.form? logicalWidth source = none := by
  unfold form?
  rw [if_neg (by omega)]

theorem SourceRange.form?_eq_none_of_after (range : SourceRange)
    (logicalWidth source : Nat)
    (after : range.sourceStart + range.sourceCount ≤ source) :
    range.form? logicalWidth source = none := by
  unfold form?
  rw [if_pos (by omega)]
  rw [if_neg (by omega)]

def SourceRange.ofSemantic {sourceWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart sourceStart
      sourceCount slotStart : Nat) : SourceRange where
  sourceStart := sourceStart
  sourceCount := sourceCount
  retained := RetainedBlock.ofSemantic block retainedStart
  slotStart := slotStart

/-- A canonical in-range wire substitution is the exact retained semantic
form. -/
theorem SourceRange.form?_ofSemantic {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth)
    (retainedStart sourceStart sourceCount slotStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (slotFits : slotStart + sourceCount ≤ block.slotCount)
    (offset : Fin sourceCount) :
    (ofSemantic block retainedStart sourceStart sourceCount slotStart).form?
        logicalWidth (sourceStart + offset.val) =
      some (block.form retainedStart fits
        ⟨slotStart + offset.val, by omega⟩) := by
  change
    (if sourceStart ≤ sourceStart + offset.val then
      if sourceStart + offset.val - sourceStart < sourceCount then
        (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotStart + (sourceStart + offset.val - sourceStart))
      else none
    else none) = _
  rw [if_pos (by omega)]
  rw [show sourceStart + offset.val - sourceStart = offset.val by omega]
  rw [if_pos offset.isLt]
  exact RetainedBlock.form?_ofSemantic block retainedStart fits
    ⟨slotStart + offset.val, by omega⟩

inductive SourceGridMode where
  | direct
  | external8
deriving Repr, DecidableEq

def SourceGridMode.format : Format SourceGridMode where
  encode
    | .direct => .atom 0
    | .external8 => .atom 1
  decode
    | .atom 0 => .ok .direct
    | .atom 1 => .ok .external8
    | _ => .error "invalid matrix source grid mode"
  decode_encode := by
    intro mode
    cases mode <;> rfl

/-- One exact two-level affine family of contiguous source runs. This keeps
regular source layouts compact without accepting stride gaps. -/
structure SourceGrid where
  sourceStart : Nat
  majorCount : Nat
  majorSourceStride : Nat
  minorCount : Nat
  minorSourceStride : Nat
  runCount : Nat
  retained : RetainedBlock
  mode : SourceGridMode
  slotStart : Nat
  majorSlotStride : Nat
  minorSlotStride : Nat
deriving Repr, DecidableEq

def SourceGrid.format : Format SourceGrid where
  encode := fun grid => .array [
    .atom grid.sourceStart,
    .atom grid.majorCount,
    .atom grid.majorSourceStride,
    .atom grid.minorCount,
    .atom grid.minorSourceStride,
    .atom grid.runCount,
    RetainedBlock.format.encode grid.retained,
    SourceGridMode.format.encode grid.mode,
    .atom grid.slotStart,
    .atom grid.majorSlotStride,
    .atom grid.minorSlotStride]
  decode
    | .array [.atom sourceStart, .atom majorCount,
        .atom majorSourceStride, .atom minorCount,
        .atom minorSourceStride, .atom runCount, retained,
        mode, .atom slotStart, .atom majorSlotStride,
        .atom minorSlotStride] => do
      pure {
        sourceStart
        majorCount
        majorSourceStride
        minorCount
        minorSourceStride
        runCount
        retained := ← RetainedBlock.format.decode retained
        mode := ← SourceGridMode.format.decode mode
        slotStart
        majorSlotStride
        minorSlotStride }
    | _ => .error "invalid matrix source grid"
  decode_encode := by
    intro grid
    cases grid
    simp [RetainedBlock.format.decode_encode,
      SourceGridMode.format.decode_encode]
    rfl

/-- Fail-closed lookup in one affine grid. Division selects a unique major
and minor cell; explicit bounds reject every gap. -/
def SourceGrid.form? (grid : SourceGrid) (logicalWidth source : Nat) :
    Option (SparseForm logicalWidth) :=
  if grid.sourceStart ≤ source then
    if 0 < grid.majorSourceStride then
      let delta := source - grid.sourceStart
      let major := delta / grid.majorSourceStride
      let majorOffset := delta % grid.majorSourceStride
      if major < grid.majorCount then
        if 0 < grid.minorSourceStride then
          let minor := majorOffset / grid.minorSourceStride
          let offset := majorOffset % grid.minorSourceStride
          if minor < grid.minorCount then
            if offset < grid.runCount then
              let slotBase := grid.slotStart + major * grid.majorSlotStride +
                minor * grid.minorSlotStride
              match grid.mode with
              | .direct =>
                  grid.retained.form? logicalWidth (slotBase + offset)
              | .external8 =>
                  grid.retained.externalForm? logicalWidth slotBase offset
            else none
          else none
        else none
      else none
    else none
  else none

def SourceGrid.ofSemantic {sourceWidth : Nat}
    (block : LowNormBlock.Block sourceWidth)
    (retainedStart sourceStart majorCount majorSourceStride minorCount
      minorSourceStride runCount slotStart majorSlotStride minorSlotStride :
      Nat) : SourceGrid where
  sourceStart := sourceStart
  majorCount := majorCount
  majorSourceStride := majorSourceStride
  minorCount := minorCount
  minorSourceStride := minorSourceStride
  runCount := runCount
  retained := RetainedBlock.ofSemantic block retainedStart
  mode := .direct
  slotStart := slotStart
  majorSlotStride := majorSlotStride
  minorSlotStride := minorSlotStride

def SourceGrid.externalOfSemantic {sourceWidth : Nat}
    (block : LowNormBlock.Block sourceWidth)
    (retainedStart sourceStart majorCount majorSourceStride minorCount
      minorSourceStride runCount slotStart majorSlotStride minorSlotStride :
      Nat) : SourceGrid :=
  { ofSemantic block retainedStart sourceStart majorCount majorSourceStride
      minorCount minorSourceStride runCount slotStart majorSlotStride
      minorSlotStride with
    mode := .external8 }

private theorem stride_div (outer offset stride : Nat)
    (positive : 0 < stride) (offsetBound : offset < stride) :
    (outer * stride + offset) / stride = outer := by
  rw [Nat.mul_comm outer stride, Nat.mul_add_div positive,
    Nat.div_eq_of_lt offsetBound, Nat.add_zero]

private theorem stride_mod (outer offset stride : Nat)
    (offsetBound : offset < stride) :
    (outer * stride + offset) % stride = offset := by
  exact Nat.mul_add_mod_of_lt offsetBound

theorem SourceGrid.form?_eq_none_of_before (grid : SourceGrid)
    (logicalWidth source : Nat) (before : source < grid.sourceStart) :
    grid.form? logicalWidth source = none := by
  unfold form?
  rw [if_neg (by omega)]

theorem SourceGrid.form?_eq_none_of_after (grid : SourceGrid)
    (logicalWidth source : Nat)
    (positive : 0 < grid.majorSourceStride)
    (after : grid.sourceStart +
      grid.majorCount * grid.majorSourceStride ≤ source) :
    grid.form? logicalWidth source = none := by
  unfold form?
  rw [if_pos (by omega), if_pos positive]
  simp only
  rw [if_neg]
  intro inside
  have lower : grid.majorCount ≤
      (source - grid.sourceStart) / grid.majorSourceStride := by
    rw [Nat.le_div_iff_mul_le positive]
    omega
  omega

theorem SourceGrid.form?_eq_none_at_gap (grid : SourceGrid)
    (logicalWidth : Nat) (major : Fin grid.majorCount)
    (minor : Fin grid.minorCount) (offset : Nat)
    (majorPositive : 0 < grid.majorSourceStride)
    (minorPositive : 0 < grid.minorSourceStride)
    (minorCellBound : minor.val * grid.minorSourceStride + offset <
      grid.majorSourceStride)
    (offsetBound : offset < grid.minorSourceStride)
    (gap : grid.runCount ≤ offset) :
    grid.form? logicalWidth
        (grid.sourceStart + major.val * grid.majorSourceStride +
          minor.val * grid.minorSourceStride + offset) = none := by
  simp only [form?]
  rw [if_pos (by omega)]
  rw [show
      grid.sourceStart + major.val * grid.majorSourceStride +
          minor.val * grid.minorSourceStride + offset - grid.sourceStart =
        major.val * grid.majorSourceStride +
          (minor.val * grid.minorSourceStride + offset) by omega]
  rw [if_pos majorPositive]
  rw [stride_div major.val
      (minor.val * grid.minorSourceStride + offset)
      grid.majorSourceStride majorPositive minorCellBound]
  rw [stride_mod major.val
      (minor.val * grid.minorSourceStride + offset)
      grid.majorSourceStride minorCellBound]
  rw [if_pos major.isLt, if_pos minorPositive]
  rw [stride_div minor.val offset grid.minorSourceStride minorPositive
      offsetBound]
  rw [stride_mod minor.val offset grid.minorSourceStride offsetBound]
  rw [if_pos minor.isLt, if_neg (by omega)]

theorem SourceGrid.form?_eq_none_at_minorAfter (grid : SourceGrid)
    (logicalWidth : Nat) (major : Fin grid.majorCount)
    (minor offset : Nat)
    (majorPositive : 0 < grid.majorSourceStride)
    (minorPositive : 0 < grid.minorSourceStride)
    (minorCellBound : minor * grid.minorSourceStride + offset <
      grid.majorSourceStride)
    (offsetBound : offset < grid.minorSourceStride)
    (after : grid.minorCount ≤ minor) :
    grid.form? logicalWidth
        (grid.sourceStart + major.val * grid.majorSourceStride +
          minor * grid.minorSourceStride + offset) = none := by
  simp only [form?]
  rw [if_pos (by omega)]
  rw [show
      grid.sourceStart + major.val * grid.majorSourceStride +
          minor * grid.minorSourceStride + offset - grid.sourceStart =
        major.val * grid.majorSourceStride +
          (minor * grid.minorSourceStride + offset) by omega]
  rw [if_pos majorPositive]
  rw [stride_div major.val (minor * grid.minorSourceStride + offset)
      grid.majorSourceStride majorPositive minorCellBound]
  rw [stride_mod major.val (minor * grid.minorSourceStride + offset)
      grid.majorSourceStride minorCellBound]
  rw [if_pos major.isLt, if_pos minorPositive]
  rw [stride_div minor offset grid.minorSourceStride minorPositive offsetBound]
  rw [stride_mod minor offset grid.minorSourceStride offsetBound]
  rw [if_neg (by omega)]

/-- A valid grid coordinate reconstructs the exact semantic retained form. -/
theorem SourceGrid.form?_ofSemantic {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth)
    (retainedStart sourceStart majorCount majorSourceStride minorCount
      minorSourceStride runCount slotStart majorSlotStride minorSlotStride :
      Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (majorPositive : 0 < majorSourceStride)
    (minorPositive : 0 < minorSourceStride)
    (major : Fin majorCount) (minor : Fin minorCount)
    (offset : Fin runCount)
    (minorCellBound : minor.val * minorSourceStride + offset.val <
      majorSourceStride)
    (offsetBound : offset.val < minorSourceStride)
    (slotBound : slotStart + major.val * majorSlotStride +
      minor.val * minorSlotStride + offset.val < block.slotCount) :
    (ofSemantic block retainedStart sourceStart majorCount majorSourceStride
      minorCount minorSourceStride runCount slotStart majorSlotStride
      minorSlotStride).form? logicalWidth
        (sourceStart + major.val * majorSourceStride +
          minor.val * minorSourceStride + offset.val) =
      some (block.form retainedStart fits ⟨_, slotBound⟩) := by
  simp only [form?, ofSemantic]
  rw [if_pos (by omega)]
  rw [show
      sourceStart + major.val * majorSourceStride +
          minor.val * minorSourceStride + offset.val - sourceStart =
        major.val * majorSourceStride +
          (minor.val * minorSourceStride + offset.val) by omega]
  rw [if_pos majorPositive]
  rw [stride_div major.val
      (minor.val * minorSourceStride + offset.val) majorSourceStride
      majorPositive minorCellBound]
  rw [stride_mod major.val
      (minor.val * minorSourceStride + offset.val) majorSourceStride
      minorCellBound]
  rw [if_pos major.isLt, if_pos minorPositive]
  rw [stride_div minor.val offset.val minorSourceStride minorPositive
      offsetBound]
  rw [stride_mod minor.val offset.val minorSourceStride offsetBound]
  rw [if_pos minor.isLt, if_pos offset.isLt]
  exact RetainedBlock.form?_ofSemantic block retainedStart fits
    ⟨slotStart + major.val * majorSlotStride +
      minor.val * minorSlotStride + offset.val, slotBound⟩

/-- A valid external grid coordinate reconstructs the exact Poseidon2
external-layer sparse form over eight consecutive retained slots. -/
theorem SourceGrid.form?_externalOfSemantic
    {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth)
    (retainedStart sourceStart majorCount majorSourceStride minorCount
      minorSourceStride runCount slotStart majorSlotStride minorSlotStride :
      Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (majorPositive : 0 < majorSourceStride)
    (minorPositive : 0 < minorSourceStride)
    (major : Fin majorCount) (minor : Fin minorCount)
    (offset : Fin runCount)
    (minorCellBound : minor.val * minorSourceStride + offset.val <
      majorSourceStride)
    (offsetBound : offset.val < minorSourceStride)
    (laneBound : offset.val < 8)
    (slotBound : ∀ lane : Fin 8,
      slotStart + major.val * majorSlotStride +
        minor.val * minorSlotStride + lane.val < block.slotCount) :
    (externalOfSemantic block retainedStart sourceStart majorCount
      majorSourceStride minorCount minorSourceStride runCount slotStart
      majorSlotStride minorSlotStride).form? logicalWidth
        (sourceStart + major.val * majorSourceStride +
          minor.val * minorSourceStride + offset.val) =
      some (SparseLayer.external (fun lane : Fin 8 =>
        block.form retainedStart fits
          ⟨slotStart + major.val * majorSlotStride +
            minor.val * minorSlotStride + lane.val, slotBound lane⟩)
        ⟨offset.val, laneBound⟩) := by
  simp only [form?, externalOfSemantic, ofSemantic]
  rw [if_pos (by omega)]
  rw [show
      sourceStart + major.val * majorSourceStride +
          minor.val * minorSourceStride + offset.val - sourceStart =
        major.val * majorSourceStride +
          (minor.val * minorSourceStride + offset.val) by omega]
  rw [if_pos majorPositive]
  rw [stride_div major.val
      (minor.val * minorSourceStride + offset.val) majorSourceStride
      majorPositive minorCellBound]
  rw [stride_mod major.val
      (minor.val * minorSourceStride + offset.val) majorSourceStride
      minorCellBound]
  rw [if_pos major.isLt, if_pos minorPositive]
  rw [stride_div minor.val offset.val minorSourceStride minorPositive
      offsetBound]
  rw [stride_mod minor.val offset.val minorSourceStride offsetBound]
  rw [if_pos minor.isLt, if_pos offset.isLt]
  exact RetainedBlock.externalForm?_ofSemantic block retainedStart fits
    (slotStart + major.val * majorSlotStride +
      minor.val * minorSlotStride) slotBound ⟨offset.val, laneBound⟩

/-- Exact source substitution table. A column must resolve through one and
only one range. -/
structure SourceSubstitution where
  ranges : List SourceRange
  grids : List SourceGrid := []
deriving Repr, DecidableEq

def SourceSubstitution.format : Format SourceSubstitution where
  encode := fun substitution => .array [
    (list SourceRange.format).encode substitution.ranges,
    (list SourceGrid.format).encode substitution.grids]
  decode
    | .array [ranges, grids] => do
      pure ⟨← (list SourceRange.format).decode ranges,
        ← (list SourceGrid.format).decode grids⟩
    | _ => .error "invalid matrix source substitution"
  decode_encode := by
    rintro ⟨ranges, grids⟩
    simp only
    rw [(list SourceRange.format).decode_encode,
      (list SourceGrid.format).decode_encode]
    rfl

/-- Fail closed on a missing, invalid, or overlapping source mapping. -/
def SourceSubstitution.form? (substitution : SourceSubstitution)
    (logicalWidth source : Nat) : Option (SparseForm logicalWidth) :=
  match substitution.ranges.filterMap
        (fun range => range.form? logicalWidth source) ++
      substitution.grids.filterMap
        (fun grid => grid.form? logicalWidth source) with
  | [form] => some form
  | _ => none

@[simp] theorem SourceSubstitution.singleton_form?
    (range : SourceRange) (logicalWidth source : Nat) :
    (SourceSubstitution.mk [range] []).form? logicalWidth source =
      range.form? logicalWidth source := by
  cases found : range.form? logicalWidth source <;>
    simp [form?, found]

/-- A one-range substitution reproduces the exact semantic retained form. -/
theorem SourceSubstitution.singleton_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth)
    (retainedStart sourceStart sourceCount slotStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (slotFits : slotStart + sourceCount ≤ block.slotCount)
    (offset : Fin sourceCount) :
    (SourceSubstitution.mk [SourceRange.ofSemantic block retainedStart
      sourceStart sourceCount slotStart] []).form? logicalWidth
        (sourceStart + offset.val) =
      some (block.form retainedStart fits
        ⟨slotStart + offset.val, by omega⟩) := by
  rw [singleton_form?]
  exact SourceRange.form?_ofSemantic block retainedStart sourceStart
    sourceCount slotStart fits slotFits offset

/-- One contiguous interval in an existing canonical package stream. -/
structure IndexRange where
  start : Nat
  count : Nat
deriving Repr, DecidableEq

def IndexRange.format : Format IndexRange where
  encode := fun range => .array [.atom range.start, .atom range.count]
  decode
    | .array [.atom start, .atom count] => .ok ⟨start, count⟩
    | _ => .error "invalid matrix index range"
  decode_encode := by
    intro range
    cases range
    rfl

def IndexRange.endExclusive (range : IndexRange) : Nat :=
  range.start + range.count

/-- Ordered selection from an existing package stream. Large monotone
selections use ranges. Small package reorderings use an exact random-access
table. -/
inductive IndexSchedule where
  | rangeList (ranges : List IndexRange)
  | indexTable (indices : Array Nat)
deriving Repr, DecidableEq

def IndexSchedule.format : Format IndexSchedule where
  encode
    | .rangeList ranges => .array [
        .atom 0, (list IndexRange.format).encode ranges]
    | .indexTable indices => .array [
        .atom 1, (list nat).encode indices.toList]
  decode
    | .array [.atom 0, ranges] => do
        pure (.rangeList (← (list IndexRange.format).decode ranges))
    | .array [.atom 1, indices] => do
        pure (.indexTable (← (list nat).decode indices).toArray)
    | _ => .error "invalid matrix index schedule"
  decode_encode := by
    intro schedule
    cases schedule <;> simp [Format.decode_encode]

def IndexSchedule.count : IndexSchedule → Nat
  | .rangeList ranges => (ranges.map IndexRange.count).sum
  | .indexTable indices => indices.size

def validIndexRanges (limit minimumStart : Nat) :
    List IndexRange → Bool
  | [] => true
  | range :: rest =>
      range.count != 0 && minimumStart ≤ range.start &&
        range.endExclusive ≤ limit &&
          validIndexRanges limit range.endExclusive rest

private def validIndexTable (limit : Nat) : List Nat → Bool
  | [] => true
  | index :: rest => index < limit && validIndexTable limit rest

/-- Reject malformed ranges or an out-of-bounds table entry. Exact table
order is package data and can represent a canonical stable partition. -/
def IndexSchedule.valid (schedule : IndexSchedule) (limit : Nat) : Bool :=
  match schedule with
  | .rangeList ranges => validIndexRanges limit 0 ranges
  | .indexTable indices => validIndexTable limit indices.toList

/-- Select one index without expanding the schedule. -/
def IndexSchedule.index? (schedule : IndexSchedule) : Nat → Option Nat
  | ordinal =>
      match schedule with
      | .rangeList ranges => select ranges ordinal
      | .indexTable indices => indices[ordinal]?
where
  select : List IndexRange → Nat → Option Nat
    | [], _ => none
    | range :: rest, ordinal =>
        if ordinal < range.count then
          some (range.start + ordinal)
        else
          select rest (ordinal - range.count)

/-- Proof-oriented ordered expansion of one index range. -/
def IndexRange.indices (range : IndexRange) : List Nat :=
  (List.range range.count).map fun offset => range.start + offset

theorem IndexRange.indices_eq_range' (range : IndexRange) :
    range.indices = List.range' range.start range.count := by
  exact List.range'_eq_map_range.symm

/-- Proof-oriented ordered expansion of a schedule. The executable
interpreter uses `index?` and does not construct this list. -/
def IndexSchedule.indices (schedule : IndexSchedule) : List Nat :=
  match schedule with
  | .rangeList ranges => ranges.flatMap IndexRange.indices
  | .indexTable indices => indices.toList

private theorem validIndexRanges_indices
    (limit minimumStart : Nat) (ranges : List IndexRange)
    (valid : validIndexRanges limit minimumStart ranges = true) :
    let indices := ranges.flatMap IndexRange.indices
    indices.Nodup ∧
      ∀ index ∈ indices, minimumStart ≤ index ∧ index < limit := by
  induction ranges generalizing minimumStart with
  | nil => simp
  | cons range rest inductionHypothesis =>
      simp only [validIndexRanges, Bool.and_eq_true,
        decide_eq_true_eq, bne_iff_ne] at valid
      rcases valid with
        ⟨⟨⟨_countNonzero, minimumLe⟩, endLe⟩, restValid⟩
      have restProperties := inductionHypothesis range.endExclusive restValid
      rcases restProperties with ⟨restNodup, restBounds⟩
      unfold IndexRange.endExclusive at endLe restBounds
      dsimp only
      simp only [List.flatMap_cons]
      constructor
      · rw [List.nodup_append]
        refine ⟨?_, restNodup, ?_⟩
        · rw [IndexRange.indices_eq_range']
          exact List.nodup_range'
        · intro first firstMember second secondMember equal
          rw [IndexRange.indices_eq_range', List.mem_range'_1] at firstMember
          have secondBounds := restBounds second secondMember
          omega
      · intro index member
        rw [List.mem_append] at member
        rcases member with headMember | tailMember
        · rw [IndexRange.indices_eq_range', List.mem_range'_1] at headMember
          omega
        · have bounds := restBounds index tailMember
          omega

/-- A valid ordered range schedule selects each source row at most once. -/
theorem IndexSchedule.rangeList_indices_nodup
    (ranges : List IndexRange) (limit : Nat)
    (valid : (IndexSchedule.rangeList ranges).valid limit = true) :
    (IndexSchedule.rangeList ranges).indices.Nodup := by
  exact (validIndexRanges_indices limit 0 ranges valid).1

/-- The same structural range validation also proves exact lower and upper
bounds for every expanded source index. -/
theorem validIndexRanges_indices_bounds
    (ranges : List IndexRange) (minimumStart limit : Nat)
    (valid : validIndexRanges limit minimumStart ranges = true) :
    ∀ index ∈ ranges.flatMap IndexRange.indices,
      minimumStart ≤ index ∧ index < limit := by
  exact (validIndexRanges_indices limit minimumStart ranges valid).2

@[simp] theorem IndexRange.indices_length (range : IndexRange) :
    range.indices.length = range.count := by
  simp [IndexRange.indices]

@[simp] theorem IndexSchedule.indices_length (schedule : IndexSchedule) :
    schedule.indices.length = schedule.count := by
  cases schedule with
  | rangeList ranges =>
      unfold IndexSchedule.indices IndexSchedule.count
      induction ranges with
      | nil => rfl
      | cons range rest inductionHypothesis =>
          simp [inductionHypothesis]
  | indexTable indices =>
      simp [IndexSchedule.indices, IndexSchedule.count]

/-- Streaming index selection is exactly lookup in the canonical ordered
schedule expansion. -/
theorem IndexSchedule.index?_eq_getElem? (schedule : IndexSchedule)
    (ordinal : Nat) :
    schedule.index? ordinal = schedule.indices[ordinal]? := by
  cases schedule with
  | indexTable indices =>
      simp [IndexSchedule.index?, IndexSchedule.indices]
  | rangeList ranges =>
      unfold IndexSchedule.index? IndexSchedule.indices
      induction ranges generalizing ordinal with
      | nil => rfl
      | cons range rest inductionHypothesis =>
          simp only [IndexSchedule.index?.select, List.flatMap_cons]
          by_cases inside : ordinal < range.count
          · rw [if_pos inside, List.getElem?_append_left]
            · simp [IndexRange.indices, inside]
            · simpa using inside
          · rw [if_neg inside, List.getElem?_append_right]
            · simpa [IndexRange.indices] using
                inductionHypothesis (ordinal - range.count)
            · simp [IndexRange.indices, Nat.le_of_not_gt inside]

@[simp] theorem IndexSchedule.singleton_index?
    (range : IndexRange) (ordinal : Nat) :
    (IndexSchedule.rangeList [range]).index? ordinal =
      if ordinal < range.count then some (range.start + ordinal) else none := by
  simp [IndexSchedule.index?, IndexSchedule.index?.select]

@[simp] theorem IndexSchedule.table_index? (indices : Array Nat)
    (ordinal : Nat) :
    (IndexSchedule.indexTable indices).index? ordinal = indices[ordinal]? := by
  rfl

end NightstreamFPrime.Export.MatrixProgram
