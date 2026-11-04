import Mathlib.Tactic.FinCases
import NightstreamFPrime.Layout.LowNormBlock
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan

namespace NightstreamFPrime.Layout.MatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

abbrev RowForms (logicalWidth : Nat) :=
  Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm logicalWidth

structure WireEntry where
  column : Nat
  coefficient : Nat
deriving Repr, DecidableEq

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

structure WireForm where
  entries : List WireEntry
deriving Repr, DecidableEq

private def decodeWireEntries? (logicalWidth : Nat) :
    List WireEntry → Option (List (ProductionRelation.SparseEntry logicalWidth))
  | [] => some []
  | entry :: rest => do
      let head ← entry.semantic? logicalWidth
      let tail ← decodeWireEntries? logicalWidth rest
      pure (head :: tail)

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

@[simp] theorem WireForm.semantic?_ofSemantic {logicalWidth : Nat}
    (form : ProductionRelation.SparseForm logicalWidth) :
    (WireForm.ofSemantic form).semantic? logicalWidth = some form := by
  cases form with
  | mk entries =>
      simp [semantic?, ofSemantic, decodeWireEntries?_ofSemantic]

structure RetainedBlock where
  kind : LowNormSlot.Kind
  slotCount : Nat
  start : Nat
deriving Repr, DecidableEq

def RetainedBlock.semantic (block : RetainedBlock) :
    LowNormBlock.Block block.slotCount where
  kind := block.kind
  slotCount := block.slotCount
  source := id

@[implicit_reducible]
def RetainedBlock.coordinateCount (block : RetainedBlock) : Nat :=
  block.slotCount * block.kind.width

@[simp] theorem RetainedBlock.semantic_coordinateCount
    (block : RetainedBlock) :
    block.semantic.coordinateCount = block.coordinateCount := by
  rfl

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

def RetainedBlock.ofSemantic {sourceWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (start : Nat) : RetainedBlock where
  kind := block.kind
  slotCount := block.slotCount
  start := start

theorem RetainedBlock.form?_ofSemantic {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + block.coordinateCount ≤ logicalWidth)
    (slot : Fin block.slotCount) :
    (ofSemantic block start).form? logicalWidth slot.val =
      some (block.form start fits slot) := by
  unfold form? ofSemantic
  rw [dite_eq_left slot.isLt]
  rw [dite_eq_left (by simpa only [coordinateCount,
    LowNormBlock.Block.coordinateCount] using fits)]
  rfl

private def fixedState16 {Alpha : Type}
    (lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7 lane8 lane9 lane10 lane11 lane12 lane13 lane14 lane15 : Alpha) :
    Fin 16 → Alpha :=
  fun lane =>
    [lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7,
      lane8, lane9, lane10, lane11, lane12, lane13, lane14, lane15].get
      ⟨lane.val, by simp⟩

def RetainedBlock.externalForm? (block : RetainedBlock)
    (logicalWidth slotBase lane : Nat) : Option (SparseForm logicalWidth) := do
  if laneBound : lane < 16 then do
    let lane0 ← block.form? logicalWidth (slotBase + 0)
    let lane1 ← block.form? logicalWidth (slotBase + 1)
    let lane2 ← block.form? logicalWidth (slotBase + 2)
    let lane3 ← block.form? logicalWidth (slotBase + 3)
    let lane4 ← block.form? logicalWidth (slotBase + 4)
    let lane5 ← block.form? logicalWidth (slotBase + 5)
    let lane6 ← block.form? logicalWidth (slotBase + 6)
    let lane7 ← block.form? logicalWidth (slotBase + 7)
    let lane8 ← block.form? logicalWidth (slotBase + 8)
    let lane9 ← block.form? logicalWidth (slotBase + 9)
    let lane10 ← block.form? logicalWidth (slotBase + 10)
    let lane11 ← block.form? logicalWidth (slotBase + 11)
    let lane12 ← block.form? logicalWidth (slotBase + 12)
    let lane13 ← block.form? logicalWidth (slotBase + 13)
    let lane14 ← block.form? logicalWidth (slotBase + 14)
    let lane15 ← block.form? logicalWidth (slotBase + 15)
    pure (SparseLayer.external
      (fixedState16 lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7 lane8 lane9 lane10 lane11 lane12 lane13 lane14 lane15)
      ⟨lane, laneBound⟩)
  else
    none

theorem RetainedBlock.externalForm?_ofSemantic
    {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart : Nat)
    (fits : retainedStart + block.coordinateCount ≤ logicalWidth)
    (slotBase : Nat)
    (slotBound : ∀ lane : Fin 16, slotBase + lane.val < block.slotCount)
    (lane : Fin 16) :
    (RetainedBlock.ofSemantic block retainedStart).externalForm?
        logicalWidth slotBase lane.val =
      some (SparseLayer.external (fun selected : Fin 16 =>
        block.form retainedStart fits
          ⟨slotBase + selected.val, slotBound selected⟩) lane) := by
  have laneEq : ∀ selected (bound : selected < 16),
      (RetainedBlock.ofSemantic block retainedStart).form? logicalWidth
          (slotBase + selected) =
        some (block.form retainedStart fits
          ⟨slotBase + selected, slotBound ⟨selected, bound⟩⟩) :=
    fun selected bound => RetainedBlock.form?_ofSemantic block retainedStart
      fits ⟨slotBase + selected, slotBound ⟨selected, bound⟩⟩
  unfold externalForm?
  rw [dite_eq_left lane.isLt]
  rw [laneEq 0 (by decide), laneEq 1 (by decide), laneEq 2 (by decide), laneEq 3 (by decide), laneEq 4 (by decide), laneEq 5 (by decide), laneEq 6 (by decide), laneEq 7 (by decide), laneEq 8 (by decide), laneEq 9 (by decide), laneEq 10 (by decide), laneEq 11 (by decide), laneEq 12 (by decide), laneEq 13 (by decide), laneEq 14 (by decide), laneEq 15 (by decide)]
  apply congrArg some
  apply congrArg (fun state => SparseLayer.external state lane)
  funext selected
  fin_cases selected <;> rfl

structure SourceRange where
  sourceStart : Nat
  sourceCount : Nat
  retained : RetainedBlock
  slotStart : Nat
deriving Repr, DecidableEq

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
  rw [ite_eq_right (by omega)]

theorem SourceRange.form?_eq_none_of_after (range : SourceRange)
    (logicalWidth source : Nat)
    (after : range.sourceStart + range.sourceCount ≤ source) :
    range.form? logicalWidth source = none := by
  unfold form?
  rw [ite_eq_left (by omega)]
  rw [ite_eq_right (by omega)]

def SourceRange.ofSemantic {sourceWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (retainedStart sourceStart
      sourceCount slotStart : Nat) : SourceRange where
  sourceStart := sourceStart
  sourceCount := sourceCount
  retained := RetainedBlock.ofSemantic block retainedStart
  slotStart := slotStart

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
  rw [ite_eq_left (by omega)]
  rw [show sourceStart + offset.val - sourceStart = offset.val by omega]
  rw [ite_eq_left offset.isLt]
  exact RetainedBlock.form?_ofSemantic block retainedStart fits
    ⟨slotStart + offset.val, by omega⟩

inductive SourceGridMode where
  | direct
  | external16
deriving Repr, DecidableEq

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
              | .external16 =>
                  grid.retained.externalForm? logicalWidth slotBase offset
            else none
          else none
        else none
      else none
    else none
  else none

@[implicit_reducible]
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

@[implicit_reducible]
def SourceGrid.externalOfSemantic {sourceWidth : Nat}
    (block : LowNormBlock.Block sourceWidth)
    (retainedStart sourceStart majorCount majorSourceStride minorCount
      minorSourceStride runCount slotStart majorSlotStride minorSlotStride :
      Nat) : SourceGrid :=
  { ofSemantic block retainedStart sourceStart majorCount majorSourceStride
      minorCount minorSourceStride runCount slotStart majorSlotStride
      minorSlotStride with
    mode := .external16 }

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
  rw [ite_eq_right (by omega)]

theorem SourceGrid.form?_eq_none_of_after (grid : SourceGrid)
    (logicalWidth source : Nat)
    (positive : 0 < grid.majorSourceStride)
    (after : grid.sourceStart +
      grid.majorCount * grid.majorSourceStride ≤ source) :
    grid.form? logicalWidth source = none := by
  unfold form?
  rw [ite_eq_left (by omega), ite_eq_left positive]
  simp only
  rw [ite_eq_right]
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
  rw [ite_eq_left (by omega)]
  rw [show
      grid.sourceStart + major.val * grid.majorSourceStride +
          minor.val * grid.minorSourceStride + offset - grid.sourceStart =
        major.val * grid.majorSourceStride +
          (minor.val * grid.minorSourceStride + offset) by omega]
  rw [ite_eq_left majorPositive]
  rw [stride_div major.val
      (minor.val * grid.minorSourceStride + offset)
      grid.majorSourceStride majorPositive minorCellBound]
  rw [stride_mod major.val
      (minor.val * grid.minorSourceStride + offset)
      grid.majorSourceStride minorCellBound]
  rw [ite_eq_left major.isLt, ite_eq_left minorPositive]
  rw [stride_div minor.val offset grid.minorSourceStride minorPositive
      offsetBound]
  rw [stride_mod minor.val offset grid.minorSourceStride offsetBound]
  rw [ite_eq_left minor.isLt, ite_eq_right (by omega)]

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
  rw [ite_eq_left (by omega)]
  rw [show
      grid.sourceStart + major.val * grid.majorSourceStride +
          minor * grid.minorSourceStride + offset - grid.sourceStart =
        major.val * grid.majorSourceStride +
          (minor * grid.minorSourceStride + offset) by omega]
  rw [ite_eq_left majorPositive]
  rw [stride_div major.val (minor * grid.minorSourceStride + offset)
      grid.majorSourceStride majorPositive minorCellBound]
  rw [stride_mod major.val (minor * grid.minorSourceStride + offset)
      grid.majorSourceStride minorCellBound]
  rw [ite_eq_left major.isLt, ite_eq_left minorPositive]
  rw [stride_div minor offset grid.minorSourceStride minorPositive offsetBound]
  rw [stride_mod minor offset grid.minorSourceStride offsetBound]
  rw [ite_eq_right (by omega)]

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
  rw [ite_eq_left (by omega)]
  rw [show
      sourceStart + major.val * majorSourceStride +
          minor.val * minorSourceStride + offset.val - sourceStart =
        major.val * majorSourceStride +
          (minor.val * minorSourceStride + offset.val) by omega]
  rw [ite_eq_left majorPositive]
  rw [stride_div major.val
      (minor.val * minorSourceStride + offset.val) majorSourceStride
      majorPositive minorCellBound]
  rw [stride_mod major.val
      (minor.val * minorSourceStride + offset.val) majorSourceStride
      minorCellBound]
  rw [ite_eq_left major.isLt, ite_eq_left minorPositive]
  rw [stride_div minor.val offset.val minorSourceStride minorPositive
      offsetBound]
  rw [stride_mod minor.val offset.val minorSourceStride offsetBound]
  rw [ite_eq_left minor.isLt, ite_eq_left offset.isLt]
  exact RetainedBlock.form?_ofSemantic block retainedStart fits
    ⟨slotStart + major.val * majorSlotStride +
      minor.val * minorSlotStride + offset.val, slotBound⟩

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
    (laneBound : offset.val < 16)
    (slotBound : ∀ lane : Fin 16,
      slotStart + major.val * majorSlotStride +
        minor.val * minorSlotStride + lane.val < block.slotCount) :
    (externalOfSemantic block retainedStart sourceStart majorCount
      majorSourceStride minorCount minorSourceStride runCount slotStart
      majorSlotStride minorSlotStride).form? logicalWidth
        (sourceStart + major.val * majorSourceStride +
          minor.val * minorSourceStride + offset.val) =
      some (SparseLayer.external (fun lane : Fin 16 =>
        block.form retainedStart fits
          ⟨slotStart + major.val * majorSlotStride +
            minor.val * minorSlotStride + lane.val, slotBound lane⟩)
        ⟨offset.val, laneBound⟩) := by
  simp only [form?, externalOfSemantic, ofSemantic]
  rw [ite_eq_left (by omega)]
  rw [show
      sourceStart + major.val * majorSourceStride +
          minor.val * minorSourceStride + offset.val - sourceStart =
        major.val * majorSourceStride +
          (minor.val * minorSourceStride + offset.val) by omega]
  rw [ite_eq_left majorPositive]
  rw [stride_div major.val
      (minor.val * minorSourceStride + offset.val) majorSourceStride
      majorPositive minorCellBound]
  rw [stride_mod major.val
      (minor.val * minorSourceStride + offset.val) majorSourceStride
      minorCellBound]
  rw [ite_eq_left major.isLt, ite_eq_left minorPositive]
  rw [stride_div minor.val offset.val minorSourceStride minorPositive
      offsetBound]
  rw [stride_mod minor.val offset.val minorSourceStride offsetBound]
  rw [ite_eq_left minor.isLt, ite_eq_left offset.isLt]
  exact RetainedBlock.externalForm?_ofSemantic block retainedStart fits
    (slotStart + major.val * majorSlotStride +
      minor.val * minorSlotStride) slotBound ⟨offset.val, laneBound⟩

structure SourceSubstitution where
  ranges : List SourceRange
  grids : List SourceGrid := []
deriving Repr, DecidableEq

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

structure IndexRange where
  start : Nat
  count : Nat
deriving Repr, DecidableEq

def IndexRange.endExclusive (range : IndexRange) : Nat :=
  range.start + range.count

inductive IndexSchedule where
  | rangeList (ranges : List IndexRange)
  | indexTable (indices : Array Nat)
deriving Repr, DecidableEq

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

def IndexSchedule.valid (schedule : IndexSchedule) (limit : Nat) : Bool :=
  match schedule with
  | .rangeList ranges => validIndexRanges limit 0 ranges
  | .indexTable indices => validIndexTable limit indices.toList

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

def IndexRange.indices (range : IndexRange) : List Nat :=
  (List.range range.count).map fun offset => range.start + offset

theorem IndexRange.indices_eq_range' (range : IndexRange) :
    range.indices = List.range' range.start range.count := by
  exact List.range'_eq_map_range.symm

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

theorem IndexSchedule.rangeList_indices_nodup
    (ranges : List IndexRange) (limit : Nat)
    (valid : (IndexSchedule.rangeList ranges).valid limit = true) :
    (IndexSchedule.rangeList ranges).indices.Nodup := by
  exact (validIndexRanges_indices limit 0 ranges valid).1

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
          · rw [ite_eq_left inside, List.getElem?_append_left]
            · simp [IndexRange.indices, inside]
            · simpa using inside
          · rw [ite_eq_right inside, List.getElem?_append_right]
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

end NightstreamFPrime.Layout.MatrixProgram
