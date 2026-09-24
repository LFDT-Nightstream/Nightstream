import NightstreamFPrime.Layout.PiRlcWideSampler.MatrixSource
import NightstreamFPrime.Layout.PiRlcWideSampler.BatchPlan
import NightstreamFPrime.Layout.MatrixProgram.Program

/-! The ordinary matrix block for the checked range rows. Its source rows
come from the certified compiler and its source map omits every helper. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.MatrixRows

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open ProductionRelation MatrixProgram

def block {columns : Nat} (compiled : RangePlan.Compiled) (oneColumn : Fin columns)
    (state : RetainedBlock) (slot start sourceStart : Nat) : Ordinary.Block where
  rows := .rangeList [⟨sourceStart, compiled.rows.length⟩]
  oneColumn := oneColumn.val
  substitution := MatrixSource.substitution state slot start

def program {columns : Nat} (compiled : RangePlan.Compiled) (oneColumn : Fin columns)
    (state : RetainedBlock) (slot start sourceStart : Nat) : MatrixProgram.Program where
  blocks := [.ordinary (block compiled oneColumn state slot start sourceStart)]

theorem rowCount {columns : Nat} (compiled : RangePlan.Compiled) (oneColumn : Fin columns)
    (state : RetainedBlock) (slot start sourceStart : Nat) :
    (program compiled oneColumn state slot start sourceStart).rowCount = 681 := by
  change compiled.rows.length + 0 = 681
  rw [compiled.rowCount]

def inputs {columns start : Nat} (compiled : RangePlan.Compiled) (oneColumn : Fin columns)
    (fits : Retained.Fits columns start) (input : Fin 4 → SparseForm columns) : compiled.program.Inputs columns where
  oneColumn := oneColumn
  sourceMap := fun _ => Retained.sourceMap start fits input

/-- Exact decoding of every checked range row. Source custody is explicit;
accepted helpers cannot supply a substituted form because no mapping exists. -/
theorem row {columns start : Nat} (compiled : RangePlan.Compiled) (oneColumn : Fin columns)
    (fits : Retained.Fits columns start) (state : RetainedBlock) (slot sourceStart : Nat)
    (input : Fin 4 → SparseForm columns)
    (loads : ∀ lane : Fin 4, (MatrixSource.inputGrid state slot).form? columns lane.val = some (input lane))
    (sourceRow : Nat → Option R1CS.Row)
    (custody : ∀ index : Fin compiled.rows.length, sourceRow (sourceStart + index.val) = some (compiled.rows.get index))
    (index : Fin compiled.rows.length) :
    (program compiled oneColumn state slot start sourceStart).row? columns sourceRow index.val =
      some (((compiled.program.compile (inputs compiled oneColumn fits input)).toPlan).forms index) := by
  let row := compiled.rows.get index
  have member : row ∈ compiled.rows := List.get_mem _ _
  have bound := compiled.bounded row member
  have active := compiled.active row member
  let sourceMap := Retained.sourceMap start fits input
  have decoded := Ordinary.Block.row?_eq_compileRow
    (block compiled oneColumn state slot start sourceStart) sourceRow index.val (sourceStart + index.val)
    row (by simp [block, IndexSchedule.index?, IndexSchedule.index?.select, index.isLt])
    row (custody index) (by simp only [block, SourceProjection.identity_row?]) sourceMap oneColumn rfl bound
    (by intro term present termBound; exact MatrixSource.form_eq fits state slot input loads ⟨term.1, termBound⟩ (active.1 term present))
    (by intro term present termBound; exact MatrixSource.form_eq fits state slot input loads ⟨term.1, termBound⟩ (active.2.1 term present))
    (by intro term present termBound; exact MatrixSource.form_eq fits state slot input loads ⟨term.1, termBound⟩ (active.2.2 term present))
  rw [program, MatrixProgram.Program.singleton_row?]
  have rowBound : index.val < (MatrixProgram.Block.ordinary (block compiled oneColumn state slot start sourceStart)).rowCount := by
    exact index.isLt
  rw [if_pos rowBound]
  change (do
    let forms ← (block compiled oneColumn state slot start sourceStart).row? columns sourceRow index.val
    pure forms.meaningfulForm) = _
  rw [decoded]
  rfl

def poseidonBlock {columns : Nat} (interface : BatchPlan.Interface columns) : RetainedBlock :=
  RetainedBlock.ofSemantic BatchPlan.poseidonBlock interface.start

def poseidonSlot (source : Fin 17) : Nat := 172 * source.val + 78

theorem poseidonFits {columns : Nat} (interface : BatchPlan.Interface columns) :
    interface.start + BatchPlan.poseidonBlock.coordinateCount ≤ columns := by
  have bound := interface.fits
  rw [BatchPlan.coordinateCount_eq] at bound
  change interface.start + 119884 ≤ columns
  omega

theorem input_forms {columns : Nat} (interface : BatchPlan.Interface columns)
    (source : Fin 17) (lane : Fin 4) :
    (MatrixSource.inputGrid (poseidonBlock interface) (poseidonSlot source)).form? columns lane.val =
      some (BatchPlan.outputState interface ⟨source.val * 2, by omega⟩ ⟨lane.val, by omega⟩) := by
  have slots : ∀ position : Fin 8,
      poseidonSlot source + (0 : Fin 1).val * 0 + (0 : Fin 1).val * 0 + position.val <
        BatchPlan.poseidonBlock.slotCount := by
    intro position
    have hs : source.val < 17 := source.isLt
    have hl : position.val < 8 := position.isLt
    change 172 * source.val + 78 + 0 * 0 + 0 * 0 + position.val < 2924
    omega
  have loaded := SourceGrid.form?_externalOfSemantic BatchPlan.poseidonBlock interface.start
    0 1 4 1 4 4 (poseidonSlot source) 0 0 (poseidonFits interface)
    (by decide) (by decide) (0 : Fin 1) (0 : Fin 1) lane
    (by omega) lane.isLt (by omega) slots
  simp only [Fin.val_zero, Nat.zero_mul, Nat.mul_zero, Nat.zero_add, Nat.add_zero] at loaded
  refine loaded.trans ?_
  apply congrArg some
  dsimp only [BatchPlan.outputState]
  apply congrArg (fun state => SparseLayer.external state ⟨lane.val, by omega⟩)
  funext position
  dsimp only [BatchPlan.sbox]
  apply congrArg (BatchPlan.poseidonBlock.form interface.start _)
  apply Fin.ext
  simp only [PoseidonRetainedSlots.finalRow_val, poseidonSlot]
  omega

def rangeProgram {columns : Nat} (compiled : RangePlan.Compiled) (interface : BatchPlan.Interface columns)
    (source : Fin 17) (sourceStart : Nat) : MatrixProgram.Program :=
  program compiled interface.oneColumn (poseidonBlock interface) (poseidonSlot source)
    (BatchPlan.rangeStart interface source) sourceStart

/-- The range consumer uses the first permutation of each scalar's pair,
exactly as the candidate semantic plan does. -/
theorem rangeRow {columns : Nat} (compiled : RangePlan.Compiled) (interface : BatchPlan.Interface columns)
    (source : Fin 17) (sourceStart : Nat) (sourceRow : Nat → Option R1CS.Row)
    (custody : ∀ index : Fin compiled.rows.length, sourceRow (sourceStart + index.val) = some (compiled.rows.get index))
    (index : Fin compiled.rows.length) :
    (rangeProgram compiled interface source sourceStart).row? columns sourceRow index.val =
      some ((BatchPlan.rangePlan compiled interface source).forms index) := by
  exact row compiled interface.oneColumn (BatchPlan.rangeFits interface source)
    (poseidonBlock interface) (poseidonSlot source) sourceStart _ (input_forms interface source) sourceRow custody index

end NightstreamFPrime.Layout.PiRlcWideSampler.MatrixRows
