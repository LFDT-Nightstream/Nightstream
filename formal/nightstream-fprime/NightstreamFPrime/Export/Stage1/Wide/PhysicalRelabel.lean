import Mathlib.Data.List.Forall2
import NightstreamFPrime.Layout.MatrixProgram.SourceProjection
import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Layout.Stage1.Wide.SourceOrder
import NightstreamFPrime.Layout.Stage1.Wide.PiRLCStarts

/-! Checked relocation of the physical prefix around the replaced sampler.
Old sampler rows and columns have no image. The final 54 digit words of each
scalar map to the new checked digit bridge; product and suffix order is kept. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalRelabel

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open Layout.Stage1

def columnRanges : List MatrixProgram.SourceProjectionRange :=
  [⟨0, 0, Spartan.sourceToSpartan PiRLCStarts.phaseLogicalStart⟩,
   ⟨Spartan.sourceToSpartan PiRLCStarts.commitmentLogicalStart,
     Layout.Stage1.Wide.SourceOrder.column Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart,
     PiRLCStarts.outputLogicalStart - PiRLCStarts.commitmentLogicalStart⟩,
   ⟨Spartan.sourceToSpartan PiRLCStarts.commitmentFreshStart,
     Layout.Stage1.Wide.SourceOrder.column Layout.Stage1.Wide.PiRLCStarts.commitmentFreshStart,
     Spartan.spartanColumnCount - Spartan.sourceToSpartan PiRLCStarts.commitmentFreshStart⟩] ++
  (List.range 17).map fun source =>
    ⟨Spartan.sourceToSpartan (PiRLCStarts.challengeWordStart source),
      Layout.Stage1.Wide.SourceOrder.column (Layout.Stage1.Wide.PiRLCStarts.challengeWordStart source), 54⟩

def projection : MatrixProgram.SourceProjection := .mapped columnRanges

/-- Ordinary archived rows use only the common source regions. The final
digit-word bridge is reserved for compact ring-product invocations. -/
def ordinaryProjection : MatrixProgram.SourceProjection := .mapped (columnRanges.take 3)

def ordinaryColumn (value : Nat) : Except String Nat :=
  match ordinaryProjection.column? value with
  | some mapped => .ok mapped
  | none => .error s!"removed ordinary source column {value}"

def column (value : Nat) : Except String Nat :=
  match projection.column? value with
  | some mapped => .ok mapped
  | none => .error s!"removed physical source column {value}"

def row (value : Nat) : Except String Nat :=
  if value < PiRLCStarts.phaseRowStart then .ok value
  else if PiRLCStarts.commitmentRowStart ≤ value then
    .ok (value - (PiRLCStarts.commitmentRowStart - Layout.Stage1.Wide.PiRLCStarts.commitmentRowStart))
  else .error s!"removed sampler source row {value}"

structure Map where
  column : Nat → Except String Nat
  row : Nat → Except String Nat

def prefixMap : Map := ⟨column, row⟩
def ordinaryMap : Map := ⟨ordinaryColumn, row⟩

theorem mapM_pairs {α β : Type} (f : α → Except String β)
    (before : List α) (after : List β) (mapped : before.mapM f = .ok after) :
    List.Forall₂ (fun a b => f a = .ok b) before after := by
  induction before generalizing after with
  | nil => simp only [List.mapM_nil] at mapped; cases mapped; exact .nil
  | cons head tail ih =>
    cases first : f head with
    | error message => simp [List.mapM_cons, first, Bind.bind, Except.bind] at mapped
    | ok value =>
      cases rest : tail.mapM f with
      | error message => simp [List.mapM_cons, first, rest, Bind.bind, Except.bind] at mapped
      | ok suffix =>
        simp [List.mapM_cons, first, rest, Bind.bind, Except.bind, Pure.pure, Except.pure] at mapped
        subst after
        exact .cons first (ih suffix rest)

namespace Map

variable (mapping : Map)

def expression (mapping : Map) : Expr → Except String Expr
  | .var index => return .var (← mapping.column index)
  | .const value => return .const value
  | .add left right => return .add (← expression mapping left) (← expression mapping right)
  | .mul left right => return .mul (← expression mapping left) (← expression mapping right)

def pullback (target : Env) : Env := fun source =>
  match mapping.column source with
  | .ok column => target column
  | .error _ => 0

/-- Successful relocation preserves every expression value. An unmapped
source cannot become an accepted constant-column read. -/
theorem expression_eval (value moved : Expr) (emitted : expression mapping value = .ok moved)
    (target : Env) : moved.eval target = value.eval (pullback mapping target) := by
  induction value generalizing moved with
  | var source =>
    cases mapped : mapping.column source with
    | error message => simp [Bind.bind, Except.bind, expression, mapped] at emitted
    | ok column =>
      simp [Bind.bind, Pure.pure, Except.bind, Except.pure, expression, mapped] at emitted
      subst moved
      simp only [Expr.eval, pullback, mapped]
  | const value =>
    simp [Pure.pure, Except.pure, expression] at emitted
    subst moved
    rfl
  | add left right leftIH rightIH =>
    cases leftMoved : expression mapping left with
    | error message => simp [Bind.bind, Except.bind, expression, leftMoved] at emitted
    | ok leftResult =>
      cases rightMoved : expression mapping right with
      | error message => simp [Bind.bind, Except.bind, expression, leftMoved, rightMoved] at emitted
      | ok rightResult =>
        simp [Bind.bind, Pure.pure, Except.bind, Except.pure, expression, leftMoved, rightMoved] at emitted
        subst moved
        simp only [Expr.eval, leftIH _ leftMoved, rightIH _ rightMoved]
  | mul left right leftIH rightIH =>
    cases leftMoved : expression mapping left with
    | error message => simp [Bind.bind, Except.bind, expression, leftMoved] at emitted
    | ok leftResult =>
      cases rightMoved : expression mapping right with
      | error message => simp [Bind.bind, Except.bind, expression, leftMoved, rightMoved] at emitted
      | ok rightResult =>
        simp [Bind.bind, Pure.pure, Except.bind, Except.pure, expression, leftMoved, rightMoved] at emitted
        subst moved
        simp only [Expr.eval, leftIH _ leftMoved, rightIH _ rightMoved]

def hint : Hint → Except String Hint
  | .bit source index => return .bit (← expression mapping source) index
  | .inverseOrZero source => return .inverseOrZero (← expression mapping source)
  | .quotientFive source => return .quotientFive (← expression mapping source)
  | .remainderFive source => return .remainderFive (← expression mapping source)

theorem hint_eval (value moved : Hint) (emitted : hint mapping value = .ok moved)
    (target : Env) : moved.eval target = value.eval (pullback mapping target) := by
  cases value with
  | bit source index =>
    cases result : expression mapping source with
    | error message => simp [hint, result] at emitted
    | ok expression =>
      simp [hint, result] at emitted
      subst moved
      simp only [Hint.eval, expression_eval mapping source expression result target]
  | inverseOrZero source =>
    cases result : expression mapping source with
    | error message => simp [hint, result] at emitted
    | ok expression =>
      simp [hint, result] at emitted
      subst moved
      simp only [Hint.eval, expression_eval mapping source expression result target]
  | quotientFive source =>
    cases result : expression mapping source with
    | error message => simp [hint, result] at emitted
    | ok expression =>
      simp [hint, result] at emitted
      subst moved
      simp only [Hint.eval, expression_eval mapping source expression result target]
  | remainderFive source =>
    cases result : expression mapping source with
    | error message => simp [hint, result] at emitted
    | ok expression =>
      simp [hint, result] at emitted
      subst moved
      simp only [Hint.eval, expression_eval mapping source expression result target]

def batch (value : WitnessBatch) : Except String WitnessBatch := do
  return {
    start := ← mapping.column value.start
    recipes := ← value.recipes.mapM (expression mapping)
    hints := ← value.hints.mapM (hint mapping) }

def term (value : SparseTerm) : Except String SparseTerm := do
  return { value with column := ← mapping.column value.column }

def combination (value : SparseCombination) : Except String SparseCombination := do
  return { value with terms := ← value.terms.mapM (term mapping) }

/-- Total notation for semantic equations after successful emission. The
emitter itself always uses the checked `column` and `row` functions. -/
def columnValue (source : Nat) : Nat := (mapping.column source).toOption.getD 0

private theorem term_pair (before after : SparseTerm) (emitted : term mapping before = .ok after) :
    (after.column, fieldValue after.coefficient) =
      (columnValue mapping before.column, fieldValue before.coefficient) := by
  cases mapped : mapping.column before.column with
  | error message => simp [term, mapped] at emitted
  | ok target =>
    simp [term, mapped] at emitted
    subst after
    simp [columnValue, mapped, Except.toOption]

theorem combination_toR1CS (before after : SparseCombination)
    (emitted : combination mapping before = .ok after) :
    after.toR1CS = R1CS.mapCombinationColumns (columnValue mapping) before.toR1CS := by
  cases mapped : before.terms.mapM (term mapping) with
  | error message => simp [combination, mapped] at emitted
  | ok terms =>
    simp [combination, mapped] at emitted
    subst after
    have pairs := mapM_pairs (term mapping) before.terms terms mapped
    have same : terms.map (fun item => (item.column, fieldValue item.coefficient)) =
        before.terms.map (fun item => (columnValue mapping item.column, fieldValue item.coefficient)) := by
      clear mapped
      generalize before.terms = original at pairs ⊢
      induction pairs with
      | nil => rfl
      | @cons a b before after emitted pairs ih =>
        simp only [List.map_cons, term_pair mapping a b emitted, ih]
    simp only [SparseCombination.toR1CS, R1CS.mapCombinationColumns, List.map_map,
      Function.comp_def, same]

def Mapped (source : Nat) : Prop := ∃ target, mapping.column source = .ok target

private theorem term_supported (before after : SparseTerm) (emitted : term mapping before = .ok after) :
    Mapped mapping before.column := by
  cases read : mapping.column before.column with
  | error message => simp [term, read] at emitted
  | ok target => exact ⟨target, read⟩

theorem combination_supported (before after : SparseCombination)
    (emitted : combination mapping before = .ok after) :
    before.toR1CS.VarsSatisfy (Mapped mapping) := by
  cases mapped : before.terms.mapM (term mapping) with
  | error message => simp [combination, mapped] at emitted
  | ok terms =>
    have pairs := mapM_pairs (term mapping) before.terms terms mapped
    have support : ∀ item ∈ before.terms, Mapped mapping item.column := by
      clear mapped emitted
      generalize before.terms = original at pairs ⊢
      induction pairs with
      | nil => simp
      | @cons a b before after pair pairs ih =>
        intro item member
        rcases List.mem_cons.mp member with equal | member
        · subst item
          exact term_supported mapping a b pair
        · exact ih item member
    intro item member
    change item ∈ before.terms.map (fun term => (term.column, fieldValue term.coefficient)) at member
    obtain ⟨original, sourceMember, same⟩ := List.mem_map.mp member
    rw [← same]
    exact support original sourceMember

def instruction (value : WitnessInstruction) : Except String WitnessInstruction := do
  return {
    rowIndex := ← mapping.row value.rowIndex
    target := ← mapping.column value.target
    a := ← combination mapping value.a
    b := ← combination mapping value.b }

def assertion (value : SparseRow) : Except String SparseRow := do
  return {
    rowIndex := ← mapping.row value.rowIndex
    a := ← combination mapping value.a
    b := ← combination mapping value.b
    c := ← combination mapping value.c }

theorem instruction_correct (before after : WitnessInstruction)
    (emitted : instruction mapping before = .ok after) :
    mapping.row before.rowIndex = .ok after.rowIndex ∧
      after.toR1CS = R1CS.mapRowColumns (columnValue mapping) before.toR1CS ∧
      before.toR1CS.VarsSatisfy (Mapped mapping) := by
  cases rowResult : mapping.row before.rowIndex with
  | error message => simp [instruction, rowResult, Bind.bind, Except.bind] at emitted
  | ok row =>
    cases targetResult : mapping.column before.target with
    | error message => simp [instruction, rowResult, targetResult, Bind.bind, Except.bind] at emitted
    | ok target =>
      cases aResult : combination mapping before.a with
      | error message => simp [instruction, rowResult, targetResult, aResult, Bind.bind, Except.bind] at emitted
      | ok a =>
        cases bResult : combination mapping before.b with
        | error message => simp [instruction, rowResult, targetResult, aResult, bResult, Bind.bind, Except.bind] at emitted
        | ok b =>
          simp [instruction, rowResult, targetResult, aResult, bResult, Bind.bind, Except.bind,
            Pure.pure, Except.pure] at emitted
          subst after
          refine ⟨rfl, ?_, ?_⟩
          · simp only [WitnessInstruction.toR1CS, R1CS.mapRowColumns,
              combination_toR1CS mapping before.a a aResult,
              combination_toR1CS mapping before.b b bResult,
              R1CS.mapCombinationColumns_ofVar]
            simp only [columnValue, targetResult, Except.toOption, Option.getD_some]
          · refine ⟨combination_supported mapping before.a a aResult,
              combination_supported mapping before.b b bResult, ?_⟩
            intro item member
            change item ∈ [(before.target, 1)] at member
            rcases List.mem_singleton.mp member with rfl
            exact ⟨target, targetResult⟩

theorem assertion_correct (before after : SparseRow)
    (emitted : assertion mapping before = .ok after) :
    mapping.row before.rowIndex = .ok after.rowIndex ∧
      after.toR1CS = R1CS.mapRowColumns (columnValue mapping) before.toR1CS ∧
      before.toR1CS.VarsSatisfy (Mapped mapping) := by
  cases rowResult : mapping.row before.rowIndex with
  | error message => simp [assertion, rowResult, Bind.bind, Except.bind] at emitted
  | ok row =>
    cases aResult : combination mapping before.a with
    | error message => simp [assertion, rowResult, aResult, Bind.bind, Except.bind] at emitted
    | ok a =>
      cases bResult : combination mapping before.b with
      | error message => simp [assertion, rowResult, aResult, bResult, Bind.bind, Except.bind] at emitted
      | ok b =>
        cases cResult : combination mapping before.c with
        | error message => simp [assertion, rowResult, aResult, bResult, cResult, Bind.bind, Except.bind] at emitted
        | ok c =>
          simp [assertion, rowResult, aResult, bResult, cResult, Bind.bind, Except.bind,
            Pure.pure, Except.pure] at emitted
          subst after
          refine ⟨rfl, ?_, combination_supported mapping before.a a aResult,
            combination_supported mapping before.b b bResult,
            combination_supported mapping before.c c cResult⟩
          simp only [SparseRow.toR1CS, R1CS.mapRowColumns,
            combination_toR1CS mapping before.a a aResult,
            combination_toR1CS mapping before.b b bResult,
            combination_toR1CS mapping before.c c cResult]

def compiledRow : Rows.CompiledRow → Except String Rows.CompiledRow
  | .witness value => return .witness (← instruction mapping value)
  | .assertion value => return .assertion (← assertion mapping value)

theorem compiledRow_correct (before after : Rows.CompiledRow)
    (emitted : compiledRow mapping before = .ok after) :
    mapping.row before.rowIndex = .ok after.rowIndex ∧
      after.toR1CS = R1CS.mapRowColumns (columnValue mapping) before.toR1CS ∧
      before.toR1CS.VarsSatisfy (Mapped mapping) := by
  cases before with
  | witness value =>
    cases moved : instruction mapping value with
    | error message => simp [compiledRow, moved] at emitted
    | ok result =>
      simp [compiledRow, moved] at emitted
      subst after
      exact instruction_correct mapping value result moved
  | assertion value =>
    cases moved : assertion mapping value with
    | error message => simp [compiledRow, moved] at emitted
    | ok result =>
      simp [compiledRow, moved] at emitted
      subst after
      exact assertion_correct mapping value result moved

def permutation (value : PermutationInvocation) : Except String PermutationInvocation := do
  return { value with
    rowStart := ← mapping.row value.rowStart
    witnessStart := ← mapping.column value.witnessStart
    inputs := ← value.inputs.mapM (combination mapping) }

def inputRange (value : CompactInputRange) : Except String CompactInputRange := do
  let first ← mapping.column value.columnStart
  for index in List.range value.inputCount do
    let mapped ← mapping.column (value.columnStart + index * value.columnStride)
    unless mapped = first + index * value.columnStride do
      throw "non-affine compact input relocation"
  return { value with columnStart := first }

def compact (value : CompactRowInvocation) : Except String CompactRowInvocation := do
  return { value with
    rowStart := ← mapping.row value.rowStart
    localStart := ← mapping.column value.localStart
    inputRanges := ← value.inputRanges.mapM (inputRange mapping) }

def chain (value : HashChain) : Except String HashChain := do
  return { value with
    rowStart := ← mapping.row value.rowStart
    inputStart := ← mapping.column value.inputStart
    witnessStart := ← mapping.column value.witnessStart
    digestStart := ← mapping.column value.digestStart }

/-- Segment endpoints describe ownership of the replaced interval. They do
not assert a column-by-column correspondence inside the old sampler. -/
def segment (value : Segment) : Except String Segment := do
  let start ← mapping.column value.start
  let stop ← mapping.column (value.start + value.length - 1)
  unless 0 < value.length && start ≤ stop do throw "invalid relocated segment"
  return { value with start := start, length := stop + 1 - start }

end Map

end NightstreamFPrime.Export.Stage1.Wide.PhysicalRelabel
