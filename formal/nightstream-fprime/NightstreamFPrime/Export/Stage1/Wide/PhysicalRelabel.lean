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

def combination (value : SparseCombination) : Except String SparseCombination := do
  return { value with terms := ← value.terms.mapM fun term => do
    return { term with column := ← mapping.column term.column } }

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
