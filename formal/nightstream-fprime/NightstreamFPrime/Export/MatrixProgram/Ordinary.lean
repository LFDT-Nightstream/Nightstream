import NightstreamFPrime.Export.MatrixProgram.SourceProjection
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan

/-!
Owns the generic executable interpreter for ordinary source R1CS rows in a
compact sparse 14-matrix program. Source rows come from the package's existing
canonical row stream. The interpreter only applies the package-carried sparse
source substitution and places the result in the fixed ordinary matrix ports.

This module does not select source rows, substitutions, or Stage 1 order.
-/

namespace NightstreamFPrime.Export.MatrixProgram.Ordinary

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

/-- Apply one fail-closed package substitution to an ordered source-term
list. -/
def compileTerms? (substitution : SourceSubstitution) (logicalWidth : Nat) :
    List (Nat × F) → Option (SparseForm logicalWidth)
  | [] => some .empty
  | term :: rest => do
      let head ← substitution.form? logicalWidth term.1
      let tail ← compileTerms? substitution logicalWidth rest
      pure (SparseForm.add (SparseForm.scale term.2 head) tail)

/-- Apply the package substitution to one affine source combination. -/
def compileCombination? (substitution : SourceSubstitution)
    {logicalWidth : Nat} (oneColumn : Fin logicalWidth)
    (combination : R1CS.LinearCombination) : Option (SparseForm logicalWidth) := do
  let terms ← compileTerms? substitution logicalWidth combination.terms
  pure (SparseForm.add
    (SparseForm.singleton oneColumn combination.constant) terms)

/-- Fail-closed ordinary-row compilation. -/
def compileRow? (substitution : SourceSubstitution) (logicalWidth oneColumn : Nat)
    (row : R1CS.Row) : Option (OrdinaryRow.Forms logicalWidth) :=
  if oneBound : oneColumn < logicalWidth then do
    let a ← compileCombination? substitution ⟨oneColumn, oneBound⟩ row.a
    let b ← compileCombination? substitution ⟨oneColumn, oneBound⟩ row.b
    let c ← compileCombination? substitution ⟨oneColumn, oneBound⟩ row.c
    pure {
      selector := SparseForm.singleton ⟨oneColumn, oneBound⟩ 1
      a := a
      b := b
      c := c }
  else
    none

/-- Complete package operands for one ordinary matrix block. -/
structure Block where
  rows : IndexSchedule
  oneColumn : Nat
  substitution : SourceSubstitution
  projection : SourceProjection := .identity
deriving Repr, DecidableEq

def Block.format : Format Block where
  encode := fun block => .array [
    IndexSchedule.format.encode block.rows,
    .atom block.oneColumn,
    SourceSubstitution.format.encode block.substitution,
    SourceProjection.format.encode block.projection]
  decode
    | .array [rows, .atom oneColumn, substitution, projection] => do
      pure ⟨← IndexSchedule.format.decode rows, oneColumn,
        ← SourceSubstitution.format.decode substitution,
        ← SourceProjection.format.decode projection⟩
    | _ => .error "invalid ordinary matrix block"
  decode_encode := by
    intro block
    cases block
    simp [IndexSchedule.format.decode_encode,
      SourceSubstitution.format.decode_encode,
      SourceProjection.format.decode_encode]
    rfl

def Block.rowCount (block : Block) : Nat :=
  block.rows.count

/-- Read and compile one selected package row without materializing the
block. -/
def Block.row? (block : Block) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat) :
    Option (OrdinaryRow.Forms logicalWidth) := do
  let sourceIndex ← block.rows.index? ordinal
  let packageRow ← sourceRow sourceIndex
  let row ← block.projection.row? packageRow
  compileRow? block.substitution logicalWidth block.oneColumn row

/-- Agreement is needed only on source columns that occur in this term
list. -/
def AgreesOnTerms {sourceWidth logicalWidth : Nat}
    (substitution : SourceSubstitution)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (terms : List (Nat × F)) : Prop :=
  ∀ term, term ∈ terms → ∀ bounded : term.1 < sourceWidth,
    substitution.form? logicalWidth term.1 =
      some (sourceMap.form ⟨term.1, bounded⟩)

/-- Exact term-list agreement with the proof-oriented source compiler. -/
theorem compileTerms?_eq_compileTerms
    {sourceWidth logicalWidth : Nat}
    (substitution : SourceSubstitution)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (terms : List (Nat × F))
    (bounded : ∀ term ∈ terms, term.1 < sourceWidth)
    (agrees : AgreesOnTerms substitution sourceMap terms) :
    compileTerms? substitution logicalWidth terms =
      some (SourceCompiler.compileTerms sourceMap terms bounded) := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      have headBound : term.1 < sourceWidth := bounded term (by simp)
      have restBound : ∀ candidate ∈ rest, candidate.1 < sourceWidth :=
        fun candidate member => bounded candidate (by simp [member])
      have restAgrees : AgreesOnTerms substitution sourceMap rest := by
        intro candidate member candidateBound
        exact agrees candidate (by simp [member]) candidateBound
      rw [compileTerms?, agrees term (by simp) headBound,
        inductionHypothesis restBound restAgrees]
      rfl

/-- Exact affine-combination agreement with the proof-oriented source
compiler. -/
theorem compileCombination?_eq_compileCombination
    {sourceWidth logicalWidth : Nat}
    (substitution : SourceSubstitution)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded sourceWidth combination)
    (agrees : AgreesOnTerms substitution sourceMap combination.terms) :
    compileCombination? substitution oneColumn combination =
      some (SourceCompiler.compileCombination sourceMap oneColumn combination
        bounded) := by
  unfold compileCombination?
  rw [compileTerms?_eq_compileTerms substitution sourceMap combination.terms
    bounded agrees]
  rfl

/-- Exact complete-row agreement with the canonical ordinary selective
compiler. -/
theorem compileRow?_eq_compileRow
    {sourceWidth logicalWidth : Nat}
    (substitution : SourceSubstitution)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (row : R1CS.Row)
    (bounded : SourceCompiler.RowBounded sourceWidth row)
    (agreesA : AgreesOnTerms substitution sourceMap row.a.terms)
    (agreesB : AgreesOnTerms substitution sourceMap row.b.terms)
    (agreesC : AgreesOnTerms substitution sourceMap row.c.terms) :
    compileRow? substitution logicalWidth oneColumn.val row =
      some (SourceCompiler.compileRow sourceMap oneColumn row bounded) := by
  unfold compileRow?
  rw [dif_pos oneColumn.isLt]
  rw [compileCombination?_eq_compileCombination substitution sourceMap
    oneColumn row.a bounded.1 agreesA]
  rw [compileCombination?_eq_compileCombination substitution sourceMap
    oneColumn row.b bounded.2.1 agreesB]
  rw [compileCombination?_eq_compileCombination substitution sourceMap
    oneColumn row.c bounded.2.2 agreesC]
  rfl

/-- An ordinary wire block that selects this source row and carries this
substitution returns the exact canonical compiled forms. -/
theorem Block.row?_eq_compileRow
    {sourceWidth logicalWidth : Nat}
    (block : Block)
    (sourceRow : Nat → Option R1CS.Row) (ordinal sourceIndex : Nat)
    (row : R1CS.Row)
    (selected : block.rows.index? ordinal = some sourceIndex)
    (packageRow : R1CS.Row)
    (loaded : sourceRow sourceIndex = some packageRow)
    (projected : block.projection.row? packageRow = some row)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (oneEqual : block.oneColumn = oneColumn.val)
    (bounded : SourceCompiler.RowBounded sourceWidth row)
    (agreesA : AgreesOnTerms block.substitution sourceMap row.a.terms)
    (agreesB : AgreesOnTerms block.substitution sourceMap row.b.terms)
    (agreesC : AgreesOnTerms block.substitution sourceMap row.c.terms) :
    block.row? logicalWidth sourceRow ordinal =
      some (SourceCompiler.compileRow sourceMap oneColumn row bounded) := by
  unfold Block.row?
  rw [selected]
  change (do
    let selectedPackageRow ← sourceRow sourceIndex
    let selectedRow ← block.projection.row? selectedPackageRow
    compileRow? block.substitution logicalWidth block.oneColumn selectedRow) = _
  rw [loaded]
  change (do
    let selectedRow ← block.projection.row? packageRow
    compileRow? block.substitution logicalWidth block.oneColumn selectedRow) = _
  rw [projected]
  rw [oneEqual]
  exact compileRow?_eq_compileRow block.substitution sourceMap oneColumn row
    bounded agreesA agreesB agreesC

end NightstreamFPrime.Export.MatrixProgram.Ordinary
