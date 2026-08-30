import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns the Lean-authored column projection applied to one package source row
before ordinary selective compilation. A projection is either the identity or
a finite set of disjoint affine ranges. Lookup fails on missing or overlapping
ranges.

This module does not select Stage 1 ranges or source rows.
-/

namespace NightstreamFPrime.Export.MatrixProgram

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout

/-- One contiguous package-column interval and its canonical source start. -/
structure SourceProjectionRange where
  packageStart : Nat
  sourceStart : Nat
  count : Nat
deriving Repr, DecidableEq

def SourceProjectionRange.format : Format SourceProjectionRange where
  encode := fun range => .array [
    .atom range.packageStart,
    .atom range.sourceStart,
    .atom range.count]
  decode
    | .array [.atom packageStart, .atom sourceStart, .atom count] =>
        .ok ⟨packageStart, sourceStart, count⟩
    | _ => .error "invalid matrix source projection range"
  decode_encode := by
    intro range
    cases range
    rfl

/-- Project one package column through one affine range. -/
def SourceProjectionRange.column? (range : SourceProjectionRange)
    (column : Nat) : Option Nat :=
  if range.packageStart ≤ column then
    let offset := column - range.packageStart
    if offset < range.count then some (range.sourceStart + offset) else none
  else
    none

theorem SourceProjectionRange.column?_at (range : SourceProjectionRange)
    (offset : Fin range.count) :
    range.column? (range.packageStart + offset.val) =
      some (range.sourceStart + offset.val) := by
  unfold column?
  rw [if_pos (by omega)]
  rw [show range.packageStart + offset.val - range.packageStart =
      offset.val by omega]
  rw [if_pos offset.isLt]

theorem SourceProjectionRange.column?_eq_none_of_before
    (range : SourceProjectionRange) (column : Nat)
    (before : column < range.packageStart) :
    range.column? column = none := by
  unfold column?
  rw [if_neg (by omega)]

theorem SourceProjectionRange.column?_eq_none_of_after
    (range : SourceProjectionRange) (column : Nat)
    (after : range.packageStart + range.count ≤ column) :
    range.column? column = none := by
  unfold column?
  rw [if_pos (by omega), if_neg (by omega)]

/-- Exact package-to-source column projection for one ordinary block. -/
inductive SourceProjection where
  | identity
  | mapped (items : List SourceProjectionRange)
deriving Repr, DecidableEq

def SourceProjection.format : Format SourceProjection where
  encode
    | .identity => .array [.atom 0]
    | .mapped items => .array [
        .atom 1, (list SourceProjectionRange.format).encode items]
  decode
    | .array [.atom 0] => .ok .identity
    | .array [.atom 1, items] => do
        pure (.mapped
          (← (list SourceProjectionRange.format).decode items))
    | _ => .error "invalid matrix source projection"
  decode_encode := by
    intro projection
    cases projection with
    | identity => rfl
    | mapped items =>
        simp [Format.decode_encode]

/-- Project one column. Range projections fail on missing or overlapping
ownership. -/
def SourceProjection.column? (projection : SourceProjection)
    (column : Nat) : Option Nat :=
  match projection with
  | .identity => some column
  | .mapped items =>
      match items.filterMap fun range => range.column? column with
      | [source] => some source
      | _ => none

@[simp] theorem SourceProjection.identity_column? (column : Nat) :
    SourceProjection.identity.column? column = some column := by
  rfl

theorem SourceProjection.mapped_two_column?
    (first second : SourceProjectionRange) (column source : Nat)
    (firstResult : first.column? column = none)
    (secondResult : second.column? column = some source) :
    (SourceProjection.mapped [first, second]).column? column = some source := by
  simp [SourceProjection.column?, firstResult, secondResult]

theorem SourceProjection.mapped_three_column?
    (first second third : SourceProjectionRange) (column source : Nat)
    (firstResult : first.column? column = none)
    (secondResult : second.column? column = none)
    (thirdResult : third.column? column = some source) :
    (SourceProjection.mapped [first, second, third]).column? column =
      some source := by
  simp [SourceProjection.column?, firstResult, secondResult, thirdResult]

theorem SourceProjection.mapped_three_first_column?
    (first second third : SourceProjectionRange) (column source : Nat)
    (firstResult : first.column? column = some source)
    (secondResult : second.column? column = none)
    (thirdResult : third.column? column = none) :
    (SourceProjection.mapped [first, second, third]).column? column =
      some source := by
  simp [SourceProjection.column?, firstResult, secondResult, thirdResult]

theorem SourceProjection.mapped_three_second_column?
    (first second third : SourceProjectionRange) (column source : Nat)
    (firstResult : first.column? column = none)
    (secondResult : second.column? column = some source)
    (thirdResult : third.column? column = none) :
    (SourceProjection.mapped [first, second, third]).column? column =
      some source := by
  simp [SourceProjection.column?, firstResult, secondResult, thirdResult]

private def projectTerms? (projection : SourceProjection) :
    List (Nat × Spec.F) → Option (List (Nat × Spec.F))
  | [] => some []
  | term :: rest => do
      let column ← projection.column? term.1
      let tail ← projectTerms? projection rest
      pure ((column, term.2) :: tail)

/-- Project every variable term of one affine combination. -/
def SourceProjection.combination? (projection : SourceProjection)
    (combination : R1CS.LinearCombination) :
    Option R1CS.LinearCombination := do
  let terms ← projectTerms? projection combination.terms
  pure ⟨combination.constant, terms⟩

/-- Project all three affine combinations of one source row. -/
def SourceProjection.row? (projection : SourceProjection)
    (row : R1CS.Row) : Option R1CS.Row := do
  let a ← projection.combination? row.a
  let b ← projection.combination? row.b
  let c ← projection.combination? row.c
  pure ⟨a, b, c⟩

private theorem projectTerms?_identity (terms : List (Nat × Spec.F)) :
    projectTerms? .identity terms = some terms := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      simp [projectTerms?, inductionHypothesis]

@[simp] theorem SourceProjection.identity_combination?
    (combination : R1CS.LinearCombination) :
    SourceProjection.identity.combination? combination = some combination := by
  cases combination
  simp [SourceProjection.combination?, projectTerms?_identity]

@[simp] theorem SourceProjection.identity_row? (row : R1CS.Row) :
    SourceProjection.identity.row? row = some row := by
  cases row
  simp [SourceProjection.row?]

private theorem projectTerms?_mapColumns
    (projection : SourceProjection) (column : Nat → Nat)
    (sourceWidth : Nat) (terms : List (Nat × Spec.F))
    (bounded : ∀ term ∈ terms, term.1 < sourceWidth)
    (leftInverse : ∀ source : Fin sourceWidth,
      projection.column? (column source.val) = some source.val) :
    projectTerms? projection
        (terms.map fun term => (column term.1, term.2)) = some terms := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      have headBound : term.1 < sourceWidth := bounded term (by simp)
      have tailBound : ∀ candidate ∈ rest, candidate.1 < sourceWidth := by
        intro candidate member
        exact bounded candidate (by simp [member])
      simp [projectTerms?, leftInverse ⟨term.1, headBound⟩,
        inductionHypothesis tailBound]

/-- A projection that is a left inverse of a column renaming recovers the
exact original affine combination. -/
theorem SourceProjection.combination?_mapColumns
    (projection : SourceProjection) (column : Nat → Nat)
    (sourceWidth : Nat) (combination : R1CS.LinearCombination)
    (bounded : combination.VarsBelow sourceWidth)
    (leftInverse : ∀ source : Fin sourceWidth,
      projection.column? (column source.val) = some source.val) :
    projection.combination? (mapCombinationColumns column combination) =
      some combination := by
  cases combination
  simp [SourceProjection.combination?, mapCombinationColumns,
    projectTerms?_mapColumns projection column sourceWidth _ bounded leftInverse]

/-- Row projection exactly cancels a proved package-column renaming. -/
theorem SourceProjection.row?_mapColumns
    (projection : SourceProjection) (column : Nat → Nat)
    (sourceWidth : Nat) (row : R1CS.Row) (bounded : row.VarsBelow sourceWidth)
    (leftInverse : ∀ source : Fin sourceWidth,
      projection.column? (column source.val) = some source.val) :
    projection.row? (mapRowColumns column row) = some row := by
  rcases row with ⟨a, b, c⟩
  rcases bounded with ⟨aBounded, bBounded, cBounded⟩
  unfold SourceProjection.row? mapRowColumns
  rw [SourceProjection.combination?_mapColumns projection column sourceWidth a
      aBounded leftInverse,
    SourceProjection.combination?_mapColumns projection column sourceWidth b
      bBounded leftInverse,
    SourceProjection.combination?_mapColumns projection column sourceWidth c
      cBounded leftInverse]
  rfl

end NightstreamFPrime.Export.MatrixProgram
