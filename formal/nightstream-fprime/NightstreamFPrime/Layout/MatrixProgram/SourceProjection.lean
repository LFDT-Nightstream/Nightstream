import NightstreamFPrime.Layout.MatrixProgram
import NightstreamFPrime.Layout.R1CS.ColumnMap
import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Layout.R1CS.Support

/-!
Owns the Lean-authored column projection applied to one package source row
before ordinary selective compilation. A projection is either the identity or
a finite set of disjoint affine ranges. Lookup fails on missing or overlapping
ranges.

This module does not select Stage 1 ranges or source rows.
-/

namespace NightstreamFPrime.Layout.MatrixProgram

open NightstreamFPrime.Layout.R1CS (mapCombinationColumns mapRowColumns)

open NightstreamFPrime.Layout

/-- One contiguous package-column interval and its canonical source start. -/
structure SourceProjectionRange where
  packageStart : Nat
  sourceStart : Nat
  count : Nat
deriving Repr, DecidableEq

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
    (terms : List (Nat × Spec.F))
    (leftInverse : ∀ term ∈ terms,
      projection.column? (column term.1) = some term.1) :
    projectTerms? projection
        (terms.map fun term => (column term.1, term.2)) = some terms := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      simp [projectTerms?, leftInverse term (by simp),
        inductionHypothesis (fun candidate member => leftInverse candidate (by simp [member]))]

theorem SourceProjection.combination?_mapColumns_supported
    (projection : SourceProjection) (column : Nat → Nat) (combination : R1CS.LinearCombination)
    (supported : combination.VarsSatisfy (fun source => projection.column? (column source) = some source)) :
    projection.combination? (mapCombinationColumns column combination) = some combination := by
  cases combination
  simp [SourceProjection.combination?, mapCombinationColumns,
    projectTerms?_mapColumns projection column _ supported]

theorem SourceProjection.row?_mapColumns_supported
    (projection : SourceProjection) (column : Nat → Nat) (row : R1CS.Row)
    (supported : row.VarsSatisfy (fun source => projection.column? (column source) = some source)) :
    projection.row? (mapRowColumns column row) = some row := by
  rcases row with ⟨a, b, c⟩
  rcases supported with ⟨aSupport, bSupport, cSupport⟩
  unfold SourceProjection.row? mapRowColumns
  rw [SourceProjection.combination?_mapColumns_supported projection column a aSupport,
    SourceProjection.combination?_mapColumns_supported projection column b bSupport,
    SourceProjection.combination?_mapColumns_supported projection column c cSupport]
  rfl

private theorem projectTerms?_mapColumns_to
    (projection : SourceProjection) (column reference : Nat → Nat)
    (terms : List (Nat × Spec.F))
    (corresponds : ∀ term ∈ terms,
      projection.column? (column term.1) = some (reference term.1)) :
    projectTerms? projection (terms.map fun term => (column term.1, term.2)) =
      some (terms.map fun term => (reference term.1, term.2)) := by
  induction terms with
  | nil => rfl
  | cons term rest ih =>
    simp [projectTerms?, corresponds term (by simp),
      ih (fun candidate member => corresponds candidate (by simp [member]))]

theorem SourceProjection.combination?_mapColumns_to
    (projection : SourceProjection) (column reference : Nat → Nat)
    (combination : R1CS.LinearCombination)
    (supported : combination.VarsSatisfy
      (fun source => projection.column? (column source) = some (reference source))) :
    projection.combination? (mapCombinationColumns column combination) =
      some (mapCombinationColumns reference combination) := by
  cases combination
  simp [SourceProjection.combination?, mapCombinationColumns,
    projectTerms?_mapColumns_to projection column reference _ supported]

theorem SourceProjection.row?_mapColumns_to
    (projection : SourceProjection) (column reference : Nat → Nat) (row : R1CS.Row)
    (supported : row.VarsSatisfy
      (fun source => projection.column? (column source) = some (reference source))) :
    projection.row? (mapRowColumns column row) = some (mapRowColumns reference row) := by
  rcases row with ⟨a, b, c⟩
  rcases supported with ⟨aSupport, bSupport, cSupport⟩
  unfold SourceProjection.row? mapRowColumns
  rw [projection.combination?_mapColumns_to column reference a aSupport,
    projection.combination?_mapColumns_to column reference b bSupport,
    projection.combination?_mapColumns_to column reference c cSupport]
  rfl

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
  exact projection.combination?_mapColumns_supported column combination
    (fun term member => leftInverse ⟨term.1, bounded term member⟩)

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

private theorem projectTerms?_compose (left right composed : SourceProjection)
    (columns : ∀ source, composed.column? source = (left.column? source).bind right.column?)
    (terms : List (Nat × Spec.F)) :
    projectTerms? composed terms = (projectTerms? left terms).bind (projectTerms? right) := by
  induction terms with
  | nil => rfl
  | cons term rest ih =>
    cases first : left.column? term.1 with
    | none => simp [projectTerms?, columns, first]
    | some intermediate =>
      cases tail : projectTerms? left rest <;>
        cases second : right.column? intermediate <;>
        simp [projectTerms?, columns, ih, first, tail, second]

private theorem combination?_compose (left right composed : SourceProjection)
    (columns : ∀ source, composed.column? source = (left.column? source).bind right.column?)
    (combination : R1CS.LinearCombination) :
    composed.combination? combination = (left.combination? combination).bind right.combination? := by
  unfold SourceProjection.combination?
  rw [projectTerms?_compose left right composed columns]
  cases projectTerms? left combination.terms <;> simp

/-- Composition at the column boundary also composes the complete source
row interpreter. Missing columns remain a decoding failure. -/
theorem SourceProjection.row?_compose_of_columns (left right composed : SourceProjection)
    (columns : ∀ source, composed.column? source = (left.column? source).bind right.column?)
    (row : R1CS.Row) :
    composed.row? row = (left.row? row).bind right.row? := by
  simp only [SourceProjection.row?, combination?_compose left right composed columns]
  cases left.combination? row.a <;>
    cases left.combination? row.b <;>
    cases left.combination? row.c <;> simp

end NightstreamFPrime.Layout.MatrixProgram
