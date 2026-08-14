import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteSchema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Decoder

/-!
Contract: fail-closed decoder for executable grouped-product rewrite data.

Owns: source-row and column bounds, canonical field coefficients, nonzero
sparse terms, nonempty slots and source-row ranges, and the five-factor limit
of one evaluation row.

Does not own: final matrix rows, source-to-final assignment encoding, rewrite
semantics, Rust conformance, production coverage, constraint necessity, or row
removal.

Emits constraints: no.

| Wire object | Checked property | Successful type |
|---|---|---|
| source term | bounded column and canonical nonzero coefficient | `DecodedSourceTerm` |
| source linear combination | canonical constant and bounded nonzero terms | `DecodedSourceLinearCombination` |
| source R1CS row | bounded row and three decoded linear combinations | `DecodedSourceR1csRow` |
| source or derived slot | nonempty final-column interval | `DecodedSourceSlot` or `DecodedDerivedSlot` |
| source range | nonempty and bounded half-open interval | `DecodedRange` |
| rewrite step | decoded fields and at most five factors | `DecodedStep` |
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire

structure DecodedSourceTerm (columns : Nat) where
  column : Fin columns
  coefficient : F
deriving DecidableEq, Repr

structure DecodedSourceLinearCombination (columns : Nat) where
  constant : F
  terms : List (DecodedSourceTerm columns)
  coefficientsNonzero : ∀ term ∈ terms, term.coefficient ≠ 0

/-- One source R1CS equation with its exact position in the source relation. -/
structure DecodedSourceR1csRow (rows columns : Nat) where
  row : Fin rows
  a : DecodedSourceLinearCombination columns
  b : DecodedSourceLinearCombination columns
  c : DecodedSourceLinearCombination columns

structure DecodedSourceSlot (sourceColumns finalColumns : Nat) where
  column : Fin sourceColumns
  start : Nat
  width : Nat
  widthPositive : 0 < width
  columnsFit : start + width ≤ finalColumns
deriving DecidableEq, Repr

structure DecodedSourceDefinition (sourceColumns : Nat) where
  target : Fin sourceColumns
  value : DecodedSourceLinearCombination sourceColumns

structure DecodedDerivedSlot (finalColumns : Nat) where
  compilerIndex : Nat
  start : Nat
  width : Nat
  widthPositive : 0 < width
  columnsFit : start + width ≤ finalColumns
deriving DecidableEq, Repr

structure DecodedRange (rows : Nat) where
  start : Nat
  stop : Nat
  nonempty : start < stop
  bounded : stop ≤ rows
deriving DecidableEq, Repr

inductive DecodedOutput (columns : Nat) where
  | source (value : DecodedSourceLinearCombination columns)
  | derivedProductSum (compilerIndex : Nat)

structure DecodedFactor (columns : Nat) where
  left : DecodedSourceLinearCombination columns
  right : DecodedSourceLinearCombination columns
  coefficient : F
  coefficientNonzero : coefficient ≠ 0

structure DecodedStep (rows columns : Nat) where
  emittedRow : Nat
  rewriteId : Nat
  kind : RawKind
  sourceRows : List (DecodedRange rows)
  output : DecodedOutput columns
  base : DecodedSourceLinearCombination columns
  previous : Option Nat
  factors : List (DecodedFactor columns)
  factorsBound : factors.length ≤ 5

def decodeSourceTerm (columns : Nat) (raw : RawSourceTerm) :
    Option (DecodedSourceTerm columns) :=
  if columnInRange : raw.column < columns then do
    let coefficient ←
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField
        raw.coefficient
    if coefficient ≠ 0 then
      pure ⟨⟨raw.column, columnInRange⟩, coefficient⟩
    else
      none
  else
    none

def SourceTermsValid {columns : Nat}
    (terms : List (DecodedSourceTerm columns)) : Prop :=
  ∀ term ∈ terms, term.coefficient ≠ 0

private instance {columns : Nat} (terms : List (DecodedSourceTerm columns)) :
    Decidable (SourceTermsValid terms) := by
  unfold SourceTermsValid
  infer_instance

def decodeSourceLinearCombination (columns : Nat)
    (raw : RawSourceLinearCombination) :
    Option (DecodedSourceLinearCombination columns) := do
  let constant ←
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField
      raw.constant
  let terms ← raw.terms.mapM (decodeSourceTerm columns)
  if valid : SourceTermsValid terms then
    pure
      { constant
        terms
        coefficientsNonzero := valid }
  else
    none

/-- Decode one source R1CS row without trusting its row or column bounds. -/
def decodeSourceR1csRow (rows columns : Nat) (raw : RawSourceR1csRow) :
    Option (DecodedSourceR1csRow rows columns) := do
  if rowInRange : raw.row < rows then
    let a ← decodeSourceLinearCombination columns raw.a
    let b ← decodeSourceLinearCombination columns raw.b
    let c ← decodeSourceLinearCombination columns raw.c
    pure { row := ⟨raw.row, rowInRange⟩, a, b, c }
  else
    none

def decodeSourceSlot (sourceColumns finalColumns : Nat)
    (raw : RawSourceSlot) :
    Option (DecodedSourceSlot sourceColumns finalColumns) :=
  if columnInRange : raw.column < sourceColumns then
    if widthPositive : 0 < raw.width then
      if columnsFit : raw.start + raw.width ≤ finalColumns then
        some
          { column := ⟨raw.column, columnInRange⟩
            start := raw.start
            width := raw.width
            widthPositive
            columnsFit }
      else
        none
    else
      none
  else
    none

def decodeSourceDefinition (sourceColumns : Nat)
    (raw : RawSourceDefinition) :
    Option (DecodedSourceDefinition sourceColumns) := do
  if targetInRange : raw.target < sourceColumns then
    let value ← decodeSourceLinearCombination sourceColumns raw.value
    pure { target := ⟨raw.target, targetInRange⟩, value }
  else
    none

def decodeDerivedSlot (finalColumns : Nat)
    (raw : RawDerivedSlot) : Option (DecodedDerivedSlot finalColumns) :=
  if widthPositive : 0 < raw.width then
    if columnsFit : raw.start + raw.width ≤ finalColumns then
      some
        { compilerIndex := raw.compilerIndex
          start := raw.start
          width := raw.width
          widthPositive
          columnsFit }
    else
      none
  else
    none

def decodeRange (rows : Nat) (raw : RawRange) :
    Option (DecodedRange rows) :=
  if nonempty : raw.start < raw.stop then
    if bounded : raw.stop ≤ rows then
      some ⟨raw.start, raw.stop, nonempty, bounded⟩
    else
      none
  else
    none

def decodeOutput (columns : Nat) : RawOutput → Option (DecodedOutput columns)
  | .source value => do
      let decoded ← decodeSourceLinearCombination columns value
      pure (.source decoded)
  | .derivedProductSum compilerIndex =>
      some (.derivedProductSum compilerIndex)

def decodeFactor (columns : Nat) (raw : RawFactor) :
    Option (DecodedFactor columns) := do
  let left ← decodeSourceLinearCombination columns raw.left
  let right ← decodeSourceLinearCombination columns raw.right
  let coefficient ←
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField
      raw.coefficient
  if nonzero : coefficient ≠ 0 then
    pure ⟨left, right, coefficient, nonzero⟩
  else
    none

/-- Decode one recurrence without trusting or repairing its source geometry. -/
def decodeStep (rows columns : Nat) (raw : RawStep) :
    Option (DecodedStep rows columns) := do
  let sourceRows ← raw.sourceRows.mapM (decodeRange rows)
  let output ← decodeOutput columns raw.output
  let base ← decodeSourceLinearCombination columns raw.base
  let factors ← raw.factors.mapM (decodeFactor columns)
  if factorsBound : factors.length ≤ 5 then
    pure
      { emittedRow := raw.emittedRow
        rewriteId := raw.rewriteId
        kind := raw.kind
        sourceRows
        output
        base
        previous := raw.previous
        factors
        factorsBound }
  else
    none

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
