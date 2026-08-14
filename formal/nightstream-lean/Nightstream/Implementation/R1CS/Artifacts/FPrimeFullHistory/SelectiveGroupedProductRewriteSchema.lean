import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema

/-!
Artifact-owned wire schema for an executable grouped-product rewrite.

Owns: untrusted source linear combinations, their final low-norm slots,
product factors, recurrence outputs, rewrite ownership, and source-row ranges.

Does not own: field semantics, row semantics, Rust conformance, production
coverage, constraint necessity, or permission to remove rows or coordinates.

Emits constraints: no.

| Wire field | Rust source | Semantic status |
|---|---|---|
| source linear combinations | checked rewrite plan | untrusted until decoded |
| retained and derived slots | checked selective layout | untrusted until decoded |
| product factors | checked rewrite plan | untrusted until decoded |
| output and predecessor | checked rewrite plan | recurrence metadata only |
| emitted row and rewrite ID | compiler ownership ledger | join keys only |
-/

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire

/-- One untrusted source-field coefficient. -/
structure RawSourceTerm where
  column : Nat
  coefficient : Nat
deriving DecidableEq, Repr

/-- One untrusted source linear combination with a separate constant. -/
structure RawSourceLinearCombination where
  constant : Nat
  terms : List RawSourceTerm
deriving DecidableEq, Repr

/-- One exact source-R1CS row claimed by a grouped rewrite. The row number is
part of the wire data so an equal equation at another source position cannot
silently replace it. -/
structure RawSourceR1csRow where
  row : Nat
  a : RawSourceLinearCombination
  b : RawSourceLinearCombination
  c : RawSourceLinearCombination
deriving DecidableEq, Repr

/-- One untrusted final low-norm slot for a retained source field. The radix
is fixed by the compiler from `width`: width 41 uses balanced radix three and
all other widths use radix two. -/
structure RawSourceSlot where
  column : Nat
  start : Nat
  width : Nat
deriving DecidableEq, Repr

/-- One compiler-validated source-column substitution. -/
structure RawSourceDefinition where
  target : Nat
  value : RawSourceLinearCombination
deriving DecidableEq, Repr

/-- One final low-norm slot for a grouped-product accumulator. -/
structure RawDerivedSlot where
  compilerIndex : Nat
  start : Nat
  width : Nat
deriving DecidableEq, Repr

/-- One source-row half-open interval. -/
structure RawRange where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

/-- The two rewrite kinds that use the five-product evaluation polynomial. -/
inductive RawKind where
  | polynomialEvaluation
  | productSum
deriving DecidableEq, Repr

/-- Source result or compiler-owned intermediate finalized by one step. -/
inductive RawOutput where
  | source (value : RawSourceLinearCombination)
  | derivedProductSum (compilerIndex : Nat)
deriving DecidableEq, Repr

/-- One scaled source product. The coefficient multiplies the left factor in
the final selective row. -/
structure RawFactor where
  left : RawSourceLinearCombination
  right : RawSourceLinearCombination
  coefficient : Nat
deriving DecidableEq, Repr

/-- Literal executable recurrence for one emitted evaluation row. -/
structure RawStep where
  emittedRow : Nat
  rewriteId : Nat
  kind : RawKind
  sourceRows : List RawRange
  output : RawOutput
  base : RawSourceLinearCombination
  previous : Option Nat
  factors : List RawFactor
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire
