/-!
Wire schema for the focused source-column decoder and rewrite provenance.

Owns: retained source-field slots, compiler linear definitions, and the
ordered partial-product accumulators that occur in the selectively emitted
`y_zcol` rows.

Does not own: artifact generation, field decoding, assignment satisfaction,
the meaning of a source column, selector truth, trace correctness, protocol
authority, or permission to remove constraints.

Emits constraints: no.

Authority boundary: these records are inert wire data. Correspondence modules
must decode coefficients, check column/slot bounds and the complete source
partition, and prove that each rewrite realizes the independently reconstructed
source program.

| Wire leaf | Mathematical obligation | Authority class |
|---|---|---|
| retained slot | final-column expansion of one source input | computed |
| linear definition | ordered compiler substitution | computed |
| rewrite step | source/base/product provenance for one compact row | checked after decoding |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

structure RawSourceTerm where
  column : Nat
  coefficient : Nat
deriving DecidableEq, Repr

structure RawSourceLinearCombination where
  constant : Nat
  terms : List RawSourceTerm
deriving DecidableEq, Repr

structure RawSourceSlot where
  column : Nat
  start : Nat
  width : Nat
deriving DecidableEq, Repr

structure RawSourceDefinition where
  target : Nat
  constant : Nat
  terms : List RawSourceTerm
deriving DecidableEq, Repr

structure RawProductFactor where
  left : RawSourceLinearCombination
  right : RawSourceLinearCombination
  coefficient : Nat
deriving DecidableEq, Repr

structure RawDerivedProductSum where
  compilerIndex : Nat
  start : Nat
  width : Nat
  factors : List RawProductFactor
  previous : Option Nat
deriving DecidableEq, Repr

/-- One half-open source-row interval owned by an executable rewrite step. -/
structure RawSourceRowBlock where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

/-- The two selective compiler families carrying source equations in this
focused slice. Retained rows and eliminated linear definitions are represented
outside this executable rewrite stream. -/
inductive RawRewriteKind where
  | polynomialEvaluation
  | productSum
deriving DecidableEq, Repr

/-- The value reconstructed by one selectively emitted row: either a source
linear combination or one named derived product-sum accumulator. -/
inductive RawRewriteOutput where
  | source (value : RawSourceLinearCombination)
  | derivedProductSum (compilerIndex : Nat)
deriving DecidableEq, Repr

/-- Inert provenance for one emitted polynomial-evaluation or product-sum row.
`base`, `previous`, and `factors` record the exact recurrence inputs; later
correspondence must decode and execute them rather than trusting `kind`. -/
structure RawRewriteStep where
  emittedRow : Nat
  rewriteId : Nat
  kind : RawRewriteKind
  sourceRows : List RawSourceRowBlock
  output : RawRewriteOutput
  base : RawSourceLinearCombination
  previous : Option Nat
  factors : List RawProductFactor
deriving DecidableEq, Repr

/-- Exact source A/B/C linear forms owned by one physically retained emitted
row. `sourceRow` is the absolute source-R1CS row index; its unique stage owner
is recovered from the checked source-leaf interval artifact. -/
structure RawRetainedStep where
  emittedRow : Nat
  sourceRow : Nat
  a : RawSourceLinearCombination
  b : RawSourceLinearCombination
  c : RawSourceLinearCombination
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
