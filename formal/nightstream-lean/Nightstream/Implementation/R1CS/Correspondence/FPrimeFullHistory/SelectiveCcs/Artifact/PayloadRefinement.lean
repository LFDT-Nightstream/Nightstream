import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Decoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Interpreter
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.FixedPointShape

/-!
Model-level placement of one decoded thirteen-matrix payload at a stabilized
selective-relation header.

Assurance tier: model-level fixed snapshot. No complete bundle for these old
fixed dimensions is exported, so this module is not artifact-checked,
Rust-conformant, or security-reduced.

Owns: successful fail-closed bundle decoding; exact row/column agreement with
the stabilized header; exhaustive role ownership of all thirteen decoded
matrices; and attachment of the independent selective polynomial and row
profile.

Does not own: canonical derivation of matrix coefficients from F' semantics,
Rust provenance, compiler convergence, source-assignment encoding, row-family
classification, seeded-sampler equality, relation acceptance, costs, or row
removal.

Emits constraints: no.

Authority boundary: decoding proves that the raw compact payload is
well-formed, not that its coefficients implement F'. The decoded matrices are
therefore refinement evidence only. Semantic authority must come from a future
canonical-matrix construction and an equality proof against this payload.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `f_prime.fixed_point.payload.decode` | exact raw variants and canonical field words decode successfully | checked | `Refinement.decodedFromRaw` |
| `f_prime.fixed_point.payload.dimensions` | decoded rows/columns equal the stabilized emitted header | checked | `Refinement.rowsExact`, `Refinement.columnsExact` |
| `f_prime.fixed_point.payload.roles` | every physical port has exactly one semantic role | computed | `decodedRelation_roleMatrix` |
| `f_prime.fixed_point.payload.relation` | transport all decoded role matrices to the stabilized dimensions | computed | `Refinement.toRelation` |
| `f_prime.fixed_point.payload.polynomial` | attach only the independent 66-term polynomial | computed | `toStructure_constraintPolynomial` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.PayloadRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Schema
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile

abbrev FixedSnapshot :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Wire.RawSnapshot

abbrev PayloadBundle :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Wire.RawBundle

/-- One raw fixed-point header and one raw matrix bundle connected through the
actual fail-closed decoder. Dimension equalities compare the decoded payload,
not caller-repeated raw metadata, with the independently validated header. -/
structure Refinement
    (rawShape : FixedSnapshot)
    (rawBundle : PayloadBundle) where
  fixedPoint :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement
      rawShape
  verifierFuel : Nat
  decoded : ProductionBundle
  decodedFromRaw :
    decodeProductionBundle verifierFuel rawBundle = some decoded
  rowsExact :
    decoded.bundle.rows = rawShape.materialized.verifier.rows
  columnsExact :
    decoded.bundle.columns = rawShape.materialized.verifier.columns

namespace Refinement

/-- The successfully decoded bundle contains exactly one matrix for every
named selective role. -/
theorem decoded_matrixCount_eq_13
    {rawShape : FixedSnapshot}
    {rawBundle : PayloadBundle}
    (refinement : Refinement rawShape rawBundle) :
    refinement.decoded.bundle.matrices.length = 13 := by
  calc
    refinement.decoded.bundle.matrices.length = Schema.portCount :=
      refinement.decoded.productionValid.valid.matrixCount
    _ = 13 := by
      exact port_count_exact

/-- Relation at the decoded payload's native dimensions. Every role selects
its sole physical port through `Role.index`. -/
def decodedRelation
    {rawShape : FixedSnapshot}
    {rawBundle : PayloadBundle}
    (refinement : Refinement rawShape rawBundle) :
    FiniteRelation refinement.decoded.bundle.rows
      refinement.decoded.bundle.columns :=
  Interpreter.ValidatedBundle.interpretRelation refinement.decoded.validated

/-- Exact compact matrix underlying one named semantic role before dimension
transport. This quantifies all roles and therefore all thirteen ports. -/
@[simp] theorem decodedRelation_roleMatrix
    {rawShape : FixedSnapshot}
    {rawBundle : PayloadBundle}
    (refinement : Refinement rawShape rawBundle)
    (role : Role) :
    (refinement.decodedRelation.matrices role) =
      fun row column =>
        CompactMatrix.valueAt
          (ValidatedBundle.matrixAt refinement.decoded.validated role.index)
          row column := by
  rfl

/-- Transport the complete decoded relation to the dimensions validated by
the fixed-point header. Equality transport changes no coefficient. -/
def toRelation
    {rawShape : FixedSnapshot}
    {rawBundle : PayloadBundle}
    (refinement : Refinement rawShape rawBundle) :
    FiniteRelation rawShape.materialized.verifier.rows
      rawShape.materialized.verifier.columns := by
  rw [← refinement.rowsExact, ← refinement.columnsExact]
  exact refinement.decodedRelation

/-- Attach the fixed-point row profile and independent selective polynomial to
the complete decoded role family. -/
def toStructure
    {rawShape : FixedSnapshot}
    {rawBundle : PayloadBundle}
    (refinement : Refinement rawShape rawBundle) :=
  refinement.toRelation.toStructure refinement.fixedPoint.toProfile

/-- The payload cannot supply or alter the selective constraint polynomial. -/
@[simp] theorem toStructure_constraintPolynomial
    {rawShape : FixedSnapshot}
    {rawBundle : PayloadBundle}
    (refinement : Refinement rawShape rawBundle) :
    refinement.toStructure.constraintPolynomial =
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial := by
  rfl

/-- Every named role reaches the correspondingly named matrix in the typed
fixed-point structure. -/
@[simp] theorem toStructure_roleMatrix
    {rawShape : FixedSnapshot}
    {rawBundle : PayloadBundle}
    (refinement : Refinement rawShape rawBundle)
    (role : Role) :
    refinement.toStructure.matrices role.index =
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowPadding.padRows
        (refinement.toRelation.matrices role) := by
  exact FiniteRelation.toStructure_roleMatrix
    refinement.toRelation refinement.fixedPoint.toProfile role

end Refinement

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.PayloadRefinement
