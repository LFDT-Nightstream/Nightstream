import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.FixedPointShapeSchema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.SelectorCoverage
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.RelationProfile
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Model-level contract from an untrusted fixed-point header to the independent
13-port selective relation shape.

Assurance tier: model-level schema/refinement; no concrete Rust snapshot is
checked here, so this is not artifact-checked, Rust-conformant, or
security-reduced.

Owns: exact equality of the terminal input, shape-only output, and materialized
headers; exact comparison with the independent sparse polynomial; 270-field
public width; actual matrix-count/arity agreement; final Phi81 width alignment;
the minimal Boolean row domain; and construction of `RelationProfile.Profile`.

Does not own: compiler convergence, a generated snapshot, matrix coefficients,
row ordering, assignment encoding, CCS/CE membership, Π_CCS output authority,
R1CS satisfaction, costs, or row removal.

Emits constraints: no.

Authority boundary: header values remain untrusted. Matrix arity and polynomial
syntax come from the independent selective semantics, while public width and
alignment come from the typed Phi81 carrier. Even a valid header proves only a
shape contract; all thirteen matrices and their assignment map still require a
separate payload refinement.

| Stage path | Mathematical obligation | Authority class | Lean owner | Rust/R1CS owner | Multiplicity source |
|---|---|---|---|---|---|
| `f_prime.fixed_point.header.stable` | terminal input = selective output = emitted header | checked | `Refinement.terminalInput_eq_materialized` | fixed-point compiler audit | one final round |
| `f_prime.fixed_point.header.matrices` | emitted matrix count = polynomial arity = independent port count | checked | `materialized_matrixCount_eq_13`, `materialized_polynomialArity_eq_13` | emitted structure and full sparse polynomial | semantic port vocabulary |
| `f_prime.fixed_point.header.polynomial` | exact independent 74-term syntax | checked | `materialized_polynomialTerms_eq` | full sparse polynomial export | `Polynomial.Semantics.terms` |
| `f_prime.fixed_point.header.public` | public input is exactly five Phi81 rings = 270 fields | checked | `materialized_publicInputLength_eq_270` | selective layout audit | `FPrimeCarrier270.alignedPublicWidth` |
| `f_prime.fixed_point.header.columns` | final width is already a complete Phi81 carrier | checked | `materialized_columns_ring_aligned` | emitted relation width | `Phi81CarrierLayout.carrierWidth` |
| `f_prime.fixed_point.header.rows` | row variables form the least Boolean cube covering physical rows | checked | `Refinement.rowDomain` | final relation rows | `RelationProfile.ExactRowDomain` |
| `f_prime.fixed_point.shape` | construct the active relation shape with thirteen matrices | computed | `Refinement.toProfile`, `profile_shape_matrixCount_eq_13` | none; semantic specification | semantic port vocabulary |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS.SelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage

def supportedSchemaVersion : Nat := 1

/-- Proof-carrying validation of one raw fixed-point snapshot. The row-domain
witness is semantic data derived from the physical row count, not a field the
raw header may declare authoritative. -/
structure Refinement (raw : RawSnapshot) where
  schemaVersion : raw.schemaVersion = supportedSchemaVersion
  terminalStable : raw.terminalInput = raw.selectiveOutput
  materializedStable : raw.selectiveOutput = raw.materialized.verifier
  rowsPositive : 0 < raw.materialized.verifier.rows
  columnsPositive : 0 < raw.materialized.verifier.columns
  publicInputExact :
    raw.materialized.verifier.publicInputLength = alignedPublicWidth
  publicInputFits :
    raw.materialized.verifier.publicInputLength <=
      raw.materialized.verifier.columns
  columnsAligned :
    Phi81CarrierLayout.carrierWidth raw.materialized.verifier.columns =
      raw.materialized.verifier.columns
  matrixCountExact :
    raw.materialized.matrixCount = RelationProfile.matrixCount
  polynomialArityMatchesMatrixCount :
    raw.materialized.verifier.polynomialArity = raw.materialized.matrixCount
  polynomialTermsExact :
    raw.materialized.verifier.polynomialTerms = expectedPolynomialTerms
  rowVariables : Nat
  rowDomain :
    RelationProfile.ExactRowDomain raw.materialized.verifier.rows rowVariables

namespace Refinement

theorem terminalInput_eq_materialized
    {raw : RawSnapshot} (refinement : Refinement raw) :
    raw.terminalInput = raw.materialized.verifier :=
  refinement.terminalStable.trans refinement.materializedStable

/-- The semantic relation profile obtained without trusting a generated matrix
count or polynomial tag. -/
def toProfile {raw : RawSnapshot} (refinement : Refinement raw) :
    RelationProfile.Profile raw.materialized.verifier.rows
      raw.materialized.verifier.columns where
  rowVariables := refinement.rowVariables
  rowDomain := refinement.rowDomain
  publicRingColumns := publicRingColumns
  publicFits := by
    have publicToColumns :
        alignedPublicWidth <= raw.materialized.verifier.columns := by
      rw [← refinement.publicInputExact]
      exact refinement.publicInputFits
    have publicToCarrier :
        alignedPublicWidth <=
          Phi81CarrierLayout.carrierWidth
            raw.materialized.verifier.columns :=
      Nat.le_trans publicToColumns
        (Phi81CarrierLayout.logicalWidth_le_carrierWidth _)
    simpa [alignedPublicWidth] using publicToCarrier

theorem materialized_publicInputLength_eq_270
    {raw : RawSnapshot} (refinement : Refinement raw) :
    raw.materialized.verifier.publicInputLength = 270 := by
  calc
    raw.materialized.verifier.publicInputLength = alignedPublicWidth :=
      refinement.publicInputExact
    _ = 270 := by decide

theorem materialized_columns_ring_aligned
    {raw : RawSnapshot} (refinement : Refinement raw) :
    Phi81CarrierLayout.carrierWidth raw.materialized.verifier.columns =
      raw.materialized.verifier.columns :=
  refinement.columnsAligned

theorem materialized_matrixCount_eq_13
    {raw : RawSnapshot} (refinement : Refinement raw) :
    raw.materialized.matrixCount = 13 := by
  rw [refinement.matrixCountExact]
  rfl

theorem materialized_polynomialArity_eq_13
    {raw : RawSnapshot} (refinement : Refinement raw) :
    raw.materialized.verifier.polynomialArity = 13 := by
  calc
    raw.materialized.verifier.polynomialArity =
        raw.materialized.matrixCount :=
      refinement.polynomialArityMatchesMatrixCount
    _ = 13 := materialized_matrixCount_eq_13 refinement

theorem materialized_polynomialTerms_eq
    {raw : RawSnapshot} (refinement : Refinement raw) :
    raw.materialized.verifier.polynomialTerms = expectedPolynomialTerms :=
  refinement.polynomialTermsExact

theorem materialized_rows_covered
    {raw : RawSnapshot} (refinement : Refinement raw) :
    raw.materialized.verifier.rows <= 2 ^ refinement.rowVariables :=
  RelationProfile.Profile.rows_covered refinement.toProfile

theorem profile_shape_matrixCount_eq_13
    {raw : RawSnapshot} (refinement : Refinement raw) :
    (RelationProfile.Profile.shape refinement.toProfile).matrixCount = 13 :=
  RelationProfile.Profile.shape_matrixCount_eq_13 refinement.toProfile

theorem profile_shape_matrixCount_ne_three
    {raw : RawSnapshot} (refinement : Refinement raw) :
    (RelationProfile.Profile.shape refinement.toProfile).matrixCount ≠ 3 :=
  RelationProfile.Profile.shape_matrixCount_ne_three refinement.toProfile

end Refinement

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape
