import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Types
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.OutputPoint
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81Evaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc

/-!
Model-level output-claim semantics for Phi81 `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS`.
Phase: source-derived output claims at the verifier-owned row and column points.
Constraint family: semantic authority only; this file emits no constraints.

Owns: the canonical claim product derived from `SplitNc.Sources.Data`;
separate fresh and running constructions; fresh logical-width zero
completion; source-binding predicates; and model-level canonical completeness
and uniqueness.

Does not own: `PiCCS.Accepted`, either SumCheck verifier, proof that `rPrime` or
`sPrime` came from a transcript, `ProtocolPolynomial.OutputMessage`, production
payload serialization, Rust, R1CS, digests, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `Claims` is prover-claim-shaped data and has no authority by
construction. `BoundToSources` binds every active claim to the sole matrices and
assignments in `SplitNc.Sources.Data`. Fresh sources use the canonical
logical-width prefix plus zero-completed carrier suffix. Running sources use
their complete carrier verbatim. The caller must supply verifier-derived
`rPrime` and `sPrime`; neither point is carried inside `Claims`.

This module deliberately does not identify its complete output product with
`PaperJoint.ProtocolPolynomial.OutputMessage`: that message contains only the
families consumed by the paper terminal, whereas these claims contain every
active output coefficient and an independent SplitNC column projection.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | output point | `rPrime` | explicit verifier-owned row point shared by every `yRing` claim |
| SplitNC | output point | `sPrime` | explicit verifier-owned padded-column point shared by every `yZcol` claim |
| `Pi_CCS` | output evaluation | `yRing[source,matrix,lane]` | evaluate the derived Phi81 coefficient-matrix image at `rPrime` |
| SplitNC | output projection | `yZcol[source,lane]` | evaluate the authoritative packed diagonal at `sPrime` over the full carrier |
| fresh source | carrier | logical prefix / completed suffix | preserve the logical assignment and set every new carrier coordinate to zero |
| running source | carrier | complete assignment | preserve every carrier coordinate verbatim |
| assurance | `yRing` branch | source binding | `YRingBoundToSources` names the CE/extraction-owned obligation independently |
| assurance | `yZcol` branch | source binding | `YZcolBoundToSources` names the SplitNC-sidecar obligation independently |
| assurance | point separation | `yRing` / `yZcol` transport | `yRing` ignores `sPrime`; `yZcol` ignores `rPrime` |
| assurance | canonical construction | completeness | `canonicalClaims` satisfies every source-bound output equation |
| assurance | canonical construction | uniqueness | any source-bound claims equal `canonicalClaims` extensionally |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-! ## Row-point `yRing` semantics -/

/-- Boolean row table for one Phi81 coefficient image derived from the sole
matrix source. This is the reusable CE-evaluation leaf; it does not depend on
the rest of a joint `Pi_CCS` source batch. -/
def yRingTableForMatrixSource
    {shape : SemanticShape}
    (matrixSource : MatrixCoefficientSource.MatrixSource F shape.paperShape
      shape.carrierWidth (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (assignment : Assignment F shape.carrierWidth)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : BooleanTable K shape.rowVariables :=
  Phi81Evaluation.table matrixSource assignment matrix lane

/-- Boolean row table for one Phi81 coefficient image and one complete-carrier
assignment. Every coefficient matrix is derived from the sole matrix source. -/
def yRingTableForAssignment
    {shape : SemanticShape}
    (data : SplitNc.Sources.Data shape)
    (assignment : Assignment F shape.carrierWidth)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : BooleanTable K shape.rowVariables :=
  yRingTableForMatrixSource data.matrixSource assignment matrix lane

/-- Evaluate one coefficient image from the sole matrix source at the
verifier-owned row point. -/
def yRingForMatrixSource
    {shape : SemanticShape}
    (matrixSource : MatrixCoefficientSource.MatrixSource F shape.paperShape
      shape.carrierWidth (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (assignment : Assignment F shape.carrierWidth)
    (rPrime : CubePoint K shape.rowVariables)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : K :=
  Phi81Evaluation.evaluate matrixSource assignment rPrime matrix lane

/-- Evaluate one derived coefficient-image table at the verifier-owned row
point. -/
def yRingForAssignment
    {shape : SemanticShape}
    (data : SplitNc.Sources.Data shape)
    (assignment : Assignment F shape.carrierWidth)
    (rPrime : CubePoint K shape.rowVariables)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : K :=
  yRingForMatrixSource data.matrixSource assignment rPrime matrix lane

/-- Canonical `yRing` claim for one source in fresh-then-running order. -/
def canonicalYRing
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : K :=
  yRingForAssignment data (data.assignment source) points.rPrime matrix lane

/-- Fresh output evaluation from the logical assignment after canonical
zero-completion to the Phi81 carrier. -/
def canonicalFreshYRing
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.freshCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : K :=
  yRingForAssignment data (data.freshAssignment source)
    points.rPrime matrix lane

/-- Running output evaluation from the caller's complete authoritative carrier. -/
def canonicalRunningYRing
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : K :=
  yRingForAssignment data (data.runningAssignments source)
    points.rPrime matrix lane

@[simp] theorem canonicalYRing_freshIndex
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.freshCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    canonicalYRing data points (SplitNc.Sources.Data.freshIndex source) matrix lane =
      canonicalFreshYRing data points source matrix lane := by
  simp [canonicalYRing, canonicalFreshYRing,
    SplitNc.Sources.Data.assignment_freshIndex]

@[simp] theorem canonicalYRing_runningIndex
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    canonicalYRing data points (SplitNc.Sources.Data.runningIndex source) matrix lane =
      canonicalRunningYRing data points source matrix lane := by
  simp [canonicalYRing, canonicalRunningYRing,
    SplitNc.Sources.Data.assignment_runningIndex]

/-! ## Column-point `yZcol` semantics -/

/-- Little-endian Boolean-basis weight of one padded column index at `sPrime`.
The later transcript refinement must prove that `sPrime` is the NC SumCheck
column conclusion; this definition does not assume that bridge. -/
def columnWeight
    {domain : FlatNcDomain}
    (sPrime : CubePoint K domain.columnVariables)
    (column : Fin domain.columnCount) : K :=
  PaperJoint.NumericBooleanDomain.testBitWeight
    ConcreteCarrier.extensionOps sPrime column

/-- One full-carrier contribution to an active `yZcol` lane. The diagonal is
the independently defined SplitNC table, not a prover-carried sidecar. -/
def yZcolTerm
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (assignment : Assignment F shape.carrierWidth)
    (sPrime : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree)
    (column : Fin shape.carrierWidth) : K :=
  K.mul
    (K.embed (SplitNc.Semantics.Nc.diagonal assignment column lane))
    (columnWeight sPrime (domain.carrierColumn covers column))

/-- Full-carrier `yZcol` evaluation for one assignment. No logical-width
truncation occurs in this fold. -/
def yZcolForAssignment
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (assignment : Assignment F shape.carrierWidth)
    (sPrime : CubePoint K domain.columnVariables)
    (lane : Fin ringDegree) : K :=
  (canonicalFinIndices shape.carrierWidth).foldl
    (fun accumulated column =>
      K.add accumulated
        (yZcolTerm covers assignment sPrime lane column))
    K.zero

/-- Canonical `yZcol` claim for one source in fresh-then-running order. -/
def canonicalYZcol
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree) : K :=
  yZcolForAssignment covers (data.assignment source) points.sPrime lane

/-- Fresh `yZcol` uses the zero-completed logical assignment. -/
def canonicalFreshYZcol
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.freshCount)
    (lane : Fin ringDegree) : K :=
  yZcolForAssignment covers (data.freshAssignment source) points.sPrime lane

/-- Running `yZcol` uses every coordinate of the complete running carrier. -/
def canonicalRunningYZcol
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.runningCount)
    (lane : Fin ringDegree) : K :=
  yZcolForAssignment covers (data.runningAssignments source) points.sPrime lane

@[simp] theorem canonicalYZcol_freshIndex
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.freshCount)
    (lane : Fin ringDegree) :
    canonicalYZcol covers data points (SplitNc.Sources.Data.freshIndex source) lane =
      canonicalFreshYZcol covers data points source lane := by
  simp [canonicalYZcol, canonicalFreshYZcol,
    SplitNc.Sources.Data.assignment_freshIndex]

@[simp] theorem canonicalYZcol_runningIndex
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.runningCount)
    (lane : Fin ringDegree) :
    canonicalYZcol covers data points (SplitNc.Sources.Data.runningIndex source) lane =
      canonicalRunningYZcol covers data points source lane := by
  simp [canonicalYZcol, canonicalRunningYZcol,
    SplitNc.Sources.Data.assignment_runningIndex]

/-- Every fresh carrier coordinate outside the original logical width is the
canonical zero supplied by `SplitNc.Sources.Data`, not an output claim. -/
theorem freshCarrier_tail_zero
    {shape : SemanticShape}
    (data : SplitNc.Sources.Data shape)
    (source : Fin shape.freshCount)
    (column : Fin shape.carrierWidth)
    (tail : shape.logicalWidth <= column.val) :
    data.freshAssignment source column = 0 := by
  exact Phi81CarrierLayout.extendAssignment_tail_zero
    0 (data.freshAssignments source) column tail

/-- Consequently a completed fresh suffix contributes zero to every active
column-projection lane. -/
theorem freshYZcolTerm_tail_zero
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (source : Fin shape.freshCount)
    (lane : Fin ringDegree)
    (column : Fin shape.carrierWidth)
    (tail : shape.logicalWidth <= column.val) :
    yZcolTerm covers (data.freshAssignment source)
      points.sPrime lane column = K.zero := by
  have assignmentZero := freshCarrier_tail_zero data source column tail
  by_cases selected : lane.val = column.val % ringDegree
  · simp [yZcolTerm, SplitNc.Semantics.Nc.diagonal, selected, assignmentZero,
      K.embed, K.mul, K.zero, Fin.zero_mul, Fin.add_zero]
    exact Fin.zero_mul _
  · simp [yZcolTerm, SplitNc.Semantics.Nc.diagonal, selected,
      K.embed, K.mul, K.zero, Fin.zero_mul, Fin.add_zero]
    exact Fin.zero_mul _

/-! ## Canonical construction and authority predicate -/

/-- The unique source-derived active claim product at the two explicit points. -/
def canonicalClaims
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain) : Claims shape where
  yRing := canonicalYRing data points
  yZcol := canonicalYZcol covers data points

/-- The full active `yRing` branch is bound to the sole matrix/assignment
source. This is the branch later CE extraction must establish. -/
def YRingBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (claims : Claims shape) : Prop :=
  forall source matrix lane,
    claims.yRing source matrix lane =
      canonicalYRing data points source matrix lane

/-- The full active `yZcol` branch is bound to the independent packed
assignment projection. This is the separate SplitNC-sidecar obligation. -/
def YZcolBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (claims : Claims shape) : Prop :=
  forall source lane,
    claims.yZcol source lane =
      canonicalYZcol covers data points source lane

/-- The CE-owned `yRing` authority predicate depends only on the row point.
Changing the independently derived NC column point cannot change it. -/
theorem yRingBoundToSources_iff_of_rPrime_eq
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : SplitNc.Sources.Data shape)
    (left right : VerifierPoints shape domain)
    (claims : Claims shape)
    (rPrime_eq : left.rPrime = right.rPrime) :
    YRingBoundToSources data left claims <->
      YRingBoundToSources data right claims := by
  have canonical_eq : forall source matrix lane,
      canonicalYRing data left source matrix lane =
        canonicalYRing data right source matrix lane := by
    intro source matrix lane
    change
      yRingForAssignment data (data.assignment source) left.rPrime matrix lane =
        yRingForAssignment data (data.assignment source) right.rPrime matrix lane
    rw [rPrime_eq]
  constructor
  · intro bound source matrix lane
    exact (bound source matrix lane).trans (canonical_eq source matrix lane)
  · intro bound source matrix lane
    exact (bound source matrix lane).trans (canonical_eq source matrix lane).symm

/-- The delayed-NC `yZcol` authority predicate depends only on the column
point. Changing the independently derived FE row point cannot change it. -/
theorem yZcolBoundToSources_iff_of_sPrime_eq
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (left right : VerifierPoints shape domain)
    (claims : Claims shape)
    (sPrime_eq : left.sPrime = right.sPrime) :
    YZcolBoundToSources covers data left claims <->
      YZcolBoundToSources covers data right claims := by
  have canonical_eq : forall source lane,
      canonicalYZcol covers data left source lane =
        canonicalYZcol covers data right source lane := by
    intro source lane
    change
      yZcolForAssignment covers (data.assignment source) left.sPrime lane =
        yZcolForAssignment covers (data.assignment source) right.sPrime lane
    rw [sPrime_eq]
  constructor
  · intro bound source lane
    exact (bound source lane).trans (canonical_eq source lane)
  · intro bound source lane
    exact (bound source lane).trans (canonical_eq source lane).symm

/-- Independent semantic authority contract for a claimed product. This is a
predicate to be established by a future verifier refinement, not acceptance. -/
structure BoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (claims : Claims shape) : Prop where
  yRing : YRingBoundToSources data points claims
  yZcol : YZcolBoundToSources covers data points claims

/-- The canonical product satisfies the CE/extraction-owned `yRing` branch. -/
theorem canonicalClaims_yRingBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain) :
    YRingBoundToSources data points (canonicalClaims covers data points) := by
  simp [YRingBoundToSources, canonicalClaims]

/-- The canonical product satisfies the independently owned SplitNC sidecar
branch. -/
theorem canonicalClaims_yZcolBoundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain) :
    YZcolBoundToSources covers data points
      (canonicalClaims covers data points) := by
  simp [YZcolBoundToSources, canonicalClaims]

/-- Model-level completeness: the source-derived construction satisfies every
coordinate of the independent authority contract. -/
theorem canonicalClaims_boundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain) :
    BoundToSources covers data points (canonicalClaims covers data points) := by
  exact {
    yRing := canonicalClaims_yRingBoundToSources covers data points
    yZcol := canonicalClaims_yZcolBoundToSources covers data points
  }

/-- Model-level uniqueness: no two different active products can satisfy all
source-derived output equations at the same verifier-owned points. -/
theorem eq_canonicalClaims_of_boundToSources
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {covers : domain.Covers shape}
    {data : SplitNc.Sources.Data shape}
    {points : VerifierPoints shape domain}
    {claims : Claims shape}
    (bound : BoundToSources covers data points claims) :
    claims = canonicalClaims covers data points := by
  apply Claims.ext
  · exact bound.yRing
  · exact bound.yZcol

/-- Exact characterization of the independent authority contract. -/
theorem boundToSources_iff_eq_canonicalClaims
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {covers : domain.Covers shape}
    {data : SplitNc.Sources.Data shape}
    {points : VerifierPoints shape domain}
    {claims : Claims shape} :
    BoundToSources covers data points claims <->
      claims = canonicalClaims covers data points := by
  constructor
  · exact eq_canonicalClaims_of_boundToSources
  · intro equal
    subst claims
    exact canonicalClaims_boundToSources covers data points

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
