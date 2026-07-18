import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath

/-!
Independent source projection for the Split-NC norm polynomial.

Protocol: SuperNeo `Pi_CCS`, split NC branch.
Phase: authoritative full-carrier source table to its nested Boolean-table MLE.
Constraint family: source projection and strict-`b = 2` cubic only; this file
emits no rows.

Owns: the complete padded column/lane table derived from `Sources.Data` and
`Semantics.Nc.diagonal`; zero column and lane tails; canonical column-then-lane
multilinear evaluation; its exact Boolean-point restriction; the concrete
extension-field cubic restriction; and its exact equivalence with independent
full-carrier norm truth.

Does not own: output `yZcol`, gamma or equality mixing, an initial claim,
SumCheck messages, transcript derivation, degree soundness, Rust, R1CS, row
emission, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: table leaves are computed solely from the independent
`Sources.Data.assignment` family. The caller provides no table value,
polynomial callback, output binding, or padding contents. `covers` ensures the
chosen Boolean domains contain the whole semantic carrier and all 54 lanes.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.source_projection.diagonal` | one live lane contains each full-carrier coefficient | computed | `paddedDiagonal`, `paddedDiagonal_live` |
| `nifs.pi_ccs.nc.source_projection.padding.column` | columns after `carrierWidth` are zero | computed | `paddedDiagonal_column_padding` |
| `nifs.pi_ccs.nc.source_projection.padding.lane` | lanes after 54 are zero | computed | `paddedDiagonal_lane_padding` |
| `nifs.pi_ccs.nc.source_projection.column_mle` | canonical column table is evaluated before lane interpolation | computed | `columnTable`, `columnValueAt` |
| `nifs.pi_ccs.nc.source_projection.lane_mle` | canonical lane table contains the column evaluations | computed | `laneTableAtColumn`, `sourceValueAt` |
| `nifs.pi_ccs.nc.source_projection.boolean` | nested MLE returns the exact padded leaf on the Boolean cube | derived | `sourceValueAt_toCubePoint_eq_embed_paddedDiagonal` |
| `nifs.pi_ccs.nc.range.cubic` | Boolean restriction is the embedded semantic cubic | derived | `rangeValueAt_toCubePoint_eq_embed_cubicResidual` |
| `nifs.pi_ccs.nc.range.boolean.completeness` | semantic truth zeros every live and padded Boolean cubic | derived | `booleanResidualsZero_of_truth` |
| `nifs.pi_ccs.nc.range.boolean.soundness` | every Boolean cubic zero implies full-carrier norm truth | checked | `truth_of_booleanResidualsZero` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.SourceProjection

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

/-- Complete padded NC table entry. Live semantic columns and active Phi81
lanes use the independent diagonal table; both padded tails are zero. -/
def paddedDiagonal
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (_covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin domain.columnCount)
    (lane : Fin domain.laneCount) : F :=
  if columnLive : column.val < shape.carrierWidth then
    if laneLive : lane.val < ringDegree then
      Semantics.Nc.diagonal (data.assignment source)
        ⟨column.val, columnLive⟩ ⟨lane.val, laneLive⟩
    else
      0
  else
    0

/-- Embedding a live carrier column and active lane preserves the exact
independent diagonal entry. -/
theorem paddedDiagonal_live
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin shape.carrierWidth)
    (lane : Fin ringDegree) :
    paddedDiagonal covers data source
        (domain.carrierColumn covers column)
        (domain.phi81Lane covers lane) =
      Semantics.Nc.diagonal (data.assignment source) column lane := by
  simp [paddedDiagonal]

/-- Every padded column after the complete semantic carrier is zero in every
lane. -/
theorem paddedDiagonal_column_padding
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin domain.columnCount)
    (lane : Fin domain.laneCount)
    (padding : shape.carrierWidth ≤ column.val) :
    paddedDiagonal covers data source column lane = 0 := by
  simp [paddedDiagonal, Nat.not_lt.mpr padding]

/-- Every padded lane after the 54 active Phi81 lanes is zero in every
column. -/
theorem paddedDiagonal_lane_padding
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin domain.columnCount)
    (lane : Fin domain.laneCount)
    (padding : ringDegree ≤ lane.val) :
    paddedDiagonal covers data source column lane = 0 := by
  simp [paddedDiagonal, Nat.not_lt.mpr padding]

/-- Canonical column table for one Boolean lane. Every leaf is source-derived
and embedded into the extension carrier. -/
def columnTable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (lane : BooleanVertex domain.laneVariables) :
    BooleanTable K domain.columnVariables :=
  BooleanTable.tabulate fun column =>
    K.embed <| paddedDiagonal covers data source
      (columnIndex column) (laneIndex lane)

/-- Evaluate the canonical column table before performing any lane
interpolation. -/
def columnValueAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : CubePoint K domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) : K :=
  (columnTable covers data source lane).evaluate
    ConcreteCarrier.extensionOps column

/-- Canonical lane table whose leaves are already-evaluated column MLEs. This
fixes the nested evaluation order as column first, then lane. -/
def laneTableAtColumn
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : CubePoint K domain.columnVariables) :
    BooleanTable K domain.laneVariables :=
  BooleanTable.tabulate fun lane =>
    columnValueAt covers data source column lane

/-- Nested canonical Boolean-table MLE of the complete padded source table. -/
def sourceValueAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (point : Point domain) : K :=
  (laneTableAtColumn covers data source point.column).evaluate
    ConcreteCarrier.extensionOps point.lane

/-- At canonical Boolean column and lane points, the nested MLE returns the
exact source-derived padded table entry. -/
theorem sourceValueAt_toCubePoint_eq_embed_paddedDiagonal
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : BooleanVertex domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) :
    sourceValueAt covers data source {
        column := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps column
        lane := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps lane } =
      K.embed (paddedDiagonal covers data source
        (columnIndex column) (laneIndex lane)) := by
  unfold sourceValueAt laneTableAtColumn
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
  rw [BooleanTable.valueAt_tabulate]
  unfold columnValueAt columnTable
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
  rw [BooleanTable.valueAt_tabulate]

/-- Numeric padded indices use the same Boolean-point restriction through the
shared little-endian index/vertex bijection. -/
theorem sourceValueAt_booleanPoint_eq_embed_paddedDiagonal
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin domain.columnCount)
    (lane : Fin domain.laneCount) :
    sourceValueAt covers data source (booleanPoint column lane) =
      K.embed (paddedDiagonal covers data source column lane) := by
  simpa [booleanPoint] using
    sourceValueAt_toCubePoint_eq_embed_paddedDiagonal
      covers data source (columnVertex column) (laneVertex lane)

/-- A live Boolean point evaluates to the authoritative full-carrier
diagonal entry. -/
theorem sourceValueAt_live
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin shape.carrierWidth)
    (lane : Fin ringDegree) :
    sourceValueAt covers data source
        (booleanPoint (domain.carrierColumn covers column)
          (domain.phi81Lane covers lane)) =
      K.embed (Semantics.Nc.diagonal
        (data.assignment source) column lane) := by
  rw [sourceValueAt_booleanPoint_eq_embed_paddedDiagonal]
  rw [paddedDiagonal_live]

/-- A padded column evaluates to extension zero. -/
theorem sourceValueAt_column_padding
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin domain.columnCount)
    (lane : Fin domain.laneCount)
    (padding : shape.carrierWidth ≤ column.val) :
    sourceValueAt covers data source (booleanPoint column lane) = K.zero := by
  rw [sourceValueAt_booleanPoint_eq_embed_paddedDiagonal]
  rw [paddedDiagonal_column_padding covers data source column lane padding]
  exact ConcreteCarrier.embed_zero

/-- A padded lane evaluates to extension zero. -/
theorem sourceValueAt_lane_padding
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin domain.columnCount)
    (lane : Fin domain.laneCount)
    (padding : ringDegree ≤ lane.val) :
    sourceValueAt covers data source (booleanPoint column lane) = K.zero := by
  rw [sourceValueAt_booleanPoint_eq_embed_paddedDiagonal]
  rw [paddedDiagonal_lane_padding covers data source column lane padding]
  exact ConcreteCarrier.embed_zero

/-- Concrete strict-`b = 2` cubic evaluated at the nested source projection. -/
def rangeValueAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (point : Point domain) : K :=
  let value := sourceValueAt covers data source point
  K.mul (K.mul (K.add value (K.embed 1)) value)
    (K.sub value (K.embed 1))

/-- On the Boolean product cube, the concrete extension cubic is exactly the
embedding of the independently defined semantic cubic residual. -/
theorem rangeValueAt_toCubePoint_eq_embed_cubicResidual
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : BooleanVertex domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) :
    rangeValueAt covers data source {
        column := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps column
        lane := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps lane } =
      K.embed (NormRange.cubicResidual
        (paddedDiagonal covers data source
          (columnIndex column) (laneIndex lane))) := by
  unfold rangeValueAt
  rw [sourceValueAt_toCubePoint_eq_embed_paddedDiagonal]
  exact NormRange.embed_cubicResidual _

/-- Numeric padded indices satisfy the same concrete cubic restriction. -/
theorem rangeValueAt_booleanPoint_eq_embed_cubicResidual
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin domain.columnCount)
    (lane : Fin domain.laneCount) :
    rangeValueAt covers data source (booleanPoint column lane) =
      K.embed (NormRange.cubicResidual
        (paddedDiagonal covers data source column lane)) := by
  simpa [booleanPoint] using
    rangeValueAt_toCubePoint_eq_embed_cubicResidual
      covers data source (columnVertex column) (laneVertex lane)

/-- At a live point the concrete cubic is exactly the authoritative diagonal
cubic, with no packed-output premise. -/
theorem rangeValueAt_live
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin shape.carrierWidth)
    (lane : Fin ringDegree) :
    rangeValueAt covers data source
        (booleanPoint (domain.carrierColumn covers column)
          (domain.phi81Lane covers lane)) =
      K.embed (NormRange.cubicResidual
        (Semantics.Nc.diagonal (data.assignment source) column lane)) := by
  rw [rangeValueAt_booleanPoint_eq_embed_cubicResidual]
  rw [paddedDiagonal_live]

/-- Every concrete cubic vanishes on the exact padded Boolean product domain.
This predicate is computed from `Sources.Data`; it contains no supplied
polynomial, output claim, or padding witness. -/
def BooleanResidualsZero
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) : Prop :=
  forall source column lane,
    rangeValueAt covers data source (booleanPoint column lane) = K.zero

/-- Independent full-carrier norm truth zeros the entire padded Boolean cubic
table. Live cells use semantic residual completeness, while both padded tails
are definitionally zero. No no-zero-divisor premise is needed. -/
theorem booleanResidualsZero_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) :
    Semantics.Nc.Truth data → BooleanResidualsZero covers data := by
  intro truth source column lane
  rw [rangeValueAt_booleanPoint_eq_embed_cubicResidual]
  have residualZero :
      NormRange.cubicResidual
          (paddedDiagonal covers data source column lane) = 0 := by
    by_cases columnLive : column.val < shape.carrierWidth
    · by_cases laneLive : lane.val < ringDegree
      · simpa [paddedDiagonal, columnLive, laneLive] using
          Semantics.Nc.residualsZero_of_truth data truth source
            ⟨column.val, columnLive⟩ ⟨lane.val, laneLive⟩
      · rw [paddedDiagonal_lane_padding covers data source column lane
          (Nat.le_of_not_gt laneLive)]
        rfl
    · rw [paddedDiagonal_column_padding covers data source column lane
          (Nat.le_of_not_gt columnLive)]
      rfl
  rw [residualZero]
  exact ConcreteCarrier.embed_zero

/-- Zero cubics on the exact padded Boolean domain imply independent
full-carrier norm truth. Restricting to each live typed column/lane recovers
the semantic residual family; only its root-classification step needs the
no-zero-divisors premise. -/
theorem truth_of_booleanResidualsZero
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) :
    BooleanResidualsZero covers data → Semantics.Nc.Truth data := by
  intro booleanResiduals
  apply Semantics.Nc.truth_of_residualsZero noZeroDivisors data
  intro source column lane
  have accepted := booleanResiduals source
    (domain.carrierColumn covers column) (domain.phi81Lane covers lane)
  rw [rangeValueAt_live covers data source column lane] at accepted
  have baseComponent := congrArg K.c0 accepted
  simpa only [K.embed, K.zero] using baseComponent

/-- The exact padded Boolean cubic relation is equivalent to independent
full-carrier norm truth. The premise is used only for soundness. -/
theorem booleanResidualsZero_iff_truth
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) :
    BooleanResidualsZero covers data <-> Semantics.Nc.Truth data := by
  exact ⟨truth_of_booleanResidualsZero noZeroDivisors covers data,
    booleanResidualsZero_of_truth covers data⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.SourceProjection
