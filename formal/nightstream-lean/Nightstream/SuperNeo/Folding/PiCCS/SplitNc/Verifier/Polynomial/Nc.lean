import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-!
Typed product-domain carrier for the independent Split-NC norm polynomial.

Protocol: SuperNeo `Pi_CCS`, split NC branch.
Phase: domain ownership before source projection or polynomial mixing.
Constraint family: column/lane point serialization only; this file emits no
rows.

Owns: the typed `{column, lane}` point, exact `column ++ lane` serialization,
fail-closed decoding, and canonical little-endian conversions between padded
numeric indices and Boolean vertices.

Does not own: source values, the NC range polynomial, gamma or equality
mixing, SumCheck messages, transcript derivation, output claims, Rust, R1CS,
row emission, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `SemanticShape` and `FlatNcDomain.Covers` remain the sole
semantic-shape and coverage owners. This module consumes those typed domains;
it does not infer coverage from a caller-selected coordinate list. Malformed
coordinate lists decode to `none`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.domain.coverage` | complete carrier and 54 active lanes fit in the padded domains | security boundary | `FlatNcDomain.Covers` |
| `nifs.pi_ccs.nc.domain.column_lane` | SumCheck coordinates are exactly `column ++ lane` | computed | `Point.coordinates` |
| `nifs.pi_ccs.nc.domain.decode` | only the exact total arity decodes | checked | `Point.decode` |
| `nifs.pi_ccs.nc.domain.column.index` | column vertices use the shared little-endian numeric order | computed | `columnIndex`, `columnVertex` |
| `nifs.pi_ccs.nc.domain.lane.index` | lane vertices use the shared little-endian numeric order | computed | `laneIndex`, `laneVertex` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Typed NC point. The serialized verifier order is column first, then lane. -/
structure Point (domain : FlatNcDomain) where
  column : CubePoint K domain.columnVariables
  lane : CubePoint K domain.laneVariables

namespace Point

/-- Product points are equal when both typed components are equal. -/
@[ext] theorem ext
    {domain : FlatNcDomain}
    (left right : Point domain)
    (column : left.column = right.column)
    (lane : left.lane = right.lane) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Exact NC coordinate serialization: column coordinates followed by lane
coordinates. -/
def coordinates
    {domain : FlatNcDomain}
    (point : Point domain) : List K :=
  point.column.coordinates ++ point.lane.coordinates

theorem coordinates_length
    {domain : FlatNcDomain}
    (point : Point domain) :
    point.coordinates.length =
      domain.columnVariables + domain.laneVariables := by
  simp [coordinates, point.column.dimension, point.lane.dimension]

/-- Internal exact-arity decoder used only after the public length check. -/
def ofCoordinates
    {domain : FlatNcDomain}
    (coordinates : List K)
    (length : coordinates.length =
      domain.columnVariables + domain.laneVariables) : Point domain where
  column := {
    coordinates := coordinates.take domain.columnVariables
    dimension := by
      rw [List.length_take, length]
      omega }
  lane := {
    coordinates := coordinates.drop domain.columnVariables
    dimension := by
      rw [List.length_drop, length]
      omega }

/-- Exact decoding is a left inverse of the fixed column-then-lane
serialization. -/
theorem ofCoordinates_coordinates
    {domain : FlatNcDomain}
    (point : Point domain)
    (length : point.coordinates.length =
      domain.columnVariables + domain.laneVariables) :
    ofCoordinates point.coordinates length = point := by
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    calc
      (point.column.coordinates ++ point.lane.coordinates).take
          domain.columnVariables =
          (point.column.coordinates ++ point.lane.coordinates).take
            point.column.coordinates.length :=
        congrArg
          (fun count =>
            (point.column.coordinates ++ point.lane.coordinates).take count)
          point.column.dimension.symm
      _ = point.column.coordinates := List.take_append_length
  · apply cubePoint_eq_of_coordinates_eq
    calc
      (point.column.coordinates ++ point.lane.coordinates).drop
          domain.columnVariables =
          (point.column.coordinates ++ point.lane.coordinates).drop
            point.column.coordinates.length :=
        congrArg
          (fun count =>
            (point.column.coordinates ++ point.lane.coordinates).drop count)
          point.column.dimension.symm
      _ = point.lane.coordinates := List.drop_append_length

/-- Fail-closed NC point decoder. The length proof is verifier-derived inside
the successful branch and is not supplied by a prover. -/
def decode
    {domain : FlatNcDomain}
    (coordinates : List K) : Option (Point domain) :=
  if length : coordinates.length =
      domain.columnVariables + domain.laneVariables then
    some (ofCoordinates coordinates length)
  else
    none

/-- A serialized typed point decodes to exactly that point. -/
theorem decode_coordinates
    {domain : FlatNcDomain}
    (point : Point domain) :
    decode point.coordinates = some point := by
  unfold decode
  rw [dif_pos point.coordinates_length]
  rw [ofCoordinates_coordinates]

/-- Wrong-arity NC points are rejected rather than truncated or padded. -/
theorem decode_eq_none_of_length_ne
    {domain : FlatNcDomain}
    (coordinates : List K)
    (different : coordinates.length ≠
      domain.columnVariables + domain.laneVariables) :
    decode (domain := domain) coordinates = (none : Option (Point domain)) := by
  simp [decode, different]

end Point

/-- Canonical padded-column index of a typed Boolean vertex. -/
def columnIndex
    {domain : FlatNcDomain}
    (vertex : BooleanVertex domain.columnVariables) :
    Fin domain.columnCount :=
  ⟨NumericBooleanDomain.index vertex, by
    simpa [FlatNcDomain.columnCount] using
      NumericBooleanDomain.index_lt_twoPow vertex⟩

/-- Canonical padded-lane index of a typed Boolean vertex. -/
def laneIndex
    {domain : FlatNcDomain}
    (vertex : BooleanVertex domain.laneVariables) :
    Fin domain.laneCount :=
  ⟨NumericBooleanDomain.index vertex, by
    simpa [FlatNcDomain.laneCount] using
      NumericBooleanDomain.index_lt_twoPow vertex⟩

/-- Canonical little-endian Boolean vertex of a padded-column index. -/
def columnVertex
    {domain : FlatNcDomain}
    (column : Fin domain.columnCount) :
    BooleanVertex domain.columnVariables :=
  NumericBooleanDomain.vertex domain.columnVariables column

/-- Canonical little-endian Boolean vertex of a padded-lane index. -/
def laneVertex
    {domain : FlatNcDomain}
    (lane : Fin domain.laneCount) :
    BooleanVertex domain.laneVariables :=
  NumericBooleanDomain.vertex domain.laneVariables lane

@[simp] theorem columnIndex_columnVertex
    {domain : FlatNcDomain}
    (column : Fin domain.columnCount) :
    columnIndex (columnVertex column) = column := by
  apply Fin.ext
  exact NumericBooleanDomain.index_vertex domain.columnVariables column

@[simp] theorem laneIndex_laneVertex
    {domain : FlatNcDomain}
    (lane : Fin domain.laneCount) :
    laneIndex (laneVertex lane) = lane := by
  apply Fin.ext
  exact NumericBooleanDomain.index_vertex domain.laneVariables lane

@[simp] theorem columnVertex_columnIndex
    {domain : FlatNcDomain}
    (vertex : BooleanVertex domain.columnVariables) :
    columnVertex (columnIndex vertex) = vertex := by
  exact NumericBooleanDomain.vertex_index vertex

@[simp] theorem laneVertex_laneIndex
    {domain : FlatNcDomain}
    (vertex : BooleanVertex domain.laneVariables) :
    laneVertex (laneIndex vertex) = vertex := by
  exact NumericBooleanDomain.vertex_index vertex

/-- Canonical field point for a pair of padded Boolean-domain indices. -/
def booleanPoint
    {domain : FlatNcDomain}
    (column : Fin domain.columnCount)
    (lane : Fin domain.laneCount) : Point domain where
  column := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
    (columnVertex column)
  lane := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
    (laneVertex lane)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
