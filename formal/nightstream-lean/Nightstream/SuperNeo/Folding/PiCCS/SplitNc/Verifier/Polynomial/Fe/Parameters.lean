import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types

/-!
Typed domain and physical degree parameters shared by the Split-NC FE
polynomial and its executable SumCheck interface.

Owns: the row-then-lane FE point, exact coordinate serialization and decoding,
the syntax-derived row degree ceiling, and the fixed quadratic lane ceiling.

Does not own: FE polynomial values, source assignments, semantic truth,
honest messages, SumCheck replay, transcript derivation, Rust, R1CS, rows,
costs, or row removal.

Emits constraints: no.

Authority boundary: the point shape and degree ceilings derive only from raw
verifier-visible dimensions and CCS terms. This module imports neither
`SplitNc.Sources` nor `SplitNc.Semantics`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.domain.row_lane` | serialize FE points exactly as `row ++ lane` | computed | `Point.coordinates`, `Point.ofCoordinates` |
| `nifs.pi_ccs.fe.degree.row` | derive the row ceiling from explicit sparse CCS terms | computed | `rowSumcheckDegreeBound` |
| `nifs.pi_ccs.fe.degree.lane` | fix the independently proved lane ceiling at two | protocol parameter | `laneSumcheckDegreeBound` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Typed FE terminal point. Mathematical notation writes `(lane,row)`, while
the executable SumCheck serialization is fixed separately as `row ++ lane`. -/
structure Point (shape : SemanticShape) (domain : FlatNcDomain) where
  row : CubePoint K shape.rowVariables
  lane : CubePoint K domain.laneVariables

namespace Point

/-- Product points are equal when their two typed components are equal. -/
@[ext] theorem ext
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (left right : Point shape domain)
    (row : left.row = right.row)
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

/-- Exact production coordinate serialization: row coordinates first, then
lane/Ajtai coordinates. -/
def coordinates
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (point : Point shape domain) : List K :=
  point.row.coordinates ++ point.lane.coordinates

theorem coordinates_length
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (point : Point shape domain) :
    point.coordinates.length = shape.rowVariables + domain.laneVariables := by
  simp [coordinates, point.row.dimension, point.lane.dimension]

/-- Decode the fixed row-then-lane serialization. The proof argument is
derived by the verifier's exact round-count check, not supplied by a prover. -/
def ofCoordinates
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (coordinates : List K)
    (length : coordinates.length =
      shape.rowVariables + domain.laneVariables) : Point shape domain where
  row := {
    coordinates := coordinates.take shape.rowVariables
    dimension := by
      rw [List.length_take, length]
      omega }
  lane := {
    coordinates := coordinates.drop shape.rowVariables
    dimension := by
      rw [List.length_drop, length]
      omega }

/-- Decoding the serialization of a typed point returns that exact point. -/
theorem ofCoordinates_coordinates
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (point : Point shape domain)
    (length : point.coordinates.length =
      shape.rowVariables + domain.laneVariables) :
    ofCoordinates point.coordinates length = point := by
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    calc
      (point.row.coordinates ++ point.lane.coordinates).take
          shape.rowVariables =
          (point.row.coordinates ++ point.lane.coordinates).take
            point.row.coordinates.length :=
        congrArg
          (fun count =>
            (point.row.coordinates ++ point.lane.coordinates).take count)
          point.row.dimension.symm
      _ = point.row.coordinates := List.take_append_length
  · apply cubePoint_eq_of_coordinates_eq
    calc
      (point.row.coordinates ++ point.lane.coordinates).drop
          shape.rowVariables =
          (point.row.coordinates ++ point.lane.coordinates).drop
            point.row.coordinates.length :=
        congrArg
          (fun count =>
            (point.row.coordinates ++ point.lane.coordinates).drop count)
          point.row.dimension.symm
      _ = point.lane.coordinates := List.drop_append_length

end Point

/-- Syntax-derived FE row-round ceiling. Declared CCS degree metadata is not
verifier authority. -/
def rowSumcheckDegreeBound
    {shape : SemanticShape}
    (input : PublicInput shape) : Nat :=
  Nat.max
    input.constraintPolynomial.canonicalEqualityGatedDegreeBound 2

/-- FE lane rounds are quadratic independently of the CCS syntax: fresh CCS
values are lane-constant before the selector, while carried values are one
lane MLE times one selector. -/
def laneSumcheckDegreeBound : Nat := 2

/-- Two inputs with identical explicit CCS terms have the same FE degree
ceiling, even if their declared metadata differs. -/
theorem rowSumcheckDegreeBound_eq_of_terms_eq
    {shape : SemanticShape}
    (left right : PublicInput shape)
    (terms : left.constraintPolynomial.terms =
      right.constraintPolynomial.terms) :
    rowSumcheckDegreeBound left = rowSumcheckDegreeBound right := by
  unfold rowSumcheckDegreeBound
  rw [CCSResidualTable.ConstraintPolynomial.canonicalEqualityGatedDegreeBound_eq_of_terms_eq
    left.constraintPolynomial right.constraintPolynomial terms]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
