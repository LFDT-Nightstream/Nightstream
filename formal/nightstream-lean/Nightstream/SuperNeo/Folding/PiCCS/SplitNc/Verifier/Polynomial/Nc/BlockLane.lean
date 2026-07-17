import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Typed product domain for the canonical Split-NC block×lane polynomial.

Assurance tier: model-level.

Owns: the typed `{block, lane}` point, exact block-then-lane serialization,
fail-closed decoding, and canonical little-endian conversions between padded
numeric indices and Boolean vertices.

Does not own: source values, the range polynomial, mixing, SumCheck,
transcripts, output claims, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: `BlockNcDomain.Covers` is the separate coverage
obligation. This module only fixes the product-domain representation and
coordinate order; malformed coordinate lists decode to `none`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.domain.point` | one typed block point and one typed lane point | computed | `Point` |
| `nifs.pi_ccs.nc.block_lane.domain.coordinates` | serialization is exactly `block ++ lane` | computed | `Point.coordinates` |
| `nifs.pi_ccs.nc.block_lane.domain.decode` | only the exact total arity decodes | checked | `Point.decode` |
| `nifs.pi_ccs.nc.block_lane.domain.block_index` | block vertices use the shared little-endian codec | computed | `BlockNcDomain.blockIndex`, `BlockNcDomain.blockVertex` |
| `nifs.pi_ccs.nc.block_lane.domain.lane_index` | lane vertices use the shared little-endian codec | computed | `BlockNcDomain.laneIndex`, `BlockNcDomain.laneVertex` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open BlockNcDomain

/-- Typed block×lane point. The verifier order is block first, then lane. -/
structure Point (domain : BlockNcDomain) where
  block : CubePoint K domain.blockVariables
  lane : CubePoint K domain.laneVariables

namespace Point

@[ext] theorem ext
    {domain : BlockNcDomain}
    (left right : Point domain)
    (block : left.block = right.block)
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

/-- Exact block-then-lane coordinate serialization. -/
def coordinates
    {domain : BlockNcDomain}
    (point : Point domain) : List K :=
  point.block.coordinates ++ point.lane.coordinates

theorem coordinates_length
    {domain : BlockNcDomain}
    (point : Point domain) :
    point.coordinates.length =
      domain.blockVariables + domain.laneVariables := by
  simp [coordinates, point.block.dimension, point.lane.dimension]

/-- Internal exact-arity decoder used only under the public length check. -/
def ofCoordinates
    {domain : BlockNcDomain}
    (coordinates : List K)
    (length : coordinates.length =
      domain.blockVariables + domain.laneVariables) : Point domain where
  block := {
    coordinates := coordinates.take domain.blockVariables
    dimension := by
      rw [List.length_take, length]
      omega }
  lane := {
    coordinates := coordinates.drop domain.blockVariables
    dimension := by
      rw [List.length_drop, length]
      omega }

/-- Exact decoding is a left inverse of block-then-lane serialization. -/
theorem ofCoordinates_coordinates
    {domain : BlockNcDomain}
    (point : Point domain)
    (length : point.coordinates.length =
      domain.blockVariables + domain.laneVariables) :
    ofCoordinates point.coordinates length = point := by
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    calc
      (point.block.coordinates ++ point.lane.coordinates).take
          domain.blockVariables =
          (point.block.coordinates ++ point.lane.coordinates).take
            point.block.coordinates.length :=
        congrArg
          (fun count =>
            (point.block.coordinates ++ point.lane.coordinates).take count)
          point.block.dimension.symm
      _ = point.block.coordinates := List.take_append_length
  · apply cubePoint_eq_of_coordinates_eq
    calc
      (point.block.coordinates ++ point.lane.coordinates).drop
          domain.blockVariables =
          (point.block.coordinates ++ point.lane.coordinates).drop
            point.block.coordinates.length :=
        congrArg
          (fun count =>
            (point.block.coordinates ++ point.lane.coordinates).drop count)
          point.block.dimension.symm
      _ = point.lane.coordinates := List.drop_append_length

/-- Fail-closed point decoder. The successful length proof is computed by
the verifier branch and is not a certificate field. -/
def decode
    {domain : BlockNcDomain}
    (coordinates : List K) : Option (Point domain) :=
  if length : coordinates.length =
      domain.blockVariables + domain.laneVariables then
    some (ofCoordinates coordinates length)
  else
    none

/-- A serialized typed point decodes to that exact point. -/
theorem decode_coordinates
    {domain : BlockNcDomain}
    (point : Point domain) :
    decode point.coordinates = some point := by
  unfold decode
  rw [dif_pos point.coordinates_length]
  rw [ofCoordinates_coordinates]

/-- Wrong-arity points reject rather than truncating or padding. -/
theorem decode_eq_none_of_length_ne
    {domain : BlockNcDomain}
    (coordinates : List K)
    (different : coordinates.length ≠
      domain.blockVariables + domain.laneVariables) :
    decode (domain := domain) coordinates = (none : Option (Point domain)) := by
  simp [decode, different]

end Point

/-- Canonical field point for padded block and lane indices. -/
def booleanPoint
    {domain : BlockNcDomain}
    (block : Fin domain.blockCount)
    (lane : Fin domain.laneCount) : Point domain where
  block := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
    (blockVertex block)
  lane := BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
    (laneVertex lane)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
