import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PointBridge

/-!
Typed decoding of the prospective ordered-commitment source message.

Assurance tier: artifact-checked layout plus model-level representation
refinement.

Owns: exact-length decoding of the committed PiDEC artifact's parent point
and fourteen child commitment blocks into the independent typed accumulator
carrier, and equality between reserialization of that carrier and the raw
field message.

Does not own: active-production shape alignment, child CE membership, selected
NIFS semantics, an emitted Rust hash call, Poseidon2 rows, collision resistance,
costs, or row removal.

Emits constraints: no.

Authority boundary: `shape` is caller supplied only to index the independent
semantic point type; `pointDimension` must prove that its row dimension equals
the artifact's physical point-pair count. This artifact has one point
coordinate and must not be presented as the separate twelve-variable
Fibonacci diagnostic or as a complete active-production relation.

| Stage path | Mathematical obligation | Authority class | Physical source | Lean owner |
|---|---|---|---|---|
| `fprime.accumulator.ordered_commitments.decode.point` | decode every parent `r` pair in order at the checked semantic dimension | checked/computed | `layout.parent.rCols` | `decodedPoint`, `encodeDecodedPoint` |
| `fprime.accumulator.ordered_commitments.decode.child` | decode each exact 18-by-54 commitment block without defaults | checked/computed | `childLayout child.commitment.dataCols` | `decodedChild`, `childFields_length` |
| `fprime.accumulator.ordered_commitments.decode.payload` | retain the typed point and child index order | computed | point then children 0 through 13 | `decodedPayload` |
| `fprime.accumulator.ordered_commitments.decode.serialize` | typed reserialization equals the exact 13,620-field raw message | derived | domain, point, children | `serialize_decodedPayload` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentTypedDecoder

open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec
open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage
open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

private theorem childLayout_mem
    (child : Fin productionGlobalParams.k) :
    childLayout child ∈ layout.children := by
  unfold childLayout
  exact List.get_mem layout.children (Fin.cast production_child_count child)

theorem parentCommitmentColumns_length :
    layout.parent.commitment.dataCols.length =
      productionProfile.commitmentWidth * ringDegree := by
  simp [layout, productionProfile, ringDegree]

/-- Canonical field values of one physical child commitment block. -/
def childFields
    (assignment : Nat -> Nat)
    (child : Fin productionGlobalParams.k) : List F :=
  values assignment (childLayout child).commitment.dataCols

@[simp] theorem childFields_length
    (assignment : Nat -> Nat)
    (child : Fin productionGlobalParams.k) :
    (childFields assignment child).length =
      productionProfile.commitmentWidth * ringDegree := by
  rw [childFields, values, List.length_map]
  rw [production_public_shape.commitmentLengths
    (childLayout child) (childLayout_mem child)]
  exact parentCommitmentColumns_length

/-- Checked row-major decoder for one child commitment. -/
def decodedChild
    (assignment : Nat -> Nat)
    (child : Fin productionGlobalParams.k) : FixedCommitment :=
  decodeCommitmentOfLength (childFields assignment child)
    (childFields_length assignment child)

/-- Checked typed point carried by the artifact's common PiDEC parent. -/
def decodedPoint
    (shape : Shape)
    (assignment : Nat -> Nat)
    (pointDimension : layout.parent.rCols.length = shape.rowVariables) :
    Point shape :=
  PointBridge.pointOfLength shape assignment
    { r := layout.parent.rCols } pointDimension

/-- Complete typed ordered-child payload decoded from the artifact. -/
def decodedPayload
    (shape : Shape)
    (assignment : Nat -> Nat)
    (pointDimension : layout.parent.rCols.length = shape.rowVariables) :
    FixedCommitmentFamilyPayload shape where
  point := decodedPoint shape assignment pointDimension
  children := decodedChild assignment

def parentPointFields (assignment : Nat -> Nat) : List F :=
  ((pairColumns layout.parent.rCols).map assignment).map residue

def orderedChildFields (assignment : Nat -> Nat) : List F :=
  layout.children.flatMap fun child =>
    child.commitment.dataCols.map fun column => residue (assignment column)

/-- The typed point codec exactly recovers the artifact's pair-major parent
point fields. -/
theorem encodeDecodedPoint
    (shape : Shape)
    (assignment : Nat -> Nat)
    (pointDimension : layout.parent.rCols.length = shape.rowVariables) :
    encodePoint (decodedPoint shape assignment pointDimension) =
      parentPointFields assignment := by
  rw [encodePoint_eq_flatten_map]
  simp [encodeK, decodedPoint, PointBridge.pointOfLength,
    decodePointColumns, extensionValues, extensionValue, parentPointFields,
    pairColumns, layout]

private theorem childLayouts :
    List.ofFn childLayout = layout.children := by
  apply List.ext_get
  · simpa using production_child_count
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    unfold childLayout
    rfl

private theorem childFieldBlocks
    (assignment : Nat -> Nat) :
    (List.ofFn (childFields assignment)).flatten =
      orderedChildFields assignment := by
  rw [orderedChildFields, ← childLayouts]
  simp only [List.flatMap, List.map_ofFn]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext child
  rfl

/-- The typed child codec exactly recovers all fourteen flat commitment
blocks in artifact child order. -/
theorem encodeDecodedChildren (assignment : Nat -> Nat) :
    encodeChildren (decodedChild assignment) =
      orderedChildFields assignment := by
  exact (encodeChildren_decodeCommitmentOfLength
    (childFields assignment) (childFields_length assignment)).trans
      (childFieldBlocks assignment)

theorem payloadFields_decomposition (assignment : Nat -> Nat) :
    payloadFields assignment =
      domainFields ++ parentPointFields assignment ++
        orderedChildFields assignment := by
  simp [payloadFields, payloadNats, domainFields_eq_residues,
    parentPointFields, orderedChildFields, List.map_flatMap,
    Function.comp_def]

/-- Reserializing the checked typed payload recovers the exact raw artifact
message, including the fresh ten-field domain prefix. -/
theorem serialize_decodedPayload
    (shape : Shape)
    (assignment : Nat -> Nat)
    (pointDimension : layout.parent.rCols.length = shape.rowVariables) :
    serialize (decodedPayload shape assignment pointDimension) =
      payloadFields assignment := by
  rw [payloadFields_decomposition]
  simp [serialize, encodeCommitmentFamily, decodedPayload,
    encodeDecodedPoint, encodeDecodedChildren]

end Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentTypedDecoder
