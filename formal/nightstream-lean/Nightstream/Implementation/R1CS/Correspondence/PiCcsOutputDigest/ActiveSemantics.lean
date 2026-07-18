import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Encoding
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types

/-!
Shape-indexed `Pi_CCS` output-message semantics for the active F-prime NIFS.

Assurance tier: model-level representation semantics.

Owns: the protocol -> source -> matrix/vector -> lane -> limb encoding tree;
exact field order for every active Split-NC output coordinate; generic field
counts; and proof that the complete pre-SIS field serialization is injective.

Does not own: acceptance or source binding of `yRing`/`yZcol`; a concrete
13-matrix shape witness; SIS maps; Poseidon2; transcript placement; Rust/R1CS
columns; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: `serialize` consumes the complete typed Split-NC output
product and omits no source, matrix, or active Phi81 lane. Its injectivity
rules out representation-level aliases before compression. It does not make
the output claims true and does not treat an equal digest as equality without
a separately stated SIS/Poseidon2 binding assumption.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest` | one outer domain and exact source count | verifier-owned shape | `serialize` |
| `nifs.pi_ccs.output_digest.source` | preserve canonical source order | computed | `sourcePayloads`, `encodeSources` |
| `nifs.pi_ccs.output_digest.source.header` | bind the exact matrix count | verifier-owned shape | `encodeSource` |
| `nifs.pi_ccs.output_digest.source.y_ring` | preserve every matrix, active lane, and `(c0,c1)` limb | checked payload encoding | `encodeSourcePayload` |
| `nifs.pi_ccs.output_digest.source.y_zcol` | preserve every active lane and `(c0,c1)` limb | checked payload encoding | `encodeSourcePayload` |
| `nifs.pi_ccs.output_digest.injective` | equal field messages imply equal complete typed outputs | derived | `serialize_injective` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- One source branch of the complete output product. -/
structure SourcePayload (matrixCount : Nat) where
  yRing : Fin matrixCount -> Fin ringDegree -> K
  yZcol : Fin ringDegree -> K

@[ext] theorem SourcePayload.ext
    {matrixCount : Nat}
    (left right : SourcePayload matrixCount)
    (yRing : forall matrix lane,
      left.yRing matrix lane = right.yRing matrix lane)
    (yZcol : forall lane, left.yZcol lane = right.yZcol lane) :
    left = right := by
  cases left with
  | mk leftYRing leftYZcol =>
      cases right with
      | mk rightYRing rightYZcol =>
          have yRingEq : leftYRing = rightYRing := by
            funext matrix lane
            exact yRing matrix lane
          have yZcolEq : leftYZcol = rightYZcol := by
            funext lane
            exact yZcol lane
          cases yRingEq
          cases yZcolEq
          rfl

/-- Exact source-indexed view of the typed Split-NC output carrier. -/
def sourcePayload
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount) : SourcePayload shape.matrixCount where
  yRing := message.yRing source
  yZcol := message.yZcol source

def sourcePayloads
    {shape : SemanticShape}
    (message : OutputMessage shape) :
    Fin shape.sourceCount -> SourcePayload shape.matrixCount :=
  fun source => sourcePayload message source

@[simp] theorem sourcePayload_yRing
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    (sourcePayload message source).yRing matrix lane =
      message.yRing source matrix lane := by
  rfl

@[simp] theorem sourcePayload_yZcol
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree) :
    (sourcePayload message source).yZcol lane =
      message.yZcol source lane := by
  rfl

theorem sourcePayloads_injective {shape : SemanticShape} :
    Function.Injective
      (sourcePayloads :
        OutputMessage shape ->
          Fin shape.sourceCount -> SourcePayload shape.matrixCount) := by
  intro left right same
  apply Claims.ext
  · intro source matrix lane
    exact congrArg (fun payload => payload.yRing matrix lane)
      (congrFun same source)
  · intro source lane
    exact congrArg (fun payload => payload.yZcol lane)
      (congrFun same source)

/-- Dynamic field count for one source, excluding its fixed domain and matrix
count header. -/
def sourcePayloadFieldCount (matrixCount : Nat) : Nat :=
  matrixCount * Encoding.kVectorFieldCount ringDegree +
    Encoding.kVectorFieldCount ringDegree

/-- Matrix-major `yRing` vectors followed by the sole `yZcol` vector. -/
def encodeSourcePayload
    {matrixCount : Nat}
    (payload : SourcePayload matrixCount) : List F :=
  Encoding.encodeKVectorFamily payload.yRing ++
    Encoding.encodeKVector payload.yZcol

@[simp] theorem encodeSourcePayload_length
    {matrixCount : Nat}
    (payload : SourcePayload matrixCount) :
    (encodeSourcePayload payload).length =
      sourcePayloadFieldCount matrixCount := by
  simp [encodeSourcePayload, sourcePayloadFieldCount]

theorem encodeSourcePayload_injective {matrixCount : Nat} :
    Function.Injective
      (encodeSourcePayload : SourcePayload matrixCount -> List F) := by
  intro left right same
  let yRingFieldCount :=
    matrixCount * Encoding.kVectorFieldCount ringDegree
  have yRingFields := congrArg (List.take yRingFieldCount) same
  have yZcolFields := congrArg (List.drop yRingFieldCount) same
  have yRingEq : left.yRing = right.yRing := by
    apply Encoding.encodeKVectorFamily_injective
    simpa [encodeSourcePayload, yRingFieldCount] using yRingFields
  have yZcolEq : left.yZcol = right.yZcol := by
    apply Encoding.encodeKVector_injective
    simpa [encodeSourcePayload, yRingFieldCount] using yZcolFields
  cases left
  cases right
  cases yRingEq
  cases yZcolEq
  rfl

/-- Complete fixed-width source block: domain, matrix count, all `yRing`
vectors, then `yZcol`. -/
def encodeSource
    {matrixCount : Nat}
    (payload : SourcePayload matrixCount) : List F :=
  Encoding.outputMessageDomainFields ++
    Encoding.fieldOfNat matrixCount :: encodeSourcePayload payload

def sourceFieldCount (matrixCount : Nat) : Nat :=
  9 + sourcePayloadFieldCount matrixCount

@[simp] theorem encodeSource_length
    {matrixCount : Nat}
    (payload : SourcePayload matrixCount) :
    (encodeSource payload).length = sourceFieldCount matrixCount := by
  simp [encodeSource, sourceFieldCount]
  omega

theorem encodeSource_injective {matrixCount : Nat} :
    Function.Injective
      (encodeSource : SourcePayload matrixCount -> List F) := by
  intro left right same
  apply encodeSourcePayload_injective
  have tails := congrArg (List.drop 9) same
  simpa [encodeSource] using tails

/-- Ordered equal-width source blocks. -/
def encodeSources
    {sourceCount matrixCount : Nat}
    (payloads : Fin sourceCount -> SourcePayload matrixCount) : List F :=
  Encoding.encodeFamily encodeSource payloads

@[simp] theorem encodeSources_length
    {sourceCount matrixCount : Nat}
    (payloads : Fin sourceCount -> SourcePayload matrixCount) :
    (encodeSources payloads).length =
      sourceCount * sourceFieldCount matrixCount := by
  simp [encodeSources,
    Encoding.encodeFamily_length encodeSource (sourceFieldCount matrixCount)
      encodeSource_length]

theorem encodeSources_injective {sourceCount matrixCount : Nat} :
    Function.Injective
      (encodeSources :
        (Fin sourceCount -> SourcePayload matrixCount) -> List F) := by
  exact Encoding.encodeFamily_injective encodeSource (by
      simp [sourceFieldCount, sourcePayloadFieldCount]
      omega)
    encodeSource_length encodeSource_injective

/-- Complete active output preimage before SIS compression. -/
def serialize
    {shape : SemanticShape}
    (message : OutputMessage shape) : List F :=
  Encoding.outputsDomainFields ++
    Encoding.fieldOfNat shape.sourceCount ::
      encodeSources (sourcePayloads message)

def fieldCount (shape : SemanticShape) : Nat :=
  8 + shape.sourceCount * sourceFieldCount shape.matrixCount

@[simp] theorem serialize_length
    {shape : SemanticShape}
    (message : OutputMessage shape) :
    (serialize message).length = fieldCount shape := by
  simp [serialize, fieldCount]
  omega

/-- No source, matrix, lane, or limb can disappear before compression. -/
theorem serialize_injective {shape : SemanticShape} :
    Function.Injective
      (serialize : OutputMessage shape -> List F) := by
  intro left right same
  apply sourcePayloads_injective
  apply encodeSources_injective
  have tails := congrArg (List.drop 8) same
  simpa [serialize] using tails

/-- The active 15-source, 13-matrix profile has this many pre-SIS fields.
This is a representation length, not an R1CS row or column count. -/
theorem serialize_length_15_sources_13_matrices
    {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    (matrixCount : shape.matrixCount = 13)
    (message : OutputMessage shape) :
    (serialize message).length = 23033 := by
  rw [serialize_length]
  simp [fieldCount, sourceFieldCount, sourcePayloadFieldCount,
    Encoding.kVectorFieldCount, sourceCount, matrixCount, ringDegree]

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSemantics
