import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Assignment semantics for the bounded fixed-point public-coordinate decoder.

Owns: interpretation of each generated coordinate owner and equality of the
resulting public vector with the independently defined typed
`FPrimeCarrier270` public projection.

Does not own: private assignment or matrix decoding, CCS/CE membership,
commitment-key alignment, producer authority, or row removal.

Emits constraints: no.

| Stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.public_decoder.owner` | 270 generated owners equal the canonical schedule | artifact-checked |
| `f_prime.fixed_point.public_decoder.value` | owner interpretation equals typed public carrier | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Wire
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

def sourceValue (source : Fin 257 → F) : RawSource → F
  | .constantOne => 1
  | .sourceField field =>
      if inRange : field < 257 then source ⟨field, inRange⟩ else 0
  | .fixedZero => 0

def artifactPublicValue (source : Fin 257 → F)
    (column : Fin PublicDecoder.alignedPublicWidth) : F :=
  sourceValue source (PublicDecoder.generatedCoordinate column).source

theorem artifactPublicValue_exact
    (source : Fin 257 → F)
    (constantOne : source ⟨0, by decide⟩ = 1)
    (column : Fin PublicDecoder.alignedPublicWidth) :
    artifactPublicValue source column =
      if inLegacy : column.val < 257 then
        source ⟨column.val, inLegacy⟩
      else 0 := by
  rw [artifactPublicValue, PublicDecoder.generatedCoordinate_exact]
  by_cases zero : column.val = 0
  · have columnEq : column = ⟨0, by decide⟩ := Fin.ext zero
    subst column
    simpa [PublicDecoder.expectedCoordinate, PublicDecoder.logicalPublicWidth,
      sourceValue] using constantOne.symm
  · by_cases inLegacy : column.val < 257
    · simp [PublicDecoder.expectedCoordinate, PublicDecoder.logicalPublicWidth, zero,
        inLegacy, sourceValue]
    · simp [PublicDecoder.expectedCoordinate, PublicDecoder.logicalPublicWidth, zero,
        inLegacy, sourceValue]

def sourcePublicPrefix (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) : Fin 257 → F :=
  fun column => legacy ⟨column.val,
    Nat.lt_of_lt_of_le column.isLt dimensions.legacyPublicFits⟩

def SourceConstantOne (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) : Prop :=
  legacy ⟨0, Nat.lt_of_lt_of_le (by decide) dimensions.legacyPublicFits⟩ = 1

def artifactColumn (dimensions : Dimensions)
    (column : Fin dimensions.shape.publicWidth) :
    Fin PublicDecoder.alignedPublicWidth :=
  ⟨column.val, by
    simpa [PublicDecoder.alignedPublicWidth] using column.isLt⟩

def artifactPublicInput (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) : PublicInput dimensions.shape :=
  fun column => artifactPublicValue (sourcePublicPrefix dimensions legacy)
    (artifactColumn dimensions column)

/-- Soundness of the generated public decoder against the independent typed
carrier. The only premise is the source relation's conventional constant-one
condition; padding and all other public owners are derived from the artifact. -/
theorem artifactPublicInput_eq_expectedPublicInput
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (constantOne : SourceConstantOne dimensions legacy) :
    artifactPublicInput dimensions legacy =
      expectedPublicInput dimensions legacy := by
  funext column
  have prefixOne :
      sourcePublicPrefix dimensions legacy ⟨0, by decide⟩ = 1 := by
    simpa [sourcePublicPrefix, SourceConstantOne] using constantOne
  rw [artifactPublicInput,
    artifactPublicValue_exact (sourcePublicPrefix dimensions legacy)
      prefixOne (artifactColumn dimensions column)]
  by_cases inLegacy : column.val < legacyPublicWidth
  · simp only [legacyPublicWidth] at inLegacy
    simp [artifactColumn, expectedPublicInput, sourcePublicPrefix,
      legacyPublicWidth, inLegacy]
  · simp only [legacyPublicWidth] at inLegacy
    simp [artifactColumn, expectedPublicInput, legacyPublicWidth, inLegacy]

/-- The artifact decoder constructs exactly the public input obtained by
projecting the independent complete carrier assignment. -/
theorem artifactPublicInput_eq_projectPublicInput
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (constantOne : SourceConstantOne dimensions legacy) :
    artifactPublicInput dimensions legacy =
      projectPublicInput (assignment dimensions legacy) := by
  rw [projectPublicInput_exact]
  exact artifactPublicInput_eq_expectedPublicInput dimensions legacy constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment
