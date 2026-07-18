import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSemantics

/-!
Independent shape profile for the `Pi_CCS` output-digest preimage.

Assurance tier: model-level representation semantics.

Owns: source/matrix counts, the exact serializer field-count formula, and the
distinct three- and thirteen-matrix specializations. Does not own relation
selection, output authority, hashing, physical columns, artifacts, or costs.

Authority boundary: thirteen matrices describe the current steady fixed-point
relation only; they are not a universal HyperNova or SuperNeo constant. A
separate refinement must prove which verifier-selected relation production uses.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest.profile.shape` | retain only source and matrix counts | verifier-selected shape | `Profile` |
| `nifs.pi_ccs.output_digest.profile.fields` | `8 + sources * (9 + (matrices + 1) * (1 + 2 * 54))` | computed | `fieldCount` |
| `nifs.pi_ccs.output_digest.profile.semantic` | agree with the typed serializer count | derived | `fieldCount_ofSemanticShape` |
| `nifs.pi_ccs.output_digest.profile.diagnostic` | direct A/B/C fixture has 6,683 fields | diagnostic specialization | `diagnosticThreeMatrix_fieldCount` |
| `nifs.pi_ccs.output_digest.profile.fixed_point` | current 13-port fixed point has 23,033 fields | model specialization | `steadyFixedPointThirteenMatrix_fieldCount` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Batch and relation arity needed to size the lossless pre-hash message. -/
structure Profile where
  sourceCount : Nat
  matrixCount : Nat
deriving DecidableEq, Repr

namespace Profile

/-- Exact field count before SIS compression. The constants are the eight-field
outer header, nine-field source header, and one width plus two limbs per lane. -/
def fieldCount (profile : Profile) : Nat :=
  8 + profile.sourceCount *
    (9 + (profile.matrixCount + 1) * (1 + 2 * ringDegree))

/-- Forget all semantic dimensions that do not affect this serializer. -/
def ofSemanticShape (shape : SemanticShape) : Profile where
  sourceCount := shape.sourceCount
  matrixCount := shape.matrixCount

/-- The independent formula is exactly the count used by the typed serializer. -/
@[simp] theorem fieldCount_ofSemanticShape (shape : SemanticShape) :
    fieldCount (ofSemanticShape shape) = ActiveSemantics.fieldCount shape := by
  simp [fieldCount, ofSemanticShape, ActiveSemantics.fieldCount,
    ActiveSemantics.sourceFieldCount, ActiveSemantics.sourcePayloadFieldCount,
    Encoding.kVectorFieldCount, Nat.add_mul]

/-- Direct R1CS-to-CCS diagnostic: one fresh plus fourteen running sources and
the standard three A/B/C matrices. -/
def diagnosticThreeMatrix : Profile where
  sourceCount := 15
  matrixCount := 3

/-- Current steady fixed-point specialization: the same batch arity and the
thirteen ports of the selectively lowered recursive relation. -/
def steadyFixedPointThirteenMatrix : Profile where
  sourceCount := 15
  matrixCount := 13

@[simp] theorem diagnosticThreeMatrix_fieldCount :
    fieldCount diagnosticThreeMatrix = 6683 := by
  decide

@[simp] theorem steadyFixedPointThirteenMatrix_fieldCount :
    fieldCount steadyFixedPointThirteenMatrix = 23033 := by
  decide

theorem diagnosticThreeMatrix_ne_steadyFixedPointThirteenMatrix :
    diagnosticThreeMatrix ≠ steadyFixedPointThirteenMatrix := by
  decide

theorem diagnosticThreeMatrix_fieldCount_ne_steadyFixedPointThirteenMatrix :
    fieldCount diagnosticThreeMatrix ≠
      fieldCount steadyFixedPointThirteenMatrix := by
  decide

end Profile
end Nightstream.Implementation.R1CS.PiCcsOutputDigest
