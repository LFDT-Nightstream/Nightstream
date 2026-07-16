import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.Generated.ProjectionBindingShapeArtifactData
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProjectionBindingSerialization

/-!
Owns: the numeric/profile bridge from the generated fixed plain F-prime Rust
artifact to the typed Pi_RLC projection-binding shape and serializer model.

Does not own: equality of the actual Rust and Lean preimage sequences, ordered
label conformance, source-field identity, native/circuit value equality, SIS
security, or R1CS row removal.

Emits constraints: no.

Authority boundary: the Rust generator proves that the production circuit's
long projection-SIS block consumes 3,616 balanced-ternary field words and that
its projection identities have the fixed plain profile. This file proves those
numbers instantiate the Lean model. It deliberately does not turn equal counts
into equality of serialized messages.

| Predicate/theorem | Rust evidence | Lean guarantee | Permits row removal? |
|---|---|---|---|
| `artifact_profile_matches_model` | validated roles, identity widths, normalized zero-pin rows | 15 inputs; 54/53 active/quotient widths; 18 + 5 + 6 + 2 identities; padded width 64 | No |
| `artifact_roles_exact` | exact diagnostic role order | generated roles equal the model's plain ownership partition | No |
| `artifact_sis_geometry_exact` | two seeded-Phi81 blocks inside the exact binding stage | 3,616 rank-2 field words followed by 108 rank-1 compression words | No |
| `artifact_plain_serializer_count_agrees` | production long-block word count | every typed plain profile's modeled serializer has the same length | No - sequence refinement open |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifact

open PiRlcChallenge
open ProjectionIdentityCertificateData

namespace ArtifactData

abbrev projectionSisRowStart :=
  ProjectionBindingShapeArtifactData.projectionSisRowStart
abbrev projectionSisRowEnd :=
  ProjectionBindingShapeArtifactData.projectionSisRowEnd
abbrev bindingBlockRowStart :=
  ProjectionBindingShapeArtifactData.bindingBlockRowStart
abbrev bindingBlockRowEnd :=
  ProjectionBindingShapeArtifactData.bindingBlockRowEnd
abbrev compressionBlockRowStart :=
  ProjectionBindingShapeArtifactData.compressionBlockRowStart
abbrev compressionBlockRowEnd :=
  ProjectionBindingShapeArtifactData.compressionBlockRowEnd
abbrev inputCount := ProjectionBindingShapeArtifactData.inputCount
abbrev activeDegree := ProjectionBindingShapeArtifactData.activeDegree
abbrev quotientDegree := ProjectionBindingShapeArtifactData.quotientDegree
abbrev commitmentLanes := ProjectionBindingShapeArtifactData.commitmentLanes
abbrev advCommitmentLanes := ProjectionBindingShapeArtifactData.advCommitmentLanes
abbrev activeXColumns := ProjectionBindingShapeArtifactData.activeXColumns
abbrev yRingRows := ProjectionBindingShapeArtifactData.yRingRows
abbrev extensionLimbs := ProjectionBindingShapeArtifactData.extensionLimbs
abbrev yZcolLimbs := ProjectionBindingShapeArtifactData.yZcolLimbs
abbrev identityCount := ProjectionBindingShapeArtifactData.identityCount
abbrev paddingTail := ProjectionBindingShapeArtifactData.paddingTail
abbrev paddedDegree := ProjectionBindingShapeArtifactData.paddedDegree
abbrev sisBlockCount := ProjectionBindingShapeArtifactData.sisBlockCount
abbrev bindingPreimageFields :=
  ProjectionBindingShapeArtifactData.bindingPreimageFields
abbrev digestCompressionFields :=
  ProjectionBindingShapeArtifactData.digestCompressionFields
abbrev bindingKappa := ProjectionBindingShapeArtifactData.bindingKappa
abbrev digestCompressionKappa :=
  ProjectionBindingShapeArtifactData.digestCompressionKappa
abbrev balancedWordWidth :=
  ProjectionBindingShapeArtifactData.balancedWordWidth
abbrev roles := ProjectionBindingShapeArtifactData.roles

end ArtifactData

/-- Model ownership order for the plain fixed profile. -/
def expectedPlainRoles : List IdentityRole :=
  (List.range PiRlcAlgebra.commitmentLanes).map IdentityRole.commitmentLane ++
    (List.range PiRlcAlgebra.activeXColumns).map IdentityRole.activeXColumn ++
    ((List.range PiRlcAlgebra.yRingRows).flatMap fun row =>
      (List.range PiRlcAlgebra.extensionLimbs).map fun limb =>
        IdentityRole.yRingLimb row limb) ++
    (List.range PiRlcAlgebra.extensionLimbs).map IdentityRole.yZcolLimb

/-- Generated fixed-profile dimensions instantiate the typed Lean parameters. -/
theorem artifact_profile_matches_model :
    ArtifactData.inputCount = PiRlcAlgebra.inputCount ∧
      ArtifactData.activeDegree = SuperNeo.d ∧
      ArtifactData.quotientDegree = SuperNeo.d - 1 ∧
      ArtifactData.commitmentLanes = PiRlcAlgebra.commitmentLanes ∧
      ArtifactData.advCommitmentLanes = 0 ∧
      ArtifactData.activeXColumns = PiRlcAlgebra.activeXColumns ∧
      ArtifactData.yRingRows = PiRlcAlgebra.yRingRows ∧
      ArtifactData.extensionLimbs = PiRlcAlgebra.extensionLimbs ∧
      ArtifactData.yZcolLimbs = PiRlcAlgebra.extensionLimbs ∧
      ArtifactData.paddingTail = PiRlcAlgebra.paddedDegree - SuperNeo.d ∧
      ArtifactData.paddedDegree = PiRlcAlgebra.paddedDegree ∧
      ArtifactData.identityCount =
        PiRlcAlgebra.commitmentLanes + PiRlcAlgebra.activeXColumns +
          PiRlcAlgebra.yRingRows * PiRlcAlgebra.extensionLimbs +
          PiRlcAlgebra.extensionLimbs := by
  decide

/-- The generated role list is the model's exact plain ownership partition. -/
theorem artifact_roles_exact :
    ArtifactData.roles = expectedPlainRoles := by
  decide

/-- Exact numeric geometry of the two production projection-SIS blocks. -/
theorem artifact_sis_geometry_exact :
    ArtifactData.projectionSisRowStart ≤ ArtifactData.bindingBlockRowStart ∧
      ArtifactData.bindingBlockRowEnd ≤ ArtifactData.compressionBlockRowStart ∧
      ArtifactData.compressionBlockRowEnd ≤ ArtifactData.projectionSisRowEnd ∧
      ArtifactData.sisBlockCount = 2 ∧
      ArtifactData.bindingPreimageFields = 3616 ∧
      ArtifactData.digestCompressionFields = 108 ∧
      ArtifactData.bindingKappa = 2 ∧
      ArtifactData.digestCompressionKappa = 1 ∧
      ArtifactData.balancedWordWidth = 41 := by
  decide

/--
Count bridge only: a typed plain-profile serializer and the production long
SIS block contain equally many fields. This theorem does not identify their
ordered elements.
-/
theorem artifact_plain_serializer_count_agrees
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    (serializeProjectionBinding profile.material).length =
      ArtifactData.bindingPreimageFields := by
  simpa [ArtifactData.bindingPreimageFields,
    ProjectionBindingShapeArtifactData.bindingPreimageFields] using
    plainFixedProfile_serialized_length shape

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifact
