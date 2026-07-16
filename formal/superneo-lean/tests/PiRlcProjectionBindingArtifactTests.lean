import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProjectionBindingShapeArtifact

namespace tests.PiRlcProjectionBindingArtifact

open SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
open SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
open SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifact

/-!
External regression checks for the generated fixed plain F-prime
projection-binding shape/count bridge.

| Check | Rust evidence | Lean guarantee | Open boundary |
|---|---|---|---|
| profile dimensions | exact identity trace and normalized zero-pin rows | artifact parameters equal the typed fixed profile | actual carrier/source-column refinement |
| role order | validated diagnostic roles | exact 18 + 5 + 6 + 2 model order | roles are not semantic authority |
| SIS geometry | two seeded-Phi81 blocks in the binding stage | 3,616 rank-2 words then 108 rank-1 words | ordered label/field-source sequence |
| count bridge | production long-block word count | modeled plain serializer has the same field count | equal counts do not imply equal messages |
-/

example :
    ArtifactData.inputCount = inputCount ∧
      ArtifactData.activeDegree = SuperNeo.d ∧
      ArtifactData.quotientDegree = SuperNeo.d - 1 ∧
      ArtifactData.commitmentLanes = commitmentLanes ∧
      ArtifactData.advCommitmentLanes = 0 ∧
      ArtifactData.activeXColumns = activeXColumns ∧
      ArtifactData.yRingRows = yRingRows ∧
      ArtifactData.extensionLimbs = extensionLimbs ∧
      ArtifactData.yZcolLimbs = extensionLimbs ∧
      ArtifactData.paddingTail = paddedDegree - SuperNeo.d ∧
      ArtifactData.paddedDegree = paddedDegree ∧
      ArtifactData.identityCount =
        commitmentLanes + activeXColumns +
          yRingRows * extensionLimbs + extensionLimbs :=
  artifact_profile_matches_model

example : ArtifactData.roles = expectedPlainRoles :=
  artifact_roles_exact

example :
    ArtifactData.projectionSisRowStart ≤ ArtifactData.bindingBlockRowStart ∧
      ArtifactData.bindingBlockRowEnd ≤ ArtifactData.compressionBlockRowStart ∧
      ArtifactData.compressionBlockRowEnd ≤ ArtifactData.projectionSisRowEnd ∧
      ArtifactData.sisBlockCount = 2 ∧
      ArtifactData.bindingPreimageFields = 3616 ∧
      ArtifactData.digestCompressionFields = 108 ∧
      ArtifactData.bindingKappa = 2 ∧
      ArtifactData.digestCompressionKappa = 1 ∧
      ArtifactData.balancedWordWidth = 41 :=
  artifact_sis_geometry_exact

example
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    (serializeProjectionBinding profile.material).length =
      ArtifactData.bindingPreimageFields :=
  artifact_plain_serializer_count_agrees shape

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifact.artifact_profile_matches_model' does not depend on any axioms -/
#guard_msgs in
#print axioms artifact_profile_matches_model

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifact.artifact_roles_exact' does not depend on any axioms -/
#guard_msgs in
#print axioms artifact_roles_exact

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifact.artifact_sis_geometry_exact' does not depend on any axioms -/
#guard_msgs in
#print axioms artifact_sis_geometry_exact

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifact.artifact_plain_serializer_count_agrees' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms artifact_plain_serializer_count_agrees

end tests.PiRlcProjectionBindingArtifact
