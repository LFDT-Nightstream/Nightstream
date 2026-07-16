import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProjectionBindingSerialization

namespace tests.PiRlcProjectionBindingShape

open SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
open SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-!
External regression and axiom-surface checks for the fixed Pi_RLC
projection-binding profile and its concrete serializer refinement.

| Check | Mathematical property | Production claim |
|---|---|---|
| active/padded pairing | A combined value is exactly `carrier.take 54`; the carrier has 64 entries and ten zero tail entries | Model-level only |
| plain family counts | 15 outputs and 31 projection lanes split as 18 + 5 + 6 + 2, with 6/2 paired padded carriers | Model-level only |
| plain serialization | Exact version-one framing emits 3,616 fields | Rust artifact and ordered-source refinement required |
| counterfactual same-X serialization | Adding all adv material without enlarging five-lane X emits 6,889 fields | Diagnostic only; not Nebula |
| axiom reports | Exported count theorems avoid native-compiler trust | Lean kernel report |
-/

example
    {carrier : List SuperNeo.F} {binding : ProjectionLaneBinding}
    (shape : PaddedProjectionLaneShape carrier binding) :
    binding.combined = carrier.take 54 ∧
      binding.combined.length = 54 ∧
      binding.quotient.length = 53 ∧
      carrier.length = 64 ∧
      carrier.drop 54 = List.replicate 10 (0 : SuperNeo.F) := by
  have laneShape := paddedProjectionLaneShape_projectionLaneShape shape
  exact ⟨paddedProjectionLaneShape_activePrefix shape,
    activePolynomialShape_length laneShape.combined,
    projectionQuotientShape_length laneShape.quotient,
    paddedCarrierShape_length shape.carrier_shape,
    paddedCarrierShape_zeroTail shape.carrier_shape⟩

example
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.outputCount = 15 ∧
      profile.material.commitmentQuotients.length = 18 ∧
      profile.material.x.length = 5 ∧
      profile.material.yRing.length = 6 ∧
      profile.yRingCarriers.length = 6 ∧
      profile.material.yZcol.length = 2 ∧
      profile.yZcolCarriers.length = 2 ∧
      List.Forall₂
        (fun carrier binding => binding.combined = carrier.take 54)
        profile.yRingCarriers profile.material.yRing ∧
      profile.yZcolCarriers.Forall
        (fun carrier =>
          carrier.drop 54 = List.replicate 10 (0 : SuperNeo.F)) ∧
      (serializeProjectionBinding profile.material).length = 3616 := by
  exact ⟨plainFixedProfile_outputCount shape,
    plainFixedProfile_commitmentQuotientCount shape,
    plainFixedProfile_xCount shape,
    plainFixedProfile_yRingCount shape,
    plainFixedProfile_yRingCarrierCount shape,
    plainFixedProfile_yZcolCount shape,
    plainFixedProfile_yZcolCarrierCount shape,
    plainFixedProfile_yRing_activePrefixes shape,
    plainFixedProfile_yZcol_zeroTails shape,
    plainFixedProfile_serialized_length shape⟩

example
    {profile : FixedProjectionProfile}
    (shape : CounterfactualAllAdvSameXProfileShape profile) :
    profile.material.combinedAdvLeaves.length = 3 ∧
      profile.material.advQuotients.length = 54 ∧
      profile.material.x.length = 5 ∧
      (serializeProjectionBinding profile.material).length = 6889 := by
  exact ⟨counterfactualAllAdvSameXProfile_advLeafCount shape,
    counterfactualAllAdvSameXProfile_advQuotientCount shape,
    by simpa [activeXColumns] using shape.families.x.count_eq,
    counterfactualAllAdvSameXProfile_serialized_length shape⟩

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.plainFixedProfile_projectionLaneCount' depends on axioms: [propext] -/
#guard_msgs in
#print axioms plainFixedProfile_projectionLaneCount

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.plainFixedProfile_serialized_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms plainFixedProfile_serialized_length

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.counterfactualAllAdvSameXProfile_serialized_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms counterfactualAllAdvSameXProfile_serialized_length

end tests.PiRlcProjectionBindingShape
