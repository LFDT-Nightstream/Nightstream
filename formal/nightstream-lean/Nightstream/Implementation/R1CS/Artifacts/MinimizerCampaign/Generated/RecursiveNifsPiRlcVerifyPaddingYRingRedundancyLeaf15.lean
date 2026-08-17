import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf15

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf100 :
    (rowsChunk wire 100).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 100).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf101 :
    (rowsChunk wire 101).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 101).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 100 ≤ k → k < 102 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is100 : k = 100
  · subst is100
    exact candLeaf100
  by_cases is101 : k = 101
  · subst is101
    exact candLeaf101
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 100 ≤ k → k < 102 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is100 : k = 100
  · subst is100
    intro scalar member
    rw [show certParts 100 = [] from rfl] at member
    cases member
  by_cases is101 : k = 101
  · subst is101
    intro scalar member
    rw [show certParts 101 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf15
