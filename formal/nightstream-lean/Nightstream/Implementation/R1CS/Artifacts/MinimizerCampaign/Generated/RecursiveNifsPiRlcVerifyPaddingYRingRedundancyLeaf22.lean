import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf22

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf123 :
    (rowsChunk wire 123).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 123).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf124 :
    (rowsChunk wire 124).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 124).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 123 ≤ k → k < 125 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is123 : k = 123
  · subst is123
    exact candLeaf123
  by_cases is124 : k = 124
  · subst is124
    exact candLeaf124
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 123 ≤ k → k < 125 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is123 : k = 123
  · subst is123
    intro scalar member
    rw [show certParts 123 = [] from rfl] at member
    cases member
  by_cases is124 : k = 124
  · subst is124
    intro scalar member
    rw [show certParts 124 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf22
