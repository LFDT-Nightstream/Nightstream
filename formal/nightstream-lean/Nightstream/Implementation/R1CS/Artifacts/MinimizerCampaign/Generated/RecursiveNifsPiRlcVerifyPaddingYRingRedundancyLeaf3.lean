import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf3

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf3 :
    (rowsChunk wire 3).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 3).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf4 :
    (rowsChunk wire 4).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 4).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf5 :
    (rowsChunk wire 5).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 5).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf6 :
    (rowsChunk wire 6).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 6).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 3 ≤ k → k < 7 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is3 : k = 3
  · subst is3
    exact candLeaf3
  by_cases is4 : k = 4
  · subst is4
    exact candLeaf4
  by_cases is5 : k = 5
  · subst is5
    exact candLeaf5
  by_cases is6 : k = 6
  · subst is6
    exact candLeaf6
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 3 ≤ k → k < 7 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is3 : k = 3
  · subst is3
    intro scalar member
    rw [show certParts 3 = [] from rfl] at member
    cases member
  by_cases is4 : k = 4
  · subst is4
    intro scalar member
    rw [show certParts 4 = [] from rfl] at member
    cases member
  by_cases is5 : k = 5
  · subst is5
    intro scalar member
    rw [show certParts 5 = [] from rfl] at member
    cases member
  by_cases is6 : k = 6
  · subst is6
    intro scalar member
    rw [show certParts 6 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf3
