import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf17

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf103 :
    (rowsChunk wire 103).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 103).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf104 :
    (rowsChunk wire 104).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 104).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf105 :
    (rowsChunk wire 105).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 105).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 103 ≤ k → k < 106 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is103 : k = 103
  · subst is103
    exact candLeaf103
  by_cases is104 : k = 104
  · subst is104
    exact candLeaf104
  by_cases is105 : k = 105
  · subst is105
    exact candLeaf105
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 103 ≤ k → k < 106 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is103 : k = 103
  · subst is103
    intro scalar member
    rw [show certParts 103 = [] from rfl] at member
    cases member
  by_cases is104 : k = 104
  · subst is104
    intro scalar member
    rw [show certParts 104 = [] from rfl] at member
    cases member
  by_cases is105 : k = 105
  · subst is105
    intro scalar member
    rw [show certParts 105 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf17
