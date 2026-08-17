import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf10

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf46 :
    (rowsChunk wire 46).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 46).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf47 :
    (rowsChunk wire 47).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 47).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf48 :
    (rowsChunk wire 48).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 48).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf49 :
    (rowsChunk wire 49).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 49).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf50 :
    (rowsChunk wire 50).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 50).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf51 :
    (rowsChunk wire 51).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 51).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf52 :
    (rowsChunk wire 52).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 52).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf53 :
    (rowsChunk wire 53).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 53).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf54 :
    (rowsChunk wire 54).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 54).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf55 :
    (rowsChunk wire 55).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 55).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf56 :
    (rowsChunk wire 56).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 56).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 46 ≤ k → k < 57 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is46 : k = 46
  · subst is46
    exact candLeaf46
  by_cases is47 : k = 47
  · subst is47
    exact candLeaf47
  by_cases is48 : k = 48
  · subst is48
    exact candLeaf48
  by_cases is49 : k = 49
  · subst is49
    exact candLeaf49
  by_cases is50 : k = 50
  · subst is50
    exact candLeaf50
  by_cases is51 : k = 51
  · subst is51
    exact candLeaf51
  by_cases is52 : k = 52
  · subst is52
    exact candLeaf52
  by_cases is53 : k = 53
  · subst is53
    exact candLeaf53
  by_cases is54 : k = 54
  · subst is54
    exact candLeaf54
  by_cases is55 : k = 55
  · subst is55
    exact candLeaf55
  by_cases is56 : k = 56
  · subst is56
    exact candLeaf56
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 46 ≤ k → k < 57 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is46 : k = 46
  · subst is46
    intro scalar member
    rw [show certParts 46 = [] from rfl] at member
    cases member
  by_cases is47 : k = 47
  · subst is47
    intro scalar member
    rw [show certParts 47 = [] from rfl] at member
    cases member
  by_cases is48 : k = 48
  · subst is48
    intro scalar member
    rw [show certParts 48 = [] from rfl] at member
    cases member
  by_cases is49 : k = 49
  · subst is49
    intro scalar member
    rw [show certParts 49 = [] from rfl] at member
    cases member
  by_cases is50 : k = 50
  · subst is50
    intro scalar member
    rw [show certParts 50 = [] from rfl] at member
    cases member
  by_cases is51 : k = 51
  · subst is51
    intro scalar member
    rw [show certParts 51 = [] from rfl] at member
    cases member
  by_cases is52 : k = 52
  · subst is52
    intro scalar member
    rw [show certParts 52 = [] from rfl] at member
    cases member
  by_cases is53 : k = 53
  · subst is53
    intro scalar member
    rw [show certParts 53 = [] from rfl] at member
    cases member
  by_cases is54 : k = 54
  · subst is54
    intro scalar member
    rw [show certParts 54 = [] from rfl] at member
    cases member
  by_cases is55 : k = 55
  · subst is55
    intro scalar member
    rw [show certParts 55 = [] from rfl] at member
    cases member
  by_cases is56 : k = 56
  · subst is56
    intro scalar member
    rw [show certParts 56 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf10
