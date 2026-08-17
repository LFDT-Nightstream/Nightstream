import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf12

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf58 :
    (rowsChunk wire 58).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 58).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf59 :
    (rowsChunk wire 59).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 59).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf60 :
    (rowsChunk wire 60).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 60).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf61 :
    (rowsChunk wire 61).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 61).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf62 :
    (rowsChunk wire 62).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 62).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf63 :
    (rowsChunk wire 63).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 63).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf64 :
    (rowsChunk wire 64).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 64).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf65 :
    (rowsChunk wire 65).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 65).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf66 :
    (rowsChunk wire 66).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 66).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf67 :
    (rowsChunk wire 67).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 67).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf68 :
    (rowsChunk wire 68).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 68).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf69 :
    (rowsChunk wire 69).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 69).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf70 :
    (rowsChunk wire 70).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 70).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf71 :
    (rowsChunk wire 71).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 71).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 58 ≤ k → k < 72 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is58 : k = 58
  · subst is58
    exact candLeaf58
  by_cases is59 : k = 59
  · subst is59
    exact candLeaf59
  by_cases is60 : k = 60
  · subst is60
    exact candLeaf60
  by_cases is61 : k = 61
  · subst is61
    exact candLeaf61
  by_cases is62 : k = 62
  · subst is62
    exact candLeaf62
  by_cases is63 : k = 63
  · subst is63
    exact candLeaf63
  by_cases is64 : k = 64
  · subst is64
    exact candLeaf64
  by_cases is65 : k = 65
  · subst is65
    exact candLeaf65
  by_cases is66 : k = 66
  · subst is66
    exact candLeaf66
  by_cases is67 : k = 67
  · subst is67
    exact candLeaf67
  by_cases is68 : k = 68
  · subst is68
    exact candLeaf68
  by_cases is69 : k = 69
  · subst is69
    exact candLeaf69
  by_cases is70 : k = 70
  · subst is70
    exact candLeaf70
  by_cases is71 : k = 71
  · subst is71
    exact candLeaf71
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 58 ≤ k → k < 72 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is58 : k = 58
  · subst is58
    intro scalar member
    rw [show certParts 58 = [] from rfl] at member
    cases member
  by_cases is59 : k = 59
  · subst is59
    intro scalar member
    rw [show certParts 59 = [] from rfl] at member
    cases member
  by_cases is60 : k = 60
  · subst is60
    intro scalar member
    rw [show certParts 60 = [] from rfl] at member
    cases member
  by_cases is61 : k = 61
  · subst is61
    intro scalar member
    rw [show certParts 61 = [] from rfl] at member
    cases member
  by_cases is62 : k = 62
  · subst is62
    intro scalar member
    rw [show certParts 62 = [] from rfl] at member
    cases member
  by_cases is63 : k = 63
  · subst is63
    intro scalar member
    rw [show certParts 63 = [] from rfl] at member
    cases member
  by_cases is64 : k = 64
  · subst is64
    intro scalar member
    rw [show certParts 64 = [] from rfl] at member
    cases member
  by_cases is65 : k = 65
  · subst is65
    intro scalar member
    rw [show certParts 65 = [] from rfl] at member
    cases member
  by_cases is66 : k = 66
  · subst is66
    intro scalar member
    rw [show certParts 66 = [] from rfl] at member
    cases member
  by_cases is67 : k = 67
  · subst is67
    intro scalar member
    rw [show certParts 67 = [] from rfl] at member
    cases member
  by_cases is68 : k = 68
  · subst is68
    intro scalar member
    rw [show certParts 68 = [] from rfl] at member
    cases member
  by_cases is69 : k = 69
  · subst is69
    intro scalar member
    rw [show certParts 69 = [] from rfl] at member
    cases member
  by_cases is70 : k = 70
  · subst is70
    intro scalar member
    rw [show certParts 70 = [] from rfl] at member
    cases member
  by_cases is71 : k = 71
  · subst is71
    intro scalar member
    rw [show certParts 71 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf12
