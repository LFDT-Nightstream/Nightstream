import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf13

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf72 :
    (rowsChunk wire 72).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 72).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf73 :
    (rowsChunk wire 73).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 73).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf74 :
    (rowsChunk wire 74).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 74).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf75 :
    (rowsChunk wire 75).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 75).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf76 :
    (rowsChunk wire 76).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 76).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf77 :
    (rowsChunk wire 77).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 77).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf78 :
    (rowsChunk wire 78).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 78).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf79 :
    (rowsChunk wire 79).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 79).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf80 :
    (rowsChunk wire 80).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 80).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf81 :
    (rowsChunk wire 81).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 81).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf82 :
    (rowsChunk wire 82).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 82).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf83 :
    (rowsChunk wire 83).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 83).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf84 :
    (rowsChunk wire 84).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 84).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf85 :
    (rowsChunk wire 85).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 85).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 72 ≤ k → k < 86 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is72 : k = 72
  · subst is72
    exact candLeaf72
  by_cases is73 : k = 73
  · subst is73
    exact candLeaf73
  by_cases is74 : k = 74
  · subst is74
    exact candLeaf74
  by_cases is75 : k = 75
  · subst is75
    exact candLeaf75
  by_cases is76 : k = 76
  · subst is76
    exact candLeaf76
  by_cases is77 : k = 77
  · subst is77
    exact candLeaf77
  by_cases is78 : k = 78
  · subst is78
    exact candLeaf78
  by_cases is79 : k = 79
  · subst is79
    exact candLeaf79
  by_cases is80 : k = 80
  · subst is80
    exact candLeaf80
  by_cases is81 : k = 81
  · subst is81
    exact candLeaf81
  by_cases is82 : k = 82
  · subst is82
    exact candLeaf82
  by_cases is83 : k = 83
  · subst is83
    exact candLeaf83
  by_cases is84 : k = 84
  · subst is84
    exact candLeaf84
  by_cases is85 : k = 85
  · subst is85
    exact candLeaf85
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 72 ≤ k → k < 86 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is72 : k = 72
  · subst is72
    intro scalar member
    rw [show certParts 72 = [] from rfl] at member
    cases member
  by_cases is73 : k = 73
  · subst is73
    intro scalar member
    rw [show certParts 73 = [] from rfl] at member
    cases member
  by_cases is74 : k = 74
  · subst is74
    intro scalar member
    rw [show certParts 74 = [] from rfl] at member
    cases member
  by_cases is75 : k = 75
  · subst is75
    intro scalar member
    rw [show certParts 75 = [] from rfl] at member
    cases member
  by_cases is76 : k = 76
  · subst is76
    intro scalar member
    rw [show certParts 76 = [] from rfl] at member
    cases member
  by_cases is77 : k = 77
  · subst is77
    intro scalar member
    rw [show certParts 77 = [] from rfl] at member
    cases member
  by_cases is78 : k = 78
  · subst is78
    intro scalar member
    rw [show certParts 78 = [] from rfl] at member
    cases member
  by_cases is79 : k = 79
  · subst is79
    intro scalar member
    rw [show certParts 79 = [] from rfl] at member
    cases member
  by_cases is80 : k = 80
  · subst is80
    intro scalar member
    rw [show certParts 80 = [] from rfl] at member
    cases member
  by_cases is81 : k = 81
  · subst is81
    intro scalar member
    rw [show certParts 81 = [] from rfl] at member
    cases member
  by_cases is82 : k = 82
  · subst is82
    intro scalar member
    rw [show certParts 82 = [] from rfl] at member
    cases member
  by_cases is83 : k = 83
  · subst is83
    intro scalar member
    rw [show certParts 83 = [] from rfl] at member
    cases member
  by_cases is84 : k = 84
  · subst is84
    intro scalar member
    rw [show certParts 84 = [] from rfl] at member
    cases member
  by_cases is85 : k = 85
  · subst is85
    intro scalar member
    rw [show certParts 85 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf13
