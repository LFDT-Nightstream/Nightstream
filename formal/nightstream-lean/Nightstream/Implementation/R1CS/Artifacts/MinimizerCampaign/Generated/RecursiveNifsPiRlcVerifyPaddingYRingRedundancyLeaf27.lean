import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf27

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf131 :
    (rowsChunk wire 131).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 131).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf132 :
    (rowsChunk wire 132).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 132).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf133 :
    (rowsChunk wire 133).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 133).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf134 :
    (rowsChunk wire 134).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 134).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf135 :
    (rowsChunk wire 135).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 135).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf136 :
    (rowsChunk wire 136).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 136).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf137 :
    (rowsChunk wire 137).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 137).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf138 :
    (rowsChunk wire 138).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 138).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf139 :
    (rowsChunk wire 139).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 139).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf140 :
    (rowsChunk wire 140).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 140).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf141 :
    (rowsChunk wire 141).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 141).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf142 :
    (rowsChunk wire 142).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 142).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf143 :
    (rowsChunk wire 143).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 143).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf144 :
    (rowsChunk wire 144).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 144).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 131 ≤ k → k < 145 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is131 : k = 131
  · subst is131
    exact candLeaf131
  by_cases is132 : k = 132
  · subst is132
    exact candLeaf132
  by_cases is133 : k = 133
  · subst is133
    exact candLeaf133
  by_cases is134 : k = 134
  · subst is134
    exact candLeaf134
  by_cases is135 : k = 135
  · subst is135
    exact candLeaf135
  by_cases is136 : k = 136
  · subst is136
    exact candLeaf136
  by_cases is137 : k = 137
  · subst is137
    exact candLeaf137
  by_cases is138 : k = 138
  · subst is138
    exact candLeaf138
  by_cases is139 : k = 139
  · subst is139
    exact candLeaf139
  by_cases is140 : k = 140
  · subst is140
    exact candLeaf140
  by_cases is141 : k = 141
  · subst is141
    exact candLeaf141
  by_cases is142 : k = 142
  · subst is142
    exact candLeaf142
  by_cases is143 : k = 143
  · subst is143
    exact candLeaf143
  by_cases is144 : k = 144
  · subst is144
    exact candLeaf144
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 131 ≤ k → k < 145 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is131 : k = 131
  · subst is131
    intro scalar member
    rw [show certParts 131 = [] from rfl] at member
    cases member
  by_cases is132 : k = 132
  · subst is132
    intro scalar member
    rw [show certParts 132 = [] from rfl] at member
    cases member
  by_cases is133 : k = 133
  · subst is133
    intro scalar member
    rw [show certParts 133 = [] from rfl] at member
    cases member
  by_cases is134 : k = 134
  · subst is134
    intro scalar member
    rw [show certParts 134 = [] from rfl] at member
    cases member
  by_cases is135 : k = 135
  · subst is135
    intro scalar member
    rw [show certParts 135 = [] from rfl] at member
    cases member
  by_cases is136 : k = 136
  · subst is136
    intro scalar member
    rw [show certParts 136 = [] from rfl] at member
    cases member
  by_cases is137 : k = 137
  · subst is137
    intro scalar member
    rw [show certParts 137 = [] from rfl] at member
    cases member
  by_cases is138 : k = 138
  · subst is138
    intro scalar member
    rw [show certParts 138 = [] from rfl] at member
    cases member
  by_cases is139 : k = 139
  · subst is139
    intro scalar member
    rw [show certParts 139 = [] from rfl] at member
    cases member
  by_cases is140 : k = 140
  · subst is140
    intro scalar member
    rw [show certParts 140 = [] from rfl] at member
    cases member
  by_cases is141 : k = 141
  · subst is141
    intro scalar member
    rw [show certParts 141 = [] from rfl] at member
    cases member
  by_cases is142 : k = 142
  · subst is142
    intro scalar member
    rw [show certParts 142 = [] from rfl] at member
    cases member
  by_cases is143 : k = 143
  · subst is143
    intro scalar member
    rw [show certParts 143 = [] from rfl] at member
    cases member
  by_cases is144 : k = 144
  · subst is144
    intro scalar member
    rw [show certParts 144 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf27
