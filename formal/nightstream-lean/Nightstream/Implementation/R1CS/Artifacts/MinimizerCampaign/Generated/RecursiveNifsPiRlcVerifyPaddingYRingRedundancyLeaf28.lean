import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf28

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf145 :
    (rowsChunk wire 145).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 145).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf146 :
    (rowsChunk wire 146).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 146).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf147 :
    (rowsChunk wire 147).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 147).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf148 :
    (rowsChunk wire 148).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 148).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf149 :
    (rowsChunk wire 149).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 149).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf150 :
    (rowsChunk wire 150).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 150).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf151 :
    (rowsChunk wire 151).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 151).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf152 :
    (rowsChunk wire 152).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 152).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf153 :
    (rowsChunk wire 153).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 153).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf154 :
    (rowsChunk wire 154).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 154).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf155 :
    (rowsChunk wire 155).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 155).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf156 :
    (rowsChunk wire 156).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 156).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf157 :
    (rowsChunk wire 157).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 157).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf158 :
    (rowsChunk wire 158).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 158).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 145 ≤ k → k < 159 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is145 : k = 145
  · subst is145
    exact candLeaf145
  by_cases is146 : k = 146
  · subst is146
    exact candLeaf146
  by_cases is147 : k = 147
  · subst is147
    exact candLeaf147
  by_cases is148 : k = 148
  · subst is148
    exact candLeaf148
  by_cases is149 : k = 149
  · subst is149
    exact candLeaf149
  by_cases is150 : k = 150
  · subst is150
    exact candLeaf150
  by_cases is151 : k = 151
  · subst is151
    exact candLeaf151
  by_cases is152 : k = 152
  · subst is152
    exact candLeaf152
  by_cases is153 : k = 153
  · subst is153
    exact candLeaf153
  by_cases is154 : k = 154
  · subst is154
    exact candLeaf154
  by_cases is155 : k = 155
  · subst is155
    exact candLeaf155
  by_cases is156 : k = 156
  · subst is156
    exact candLeaf156
  by_cases is157 : k = 157
  · subst is157
    exact candLeaf157
  by_cases is158 : k = 158
  · subst is158
    exact candLeaf158
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 145 ≤ k → k < 159 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is145 : k = 145
  · subst is145
    intro scalar member
    rw [show certParts 145 = [] from rfl] at member
    cases member
  by_cases is146 : k = 146
  · subst is146
    intro scalar member
    rw [show certParts 146 = [] from rfl] at member
    cases member
  by_cases is147 : k = 147
  · subst is147
    intro scalar member
    rw [show certParts 147 = [] from rfl] at member
    cases member
  by_cases is148 : k = 148
  · subst is148
    intro scalar member
    rw [show certParts 148 = [] from rfl] at member
    cases member
  by_cases is149 : k = 149
  · subst is149
    intro scalar member
    rw [show certParts 149 = [] from rfl] at member
    cases member
  by_cases is150 : k = 150
  · subst is150
    intro scalar member
    rw [show certParts 150 = [] from rfl] at member
    cases member
  by_cases is151 : k = 151
  · subst is151
    intro scalar member
    rw [show certParts 151 = [] from rfl] at member
    cases member
  by_cases is152 : k = 152
  · subst is152
    intro scalar member
    rw [show certParts 152 = [] from rfl] at member
    cases member
  by_cases is153 : k = 153
  · subst is153
    intro scalar member
    rw [show certParts 153 = [] from rfl] at member
    cases member
  by_cases is154 : k = 154
  · subst is154
    intro scalar member
    rw [show certParts 154 = [] from rfl] at member
    cases member
  by_cases is155 : k = 155
  · subst is155
    intro scalar member
    rw [show certParts 155 = [] from rfl] at member
    cases member
  by_cases is156 : k = 156
  · subst is156
    intro scalar member
    rw [show certParts 156 = [] from rfl] at member
    cases member
  by_cases is157 : k = 157
  · subst is157
    intro scalar member
    rw [show certParts 157 = [] from rfl] at member
    cases member
  by_cases is158 : k = 158
  · subst is158
    intro scalar member
    rw [show certParts 158 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf28
