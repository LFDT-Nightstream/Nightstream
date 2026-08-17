import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf29

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf159 :
    (rowsChunk wire 159).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 159).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf160 :
    (rowsChunk wire 160).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 160).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf161 :
    (rowsChunk wire 161).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 161).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf162 :
    (rowsChunk wire 162).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 162).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf163 :
    (rowsChunk wire 163).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 163).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf164 :
    (rowsChunk wire 164).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 164).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf165 :
    (rowsChunk wire 165).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 165).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf166 :
    (rowsChunk wire 166).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 166).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf167 :
    (rowsChunk wire 167).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 167).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf168 :
    (rowsChunk wire 168).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 168).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 159 ≤ k → k < 169 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is159 : k = 159
  · subst is159
    exact candLeaf159
  by_cases is160 : k = 160
  · subst is160
    exact candLeaf160
  by_cases is161 : k = 161
  · subst is161
    exact candLeaf161
  by_cases is162 : k = 162
  · subst is162
    exact candLeaf162
  by_cases is163 : k = 163
  · subst is163
    exact candLeaf163
  by_cases is164 : k = 164
  · subst is164
    exact candLeaf164
  by_cases is165 : k = 165
  · subst is165
    exact candLeaf165
  by_cases is166 : k = 166
  · subst is166
    exact candLeaf166
  by_cases is167 : k = 167
  · subst is167
    exact candLeaf167
  by_cases is168 : k = 168
  · subst is168
    exact candLeaf168
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 159 ≤ k → k < 169 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is159 : k = 159
  · subst is159
    intro scalar member
    rw [show certParts 159 = [] from rfl] at member
    cases member
  by_cases is160 : k = 160
  · subst is160
    intro scalar member
    rw [show certParts 160 = [] from rfl] at member
    cases member
  by_cases is161 : k = 161
  · subst is161
    intro scalar member
    rw [show certParts 161 = [] from rfl] at member
    cases member
  by_cases is162 : k = 162
  · subst is162
    intro scalar member
    rw [show certParts 162 = [] from rfl] at member
    cases member
  by_cases is163 : k = 163
  · subst is163
    intro scalar member
    rw [show certParts 163 = [] from rfl] at member
    cases member
  by_cases is164 : k = 164
  · subst is164
    intro scalar member
    rw [show certParts 164 = [] from rfl] at member
    cases member
  by_cases is165 : k = 165
  · subst is165
    intro scalar member
    rw [show certParts 165 = [] from rfl] at member
    cases member
  by_cases is166 : k = 166
  · subst is166
    intro scalar member
    rw [show certParts 166 = [] from rfl] at member
    cases member
  by_cases is167 : k = 167
  · subst is167
    intro scalar member
    rw [show certParts 167 = [] from rfl] at member
    cases member
  by_cases is168 : k = 168
  · subst is168
    intro scalar member
    rw [show certParts 168 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf29
