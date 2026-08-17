import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf21

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf109 :
    (rowsChunk wire 109).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 109).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf110 :
    (rowsChunk wire 110).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 110).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf111 :
    (rowsChunk wire 111).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 111).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf112 :
    (rowsChunk wire 112).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 112).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf113 :
    (rowsChunk wire 113).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 113).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf114 :
    (rowsChunk wire 114).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 114).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf115 :
    (rowsChunk wire 115).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 115).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf116 :
    (rowsChunk wire 116).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 116).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf117 :
    (rowsChunk wire 117).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 117).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf118 :
    (rowsChunk wire 118).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 118).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf119 :
    (rowsChunk wire 119).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 119).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf120 :
    (rowsChunk wire 120).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 120).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf121 :
    (rowsChunk wire 121).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 121).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf122 :
    (rowsChunk wire 122).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 122).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 109 ≤ k → k < 123 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is109 : k = 109
  · subst is109
    exact candLeaf109
  by_cases is110 : k = 110
  · subst is110
    exact candLeaf110
  by_cases is111 : k = 111
  · subst is111
    exact candLeaf111
  by_cases is112 : k = 112
  · subst is112
    exact candLeaf112
  by_cases is113 : k = 113
  · subst is113
    exact candLeaf113
  by_cases is114 : k = 114
  · subst is114
    exact candLeaf114
  by_cases is115 : k = 115
  · subst is115
    exact candLeaf115
  by_cases is116 : k = 116
  · subst is116
    exact candLeaf116
  by_cases is117 : k = 117
  · subst is117
    exact candLeaf117
  by_cases is118 : k = 118
  · subst is118
    exact candLeaf118
  by_cases is119 : k = 119
  · subst is119
    exact candLeaf119
  by_cases is120 : k = 120
  · subst is120
    exact candLeaf120
  by_cases is121 : k = 121
  · subst is121
    exact candLeaf121
  by_cases is122 : k = 122
  · subst is122
    exact candLeaf122
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 109 ≤ k → k < 123 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is109 : k = 109
  · subst is109
    intro scalar member
    rw [show certParts 109 = [] from rfl] at member
    cases member
  by_cases is110 : k = 110
  · subst is110
    intro scalar member
    rw [show certParts 110 = [] from rfl] at member
    cases member
  by_cases is111 : k = 111
  · subst is111
    intro scalar member
    rw [show certParts 111 = [] from rfl] at member
    cases member
  by_cases is112 : k = 112
  · subst is112
    intro scalar member
    rw [show certParts 112 = [] from rfl] at member
    cases member
  by_cases is113 : k = 113
  · subst is113
    intro scalar member
    rw [show certParts 113 = [] from rfl] at member
    cases member
  by_cases is114 : k = 114
  · subst is114
    intro scalar member
    rw [show certParts 114 = [] from rfl] at member
    cases member
  by_cases is115 : k = 115
  · subst is115
    intro scalar member
    rw [show certParts 115 = [] from rfl] at member
    cases member
  by_cases is116 : k = 116
  · subst is116
    intro scalar member
    rw [show certParts 116 = [] from rfl] at member
    cases member
  by_cases is117 : k = 117
  · subst is117
    intro scalar member
    rw [show certParts 117 = [] from rfl] at member
    cases member
  by_cases is118 : k = 118
  · subst is118
    intro scalar member
    rw [show certParts 118 = [] from rfl] at member
    cases member
  by_cases is119 : k = 119
  · subst is119
    intro scalar member
    rw [show certParts 119 = [] from rfl] at member
    cases member
  by_cases is120 : k = 120
  · subst is120
    intro scalar member
    rw [show certParts 120 = [] from rfl] at member
    cases member
  by_cases is121 : k = 121
  · subst is121
    intro scalar member
    rw [show certParts 121 = [] from rfl] at member
    cases member
  by_cases is122 : k = 122
  · subst is122
    intro scalar member
    rw [show certParts 122 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf21
