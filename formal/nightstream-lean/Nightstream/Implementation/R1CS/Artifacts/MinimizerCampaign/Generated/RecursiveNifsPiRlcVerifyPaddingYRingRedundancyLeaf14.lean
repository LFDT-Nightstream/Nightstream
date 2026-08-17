import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf14

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf86 :
    (rowsChunk wire 86).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 86).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf87 :
    (rowsChunk wire 87).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 87).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf88 :
    (rowsChunk wire 88).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 88).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf89 :
    (rowsChunk wire 89).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 89).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf90 :
    (rowsChunk wire 90).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 90).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf91 :
    (rowsChunk wire 91).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 91).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf92 :
    (rowsChunk wire 92).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 92).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf93 :
    (rowsChunk wire 93).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 93).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf94 :
    (rowsChunk wire 94).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 94).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf95 :
    (rowsChunk wire 95).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 95).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf96 :
    (rowsChunk wire 96).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 96).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf97 :
    (rowsChunk wire 97).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 97).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf98 :
    (rowsChunk wire 98).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 98).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf99 :
    (rowsChunk wire 99).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 99).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 86 ≤ k → k < 100 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is86 : k = 86
  · subst is86
    exact candLeaf86
  by_cases is87 : k = 87
  · subst is87
    exact candLeaf87
  by_cases is88 : k = 88
  · subst is88
    exact candLeaf88
  by_cases is89 : k = 89
  · subst is89
    exact candLeaf89
  by_cases is90 : k = 90
  · subst is90
    exact candLeaf90
  by_cases is91 : k = 91
  · subst is91
    exact candLeaf91
  by_cases is92 : k = 92
  · subst is92
    exact candLeaf92
  by_cases is93 : k = 93
  · subst is93
    exact candLeaf93
  by_cases is94 : k = 94
  · subst is94
    exact candLeaf94
  by_cases is95 : k = 95
  · subst is95
    exact candLeaf95
  by_cases is96 : k = 96
  · subst is96
    exact candLeaf96
  by_cases is97 : k = 97
  · subst is97
    exact candLeaf97
  by_cases is98 : k = 98
  · subst is98
    exact candLeaf98
  by_cases is99 : k = 99
  · subst is99
    exact candLeaf99
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 86 ≤ k → k < 100 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is86 : k = 86
  · subst is86
    intro scalar member
    rw [show certParts 86 = [] from rfl] at member
    cases member
  by_cases is87 : k = 87
  · subst is87
    intro scalar member
    rw [show certParts 87 = [] from rfl] at member
    cases member
  by_cases is88 : k = 88
  · subst is88
    intro scalar member
    rw [show certParts 88 = [] from rfl] at member
    cases member
  by_cases is89 : k = 89
  · subst is89
    intro scalar member
    rw [show certParts 89 = [] from rfl] at member
    cases member
  by_cases is90 : k = 90
  · subst is90
    intro scalar member
    rw [show certParts 90 = [] from rfl] at member
    cases member
  by_cases is91 : k = 91
  · subst is91
    intro scalar member
    rw [show certParts 91 = [] from rfl] at member
    cases member
  by_cases is92 : k = 92
  · subst is92
    intro scalar member
    rw [show certParts 92 = [] from rfl] at member
    cases member
  by_cases is93 : k = 93
  · subst is93
    intro scalar member
    rw [show certParts 93 = [] from rfl] at member
    cases member
  by_cases is94 : k = 94
  · subst is94
    intro scalar member
    rw [show certParts 94 = [] from rfl] at member
    cases member
  by_cases is95 : k = 95
  · subst is95
    intro scalar member
    rw [show certParts 95 = [] from rfl] at member
    cases member
  by_cases is96 : k = 96
  · subst is96
    intro scalar member
    rw [show certParts 96 = [] from rfl] at member
    cases member
  by_cases is97 : k = 97
  · subst is97
    intro scalar member
    rw [show certParts 97 = [] from rfl] at member
    cases member
  by_cases is98 : k = 98
  · subst is98
    intro scalar member
    rw [show certParts 98 = [] from rfl] at member
    cases member
  by_cases is99 : k = 99
  · subst is99
    intro scalar member
    rw [show certParts 99 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf14
