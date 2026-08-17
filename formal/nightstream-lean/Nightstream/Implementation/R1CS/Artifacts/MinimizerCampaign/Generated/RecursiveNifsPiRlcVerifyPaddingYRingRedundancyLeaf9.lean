import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf9

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf32 :
    (rowsChunk wire 32).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 32).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf33 :
    (rowsChunk wire 33).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 33).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf34 :
    (rowsChunk wire 34).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 34).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf35 :
    (rowsChunk wire 35).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 35).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf36 :
    (rowsChunk wire 36).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 36).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf37 :
    (rowsChunk wire 37).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 37).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf38 :
    (rowsChunk wire 38).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 38).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf39 :
    (rowsChunk wire 39).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 39).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf40 :
    (rowsChunk wire 40).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 40).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf41 :
    (rowsChunk wire 41).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 41).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf42 :
    (rowsChunk wire 42).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 42).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf43 :
    (rowsChunk wire 43).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 43).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf44 :
    (rowsChunk wire 44).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 44).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf45 :
    (rowsChunk wire 45).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 45).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 32 ≤ k → k < 46 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is32 : k = 32
  · subst is32
    exact candLeaf32
  by_cases is33 : k = 33
  · subst is33
    exact candLeaf33
  by_cases is34 : k = 34
  · subst is34
    exact candLeaf34
  by_cases is35 : k = 35
  · subst is35
    exact candLeaf35
  by_cases is36 : k = 36
  · subst is36
    exact candLeaf36
  by_cases is37 : k = 37
  · subst is37
    exact candLeaf37
  by_cases is38 : k = 38
  · subst is38
    exact candLeaf38
  by_cases is39 : k = 39
  · subst is39
    exact candLeaf39
  by_cases is40 : k = 40
  · subst is40
    exact candLeaf40
  by_cases is41 : k = 41
  · subst is41
    exact candLeaf41
  by_cases is42 : k = 42
  · subst is42
    exact candLeaf42
  by_cases is43 : k = 43
  · subst is43
    exact candLeaf43
  by_cases is44 : k = 44
  · subst is44
    exact candLeaf44
  by_cases is45 : k = 45
  · subst is45
    exact candLeaf45
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 32 ≤ k → k < 46 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is32 : k = 32
  · subst is32
    intro scalar member
    rw [show certParts 32 = [] from rfl] at member
    cases member
  by_cases is33 : k = 33
  · subst is33
    intro scalar member
    rw [show certParts 33 = [] from rfl] at member
    cases member
  by_cases is34 : k = 34
  · subst is34
    intro scalar member
    rw [show certParts 34 = [] from rfl] at member
    cases member
  by_cases is35 : k = 35
  · subst is35
    intro scalar member
    rw [show certParts 35 = [] from rfl] at member
    cases member
  by_cases is36 : k = 36
  · subst is36
    intro scalar member
    rw [show certParts 36 = [] from rfl] at member
    cases member
  by_cases is37 : k = 37
  · subst is37
    intro scalar member
    rw [show certParts 37 = [] from rfl] at member
    cases member
  by_cases is38 : k = 38
  · subst is38
    intro scalar member
    rw [show certParts 38 = [] from rfl] at member
    cases member
  by_cases is39 : k = 39
  · subst is39
    intro scalar member
    rw [show certParts 39 = [] from rfl] at member
    cases member
  by_cases is40 : k = 40
  · subst is40
    intro scalar member
    rw [show certParts 40 = [] from rfl] at member
    cases member
  by_cases is41 : k = 41
  · subst is41
    intro scalar member
    rw [show certParts 41 = [] from rfl] at member
    cases member
  by_cases is42 : k = 42
  · subst is42
    intro scalar member
    rw [show certParts 42 = [] from rfl] at member
    cases member
  by_cases is43 : k = 43
  · subst is43
    intro scalar member
    rw [show certParts 43 = [] from rfl] at member
    cases member
  by_cases is44 : k = 44
  · subst is44
    intro scalar member
    rw [show certParts 44 = [] from rfl] at member
    cases member
  by_cases is45 : k = 45
  · subst is45
    intro scalar member
    rw [show certParts 45 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf9
