import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf6

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf9 :
    (rowsChunk wire 9).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 9).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf10 :
    (rowsChunk wire 10).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 10).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf11 :
    (rowsChunk wire 11).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 11).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf12 :
    (rowsChunk wire 12).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 12).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf13 :
    (rowsChunk wire 13).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 13).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf14 :
    (rowsChunk wire 14).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 14).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf15 :
    (rowsChunk wire 15).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 15).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf16 :
    (rowsChunk wire 16).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 16).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 9 ≤ k → k < 17 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is9 : k = 9
  · subst is9
    exact candLeaf9
  by_cases is10 : k = 10
  · subst is10
    exact candLeaf10
  by_cases is11 : k = 11
  · subst is11
    exact candLeaf11
  by_cases is12 : k = 12
  · subst is12
    exact candLeaf12
  by_cases is13 : k = 13
  · subst is13
    exact candLeaf13
  by_cases is14 : k = 14
  · subst is14
    exact candLeaf14
  by_cases is15 : k = 15
  · subst is15
    exact candLeaf15
  by_cases is16 : k = 16
  · subst is16
    exact candLeaf16
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 9 ≤ k → k < 17 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is9 : k = 9
  · subst is9
    intro scalar member
    rw [show certParts 9 = [] from rfl] at member
    cases member
  by_cases is10 : k = 10
  · subst is10
    intro scalar member
    rw [show certParts 10 = [] from rfl] at member
    cases member
  by_cases is11 : k = 11
  · subst is11
    intro scalar member
    rw [show certParts 11 = [] from rfl] at member
    cases member
  by_cases is12 : k = 12
  · subst is12
    intro scalar member
    rw [show certParts 12 = [] from rfl] at member
    cases member
  by_cases is13 : k = 13
  · subst is13
    intro scalar member
    rw [show certParts 13 = [] from rfl] at member
    cases member
  by_cases is14 : k = 14
  · subst is14
    intro scalar member
    rw [show certParts 14 = [] from rfl] at member
    cases member
  by_cases is15 : k = 15
  · subst is15
    intro scalar member
    rw [show certParts 15 = [] from rfl] at member
    cases member
  by_cases is16 : k = 16
  · subst is16
    intro scalar member
    rw [show certParts 16 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf6
