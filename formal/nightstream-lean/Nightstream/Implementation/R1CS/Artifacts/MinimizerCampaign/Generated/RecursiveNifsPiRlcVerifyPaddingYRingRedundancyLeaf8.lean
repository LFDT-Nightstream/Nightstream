import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf8

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem candLeaf18 :
    (rowsChunk wire 18).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 18).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf19 :
    (rowsChunk wire 19).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 19).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf20 :
    (rowsChunk wire 20).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 20).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf21 :
    (rowsChunk wire 21).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 21).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf22 :
    (rowsChunk wire 22).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 22).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf23 :
    (rowsChunk wire 23).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 23).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf24 :
    (rowsChunk wire 24).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 24).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf25 :
    (rowsChunk wire 25).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 25).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf26 :
    (rowsChunk wire 26).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 26).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf27 :
    (rowsChunk wire 27).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 27).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf28 :
    (rowsChunk wire 28).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 28).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf29 :
    (rowsChunk wire 29).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 29).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf30 :
    (rowsChunk wire 30).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 30).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf31 :
    (rowsChunk wire 31).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 31).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candGroup :
    ∀ k, 18 ≤ k → k < 32 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is18 : k = 18
  · subst is18
    exact candLeaf18
  by_cases is19 : k = 19
  · subst is19
    exact candLeaf19
  by_cases is20 : k = 20
  · subst is20
    exact candLeaf20
  by_cases is21 : k = 21
  · subst is21
    exact candLeaf21
  by_cases is22 : k = 22
  · subst is22
    exact candLeaf22
  by_cases is23 : k = 23
  · subst is23
    exact candLeaf23
  by_cases is24 : k = 24
  · subst is24
    exact candLeaf24
  by_cases is25 : k = 25
  · subst is25
    exact candLeaf25
  by_cases is26 : k = 26
  · subst is26
    exact candLeaf26
  by_cases is27 : k = 27
  · subst is27
    exact candLeaf27
  by_cases is28 : k = 28
  · subst is28
    exact candLeaf28
  by_cases is29 : k = 29
  · subst is29
    exact candLeaf29
  by_cases is30 : k = 30
  · subst is30
    exact candLeaf30
  by_cases is31 : k = 31
  · subst is31
    exact candLeaf31
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 18 ≤ k → k < 32 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is18 : k = 18
  · subst is18
    intro scalar member
    rw [show certParts 18 = [] from rfl] at member
    cases member
  by_cases is19 : k = 19
  · subst is19
    intro scalar member
    rw [show certParts 19 = [] from rfl] at member
    cases member
  by_cases is20 : k = 20
  · subst is20
    intro scalar member
    rw [show certParts 20 = [] from rfl] at member
    cases member
  by_cases is21 : k = 21
  · subst is21
    intro scalar member
    rw [show certParts 21 = [] from rfl] at member
    cases member
  by_cases is22 : k = 22
  · subst is22
    intro scalar member
    rw [show certParts 22 = [] from rfl] at member
    cases member
  by_cases is23 : k = 23
  · subst is23
    intro scalar member
    rw [show certParts 23 = [] from rfl] at member
    cases member
  by_cases is24 : k = 24
  · subst is24
    intro scalar member
    rw [show certParts 24 = [] from rfl] at member
    cases member
  by_cases is25 : k = 25
  · subst is25
    intro scalar member
    rw [show certParts 25 = [] from rfl] at member
    cases member
  by_cases is26 : k = 26
  · subst is26
    intro scalar member
    rw [show certParts 26 = [] from rfl] at member
    cases member
  by_cases is27 : k = 27
  · subst is27
    intro scalar member
    rw [show certParts 27 = [] from rfl] at member
    cases member
  by_cases is28 : k = 28
  · subst is28
    intro scalar member
    rw [show certParts 28 = [] from rfl] at member
    cases member
  by_cases is29 : k = 29
  · subst is29
    intro scalar member
    rw [show certParts 29 = [] from rfl] at member
    cases member
  by_cases is30 : k = 30
  · subst is30
    intro scalar member
    rw [show certParts 30 = [] from rfl] at member
    cases member
  by_cases is31 : k = 31
  · subst is31
    intro scalar member
    rw [show certParts 31 = [] from rfl] at member
    cases member
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf8
