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

theorem chunkLeaf131 :
    ((rowsChunk wire 131).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 131).map (fun scalar => scalar.candidate)) ∧
      ((certParts 131).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf132 :
    ((rowsChunk wire 132).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 132).map (fun scalar => scalar.candidate)) ∧
      ((certParts 132).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf133 :
    ((rowsChunk wire 133).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 133).map (fun scalar => scalar.candidate)) ∧
      ((certParts 133).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf134 :
    ((rowsChunk wire 134).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 134).map (fun scalar => scalar.candidate)) ∧
      ((certParts 134).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf135 :
    ((rowsChunk wire 135).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 135).map (fun scalar => scalar.candidate)) ∧
      ((certParts 135).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf136 :
    ((rowsChunk wire 136).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 136).map (fun scalar => scalar.candidate)) ∧
      ((certParts 136).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf137 :
    ((rowsChunk wire 137).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 137).map (fun scalar => scalar.candidate)) ∧
      ((certParts 137).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf138 :
    ((rowsChunk wire 138).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 138).map (fun scalar => scalar.candidate)) ∧
      ((certParts 138).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf139 :
    ((rowsChunk wire 139).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 139).map (fun scalar => scalar.candidate)) ∧
      ((certParts 139).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf140 :
    ((rowsChunk wire 140).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 140).map (fun scalar => scalar.candidate)) ∧
      ((certParts 140).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf141 :
    ((rowsChunk wire 141).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 141).map (fun scalar => scalar.candidate)) ∧
      ((certParts 141).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf142 :
    ((rowsChunk wire 142).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 142).map (fun scalar => scalar.candidate)) ∧
      ((certParts 142).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf143 :
    ((rowsChunk wire 143).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 143).map (fun scalar => scalar.candidate)) ∧
      ((certParts 143).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf144 :
    ((rowsChunk wire 144).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 144).map (fun scalar => scalar.candidate)) ∧
      ((certParts 144).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 131 ≤ k → k < 145 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is131 : k = 131
  · subst is131
    exact (chunkLeaf131).1
  by_cases is132 : k = 132
  · subst is132
    exact (chunkLeaf132).1
  by_cases is133 : k = 133
  · subst is133
    exact (chunkLeaf133).1
  by_cases is134 : k = 134
  · subst is134
    exact (chunkLeaf134).1
  by_cases is135 : k = 135
  · subst is135
    exact (chunkLeaf135).1
  by_cases is136 : k = 136
  · subst is136
    exact (chunkLeaf136).1
  by_cases is137 : k = 137
  · subst is137
    exact (chunkLeaf137).1
  by_cases is138 : k = 138
  · subst is138
    exact (chunkLeaf138).1
  by_cases is139 : k = 139
  · subst is139
    exact (chunkLeaf139).1
  by_cases is140 : k = 140
  · subst is140
    exact (chunkLeaf140).1
  by_cases is141 : k = 141
  · subst is141
    exact (chunkLeaf141).1
  by_cases is142 : k = 142
  · subst is142
    exact (chunkLeaf142).1
  by_cases is143 : k = 143
  · subst is143
    exact (chunkLeaf143).1
  by_cases is144 : k = 144
  · subst is144
    exact (chunkLeaf144).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 131 ≤ k → k < 145 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is131 : k = 131
  · subst is131
    exact (chunkLeaf131).2
  by_cases is132 : k = 132
  · subst is132
    exact (chunkLeaf132).2
  by_cases is133 : k = 133
  · subst is133
    exact (chunkLeaf133).2
  by_cases is134 : k = 134
  · subst is134
    exact (chunkLeaf134).2
  by_cases is135 : k = 135
  · subst is135
    exact (chunkLeaf135).2
  by_cases is136 : k = 136
  · subst is136
    exact (chunkLeaf136).2
  by_cases is137 : k = 137
  · subst is137
    exact (chunkLeaf137).2
  by_cases is138 : k = 138
  · subst is138
    exact (chunkLeaf138).2
  by_cases is139 : k = 139
  · subst is139
    exact (chunkLeaf139).2
  by_cases is140 : k = 140
  · subst is140
    exact (chunkLeaf140).2
  by_cases is141 : k = 141
  · subst is141
    exact (chunkLeaf141).2
  by_cases is142 : k = 142
  · subst is142
    exact (chunkLeaf142).2
  by_cases is143 : k = 143
  · subst is143
    exact (chunkLeaf143).2
  by_cases is144 : k = 144
  · subst is144
    exact (chunkLeaf144).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf27
