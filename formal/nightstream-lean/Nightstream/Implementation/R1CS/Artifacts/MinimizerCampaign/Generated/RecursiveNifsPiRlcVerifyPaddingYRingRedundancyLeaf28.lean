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

theorem chunkLeaf145 :
    ((rowsChunk wire 145).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 145).map (fun scalar => scalar.candidate)) ∧
      ((certParts 145).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf146 :
    ((rowsChunk wire 146).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 146).map (fun scalar => scalar.candidate)) ∧
      ((certParts 146).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf147 :
    ((rowsChunk wire 147).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 147).map (fun scalar => scalar.candidate)) ∧
      ((certParts 147).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf148 :
    ((rowsChunk wire 148).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 148).map (fun scalar => scalar.candidate)) ∧
      ((certParts 148).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf149 :
    ((rowsChunk wire 149).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 149).map (fun scalar => scalar.candidate)) ∧
      ((certParts 149).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf150 :
    ((rowsChunk wire 150).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 150).map (fun scalar => scalar.candidate)) ∧
      ((certParts 150).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf151 :
    ((rowsChunk wire 151).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 151).map (fun scalar => scalar.candidate)) ∧
      ((certParts 151).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf152 :
    ((rowsChunk wire 152).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 152).map (fun scalar => scalar.candidate)) ∧
      ((certParts 152).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf153 :
    ((rowsChunk wire 153).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 153).map (fun scalar => scalar.candidate)) ∧
      ((certParts 153).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf154 :
    ((rowsChunk wire 154).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 154).map (fun scalar => scalar.candidate)) ∧
      ((certParts 154).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf155 :
    ((rowsChunk wire 155).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 155).map (fun scalar => scalar.candidate)) ∧
      ((certParts 155).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf156 :
    ((rowsChunk wire 156).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 156).map (fun scalar => scalar.candidate)) ∧
      ((certParts 156).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf157 :
    ((rowsChunk wire 157).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 157).map (fun scalar => scalar.candidate)) ∧
      ((certParts 157).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf158 :
    ((rowsChunk wire 158).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 158).map (fun scalar => scalar.candidate)) ∧
      ((certParts 158).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 145 ≤ k → k < 159 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is145 : k = 145
  · subst is145
    exact (chunkLeaf145).1
  by_cases is146 : k = 146
  · subst is146
    exact (chunkLeaf146).1
  by_cases is147 : k = 147
  · subst is147
    exact (chunkLeaf147).1
  by_cases is148 : k = 148
  · subst is148
    exact (chunkLeaf148).1
  by_cases is149 : k = 149
  · subst is149
    exact (chunkLeaf149).1
  by_cases is150 : k = 150
  · subst is150
    exact (chunkLeaf150).1
  by_cases is151 : k = 151
  · subst is151
    exact (chunkLeaf151).1
  by_cases is152 : k = 152
  · subst is152
    exact (chunkLeaf152).1
  by_cases is153 : k = 153
  · subst is153
    exact (chunkLeaf153).1
  by_cases is154 : k = 154
  · subst is154
    exact (chunkLeaf154).1
  by_cases is155 : k = 155
  · subst is155
    exact (chunkLeaf155).1
  by_cases is156 : k = 156
  · subst is156
    exact (chunkLeaf156).1
  by_cases is157 : k = 157
  · subst is157
    exact (chunkLeaf157).1
  by_cases is158 : k = 158
  · subst is158
    exact (chunkLeaf158).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 145 ≤ k → k < 159 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is145 : k = 145
  · subst is145
    exact (chunkLeaf145).2
  by_cases is146 : k = 146
  · subst is146
    exact (chunkLeaf146).2
  by_cases is147 : k = 147
  · subst is147
    exact (chunkLeaf147).2
  by_cases is148 : k = 148
  · subst is148
    exact (chunkLeaf148).2
  by_cases is149 : k = 149
  · subst is149
    exact (chunkLeaf149).2
  by_cases is150 : k = 150
  · subst is150
    exact (chunkLeaf150).2
  by_cases is151 : k = 151
  · subst is151
    exact (chunkLeaf151).2
  by_cases is152 : k = 152
  · subst is152
    exact (chunkLeaf152).2
  by_cases is153 : k = 153
  · subst is153
    exact (chunkLeaf153).2
  by_cases is154 : k = 154
  · subst is154
    exact (chunkLeaf154).2
  by_cases is155 : k = 155
  · subst is155
    exact (chunkLeaf155).2
  by_cases is156 : k = 156
  · subst is156
    exact (chunkLeaf156).2
  by_cases is157 : k = 157
  · subst is157
    exact (chunkLeaf157).2
  by_cases is158 : k = 158
  · subst is158
    exact (chunkLeaf158).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf28
