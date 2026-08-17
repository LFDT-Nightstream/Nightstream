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

theorem chunkLeaf159 :
    ((rowsChunk wire 159).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 159).map (fun scalar => scalar.candidate)) ∧
      ((certParts 159).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf160 :
    ((rowsChunk wire 160).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 160).map (fun scalar => scalar.candidate)) ∧
      ((certParts 160).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf161 :
    ((rowsChunk wire 161).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 161).map (fun scalar => scalar.candidate)) ∧
      ((certParts 161).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf162 :
    ((rowsChunk wire 162).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 162).map (fun scalar => scalar.candidate)) ∧
      ((certParts 162).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf163 :
    ((rowsChunk wire 163).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 163).map (fun scalar => scalar.candidate)) ∧
      ((certParts 163).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf164 :
    ((rowsChunk wire 164).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 164).map (fun scalar => scalar.candidate)) ∧
      ((certParts 164).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf165 :
    ((rowsChunk wire 165).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 165).map (fun scalar => scalar.candidate)) ∧
      ((certParts 165).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf166 :
    ((rowsChunk wire 166).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 166).map (fun scalar => scalar.candidate)) ∧
      ((certParts 166).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf167 :
    ((rowsChunk wire 167).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 167).map (fun scalar => scalar.candidate)) ∧
      ((certParts 167).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf168 :
    ((rowsChunk wire 168).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 168).map (fun scalar => scalar.candidate)) ∧
      ((certParts 168).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 159 ≤ k → k < 169 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is159 : k = 159
  · subst is159
    exact (chunkLeaf159).1
  by_cases is160 : k = 160
  · subst is160
    exact (chunkLeaf160).1
  by_cases is161 : k = 161
  · subst is161
    exact (chunkLeaf161).1
  by_cases is162 : k = 162
  · subst is162
    exact (chunkLeaf162).1
  by_cases is163 : k = 163
  · subst is163
    exact (chunkLeaf163).1
  by_cases is164 : k = 164
  · subst is164
    exact (chunkLeaf164).1
  by_cases is165 : k = 165
  · subst is165
    exact (chunkLeaf165).1
  by_cases is166 : k = 166
  · subst is166
    exact (chunkLeaf166).1
  by_cases is167 : k = 167
  · subst is167
    exact (chunkLeaf167).1
  by_cases is168 : k = 168
  · subst is168
    exact (chunkLeaf168).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 159 ≤ k → k < 169 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is159 : k = 159
  · subst is159
    exact (chunkLeaf159).2
  by_cases is160 : k = 160
  · subst is160
    exact (chunkLeaf160).2
  by_cases is161 : k = 161
  · subst is161
    exact (chunkLeaf161).2
  by_cases is162 : k = 162
  · subst is162
    exact (chunkLeaf162).2
  by_cases is163 : k = 163
  · subst is163
    exact (chunkLeaf163).2
  by_cases is164 : k = 164
  · subst is164
    exact (chunkLeaf164).2
  by_cases is165 : k = 165
  · subst is165
    exact (chunkLeaf165).2
  by_cases is166 : k = 166
  · subst is166
    exact (chunkLeaf166).2
  by_cases is167 : k = 167
  · subst is167
    exact (chunkLeaf167).2
  by_cases is168 : k = 168
  · subst is168
    exact (chunkLeaf168).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf29
