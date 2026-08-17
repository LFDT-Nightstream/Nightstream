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

theorem chunkLeaf32 :
    ((rowsChunk wire 32).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 32).map (fun scalar => scalar.candidate)) ∧
      ((certParts 32).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf33 :
    ((rowsChunk wire 33).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 33).map (fun scalar => scalar.candidate)) ∧
      ((certParts 33).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf34 :
    ((rowsChunk wire 34).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 34).map (fun scalar => scalar.candidate)) ∧
      ((certParts 34).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf35 :
    ((rowsChunk wire 35).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 35).map (fun scalar => scalar.candidate)) ∧
      ((certParts 35).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf36 :
    ((rowsChunk wire 36).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 36).map (fun scalar => scalar.candidate)) ∧
      ((certParts 36).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf37 :
    ((rowsChunk wire 37).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 37).map (fun scalar => scalar.candidate)) ∧
      ((certParts 37).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf38 :
    ((rowsChunk wire 38).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 38).map (fun scalar => scalar.candidate)) ∧
      ((certParts 38).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf39 :
    ((rowsChunk wire 39).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 39).map (fun scalar => scalar.candidate)) ∧
      ((certParts 39).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf40 :
    ((rowsChunk wire 40).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 40).map (fun scalar => scalar.candidate)) ∧
      ((certParts 40).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf41 :
    ((rowsChunk wire 41).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 41).map (fun scalar => scalar.candidate)) ∧
      ((certParts 41).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf42 :
    ((rowsChunk wire 42).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 42).map (fun scalar => scalar.candidate)) ∧
      ((certParts 42).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf43 :
    ((rowsChunk wire 43).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 43).map (fun scalar => scalar.candidate)) ∧
      ((certParts 43).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf44 :
    ((rowsChunk wire 44).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 44).map (fun scalar => scalar.candidate)) ∧
      ((certParts 44).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf45 :
    ((rowsChunk wire 45).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 45).map (fun scalar => scalar.candidate)) ∧
      ((certParts 45).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 32 ≤ k → k < 46 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is32 : k = 32
  · subst is32
    exact (chunkLeaf32).1
  by_cases is33 : k = 33
  · subst is33
    exact (chunkLeaf33).1
  by_cases is34 : k = 34
  · subst is34
    exact (chunkLeaf34).1
  by_cases is35 : k = 35
  · subst is35
    exact (chunkLeaf35).1
  by_cases is36 : k = 36
  · subst is36
    exact (chunkLeaf36).1
  by_cases is37 : k = 37
  · subst is37
    exact (chunkLeaf37).1
  by_cases is38 : k = 38
  · subst is38
    exact (chunkLeaf38).1
  by_cases is39 : k = 39
  · subst is39
    exact (chunkLeaf39).1
  by_cases is40 : k = 40
  · subst is40
    exact (chunkLeaf40).1
  by_cases is41 : k = 41
  · subst is41
    exact (chunkLeaf41).1
  by_cases is42 : k = 42
  · subst is42
    exact (chunkLeaf42).1
  by_cases is43 : k = 43
  · subst is43
    exact (chunkLeaf43).1
  by_cases is44 : k = 44
  · subst is44
    exact (chunkLeaf44).1
  by_cases is45 : k = 45
  · subst is45
    exact (chunkLeaf45).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 32 ≤ k → k < 46 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is32 : k = 32
  · subst is32
    exact (chunkLeaf32).2
  by_cases is33 : k = 33
  · subst is33
    exact (chunkLeaf33).2
  by_cases is34 : k = 34
  · subst is34
    exact (chunkLeaf34).2
  by_cases is35 : k = 35
  · subst is35
    exact (chunkLeaf35).2
  by_cases is36 : k = 36
  · subst is36
    exact (chunkLeaf36).2
  by_cases is37 : k = 37
  · subst is37
    exact (chunkLeaf37).2
  by_cases is38 : k = 38
  · subst is38
    exact (chunkLeaf38).2
  by_cases is39 : k = 39
  · subst is39
    exact (chunkLeaf39).2
  by_cases is40 : k = 40
  · subst is40
    exact (chunkLeaf40).2
  by_cases is41 : k = 41
  · subst is41
    exact (chunkLeaf41).2
  by_cases is42 : k = 42
  · subst is42
    exact (chunkLeaf42).2
  by_cases is43 : k = 43
  · subst is43
    exact (chunkLeaf43).2
  by_cases is44 : k = 44
  · subst is44
    exact (chunkLeaf44).2
  by_cases is45 : k = 45
  · subst is45
    exact (chunkLeaf45).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf9
