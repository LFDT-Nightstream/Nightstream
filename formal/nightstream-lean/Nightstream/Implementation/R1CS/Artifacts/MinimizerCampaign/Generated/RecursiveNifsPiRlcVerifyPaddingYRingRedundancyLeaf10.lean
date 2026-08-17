import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf10

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf46 :
    ((rowsChunk wire 46).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 46).map (fun scalar => scalar.candidate)) ∧
      ((certParts 46).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf47 :
    ((rowsChunk wire 47).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 47).map (fun scalar => scalar.candidate)) ∧
      ((certParts 47).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf48 :
    ((rowsChunk wire 48).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 48).map (fun scalar => scalar.candidate)) ∧
      ((certParts 48).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf49 :
    ((rowsChunk wire 49).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 49).map (fun scalar => scalar.candidate)) ∧
      ((certParts 49).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf50 :
    ((rowsChunk wire 50).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 50).map (fun scalar => scalar.candidate)) ∧
      ((certParts 50).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf51 :
    ((rowsChunk wire 51).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 51).map (fun scalar => scalar.candidate)) ∧
      ((certParts 51).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf52 :
    ((rowsChunk wire 52).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 52).map (fun scalar => scalar.candidate)) ∧
      ((certParts 52).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf53 :
    ((rowsChunk wire 53).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 53).map (fun scalar => scalar.candidate)) ∧
      ((certParts 53).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf54 :
    ((rowsChunk wire 54).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 54).map (fun scalar => scalar.candidate)) ∧
      ((certParts 54).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf55 :
    ((rowsChunk wire 55).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 55).map (fun scalar => scalar.candidate)) ∧
      ((certParts 55).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf56 :
    ((rowsChunk wire 56).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 56).map (fun scalar => scalar.candidate)) ∧
      ((certParts 56).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 46 ≤ k → k < 57 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is46 : k = 46
  · subst is46
    exact (chunkLeaf46).1
  by_cases is47 : k = 47
  · subst is47
    exact (chunkLeaf47).1
  by_cases is48 : k = 48
  · subst is48
    exact (chunkLeaf48).1
  by_cases is49 : k = 49
  · subst is49
    exact (chunkLeaf49).1
  by_cases is50 : k = 50
  · subst is50
    exact (chunkLeaf50).1
  by_cases is51 : k = 51
  · subst is51
    exact (chunkLeaf51).1
  by_cases is52 : k = 52
  · subst is52
    exact (chunkLeaf52).1
  by_cases is53 : k = 53
  · subst is53
    exact (chunkLeaf53).1
  by_cases is54 : k = 54
  · subst is54
    exact (chunkLeaf54).1
  by_cases is55 : k = 55
  · subst is55
    exact (chunkLeaf55).1
  by_cases is56 : k = 56
  · subst is56
    exact (chunkLeaf56).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 46 ≤ k → k < 57 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is46 : k = 46
  · subst is46
    exact (chunkLeaf46).2
  by_cases is47 : k = 47
  · subst is47
    exact (chunkLeaf47).2
  by_cases is48 : k = 48
  · subst is48
    exact (chunkLeaf48).2
  by_cases is49 : k = 49
  · subst is49
    exact (chunkLeaf49).2
  by_cases is50 : k = 50
  · subst is50
    exact (chunkLeaf50).2
  by_cases is51 : k = 51
  · subst is51
    exact (chunkLeaf51).2
  by_cases is52 : k = 52
  · subst is52
    exact (chunkLeaf52).2
  by_cases is53 : k = 53
  · subst is53
    exact (chunkLeaf53).2
  by_cases is54 : k = 54
  · subst is54
    exact (chunkLeaf54).2
  by_cases is55 : k = 55
  · subst is55
    exact (chunkLeaf55).2
  by_cases is56 : k = 56
  · subst is56
    exact (chunkLeaf56).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf10
