import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf3

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf3 :
    ((rowsChunk wire 3).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 3).map (fun scalar => scalar.candidate)) ∧
      ((certParts 3).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf4 :
    ((rowsChunk wire 4).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 4).map (fun scalar => scalar.candidate)) ∧
      ((certParts 4).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf5 :
    ((rowsChunk wire 5).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 5).map (fun scalar => scalar.candidate)) ∧
      ((certParts 5).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf6 :
    ((rowsChunk wire 6).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 6).map (fun scalar => scalar.candidate)) ∧
      ((certParts 6).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 3 ≤ k → k < 7 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is3 : k = 3
  · subst is3
    exact (chunkLeaf3).1
  by_cases is4 : k = 4
  · subst is4
    exact (chunkLeaf4).1
  by_cases is5 : k = 5
  · subst is5
    exact (chunkLeaf5).1
  by_cases is6 : k = 6
  · subst is6
    exact (chunkLeaf6).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 3 ≤ k → k < 7 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is3 : k = 3
  · subst is3
    exact (chunkLeaf3).2
  by_cases is4 : k = 4
  · subst is4
    exact (chunkLeaf4).2
  by_cases is5 : k = 5
  · subst is5
    exact (chunkLeaf5).2
  by_cases is6 : k = 6
  · subst is6
    exact (chunkLeaf6).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf3
