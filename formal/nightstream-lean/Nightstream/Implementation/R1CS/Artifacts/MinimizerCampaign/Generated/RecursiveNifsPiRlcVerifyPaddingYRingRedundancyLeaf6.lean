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

theorem chunkLeaf9 :
    ((rowsChunk wire 9).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 9).map (fun scalar => scalar.candidate)) ∧
      ((certParts 9).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf10 :
    ((rowsChunk wire 10).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 10).map (fun scalar => scalar.candidate)) ∧
      ((certParts 10).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf11 :
    ((rowsChunk wire 11).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 11).map (fun scalar => scalar.candidate)) ∧
      ((certParts 11).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf12 :
    ((rowsChunk wire 12).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 12).map (fun scalar => scalar.candidate)) ∧
      ((certParts 12).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf13 :
    ((rowsChunk wire 13).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 13).map (fun scalar => scalar.candidate)) ∧
      ((certParts 13).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf14 :
    ((rowsChunk wire 14).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 14).map (fun scalar => scalar.candidate)) ∧
      ((certParts 14).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf15 :
    ((rowsChunk wire 15).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 15).map (fun scalar => scalar.candidate)) ∧
      ((certParts 15).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf16 :
    ((rowsChunk wire 16).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 16).map (fun scalar => scalar.candidate)) ∧
      ((certParts 16).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 9 ≤ k → k < 17 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is9 : k = 9
  · subst is9
    exact (chunkLeaf9).1
  by_cases is10 : k = 10
  · subst is10
    exact (chunkLeaf10).1
  by_cases is11 : k = 11
  · subst is11
    exact (chunkLeaf11).1
  by_cases is12 : k = 12
  · subst is12
    exact (chunkLeaf12).1
  by_cases is13 : k = 13
  · subst is13
    exact (chunkLeaf13).1
  by_cases is14 : k = 14
  · subst is14
    exact (chunkLeaf14).1
  by_cases is15 : k = 15
  · subst is15
    exact (chunkLeaf15).1
  by_cases is16 : k = 16
  · subst is16
    exact (chunkLeaf16).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 9 ≤ k → k < 17 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is9 : k = 9
  · subst is9
    exact (chunkLeaf9).2
  by_cases is10 : k = 10
  · subst is10
    exact (chunkLeaf10).2
  by_cases is11 : k = 11
  · subst is11
    exact (chunkLeaf11).2
  by_cases is12 : k = 12
  · subst is12
    exact (chunkLeaf12).2
  by_cases is13 : k = 13
  · subst is13
    exact (chunkLeaf13).2
  by_cases is14 : k = 14
  · subst is14
    exact (chunkLeaf14).2
  by_cases is15 : k = 15
  · subst is15
    exact (chunkLeaf15).2
  by_cases is16 : k = 16
  · subst is16
    exact (chunkLeaf16).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf6
