import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf17

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf103 :
    ((rowsChunk wire 103).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 103).map (fun scalar => scalar.candidate)) ∧
      ((certParts 103).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf104 :
    ((rowsChunk wire 104).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 104).map (fun scalar => scalar.candidate)) ∧
      ((certParts 104).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf105 :
    ((rowsChunk wire 105).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 105).map (fun scalar => scalar.candidate)) ∧
      ((certParts 105).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 103 ≤ k → k < 106 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is103 : k = 103
  · subst is103
    exact (chunkLeaf103).1
  by_cases is104 : k = 104
  · subst is104
    exact (chunkLeaf104).1
  by_cases is105 : k = 105
  · subst is105
    exact (chunkLeaf105).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 103 ≤ k → k < 106 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is103 : k = 103
  · subst is103
    exact (chunkLeaf103).2
  by_cases is104 : k = 104
  · subst is104
    exact (chunkLeaf104).2
  by_cases is105 : k = 105
  · subst is105
    exact (chunkLeaf105).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf17
