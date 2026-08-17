import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf24

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf126 :
    ((rowsChunk wire 126).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 126).map (fun scalar => scalar.candidate)) ∧
      ((certParts 126).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf127 :
    ((rowsChunk wire 127).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 127).map (fun scalar => scalar.candidate)) ∧
      ((certParts 127).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf128 :
    ((rowsChunk wire 128).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 128).map (fun scalar => scalar.candidate)) ∧
      ((certParts 128).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 126 ≤ k → k < 129 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (chunkLeaf126).1
  by_cases is127 : k = 127
  · subst is127
    exact (chunkLeaf127).1
  by_cases is128 : k = 128
  · subst is128
    exact (chunkLeaf128).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 126 ≤ k → k < 129 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (chunkLeaf126).2
  by_cases is127 : k = 127
  · subst is127
    exact (chunkLeaf127).2
  by_cases is128 : k = 128
  · subst is128
    exact (chunkLeaf128).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf24
