import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf15

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf100 :
    ((rowsChunk wire 100).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 100).map (fun scalar => scalar.candidate)) ∧
      ((certParts 100).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf101 :
    ((rowsChunk wire 101).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 101).map (fun scalar => scalar.candidate)) ∧
      ((certParts 101).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 100 ≤ k → k < 102 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is100 : k = 100
  · subst is100
    exact (chunkLeaf100).1
  by_cases is101 : k = 101
  · subst is101
    exact (chunkLeaf101).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 100 ≤ k → k < 102 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is100 : k = 100
  · subst is100
    exact (chunkLeaf100).2
  by_cases is101 : k = 101
  · subst is101
    exact (chunkLeaf101).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf15
