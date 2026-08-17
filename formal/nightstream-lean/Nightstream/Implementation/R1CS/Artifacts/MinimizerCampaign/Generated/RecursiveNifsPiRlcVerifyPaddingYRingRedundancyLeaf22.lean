import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf22

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf123 :
    ((rowsChunk wire 123).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 123).map (fun scalar => scalar.candidate)) ∧
      ((certParts 123).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf124 :
    ((rowsChunk wire 124).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 124).map (fun scalar => scalar.candidate)) ∧
      ((certParts 124).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 123 ≤ k → k < 125 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is123 : k = 123
  · subst is123
    exact (chunkLeaf123).1
  by_cases is124 : k = 124
  · subst is124
    exact (chunkLeaf124).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 123 ≤ k → k < 125 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is123 : k = 123
  · subst is123
    exact (chunkLeaf123).2
  by_cases is124 : k = 124
  · subst is124
    exact (chunkLeaf124).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf22
