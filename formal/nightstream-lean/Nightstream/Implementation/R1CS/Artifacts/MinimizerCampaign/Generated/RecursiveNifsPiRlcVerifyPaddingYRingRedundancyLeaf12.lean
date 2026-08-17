import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf12

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf58 :
    ((rowsChunk wire 58).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 58).map (fun scalar => scalar.candidate)) ∧
      ((certParts 58).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf59 :
    ((rowsChunk wire 59).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 59).map (fun scalar => scalar.candidate)) ∧
      ((certParts 59).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf60 :
    ((rowsChunk wire 60).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 60).map (fun scalar => scalar.candidate)) ∧
      ((certParts 60).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf61 :
    ((rowsChunk wire 61).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 61).map (fun scalar => scalar.candidate)) ∧
      ((certParts 61).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf62 :
    ((rowsChunk wire 62).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 62).map (fun scalar => scalar.candidate)) ∧
      ((certParts 62).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf63 :
    ((rowsChunk wire 63).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 63).map (fun scalar => scalar.candidate)) ∧
      ((certParts 63).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf64 :
    ((rowsChunk wire 64).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 64).map (fun scalar => scalar.candidate)) ∧
      ((certParts 64).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf65 :
    ((rowsChunk wire 65).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 65).map (fun scalar => scalar.candidate)) ∧
      ((certParts 65).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf66 :
    ((rowsChunk wire 66).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 66).map (fun scalar => scalar.candidate)) ∧
      ((certParts 66).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf67 :
    ((rowsChunk wire 67).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 67).map (fun scalar => scalar.candidate)) ∧
      ((certParts 67).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf68 :
    ((rowsChunk wire 68).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 68).map (fun scalar => scalar.candidate)) ∧
      ((certParts 68).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf69 :
    ((rowsChunk wire 69).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 69).map (fun scalar => scalar.candidate)) ∧
      ((certParts 69).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf70 :
    ((rowsChunk wire 70).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 70).map (fun scalar => scalar.candidate)) ∧
      ((certParts 70).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf71 :
    ((rowsChunk wire 71).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 71).map (fun scalar => scalar.candidate)) ∧
      ((certParts 71).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 58 ≤ k → k < 72 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is58 : k = 58
  · subst is58
    exact (chunkLeaf58).1
  by_cases is59 : k = 59
  · subst is59
    exact (chunkLeaf59).1
  by_cases is60 : k = 60
  · subst is60
    exact (chunkLeaf60).1
  by_cases is61 : k = 61
  · subst is61
    exact (chunkLeaf61).1
  by_cases is62 : k = 62
  · subst is62
    exact (chunkLeaf62).1
  by_cases is63 : k = 63
  · subst is63
    exact (chunkLeaf63).1
  by_cases is64 : k = 64
  · subst is64
    exact (chunkLeaf64).1
  by_cases is65 : k = 65
  · subst is65
    exact (chunkLeaf65).1
  by_cases is66 : k = 66
  · subst is66
    exact (chunkLeaf66).1
  by_cases is67 : k = 67
  · subst is67
    exact (chunkLeaf67).1
  by_cases is68 : k = 68
  · subst is68
    exact (chunkLeaf68).1
  by_cases is69 : k = 69
  · subst is69
    exact (chunkLeaf69).1
  by_cases is70 : k = 70
  · subst is70
    exact (chunkLeaf70).1
  by_cases is71 : k = 71
  · subst is71
    exact (chunkLeaf71).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 58 ≤ k → k < 72 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is58 : k = 58
  · subst is58
    exact (chunkLeaf58).2
  by_cases is59 : k = 59
  · subst is59
    exact (chunkLeaf59).2
  by_cases is60 : k = 60
  · subst is60
    exact (chunkLeaf60).2
  by_cases is61 : k = 61
  · subst is61
    exact (chunkLeaf61).2
  by_cases is62 : k = 62
  · subst is62
    exact (chunkLeaf62).2
  by_cases is63 : k = 63
  · subst is63
    exact (chunkLeaf63).2
  by_cases is64 : k = 64
  · subst is64
    exact (chunkLeaf64).2
  by_cases is65 : k = 65
  · subst is65
    exact (chunkLeaf65).2
  by_cases is66 : k = 66
  · subst is66
    exact (chunkLeaf66).2
  by_cases is67 : k = 67
  · subst is67
    exact (chunkLeaf67).2
  by_cases is68 : k = 68
  · subst is68
    exact (chunkLeaf68).2
  by_cases is69 : k = 69
  · subst is69
    exact (chunkLeaf69).2
  by_cases is70 : k = 70
  · subst is70
    exact (chunkLeaf70).2
  by_cases is71 : k = 71
  · subst is71
    exact (chunkLeaf71).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf12
