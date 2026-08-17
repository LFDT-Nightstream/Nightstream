import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf14

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf86 :
    ((rowsChunk wire 86).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 86).map (fun scalar => scalar.candidate)) ∧
      ((certParts 86).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf87 :
    ((rowsChunk wire 87).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 87).map (fun scalar => scalar.candidate)) ∧
      ((certParts 87).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf88 :
    ((rowsChunk wire 88).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 88).map (fun scalar => scalar.candidate)) ∧
      ((certParts 88).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf89 :
    ((rowsChunk wire 89).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 89).map (fun scalar => scalar.candidate)) ∧
      ((certParts 89).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf90 :
    ((rowsChunk wire 90).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 90).map (fun scalar => scalar.candidate)) ∧
      ((certParts 90).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf91 :
    ((rowsChunk wire 91).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 91).map (fun scalar => scalar.candidate)) ∧
      ((certParts 91).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf92 :
    ((rowsChunk wire 92).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 92).map (fun scalar => scalar.candidate)) ∧
      ((certParts 92).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf93 :
    ((rowsChunk wire 93).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 93).map (fun scalar => scalar.candidate)) ∧
      ((certParts 93).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf94 :
    ((rowsChunk wire 94).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 94).map (fun scalar => scalar.candidate)) ∧
      ((certParts 94).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf95 :
    ((rowsChunk wire 95).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 95).map (fun scalar => scalar.candidate)) ∧
      ((certParts 95).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf96 :
    ((rowsChunk wire 96).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 96).map (fun scalar => scalar.candidate)) ∧
      ((certParts 96).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf97 :
    ((rowsChunk wire 97).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 97).map (fun scalar => scalar.candidate)) ∧
      ((certParts 97).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf98 :
    ((rowsChunk wire 98).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 98).map (fun scalar => scalar.candidate)) ∧
      ((certParts 98).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf99 :
    ((rowsChunk wire 99).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 99).map (fun scalar => scalar.candidate)) ∧
      ((certParts 99).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 86 ≤ k → k < 100 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is86 : k = 86
  · subst is86
    exact (chunkLeaf86).1
  by_cases is87 : k = 87
  · subst is87
    exact (chunkLeaf87).1
  by_cases is88 : k = 88
  · subst is88
    exact (chunkLeaf88).1
  by_cases is89 : k = 89
  · subst is89
    exact (chunkLeaf89).1
  by_cases is90 : k = 90
  · subst is90
    exact (chunkLeaf90).1
  by_cases is91 : k = 91
  · subst is91
    exact (chunkLeaf91).1
  by_cases is92 : k = 92
  · subst is92
    exact (chunkLeaf92).1
  by_cases is93 : k = 93
  · subst is93
    exact (chunkLeaf93).1
  by_cases is94 : k = 94
  · subst is94
    exact (chunkLeaf94).1
  by_cases is95 : k = 95
  · subst is95
    exact (chunkLeaf95).1
  by_cases is96 : k = 96
  · subst is96
    exact (chunkLeaf96).1
  by_cases is97 : k = 97
  · subst is97
    exact (chunkLeaf97).1
  by_cases is98 : k = 98
  · subst is98
    exact (chunkLeaf98).1
  by_cases is99 : k = 99
  · subst is99
    exact (chunkLeaf99).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 86 ≤ k → k < 100 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is86 : k = 86
  · subst is86
    exact (chunkLeaf86).2
  by_cases is87 : k = 87
  · subst is87
    exact (chunkLeaf87).2
  by_cases is88 : k = 88
  · subst is88
    exact (chunkLeaf88).2
  by_cases is89 : k = 89
  · subst is89
    exact (chunkLeaf89).2
  by_cases is90 : k = 90
  · subst is90
    exact (chunkLeaf90).2
  by_cases is91 : k = 91
  · subst is91
    exact (chunkLeaf91).2
  by_cases is92 : k = 92
  · subst is92
    exact (chunkLeaf92).2
  by_cases is93 : k = 93
  · subst is93
    exact (chunkLeaf93).2
  by_cases is94 : k = 94
  · subst is94
    exact (chunkLeaf94).2
  by_cases is95 : k = 95
  · subst is95
    exact (chunkLeaf95).2
  by_cases is96 : k = 96
  · subst is96
    exact (chunkLeaf96).2
  by_cases is97 : k = 97
  · subst is97
    exact (chunkLeaf97).2
  by_cases is98 : k = 98
  · subst is98
    exact (chunkLeaf98).2
  by_cases is99 : k = 99
  · subst is99
    exact (chunkLeaf99).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf14
