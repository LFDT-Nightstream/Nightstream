import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf21

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf109 :
    ((rowsChunk wire 109).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 109).map (fun scalar => scalar.candidate)) ∧
      ((certParts 109).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf110 :
    ((rowsChunk wire 110).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 110).map (fun scalar => scalar.candidate)) ∧
      ((certParts 110).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf111 :
    ((rowsChunk wire 111).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 111).map (fun scalar => scalar.candidate)) ∧
      ((certParts 111).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf112 :
    ((rowsChunk wire 112).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 112).map (fun scalar => scalar.candidate)) ∧
      ((certParts 112).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf113 :
    ((rowsChunk wire 113).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 113).map (fun scalar => scalar.candidate)) ∧
      ((certParts 113).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf114 :
    ((rowsChunk wire 114).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 114).map (fun scalar => scalar.candidate)) ∧
      ((certParts 114).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf115 :
    ((rowsChunk wire 115).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 115).map (fun scalar => scalar.candidate)) ∧
      ((certParts 115).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf116 :
    ((rowsChunk wire 116).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 116).map (fun scalar => scalar.candidate)) ∧
      ((certParts 116).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf117 :
    ((rowsChunk wire 117).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 117).map (fun scalar => scalar.candidate)) ∧
      ((certParts 117).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf118 :
    ((rowsChunk wire 118).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 118).map (fun scalar => scalar.candidate)) ∧
      ((certParts 118).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf119 :
    ((rowsChunk wire 119).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 119).map (fun scalar => scalar.candidate)) ∧
      ((certParts 119).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf120 :
    ((rowsChunk wire 120).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 120).map (fun scalar => scalar.candidate)) ∧
      ((certParts 120).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf121 :
    ((rowsChunk wire 121).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 121).map (fun scalar => scalar.candidate)) ∧
      ((certParts 121).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf122 :
    ((rowsChunk wire 122).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 122).map (fun scalar => scalar.candidate)) ∧
      ((certParts 122).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 109 ≤ k → k < 123 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is109 : k = 109
  · subst is109
    exact (chunkLeaf109).1
  by_cases is110 : k = 110
  · subst is110
    exact (chunkLeaf110).1
  by_cases is111 : k = 111
  · subst is111
    exact (chunkLeaf111).1
  by_cases is112 : k = 112
  · subst is112
    exact (chunkLeaf112).1
  by_cases is113 : k = 113
  · subst is113
    exact (chunkLeaf113).1
  by_cases is114 : k = 114
  · subst is114
    exact (chunkLeaf114).1
  by_cases is115 : k = 115
  · subst is115
    exact (chunkLeaf115).1
  by_cases is116 : k = 116
  · subst is116
    exact (chunkLeaf116).1
  by_cases is117 : k = 117
  · subst is117
    exact (chunkLeaf117).1
  by_cases is118 : k = 118
  · subst is118
    exact (chunkLeaf118).1
  by_cases is119 : k = 119
  · subst is119
    exact (chunkLeaf119).1
  by_cases is120 : k = 120
  · subst is120
    exact (chunkLeaf120).1
  by_cases is121 : k = 121
  · subst is121
    exact (chunkLeaf121).1
  by_cases is122 : k = 122
  · subst is122
    exact (chunkLeaf122).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 109 ≤ k → k < 123 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is109 : k = 109
  · subst is109
    exact (chunkLeaf109).2
  by_cases is110 : k = 110
  · subst is110
    exact (chunkLeaf110).2
  by_cases is111 : k = 111
  · subst is111
    exact (chunkLeaf111).2
  by_cases is112 : k = 112
  · subst is112
    exact (chunkLeaf112).2
  by_cases is113 : k = 113
  · subst is113
    exact (chunkLeaf113).2
  by_cases is114 : k = 114
  · subst is114
    exact (chunkLeaf114).2
  by_cases is115 : k = 115
  · subst is115
    exact (chunkLeaf115).2
  by_cases is116 : k = 116
  · subst is116
    exact (chunkLeaf116).2
  by_cases is117 : k = 117
  · subst is117
    exact (chunkLeaf117).2
  by_cases is118 : k = 118
  · subst is118
    exact (chunkLeaf118).2
  by_cases is119 : k = 119
  · subst is119
    exact (chunkLeaf119).2
  by_cases is120 : k = 120
  · subst is120
    exact (chunkLeaf120).2
  by_cases is121 : k = 121
  · subst is121
    exact (chunkLeaf121).2
  by_cases is122 : k = 122
  · subst is122
    exact (chunkLeaf122).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf21
