import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf13

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf72 :
    ((rowsChunk wire 72).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 72).map (fun scalar => scalar.candidate)) ∧
      ((certParts 72).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf73 :
    ((rowsChunk wire 73).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 73).map (fun scalar => scalar.candidate)) ∧
      ((certParts 73).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf74 :
    ((rowsChunk wire 74).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 74).map (fun scalar => scalar.candidate)) ∧
      ((certParts 74).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf75 :
    ((rowsChunk wire 75).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 75).map (fun scalar => scalar.candidate)) ∧
      ((certParts 75).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf76 :
    ((rowsChunk wire 76).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 76).map (fun scalar => scalar.candidate)) ∧
      ((certParts 76).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf77 :
    ((rowsChunk wire 77).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 77).map (fun scalar => scalar.candidate)) ∧
      ((certParts 77).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf78 :
    ((rowsChunk wire 78).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 78).map (fun scalar => scalar.candidate)) ∧
      ((certParts 78).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf79 :
    ((rowsChunk wire 79).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 79).map (fun scalar => scalar.candidate)) ∧
      ((certParts 79).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf80 :
    ((rowsChunk wire 80).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 80).map (fun scalar => scalar.candidate)) ∧
      ((certParts 80).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf81 :
    ((rowsChunk wire 81).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 81).map (fun scalar => scalar.candidate)) ∧
      ((certParts 81).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf82 :
    ((rowsChunk wire 82).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 82).map (fun scalar => scalar.candidate)) ∧
      ((certParts 82).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf83 :
    ((rowsChunk wire 83).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 83).map (fun scalar => scalar.candidate)) ∧
      ((certParts 83).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf84 :
    ((rowsChunk wire 84).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 84).map (fun scalar => scalar.candidate)) ∧
      ((certParts 84).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf85 :
    ((rowsChunk wire 85).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 85).map (fun scalar => scalar.candidate)) ∧
      ((certParts 85).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 72 ≤ k → k < 86 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is72 : k = 72
  · subst is72
    exact (chunkLeaf72).1
  by_cases is73 : k = 73
  · subst is73
    exact (chunkLeaf73).1
  by_cases is74 : k = 74
  · subst is74
    exact (chunkLeaf74).1
  by_cases is75 : k = 75
  · subst is75
    exact (chunkLeaf75).1
  by_cases is76 : k = 76
  · subst is76
    exact (chunkLeaf76).1
  by_cases is77 : k = 77
  · subst is77
    exact (chunkLeaf77).1
  by_cases is78 : k = 78
  · subst is78
    exact (chunkLeaf78).1
  by_cases is79 : k = 79
  · subst is79
    exact (chunkLeaf79).1
  by_cases is80 : k = 80
  · subst is80
    exact (chunkLeaf80).1
  by_cases is81 : k = 81
  · subst is81
    exact (chunkLeaf81).1
  by_cases is82 : k = 82
  · subst is82
    exact (chunkLeaf82).1
  by_cases is83 : k = 83
  · subst is83
    exact (chunkLeaf83).1
  by_cases is84 : k = 84
  · subst is84
    exact (chunkLeaf84).1
  by_cases is85 : k = 85
  · subst is85
    exact (chunkLeaf85).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 72 ≤ k → k < 86 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is72 : k = 72
  · subst is72
    exact (chunkLeaf72).2
  by_cases is73 : k = 73
  · subst is73
    exact (chunkLeaf73).2
  by_cases is74 : k = 74
  · subst is74
    exact (chunkLeaf74).2
  by_cases is75 : k = 75
  · subst is75
    exact (chunkLeaf75).2
  by_cases is76 : k = 76
  · subst is76
    exact (chunkLeaf76).2
  by_cases is77 : k = 77
  · subst is77
    exact (chunkLeaf77).2
  by_cases is78 : k = 78
  · subst is78
    exact (chunkLeaf78).2
  by_cases is79 : k = 79
  · subst is79
    exact (chunkLeaf79).2
  by_cases is80 : k = 80
  · subst is80
    exact (chunkLeaf80).2
  by_cases is81 : k = 81
  · subst is81
    exact (chunkLeaf81).2
  by_cases is82 : k = 82
  · subst is82
    exact (chunkLeaf82).2
  by_cases is83 : k = 83
  · subst is83
    exact (chunkLeaf83).2
  by_cases is84 : k = 84
  · subst is84
    exact (chunkLeaf84).2
  by_cases is85 : k = 85
  · subst is85
    exact (chunkLeaf85).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf13
