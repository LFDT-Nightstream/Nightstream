import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

/-!
GENERATED FILE - do not edit by hand.

Bounded redundancy leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf8

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyParts

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf18 :
    ((rowsChunk wire 18).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 18).map (fun scalar => scalar.candidate)) ∧
      ((certParts 18).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf19 :
    ((rowsChunk wire 19).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 19).map (fun scalar => scalar.candidate)) ∧
      ((certParts 19).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf20 :
    ((rowsChunk wire 20).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 20).map (fun scalar => scalar.candidate)) ∧
      ((certParts 20).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf21 :
    ((rowsChunk wire 21).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 21).map (fun scalar => scalar.candidate)) ∧
      ((certParts 21).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf22 :
    ((rowsChunk wire 22).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 22).map (fun scalar => scalar.candidate)) ∧
      ((certParts 22).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf23 :
    ((rowsChunk wire 23).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 23).map (fun scalar => scalar.candidate)) ∧
      ((certParts 23).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf24 :
    ((rowsChunk wire 24).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 24).map (fun scalar => scalar.candidate)) ∧
      ((certParts 24).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf25 :
    ((rowsChunk wire 25).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 25).map (fun scalar => scalar.candidate)) ∧
      ((certParts 25).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf26 :
    ((rowsChunk wire 26).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 26).map (fun scalar => scalar.candidate)) ∧
      ((certParts 26).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf27 :
    ((rowsChunk wire 27).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 27).map (fun scalar => scalar.candidate)) ∧
      ((certParts 27).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf28 :
    ((rowsChunk wire 28).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 28).map (fun scalar => scalar.candidate)) ∧
      ((certParts 28).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf29 :
    ((rowsChunk wire 29).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 29).map (fun scalar => scalar.candidate)) ∧
      ((certParts 29).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf30 :
    ((rowsChunk wire 30).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 30).map (fun scalar => scalar.candidate)) ∧
      ((certParts 30).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem chunkLeaf31 :
    ((rowsChunk wire 31).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 31).map (fun scalar => scalar.candidate)) ∧
      ((certParts 31).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true) := by
  native_decide

theorem candGroup :
    ∀ k, 18 ≤ k → k < 32 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is18 : k = 18
  · subst is18
    exact (chunkLeaf18).1
  by_cases is19 : k = 19
  · subst is19
    exact (chunkLeaf19).1
  by_cases is20 : k = 20
  · subst is20
    exact (chunkLeaf20).1
  by_cases is21 : k = 21
  · subst is21
    exact (chunkLeaf21).1
  by_cases is22 : k = 22
  · subst is22
    exact (chunkLeaf22).1
  by_cases is23 : k = 23
  · subst is23
    exact (chunkLeaf23).1
  by_cases is24 : k = 24
  · subst is24
    exact (chunkLeaf24).1
  by_cases is25 : k = 25
  · subst is25
    exact (chunkLeaf25).1
  by_cases is26 : k = 26
  · subst is26
    exact (chunkLeaf26).1
  by_cases is27 : k = 27
  · subst is27
    exact (chunkLeaf27).1
  by_cases is28 : k = 28
  · subst is28
    exact (chunkLeaf28).1
  by_cases is29 : k = 29
  · subst is29
    exact (chunkLeaf29).1
  by_cases is30 : k = 30
  · subst is30
    exact (chunkLeaf30).1
  by_cases is31 : k = 31
  · subst is31
    exact (chunkLeaf31).1
  exact absurd upper (by omega)


theorem scalarGroup :
    ∀ k, 18 ≤ k → k < 32 →
      (certParts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire certPlan certFamily)) = true := by
  intro k lower upper
  by_cases is18 : k = 18
  · subst is18
    exact (chunkLeaf18).2
  by_cases is19 : k = 19
  · subst is19
    exact (chunkLeaf19).2
  by_cases is20 : k = 20
  · subst is20
    exact (chunkLeaf20).2
  by_cases is21 : k = 21
  · subst is21
    exact (chunkLeaf21).2
  by_cases is22 : k = 22
  · subst is22
    exact (chunkLeaf22).2
  by_cases is23 : k = 23
  · subst is23
    exact (chunkLeaf23).2
  by_cases is24 : k = 24
  · subst is24
    exact (chunkLeaf24).2
  by_cases is25 : k = 25
  · subst is25
    exact (chunkLeaf25).2
  by_cases is26 : k = 26
  · subst is26
    exact (chunkLeaf26).2
  by_cases is27 : k = 27
  · subst is27
    exact (chunkLeaf27).2
  by_cases is28 : k = 28
  · subst is28
    exact (chunkLeaf28).2
  by_cases is29 : k = 29
  · subst is29
    exact (chunkLeaf29).2
  by_cases is30 : k = 30
  · subst is30
    exact (chunkLeaf30).2
  by_cases is31 : k = 31
  · subst is31
    exact (chunkLeaf31).2
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf8
