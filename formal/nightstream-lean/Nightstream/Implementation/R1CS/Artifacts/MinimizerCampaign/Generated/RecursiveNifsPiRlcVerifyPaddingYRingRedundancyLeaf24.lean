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

theorem candLeaf126 :
    (rowsChunk wire 126).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 126).map (fun scalar => scalar.candidate) := by
  native_decide

theorem candLeaf127 :
    (rowsChunk wire 127).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 127).map (fun scalar => scalar.candidate) := by
  native_decide

theorem coveredLeaf127x0 :
    supportsCovered wire 0 (certParts 127) = true := by
  native_decide

theorem coveredLeaf127x128 :
    supportsCovered wire 128 (certParts 127) = true := by
  native_decide

theorem homesLeaf127 :
    (leafSupports (certParts 127)).all (fun source =>
      decide (source.sourceIndex / wire.chunkRows ∈ [0, 128])) = true := by
  native_decide

theorem shapeLeaf127 :
    scalarShapeOk certPlan certFamily (certParts 127) = true := by
  native_decide

theorem candLeaf128 :
    (rowsChunk wire 128).filter
        (fun row => decide (row.family = certFamily)) =
      (certParts 128).map (fun scalar => scalar.candidate) := by
  native_decide

theorem coveredLeaf128x0 :
    supportsCovered wire 0 (certParts 128) = true := by
  native_decide

theorem coveredLeaf128x128 :
    supportsCovered wire 128 (certParts 128) = true := by
  native_decide

theorem homesLeaf128 :
    (leafSupports (certParts 128)).all (fun source =>
      decide (source.sourceIndex / wire.chunkRows ∈ [0, 128])) = true := by
  native_decide

theorem shapeLeaf128 :
    scalarShapeOk certPlan certFamily (certParts 128) = true := by
  native_decide

theorem candGroup :
    ∀ k, 126 ≤ k → k < 129 →
      (rowsChunk wire k).filter
          (fun row => decide (row.family = certFamily)) =
        (certParts k).map (fun scalar => scalar.candidate) := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact candLeaf126
  by_cases is127 : k = 127
  · subst is127
    exact candLeaf127
  by_cases is128 : k = 128
  · subst is128
    exact candLeaf128
  exact absurd upper (by omega)


theorem scalarsGroup :
    ∀ k, 126 ≤ k → k < 129 → ∀ scalar ∈ certParts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ certPlan ∧
              support.source.family ≠ certFamily := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    intro scalar member
    rw [show certParts 126 = [] from rfl] at member
    cases member
  by_cases is127 : k = 127
  · subst is127
    exact scalar_facts_of_leaf_checks wire certPlan certFamily
      (certParts 127) [0, 128]
      (by
        intro chunk member
        simp only [List.mem_cons, List.not_mem_nil, or_false] at member
        rcases member with rfl | rfl
        · exact coveredLeaf127x0
        · exact coveredLeaf127x128
      )
      homesLeaf127 shapeLeaf127
  by_cases is128 : k = 128
  · subst is128
    exact scalar_facts_of_leaf_checks wire certPlan certFamily
      (certParts 128) [0, 128]
      (by
        intro chunk member
        simp only [List.mem_cons, List.not_mem_nil, or_false] at member
        rcases member with rfl | rfl
        · exact coveredLeaf128x0
        · exact coveredLeaf128x128
      )
      homesLeaf128 shapeLeaf128
  exact absurd upper (by omega)


end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveNifsPiRlcVerifyPaddingYRingRedundancyLeaf24
