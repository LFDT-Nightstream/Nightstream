import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf154 :
    (chunkFacts (rowsChunk wire 154) 39424 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 154 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk154) := by
  native_decide

theorem chunkLeaf155 :
    (chunkFacts (rowsChunk wire 155) 39680 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 155 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk155) := by
  native_decide

theorem chunkLeaf156 :
    (chunkFacts (rowsChunk wire 156) 39936 13 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 156 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk156) := by
  native_decide

theorem censusGroup :
    ∀ k, 154 ≤ k → k < 157 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is154 : k = 154
  · subst is154
    exact (chunkFacts_split (chunkLeaf154).1).1
  by_cases is155 : k = 155
  · subst is155
    exact (chunkFacts_split (chunkLeaf155).1).1
  by_cases is156 : k = 156
  · subst is156
    exact (chunkFacts_split (chunkLeaf156).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 154 ≤ k → k < 157 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is154 : k = 154
  · subst is154
    exact (chunkFacts_split (chunkLeaf154).1).2.1
  by_cases is155 : k = 155
  · subst is155
    exact (chunkFacts_split (chunkLeaf155).1).2.1
  by_cases is156 : k = 156
  · subst is156
    exact (chunkFacts_split (chunkLeaf156).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 154 ≤ k → k < 157 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is154 : k = 154
  · subst is154
    exact (chunkFacts_split (chunkLeaf154).1).2.2.1
  by_cases is155 : k = 155
  · subst is155
    exact (chunkFacts_split (chunkLeaf155).1).2.2.1
  by_cases is156 : k = 156
  · subst is156
    exact (chunkFacts_split (chunkLeaf156).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11
