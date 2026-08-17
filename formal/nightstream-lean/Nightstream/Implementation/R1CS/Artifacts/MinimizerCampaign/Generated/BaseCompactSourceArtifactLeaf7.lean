import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf98 :
    (chunkFacts (rowsChunk wire 98) 25088 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 98 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk98) := by
  native_decide

theorem chunkLeaf99 :
    (chunkFacts (rowsChunk wire 99) 25344 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 99 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk99) := by
  native_decide

theorem chunkLeaf100 :
    (chunkFacts (rowsChunk wire 100) 25600 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 100 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk100) := by
  native_decide

theorem chunkLeaf101 :
    (chunkFacts (rowsChunk wire 101) 25856 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 101 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk101) := by
  native_decide

theorem chunkLeaf102 :
    (chunkFacts (rowsChunk wire 102) 26112 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 102 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk102) := by
  native_decide

theorem chunkLeaf103 :
    (chunkFacts (rowsChunk wire 103) 26368 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 103 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk103) := by
  native_decide

theorem chunkLeaf104 :
    (chunkFacts (rowsChunk wire 104) 26624 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 104 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk104) := by
  native_decide

theorem chunkLeaf105 :
    (chunkFacts (rowsChunk wire 105) 26880 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 105 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk105) := by
  native_decide

theorem chunkLeaf106 :
    (chunkFacts (rowsChunk wire 106) 27136 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 106 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk106) := by
  native_decide

theorem chunkLeaf107 :
    (chunkFacts (rowsChunk wire 107) 27392 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 107 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk107) := by
  native_decide

theorem chunkLeaf108 :
    (chunkFacts (rowsChunk wire 108) 27648 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 108 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk108) := by
  native_decide

theorem chunkLeaf109 :
    (chunkFacts (rowsChunk wire 109) 27904 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 109 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk109) := by
  native_decide

theorem chunkLeaf110 :
    (chunkFacts (rowsChunk wire 110) 28160 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 110 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk110) := by
  native_decide

theorem chunkLeaf111 :
    (chunkFacts (rowsChunk wire 111) 28416 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 111 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk111) := by
  native_decide

theorem censusGroup :
    ∀ k, 98 ≤ k → k < 112 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is98 : k = 98
  · subst is98
    exact (chunkFacts_split (chunkLeaf98).1).1
  by_cases is99 : k = 99
  · subst is99
    exact (chunkFacts_split (chunkLeaf99).1).1
  by_cases is100 : k = 100
  · subst is100
    exact (chunkFacts_split (chunkLeaf100).1).1
  by_cases is101 : k = 101
  · subst is101
    exact (chunkFacts_split (chunkLeaf101).1).1
  by_cases is102 : k = 102
  · subst is102
    exact (chunkFacts_split (chunkLeaf102).1).1
  by_cases is103 : k = 103
  · subst is103
    exact (chunkFacts_split (chunkLeaf103).1).1
  by_cases is104 : k = 104
  · subst is104
    exact (chunkFacts_split (chunkLeaf104).1).1
  by_cases is105 : k = 105
  · subst is105
    exact (chunkFacts_split (chunkLeaf105).1).1
  by_cases is106 : k = 106
  · subst is106
    exact (chunkFacts_split (chunkLeaf106).1).1
  by_cases is107 : k = 107
  · subst is107
    exact (chunkFacts_split (chunkLeaf107).1).1
  by_cases is108 : k = 108
  · subst is108
    exact (chunkFacts_split (chunkLeaf108).1).1
  by_cases is109 : k = 109
  · subst is109
    exact (chunkFacts_split (chunkLeaf109).1).1
  by_cases is110 : k = 110
  · subst is110
    exact (chunkFacts_split (chunkLeaf110).1).1
  by_cases is111 : k = 111
  · subst is111
    exact (chunkFacts_split (chunkLeaf111).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 98 ≤ k → k < 112 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is98 : k = 98
  · subst is98
    exact (chunkFacts_split (chunkLeaf98).1).2.1
  by_cases is99 : k = 99
  · subst is99
    exact (chunkFacts_split (chunkLeaf99).1).2.1
  by_cases is100 : k = 100
  · subst is100
    exact (chunkFacts_split (chunkLeaf100).1).2.1
  by_cases is101 : k = 101
  · subst is101
    exact (chunkFacts_split (chunkLeaf101).1).2.1
  by_cases is102 : k = 102
  · subst is102
    exact (chunkFacts_split (chunkLeaf102).1).2.1
  by_cases is103 : k = 103
  · subst is103
    exact (chunkFacts_split (chunkLeaf103).1).2.1
  by_cases is104 : k = 104
  · subst is104
    exact (chunkFacts_split (chunkLeaf104).1).2.1
  by_cases is105 : k = 105
  · subst is105
    exact (chunkFacts_split (chunkLeaf105).1).2.1
  by_cases is106 : k = 106
  · subst is106
    exact (chunkFacts_split (chunkLeaf106).1).2.1
  by_cases is107 : k = 107
  · subst is107
    exact (chunkFacts_split (chunkLeaf107).1).2.1
  by_cases is108 : k = 108
  · subst is108
    exact (chunkFacts_split (chunkLeaf108).1).2.1
  by_cases is109 : k = 109
  · subst is109
    exact (chunkFacts_split (chunkLeaf109).1).2.1
  by_cases is110 : k = 110
  · subst is110
    exact (chunkFacts_split (chunkLeaf110).1).2.1
  by_cases is111 : k = 111
  · subst is111
    exact (chunkFacts_split (chunkLeaf111).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 98 ≤ k → k < 112 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is98 : k = 98
  · subst is98
    exact (chunkFacts_split (chunkLeaf98).1).2.2.1
  by_cases is99 : k = 99
  · subst is99
    exact (chunkFacts_split (chunkLeaf99).1).2.2.1
  by_cases is100 : k = 100
  · subst is100
    exact (chunkFacts_split (chunkLeaf100).1).2.2.1
  by_cases is101 : k = 101
  · subst is101
    exact (chunkFacts_split (chunkLeaf101).1).2.2.1
  by_cases is102 : k = 102
  · subst is102
    exact (chunkFacts_split (chunkLeaf102).1).2.2.1
  by_cases is103 : k = 103
  · subst is103
    exact (chunkFacts_split (chunkLeaf103).1).2.2.1
  by_cases is104 : k = 104
  · subst is104
    exact (chunkFacts_split (chunkLeaf104).1).2.2.1
  by_cases is105 : k = 105
  · subst is105
    exact (chunkFacts_split (chunkLeaf105).1).2.2.1
  by_cases is106 : k = 106
  · subst is106
    exact (chunkFacts_split (chunkLeaf106).1).2.2.1
  by_cases is107 : k = 107
  · subst is107
    exact (chunkFacts_split (chunkLeaf107).1).2.2.1
  by_cases is108 : k = 108
  · subst is108
    exact (chunkFacts_split (chunkLeaf108).1).2.2.1
  by_cases is109 : k = 109
  · subst is109
    exact (chunkFacts_split (chunkLeaf109).1).2.2.1
  by_cases is110 : k = 110
  · subst is110
    exact (chunkFacts_split (chunkLeaf110).1).2.2.1
  by_cases is111 : k = 111
  · subst is111
    exact (chunkFacts_split (chunkLeaf111).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7
