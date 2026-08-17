import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf140 :
    (chunkFacts (rowsChunk wire 140) 35840 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 140 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk140) := by
  native_decide

theorem chunkLeaf141 :
    (chunkFacts (rowsChunk wire 141) 36096 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 141 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk141) := by
  native_decide

theorem chunkLeaf142 :
    (chunkFacts (rowsChunk wire 142) 36352 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 142 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk142) := by
  native_decide

theorem chunkLeaf143 :
    (chunkFacts (rowsChunk wire 143) 36608 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 143 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk143) := by
  native_decide

theorem chunkLeaf144 :
    (chunkFacts (rowsChunk wire 144) 36864 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 144 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk144) := by
  native_decide

theorem chunkLeaf145 :
    (chunkFacts (rowsChunk wire 145) 37120 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 145 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk145) := by
  native_decide

theorem chunkLeaf146 :
    (chunkFacts (rowsChunk wire 146) 37376 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 146 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk146) := by
  native_decide

theorem chunkLeaf147 :
    (chunkFacts (rowsChunk wire 147) 37632 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 147 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk147) := by
  native_decide

theorem chunkLeaf148 :
    (chunkFacts (rowsChunk wire 148) 37888 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 148 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk148) := by
  native_decide

theorem chunkLeaf149 :
    (chunkFacts (rowsChunk wire 149) 38144 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 149 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk149) := by
  native_decide

theorem chunkLeaf150 :
    (chunkFacts (rowsChunk wire 150) 38400 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 150 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk150) := by
  native_decide

theorem chunkLeaf151 :
    (chunkFacts (rowsChunk wire 151) 38656 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 151 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk151) := by
  native_decide

theorem chunkLeaf152 :
    (chunkFacts (rowsChunk wire 152) 38912 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 152 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk152) := by
  native_decide

theorem chunkLeaf153 :
    (chunkFacts (rowsChunk wire 153) 39168 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 153 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk153) := by
  native_decide

theorem censusGroup :
    ∀ k, 140 ≤ k → k < 154 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is140 : k = 140
  · subst is140
    exact (chunkFacts_split (chunkLeaf140).1).1
  by_cases is141 : k = 141
  · subst is141
    exact (chunkFacts_split (chunkLeaf141).1).1
  by_cases is142 : k = 142
  · subst is142
    exact (chunkFacts_split (chunkLeaf142).1).1
  by_cases is143 : k = 143
  · subst is143
    exact (chunkFacts_split (chunkLeaf143).1).1
  by_cases is144 : k = 144
  · subst is144
    exact (chunkFacts_split (chunkLeaf144).1).1
  by_cases is145 : k = 145
  · subst is145
    exact (chunkFacts_split (chunkLeaf145).1).1
  by_cases is146 : k = 146
  · subst is146
    exact (chunkFacts_split (chunkLeaf146).1).1
  by_cases is147 : k = 147
  · subst is147
    exact (chunkFacts_split (chunkLeaf147).1).1
  by_cases is148 : k = 148
  · subst is148
    exact (chunkFacts_split (chunkLeaf148).1).1
  by_cases is149 : k = 149
  · subst is149
    exact (chunkFacts_split (chunkLeaf149).1).1
  by_cases is150 : k = 150
  · subst is150
    exact (chunkFacts_split (chunkLeaf150).1).1
  by_cases is151 : k = 151
  · subst is151
    exact (chunkFacts_split (chunkLeaf151).1).1
  by_cases is152 : k = 152
  · subst is152
    exact (chunkFacts_split (chunkLeaf152).1).1
  by_cases is153 : k = 153
  · subst is153
    exact (chunkFacts_split (chunkLeaf153).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 140 ≤ k → k < 154 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is140 : k = 140
  · subst is140
    exact (chunkFacts_split (chunkLeaf140).1).2.1
  by_cases is141 : k = 141
  · subst is141
    exact (chunkFacts_split (chunkLeaf141).1).2.1
  by_cases is142 : k = 142
  · subst is142
    exact (chunkFacts_split (chunkLeaf142).1).2.1
  by_cases is143 : k = 143
  · subst is143
    exact (chunkFacts_split (chunkLeaf143).1).2.1
  by_cases is144 : k = 144
  · subst is144
    exact (chunkFacts_split (chunkLeaf144).1).2.1
  by_cases is145 : k = 145
  · subst is145
    exact (chunkFacts_split (chunkLeaf145).1).2.1
  by_cases is146 : k = 146
  · subst is146
    exact (chunkFacts_split (chunkLeaf146).1).2.1
  by_cases is147 : k = 147
  · subst is147
    exact (chunkFacts_split (chunkLeaf147).1).2.1
  by_cases is148 : k = 148
  · subst is148
    exact (chunkFacts_split (chunkLeaf148).1).2.1
  by_cases is149 : k = 149
  · subst is149
    exact (chunkFacts_split (chunkLeaf149).1).2.1
  by_cases is150 : k = 150
  · subst is150
    exact (chunkFacts_split (chunkLeaf150).1).2.1
  by_cases is151 : k = 151
  · subst is151
    exact (chunkFacts_split (chunkLeaf151).1).2.1
  by_cases is152 : k = 152
  · subst is152
    exact (chunkFacts_split (chunkLeaf152).1).2.1
  by_cases is153 : k = 153
  · subst is153
    exact (chunkFacts_split (chunkLeaf153).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 140 ≤ k → k < 154 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is140 : k = 140
  · subst is140
    exact (chunkFacts_split (chunkLeaf140).1).2.2.1
  by_cases is141 : k = 141
  · subst is141
    exact (chunkFacts_split (chunkLeaf141).1).2.2.1
  by_cases is142 : k = 142
  · subst is142
    exact (chunkFacts_split (chunkLeaf142).1).2.2.1
  by_cases is143 : k = 143
  · subst is143
    exact (chunkFacts_split (chunkLeaf143).1).2.2.1
  by_cases is144 : k = 144
  · subst is144
    exact (chunkFacts_split (chunkLeaf144).1).2.2.1
  by_cases is145 : k = 145
  · subst is145
    exact (chunkFacts_split (chunkLeaf145).1).2.2.1
  by_cases is146 : k = 146
  · subst is146
    exact (chunkFacts_split (chunkLeaf146).1).2.2.1
  by_cases is147 : k = 147
  · subst is147
    exact (chunkFacts_split (chunkLeaf147).1).2.2.1
  by_cases is148 : k = 148
  · subst is148
    exact (chunkFacts_split (chunkLeaf148).1).2.2.1
  by_cases is149 : k = 149
  · subst is149
    exact (chunkFacts_split (chunkLeaf149).1).2.2.1
  by_cases is150 : k = 150
  · subst is150
    exact (chunkFacts_split (chunkLeaf150).1).2.2.1
  by_cases is151 : k = 151
  · subst is151
    exact (chunkFacts_split (chunkLeaf151).1).2.2.1
  by_cases is152 : k = 152
  · subst is152
    exact (chunkFacts_split (chunkLeaf152).1).2.2.1
  by_cases is153 : k = 153
  · subst is153
    exact (chunkFacts_split (chunkLeaf153).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10
