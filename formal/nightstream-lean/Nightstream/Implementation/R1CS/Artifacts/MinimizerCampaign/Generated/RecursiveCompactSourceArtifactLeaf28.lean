import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf28

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf145 :
    chunkFacts (rowsChunk wire 145) 9502720 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf146 :
    chunkFacts (rowsChunk wire 146) 9568256 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf147 :
    chunkFacts (rowsChunk wire 147) 9633792 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf148 :
    chunkFacts (rowsChunk wire 148) 9699328 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf149 :
    chunkFacts (rowsChunk wire 149) 9764864 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf150 :
    chunkFacts (rowsChunk wire 150) 9830400 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf151 :
    chunkFacts (rowsChunk wire 151) 9895936 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf152 :
    chunkFacts (rowsChunk wire 152) 9961472 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf153 :
    chunkFacts (rowsChunk wire 153) 10027008 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf154 :
    chunkFacts (rowsChunk wire 154) 10092544 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf155 :
    chunkFacts (rowsChunk wire 155) 10158080 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf156 :
    chunkFacts (rowsChunk wire 156) 10223616 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf157 :
    chunkFacts (rowsChunk wire 157) 10289152 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf158 :
    chunkFacts (rowsChunk wire 158) 10354688 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 145 ≤ k → k < 159 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is145 : k = 145
  · subst is145
    exact (chunkFacts_split chunkLeaf145).1
  by_cases is146 : k = 146
  · subst is146
    exact (chunkFacts_split chunkLeaf146).1
  by_cases is147 : k = 147
  · subst is147
    exact (chunkFacts_split chunkLeaf147).1
  by_cases is148 : k = 148
  · subst is148
    exact (chunkFacts_split chunkLeaf148).1
  by_cases is149 : k = 149
  · subst is149
    exact (chunkFacts_split chunkLeaf149).1
  by_cases is150 : k = 150
  · subst is150
    exact (chunkFacts_split chunkLeaf150).1
  by_cases is151 : k = 151
  · subst is151
    exact (chunkFacts_split chunkLeaf151).1
  by_cases is152 : k = 152
  · subst is152
    exact (chunkFacts_split chunkLeaf152).1
  by_cases is153 : k = 153
  · subst is153
    exact (chunkFacts_split chunkLeaf153).1
  by_cases is154 : k = 154
  · subst is154
    exact (chunkFacts_split chunkLeaf154).1
  by_cases is155 : k = 155
  · subst is155
    exact (chunkFacts_split chunkLeaf155).1
  by_cases is156 : k = 156
  · subst is156
    exact (chunkFacts_split chunkLeaf156).1
  by_cases is157 : k = 157
  · subst is157
    exact (chunkFacts_split chunkLeaf157).1
  by_cases is158 : k = 158
  · subst is158
    exact (chunkFacts_split chunkLeaf158).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 145 ≤ k → k < 159 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is145 : k = 145
  · subst is145
    exact (chunkFacts_split chunkLeaf145).2.1
  by_cases is146 : k = 146
  · subst is146
    exact (chunkFacts_split chunkLeaf146).2.1
  by_cases is147 : k = 147
  · subst is147
    exact (chunkFacts_split chunkLeaf147).2.1
  by_cases is148 : k = 148
  · subst is148
    exact (chunkFacts_split chunkLeaf148).2.1
  by_cases is149 : k = 149
  · subst is149
    exact (chunkFacts_split chunkLeaf149).2.1
  by_cases is150 : k = 150
  · subst is150
    exact (chunkFacts_split chunkLeaf150).2.1
  by_cases is151 : k = 151
  · subst is151
    exact (chunkFacts_split chunkLeaf151).2.1
  by_cases is152 : k = 152
  · subst is152
    exact (chunkFacts_split chunkLeaf152).2.1
  by_cases is153 : k = 153
  · subst is153
    exact (chunkFacts_split chunkLeaf153).2.1
  by_cases is154 : k = 154
  · subst is154
    exact (chunkFacts_split chunkLeaf154).2.1
  by_cases is155 : k = 155
  · subst is155
    exact (chunkFacts_split chunkLeaf155).2.1
  by_cases is156 : k = 156
  · subst is156
    exact (chunkFacts_split chunkLeaf156).2.1
  by_cases is157 : k = 157
  · subst is157
    exact (chunkFacts_split chunkLeaf157).2.1
  by_cases is158 : k = 158
  · subst is158
    exact (chunkFacts_split chunkLeaf158).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 145 ≤ k → k < 159 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is145 : k = 145
  · subst is145
    exact (chunkFacts_split chunkLeaf145).2.2.1
  by_cases is146 : k = 146
  · subst is146
    exact (chunkFacts_split chunkLeaf146).2.2.1
  by_cases is147 : k = 147
  · subst is147
    exact (chunkFacts_split chunkLeaf147).2.2.1
  by_cases is148 : k = 148
  · subst is148
    exact (chunkFacts_split chunkLeaf148).2.2.1
  by_cases is149 : k = 149
  · subst is149
    exact (chunkFacts_split chunkLeaf149).2.2.1
  by_cases is150 : k = 150
  · subst is150
    exact (chunkFacts_split chunkLeaf150).2.2.1
  by_cases is151 : k = 151
  · subst is151
    exact (chunkFacts_split chunkLeaf151).2.2.1
  by_cases is152 : k = 152
  · subst is152
    exact (chunkFacts_split chunkLeaf152).2.2.1
  by_cases is153 : k = 153
  · subst is153
    exact (chunkFacts_split chunkLeaf153).2.2.1
  by_cases is154 : k = 154
  · subst is154
    exact (chunkFacts_split chunkLeaf154).2.2.1
  by_cases is155 : k = 155
  · subst is155
    exact (chunkFacts_split chunkLeaf155).2.2.1
  by_cases is156 : k = 156
  · subst is156
    exact (chunkFacts_split chunkLeaf156).2.2.1
  by_cases is157 : k = 157
  · subst is157
    exact (chunkFacts_split chunkLeaf157).2.2.1
  by_cases is158 : k = 158
  · subst is158
    exact (chunkFacts_split chunkLeaf158).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf28
