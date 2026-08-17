import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf29

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf159 :
    chunkFacts (rowsChunk wire 159) 10420224 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf160 :
    chunkFacts (rowsChunk wire 160) 10485760 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf161 :
    chunkFacts (rowsChunk wire 161) 10551296 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf162 :
    chunkFacts (rowsChunk wire 162) 10616832 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf163 :
    chunkFacts (rowsChunk wire 163) 10682368 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf164 :
    chunkFacts (rowsChunk wire 164) 10747904 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf165 :
    chunkFacts (rowsChunk wire 165) 10813440 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf166 :
    chunkFacts (rowsChunk wire 166) 10878976 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf167 :
    chunkFacts (rowsChunk wire 167) 10944512 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf168 :
    chunkFacts (rowsChunk wire 168) 11010048 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 159 ≤ k → k < 169 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is159 : k = 159
  · subst is159
    exact (chunkFacts_split chunkLeaf159).1
  by_cases is160 : k = 160
  · subst is160
    exact (chunkFacts_split chunkLeaf160).1
  by_cases is161 : k = 161
  · subst is161
    exact (chunkFacts_split chunkLeaf161).1
  by_cases is162 : k = 162
  · subst is162
    exact (chunkFacts_split chunkLeaf162).1
  by_cases is163 : k = 163
  · subst is163
    exact (chunkFacts_split chunkLeaf163).1
  by_cases is164 : k = 164
  · subst is164
    exact (chunkFacts_split chunkLeaf164).1
  by_cases is165 : k = 165
  · subst is165
    exact (chunkFacts_split chunkLeaf165).1
  by_cases is166 : k = 166
  · subst is166
    exact (chunkFacts_split chunkLeaf166).1
  by_cases is167 : k = 167
  · subst is167
    exact (chunkFacts_split chunkLeaf167).1
  by_cases is168 : k = 168
  · subst is168
    exact (chunkFacts_split chunkLeaf168).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 159 ≤ k → k < 169 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is159 : k = 159
  · subst is159
    exact (chunkFacts_split chunkLeaf159).2.1
  by_cases is160 : k = 160
  · subst is160
    exact (chunkFacts_split chunkLeaf160).2.1
  by_cases is161 : k = 161
  · subst is161
    exact (chunkFacts_split chunkLeaf161).2.1
  by_cases is162 : k = 162
  · subst is162
    exact (chunkFacts_split chunkLeaf162).2.1
  by_cases is163 : k = 163
  · subst is163
    exact (chunkFacts_split chunkLeaf163).2.1
  by_cases is164 : k = 164
  · subst is164
    exact (chunkFacts_split chunkLeaf164).2.1
  by_cases is165 : k = 165
  · subst is165
    exact (chunkFacts_split chunkLeaf165).2.1
  by_cases is166 : k = 166
  · subst is166
    exact (chunkFacts_split chunkLeaf166).2.1
  by_cases is167 : k = 167
  · subst is167
    exact (chunkFacts_split chunkLeaf167).2.1
  by_cases is168 : k = 168
  · subst is168
    exact (chunkFacts_split chunkLeaf168).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 159 ≤ k → k < 169 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is159 : k = 159
  · subst is159
    exact (chunkFacts_split chunkLeaf159).2.2.1
  by_cases is160 : k = 160
  · subst is160
    exact (chunkFacts_split chunkLeaf160).2.2.1
  by_cases is161 : k = 161
  · subst is161
    exact (chunkFacts_split chunkLeaf161).2.2.1
  by_cases is162 : k = 162
  · subst is162
    exact (chunkFacts_split chunkLeaf162).2.2.1
  by_cases is163 : k = 163
  · subst is163
    exact (chunkFacts_split chunkLeaf163).2.2.1
  by_cases is164 : k = 164
  · subst is164
    exact (chunkFacts_split chunkLeaf164).2.2.1
  by_cases is165 : k = 165
  · subst is165
    exact (chunkFacts_split chunkLeaf165).2.2.1
  by_cases is166 : k = 166
  · subst is166
    exact (chunkFacts_split chunkLeaf166).2.2.1
  by_cases is167 : k = 167
  · subst is167
    exact (chunkFacts_split chunkLeaf167).2.2.1
  by_cases is168 : k = 168
  · subst is168
    exact (chunkFacts_split chunkLeaf168).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf29
