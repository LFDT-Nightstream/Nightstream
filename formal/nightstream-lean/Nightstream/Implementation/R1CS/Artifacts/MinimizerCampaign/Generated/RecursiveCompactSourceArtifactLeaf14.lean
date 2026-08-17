import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf14

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf86 :
    chunkFacts (rowsChunk wire 86) 5636096 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf87 :
    chunkFacts (rowsChunk wire 87) 5701632 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf88 :
    chunkFacts (rowsChunk wire 88) 5767168 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf89 :
    chunkFacts (rowsChunk wire 89) 5832704 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf90 :
    chunkFacts (rowsChunk wire 90) 5898240 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf91 :
    chunkFacts (rowsChunk wire 91) 5963776 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf92 :
    chunkFacts (rowsChunk wire 92) 6029312 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf93 :
    chunkFacts (rowsChunk wire 93) 6094848 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf94 :
    chunkFacts (rowsChunk wire 94) 6160384 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf95 :
    chunkFacts (rowsChunk wire 95) 6225920 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf96 :
    chunkFacts (rowsChunk wire 96) 6291456 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf97 :
    chunkFacts (rowsChunk wire 97) 6356992 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf98 :
    chunkFacts (rowsChunk wire 98) 6422528 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf99 :
    chunkFacts (rowsChunk wire 99) 6488064 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 86 ≤ k → k < 100 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is86 : k = 86
  · subst is86
    exact (chunkFacts_split chunkLeaf86).1
  by_cases is87 : k = 87
  · subst is87
    exact (chunkFacts_split chunkLeaf87).1
  by_cases is88 : k = 88
  · subst is88
    exact (chunkFacts_split chunkLeaf88).1
  by_cases is89 : k = 89
  · subst is89
    exact (chunkFacts_split chunkLeaf89).1
  by_cases is90 : k = 90
  · subst is90
    exact (chunkFacts_split chunkLeaf90).1
  by_cases is91 : k = 91
  · subst is91
    exact (chunkFacts_split chunkLeaf91).1
  by_cases is92 : k = 92
  · subst is92
    exact (chunkFacts_split chunkLeaf92).1
  by_cases is93 : k = 93
  · subst is93
    exact (chunkFacts_split chunkLeaf93).1
  by_cases is94 : k = 94
  · subst is94
    exact (chunkFacts_split chunkLeaf94).1
  by_cases is95 : k = 95
  · subst is95
    exact (chunkFacts_split chunkLeaf95).1
  by_cases is96 : k = 96
  · subst is96
    exact (chunkFacts_split chunkLeaf96).1
  by_cases is97 : k = 97
  · subst is97
    exact (chunkFacts_split chunkLeaf97).1
  by_cases is98 : k = 98
  · subst is98
    exact (chunkFacts_split chunkLeaf98).1
  by_cases is99 : k = 99
  · subst is99
    exact (chunkFacts_split chunkLeaf99).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 86 ≤ k → k < 100 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is86 : k = 86
  · subst is86
    exact (chunkFacts_split chunkLeaf86).2.1
  by_cases is87 : k = 87
  · subst is87
    exact (chunkFacts_split chunkLeaf87).2.1
  by_cases is88 : k = 88
  · subst is88
    exact (chunkFacts_split chunkLeaf88).2.1
  by_cases is89 : k = 89
  · subst is89
    exact (chunkFacts_split chunkLeaf89).2.1
  by_cases is90 : k = 90
  · subst is90
    exact (chunkFacts_split chunkLeaf90).2.1
  by_cases is91 : k = 91
  · subst is91
    exact (chunkFacts_split chunkLeaf91).2.1
  by_cases is92 : k = 92
  · subst is92
    exact (chunkFacts_split chunkLeaf92).2.1
  by_cases is93 : k = 93
  · subst is93
    exact (chunkFacts_split chunkLeaf93).2.1
  by_cases is94 : k = 94
  · subst is94
    exact (chunkFacts_split chunkLeaf94).2.1
  by_cases is95 : k = 95
  · subst is95
    exact (chunkFacts_split chunkLeaf95).2.1
  by_cases is96 : k = 96
  · subst is96
    exact (chunkFacts_split chunkLeaf96).2.1
  by_cases is97 : k = 97
  · subst is97
    exact (chunkFacts_split chunkLeaf97).2.1
  by_cases is98 : k = 98
  · subst is98
    exact (chunkFacts_split chunkLeaf98).2.1
  by_cases is99 : k = 99
  · subst is99
    exact (chunkFacts_split chunkLeaf99).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 86 ≤ k → k < 100 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is86 : k = 86
  · subst is86
    exact (chunkFacts_split chunkLeaf86).2.2.1
  by_cases is87 : k = 87
  · subst is87
    exact (chunkFacts_split chunkLeaf87).2.2.1
  by_cases is88 : k = 88
  · subst is88
    exact (chunkFacts_split chunkLeaf88).2.2.1
  by_cases is89 : k = 89
  · subst is89
    exact (chunkFacts_split chunkLeaf89).2.2.1
  by_cases is90 : k = 90
  · subst is90
    exact (chunkFacts_split chunkLeaf90).2.2.1
  by_cases is91 : k = 91
  · subst is91
    exact (chunkFacts_split chunkLeaf91).2.2.1
  by_cases is92 : k = 92
  · subst is92
    exact (chunkFacts_split chunkLeaf92).2.2.1
  by_cases is93 : k = 93
  · subst is93
    exact (chunkFacts_split chunkLeaf93).2.2.1
  by_cases is94 : k = 94
  · subst is94
    exact (chunkFacts_split chunkLeaf94).2.2.1
  by_cases is95 : k = 95
  · subst is95
    exact (chunkFacts_split chunkLeaf95).2.2.1
  by_cases is96 : k = 96
  · subst is96
    exact (chunkFacts_split chunkLeaf96).2.2.1
  by_cases is97 : k = 97
  · subst is97
    exact (chunkFacts_split chunkLeaf97).2.2.1
  by_cases is98 : k = 98
  · subst is98
    exact (chunkFacts_split chunkLeaf98).2.2.1
  by_cases is99 : k = 99
  · subst is99
    exact (chunkFacts_split chunkLeaf99).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf14
