import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf13

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf72 :
    chunkFacts (rowsChunk wire 72) 4718592 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf73 :
    chunkFacts (rowsChunk wire 73) 4784128 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf74 :
    chunkFacts (rowsChunk wire 74) 4849664 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf75 :
    chunkFacts (rowsChunk wire 75) 4915200 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf76 :
    chunkFacts (rowsChunk wire 76) 4980736 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf77 :
    chunkFacts (rowsChunk wire 77) 5046272 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf78 :
    chunkFacts (rowsChunk wire 78) 5111808 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf79 :
    chunkFacts (rowsChunk wire 79) 5177344 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf80 :
    chunkFacts (rowsChunk wire 80) 5242880 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf81 :
    chunkFacts (rowsChunk wire 81) 5308416 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf82 :
    chunkFacts (rowsChunk wire 82) 5373952 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf83 :
    chunkFacts (rowsChunk wire 83) 5439488 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf84 :
    chunkFacts (rowsChunk wire 84) 5505024 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf85 :
    chunkFacts (rowsChunk wire 85) 5570560 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 72 ≤ k → k < 86 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is72 : k = 72
  · subst is72
    exact (chunkFacts_split chunkLeaf72).1
  by_cases is73 : k = 73
  · subst is73
    exact (chunkFacts_split chunkLeaf73).1
  by_cases is74 : k = 74
  · subst is74
    exact (chunkFacts_split chunkLeaf74).1
  by_cases is75 : k = 75
  · subst is75
    exact (chunkFacts_split chunkLeaf75).1
  by_cases is76 : k = 76
  · subst is76
    exact (chunkFacts_split chunkLeaf76).1
  by_cases is77 : k = 77
  · subst is77
    exact (chunkFacts_split chunkLeaf77).1
  by_cases is78 : k = 78
  · subst is78
    exact (chunkFacts_split chunkLeaf78).1
  by_cases is79 : k = 79
  · subst is79
    exact (chunkFacts_split chunkLeaf79).1
  by_cases is80 : k = 80
  · subst is80
    exact (chunkFacts_split chunkLeaf80).1
  by_cases is81 : k = 81
  · subst is81
    exact (chunkFacts_split chunkLeaf81).1
  by_cases is82 : k = 82
  · subst is82
    exact (chunkFacts_split chunkLeaf82).1
  by_cases is83 : k = 83
  · subst is83
    exact (chunkFacts_split chunkLeaf83).1
  by_cases is84 : k = 84
  · subst is84
    exact (chunkFacts_split chunkLeaf84).1
  by_cases is85 : k = 85
  · subst is85
    exact (chunkFacts_split chunkLeaf85).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 72 ≤ k → k < 86 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is72 : k = 72
  · subst is72
    exact (chunkFacts_split chunkLeaf72).2.1
  by_cases is73 : k = 73
  · subst is73
    exact (chunkFacts_split chunkLeaf73).2.1
  by_cases is74 : k = 74
  · subst is74
    exact (chunkFacts_split chunkLeaf74).2.1
  by_cases is75 : k = 75
  · subst is75
    exact (chunkFacts_split chunkLeaf75).2.1
  by_cases is76 : k = 76
  · subst is76
    exact (chunkFacts_split chunkLeaf76).2.1
  by_cases is77 : k = 77
  · subst is77
    exact (chunkFacts_split chunkLeaf77).2.1
  by_cases is78 : k = 78
  · subst is78
    exact (chunkFacts_split chunkLeaf78).2.1
  by_cases is79 : k = 79
  · subst is79
    exact (chunkFacts_split chunkLeaf79).2.1
  by_cases is80 : k = 80
  · subst is80
    exact (chunkFacts_split chunkLeaf80).2.1
  by_cases is81 : k = 81
  · subst is81
    exact (chunkFacts_split chunkLeaf81).2.1
  by_cases is82 : k = 82
  · subst is82
    exact (chunkFacts_split chunkLeaf82).2.1
  by_cases is83 : k = 83
  · subst is83
    exact (chunkFacts_split chunkLeaf83).2.1
  by_cases is84 : k = 84
  · subst is84
    exact (chunkFacts_split chunkLeaf84).2.1
  by_cases is85 : k = 85
  · subst is85
    exact (chunkFacts_split chunkLeaf85).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 72 ≤ k → k < 86 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is72 : k = 72
  · subst is72
    exact (chunkFacts_split chunkLeaf72).2.2.1
  by_cases is73 : k = 73
  · subst is73
    exact (chunkFacts_split chunkLeaf73).2.2.1
  by_cases is74 : k = 74
  · subst is74
    exact (chunkFacts_split chunkLeaf74).2.2.1
  by_cases is75 : k = 75
  · subst is75
    exact (chunkFacts_split chunkLeaf75).2.2.1
  by_cases is76 : k = 76
  · subst is76
    exact (chunkFacts_split chunkLeaf76).2.2.1
  by_cases is77 : k = 77
  · subst is77
    exact (chunkFacts_split chunkLeaf77).2.2.1
  by_cases is78 : k = 78
  · subst is78
    exact (chunkFacts_split chunkLeaf78).2.2.1
  by_cases is79 : k = 79
  · subst is79
    exact (chunkFacts_split chunkLeaf79).2.2.1
  by_cases is80 : k = 80
  · subst is80
    exact (chunkFacts_split chunkLeaf80).2.2.1
  by_cases is81 : k = 81
  · subst is81
    exact (chunkFacts_split chunkLeaf81).2.2.1
  by_cases is82 : k = 82
  · subst is82
    exact (chunkFacts_split chunkLeaf82).2.2.1
  by_cases is83 : k = 83
  · subst is83
    exact (chunkFacts_split chunkLeaf83).2.2.1
  by_cases is84 : k = 84
  · subst is84
    exact (chunkFacts_split chunkLeaf84).2.2.1
  by_cases is85 : k = 85
  · subst is85
    exact (chunkFacts_split chunkLeaf85).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf13
