import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf10

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf46 :
    chunkFacts (rowsChunk wire 46) 3014656 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf47 :
    chunkFacts (rowsChunk wire 47) 3080192 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf48 :
    chunkFacts (rowsChunk wire 48) 3145728 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf49 :
    chunkFacts (rowsChunk wire 49) 3211264 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf50 :
    chunkFacts (rowsChunk wire 50) 3276800 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf51 :
    chunkFacts (rowsChunk wire 51) 3342336 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf52 :
    chunkFacts (rowsChunk wire 52) 3407872 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf53 :
    chunkFacts (rowsChunk wire 53) 3473408 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf54 :
    chunkFacts (rowsChunk wire 54) 3538944 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf55 :
    chunkFacts (rowsChunk wire 55) 3604480 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf56 :
    chunkFacts (rowsChunk wire 56) 3670016 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 46 ≤ k → k < 57 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is46 : k = 46
  · subst is46
    exact (chunkFacts_split chunkLeaf46).1
  by_cases is47 : k = 47
  · subst is47
    exact (chunkFacts_split chunkLeaf47).1
  by_cases is48 : k = 48
  · subst is48
    exact (chunkFacts_split chunkLeaf48).1
  by_cases is49 : k = 49
  · subst is49
    exact (chunkFacts_split chunkLeaf49).1
  by_cases is50 : k = 50
  · subst is50
    exact (chunkFacts_split chunkLeaf50).1
  by_cases is51 : k = 51
  · subst is51
    exact (chunkFacts_split chunkLeaf51).1
  by_cases is52 : k = 52
  · subst is52
    exact (chunkFacts_split chunkLeaf52).1
  by_cases is53 : k = 53
  · subst is53
    exact (chunkFacts_split chunkLeaf53).1
  by_cases is54 : k = 54
  · subst is54
    exact (chunkFacts_split chunkLeaf54).1
  by_cases is55 : k = 55
  · subst is55
    exact (chunkFacts_split chunkLeaf55).1
  by_cases is56 : k = 56
  · subst is56
    exact (chunkFacts_split chunkLeaf56).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 46 ≤ k → k < 57 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is46 : k = 46
  · subst is46
    exact (chunkFacts_split chunkLeaf46).2.1
  by_cases is47 : k = 47
  · subst is47
    exact (chunkFacts_split chunkLeaf47).2.1
  by_cases is48 : k = 48
  · subst is48
    exact (chunkFacts_split chunkLeaf48).2.1
  by_cases is49 : k = 49
  · subst is49
    exact (chunkFacts_split chunkLeaf49).2.1
  by_cases is50 : k = 50
  · subst is50
    exact (chunkFacts_split chunkLeaf50).2.1
  by_cases is51 : k = 51
  · subst is51
    exact (chunkFacts_split chunkLeaf51).2.1
  by_cases is52 : k = 52
  · subst is52
    exact (chunkFacts_split chunkLeaf52).2.1
  by_cases is53 : k = 53
  · subst is53
    exact (chunkFacts_split chunkLeaf53).2.1
  by_cases is54 : k = 54
  · subst is54
    exact (chunkFacts_split chunkLeaf54).2.1
  by_cases is55 : k = 55
  · subst is55
    exact (chunkFacts_split chunkLeaf55).2.1
  by_cases is56 : k = 56
  · subst is56
    exact (chunkFacts_split chunkLeaf56).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 46 ≤ k → k < 57 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is46 : k = 46
  · subst is46
    exact (chunkFacts_split chunkLeaf46).2.2.1
  by_cases is47 : k = 47
  · subst is47
    exact (chunkFacts_split chunkLeaf47).2.2.1
  by_cases is48 : k = 48
  · subst is48
    exact (chunkFacts_split chunkLeaf48).2.2.1
  by_cases is49 : k = 49
  · subst is49
    exact (chunkFacts_split chunkLeaf49).2.2.1
  by_cases is50 : k = 50
  · subst is50
    exact (chunkFacts_split chunkLeaf50).2.2.1
  by_cases is51 : k = 51
  · subst is51
    exact (chunkFacts_split chunkLeaf51).2.2.1
  by_cases is52 : k = 52
  · subst is52
    exact (chunkFacts_split chunkLeaf52).2.2.1
  by_cases is53 : k = 53
  · subst is53
    exact (chunkFacts_split chunkLeaf53).2.2.1
  by_cases is54 : k = 54
  · subst is54
    exact (chunkFacts_split chunkLeaf54).2.2.1
  by_cases is55 : k = 55
  · subst is55
    exact (chunkFacts_split chunkLeaf55).2.2.1
  by_cases is56 : k = 56
  · subst is56
    exact (chunkFacts_split chunkLeaf56).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf10
