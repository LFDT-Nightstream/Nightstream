import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf8

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf18 :
    chunkFacts (rowsChunk wire 18) 1179648 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf19 :
    chunkFacts (rowsChunk wire 19) 1245184 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf20 :
    chunkFacts (rowsChunk wire 20) 1310720 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf21 :
    chunkFacts (rowsChunk wire 21) 1376256 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf22 :
    chunkFacts (rowsChunk wire 22) 1441792 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf23 :
    chunkFacts (rowsChunk wire 23) 1507328 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf24 :
    chunkFacts (rowsChunk wire 24) 1572864 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf25 :
    chunkFacts (rowsChunk wire 25) 1638400 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf26 :
    chunkFacts (rowsChunk wire 26) 1703936 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf27 :
    chunkFacts (rowsChunk wire 27) 1769472 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf28 :
    chunkFacts (rowsChunk wire 28) 1835008 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf29 :
    chunkFacts (rowsChunk wire 29) 1900544 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf30 :
    chunkFacts (rowsChunk wire 30) 1966080 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf31 :
    chunkFacts (rowsChunk wire 31) 2031616 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 18 ≤ k → k < 32 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is18 : k = 18
  · subst is18
    exact (chunkFacts_split chunkLeaf18).1
  by_cases is19 : k = 19
  · subst is19
    exact (chunkFacts_split chunkLeaf19).1
  by_cases is20 : k = 20
  · subst is20
    exact (chunkFacts_split chunkLeaf20).1
  by_cases is21 : k = 21
  · subst is21
    exact (chunkFacts_split chunkLeaf21).1
  by_cases is22 : k = 22
  · subst is22
    exact (chunkFacts_split chunkLeaf22).1
  by_cases is23 : k = 23
  · subst is23
    exact (chunkFacts_split chunkLeaf23).1
  by_cases is24 : k = 24
  · subst is24
    exact (chunkFacts_split chunkLeaf24).1
  by_cases is25 : k = 25
  · subst is25
    exact (chunkFacts_split chunkLeaf25).1
  by_cases is26 : k = 26
  · subst is26
    exact (chunkFacts_split chunkLeaf26).1
  by_cases is27 : k = 27
  · subst is27
    exact (chunkFacts_split chunkLeaf27).1
  by_cases is28 : k = 28
  · subst is28
    exact (chunkFacts_split chunkLeaf28).1
  by_cases is29 : k = 29
  · subst is29
    exact (chunkFacts_split chunkLeaf29).1
  by_cases is30 : k = 30
  · subst is30
    exact (chunkFacts_split chunkLeaf30).1
  by_cases is31 : k = 31
  · subst is31
    exact (chunkFacts_split chunkLeaf31).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 18 ≤ k → k < 32 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is18 : k = 18
  · subst is18
    exact (chunkFacts_split chunkLeaf18).2.1
  by_cases is19 : k = 19
  · subst is19
    exact (chunkFacts_split chunkLeaf19).2.1
  by_cases is20 : k = 20
  · subst is20
    exact (chunkFacts_split chunkLeaf20).2.1
  by_cases is21 : k = 21
  · subst is21
    exact (chunkFacts_split chunkLeaf21).2.1
  by_cases is22 : k = 22
  · subst is22
    exact (chunkFacts_split chunkLeaf22).2.1
  by_cases is23 : k = 23
  · subst is23
    exact (chunkFacts_split chunkLeaf23).2.1
  by_cases is24 : k = 24
  · subst is24
    exact (chunkFacts_split chunkLeaf24).2.1
  by_cases is25 : k = 25
  · subst is25
    exact (chunkFacts_split chunkLeaf25).2.1
  by_cases is26 : k = 26
  · subst is26
    exact (chunkFacts_split chunkLeaf26).2.1
  by_cases is27 : k = 27
  · subst is27
    exact (chunkFacts_split chunkLeaf27).2.1
  by_cases is28 : k = 28
  · subst is28
    exact (chunkFacts_split chunkLeaf28).2.1
  by_cases is29 : k = 29
  · subst is29
    exact (chunkFacts_split chunkLeaf29).2.1
  by_cases is30 : k = 30
  · subst is30
    exact (chunkFacts_split chunkLeaf30).2.1
  by_cases is31 : k = 31
  · subst is31
    exact (chunkFacts_split chunkLeaf31).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 18 ≤ k → k < 32 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is18 : k = 18
  · subst is18
    exact (chunkFacts_split chunkLeaf18).2.2.1
  by_cases is19 : k = 19
  · subst is19
    exact (chunkFacts_split chunkLeaf19).2.2.1
  by_cases is20 : k = 20
  · subst is20
    exact (chunkFacts_split chunkLeaf20).2.2.1
  by_cases is21 : k = 21
  · subst is21
    exact (chunkFacts_split chunkLeaf21).2.2.1
  by_cases is22 : k = 22
  · subst is22
    exact (chunkFacts_split chunkLeaf22).2.2.1
  by_cases is23 : k = 23
  · subst is23
    exact (chunkFacts_split chunkLeaf23).2.2.1
  by_cases is24 : k = 24
  · subst is24
    exact (chunkFacts_split chunkLeaf24).2.2.1
  by_cases is25 : k = 25
  · subst is25
    exact (chunkFacts_split chunkLeaf25).2.2.1
  by_cases is26 : k = 26
  · subst is26
    exact (chunkFacts_split chunkLeaf26).2.2.1
  by_cases is27 : k = 27
  · subst is27
    exact (chunkFacts_split chunkLeaf27).2.2.1
  by_cases is28 : k = 28
  · subst is28
    exact (chunkFacts_split chunkLeaf28).2.2.1
  by_cases is29 : k = 29
  · subst is29
    exact (chunkFacts_split chunkLeaf29).2.2.1
  by_cases is30 : k = 30
  · subst is30
    exact (chunkFacts_split chunkLeaf30).2.2.1
  by_cases is31 : k = 31
  · subst is31
    exact (chunkFacts_split chunkLeaf31).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf8
