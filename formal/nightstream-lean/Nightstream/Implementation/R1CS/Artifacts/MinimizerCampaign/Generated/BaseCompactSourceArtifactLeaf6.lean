import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf84 :
    (chunkFacts (rowsChunk wire 84) 21504 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 84 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk84) := by
  native_decide

theorem chunkLeaf85 :
    (chunkFacts (rowsChunk wire 85) 21760 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 85 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk85) := by
  native_decide

theorem chunkLeaf86 :
    (chunkFacts (rowsChunk wire 86) 22016 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 86 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk86) := by
  native_decide

theorem chunkLeaf87 :
    (chunkFacts (rowsChunk wire 87) 22272 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 87 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk87) := by
  native_decide

theorem chunkLeaf88 :
    (chunkFacts (rowsChunk wire 88) 22528 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 88 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk88) := by
  native_decide

theorem chunkLeaf89 :
    (chunkFacts (rowsChunk wire 89) 22784 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 89 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk89) := by
  native_decide

theorem chunkLeaf90 :
    (chunkFacts (rowsChunk wire 90) 23040 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 90 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk90) := by
  native_decide

theorem chunkLeaf91 :
    (chunkFacts (rowsChunk wire 91) 23296 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 91 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk91) := by
  native_decide

theorem chunkLeaf92 :
    (chunkFacts (rowsChunk wire 92) 23552 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 92 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk92) := by
  native_decide

theorem chunkLeaf93 :
    (chunkFacts (rowsChunk wire 93) 23808 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 93 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk93) := by
  native_decide

theorem chunkLeaf94 :
    (chunkFacts (rowsChunk wire 94) 24064 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 94 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk94) := by
  native_decide

theorem chunkLeaf95 :
    (chunkFacts (rowsChunk wire 95) 24320 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 95 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk95) := by
  native_decide

theorem chunkLeaf96 :
    (chunkFacts (rowsChunk wire 96) 24576 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 96 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk96) := by
  native_decide

theorem chunkLeaf97 :
    (chunkFacts (rowsChunk wire 97) 24832 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 97 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk97) := by
  native_decide

theorem censusGroup :
    ∀ k, 84 ≤ k → k < 98 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is84 : k = 84
  · subst is84
    exact (chunkFacts_split (chunkLeaf84).1).1
  by_cases is85 : k = 85
  · subst is85
    exact (chunkFacts_split (chunkLeaf85).1).1
  by_cases is86 : k = 86
  · subst is86
    exact (chunkFacts_split (chunkLeaf86).1).1
  by_cases is87 : k = 87
  · subst is87
    exact (chunkFacts_split (chunkLeaf87).1).1
  by_cases is88 : k = 88
  · subst is88
    exact (chunkFacts_split (chunkLeaf88).1).1
  by_cases is89 : k = 89
  · subst is89
    exact (chunkFacts_split (chunkLeaf89).1).1
  by_cases is90 : k = 90
  · subst is90
    exact (chunkFacts_split (chunkLeaf90).1).1
  by_cases is91 : k = 91
  · subst is91
    exact (chunkFacts_split (chunkLeaf91).1).1
  by_cases is92 : k = 92
  · subst is92
    exact (chunkFacts_split (chunkLeaf92).1).1
  by_cases is93 : k = 93
  · subst is93
    exact (chunkFacts_split (chunkLeaf93).1).1
  by_cases is94 : k = 94
  · subst is94
    exact (chunkFacts_split (chunkLeaf94).1).1
  by_cases is95 : k = 95
  · subst is95
    exact (chunkFacts_split (chunkLeaf95).1).1
  by_cases is96 : k = 96
  · subst is96
    exact (chunkFacts_split (chunkLeaf96).1).1
  by_cases is97 : k = 97
  · subst is97
    exact (chunkFacts_split (chunkLeaf97).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 84 ≤ k → k < 98 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is84 : k = 84
  · subst is84
    exact (chunkFacts_split (chunkLeaf84).1).2.1
  by_cases is85 : k = 85
  · subst is85
    exact (chunkFacts_split (chunkLeaf85).1).2.1
  by_cases is86 : k = 86
  · subst is86
    exact (chunkFacts_split (chunkLeaf86).1).2.1
  by_cases is87 : k = 87
  · subst is87
    exact (chunkFacts_split (chunkLeaf87).1).2.1
  by_cases is88 : k = 88
  · subst is88
    exact (chunkFacts_split (chunkLeaf88).1).2.1
  by_cases is89 : k = 89
  · subst is89
    exact (chunkFacts_split (chunkLeaf89).1).2.1
  by_cases is90 : k = 90
  · subst is90
    exact (chunkFacts_split (chunkLeaf90).1).2.1
  by_cases is91 : k = 91
  · subst is91
    exact (chunkFacts_split (chunkLeaf91).1).2.1
  by_cases is92 : k = 92
  · subst is92
    exact (chunkFacts_split (chunkLeaf92).1).2.1
  by_cases is93 : k = 93
  · subst is93
    exact (chunkFacts_split (chunkLeaf93).1).2.1
  by_cases is94 : k = 94
  · subst is94
    exact (chunkFacts_split (chunkLeaf94).1).2.1
  by_cases is95 : k = 95
  · subst is95
    exact (chunkFacts_split (chunkLeaf95).1).2.1
  by_cases is96 : k = 96
  · subst is96
    exact (chunkFacts_split (chunkLeaf96).1).2.1
  by_cases is97 : k = 97
  · subst is97
    exact (chunkFacts_split (chunkLeaf97).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 84 ≤ k → k < 98 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is84 : k = 84
  · subst is84
    exact (chunkFacts_split (chunkLeaf84).1).2.2.1
  by_cases is85 : k = 85
  · subst is85
    exact (chunkFacts_split (chunkLeaf85).1).2.2.1
  by_cases is86 : k = 86
  · subst is86
    exact (chunkFacts_split (chunkLeaf86).1).2.2.1
  by_cases is87 : k = 87
  · subst is87
    exact (chunkFacts_split (chunkLeaf87).1).2.2.1
  by_cases is88 : k = 88
  · subst is88
    exact (chunkFacts_split (chunkLeaf88).1).2.2.1
  by_cases is89 : k = 89
  · subst is89
    exact (chunkFacts_split (chunkLeaf89).1).2.2.1
  by_cases is90 : k = 90
  · subst is90
    exact (chunkFacts_split (chunkLeaf90).1).2.2.1
  by_cases is91 : k = 91
  · subst is91
    exact (chunkFacts_split (chunkLeaf91).1).2.2.1
  by_cases is92 : k = 92
  · subst is92
    exact (chunkFacts_split (chunkLeaf92).1).2.2.1
  by_cases is93 : k = 93
  · subst is93
    exact (chunkFacts_split (chunkLeaf93).1).2.2.1
  by_cases is94 : k = 94
  · subst is94
    exact (chunkFacts_split (chunkLeaf94).1).2.2.1
  by_cases is95 : k = 95
  · subst is95
    exact (chunkFacts_split (chunkLeaf95).1).2.2.1
  by_cases is96 : k = 96
  · subst is96
    exact (chunkFacts_split (chunkLeaf96).1).2.2.1
  by_cases is97 : k = 97
  · subst is97
    exact (chunkFacts_split (chunkLeaf97).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6
