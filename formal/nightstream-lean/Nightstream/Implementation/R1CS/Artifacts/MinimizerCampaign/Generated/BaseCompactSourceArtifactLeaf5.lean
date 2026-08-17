import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf70 :
    (chunkFacts (rowsChunk wire 70) 17920 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 70 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk70) := by
  native_decide

theorem chunkLeaf71 :
    (chunkFacts (rowsChunk wire 71) 18176 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 71 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk71) := by
  native_decide

theorem chunkLeaf72 :
    (chunkFacts (rowsChunk wire 72) 18432 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 72 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk72) := by
  native_decide

theorem chunkLeaf73 :
    (chunkFacts (rowsChunk wire 73) 18688 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 73 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk73) := by
  native_decide

theorem chunkLeaf74 :
    (chunkFacts (rowsChunk wire 74) 18944 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 74 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk74) := by
  native_decide

theorem chunkLeaf75 :
    (chunkFacts (rowsChunk wire 75) 19200 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 75 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk75) := by
  native_decide

theorem chunkLeaf76 :
    (chunkFacts (rowsChunk wire 76) 19456 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 76 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk76) := by
  native_decide

theorem chunkLeaf77 :
    (chunkFacts (rowsChunk wire 77) 19712 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 77 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk77) := by
  native_decide

theorem chunkLeaf78 :
    (chunkFacts (rowsChunk wire 78) 19968 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 78 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk78) := by
  native_decide

theorem chunkLeaf79 :
    (chunkFacts (rowsChunk wire 79) 20224 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 79 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk79) := by
  native_decide

theorem chunkLeaf80 :
    (chunkFacts (rowsChunk wire 80) 20480 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 80 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk80) := by
  native_decide

theorem chunkLeaf81 :
    (chunkFacts (rowsChunk wire 81) 20736 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 81 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk81) := by
  native_decide

theorem chunkLeaf82 :
    (chunkFacts (rowsChunk wire 82) 20992 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 82 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk82) := by
  native_decide

theorem chunkLeaf83 :
    (chunkFacts (rowsChunk wire 83) 21248 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 83 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk83) := by
  native_decide

theorem censusGroup :
    ∀ k, 70 ≤ k → k < 84 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is70 : k = 70
  · subst is70
    exact (chunkFacts_split (chunkLeaf70).1).1
  by_cases is71 : k = 71
  · subst is71
    exact (chunkFacts_split (chunkLeaf71).1).1
  by_cases is72 : k = 72
  · subst is72
    exact (chunkFacts_split (chunkLeaf72).1).1
  by_cases is73 : k = 73
  · subst is73
    exact (chunkFacts_split (chunkLeaf73).1).1
  by_cases is74 : k = 74
  · subst is74
    exact (chunkFacts_split (chunkLeaf74).1).1
  by_cases is75 : k = 75
  · subst is75
    exact (chunkFacts_split (chunkLeaf75).1).1
  by_cases is76 : k = 76
  · subst is76
    exact (chunkFacts_split (chunkLeaf76).1).1
  by_cases is77 : k = 77
  · subst is77
    exact (chunkFacts_split (chunkLeaf77).1).1
  by_cases is78 : k = 78
  · subst is78
    exact (chunkFacts_split (chunkLeaf78).1).1
  by_cases is79 : k = 79
  · subst is79
    exact (chunkFacts_split (chunkLeaf79).1).1
  by_cases is80 : k = 80
  · subst is80
    exact (chunkFacts_split (chunkLeaf80).1).1
  by_cases is81 : k = 81
  · subst is81
    exact (chunkFacts_split (chunkLeaf81).1).1
  by_cases is82 : k = 82
  · subst is82
    exact (chunkFacts_split (chunkLeaf82).1).1
  by_cases is83 : k = 83
  · subst is83
    exact (chunkFacts_split (chunkLeaf83).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 70 ≤ k → k < 84 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is70 : k = 70
  · subst is70
    exact (chunkFacts_split (chunkLeaf70).1).2.1
  by_cases is71 : k = 71
  · subst is71
    exact (chunkFacts_split (chunkLeaf71).1).2.1
  by_cases is72 : k = 72
  · subst is72
    exact (chunkFacts_split (chunkLeaf72).1).2.1
  by_cases is73 : k = 73
  · subst is73
    exact (chunkFacts_split (chunkLeaf73).1).2.1
  by_cases is74 : k = 74
  · subst is74
    exact (chunkFacts_split (chunkLeaf74).1).2.1
  by_cases is75 : k = 75
  · subst is75
    exact (chunkFacts_split (chunkLeaf75).1).2.1
  by_cases is76 : k = 76
  · subst is76
    exact (chunkFacts_split (chunkLeaf76).1).2.1
  by_cases is77 : k = 77
  · subst is77
    exact (chunkFacts_split (chunkLeaf77).1).2.1
  by_cases is78 : k = 78
  · subst is78
    exact (chunkFacts_split (chunkLeaf78).1).2.1
  by_cases is79 : k = 79
  · subst is79
    exact (chunkFacts_split (chunkLeaf79).1).2.1
  by_cases is80 : k = 80
  · subst is80
    exact (chunkFacts_split (chunkLeaf80).1).2.1
  by_cases is81 : k = 81
  · subst is81
    exact (chunkFacts_split (chunkLeaf81).1).2.1
  by_cases is82 : k = 82
  · subst is82
    exact (chunkFacts_split (chunkLeaf82).1).2.1
  by_cases is83 : k = 83
  · subst is83
    exact (chunkFacts_split (chunkLeaf83).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 70 ≤ k → k < 84 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is70 : k = 70
  · subst is70
    exact (chunkFacts_split (chunkLeaf70).1).2.2.1
  by_cases is71 : k = 71
  · subst is71
    exact (chunkFacts_split (chunkLeaf71).1).2.2.1
  by_cases is72 : k = 72
  · subst is72
    exact (chunkFacts_split (chunkLeaf72).1).2.2.1
  by_cases is73 : k = 73
  · subst is73
    exact (chunkFacts_split (chunkLeaf73).1).2.2.1
  by_cases is74 : k = 74
  · subst is74
    exact (chunkFacts_split (chunkLeaf74).1).2.2.1
  by_cases is75 : k = 75
  · subst is75
    exact (chunkFacts_split (chunkLeaf75).1).2.2.1
  by_cases is76 : k = 76
  · subst is76
    exact (chunkFacts_split (chunkLeaf76).1).2.2.1
  by_cases is77 : k = 77
  · subst is77
    exact (chunkFacts_split (chunkLeaf77).1).2.2.1
  by_cases is78 : k = 78
  · subst is78
    exact (chunkFacts_split (chunkLeaf78).1).2.2.1
  by_cases is79 : k = 79
  · subst is79
    exact (chunkFacts_split (chunkLeaf79).1).2.2.1
  by_cases is80 : k = 80
  · subst is80
    exact (chunkFacts_split (chunkLeaf80).1).2.2.1
  by_cases is81 : k = 81
  · subst is81
    exact (chunkFacts_split (chunkLeaf81).1).2.2.1
  by_cases is82 : k = 82
  · subst is82
    exact (chunkFacts_split (chunkLeaf82).1).2.2.1
  by_cases is83 : k = 83
  · subst is83
    exact (chunkFacts_split (chunkLeaf83).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5
