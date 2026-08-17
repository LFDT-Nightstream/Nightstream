import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf42 :
    (chunkFacts (rowsChunk wire 42) 10752 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 42 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk42) := by
  native_decide

theorem chunkLeaf43 :
    (chunkFacts (rowsChunk wire 43) 11008 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 43 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk43) := by
  native_decide

theorem chunkLeaf44 :
    (chunkFacts (rowsChunk wire 44) 11264 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 44 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk44) := by
  native_decide

theorem chunkLeaf45 :
    (chunkFacts (rowsChunk wire 45) 11520 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 45 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk45) := by
  native_decide

theorem chunkLeaf46 :
    (chunkFacts (rowsChunk wire 46) 11776 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 46 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk46) := by
  native_decide

theorem chunkLeaf47 :
    (chunkFacts (rowsChunk wire 47) 12032 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 47 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk47) := by
  native_decide

theorem chunkLeaf48 :
    (chunkFacts (rowsChunk wire 48) 12288 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 48 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk48) := by
  native_decide

theorem chunkLeaf49 :
    (chunkFacts (rowsChunk wire 49) 12544 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 49 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk49) := by
  native_decide

theorem chunkLeaf50 :
    (chunkFacts (rowsChunk wire 50) 12800 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 50 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk50) := by
  native_decide

theorem chunkLeaf51 :
    (chunkFacts (rowsChunk wire 51) 13056 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 51 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk51) := by
  native_decide

theorem chunkLeaf52 :
    (chunkFacts (rowsChunk wire 52) 13312 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 52 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk52) := by
  native_decide

theorem chunkLeaf53 :
    (chunkFacts (rowsChunk wire 53) 13568 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 53 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk53) := by
  native_decide

theorem chunkLeaf54 :
    (chunkFacts (rowsChunk wire 54) 13824 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 54 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk54) := by
  native_decide

theorem chunkLeaf55 :
    (chunkFacts (rowsChunk wire 55) 14080 256 39949 38626
      wire.completeFamilies
      ["fprime.base.step.source"] = true) ∧
      (rowsChunk wire 55 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk55) := by
  native_decide

theorem presence5 :
    (rowsChunk wire 55).any
      (fun row => decide (row.family = "fprime.base.step.source")) = true :=
  presence_of_chunkFacts (chunkLeaf55).1 (by decide)

theorem censusGroup :
    ∀ k, 42 ≤ k → k < 56 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is42 : k = 42
  · subst is42
    exact (chunkFacts_split (chunkLeaf42).1).1
  by_cases is43 : k = 43
  · subst is43
    exact (chunkFacts_split (chunkLeaf43).1).1
  by_cases is44 : k = 44
  · subst is44
    exact (chunkFacts_split (chunkLeaf44).1).1
  by_cases is45 : k = 45
  · subst is45
    exact (chunkFacts_split (chunkLeaf45).1).1
  by_cases is46 : k = 46
  · subst is46
    exact (chunkFacts_split (chunkLeaf46).1).1
  by_cases is47 : k = 47
  · subst is47
    exact (chunkFacts_split (chunkLeaf47).1).1
  by_cases is48 : k = 48
  · subst is48
    exact (chunkFacts_split (chunkLeaf48).1).1
  by_cases is49 : k = 49
  · subst is49
    exact (chunkFacts_split (chunkLeaf49).1).1
  by_cases is50 : k = 50
  · subst is50
    exact (chunkFacts_split (chunkLeaf50).1).1
  by_cases is51 : k = 51
  · subst is51
    exact (chunkFacts_split (chunkLeaf51).1).1
  by_cases is52 : k = 52
  · subst is52
    exact (chunkFacts_split (chunkLeaf52).1).1
  by_cases is53 : k = 53
  · subst is53
    exact (chunkFacts_split (chunkLeaf53).1).1
  by_cases is54 : k = 54
  · subst is54
    exact (chunkFacts_split (chunkLeaf54).1).1
  by_cases is55 : k = 55
  · subst is55
    exact (chunkFacts_split (chunkLeaf55).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 42 ≤ k → k < 56 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is42 : k = 42
  · subst is42
    exact (chunkFacts_split (chunkLeaf42).1).2.1
  by_cases is43 : k = 43
  · subst is43
    exact (chunkFacts_split (chunkLeaf43).1).2.1
  by_cases is44 : k = 44
  · subst is44
    exact (chunkFacts_split (chunkLeaf44).1).2.1
  by_cases is45 : k = 45
  · subst is45
    exact (chunkFacts_split (chunkLeaf45).1).2.1
  by_cases is46 : k = 46
  · subst is46
    exact (chunkFacts_split (chunkLeaf46).1).2.1
  by_cases is47 : k = 47
  · subst is47
    exact (chunkFacts_split (chunkLeaf47).1).2.1
  by_cases is48 : k = 48
  · subst is48
    exact (chunkFacts_split (chunkLeaf48).1).2.1
  by_cases is49 : k = 49
  · subst is49
    exact (chunkFacts_split (chunkLeaf49).1).2.1
  by_cases is50 : k = 50
  · subst is50
    exact (chunkFacts_split (chunkLeaf50).1).2.1
  by_cases is51 : k = 51
  · subst is51
    exact (chunkFacts_split (chunkLeaf51).1).2.1
  by_cases is52 : k = 52
  · subst is52
    exact (chunkFacts_split (chunkLeaf52).1).2.1
  by_cases is53 : k = 53
  · subst is53
    exact (chunkFacts_split (chunkLeaf53).1).2.1
  by_cases is54 : k = 54
  · subst is54
    exact (chunkFacts_split (chunkLeaf54).1).2.1
  by_cases is55 : k = 55
  · subst is55
    exact (chunkFacts_split (chunkLeaf55).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 42 ≤ k → k < 56 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is42 : k = 42
  · subst is42
    exact (chunkFacts_split (chunkLeaf42).1).2.2.1
  by_cases is43 : k = 43
  · subst is43
    exact (chunkFacts_split (chunkLeaf43).1).2.2.1
  by_cases is44 : k = 44
  · subst is44
    exact (chunkFacts_split (chunkLeaf44).1).2.2.1
  by_cases is45 : k = 45
  · subst is45
    exact (chunkFacts_split (chunkLeaf45).1).2.2.1
  by_cases is46 : k = 46
  · subst is46
    exact (chunkFacts_split (chunkLeaf46).1).2.2.1
  by_cases is47 : k = 47
  · subst is47
    exact (chunkFacts_split (chunkLeaf47).1).2.2.1
  by_cases is48 : k = 48
  · subst is48
    exact (chunkFacts_split (chunkLeaf48).1).2.2.1
  by_cases is49 : k = 49
  · subst is49
    exact (chunkFacts_split (chunkLeaf49).1).2.2.1
  by_cases is50 : k = 50
  · subst is50
    exact (chunkFacts_split (chunkLeaf50).1).2.2.1
  by_cases is51 : k = 51
  · subst is51
    exact (chunkFacts_split (chunkLeaf51).1).2.2.1
  by_cases is52 : k = 52
  · subst is52
    exact (chunkFacts_split (chunkLeaf52).1).2.2.1
  by_cases is53 : k = 53
  · subst is53
    exact (chunkFacts_split (chunkLeaf53).1).2.2.1
  by_cases is54 : k = 54
  · subst is54
    exact (chunkFacts_split (chunkLeaf54).1).2.2.1
  by_cases is55 : k = 55
  · subst is55
    exact (chunkFacts_split (chunkLeaf55).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3
