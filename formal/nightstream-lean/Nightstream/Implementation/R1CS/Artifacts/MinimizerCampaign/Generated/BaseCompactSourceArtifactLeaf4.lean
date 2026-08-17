import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf56 :
    (chunkFacts (rowsChunk wire 56) 14336 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 56 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk56) := by
  native_decide

theorem chunkLeaf57 :
    (chunkFacts (rowsChunk wire 57) 14592 256 39949 38626
      wire.completeFamilies
      ["fprime.base.step.advance",
       "fprime.base.step.initial"] = true) ∧
      (rowsChunk wire 57 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk57) := by
  native_decide

theorem chunkLeaf58 :
    (chunkFacts (rowsChunk wire 58) 14848 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 58 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk58) := by
  native_decide

theorem chunkLeaf59 :
    (chunkFacts (rowsChunk wire 59) 15104 256 39949 38626
      wire.completeFamilies
      ["fprime.base.step.output"] = true) ∧
      (rowsChunk wire 59 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk59) := by
  native_decide

theorem chunkLeaf60 :
    (chunkFacts (rowsChunk wire 60) 15360 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 60 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk60) := by
  native_decide

theorem chunkLeaf61 :
    (chunkFacts (rowsChunk wire 61) 15616 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 61 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk61) := by
  native_decide

theorem chunkLeaf62 :
    (chunkFacts (rowsChunk wire 62) 15872 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 62 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk62) := by
  native_decide

theorem chunkLeaf63 :
    (chunkFacts (rowsChunk wire 63) 16128 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 63 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk63) := by
  native_decide

theorem chunkLeaf64 :
    (chunkFacts (rowsChunk wire 64) 16384 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 64 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk64) := by
  native_decide

theorem chunkLeaf65 :
    (chunkFacts (rowsChunk wire 65) 16640 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 65 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk65) := by
  native_decide

theorem chunkLeaf66 :
    (chunkFacts (rowsChunk wire 66) 16896 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 66 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk66) := by
  native_decide

theorem chunkLeaf67 :
    (chunkFacts (rowsChunk wire 67) 17152 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 67 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk67) := by
  native_decide

theorem chunkLeaf68 :
    (chunkFacts (rowsChunk wire 68) 17408 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 68 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk68) := by
  native_decide

theorem chunkLeaf69 :
    (chunkFacts (rowsChunk wire 69) 17664 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 69 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk69) := by
  native_decide

theorem presence1 :
    (rowsChunk wire 57).any
      (fun row => decide (row.family = "fprime.base.step.advance")) = true :=
  presence_of_chunkFacts (chunkLeaf57).1 (by decide)

theorem presence2 :
    (rowsChunk wire 57).any
      (fun row => decide (row.family = "fprime.base.step.initial")) = true :=
  presence_of_chunkFacts (chunkLeaf57).1 (by decide)

theorem presence3 :
    (rowsChunk wire 59).any
      (fun row => decide (row.family = "fprime.base.step.output")) = true :=
  presence_of_chunkFacts (chunkLeaf59).1 (by decide)

theorem censusGroup :
    ∀ k, 56 ≤ k → k < 70 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is56 : k = 56
  · subst is56
    exact (chunkFacts_split (chunkLeaf56).1).1
  by_cases is57 : k = 57
  · subst is57
    exact (chunkFacts_split (chunkLeaf57).1).1
  by_cases is58 : k = 58
  · subst is58
    exact (chunkFacts_split (chunkLeaf58).1).1
  by_cases is59 : k = 59
  · subst is59
    exact (chunkFacts_split (chunkLeaf59).1).1
  by_cases is60 : k = 60
  · subst is60
    exact (chunkFacts_split (chunkLeaf60).1).1
  by_cases is61 : k = 61
  · subst is61
    exact (chunkFacts_split (chunkLeaf61).1).1
  by_cases is62 : k = 62
  · subst is62
    exact (chunkFacts_split (chunkLeaf62).1).1
  by_cases is63 : k = 63
  · subst is63
    exact (chunkFacts_split (chunkLeaf63).1).1
  by_cases is64 : k = 64
  · subst is64
    exact (chunkFacts_split (chunkLeaf64).1).1
  by_cases is65 : k = 65
  · subst is65
    exact (chunkFacts_split (chunkLeaf65).1).1
  by_cases is66 : k = 66
  · subst is66
    exact (chunkFacts_split (chunkLeaf66).1).1
  by_cases is67 : k = 67
  · subst is67
    exact (chunkFacts_split (chunkLeaf67).1).1
  by_cases is68 : k = 68
  · subst is68
    exact (chunkFacts_split (chunkLeaf68).1).1
  by_cases is69 : k = 69
  · subst is69
    exact (chunkFacts_split (chunkLeaf69).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 56 ≤ k → k < 70 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is56 : k = 56
  · subst is56
    exact (chunkFacts_split (chunkLeaf56).1).2.1
  by_cases is57 : k = 57
  · subst is57
    exact (chunkFacts_split (chunkLeaf57).1).2.1
  by_cases is58 : k = 58
  · subst is58
    exact (chunkFacts_split (chunkLeaf58).1).2.1
  by_cases is59 : k = 59
  · subst is59
    exact (chunkFacts_split (chunkLeaf59).1).2.1
  by_cases is60 : k = 60
  · subst is60
    exact (chunkFacts_split (chunkLeaf60).1).2.1
  by_cases is61 : k = 61
  · subst is61
    exact (chunkFacts_split (chunkLeaf61).1).2.1
  by_cases is62 : k = 62
  · subst is62
    exact (chunkFacts_split (chunkLeaf62).1).2.1
  by_cases is63 : k = 63
  · subst is63
    exact (chunkFacts_split (chunkLeaf63).1).2.1
  by_cases is64 : k = 64
  · subst is64
    exact (chunkFacts_split (chunkLeaf64).1).2.1
  by_cases is65 : k = 65
  · subst is65
    exact (chunkFacts_split (chunkLeaf65).1).2.1
  by_cases is66 : k = 66
  · subst is66
    exact (chunkFacts_split (chunkLeaf66).1).2.1
  by_cases is67 : k = 67
  · subst is67
    exact (chunkFacts_split (chunkLeaf67).1).2.1
  by_cases is68 : k = 68
  · subst is68
    exact (chunkFacts_split (chunkLeaf68).1).2.1
  by_cases is69 : k = 69
  · subst is69
    exact (chunkFacts_split (chunkLeaf69).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 56 ≤ k → k < 70 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is56 : k = 56
  · subst is56
    exact (chunkFacts_split (chunkLeaf56).1).2.2.1
  by_cases is57 : k = 57
  · subst is57
    exact (chunkFacts_split (chunkLeaf57).1).2.2.1
  by_cases is58 : k = 58
  · subst is58
    exact (chunkFacts_split (chunkLeaf58).1).2.2.1
  by_cases is59 : k = 59
  · subst is59
    exact (chunkFacts_split (chunkLeaf59).1).2.2.1
  by_cases is60 : k = 60
  · subst is60
    exact (chunkFacts_split (chunkLeaf60).1).2.2.1
  by_cases is61 : k = 61
  · subst is61
    exact (chunkFacts_split (chunkLeaf61).1).2.2.1
  by_cases is62 : k = 62
  · subst is62
    exact (chunkFacts_split (chunkLeaf62).1).2.2.1
  by_cases is63 : k = 63
  · subst is63
    exact (chunkFacts_split (chunkLeaf63).1).2.2.1
  by_cases is64 : k = 64
  · subst is64
    exact (chunkFacts_split (chunkLeaf64).1).2.2.1
  by_cases is65 : k = 65
  · subst is65
    exact (chunkFacts_split (chunkLeaf65).1).2.2.1
  by_cases is66 : k = 66
  · subst is66
    exact (chunkFacts_split (chunkLeaf66).1).2.2.1
  by_cases is67 : k = 67
  · subst is67
    exact (chunkFacts_split (chunkLeaf67).1).2.2.1
  by_cases is68 : k = 68
  · subst is68
    exact (chunkFacts_split (chunkLeaf68).1).2.2.1
  by_cases is69 : k = 69
  · subst is69
    exact (chunkFacts_split (chunkLeaf69).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4
