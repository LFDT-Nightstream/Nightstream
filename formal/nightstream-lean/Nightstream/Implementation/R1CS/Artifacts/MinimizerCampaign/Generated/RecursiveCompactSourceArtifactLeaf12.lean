import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf58 :
    chunkFacts (rowsChunk wire 58) 3801088 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf59 :
    chunkFacts (rowsChunk wire 59) 3866624 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf60 :
    chunkFacts (rowsChunk wire 60) 3932160 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.challenges"] = true := by
  native_decide

theorem chunkLeaf61 :
    chunkFacts (rowsChunk wire 61) 3997696 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.sumcheck"] = true := by
  native_decide

theorem chunkLeaf62 :
    chunkFacts (rowsChunk wire 62) 4063232 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf63 :
    chunkFacts (rowsChunk wire 63) 4128768 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.terminal"] = true := by
  native_decide

theorem chunkLeaf64 :
    chunkFacts (rowsChunk wire 64) 4194304 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.output_digest.preimage.outer_header",
       "nifs.pi_ccs.padded_row.output_digest.preimage.source_headers",
       "nifs.pi_ccs.padded_row.output_digest.preimage.y_ring",
       "nifs.pi_ccs.padded_row.output_digest.sis",
       "nifs.pi_ccs.padded_row.output_transcript"] = true := by
  native_decide

theorem chunkLeaf65 :
    chunkFacts (rowsChunk wire 65) 4259840 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf66 :
    chunkFacts (rowsChunk wire 66) 4325376 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf67 :
    chunkFacts (rowsChunk wire 67) 4390912 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf68 :
    chunkFacts (rowsChunk wire 68) 4456448 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf69 :
    chunkFacts (rowsChunk wire 69) 4521984 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf70 :
    chunkFacts (rowsChunk wire 70) 4587520 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf71 :
    chunkFacts (rowsChunk wire 71) 4653056 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem presence15 :
    (rowsChunk wire 60).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.challenges")) = true :=
  presence_of_chunkFacts chunkLeaf60 (by decide)

theorem presence17 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.outer_header")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence18 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.source_headers")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence19 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.y_ring")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence20 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.sis")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence21 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_transcript")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence23 :
    (rowsChunk wire 61).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.sumcheck")) = true :=
  presence_of_chunkFacts chunkLeaf61 (by decide)

theorem presence24 :
    (rowsChunk wire 63).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.terminal")) = true :=
  presence_of_chunkFacts chunkLeaf63 (by decide)

theorem censusGroup :
    ∀ k, 58 ≤ k → k < 72 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is58 : k = 58
  · subst is58
    exact (chunkFacts_split chunkLeaf58).1
  by_cases is59 : k = 59
  · subst is59
    exact (chunkFacts_split chunkLeaf59).1
  by_cases is60 : k = 60
  · subst is60
    exact (chunkFacts_split chunkLeaf60).1
  by_cases is61 : k = 61
  · subst is61
    exact (chunkFacts_split chunkLeaf61).1
  by_cases is62 : k = 62
  · subst is62
    exact (chunkFacts_split chunkLeaf62).1
  by_cases is63 : k = 63
  · subst is63
    exact (chunkFacts_split chunkLeaf63).1
  by_cases is64 : k = 64
  · subst is64
    exact (chunkFacts_split chunkLeaf64).1
  by_cases is65 : k = 65
  · subst is65
    exact (chunkFacts_split chunkLeaf65).1
  by_cases is66 : k = 66
  · subst is66
    exact (chunkFacts_split chunkLeaf66).1
  by_cases is67 : k = 67
  · subst is67
    exact (chunkFacts_split chunkLeaf67).1
  by_cases is68 : k = 68
  · subst is68
    exact (chunkFacts_split chunkLeaf68).1
  by_cases is69 : k = 69
  · subst is69
    exact (chunkFacts_split chunkLeaf69).1
  by_cases is70 : k = 70
  · subst is70
    exact (chunkFacts_split chunkLeaf70).1
  by_cases is71 : k = 71
  · subst is71
    exact (chunkFacts_split chunkLeaf71).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 58 ≤ k → k < 72 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is58 : k = 58
  · subst is58
    exact (chunkFacts_split chunkLeaf58).2.1
  by_cases is59 : k = 59
  · subst is59
    exact (chunkFacts_split chunkLeaf59).2.1
  by_cases is60 : k = 60
  · subst is60
    exact (chunkFacts_split chunkLeaf60).2.1
  by_cases is61 : k = 61
  · subst is61
    exact (chunkFacts_split chunkLeaf61).2.1
  by_cases is62 : k = 62
  · subst is62
    exact (chunkFacts_split chunkLeaf62).2.1
  by_cases is63 : k = 63
  · subst is63
    exact (chunkFacts_split chunkLeaf63).2.1
  by_cases is64 : k = 64
  · subst is64
    exact (chunkFacts_split chunkLeaf64).2.1
  by_cases is65 : k = 65
  · subst is65
    exact (chunkFacts_split chunkLeaf65).2.1
  by_cases is66 : k = 66
  · subst is66
    exact (chunkFacts_split chunkLeaf66).2.1
  by_cases is67 : k = 67
  · subst is67
    exact (chunkFacts_split chunkLeaf67).2.1
  by_cases is68 : k = 68
  · subst is68
    exact (chunkFacts_split chunkLeaf68).2.1
  by_cases is69 : k = 69
  · subst is69
    exact (chunkFacts_split chunkLeaf69).2.1
  by_cases is70 : k = 70
  · subst is70
    exact (chunkFacts_split chunkLeaf70).2.1
  by_cases is71 : k = 71
  · subst is71
    exact (chunkFacts_split chunkLeaf71).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 58 ≤ k → k < 72 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is58 : k = 58
  · subst is58
    exact (chunkFacts_split chunkLeaf58).2.2.1
  by_cases is59 : k = 59
  · subst is59
    exact (chunkFacts_split chunkLeaf59).2.2.1
  by_cases is60 : k = 60
  · subst is60
    exact (chunkFacts_split chunkLeaf60).2.2.1
  by_cases is61 : k = 61
  · subst is61
    exact (chunkFacts_split chunkLeaf61).2.2.1
  by_cases is62 : k = 62
  · subst is62
    exact (chunkFacts_split chunkLeaf62).2.2.1
  by_cases is63 : k = 63
  · subst is63
    exact (chunkFacts_split chunkLeaf63).2.2.1
  by_cases is64 : k = 64
  · subst is64
    exact (chunkFacts_split chunkLeaf64).2.2.1
  by_cases is65 : k = 65
  · subst is65
    exact (chunkFacts_split chunkLeaf65).2.2.1
  by_cases is66 : k = 66
  · subst is66
    exact (chunkFacts_split chunkLeaf66).2.2.1
  by_cases is67 : k = 67
  · subst is67
    exact (chunkFacts_split chunkLeaf67).2.2.1
  by_cases is68 : k = 68
  · subst is68
    exact (chunkFacts_split chunkLeaf68).2.2.1
  by_cases is69 : k = 69
  · subst is69
    exact (chunkFacts_split chunkLeaf69).2.2.1
  by_cases is70 : k = 70
  · subst is70
    exact (chunkFacts_split chunkLeaf70).2.2.1
  by_cases is71 : k = 71
  · subst is71
    exact (chunkFacts_split chunkLeaf71).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12
