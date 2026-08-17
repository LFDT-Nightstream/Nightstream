import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf14 :
    (chunkFacts (rowsChunk wire 14) 3584 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 14 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk14) := by
  native_decide

theorem chunkLeaf15 :
    (chunkFacts (rowsChunk wire 15) 3840 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 15 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk15) := by
  native_decide

theorem chunkLeaf16 :
    (chunkFacts (rowsChunk wire 16) 4096 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 16 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk16) := by
  native_decide

theorem chunkLeaf17 :
    (chunkFacts (rowsChunk wire 17) 4352 256 39949 38626
      wire.completeFamilies
      ["fprime.base.step.prelude"] = true) ∧
      (rowsChunk wire 17 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk17) := by
  native_decide

theorem chunkLeaf18 :
    (chunkFacts (rowsChunk wire 18) 4608 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 18 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk18) := by
  native_decide

theorem chunkLeaf19 :
    (chunkFacts (rowsChunk wire 19) 4864 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 19 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk19) := by
  native_decide

theorem chunkLeaf20 :
    (chunkFacts (rowsChunk wire 20) 5120 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 20 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk20) := by
  native_decide

theorem chunkLeaf21 :
    (chunkFacts (rowsChunk wire 21) 5376 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 21 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk21) := by
  native_decide

theorem chunkLeaf22 :
    (chunkFacts (rowsChunk wire 22) 5632 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 22 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk22) := by
  native_decide

theorem chunkLeaf23 :
    (chunkFacts (rowsChunk wire 23) 5888 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 23 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk23) := by
  native_decide

theorem chunkLeaf24 :
    (chunkFacts (rowsChunk wire 24) 6144 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 24 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk24) := by
  native_decide

theorem chunkLeaf25 :
    (chunkFacts (rowsChunk wire 25) 6400 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 25 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk25) := by
  native_decide

theorem chunkLeaf26 :
    (chunkFacts (rowsChunk wire 26) 6656 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 26 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk26) := by
  native_decide

theorem chunkLeaf27 :
    (chunkFacts (rowsChunk wire 27) 6912 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 27 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk27) := by
  native_decide

theorem presence4 :
    (rowsChunk wire 17).any
      (fun row => decide (row.family = "fprime.base.step.prelude")) = true :=
  presence_of_chunkFacts (chunkLeaf17).1 (by decide)

theorem censusGroup :
    ∀ k, 14 ≤ k → k < 28 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is14 : k = 14
  · subst is14
    exact (chunkFacts_split (chunkLeaf14).1).1
  by_cases is15 : k = 15
  · subst is15
    exact (chunkFacts_split (chunkLeaf15).1).1
  by_cases is16 : k = 16
  · subst is16
    exact (chunkFacts_split (chunkLeaf16).1).1
  by_cases is17 : k = 17
  · subst is17
    exact (chunkFacts_split (chunkLeaf17).1).1
  by_cases is18 : k = 18
  · subst is18
    exact (chunkFacts_split (chunkLeaf18).1).1
  by_cases is19 : k = 19
  · subst is19
    exact (chunkFacts_split (chunkLeaf19).1).1
  by_cases is20 : k = 20
  · subst is20
    exact (chunkFacts_split (chunkLeaf20).1).1
  by_cases is21 : k = 21
  · subst is21
    exact (chunkFacts_split (chunkLeaf21).1).1
  by_cases is22 : k = 22
  · subst is22
    exact (chunkFacts_split (chunkLeaf22).1).1
  by_cases is23 : k = 23
  · subst is23
    exact (chunkFacts_split (chunkLeaf23).1).1
  by_cases is24 : k = 24
  · subst is24
    exact (chunkFacts_split (chunkLeaf24).1).1
  by_cases is25 : k = 25
  · subst is25
    exact (chunkFacts_split (chunkLeaf25).1).1
  by_cases is26 : k = 26
  · subst is26
    exact (chunkFacts_split (chunkLeaf26).1).1
  by_cases is27 : k = 27
  · subst is27
    exact (chunkFacts_split (chunkLeaf27).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 14 ≤ k → k < 28 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is14 : k = 14
  · subst is14
    exact (chunkFacts_split (chunkLeaf14).1).2.1
  by_cases is15 : k = 15
  · subst is15
    exact (chunkFacts_split (chunkLeaf15).1).2.1
  by_cases is16 : k = 16
  · subst is16
    exact (chunkFacts_split (chunkLeaf16).1).2.1
  by_cases is17 : k = 17
  · subst is17
    exact (chunkFacts_split (chunkLeaf17).1).2.1
  by_cases is18 : k = 18
  · subst is18
    exact (chunkFacts_split (chunkLeaf18).1).2.1
  by_cases is19 : k = 19
  · subst is19
    exact (chunkFacts_split (chunkLeaf19).1).2.1
  by_cases is20 : k = 20
  · subst is20
    exact (chunkFacts_split (chunkLeaf20).1).2.1
  by_cases is21 : k = 21
  · subst is21
    exact (chunkFacts_split (chunkLeaf21).1).2.1
  by_cases is22 : k = 22
  · subst is22
    exact (chunkFacts_split (chunkLeaf22).1).2.1
  by_cases is23 : k = 23
  · subst is23
    exact (chunkFacts_split (chunkLeaf23).1).2.1
  by_cases is24 : k = 24
  · subst is24
    exact (chunkFacts_split (chunkLeaf24).1).2.1
  by_cases is25 : k = 25
  · subst is25
    exact (chunkFacts_split (chunkLeaf25).1).2.1
  by_cases is26 : k = 26
  · subst is26
    exact (chunkFacts_split (chunkLeaf26).1).2.1
  by_cases is27 : k = 27
  · subst is27
    exact (chunkFacts_split (chunkLeaf27).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 14 ≤ k → k < 28 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is14 : k = 14
  · subst is14
    exact (chunkFacts_split (chunkLeaf14).1).2.2.1
  by_cases is15 : k = 15
  · subst is15
    exact (chunkFacts_split (chunkLeaf15).1).2.2.1
  by_cases is16 : k = 16
  · subst is16
    exact (chunkFacts_split (chunkLeaf16).1).2.2.1
  by_cases is17 : k = 17
  · subst is17
    exact (chunkFacts_split (chunkLeaf17).1).2.2.1
  by_cases is18 : k = 18
  · subst is18
    exact (chunkFacts_split (chunkLeaf18).1).2.2.1
  by_cases is19 : k = 19
  · subst is19
    exact (chunkFacts_split (chunkLeaf19).1).2.2.1
  by_cases is20 : k = 20
  · subst is20
    exact (chunkFacts_split (chunkLeaf20).1).2.2.1
  by_cases is21 : k = 21
  · subst is21
    exact (chunkFacts_split (chunkLeaf21).1).2.2.1
  by_cases is22 : k = 22
  · subst is22
    exact (chunkFacts_split (chunkLeaf22).1).2.2.1
  by_cases is23 : k = 23
  · subst is23
    exact (chunkFacts_split (chunkLeaf23).1).2.2.1
  by_cases is24 : k = 24
  · subst is24
    exact (chunkFacts_split (chunkLeaf24).1).2.2.1
  by_cases is25 : k = 25
  · subst is25
    exact (chunkFacts_split (chunkLeaf25).1).2.2.1
  by_cases is26 : k = 26
  · subst is26
    exact (chunkFacts_split (chunkLeaf26).1).2.2.1
  by_cases is27 : k = 27
  · subst is27
    exact (chunkFacts_split (chunkLeaf27).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1
