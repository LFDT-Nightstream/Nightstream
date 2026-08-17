import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf28 :
    (chunkFacts (rowsChunk wire 28) 7168 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 28 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk28) := by
  native_decide

theorem chunkLeaf29 :
    (chunkFacts (rowsChunk wire 29) 7424 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 29 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk29) := by
  native_decide

theorem chunkLeaf30 :
    (chunkFacts (rowsChunk wire 30) 7680 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 30 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk30) := by
  native_decide

theorem chunkLeaf31 :
    (chunkFacts (rowsChunk wire 31) 7936 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 31 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk31) := by
  native_decide

theorem chunkLeaf32 :
    (chunkFacts (rowsChunk wire 32) 8192 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 32 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk32) := by
  native_decide

theorem chunkLeaf33 :
    (chunkFacts (rowsChunk wire 33) 8448 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 33 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk33) := by
  native_decide

theorem chunkLeaf34 :
    (chunkFacts (rowsChunk wire 34) 8704 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 34 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk34) := by
  native_decide

theorem chunkLeaf35 :
    (chunkFacts (rowsChunk wire 35) 8960 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 35 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk35) := by
  native_decide

theorem chunkLeaf36 :
    (chunkFacts (rowsChunk wire 36) 9216 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 36 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk36) := by
  native_decide

theorem chunkLeaf37 :
    (chunkFacts (rowsChunk wire 37) 9472 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 37 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk37) := by
  native_decide

theorem chunkLeaf38 :
    (chunkFacts (rowsChunk wire 38) 9728 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 38 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk38) := by
  native_decide

theorem chunkLeaf39 :
    (chunkFacts (rowsChunk wire 39) 9984 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 39 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk39) := by
  native_decide

theorem chunkLeaf40 :
    (chunkFacts (rowsChunk wire 40) 10240 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 40 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk40) := by
  native_decide

theorem chunkLeaf41 :
    (chunkFacts (rowsChunk wire 41) 10496 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 41 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk41) := by
  native_decide

theorem censusGroup :
    ∀ k, 28 ≤ k → k < 42 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is28 : k = 28
  · subst is28
    exact (chunkFacts_split (chunkLeaf28).1).1
  by_cases is29 : k = 29
  · subst is29
    exact (chunkFacts_split (chunkLeaf29).1).1
  by_cases is30 : k = 30
  · subst is30
    exact (chunkFacts_split (chunkLeaf30).1).1
  by_cases is31 : k = 31
  · subst is31
    exact (chunkFacts_split (chunkLeaf31).1).1
  by_cases is32 : k = 32
  · subst is32
    exact (chunkFacts_split (chunkLeaf32).1).1
  by_cases is33 : k = 33
  · subst is33
    exact (chunkFacts_split (chunkLeaf33).1).1
  by_cases is34 : k = 34
  · subst is34
    exact (chunkFacts_split (chunkLeaf34).1).1
  by_cases is35 : k = 35
  · subst is35
    exact (chunkFacts_split (chunkLeaf35).1).1
  by_cases is36 : k = 36
  · subst is36
    exact (chunkFacts_split (chunkLeaf36).1).1
  by_cases is37 : k = 37
  · subst is37
    exact (chunkFacts_split (chunkLeaf37).1).1
  by_cases is38 : k = 38
  · subst is38
    exact (chunkFacts_split (chunkLeaf38).1).1
  by_cases is39 : k = 39
  · subst is39
    exact (chunkFacts_split (chunkLeaf39).1).1
  by_cases is40 : k = 40
  · subst is40
    exact (chunkFacts_split (chunkLeaf40).1).1
  by_cases is41 : k = 41
  · subst is41
    exact (chunkFacts_split (chunkLeaf41).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 28 ≤ k → k < 42 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is28 : k = 28
  · subst is28
    exact (chunkFacts_split (chunkLeaf28).1).2.1
  by_cases is29 : k = 29
  · subst is29
    exact (chunkFacts_split (chunkLeaf29).1).2.1
  by_cases is30 : k = 30
  · subst is30
    exact (chunkFacts_split (chunkLeaf30).1).2.1
  by_cases is31 : k = 31
  · subst is31
    exact (chunkFacts_split (chunkLeaf31).1).2.1
  by_cases is32 : k = 32
  · subst is32
    exact (chunkFacts_split (chunkLeaf32).1).2.1
  by_cases is33 : k = 33
  · subst is33
    exact (chunkFacts_split (chunkLeaf33).1).2.1
  by_cases is34 : k = 34
  · subst is34
    exact (chunkFacts_split (chunkLeaf34).1).2.1
  by_cases is35 : k = 35
  · subst is35
    exact (chunkFacts_split (chunkLeaf35).1).2.1
  by_cases is36 : k = 36
  · subst is36
    exact (chunkFacts_split (chunkLeaf36).1).2.1
  by_cases is37 : k = 37
  · subst is37
    exact (chunkFacts_split (chunkLeaf37).1).2.1
  by_cases is38 : k = 38
  · subst is38
    exact (chunkFacts_split (chunkLeaf38).1).2.1
  by_cases is39 : k = 39
  · subst is39
    exact (chunkFacts_split (chunkLeaf39).1).2.1
  by_cases is40 : k = 40
  · subst is40
    exact (chunkFacts_split (chunkLeaf40).1).2.1
  by_cases is41 : k = 41
  · subst is41
    exact (chunkFacts_split (chunkLeaf41).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 28 ≤ k → k < 42 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is28 : k = 28
  · subst is28
    exact (chunkFacts_split (chunkLeaf28).1).2.2.1
  by_cases is29 : k = 29
  · subst is29
    exact (chunkFacts_split (chunkLeaf29).1).2.2.1
  by_cases is30 : k = 30
  · subst is30
    exact (chunkFacts_split (chunkLeaf30).1).2.2.1
  by_cases is31 : k = 31
  · subst is31
    exact (chunkFacts_split (chunkLeaf31).1).2.2.1
  by_cases is32 : k = 32
  · subst is32
    exact (chunkFacts_split (chunkLeaf32).1).2.2.1
  by_cases is33 : k = 33
  · subst is33
    exact (chunkFacts_split (chunkLeaf33).1).2.2.1
  by_cases is34 : k = 34
  · subst is34
    exact (chunkFacts_split (chunkLeaf34).1).2.2.1
  by_cases is35 : k = 35
  · subst is35
    exact (chunkFacts_split (chunkLeaf35).1).2.2.1
  by_cases is36 : k = 36
  · subst is36
    exact (chunkFacts_split (chunkLeaf36).1).2.2.1
  by_cases is37 : k = 37
  · subst is37
    exact (chunkFacts_split (chunkLeaf37).1).2.2.1
  by_cases is38 : k = 38
  · subst is38
    exact (chunkFacts_split (chunkLeaf38).1).2.2.1
  by_cases is39 : k = 39
  · subst is39
    exact (chunkFacts_split (chunkLeaf39).1).2.2.1
  by_cases is40 : k = 40
  · subst is40
    exact (chunkFacts_split (chunkLeaf40).1).2.2.1
  by_cases is41 : k = 41
  · subst is41
    exact (chunkFacts_split (chunkLeaf41).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2
