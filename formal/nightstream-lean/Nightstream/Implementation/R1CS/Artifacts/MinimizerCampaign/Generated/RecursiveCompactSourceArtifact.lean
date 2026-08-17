import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf1
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf2
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf3
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf4
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf5
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf6
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf7
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf8
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf9
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf10
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf11
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf13
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf14
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf15
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf19
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf21
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf22
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf25
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf27
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf28
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf29
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf30
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31

/-!
GENERATED FILE - do not edit by hand.

Assembly of the chunk-aligned compact source artifact. All heavy
facts live in the bounded leaf modules; this module only
dispatches them and applies the structural composition theorems.
Exact validation is discharged by proof, never by evaluation.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifact

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

export Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire (families wire sourceArtifact reviewedPlan reviewedPlan_subset chunkRows_eq totalRows_eq chunkCount_eq)

theorem censuses :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k bound
  rw [chunkCount_eq] at bound
  by_cases group0 : k < 1
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.censusGroup k (by omega) group0
  by_cases group1 : k < 2
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf1.censusGroup k (by omega) group1
  by_cases group2 : k < 3
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf2.censusGroup k (by omega) group2
  by_cases group3 : k < 7
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf3.censusGroup k (by omega) group3
  by_cases group4 : k < 8
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf4.censusGroup k (by omega) group4
  by_cases group5 : k < 9
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf5.censusGroup k (by omega) group5
  by_cases group6 : k < 17
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf6.censusGroup k (by omega) group6
  by_cases group7 : k < 18
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf7.censusGroup k (by omega) group7
  by_cases group8 : k < 32
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf8.censusGroup k (by omega) group8
  by_cases group9 : k < 46
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf9.censusGroup k (by omega) group9
  by_cases group10 : k < 57
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf10.censusGroup k (by omega) group10
  by_cases group11 : k < 58
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf11.censusGroup k (by omega) group11
  by_cases group12 : k < 72
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.censusGroup k (by omega) group12
  by_cases group13 : k < 86
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf13.censusGroup k (by omega) group13
  by_cases group14 : k < 100
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf14.censusGroup k (by omega) group14
  by_cases group15 : k < 102
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf15.censusGroup k (by omega) group15
  by_cases group16 : k < 103
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16.censusGroup k (by omega) group16
  by_cases group17 : k < 106
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.censusGroup k (by omega) group17
  by_cases group18 : k < 107
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.censusGroup k (by omega) group18
  by_cases group19 : k < 108
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf19.censusGroup k (by omega) group19
  by_cases group20 : k < 109
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.censusGroup k (by omega) group20
  by_cases group21 : k < 123
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf21.censusGroup k (by omega) group21
  by_cases group22 : k < 125
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf22.censusGroup k (by omega) group22
  by_cases group23 : k < 126
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23.censusGroup k (by omega) group23
  by_cases group24 : k < 129
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.censusGroup k (by omega) group24
  by_cases group25 : k < 130
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf25.censusGroup k (by omega) group25
  by_cases group26 : k < 131
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26.censusGroup k (by omega) group26
  by_cases group27 : k < 145
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf27.censusGroup k (by omega) group27
  by_cases group28 : k < 159
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf28.censusGroup k (by omega) group28
  by_cases group29 : k < 169
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf29.censusGroup k (by omega) group29
  by_cases group30 : k < 170
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf30.censusGroup k (by omega) group30
  by_cases group31 : k < 171
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31.censusGroup k (by omega) group31
  exact absurd bound (by omega)

theorem rowsWf :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  by_cases group0 : k < 1
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.wfGroup k (by omega) group0
  by_cases group1 : k < 2
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf1.wfGroup k (by omega) group1
  by_cases group2 : k < 3
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf2.wfGroup k (by omega) group2
  by_cases group3 : k < 7
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf3.wfGroup k (by omega) group3
  by_cases group4 : k < 8
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf4.wfGroup k (by omega) group4
  by_cases group5 : k < 9
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf5.wfGroup k (by omega) group5
  by_cases group6 : k < 17
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf6.wfGroup k (by omega) group6
  by_cases group7 : k < 18
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf7.wfGroup k (by omega) group7
  by_cases group8 : k < 32
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf8.wfGroup k (by omega) group8
  by_cases group9 : k < 46
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf9.wfGroup k (by omega) group9
  by_cases group10 : k < 57
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf10.wfGroup k (by omega) group10
  by_cases group11 : k < 58
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf11.wfGroup k (by omega) group11
  by_cases group12 : k < 72
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.wfGroup k (by omega) group12
  by_cases group13 : k < 86
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf13.wfGroup k (by omega) group13
  by_cases group14 : k < 100
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf14.wfGroup k (by omega) group14
  by_cases group15 : k < 102
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf15.wfGroup k (by omega) group15
  by_cases group16 : k < 103
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16.wfGroup k (by omega) group16
  by_cases group17 : k < 106
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.wfGroup k (by omega) group17
  by_cases group18 : k < 107
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.wfGroup k (by omega) group18
  by_cases group19 : k < 108
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf19.wfGroup k (by omega) group19
  by_cases group20 : k < 109
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.wfGroup k (by omega) group20
  by_cases group21 : k < 123
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf21.wfGroup k (by omega) group21
  by_cases group22 : k < 125
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf22.wfGroup k (by omega) group22
  by_cases group23 : k < 126
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23.wfGroup k (by omega) group23
  by_cases group24 : k < 129
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.wfGroup k (by omega) group24
  by_cases group25 : k < 130
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf25.wfGroup k (by omega) group25
  by_cases group26 : k < 131
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26.wfGroup k (by omega) group26
  by_cases group27 : k < 145
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf27.wfGroup k (by omega) group27
  by_cases group28 : k < 159
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf28.wfGroup k (by omega) group28
  by_cases group29 : k < 169
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf29.wfGroup k (by omega) group29
  by_cases group30 : k < 170
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf30.wfGroup k (by omega) group30
  by_cases group31 : k < 171
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31.wfGroup k (by omega) group31
  exact absurd bound (by omega)

theorem familiesCovered :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  by_cases group0 : k < 1
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.coverGroup k (by omega) group0
  by_cases group1 : k < 2
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf1.coverGroup k (by omega) group1
  by_cases group2 : k < 3
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf2.coverGroup k (by omega) group2
  by_cases group3 : k < 7
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf3.coverGroup k (by omega) group3
  by_cases group4 : k < 8
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf4.coverGroup k (by omega) group4
  by_cases group5 : k < 9
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf5.coverGroup k (by omega) group5
  by_cases group6 : k < 17
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf6.coverGroup k (by omega) group6
  by_cases group7 : k < 18
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf7.coverGroup k (by omega) group7
  by_cases group8 : k < 32
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf8.coverGroup k (by omega) group8
  by_cases group9 : k < 46
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf9.coverGroup k (by omega) group9
  by_cases group10 : k < 57
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf10.coverGroup k (by omega) group10
  by_cases group11 : k < 58
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf11.coverGroup k (by omega) group11
  by_cases group12 : k < 72
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.coverGroup k (by omega) group12
  by_cases group13 : k < 86
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf13.coverGroup k (by omega) group13
  by_cases group14 : k < 100
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf14.coverGroup k (by omega) group14
  by_cases group15 : k < 102
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf15.coverGroup k (by omega) group15
  by_cases group16 : k < 103
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16.coverGroup k (by omega) group16
  by_cases group17 : k < 106
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.coverGroup k (by omega) group17
  by_cases group18 : k < 107
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.coverGroup k (by omega) group18
  by_cases group19 : k < 108
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf19.coverGroup k (by omega) group19
  by_cases group20 : k < 109
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.coverGroup k (by omega) group20
  by_cases group21 : k < 123
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf21.coverGroup k (by omega) group21
  by_cases group22 : k < 125
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf22.coverGroup k (by omega) group22
  by_cases group23 : k < 126
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23.coverGroup k (by omega) group23
  by_cases group24 : k < 129
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.coverGroup k (by omega) group24
  by_cases group25 : k < 130
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf25.coverGroup k (by omega) group25
  by_cases group26 : k < 131
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26.coverGroup k (by omega) group26
  by_cases group27 : k < 145
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf27.coverGroup k (by omega) group27
  by_cases group28 : k < 159
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf28.coverGroup k (by omega) group28
  by_cases group29 : k < 169
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf29.coverGroup k (by omega) group29
  by_cases group30 : k < 170
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf30.coverGroup k (by omega) group30
  by_cases group31 : k < 171
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31.coverGroup k (by omega) group31
  exact absurd bound (by omega)

theorem chunkArithmeticFull :
    ∀ k, k + 1 < wire.chunkCount → wire.chunkLength k = wire.chunkRows := by
  intro k bound
  rw [chunkCount_eq] at bound
  simp only [Wire.chunkLength, Wire.chunkStart, chunkRows_eq, totalRows_eq]
  omega

theorem chunkArithmeticLast :
    wire.chunkCount ≠ 0 →
      (wire.chunkCount - 1) * wire.chunkRows +
        wire.chunkLength (wire.chunkCount - 1) = wire.totalRows := by
  intro _
  simp only [Wire.chunkLength, Wire.chunkStart, chunkCount_eq, chunkRows_eq, totalRows_eq]
  omega

theorem chunkArithmeticLead :
    wire.chunkCount ≠ 0 →
      (wire.chunkCount - 1) * wire.chunkRows ≤ wire.totalRows := by
  intro _
  simp only [chunkCount_eq, chunkRows_eq, totalRows_eq]
  omega

theorem chunkArithmeticEmpty :
    wire.chunkCount = 0 → wire.totalRows = 0 := by
  intro h
  rw [chunkCount_eq] at h
  exact absurd h (by decide)

theorem familyPresence :
    sourceArtifact.completeFamilies.all
      (fun family =>
        sourceArtifact.rows.any
          (fun row => decide (row.family = family))) = true := by
  rw [List.all_eq_true]
  intro family membership
  have present : ∃ chunk, chunk < wire.chunkCount ∧
      (rowsChunk wire chunk).any
        (fun row => decide (row.family = family)) = true := by
    fin_cases membership
    · exact ⟨0, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.presence0⟩
    · exact ⟨130, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26.presence1⟩
    · exact ⟨170, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31.presence2⟩
    · exact ⟨130, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26.presence3⟩
    · exact ⟨170, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31.presence4⟩
    · exact ⟨128, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence5⟩
    · exact ⟨170, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31.presence6⟩
    · exact ⟨0, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.presence7⟩
    · exact ⟨128, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence8⟩
    · exact ⟨128, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence9⟩
    · exact ⟨128, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence10⟩
    · exact ⟨0, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.presence11⟩
    · exact ⟨0, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.presence12⟩
    · exact ⟨0, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.presence13⟩
    · exact ⟨0, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0.presence14⟩
    · exact ⟨60, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.presence15⟩
    · exact ⟨102, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16.presence16⟩
    · exact ⟨64, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.presence17⟩
    · exact ⟨64, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.presence18⟩
    · exact ⟨64, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.presence19⟩
    · exact ⟨64, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.presence20⟩
    · exact ⟨64, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.presence21⟩
    · exact ⟨57, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf11.presence22⟩
    · exact ⟨61, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.presence23⟩
    · exact ⟨63, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12.presence24⟩
    · exact ⟨128, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence25⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence26⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence27⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence28⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence29⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence30⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence31⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence32⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence33⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence34⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence35⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence36⟩
    · exact ⟨103, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17.presence37⟩
    · exact ⟨106, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.presence38⟩
    · exact ⟨106, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.presence39⟩
    · exact ⟨106, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.presence40⟩
    · exact ⟨106, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.presence41⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence42⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence43⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence44⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence45⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence46⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence47⟩
    · exact ⟨125, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23.presence48⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence49⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence50⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence51⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence52⟩
    · exact ⟨125, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23.presence53⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence54⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence55⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence56⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence57⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence58⟩
    · exact ⟨126, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence59⟩
    · exact ⟨127, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence60⟩
    · exact ⟨127, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence61⟩
    · exact ⟨127, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence62⟩
    · exact ⟨127, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence63⟩
    · exact ⟨127, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence64⟩
    · exact ⟨127, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence65⟩
    · exact ⟨127, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence66⟩
    · exact ⟨106, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.presence67⟩
    · exact ⟨106, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.presence68⟩
    · exact ⟨108, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.presence69⟩
    · exact ⟨108, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.presence70⟩
    · exact ⟨106, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.presence71⟩
    · exact ⟨108, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.presence72⟩
    · exact ⟨106, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18.presence73⟩
    · exact ⟨108, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.presence74⟩
    · exact ⟨108, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.presence75⟩
    · exact ⟨108, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20.presence76⟩
    · exact ⟨125, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23.presence77⟩
    · exact ⟨125, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23.presence78⟩
    · exact ⟨125, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23.presence79⟩
    · exact ⟨128, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf24.presence80⟩
    · exact ⟨102, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16.presence81⟩
  rcases present with ⟨chunk, chunkBound, chunkAny⟩
  rw [List.any_eq_true] at chunkAny ⊢
  rcases chunkAny with ⟨row, rowMember, rowFamily⟩
  refine ⟨row, ?_, rowFamily⟩
  show row ∈ artifactRows wire
  unfold artifactRows
  exact List.mem_flatMap.mpr ⟨chunk, List.mem_range.mpr chunkBound, rowMember⟩

theorem scalarFacts :
    sourceArtifact.schema = Artifact.supportedSchema ∧
      sourceArtifact.profile ≠ "" ∧
      sourceArtifact.scope ∈ Artifact.scopes ∧
      sourceArtifact.diagnosticDigest ≠ "" ∧
      sourceArtifact.fieldModulus = Artifact.goldilocksModulusDecimal ∧
      0 < sourceArtifact.totalRows ∧
      0 < sourceArtifact.columnCount ∧
      0 < sourceArtifact.publicInputCount ∧
      sourceArtifact.publicInputCount ≤ sourceArtifact.columnCount ∧
      sourceArtifact.constantOneColumn < sourceArtifact.publicInputCount ∧
      sourceArtifact.completeFamilies.Nodup ∧
      sourceArtifact.completeFamilies.all
        (fun family => decide (family ≠ "")) = true := by
  native_decide

theorem sourceArtifact_indexCensus :
    (artifactRows wire).map (fun row => row.sourceIndex) =
      List.range wire.totalRows :=
  covers_indexes_of_chunks wire censuses chunkArithmeticFull
    chunkArithmeticLast chunkArithmeticLead chunkArithmeticEmpty

theorem sourceArtifact_coversFullRelation :
    sourceArtifact.CoversFullRelation :=
  coversFullRelation_of_chunks wire censuses chunkArithmeticFull
    chunkArithmeticLast chunkArithmeticLead chunkArithmeticEmpty familiesCovered

theorem sourceArtifact_wellFormed : sourceArtifact.WellFormed :=
  wellFormed_of_chunks wire scalarFacts sourceArtifact_indexCensus
    rowsWf familyPresence

theorem sourceArtifact_exactValidation :
    Artifact.ExactValidation sourceArtifact sourceArtifact = true :=
  exactValidation_self sourceArtifact_wellFormed

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifact
