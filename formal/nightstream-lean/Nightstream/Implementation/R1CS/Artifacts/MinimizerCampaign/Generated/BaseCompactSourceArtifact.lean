import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11

/-!
GENERATED FILE - do not edit by hand.

Assembly of the chunk-aligned compact source artifact. All heavy
facts live in the bounded leaf modules; this module only
dispatches them and applies the structural composition theorems.
Exact validation is discharged by proof, never by evaluation.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

export Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire (families wire sourceArtifact reviewedPlan reviewedPlan_subset chunkRows_eq totalRows_eq chunkCount_eq)

theorem censuses :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k bound
  rw [chunkCount_eq] at bound
  by_cases group0 : k < 14
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.censusGroup k (by omega) group0
  by_cases group1 : k < 28
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.censusGroup k (by omega) group1
  by_cases group2 : k < 42
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.censusGroup k (by omega) group2
  by_cases group3 : k < 56
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.censusGroup k (by omega) group3
  by_cases group4 : k < 70
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.censusGroup k (by omega) group4
  by_cases group5 : k < 84
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.censusGroup k (by omega) group5
  by_cases group6 : k < 98
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.censusGroup k (by omega) group6
  by_cases group7 : k < 112
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.censusGroup k (by omega) group7
  by_cases group8 : k < 126
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.censusGroup k (by omega) group8
  by_cases group9 : k < 140
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.censusGroup k (by omega) group9
  by_cases group10 : k < 154
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.censusGroup k (by omega) group10
  by_cases group11 : k < 157
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.censusGroup k (by omega) group11
  exact absurd bound (by omega)

theorem rowsWf :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  by_cases group0 : k < 14
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.wfGroup k (by omega) group0
  by_cases group1 : k < 28
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.wfGroup k (by omega) group1
  by_cases group2 : k < 42
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.wfGroup k (by omega) group2
  by_cases group3 : k < 56
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.wfGroup k (by omega) group3
  by_cases group4 : k < 70
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.wfGroup k (by omega) group4
  by_cases group5 : k < 84
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.wfGroup k (by omega) group5
  by_cases group6 : k < 98
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.wfGroup k (by omega) group6
  by_cases group7 : k < 112
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.wfGroup k (by omega) group7
  by_cases group8 : k < 126
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.wfGroup k (by omega) group8
  by_cases group9 : k < 140
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.wfGroup k (by omega) group9
  by_cases group10 : k < 154
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.wfGroup k (by omega) group10
  by_cases group11 : k < 157
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.wfGroup k (by omega) group11
  exact absurd bound (by omega)

theorem familiesCovered :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  by_cases group0 : k < 14
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.coverGroup k (by omega) group0
  by_cases group1 : k < 28
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.coverGroup k (by omega) group1
  by_cases group2 : k < 42
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf2.coverGroup k (by omega) group2
  by_cases group3 : k < 56
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.coverGroup k (by omega) group3
  by_cases group4 : k < 70
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.coverGroup k (by omega) group4
  by_cases group5 : k < 84
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf5.coverGroup k (by omega) group5
  by_cases group6 : k < 98
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf6.coverGroup k (by omega) group6
  by_cases group7 : k < 112
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf7.coverGroup k (by omega) group7
  by_cases group8 : k < 126
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8.coverGroup k (by omega) group8
  by_cases group9 : k < 140
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9.coverGroup k (by omega) group9
  by_cases group10 : k < 154
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf10.coverGroup k (by omega) group10
  by_cases group11 : k < 157
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf11.coverGroup k (by omega) group11
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
    · exact ⟨0, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0.presence0⟩
    · exact ⟨57, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.presence1⟩
    · exact ⟨57, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.presence2⟩
    · exact ⟨59, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4.presence3⟩
    · exact ⟨17, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1.presence4⟩
    · exact ⟨55, by rw [chunkCount_eq]; decide, Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf3.presence5⟩
  rcases present with ⟨chunk, chunkBound, chunkAny⟩
  rw [List.any_eq_true] at chunkAny ⊢
  rcases chunkAny with ⟨row, rowMember, rowFamily⟩
  refine ⟨row, ?_, rowFamily⟩
  show row ∈ artifactRows wire
  unfold artifactRows
  exact List.mem_flatMap.mpr ⟨chunk, List.mem_range.mpr chunkBound, rowMember⟩

theorem scalarFacts :
    wire.schema = Artifact.supportedSchema ∧
      wire.profile ≠ "" ∧
      wire.scope ∈ Artifact.scopes ∧
      wire.diagnosticDigest ≠ "" ∧
      wire.fieldModulus = Artifact.goldilocksModulusDecimal ∧
      0 < wire.totalRows ∧
      0 < wire.columnCount ∧
      0 < wire.publicInputCount ∧
      wire.publicInputCount ≤ wire.columnCount ∧
      wire.constantOneColumn < wire.publicInputCount ∧
      wire.completeFamilies.Nodup ∧
      wire.completeFamilies.all
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

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact
