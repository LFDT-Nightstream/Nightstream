use super::*;

pub(super) fn render_nested_owner_aggregate() -> String {
    let imports = NESTED_OWNER_SPECS
        .iter()
        .map(|spec| {
            format!(
                "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistory{}Artifact",
                spec.module_suffix
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let extra_imports = r#"import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryManifestData
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsRecursiveAllocation
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsRecursiveAuthority
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsRecursiveOutputBinding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsTerminalAllocation
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsTerminalAuthorityTail
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsTerminalOutputBinding
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPiDecArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiRlcRecursiveLinearFolds
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiRlcRecursiveShape
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiRlcTerminalLinearFolds
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiRlcTerminalShape
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryProjectionArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryProjectionRoles
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryProjectionSound
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistorySumcheckArtifact"#;
    let body = r#"/-!
Exact manifest-ordered row decomposition for the recursive and terminal PiCCS
and PiRLC parent owners.  Each large residual is reconstructed by a compact
owner certificate; repeated algebraic families reuse their checked compiler.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNestedOwners

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def recursivePiCcsResidualOwners : List Owner :=
  [ FPrimeFullHistoryRecursivePiCcsFreshDigests.owner
  , FPrimeFullHistoryRecursivePiCcsRunningAuthority.owner
  , FPrimeFullHistoryRecursivePiCcsTranscript.owner
  , FPrimeFullHistoryRecursivePiCcsFeInitial.owner
  , FPrimeFullHistoryRecursivePiCcsFeOptionalClaim.owner
  , FPrimeFullHistoryRecursivePiCcsFeSumcheck.owner
  , FPrimeFullHistoryRecursivePiCcsNcSumcheck.owner
  , FPrimeFullHistoryRecursivePiCcsFeTerminal.owner
  , FPrimeFullHistoryRecursivePiCcsNcTerminal.owner
  , FPrimeFullHistoryRecursivePiCcsCatchup.owner
  , FPrimeFullHistoryRecursivePiCcsOutputMessageHashes.owner ]

def terminalPiCcsResidualOwners : List Owner :=
  [ FPrimeFullHistoryTerminalPiCcsFreshDigests.owner
  , FPrimeFullHistoryTerminalPiCcsRunningAuthority.owner
  , FPrimeFullHistoryTerminalPiCcsTranscript.owner
  , FPrimeFullHistoryTerminalPiCcsFeInitial.owner
  , FPrimeFullHistoryTerminalPiCcsFeOptionalClaim.owner
  , FPrimeFullHistoryTerminalPiCcsFeSumcheck.owner
  , FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner
  , FPrimeFullHistoryTerminalPiCcsFeTerminal.owner
  , FPrimeFullHistoryTerminalPiCcsNcTerminal.owner
  , FPrimeFullHistoryTerminalPiCcsCatchup.owner
  , FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.owner ]

def recursivePiRlcResidualOwners : List Owner :=
  [ FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner
  , FPrimeFullHistoryRecursivePiRlcProjectionBinding.owner ]

def terminalPiRlcResidualOwners : List Owner :=
  [ FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner
  , FPrimeFullHistoryTerminalPiRlcProjectionBinding.owner ]

def terminalPiCcsAuthorityRows : List Row :=
  FPrimeFullHistoryPiDec.terminalCeRows ++
    FPrimeFullHistoryPiCcsTerminalAuthorityTail.rows

theorem terminalPiCcsAuthorityRows_length :
    terminalPiCcsAuthorityRows.length = 11887 := by
  simp [terminalPiCcsAuthorityRows, FPrimeFullHistoryPiDec.terminalCeRows_length,
    FPrimeFullHistoryPiCcsTerminalAuthorityTail.rows_length,
    FPrimeFullHistoryPiDec.rowCount,
    FPrimeFullHistoryPiCcsTerminalAuthorityTail.rowCount]

def recursiveProjectionSharedRows : List Row :=
  FPrimeFullHistoryProjection.mappedRows
    FPrimeFullHistoryProjection.RecursiveSharedInstructions
    FPrimeFullHistoryProjection.recursiveSharedMap

def terminalProjectionSharedRows : List Row :=
  FPrimeFullHistoryProjection.mappedRows
    FPrimeFullHistoryProjection.TerminalSharedInstructions
    FPrimeFullHistoryProjection.terminalSharedMap

theorem recursiveProjectionSharedRows_length :
    recursiveProjectionSharedRows.length = 380 := by native_decide

theorem terminalProjectionSharedRows_length :
    terminalProjectionSharedRows.length = 1892 := by native_decide

def recursiveProjectionIdentityPieces : List (List Row) :=
  FPrimeFullHistoryProjection.recursiveTraces.map (fun trace =>
      (FPrimeFullHistoryProjection.traceRows trace).drop 380) ++
    [FPrimeFullHistoryProjectionRoles.recursiveGlueRows]

def terminalProjectionIdentityPieces : List (List Row) :=
  FPrimeFullHistoryProjection.terminalTraces.map (fun trace =>
      (FPrimeFullHistoryProjection.traceRows trace).drop 1892) ++
    [FPrimeFullHistoryProjectionRoles.terminalGlueRows]

def recursiveProjectionIdentityRows : List Row :=
  recursiveProjectionIdentityPieces.flatten

def terminalProjectionIdentityRows : List Row :=
  terminalProjectionIdentityPieces.flatten

theorem recursiveProjectionIdentityRows_length :
    recursiveProjectionIdentityRows.length = 10516 := by native_decide

theorem terminalProjectionIdentityRows_length :
    terminalProjectionIdentityRows.length = 60692 := by native_decide

def recursivePiCcsPieces : List (List Row) :=
  [ FPrimeFullHistoryPiCcsRecursiveAllocation.rows
  , FPrimeFullHistoryPiCcsRecursiveAuthority.rows
  , FPrimeFullHistoryRecursivePiCcsFreshDigests.rows
  , FPrimeFullHistoryRecursivePiCcsRunningAuthority.rows
  , FPrimeFullHistoryRecursivePiCcsTranscript.rows
  , FPrimeFullHistoryRecursivePiCcsFeInitial.rows
  , FPrimeFullHistoryRecursivePiCcsFeOptionalClaim.rows
  , FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows
  , FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows
  , FPrimeFullHistoryPiCcsRecursiveOutputBinding.rows
  , FPrimeFullHistoryRecursivePiCcsFeTerminal.rows
  , FPrimeFullHistoryRecursivePiCcsNcTerminal.rows
  , FPrimeFullHistoryRecursivePiCcsCatchup.rows
  , FPrimeFullHistoryRecursivePiCcsOutputMessageHashes.rows ]

def terminalPiCcsPieces : List (List Row) :=
  [ FPrimeFullHistoryPiCcsTerminalAllocation.rows
  , terminalPiCcsAuthorityRows
  , FPrimeFullHistoryTerminalPiCcsFreshDigests.rows
  , FPrimeFullHistoryTerminalPiCcsRunningAuthority.rows
  , FPrimeFullHistoryTerminalPiCcsTranscript.rows
  , FPrimeFullHistoryTerminalPiCcsFeInitial.rows
  , FPrimeFullHistoryTerminalPiCcsFeOptionalClaim.rows
  , FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows
  , FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows
  , FPrimeFullHistoryPiCcsTerminalOutputBinding.rows
  , FPrimeFullHistoryTerminalPiCcsFeTerminal.rows
  , FPrimeFullHistoryTerminalPiCcsNcTerminal.rows
  , FPrimeFullHistoryTerminalPiCcsCatchup.rows
  , FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.rows ]

def recursivePiRlcPieces : List (List Row) :=
  [ FPrimeFullHistoryRecursivePiRlcTranscriptRhos.rows
  , FPrimeFullHistoryPiRlcRecursiveShape.rows
  , FPrimeFullHistoryPiRlcRecursiveLinearFolds.rows
  , FPrimeFullHistoryRecursivePiRlcProjectionBinding.rows
  , recursiveProjectionSharedRows
  , recursiveProjectionIdentityRows ]

def terminalPiRlcPieces : List (List Row) :=
  [ FPrimeFullHistoryTerminalPiRlcTranscriptRhos.rows
  , FPrimeFullHistoryPiRlcTerminalShape.rows
  , FPrimeFullHistoryPiRlcTerminalLinearFolds.rows
  , FPrimeFullHistoryTerminalPiRlcProjectionBinding.rows
  , terminalProjectionSharedRows
  , terminalProjectionIdentityRows ]

def recursivePiCcsRows : List Row := recursivePiCcsPieces.flatten
def terminalPiCcsRows : List Row := terminalPiCcsPieces.flatten
def recursivePiRlcRows : List Row := recursivePiRlcPieces.flatten
def terminalPiRlcRows : List Row := terminalPiRlcPieces.flatten

theorem recursivePiCcsRows_length : recursivePiCcsRows.length = 320528 := by
  simp [recursivePiCcsRows, recursivePiCcsPieces,
    FPrimeFullHistoryPiCcsRecursiveAllocation.rows_length,
    FPrimeFullHistoryPiCcsRecursiveAuthority.rows_length,
    FPrimeFullHistoryPiCcsRecursiveOutputBinding.rows_length,
    FPrimeFullHistoryRecursivePiCcsFreshDigests.rows_length,
    FPrimeFullHistoryRecursivePiCcsRunningAuthority.rows_length,
    FPrimeFullHistoryRecursivePiCcsTranscript.rows_length,
    FPrimeFullHistoryRecursivePiCcsFeInitial.rows_length,
    FPrimeFullHistoryRecursivePiCcsFeOptionalClaim.rows_length,
    FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows_length,
    FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows_length,
    FPrimeFullHistoryRecursivePiCcsFeTerminal.rows_length,
    FPrimeFullHistoryRecursivePiCcsNcTerminal.rows_length,
    FPrimeFullHistoryRecursivePiCcsCatchup.rows_length,
    FPrimeFullHistoryRecursivePiCcsOutputMessageHashes.rows_length]
  native_decide

theorem terminalPiCcsRows_length : terminalPiCcsRows.length = 1602001 := by
  simp [terminalPiCcsRows, terminalPiCcsPieces, terminalPiCcsAuthorityRows_length,
    FPrimeFullHistoryPiCcsTerminalAllocation.rows_length,
    FPrimeFullHistoryPiCcsTerminalOutputBinding.rows_length,
    FPrimeFullHistoryTerminalPiCcsFreshDigests.rows_length,
    FPrimeFullHistoryTerminalPiCcsRunningAuthority.rows_length,
    FPrimeFullHistoryTerminalPiCcsTranscript.rows_length,
    FPrimeFullHistoryTerminalPiCcsFeInitial.rows_length,
    FPrimeFullHistoryTerminalPiCcsFeOptionalClaim.rows_length,
    FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows_length,
    FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows_length,
    FPrimeFullHistoryTerminalPiCcsFeTerminal.rows_length,
    FPrimeFullHistoryTerminalPiCcsNcTerminal.rows_length,
    FPrimeFullHistoryTerminalPiCcsCatchup.rows_length,
    FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.rows_length]
  native_decide

theorem recursivePiRlcRows_length : recursivePiRlcRows.length = 496739 := by
  rw [recursivePiRlcRows, List.length_flatten]
  simp only [recursivePiRlcPieces, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    recursiveProjectionSharedRows_length, recursiveProjectionIdentityRows_length,
    FPrimeFullHistoryPiRlcRecursiveShape.rows_length,
    FPrimeFullHistoryPiRlcRecursiveLinearFolds.rows_length,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.rows_length,
    FPrimeFullHistoryRecursivePiRlcProjectionBinding.rows_length]
  native_decide

theorem terminalPiRlcRows_length : terminalPiRlcRows.length = 666223 := by
  rw [terminalPiRlcRows, List.length_flatten]
  simp only [terminalPiRlcPieces, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    terminalProjectionSharedRows_length, terminalProjectionIdentityRows_length,
    FPrimeFullHistoryPiRlcTerminalShape.rows_length,
    FPrimeFullHistoryPiRlcTerminalLinearFolds.rows_length,
    FPrimeFullHistoryTerminalPiRlcTranscriptRhos.rows_length,
    FPrimeFullHistoryTerminalPiRlcProjectionBinding.rows_length]
  native_decide

theorem recursivePiCcsRows_length_matches_manifest :
    recursivePiCcsRows.length =
      (FPrimeFullHistoryManifest.recursiveNifsFamilies[0]!).rowCount := by
  simpa using recursivePiCcsRows_length

theorem recursivePiRlcRows_length_matches_manifest :
    recursivePiRlcRows.length =
      (FPrimeFullHistoryManifest.recursiveNifsFamilies[1]!).rowCount := by
  simpa using recursivePiRlcRows_length

theorem terminalPiCcsRows_length_matches_manifest :
    terminalPiCcsRows.length =
      (FPrimeFullHistoryManifest.terminalNifsFamilies[1]!).rowCount := by
  simpa using terminalPiCcsRows_length

theorem terminalPiRlcRows_length_matches_manifest :
    terminalPiRlcRows.length =
      (FPrimeFullHistoryManifest.terminalNifsFamilies[2]!).rowCount := by
  simpa using terminalPiRlcRows_length

theorem recursivePiCcs_satisfies_iff (assignment : Nat → Nat) :
    Satisfies recursivePiCcsRows assignment ↔
      ∀ rows ∈ recursivePiCcsPieces, Satisfies rows assignment := by
  exact satisfies_flatten_iff recursivePiCcsPieces assignment

theorem terminalPiCcs_satisfies_iff (assignment : Nat → Nat) :
    Satisfies terminalPiCcsRows assignment ↔
      ∀ rows ∈ terminalPiCcsPieces, Satisfies rows assignment := by
  exact satisfies_flatten_iff terminalPiCcsPieces assignment

theorem recursivePiRlc_satisfies_iff (assignment : Nat → Nat) :
    Satisfies recursivePiRlcRows assignment ↔
      ∀ rows ∈ recursivePiRlcPieces, Satisfies rows assignment := by
  exact satisfies_flatten_iff recursivePiRlcPieces assignment

theorem terminalPiRlc_satisfies_iff (assignment : Nat → Nat) :
    Satisfies terminalPiRlcRows assignment ↔
      ∀ rows ∈ terminalPiRlcPieces, Satisfies rows assignment := by
  exact satisfies_flatten_iff terminalPiRlcPieces assignment

theorem recursiveFeSumcheckRows_in_owner :
    rowsIncluded FPrimeFullHistorySumcheckArtifact.recursiveFeRows
      FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows = true := by
  native_decide

theorem recursiveNcSumcheckRows_in_owner :
    rowsIncluded FPrimeFullHistorySumcheckArtifact.recursiveNcRows
      FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows = true := by
  native_decide

theorem terminalFeSumcheckRows_in_owner :
    rowsIncluded FPrimeFullHistorySumcheckArtifact.terminalFeRows
      FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows = true := by
  native_decide

theorem terminalNcSumcheckRows_in_owner :
    rowsIncluded FPrimeFullHistorySumcheckArtifact.terminalNcRows
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows = true := by
  native_decide

theorem recursivePiCcs_feSumcheck_satisfies {assignment : Nat → Nat}
    (satisfies : Satisfies recursivePiCcsRows assignment) :
    Satisfies FPrimeFullHistorySumcheckArtifact.recursiveFeRows assignment := by
  have pieces := (recursivePiCcs_satisfies_iff assignment).mp satisfies
  have ownerRows := pieces FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows
    (by simp [recursivePiCcsPieces])
  intro row rowMember
  exact ownerRows row
    (rowsIncluded_sound recursiveFeSumcheckRows_in_owner row rowMember)

theorem recursivePiCcs_ncSumcheck_satisfies {assignment : Nat → Nat}
    (satisfies : Satisfies recursivePiCcsRows assignment) :
    Satisfies FPrimeFullHistorySumcheckArtifact.recursiveNcRows assignment := by
  have pieces := (recursivePiCcs_satisfies_iff assignment).mp satisfies
  have ownerRows := pieces FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows
    (by simp [recursivePiCcsPieces])
  intro row rowMember
  exact ownerRows row
    (rowsIncluded_sound recursiveNcSumcheckRows_in_owner row rowMember)

theorem terminalPiCcs_feSumcheck_satisfies {assignment : Nat → Nat}
    (satisfies : Satisfies terminalPiCcsRows assignment) :
    Satisfies FPrimeFullHistorySumcheckArtifact.terminalFeRows assignment := by
  have pieces := (terminalPiCcs_satisfies_iff assignment).mp satisfies
  have ownerRows := pieces FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows
    (by simp [terminalPiCcsPieces])
  intro row rowMember
  exact ownerRows row
    (rowsIncluded_sound terminalFeSumcheckRows_in_owner row rowMember)

theorem terminalPiCcs_ncSumcheck_satisfies {assignment : Nat → Nat}
    (satisfies : Satisfies terminalPiCcsRows assignment) :
    Satisfies FPrimeFullHistorySumcheckArtifact.terminalNcRows assignment := by
  have pieces := (terminalPiCcs_satisfies_iff assignment).mp satisfies
  have ownerRows := pieces FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows
    (by simp [terminalPiCcsPieces])
  intro row rowMember
  exact ownerRows row
    (rowsIncluded_sound terminalNcSumcheckRows_in_owner row rowMember)

theorem recursiveProjectionGlueRows_mem :
    FPrimeFullHistoryProjectionRoles.recursiveGlueRows ∈
      recursiveProjectionIdentityPieces := by
  exact List.mem_append_right _ (by simp)

theorem terminalProjectionGlueRows_mem :
    FPrimeFullHistoryProjectionRoles.terminalGlueRows ∈
      terminalProjectionIdentityPieces := by
  exact List.mem_append_right _ (by simp)

theorem recursivePiRlc_projectionGlue_satisfies {assignment : Nat → Nat}
    (satisfies : Satisfies recursivePiRlcRows assignment) :
    Satisfies FPrimeFullHistoryProjectionRoles.recursiveGlueRows assignment := by
  have pieces := (recursivePiRlc_satisfies_iff assignment).mp satisfies
  have identities := pieces recursiveProjectionIdentityRows
    (by simp [recursivePiRlcPieces])
  exact (satisfies_flatten_iff recursiveProjectionIdentityPieces assignment).mp
    (by simpa [recursiveProjectionIdentityRows] using identities) _
    recursiveProjectionGlueRows_mem

theorem terminalPiRlc_projectionGlue_satisfies {assignment : Nat → Nat}
    (satisfies : Satisfies terminalPiRlcRows assignment) :
    Satisfies FPrimeFullHistoryProjectionRoles.terminalGlueRows assignment := by
  have pieces := (terminalPiRlc_satisfies_iff assignment).mp satisfies
  have identities := pieces terminalProjectionIdentityRows
    (by simp [terminalPiRlcPieces])
  exact (satisfies_flatten_iff terminalProjectionIdentityPieces assignment).mp
    (by simpa [terminalProjectionIdentityRows] using identities) _
    terminalProjectionGlueRows_mem

theorem recursiveTraceRows_partition :
    ∀ trace ∈ FPrimeFullHistoryProjection.recursiveTraces,
      FPrimeFullHistoryProjection.traceRows trace =
        recursiveProjectionSharedRows ++
          (FPrimeFullHistoryProjection.traceRows trace).drop 380 := by
  native_decide

theorem terminalTraceRows_partition :
    ∀ trace ∈ FPrimeFullHistoryProjection.terminalTraces,
      FPrimeFullHistoryProjection.traceRows trace =
        terminalProjectionSharedRows ++
          (FPrimeFullHistoryProjection.traceRows trace).drop 1892 := by
  native_decide

/-- Satisfaction of the exact recursive PiRLC parent rows supplies the
projection semantics consumed by the probabilistic projection theorem. -/
theorem recursivePiRlc_projectionHolds {assignment : Nat → Nat}
    (satisfies : Satisfies recursivePiRlcRows assignment) :
    FPrimeFullHistoryProjection.RecursiveHolds assignment := by
  have pieces := (recursivePiRlc_satisfies_iff assignment).mp satisfies
  have shared : Satisfies recursiveProjectionSharedRows assignment :=
    pieces _ (by simp [recursivePiRlcPieces])
  have identities : Satisfies recursiveProjectionIdentityRows assignment :=
    pieces _ (by simp [recursivePiRlcPieces])
  have identityPieces :
      ∀ rows ∈ recursiveProjectionIdentityPieces,
        Satisfies rows assignment := by
    exact (satisfies_flatten_iff recursiveProjectionIdentityPieces assignment).mp
      (by simpa [recursiveProjectionIdentityRows] using identities)
  intro trace traceMember
  rw [recursiveTraceRows_partition trace traceMember]
  intro row rowMember
  rcases List.mem_append.mp rowMember with sharedMember | identityMember
  · exact shared row sharedMember
  · exact identityPieces _
      (List.mem_append_left _
        (List.mem_map.mpr ⟨trace, traceMember, rfl⟩)) row identityMember

/-- Terminal-fold counterpart of `recursivePiRlc_projectionHolds`. -/
theorem terminalPiRlc_projectionHolds {assignment : Nat → Nat}
    (satisfies : Satisfies terminalPiRlcRows assignment) :
    FPrimeFullHistoryProjection.TerminalHolds assignment := by
  have pieces := (terminalPiRlc_satisfies_iff assignment).mp satisfies
  have shared : Satisfies terminalProjectionSharedRows assignment :=
    pieces _ (by simp [terminalPiRlcPieces])
  have identities : Satisfies terminalProjectionIdentityRows assignment :=
    pieces _ (by simp [terminalPiRlcPieces])
  have identityPieces :
      ∀ rows ∈ terminalProjectionIdentityPieces,
        Satisfies rows assignment := by
    exact (satisfies_flatten_iff terminalProjectionIdentityPieces assignment).mp
      (by simpa [terminalProjectionIdentityRows] using identities)
  intro trace traceMember
  rw [terminalTraceRows_partition trace traceMember]
  intro row rowMember
  rcases List.mem_append.mp rowMember with sharedMember | identityMember
  · exact shared row sharedMember
  · exact identityPieces _
      (List.mem_append_left _
        (List.mem_map.mpr ⟨trace, traceMember, rfl⟩)) row identityMember

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNestedOwners
"#;
    format!("{imports}\n{extra_imports}\n\n{body}")
}
