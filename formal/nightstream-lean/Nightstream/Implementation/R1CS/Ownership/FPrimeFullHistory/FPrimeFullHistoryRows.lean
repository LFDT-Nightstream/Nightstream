import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryManifest
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryNestedOwners
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryBaseArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveTranscriptArtifact
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPriorLinkArtifact
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCounterSound
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveOutputArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStateLinkArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalTranscriptArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursivePointBindingArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPointBindingArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalRunningLinkArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalParentLinkArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalLinkArtifact
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorArtifact
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityArtifact
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPublicPinsArtifact
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalCeArtifact

/-!
Contract: the exact manifest-ordered sparse-row list for the supported
plain/stateless `[1,1]` full-history artifact.

This is the missing structural bridge between the Rust-generated 4,193,134-row
artifact and the semantic owner theorems.  The recursive counter is handled
carefully: its first 138 rows occur in the recursive prelude, while only its
remaining 522 rows are emitted by the later counter owner.  No row is inserted
twice in `fullRows`.

Hashes remain drift metadata.  Semantic authority comes from satisfaction of
the reconstructed sparse rows below.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRows

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576

/-- The counter compiler spans two manifest owners. -/
def counterInputRows : List Row :=
  FPrimeFullHistoryCounterSound.globalRows.take
    FPrimeFullHistoryCounter.inputRowCount

def counterTransitionRows : List Row :=
  FPrimeFullHistoryCounterSound.globalRows.drop
    FPrimeFullHistoryCounter.inputRowCount

theorem counterInputRows_length : counterInputRows.length = 138 := by
  native_decide

theorem counterTransitionRows_length : counterTransitionRows.length = 522 := by
  native_decide

theorem counterInputRows_in_prelude :
    rowsIncluded counterInputRows FPrimeFullHistoryRecursivePrelude.rows = true := by
  native_decide

theorem counterRows_partition :
    counterInputRows ++ counterTransitionRows =
      FPrimeFullHistoryCounterSound.globalRows := by
  exact List.take_append_drop _ _

/-- Exact recursive NIFS parent, in production emission order. -/
def recursiveNifsPieces : List (List Row) :=
  [ FPrimeFullHistoryNestedOwners.recursivePiCcsRows
  , FPrimeFullHistoryNestedOwners.recursivePiRlcRows
  , FPrimeFullHistoryPiDec.recursiveRows
  , FPrimeFullHistoryRecursivePointBinding.rows ]

def recursiveNifsRows : List Row := recursiveNifsPieces.flatten

theorem recursiveNifsRows_length : recursiveNifsRows.length = 827866 := by
  simp [recursiveNifsRows, recursiveNifsPieces,
    FPrimeFullHistoryNestedOwners.recursivePiCcsRows_length,
    FPrimeFullHistoryNestedOwners.recursivePiRlcRows_length,
    FPrimeFullHistoryPiDec.recursiveRows_length,
    FPrimeFullHistoryRecursivePointBinding.rows_length,
    FPrimeFullHistoryPiDec.rowCount,
    FPrimeFullHistoryRecursivePointBinding.rowCount]

/-- Exact recursive top-level owner. -/
def recursivePieces : List (List Row) :=
  [ FPrimeFullHistoryRecursivePrelude.rows
  , FPrimeFullHistoryRecursiveTranscriptArtifact.ownerRows
  , recursiveNifsRows
  , FPrimeFullHistoryPriorLink.rows
  , FPrimeFullHistoryRecursiveAccumulator.rows
  , counterTransitionRows
  , FPrimeFullHistoryRecursiveOutput.rows ]

def recursiveRows : List Row := recursivePieces.flatten

theorem recursiveRows_length : recursiveRows.length = 1112745 := by
  simp [recursiveRows, recursivePieces, recursiveNifsRows_length,
    counterTransitionRows_length,
    FPrimeFullHistoryRecursivePrelude.rows_length,
    FPrimeFullHistoryRecursivePrelude.rowCount,
    FPrimeFullHistoryRecursiveTranscriptArtifact.ownerRows_length,
    FPrimeFullHistoryRecursiveTranscriptArtifact.rowCount,
    FPrimeFullHistoryPriorLink.rows_length,
    FPrimeFullHistoryPriorLink.rowCount,
    FPrimeFullHistoryRecursiveAccumulator.rows_length,
    FPrimeFullHistoryRecursiveAccumulator.rowCount,
    FPrimeFullHistoryRecursiveOutput.rows_length,
    FPrimeFullHistoryRecursiveOutput.rowCount]

/-- Exact terminal NIFS parent, in production emission order. -/
def terminalNifsPieces : List (List Row) :=
  [ FPrimeFullHistoryTerminalTranscriptArtifact.ownerRows
  , FPrimeFullHistoryNestedOwners.terminalPiCcsRows
  , FPrimeFullHistoryNestedOwners.terminalPiRlcRows
  , FPrimeFullHistoryPiDec.terminalRows
  , FPrimeFullHistoryTerminalPointBinding.rows ]

def terminalNifsRows : List Row := terminalNifsPieces.flatten

theorem terminalNifsRows_length : terminalNifsRows.length = 2278831 := by
  simp [terminalNifsRows, terminalNifsPieces,
    FPrimeFullHistoryTerminalTranscriptArtifact.ownerRows_length,
    FPrimeFullHistoryTerminalTranscriptArtifact.rowCount,
    FPrimeFullHistoryNestedOwners.terminalPiCcsRows_length,
    FPrimeFullHistoryNestedOwners.terminalPiRlcRows_length,
    FPrimeFullHistoryPiDec.terminalRows_length,
    FPrimeFullHistoryTerminalPointBinding.rows_length,
    FPrimeFullHistoryPiDec.rowCount,
    FPrimeFullHistoryTerminalPointBinding.rowCount]

/-- Exact terminal-fold top-level owner. -/
def terminalPieces : List (List Row) :=
  [ terminalNifsRows
  , FPrimeFullHistoryTerminalRunningLink.rows
  , FPrimeFullHistoryTerminalParentLink.rows
  , FPrimeFullHistoryTerminalLink.rows
  , FPrimeFullHistoryTerminalAccumulator.rows ]

def terminalRows : List Row := terminalPieces.flatten

theorem terminalRows_length : terminalRows.length = 2549400 := by
  simp [terminalRows, terminalPieces, terminalNifsRows_length,
    FPrimeFullHistoryTerminalRunningLink.rows_length,
    FPrimeFullHistoryTerminalParentLink.rows_length,
    FPrimeFullHistoryTerminalLink.rows_length,
    FPrimeFullHistoryTerminalAccumulator.rows_length,
    FPrimeFullHistoryTerminalRunningLink.rowCount,
    FPrimeFullHistoryTerminalParentLink.rowCount,
    FPrimeFullHistoryTerminalLink.rowCount,
    FPrimeFullHistoryTerminalAccumulator.rowCount]

theorem terminalCeRows_length :
    FPrimeFullHistoryTerminalCe.terminalCeRows.length = 301588 := by
  simp [FPrimeFullHistoryTerminalCe.terminalCeRows,
    FPrimeFullHistoryTerminalCe.claimRows,
    FPrimeFullHistoryTerminalCe.columnMaps,
    FPrimeFullHistoryTerminalCe.rows_length]

/-- The seven top-level manifest owners as actual sparse-row lists. -/
def topLevelPieces : List (List Row) :=
  [ FPrimeFullHistoryBase.rows
  , recursiveRows
  , FPrimeFullHistoryStateLink.rows
  , terminalRows
  , FPrimeFullHistoryTerminalContinuity.rows
  , FPrimeFullHistoryPublicPins.rows
  , FPrimeFullHistoryTerminalCe.terminalCeRows ]

def fullRows : List Row := topLevelPieces.flatten

theorem fullRows_length :
    fullRows.length = FPrimeFullHistoryManifest.totalRows := by
  simp [fullRows, topLevelPieces, recursiveRows_length, terminalRows_length,
    terminalCeRows_length, FPrimeFullHistoryBase.rows_length,
    FPrimeFullHistoryStateLink.rows_length,
    FPrimeFullHistoryTerminalContinuity.rows_length,
    FPrimeFullHistoryPublicPins.rows_length,
    FPrimeFullHistoryManifest.totalRows,
    FPrimeFullHistoryBase.rowCount,
    FPrimeFullHistoryStateLink.rowCount,
    FPrimeFullHistoryTerminalContinuity.rowCount,
    FPrimeFullHistoryPublicPins.rowCount]

theorem recursiveNifs_satisfies_iff (assignment : Nat → Nat) :
    Satisfies recursiveNifsRows assignment ↔
      ∀ rows ∈ recursiveNifsPieces, Satisfies rows assignment :=
  satisfies_flatten_iff recursiveNifsPieces assignment

theorem recursive_satisfies_iff (assignment : Nat → Nat) :
    Satisfies recursiveRows assignment ↔
      ∀ rows ∈ recursivePieces, Satisfies rows assignment :=
  satisfies_flatten_iff recursivePieces assignment

theorem terminalNifs_satisfies_iff (assignment : Nat → Nat) :
    Satisfies terminalNifsRows assignment ↔
      ∀ rows ∈ terminalNifsPieces, Satisfies rows assignment :=
  satisfies_flatten_iff terminalNifsPieces assignment

theorem terminal_satisfies_iff (assignment : Nat → Nat) :
    Satisfies terminalRows assignment ↔
      ∀ rows ∈ terminalPieces, Satisfies rows assignment :=
  satisfies_flatten_iff terminalPieces assignment

theorem full_satisfies_iff (assignment : Nat → Nat) :
    Satisfies fullRows assignment ↔
      ∀ rows ∈ topLevelPieces, Satisfies rows assignment :=
  satisfies_flatten_iff topLevelPieces assignment

/-- Recover satisfaction of all 660 counter rows from their two exact owners. -/
theorem counter_satisfies_of_prelude_and_transition
    {assignment : Nat → Nat}
    (prelude : Satisfies FPrimeFullHistoryRecursivePrelude.rows assignment)
    (transition : Satisfies counterTransitionRows assignment) :
    Satisfies FPrimeFullHistoryCounterSound.globalRows assignment := by
  rw [← counterRows_partition]
  apply (satisfies_flatten_iff [counterInputRows, counterTransitionRows]
    assignment).mpr
  intro rows member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · intro row rowMember
    exact prelude row (rowsIncluded_sound counterInputRows_in_prelude row rowMember)
  · exact transition

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRows
