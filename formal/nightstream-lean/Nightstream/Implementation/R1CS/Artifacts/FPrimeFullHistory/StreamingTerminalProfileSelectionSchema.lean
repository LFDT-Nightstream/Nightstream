import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: compact exact-row schema for terminal profile selection.

The three rows bind the terminal schedule, lifecycle, and phase selectors to
one. They do not own the selected arm's semantic rows.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfileSelection.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceArtifactIdentity : String
  lifecycleScope : String
  rowFamily : String
  rowStart : Nat
  rowStop : Nat
  columnCount : Nat
  selectorColumns : List Nat
deriving DecidableEq, Repr

def selectorRow (column : Nat) : Row :=
  ⟨[(0, goldilocksP - 1), (column, 1)], [(0, 1)], []⟩

def RawArtifact.rows (artifact : RawArtifact) : List Row :=
  artifact.selectorColumns.map selectorRow

def RawArtifact.Satisfied
    (artifact : RawArtifact) (assignment : Nat -> Nat) : Prop :=
  Satisfies artifact.rows assignment

structure RawArtifact.Valid (artifact : RawArtifact) : Prop where
  schemaVersion : artifact.schemaVersion = 1
  profileId : artifact.profileId =
    "nightstream/goldilocks/streaming-terminal-lifecycle/v1"
  sourceArtifactIdentity : artifact.sourceArtifactIdentity =
    "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1"
  lifecycleScope : artifact.lifecycleScope = "recursive-terminal-arm-435"
  rowFamily : artifact.rowFamily = "terminal.streaming.profile_selection"
  rowCount : artifact.rowStop - artifact.rowStart = 3
  selectorCount : artifact.selectorColumns.length = 3
  selectorsInside :
    artifact.selectorColumns.getD 0 0 < artifact.columnCount /\
      artifact.selectorColumns.getD 1 0 < artifact.columnCount /\
      artifact.selectorColumns.getD 2 0 < artifact.columnCount

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfileSelection.Artifact
