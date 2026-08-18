import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: compact final-link schema for the terminal Nebula-state-digest family.

Rust owns the full 19,353-row family. This schema owns its four final links and
the exact source-row location needed for a checked removal counterexample.

Assurance tier: model-level.

Emits constraints: no. It describes four emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

def digestFields : Nat := 4
def familyRows : Nat := 19353

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceIdentity : String
  sourceRowsSha256 : String
  rowCount : Nat
  columnCount : Nat
  sourceRowStart : Nat
  finalRowStart : Nat
  hashOutputColumns : List Nat
  xOutStateColumns : List Nat
  baselineDigestValue : Nat
  equalityRowStart : Nat
  selectedSourceRow : Nat
deriving DecidableEq, Repr

def RawArtifact.equalityRows (artifact : RawArtifact) : List Row :=
  (List.range digestFields).map fun lane =>
    builderLinearRow (artifact.xOutStateColumns.getD lane 0)
      [(artifact.hashOutputColumns.getD lane 0, 1)]

def RawArtifact.LinkSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.equalityRows assignment

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 1 ∧
    artifact.profileId =
      "nightstream/goldilocks/streaming-terminal-nebula-state-digest-link/v1" ∧
    artifact.sourceIdentity =
      "rust:streaming-terminal-nebula-state-digest-link/v1" ∧
    artifact.sourceRowsSha256.length = 64 ∧
    artifact.rowCount = familyRows ∧
    artifact.columnCount = 23087 ∧
    artifact.sourceRowStart = 3660 ∧
    artifact.finalRowStart = artifact.sourceRowStart ∧
    artifact.hashOutputColumns.length = digestFields ∧
    (∀ column ∈ artifact.hashOutputColumns, column < artifact.columnCount) ∧
    artifact.xOutStateColumns = [29, 30, 31, 32] ∧
    artifact.baselineDigestValue < goldilocksP ∧
    artifact.equalityRowStart + digestFields = artifact.rowCount ∧
    artifact.selectedSourceRow =
      artifact.sourceRowStart + artifact.equalityRowStart ∧
    artifact.selectedSourceRow = 23009

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact
