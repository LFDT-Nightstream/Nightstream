import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext

/-! Structural validation of the exact full-layout terminal XOut context artifact. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullXOutContext

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext

structure FullValid : Prop where
  schemaVersion : rawArtifact.schemaVersion = 1
  profileId : rawArtifact.profileId =
    "nightstream/goldilocks/streaming-terminal-full-x-out-context/v1"
  sourceIdentity : rawArtifact.sourceIdentity =
    "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1"
  legacyShaNonAuthoritative : rawArtifact.sourceRowsSha256 = ""
  lifecycleScope : lifecycleScope = "recursive-terminal-arm-435"
  rowOwnership : rowStop - rowStart = rawArtifact.rowCount
  rowCount : rawArtifact.rowCount = 24
  xOutCount : rawArtifact.xOutColumns.length = 32
  sourceCounts : rawArtifact.vkFsSourceColumns.length = 4 ∧
    rawArtifact.piCcsHeaderSourceColumns.length = 4 ∧
    rawArtifact.boundarySourceColumns.length = 4 ∧
    rawArtifact.accumulatorSourceColumns.length = 4
  constantsCanonical : rawArtifact.domainTag < goldilocksP ∧
    rawArtifact.acceptedWorkItems < goldilocksP ∧
    rawArtifact.nebulaMarker < goldilocksP
  columnsInside : ∀ column ∈ rawArtifact.xOutColumns ++
      rawArtifact.vkFsSourceColumns ++ rawArtifact.piCcsHeaderSourceColumns ++
      rawArtifact.boundarySourceColumns ++ rawArtifact.accumulatorSourceColumns,
    column < rawArtifact.columnCount

theorem rawArtifact_valid : FullValid := by
  refine {
    schemaVersion := rfl
    profileId := rfl
    sourceIdentity := rfl
    legacyShaNonAuthoritative := rfl
    lifecycleScope := rfl
    rowOwnership := by decide
    rowCount := rfl
    xOutCount := rfl
    sourceCounts := by decide
    constantsCanonical := by decide
    columnsInside := by decide }

/-- Rust's canonical sparse snapshot sorts the copied input before the newly
allocated output. The builder form has the same two operands in reverse order. -/
def canonicalizeLinearRow (row : Row) : Row :=
  match row.a with
  | [] => row
  | head :: tail => ⟨tail ++ [head], row.b, row.c⟩

def canonicalRows : List Row :=
  rawArtifact.contextRows.map canonicalizeLinearRow

def Satisfied (assignment : Nat → Nat) : Prop :=
  Satisfies canonicalRows assignment

theorem canonicalRows_length : canonicalRows.length = rawArtifact.rowCount := by
  norm_num [canonicalRows, RawArtifact.contextRows, copyRows, rawArtifact]

/-- Exact Rust-emitted order of the 32 decoded terminal XOut fields. -/
theorem xOutColumns_exact :
    rawArtifact.xOutColumns =
      List.ofFn fun lane : Fin 32 => 28041899 + lane.val := by
  rfl

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullXOutContext
