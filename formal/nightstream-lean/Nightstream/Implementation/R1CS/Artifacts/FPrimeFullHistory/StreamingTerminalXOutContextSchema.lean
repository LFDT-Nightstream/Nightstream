import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: compact exact-row schema for the terminal XOut context family.

Rust compares all 24 source rows with `contextRows`. This schema owns only
the verifier-derived XOut context bindings. It does not own either retained
Poseidon2 family or terminal acceptance.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceIdentity : String
  sourceRowsSha256 : String
  rowCount : Nat
  columnCount : Nat
  domainTag : Nat
  acceptedWorkItems : Nat
  nebulaMarker : Nat
  baselineChangedValue : Nat
  mutatedChangedValue : Nat
  xOutColumns : List Nat
  vkFsSourceColumns : List Nat
  piCcsHeaderSourceColumns : List Nat
  boundarySourceColumns : List Nat
  accumulatorSourceColumns : List Nat
deriving DecidableEq, Repr

def copyRows (outputs inputs : List Nat) : List Row :=
  List.zipWith (fun output input => builderLinearRow output [(input, 1)])
    outputs inputs

def RawArtifact.contextRows (artifact : RawArtifact) : List Row :=
  [builderLinearRow (artifact.xOutColumns.getD 0 0)
      [(0, artifact.domainTag)]] ++
    copyRows ((artifact.xOutColumns.drop 1).take 4)
      artifact.vkFsSourceColumns ++
    copyRows ((artifact.xOutColumns.drop 5).take 4)
      artifact.piCcsHeaderSourceColumns ++
    [builderLinearRow (artifact.xOutColumns.getD 9 0)
        [(0, artifact.acceptedWorkItems)],
      builderLinearRow (artifact.xOutColumns.getD 10 0) [],
      builderLinearRow (artifact.xOutColumns.getD 11 0)
        [(0, artifact.acceptedWorkItems)],
      builderLinearRow (artifact.xOutColumns.getD 12 0) [],
      builderLinearRow (artifact.xOutColumns.getD 13 0) [(0, 1)],
      builderLinearRow (artifact.xOutColumns.getD 14 0) []] ++
    copyRows ((artifact.xOutColumns.drop 15).take 4)
      artifact.boundarySourceColumns ++
    copyRows ((artifact.xOutColumns.drop 23).take 4)
      artifact.accumulatorSourceColumns ++
    [builderLinearRow (artifact.xOutColumns.getD 27 0)
      [(0, artifact.nebulaMarker)]]

def RawArtifact.Satisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.contextRows assignment

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 1 ∧
    artifact.profileId =
      "nightstream/goldilocks/streaming-terminal-x-out-context/v1" ∧
    artifact.sourceIdentity = "rust:streaming-terminal-x-out-context/v1" ∧
    artifact.sourceRowsSha256 =
      "89aae9a5eb9aa1f455cb97d60b648c7fdd03d729935d6d6cc87fe5419773173d" ∧
    artifact.rowCount = 24 ∧
    artifact.xOutColumns.length = 32 ∧
    artifact.vkFsSourceColumns.length = 4 ∧
    artifact.piCcsHeaderSourceColumns.length = 4 ∧
    artifact.boundarySourceColumns.length = 4 ∧
    artifact.accumulatorSourceColumns.length = 4 ∧
    artifact.domainTag < goldilocksP ∧
    artifact.acceptedWorkItems < goldilocksP ∧
    artifact.nebulaMarker < goldilocksP ∧
    artifact.baselineChangedValue < goldilocksP ∧
    artifact.mutatedChangedValue < goldilocksP ∧
    artifact.baselineChangedValue ≠ artifact.mutatedChangedValue

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact
