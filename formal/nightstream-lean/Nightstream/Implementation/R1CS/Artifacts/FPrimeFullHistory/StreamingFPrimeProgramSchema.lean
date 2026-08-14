/-!
Wire schema for the compact Rust streaming F-prime program artifact.

Owns proof-free phase runs and executable geometry checks.

Does not own phase meanings, phase-local constraints, lifecycle semantics,
relation dimensions, Rust conformance, or security reduction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact

structure RawRun where
  phaseCode : Nat
  firstIndex : Nat
  count : Nat
  deriving DecidableEq, Inhabited, Repr

def RawRun.expand (run : RawRun) : List (Nat × Nat) :=
  (List.range run.count).map fun offset =>
    (run.phaseCode, run.firstIndex + offset)

structure RawProgram where
  schemaVersion : Nat
  stateChunkFields : Nat
  priorStateFrameFields : Nat
  priorStateChunks : Nat
  claimFrameFields : Nat
  claimChunkFields : Nat
  claimChunks : Nat
  piCcsRounds : Nat
  piRlcFamilies : Nat
  successorPrefixFrameFields : Nat
  successorPrefixChunks : Nat
  workItemCount : Nat
  runs : List RawRun
  deriving DecidableEq, Repr

def RawProgram.expanded (raw : RawProgram) : List (Nat × Nat) :=
  raw.runs.flatMap RawRun.expand

def RawRun.valid (run : RawRun) : Bool :=
  run.phaseCode < 19 && 0 < run.count

def ProgramValid (raw : RawProgram) : Prop :=
  raw.schemaVersion = 2 /\
    0 < raw.stateChunkFields /\
    0 < raw.priorStateChunks /\
    (raw.priorStateChunks - 1) * raw.stateChunkFields <
      raw.priorStateFrameFields /\
    raw.priorStateFrameFields <=
      raw.priorStateChunks * raw.stateChunkFields /\
    0 < raw.claimChunkFields /\
    0 < raw.claimChunks /\
    (raw.claimChunks - 1) * raw.claimChunkFields < raw.claimFrameFields /\
    raw.claimFrameFields <= raw.claimChunks * raw.claimChunkFields /\
    0 < raw.successorPrefixChunks /\
    (raw.successorPrefixChunks - 1) * raw.stateChunkFields <
      raw.successorPrefixFrameFields /\
    raw.successorPrefixFrameFields <=
      raw.successorPrefixChunks * raw.stateChunkFields /\
    raw.runs.all RawRun.valid = true /\
    raw.expanded.length = raw.workItemCount

instance programValidDecidable (raw : RawProgram) :
    Decidable (ProgramValid raw) := by
  unfold ProgramValid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact
