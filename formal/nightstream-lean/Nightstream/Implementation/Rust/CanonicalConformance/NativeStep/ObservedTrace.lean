import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Core

/-!
Contract: proof-free observations from one instrumented native F' step.

This module types the public call-site data recorded by Rust.  It owns only
structural encodings and event names.  In particular, a transcript snapshot is
an opaque observed value: retaining it beside the exact prefix does not prove
Poseidon2 correctness.  Likewise, an observed NIFS result records the call
result without proving NIFS soundness.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

/-- Canonical unsigned representative of one exported Goldilocks element. -/
abbrev RawField := Nat

def goldilocksModulus : Nat :=
  18446744069414584321

def twoPow32 : Nat :=
  4294967296

def rawFieldOfNat (value : Nat) : RawField :=
  value % goldilocksModulus

/-- Rust encodes each `u64` as low and high 32-bit field limbs. -/
def u64Halves (value : Nat) : List RawField :=
  [value % twoPow32, value / twoPow32]

/-- Typed values whose concrete four-field encodings occur in native calls. -/
inductive RawEncodingKey where
  | digest (value : Digest)
  | header (value : Header)
  | nebula (value : Nebula)
  | nebulaDigest (value : NebulaDigest)
deriving Repr, DecidableEq

structure RawEncodingEntry where
  key : RawEncodingKey
  fields : List RawField
deriving Repr, DecidableEq

abbrev RawEncodingTable := List RawEncodingEntry

def lookupRawFields
    (table : RawEncodingTable)
    (key : RawEncodingKey) : List RawField :=
  match table with
  | [] => []
  | entry :: rest =>
      if entry.key = key then entry.fields else lookupRawFields rest key

def RawFieldsWellFormed (fields : List RawField) : Bool :=
  fields.all fun field => decide (field < goldilocksModulus)

def RawEncodingTableWellFormed (table : RawEncodingTable) : Bool :=
  decide ((table.map fun entry => entry.key).Nodup) &&
  table.all fun entry =>
    decide (entry.fields.length = 4) &&
    RawFieldsWellFormed entry.fields

/-- Closed labels prevent a renderer from silently accepting a new domain or
reordered label spelling. -/
inductive TranscriptLabel where
  | fPrimeStep
  | vkFs
  | piCcsHeader
  | chunkCountIn
  | stepCountIn
  | z0
  | ziIn
  | pc
  | semanticStateIn
  | accDigestIn
  | publicTraceIn
  | nebulaLaneIn
  | chunkDigest
deriving Repr, DecidableEq

structure TranscriptAppend where
  label : TranscriptLabel
  fields : List RawField
deriving Repr, DecidableEq

/-- Exact snapshot returned by the transcript immediately before NIFS.  Its
contents are retained, not recomputed as a Poseidon2 claim. -/
structure TranscriptSnapshot where
  state : List RawField
  /-- Opaque sponge cursor retained verbatim.  This layer deliberately places
  no arithmetic or rate claim on it. -/
  absorbed : Nat
deriving Repr, DecidableEq

structure TranscriptReceipt where
  label : TranscriptLabel
  orderedAppends : List TranscriptAppend
  prefixSnapshot : TranscriptSnapshot
deriving Repr, DecidableEq

inductive EventKind where
  | chunkDigest
  | dispatch
  | transcriptStarted
  | transcriptAppend
  | transcriptPrefix
  | nifsVerify
  | runningDigest
  | stateAdvanced
  | verifierDigestRead
  | piCcsHeaderRead
  | nebulaDigest
  | stateXOutHash
deriving Repr, DecidableEq

inductive ExecutionStage where
  | entry
  | chunkDigest
  | dispatch
  | nifs
  | nebula
  | advance
  | semantic
  | xOut
  | complete
deriving Repr, DecidableEq

structure ChunkDigestCall where
  startIndex : Nat
  orderedClaims : List Fresh
  output : Digest
deriving Repr, DecidableEq

structure NifsCallReceipt where
  running : Running
  fresh : List Fresh
  proof : NifsProof
  /-- `none` is the observed rejecting result; it is not a soundness claim. -/
  outcome : Option Running
deriving Repr, DecidableEq

structure RunningDigestCall where
  running : Running
  relationColumns : Nat
  output : Digest
deriving Repr, DecidableEq

structure NebulaDigestCall where
  input : Nebula
  output : NebulaDigest
deriving Repr, DecidableEq

structure StateXOutHashCall where
  rawPreimage : List RawField
  outputDigest : Digest
  /-- Digest reconstructed from the observed `EncInst` output bits. -/
  output : Digest
deriving Repr, DecidableEq

/-- Exact singleton calls and their source order for one native execution. -/
structure ObservedTrace where
  executionOrder : List EventKind
  chunkDigest : Option ChunkDigestCall
  transcript : Option TranscriptReceipt
  nifsCall : Option NifsCallReceipt
  runningDigest : Option RunningDigestCall
  advancedState : Option NativeState
  verifierDigestRead : Option Digest
  piCcsHeaderRead : Option Header
  nebulaDigest : Option NebulaDigestCall
  stateXOutHash : Option StateXOutHashCall
  finalStage : ExecutionStage
deriving Repr, DecidableEq

private def digestFields
    (table : RawEncodingTable)
    (digest : Digest) : List RawField :=
  lookupRawFields table (.digest digest)

private def headerFields
    (table : RawEncodingTable)
    (header : Header) : List RawField :=
  lookupRawFields table (.header header)

private def nebulaFields
    (table : RawEncodingTable)
    (nebula : Nebula) : List RawField :=
  lookupRawFields table (.nebula nebula)

private def nebulaDigestFields
    (table : RawEncodingTable)
    (digest : NebulaDigest) : List RawField :=
  lookupRawFields table (.nebulaDigest digest)

/-- The exact pre-NIFS native append sequence.  This establishes label,
ordering, and field dataflow only; it does not recompute the snapshot. -/
def expectedTranscriptAppends
    (table : RawEncodingTable)
    (vkFsDigest : Digest)
    (header : Header)
    (prior : NativeState)
    (chunk : Digest) : List TranscriptAppend :=
  [
    ⟨.vkFs, digestFields table vkFsDigest⟩,
    ⟨.piCcsHeader, headerFields table header⟩,
    ⟨.chunkCountIn, [rawFieldOfNat prior.chunkCount]⟩,
    ⟨.stepCountIn, [rawFieldOfNat prior.stepCount]⟩,
    ⟨.z0, digestFields table prior.z0⟩,
    ⟨.ziIn, digestFields table prior.zi⟩,
    ⟨.pc, [rawFieldOfNat prior.pc]⟩,
    ⟨.semanticStateIn, digestFields table prior.semanticState⟩,
    ⟨.accDigestIn, digestFields table prior.accumulatorDigest⟩,
    ⟨.publicTraceIn, digestFields table prior.publicTrace⟩
  ] ++
  (match prior.nebula with
    | none => []
    | some lane => [⟨.nebulaLaneIn, nebulaFields table lane⟩]) ++
  [⟨.chunkDigest, digestFields table chunk⟩]

private def optionalDigestFields
    (table : RawEncodingTable) : Option Digest → List RawField
  | none => []
  | some digest => digestFields table digest

private def optionalNebulaDigestFields
    (table : RawEncodingTable) : Option NebulaDigest → List RawField
  | none => []
  | some digest =>
      [rawFieldOfNat 0x4e424c41] ++ nebulaDigestFields table digest

/-- Pure encoder for the exact field vector passed to Rust's state-output
hash.  Hash evaluation itself remains opaque. -/
def encodeStateXOutPreimage
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    List RawField :=
  [rawFieldOfNat 0x4e460002] ++
  digestFields table preimage.vkFsDigest ++
  headerFields table preimage.piCcsHeader ++
  u64Halves preimage.chunkCount ++
  u64Halves preimage.stepCount ++
  u64Halves preimage.pc ++
  digestFields table preimage.currentBoundary ++
  optionalDigestFields table preimage.semanticState ++
  digestFields table preimage.construction2Accumulator ++
  optionalNebulaDigestFields table preimage.nebula

/-- Public definitional expansion used by source-program refinements. -/
theorem encodeStateXOutPreimage_expansion
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    encodeStateXOutPreimage table preimage =
      [rawFieldOfNat 0x4e460002] ++
      lookupRawFields table (.digest preimage.vkFsDigest) ++
      lookupRawFields table (.header preimage.piCcsHeader) ++
      u64Halves preimage.chunkCount ++
      u64Halves preimage.stepCount ++
      u64Halves preimage.pc ++
      lookupRawFields table (.digest preimage.currentBoundary) ++
      (match preimage.semanticState with
        | none => []
        | some digest => lookupRawFields table (.digest digest)) ++
      lookupRawFields table
        (.digest preimage.construction2Accumulator) ++
      (match preimage.nebula with
        | none => []
        | some digest =>
            [rawFieldOfNat 0x4e424c41] ++
              lookupRawFields table (.nebulaDigest digest)) := by
  rfl

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
