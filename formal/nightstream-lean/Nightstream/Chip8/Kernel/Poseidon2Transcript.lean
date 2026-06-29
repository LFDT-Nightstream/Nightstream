import Nightstream.Chip8.Kernel.Root0Preimage
import SuperNeo.Primitives.Field

/-!
Owns the exact Rust-compatible packing semantics for the CHIP-8
Poseidon2-over-Goldilocks transcript. This file fixes transcript initialization,
message packing, `u64` packing, and the absorb-word view consumed by `root0`
and later challenge/digest owners. It does not own the Poseidon2 permutation
round function itself.
-/

namespace Nightstream.Chip8.Poseidon2Transcript

open Nightstream.Chip8
open Nightstream.Chip8.MetaPubEncoding
open Nightstream.Chip8.Root0Preimage

abbrev FieldElem := SuperNeo.F
abbrev Byte := UInt8

def poseidon2AppDomain : String := "neo/transcript/v1|poseidon2-goldilocks-w8-r4"
def simpleKernelTranscriptAppLabel : String := "neo.fold.next/chip8/simple_kernel"
def transcriptSeedLabel : String := "chip8/kernel/transcript_seed"
def challengeLabelLabel : String := "chal/label"
def domainGateWord : Nat := 1

def utf8Bytes (s : String) : List Byte :=
  s.toUTF8.data.toList

def littleEndianNat : List Byte → Nat
  | [] => 0
  | b :: bs => b.toNat + 256 * littleEndianNat bs

def packBytesWordsAux : Nat → List Byte → List Nat
  | 0, _ => []
  | _ + 1, [] => []
  | n + 1, bytes@(_ :: _) => littleEndianNat (bytes.take 7) :: packBytesWordsAux n (bytes.drop 7)

def packBytesWords (bytes : List Byte) : List Nat :=
  packBytesWordsAux bytes.length bytes

def absorbPackedBytesWithLenWords (bytes : List Byte) : List Nat :=
  bytes.length :: packBytesWords bytes

def u64LoWord (value : Nat) : Nat :=
  value % 4294967296

def u64HiWord (value : Nat) : Nat :=
  value / 4294967296

def splitU64Words (value : Nat) : List Nat :=
  [u64LoWord value, u64HiWord value]

def u64PayloadWords : List Nat → List Nat
  | [] => []
  | value :: values => splitU64Words value ++ u64PayloadWords values

def appendMessageWords (label : String) (msg : List Byte) : List Nat :=
  absorbPackedBytesWithLenWords (utf8Bytes label) ++
    absorbPackedBytesWithLenWords msg

def appendU64Words (label : String) (values : List Nat) : List Nat :=
  absorbPackedBytesWithLenWords (utf8Bytes label) ++
    [values.length] ++ u64PayloadWords values

def toFieldElems (words : List Nat) : List FieldElem :=
  words.map SuperNeo.F.ofNat

inductive TranscriptOp where
  | appendMessage (label : String) (msg : List Byte)
  | appendU64s (label : String) (values : List Nat)
  | digest32
  | challengeField (label : String)
deriving DecidableEq, Repr

def transcriptOpLabel? : TranscriptOp → Option String
  | .appendMessage label _ => some label
  | .appendU64s label _ => some label
  | .digest32 => none
  | .challengeField _ => some challengeLabelLabel

def transcriptOpAbsorbWords : TranscriptOp → List Nat
  | .appendMessage label msg => appendMessageWords label msg
  | .appendU64s label values => appendU64Words label values
  | .digest32 => [domainGateWord]
  | .challengeField label => appendMessageWords challengeLabelLabel (utf8Bytes label) ++ [domainGateWord]

def transcriptOpsAbsorbWords : List TranscriptOp → List Nat
  | [] => []
  | op :: ops => transcriptOpAbsorbWords op ++ transcriptOpsAbsorbWords ops

def transcriptOpsAbsorbFields (ops : List TranscriptOp) : List FieldElem :=
  toFieldElems (transcriptOpsAbsorbWords ops)

def encodeMetaAbsorbOp
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte) :
    Root0MetaAbsorbOp Digest RootParamsId → TranscriptOp
  | .absorbU64s label values => .appendU64s label values
  | .absorbDigest label value => .appendMessage label (encodeDigest value)
  | .absorbRootParamsId label value => .appendMessage label (encodeRootParamsId value)

def encodeRoot0PreimageOp
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte) :
    Root0PreimageOp CommitmentDigest Digest RootParamsId → TranscriptOp
  | .absorbCommitment label _ digest => .appendMessage label (encodeCommitmentDigest digest)
  | .absorbMeta op => encodeMetaAbsorbOp encodeDigest encodeRootParamsId op

@[simp] theorem encodeMetaAbsorbOp_label?
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    (op : Root0MetaAbsorbOp Digest RootParamsId) :
    transcriptOpLabel? (encodeMetaAbsorbOp encodeDigest encodeRootParamsId op) =
      some (root0MetaAbsorbOpLabel op) := by
  cases op <;> rfl

@[simp] theorem encodeRoot0PreimageOp_label?
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    (op : Root0PreimageOp CommitmentDigest Digest RootParamsId) :
    transcriptOpLabel? (encodeRoot0PreimageOp encodeCommitmentDigest encodeDigest encodeRootParamsId op) =
      some (root0PreimageOpLabel op) := by
  cases op with
  | absorbCommitment label id digest =>
      simp [encodeRoot0PreimageOp, transcriptOpLabel?, root0PreimageOpLabel]
  | absorbMeta metaOp =>
      cases metaOp <;> simp [encodeRoot0PreimageOp, encodeMetaAbsorbOp, transcriptOpLabel?,
        root0PreimageOpLabel, root0MetaAbsorbOpLabel]

def root0TranscriptPreludeOps (transcriptSeed : List Byte) : List TranscriptOp :=
  [ .appendMessage poseidon2AppDomain (utf8Bytes simpleKernelTranscriptAppLabel)
  , .appendMessage transcriptSeedLabel transcriptSeed
  ]

def root0TranscriptOps
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    (transcriptSeed : List Byte)
    (bindings : List (Root0CommitmentDigestBinding CommitmentDigest))
    (pubMeta : KernelMetaPub Digest RootParamsId) :
    List TranscriptOp :=
  root0TranscriptPreludeOps transcriptSeed ++
    (root0Preimage bindings pubMeta).map
      (encodeRoot0PreimageOp encodeCommitmentDigest encodeDigest encodeRootParamsId)

def root0DigestOps
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    (transcriptSeed : List Byte)
    (bindings : List (Root0CommitmentDigestBinding CommitmentDigest))
    (pubMeta : KernelMetaPub Digest RootParamsId) :
    List TranscriptOp :=
  root0TranscriptOps encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta ++
    [.digest32]

def root0DigestInputWords
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    (transcriptSeed : List Byte)
    (bindings : List (Root0CommitmentDigestBinding CommitmentDigest))
    (pubMeta : KernelMetaPub Digest RootParamsId) :
    List Nat :=
  transcriptOpsAbsorbWords
    (root0DigestOps encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta)

def root0DigestInputFields
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    (transcriptSeed : List Byte)
    (bindings : List (Root0CommitmentDigestBinding CommitmentDigest))
    (pubMeta : KernelMetaPub Digest RootParamsId) :
    List FieldElem :=
  toFieldElems
    (root0DigestInputWords
      encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta)

@[simp] theorem packBytesWords_nil :
    packBytesWords [] = [] := by
  simp [packBytesWords, packBytesWordsAux]

theorem absorbPackedBytesWithLenWords_nonempty (bytes : List Byte) :
    (absorbPackedBytesWithLenWords bytes).length > 0 := by
  simp [absorbPackedBytesWithLenWords]

theorem appendMessageWords_prefix (label : String) (msg : List Byte) :
    (appendMessageWords label msg).take 1 = [utf8Bytes label |>.length] := by
  simp [appendMessageWords, absorbPackedBytesWithLenWords]

@[simp] theorem root0TranscriptPrelude_labels (transcriptSeed : List Byte) :
    (root0TranscriptPreludeOps transcriptSeed).filterMap transcriptOpLabel? =
      [poseidon2AppDomain, transcriptSeedLabel] := by
  simp [root0TranscriptPreludeOps, transcriptOpLabel?]

theorem root0TranscriptOps_labels_of_conform
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    {bindings : List (Root0CommitmentDigestBinding CommitmentDigest)}
    {pubMeta : KernelMetaPub Digest RootParamsId}
    (transcriptSeed : List Byte)
    (hConform : root0CommitmentDigestBindingsConform bindings) :
    (root0TranscriptOps
        encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta).filterMap
        transcriptOpLabel? =
      [poseidon2AppDomain, transcriptSeedLabel] ++
        root0MetaPubLabels.take 3 ++ root0CommitmentLabels ++ root0MetaPubLabels.drop 3 := by
  have hEncoded :
      ((root0Preimage bindings pubMeta).map
          (encodeRoot0PreimageOp encodeCommitmentDigest encodeDigest encodeRootParamsId)).filterMap
          transcriptOpLabel? =
        (root0Preimage bindings pubMeta).map root0PreimageOpLabel := by
    induction root0Preimage bindings pubMeta with
    | nil =>
        simp
    | cons op ops ih =>
        cases op with
        | absorbCommitment label id digest =>
            simp [ih]
        | absorbMeta op =>
            cases op <;> simp [ih]
  simp [root0TranscriptOps, root0TranscriptPrelude_labels, hEncoded,
    root0Preimage_labels_of_conform hConform]

theorem root0DigestOps_labels_of_conform
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    {bindings : List (Root0CommitmentDigestBinding CommitmentDigest)}
    {pubMeta : KernelMetaPub Digest RootParamsId}
    (transcriptSeed : List Byte)
    (hConform : root0CommitmentDigestBindingsConform bindings) :
    (root0DigestOps
        encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta).filterMap
        transcriptOpLabel? =
      [poseidon2AppDomain, transcriptSeedLabel] ++
        root0MetaPubLabels.take 3 ++ root0CommitmentLabels ++ root0MetaPubLabels.drop 3 := by
  simp [root0DigestOps, root0TranscriptOps_labels_of_conform
    encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed hConform, transcriptOpLabel?]

theorem root0DigestInputWords_suffix
    (encodeCommitmentDigest : CommitmentDigest → List Byte)
    (encodeDigest : Digest → List Byte)
    (encodeRootParamsId : RootParamsId → List Byte)
    (transcriptSeed : List Byte)
    (bindings : List (Root0CommitmentDigestBinding CommitmentDigest))
    (pubMeta : KernelMetaPub Digest RootParamsId) :
    (root0DigestInputWords
        encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta).getLast? =
      some domainGateWord := by
  have hAppend :
      transcriptOpsAbsorbWords
          (root0TranscriptOps
              encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta ++
            [.digest32]) =
        transcriptOpsAbsorbWords
            (root0TranscriptOps
              encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta) ++
          [domainGateWord] := by
    induction root0TranscriptOps
        encodeCommitmentDigest encodeDigest encodeRootParamsId transcriptSeed bindings pubMeta with
    | nil =>
        simp [transcriptOpsAbsorbWords, transcriptOpAbsorbWords]
    | cons op ops ih =>
        simp [transcriptOpsAbsorbWords, ih, List.append_assoc]
  simp [root0DigestInputWords, root0DigestOps, hAppend]

end Nightstream.Chip8.Poseidon2Transcript
