import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursivePiRlcTranscriptRhosArtifact

/-!
Exact Poseidon call schedule for the recursive-profile Π_RLC scalar sampler.

Owns: the six call descriptors that implement the one recursive-profile scalar
domain transition and its four digest blocks, plus proofs that those exact
descriptors occur in the generated owner.

Does not own: call semantics, constant-pin meaning, canonical-u64 meaning,
chunk selection, the pre-Π_RLC transcript prefix, native Rust conformance, or
row/cost authority outside this exact owner.

Emits constraints: no. This file only names existing artifact pieces.

Authority boundary: membership in a generated owner is structural evidence
only. Later modules must use `TranscriptCertificate.CallAccepted` to derive
the permutation result and must separately prove every input-wire connection.

| Protocol | Phase | Constraint family | Exact artifact obligation |
|---|---|---|---|
| `Pi_RLC` | scalar domain | full-cursor permutation | appending `[2, 0, coordinate]` crosses the rate boundary |
| `Pi_RLC` | digest block 0 | full-cursor permutation | the first block fills the remaining rate lanes before squeeze |
| `Pi_RLC` | digest block 0 | squeeze permutation | absorbed squeeze-one determines the first four digest lanes |
| `Pi_RLC` | digest blocks 1-3 | squeeze permutation | each later `[2, 1, counter, 1]` block uses one permutation |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Schedule

open Nightstream.Implementation.R1CS.OwnerCertificate

namespace Artifact

/-- Rate-boundary permutation triggered by the final scalar-domain word. -/
def enterScalarCall : Poseidon2Call.Call :=
  { rowStart := 1209
    rowEnd := 1809
    inputColumns :=
      [348830, 348831, 350046, 350047, 350042, 350043, 350044, 350045]
    firstAllocatedColumn := 350049 }

/-- First-block permutation triggered before squeeze-one can be absorbed. -/
def block0FullCursorCall : Poseidon2Call.Call :=
  { rowStart := 1814
    rowEnd := 2414
    inputColumns :=
      [350048, 350650, 350651, 350652, 350645, 350646, 350647, 350648]
    firstAllocatedColumn := 350654 }

/-- Squeeze permutation producing digest block zero. -/
def block0DigestCall : Poseidon2Call.Call :=
  { rowStart := 2414
    rowEnd := 3014
    inputColumns :=
      [350653, 351247, 351248, 351249, 351250, 351251, 351252, 351253]
    firstAllocatedColumn := 351254 }

/-- Squeeze permutation producing digest block one. -/
def block1DigestCall : Poseidon2Call.Call :=
  { rowStart := 3710
    rowEnd := 4310
    inputColumns :=
      [352486, 352487, 352488, 352489, 351850, 351851, 351852, 351853]
    firstAllocatedColumn := 352490 }

/-- Squeeze permutation producing digest block two. -/
def block2DigestCall : Poseidon2Call.Call :=
  { rowStart := 5006
    rowEnd := 5606
    inputColumns :=
      [353722, 353723, 353724, 353725, 353086, 353087, 353088, 353089]
    firstAllocatedColumn := 353726 }

/-- Squeeze permutation producing digest block three. -/
def block3DigestCall : Poseidon2Call.Call :=
  { rowStart := 6302
    rowEnd := 6902
    inputColumns :=
      [354958, 354959, 354960, 354961, 354322, 354323, 354324, 354325]
    firstAllocatedColumn := 354962 }

def enterScalarPiece : Piece :=
  { rowStart := 352496
    rowEnd := 353096
    payload := .poseidon enterScalarCall }

def block0FullCursorPiece : Piece :=
  { rowStart := 353101
    rowEnd := 353701
    payload := .poseidon block0FullCursorCall }

def block0DigestPiece : Piece :=
  { rowStart := 353701
    rowEnd := 354301
    payload := .poseidon block0DigestCall }

def block1DigestPiece : Piece :=
  { rowStart := 354997
    rowEnd := 355597
    payload := .poseidon block1DigestCall }

def block2DigestPiece : Piece :=
  { rowStart := 356293
    rowEnd := 356893
    payload := .poseidon block2DigestCall }

def block3DigestPiece : Piece :=
  { rowStart := 357589
    rowEnd := 358189
    payload := .poseidon block3DigestCall }

theorem enterScalarPiece_mem :
    enterScalarPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [enterScalarPiece, enterScalarCall,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0]

theorem block0FullCursorPiece_mem :
    block0FullCursorPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block0FullCursorPiece, block0FullCursorCall,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0]

theorem block0DigestPiece_mem :
    block0DigestPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block0DigestPiece, block0DigestCall,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0]

theorem block1DigestPiece_mem :
    block1DigestPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block1DigestPiece, block1DigestCall,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0]

theorem block2DigestPiece_mem :
    block2DigestPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block2DigestPiece, block2DigestCall,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0]

theorem block3DigestPiece_mem :
    block3DigestPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block3DigestPiece, block3DigestCall,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0]

end Artifact

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Schedule
