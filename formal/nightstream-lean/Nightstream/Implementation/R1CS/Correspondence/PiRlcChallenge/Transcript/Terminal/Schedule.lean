import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows

/-!
Exact terminal-profile Poseidon2 call tree for all fifteen `Pi_RLC` scalars.

Assurance tier: implementation/R1CS structural correspondence. This file names
the affine scalar/call layout and proves that every named call is exactly the
corresponding generated owner piece. It assigns no transcript meaning by row
order alone.

Owns: the terminal `rho -> call phase` tree; exact call descriptors; exact
input-column lists; scalar-zero versus successor entry-boundary shape; and
generated-owner membership for all 76 sampler Poseidon2 calls.

Does not own: constant-pin values, Poseidon2 call semantics, inter-call state
connectivity, canonical field decomposition, rejection selection, coefficient
assembly, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: a generated piece is structural evidence only. Later
modules must independently decode every pin, replay every Poseidon2 call, and
prove that output state and candidate fields come from the same verifier-owned
transcript execution.

| Protocol | Phase | Constraint family | Multiplicity | Exact structural obligation |
|---|---|---|---:|---|
| `Pi_RLC` | scalar 0 entry | rate-boundary Poseidon2 | 1 | cursor-2 scalar domain crosses the rate boundary |
| `Pi_RLC` | scalars 1-14 entry/block 0 | shared rate-boundary Poseidon2 | 14 | cursor-0 scalar domain plus block-length word crosses the boundary |
| `Pi_RLC` | block 0 digest | squeeze Poseidon2 | 15 | first four candidate fields and successor state |
| `Pi_RLC` | blocks 1-3 | squeeze Poseidon2 | 45 | three further candidate blocks and successor states |
| `Pi_RLC` | complete call tree | exact owner pieces | 76 | every call descriptor occurs at its protocol address |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.Schedule

open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

def scalarColumnStride : Nat := 7984
def scalarRowStride : Nat := 8387

/-- Constants allocated before one scalar's entry and block-zero boundary. -/
def domainBase (rho : Fin ScalarRows.scalarCount) : Nat :=
  2555251 + scalarColumnStride * rho.val

/-- First allocated column of scalar zero's dedicated entry permutation, or
of a successor scalar's combined entry/block-zero boundary permutation. -/
def entryFirstAllocated (rho : Fin ScalarRows.scalarCount) : Nat :=
  if rho.val = 0 then 2554654 else domainBase rho + 6

/-- High four lanes of the state preceding a nonzero scalar. -/
def previousHighColumns (rho : Fin ScalarRows.scalarCount) : List Nat :=
  [2552179 + scalarColumnStride * rho.val,
   2552180 + scalarColumnStride * rho.val,
   2552181 + scalarColumnStride * rho.val,
   2552182 + scalarColumnStride * rho.val]

/-- Scalar zero crosses the rate boundary while entering its scalar domain.
Every successor crosses only after also absorbing block zero's length word. -/
def entryBoundaryCall (rho : Fin ScalarRows.scalarCount) : Poseidon2Call.Call :=
  if rho.val = 0 then
    { rowStart := 1209
      rowEnd := 1809
      inputColumns :=
        [2553435, 2553436, 2554651, 2554652,
         2554647, 2554648, 2554649, 2554650]
      firstAllocatedColumn := 2554654 }
  else
    { rowStart := 1812 + scalarRowStride * rho.val
      rowEnd := 2412 + scalarRowStride * rho.val
      inputColumns :=
        [domainBase rho, domainBase rho + 1, domainBase rho + 2,
         domainBase rho + 4] ++ previousHighColumns rho
      firstAllocatedColumn := entryFirstAllocated rho }

/-- Scalar zero alone needs a second rate-boundary permutation after entering
its domain and before the block-zero squeeze permutation. -/
def scalar0Block0FullCursorCall : Poseidon2Call.Call :=
  { rowStart := 1814
    rowEnd := 2414
    inputColumns :=
      [2554653, 2555255, 2555256, 2555257,
       2555250, 2555251, 2555252, 2555253]
    firstAllocatedColumn := 2555259 }

/-- Exact block-zero digest input. Scalar zero consumes a separate full-cursor
call; successors reuse their combined entry-boundary output. -/
def block0InputColumns (rho : Fin ScalarRows.scalarCount) : List Nat :=
  if rho.val = 0 then
    [2555258, 2555852, 2555853, 2555854,
     2555855, 2555856, 2555857, 2555858]
  else
    [domainBase rho + 5,
     entryFirstAllocated rho + 600,
     entryFirstAllocated rho + 601,
     entryFirstAllocated rho + 595,
     entryFirstAllocated rho + 596,
     entryFirstAllocated rho + 597,
     entryFirstAllocated rho + 598,
     entryFirstAllocated rho + 599]

def block0DigestCall (rho : Fin ScalarRows.scalarCount) : Poseidon2Call.Call :=
  { rowStart := 2414 + scalarRowStride * rho.val
    rowEnd := 3014 + scalarRowStride * rho.val
    inputColumns := block0InputColumns rho
    firstAllocatedColumn := 2555859 + scalarColumnStride * rho.val }

def laterBlockPinBase
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) : Nat :=
  2557091 + scalarColumnStride * rho.val + 1236 * block.val

def priorDigestHighBase
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) : Nat :=
  2556455 + scalarColumnStride * rho.val + 1236 * block.val

/-- Uniform calls for digest blocks one, two, and three. -/
def laterDigestCall
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) : Poseidon2Call.Call :=
  let pin := laterBlockPinBase rho block
  let prior := priorDigestHighBase rho block
  { rowStart := 3710 + scalarRowStride * rho.val + 1296 * block.val
    rowEnd := 4310 + scalarRowStride * rho.val + 1296 * block.val
    inputColumns :=
      [pin, pin + 1, pin + 2, pin + 3,
       prior, prior + 1, prior + 2, prior + 3]
    firstAllocatedColumn := pin + 4 }

def block1DigestCall (rho : Fin ScalarRows.scalarCount) : Poseidon2Call.Call :=
  laterDigestCall rho ⟨0, by decide⟩

def block2DigestCall (rho : Fin ScalarRows.scalarCount) : Poseidon2Call.Call :=
  laterDigestCall rho ⟨1, by decide⟩

def block3DigestCall (rho : Fin ScalarRows.scalarCount) : Poseidon2Call.Call :=
  laterDigestCall rho ⟨2, by decide⟩

/-! ## Exact owner addresses -/

def entryPieceIndex (rho : Fin ScalarRows.scalarCount) :
    Fin ScalarRows.pieceCount :=
  ⟨if rho.val = 0 then 5 else 6 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [ScalarRows.scalarCount, ScalarRows.pieceCount] at rhoLt ⊢
    split <;> omega⟩

def block0PieceIndex (rho : Fin ScalarRows.scalarCount) :
    Fin ScalarRows.pieceCount :=
  ⟨8 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [ScalarRows.scalarCount, ScalarRows.pieceCount] at rhoLt ⊢
    omega⟩

def block1PieceIndex (rho : Fin ScalarRows.scalarCount) :
    Fin ScalarRows.pieceCount :=
  ⟨18 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [ScalarRows.scalarCount, ScalarRows.pieceCount] at rhoLt ⊢
    omega⟩

def block2PieceIndex (rho : Fin ScalarRows.scalarCount) :
    Fin ScalarRows.pieceCount :=
  ⟨28 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [ScalarRows.scalarCount, ScalarRows.pieceCount] at rhoLt ⊢
    omega⟩

def block3PieceIndex (rho : Fin ScalarRows.scalarCount) :
    Fin ScalarRows.pieceCount :=
  ⟨38 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [ScalarRows.scalarCount, ScalarRows.pieceCount] at rhoLt ⊢
    omega⟩

def entryPiece (rho : Fin ScalarRows.scalarCount) : Piece :=
  ScalarRows.pieceAt (entryPieceIndex rho)

def scalar0Block0FullCursorPiece : Piece :=
  ScalarRows.pieceAt ⟨7, by decide⟩

def block0Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  ScalarRows.pieceAt (block0PieceIndex rho)

def block1Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  ScalarRows.pieceAt (block1PieceIndex rho)

def block2Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  ScalarRows.pieceAt (block2PieceIndex rho)

def block3Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  ScalarRows.pieceAt (block3PieceIndex rho)

def expectedEntryPiece (rho : Fin ScalarRows.scalarCount) : Piece :=
  { rowStart := if rho.val = 0 then 2727315
      else 2727918 + scalarRowStride * rho.val
    rowEnd := if rho.val = 0 then 2727915
      else 2728518 + scalarRowStride * rho.val
    payload := .poseidon (entryBoundaryCall rho) }

def expectedScalar0Block0FullCursorPiece : Piece :=
  { rowStart := 2727920
    rowEnd := 2728520
    payload := .poseidon scalar0Block0FullCursorCall }

def expectedBlock0Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  { rowStart := 2728520 + scalarRowStride * rho.val
    rowEnd := 2729120 + scalarRowStride * rho.val
    payload := .poseidon (block0DigestCall rho) }

def expectedBlock1Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  { rowStart := 2729816 + scalarRowStride * rho.val
    rowEnd := 2730416 + scalarRowStride * rho.val
    payload := .poseidon (block1DigestCall rho) }

def expectedBlock2Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  { rowStart := 2731112 + scalarRowStride * rho.val
    rowEnd := 2731712 + scalarRowStride * rho.val
    payload := .poseidon (block2DigestCall rho) }

def expectedBlock3Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  { rowStart := 2732408 + scalarRowStride * rho.val
    rowEnd := 2733008 + scalarRowStride * rho.val
    payload := .poseidon (block3DigestCall rho) }

/-- Closed protocol/phase/call tree. Kernel reduction checks all fifteen
scalar coordinates and all five named call families against the exact owner. -/
theorem callTree_eq : forall rho : Fin ScalarRows.scalarCount,
    entryPiece rho = expectedEntryPiece rho /\
    block0Piece rho = expectedBlock0Piece rho /\
    block1Piece rho = expectedBlock1Piece rho /\
    block2Piece rho = expectedBlock2Piece rho /\
    block3Piece rho = expectedBlock3Piece rho := by
  decide

theorem scalar0Block0FullCursorPiece_eq :
    scalar0Block0FullCursorPiece = expectedScalar0Block0FullCursorPiece := by
  decide

/-- Complete 76-call structural tree: the scalar-zero-only full-cursor call
plus five indexed call families for all fifteen scalars. -/
theorem completeCallTree_eq :
    scalar0Block0FullCursorPiece = expectedScalar0Block0FullCursorPiece /\
    forall rho : Fin ScalarRows.scalarCount,
      entryPiece rho = expectedEntryPiece rho /\
      block0Piece rho = expectedBlock0Piece rho /\
      block1Piece rho = expectedBlock1Piece rho /\
      block2Piece rho = expectedBlock2Piece rho /\
      block3Piece rho = expectedBlock3Piece rho :=
  ⟨scalar0Block0FullCursorPiece_eq, callTree_eq⟩

theorem entryPiece_eq (rho : Fin ScalarRows.scalarCount) :
    entryPiece rho = expectedEntryPiece rho :=
  (callTree_eq rho).1

theorem block0Piece_eq (rho : Fin ScalarRows.scalarCount) :
    block0Piece rho = expectedBlock0Piece rho :=
  (callTree_eq rho).2.1

theorem block1Piece_eq (rho : Fin ScalarRows.scalarCount) :
    block1Piece rho = expectedBlock1Piece rho :=
  (callTree_eq rho).2.2.1

theorem block2Piece_eq (rho : Fin ScalarRows.scalarCount) :
    block2Piece rho = expectedBlock2Piece rho :=
  (callTree_eq rho).2.2.2.1

theorem block3Piece_eq (rho : Fin ScalarRows.scalarCount) :
    block3Piece rho = expectedBlock3Piece rho :=
  (callTree_eq rho).2.2.2.2

theorem entryPiece_mem (rho : Fin ScalarRows.scalarCount) :
    entryPiece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem scalar0Block0FullCursorPiece_mem :
    scalar0Block0FullCursorPiece ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem block0Piece_mem (rho : Fin ScalarRows.scalarCount) :
    block0Piece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem block1Piece_mem (rho : Fin ScalarRows.scalarCount) :
    block1Piece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem block2Piece_mem (rho : Fin ScalarRows.scalarCount) :
    block2Piece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem block3Piece_mem (rho : Fin ScalarRows.scalarCount) :
    block3Piece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.Schedule
