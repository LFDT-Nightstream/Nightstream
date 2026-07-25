import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkCanonicalRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryProductionDigestCodec

/-!
Contract: construct the current 270-row plain terminal-link owner from the
captured full-history producer and 257-row logical-link snapshot.

Assurance tier: artifact-checked honest completion.

Owns:
- an explicit local assignment that copies the captured producer bits and
  terminal logical prefix into the current isolated owner;
- verifier-fixed zero values for exactly the thirteen added padding columns;
- satisfaction of all current isolated rows from the captured producer and
  logical-link rows;
- one selected typed digest shared by the recursive output, completed current
  carrier, selected codec, and paper public input.

Does not own: a current generated full-history artifact, placement of these
local columns in such an artifact, host shape checks, equality of whole
programs, compiled-Rust semantics, or Poseidon2 collision resistance.  This is
an honest assignment construction, not a claim that the stale 257-row list is
the current 270-row list.

Emits constraints: no; it constructs an assignment for the already selected
current isolated owner.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkCompletion

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

abbrev ProductionDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

/-- Local current-owner assignment. Columns `1..256` copy the captured
producer bits, column `257` and columns `258..513` copy the captured logical
fresh input, and every later column is verifier-fixed zero. -/
def completedAssignment
    (assignment : Nat -> Nat)
    (column : Nat) : Nat :=
  if column = 0 then
    assignment 0
  else if column <= 256 then
    assignment
      (FPrimeFullHistoryTerminalLinkSound.lastXOutBitCol (column - 1))
  else if column = 257 then
    assignment FPrimeFullHistoryTerminalLinkSound.freshOneCol
  else if column <= 513 then
    assignment
      (FPrimeFullHistoryTerminalLinkSound.freshBitCol (column - 258))
  else
    0

@[simp] theorem completedAssignment_zero
    (assignment : Nat -> Nat) :
    completedAssignment assignment 0 = assignment 0 := by
  rfl

@[simp] theorem completedAssignment_lastXOutBit
    (assignment : Nat -> Nat)
    (bit : Nat)
    (bitLt : bit < 256) :
    completedAssignment assignment
        (FPrimeTerminalLink.lastXOutBitCol bit) =
      assignment
        (FPrimeFullHistoryTerminalLinkSound.lastXOutBitCol bit) := by
  unfold completedAssignment FPrimeTerminalLink.lastXOutBitCol
  rw [if_neg (by omega), if_pos (by omega)]
  congr 2
  omega

@[simp] theorem completedAssignment_freshOne
    (assignment : Nat -> Nat) :
    completedAssignment assignment FPrimeTerminalLink.freshOneCol =
      assignment FPrimeFullHistoryTerminalLinkSound.freshOneCol := by
  rfl

@[simp] theorem completedAssignment_freshBit
    (assignment : Nat -> Nat)
    (bit : Nat)
    (bitLt : bit < 256) :
    completedAssignment assignment (FPrimeTerminalLink.freshBitCol bit) =
      assignment
        (FPrimeFullHistoryTerminalLinkSound.freshBitCol bit) := by
  unfold completedAssignment FPrimeTerminalLink.freshBitCol
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
    if_pos (by omega)]
  apply congrArg assignment
  apply congrArg FPrimeFullHistoryTerminalLinkSound.freshBitCol
  omega

@[simp] theorem completedAssignment_padding
    (assignment : Nat -> Nat)
    (padding : Nat)
    (paddingLt : padding < 13) :
    completedAssignment assignment
        (FPrimeTerminalLink.freshPaddingCol padding) = 0 := by
  unfold completedAssignment FPrimeTerminalLink.freshPaddingCol
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
    if_neg (by omega)]

/-- The completed local assignment remains a canonical Goldilocks
representative at every column. -/
theorem completedAssignment_canonical
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP) :
    forall column, completedAssignment assignment column < goldilocksP := by
  intro column
  unfold completedAssignment
  by_cases zero : column = 0
  · rw [if_pos zero]
    exact canonical 0
  · rw [if_neg zero]
    by_cases producer : column <= 256
    · rw [if_pos producer]
      exact canonical _
    · rw [if_neg producer]
      by_cases affine : column = 257
      · rw [if_pos affine]
        exact canonical _
      · rw [if_neg affine]
        by_cases fresh : column <= 513
        · rw [if_pos fresh]
          exact canonical _
        · rw [if_neg fresh]
          decide

@[simp] theorem completedAssignment_one
    {assignment : Nat -> Nat}
    (one : assignment 0 = 1) :
    completedAssignment assignment 0 = 1 := by
  simpa using one

/-- The captured logical-link rows construct all current isolated rows; the
only new values are the thirteen explicit zero padding coordinates. -/
theorem currentRows_of_snapshotLinkRows
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (snapshotRows :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment) :
    Satisfies FPrimeTerminalLink.rows
      (completedAssignment assignment) := by
  have snapshot :=
    FPrimeFullHistoryTerminalLinkSound.sound
      canonical one snapshotRows
  apply
    FPrimeTerminalLinkSound.fPrimeTerminalLink_complete
      (completedAssignment_canonical canonical)
      (completedAssignment_one one)
  refine {
    affineOne := ?_
    linked := ?_
    paddingZero := ?_
  }
  · simpa using snapshot.affineOne
  · intro bit bitLt
    simpa [bitLt] using snapshot.linked bit bitLt
  · intro padding paddingLt
    exact completedAssignment_padding assignment padding paddingLt

/-- Exact producer-coordinate alignment for the completed local assignment.
The digest is reconstructed from the captured output owner, never supplied as
an independent authority. -/
theorem completedAssignment_producerAligned
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (encoding :
      FPrimeEncodingSound.Holds
        (FPrimeFullHistoryOutputEncodingSound.Pulled assignment)) :
    FPrimeTerminalLinkCanonicalRefinement.ProducerAligned
      (FPrimeFullHistoryTerminalLogicalLinkSound.outputDigest
        assignment canonical)
      (completedAssignment assignment) := by
  intro lane bit
  rw [completedAssignment_lastXOutBit assignment
    (lane.val * 64 + bit.val) (by
      have laneLt := lane.isLt
      have bitLt := bit.isLt
      omega)]
  have encoded :=
    FPrimeEncodingCanonicalBits.publicBit_eq_encodedBit
      (Relabel.canonical canonical) encoding lane bit
  change
    assignment
        (Relabel.column FPrimeFullHistoryOutputEncoding.columnMap
          (FPrimeEncoding.publicBitCol lane.val bit.val)) =
      CanonicalPlainCarrierLink.encodedBit
        (FPrimeFullHistoryTerminalLogicalLinkSound.outputDigest
          assignment canonical)
        lane bit at encoded
  rw [FPrimeFullHistoryOutputEncodingSound.publicBitColumnMap lane bit]
    at encoded
  exact encoded

/-- Complete constructive bridge across the captured/current boundary.

The assignment for the current owner is explicit and local. The theorem does
not identify it with a generated current full-history placement. -/
theorem output_and_snapshot_rows_construct_currentPlainOwner
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputRows :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (snapshotLinkRows :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment) :
    exists digest : ProductionDigest,
      Satisfies FPrimeTerminalLink.rows
          (completedAssignment assignment) /\
        FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
            (completedAssignment assignment) =
          CanonicalPlainCarrierLink.encodeClaim digest /\
        (digestCodec.encode digest).map (fun field => field.val) =
          FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment /\
        digestCodec.decode (digestCodec.encode digest) = some digest /\
        FPrimeFullHistoryTerminalLogicalLinkSound.terminalLogicalPublic
            assignment =
          Nightstream.Implementation.Encoding.FPrime.encodePublicInput
            digest := by
  let facts :=
    FPrimeFullHistoryRecursiveOutputSound.sound
      goldilocksPrime canonical one outputRows
  let digest :=
    FPrimeFullHistoryProductionDigestCodec.decodedDigest
      assignment facts.encoding
  have digestEq :
      digest =
        FPrimeFullHistoryTerminalLogicalLinkSound.outputDigest
          assignment canonical :=
    FPrimeFullHistoryProductionDigestCodec.decodedDigest_eq_logicalLinkDigest
      assignment canonical facts.encoding
  have currentRows :=
    currentRows_of_snapshotLinkRows
      canonical one snapshotLinkRows
  have aligned :
      FPrimeTerminalLinkCanonicalRefinement.ProducerAligned
        digest (completedAssignment assignment) := by
    rw [digestEq]
    exact completedAssignment_producerAligned canonical facts.encoding
  have currentAccepted :=
    FPrimeTerminalLinkCanonicalRefinement.check_of_satisfies
      digest
      (completedAssignment_canonical canonical)
      (completedAssignment_one one)
      currentRows aligned
  have currentClaim :
      FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
          (completedAssignment assignment) =
        CanonicalPlainCarrierLink.encodeClaim digest :=
    (CanonicalPlainCarrierLink.check_eq_true_iff
      digest
      (FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
        (completedAssignment assignment))).mp currentAccepted
  have codecXOut :
      (digestCodec.encode digest).map (fun field => field.val) =
        FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment := by
    rw [
      FPrimeFullHistoryProductionDigestCodec.codec_values_eq_outputDigest
        assignment facts.encoding
    ]
    exact
      FPrimeFullHistoryRecursiveOutputSound.outputDigest_eq_xOutColumns
        assignment
  have snapshot :=
    FPrimeFullHistoryTerminalLinkSound.sound
      canonical one snapshotLinkRows
  refine
    ⟨digest, currentRows, currentClaim, codecXOut,
      FPrimeFullHistoryProductionDigestCodec.codec_roundtrip
        assignment facts.encoding, ?_⟩
  exact
    FPrimeFullHistoryProductionDigestCodec.terminalLogicalPublic_eq_encodePublicInput
      canonical facts.encoding snapshot

end Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkCompletion
