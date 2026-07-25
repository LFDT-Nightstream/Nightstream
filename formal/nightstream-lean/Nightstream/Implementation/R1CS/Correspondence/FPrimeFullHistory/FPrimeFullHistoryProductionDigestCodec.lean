import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryRecursiveOutputSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalLogicalLinkSound

/-!
Contract: decode the exact recursive-output owner into the selected typed
production digest codec.

Assurance tier: artifact-checked output refinement.

Owns:
- construction of one canonical four-lane production digest from the exact
  output-encoding facts;
- equality between that digest's codec coordinates and the four physical
  `x_out` wires;
- successful round-trip through the selected production digest decoder.
- equality between the captured terminal logical public input and the
  paper-owned canonical public-input encoding of that same typed digest.

Does not own: a state codec, the compact optional/linked adapter value at this
call site, equality of the whole generated artifact with the selected typed
lowering program, the thirteen current-production plain-carrier padding rows,
Rust-source or compiled-Rust semantics, or Poseidon2 collision resistance.

Emits constraints: no; it decodes an existing receipt-owned output block.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

abbrev ProductionDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

/-- The canonical typed digest determined by the four physical digest
coordinates. The encoding facts supply the lane bounds. -/
def decodedDigest
    (assignment : Nat -> Nat)
    (encoding :
      FPrimeEncodingSound.Holds
        (FPrimeFullHistoryOutputEncodingSound.Pulled assignment)) :
    ProductionDigest :=
  fun lane =>
    ⟨FPrimeFullHistoryOutputEncodingSound.Pulled assignment
        (FPrimeEncoding.digestCol lane.val),
      by
        have canonical :=
          encoding.laneCanonical lane.val lane.isLt
        rw [canonical.1]
        exact canonical.2⟩

@[simp] theorem decodedDigest_value
    (assignment : Nat -> Nat)
    (encoding :
      FPrimeEncodingSound.Holds
        (FPrimeFullHistoryOutputEncodingSound.Pulled assignment))
    (lane : Fin 4) :
    (decodedDigest assignment encoding lane).val =
      assignment
        (Relabel.column FPrimeFullHistoryOutputEncoding.columnMap
          (FPrimeEncoding.digestCol lane.val)) :=
  rfl

private theorem rangeFour :
    List.range 4 = [0, 1, 2, 3] := by
  decide

/-- The selected digest codec reads the exact four artifact output lanes in
their physical lane order. -/
theorem codec_values_eq_outputDigest
    (assignment : Nat -> Nat)
    (encoding :
      FPrimeEncodingSound.Holds
        (FPrimeFullHistoryOutputEncodingSound.Pulled assignment)) :
    (digestCodec.encode (decodedDigest assignment encoding)).map
        (fun field => field.val) =
      FPrimeFullHistoryOutputEncodingSound.outputDigest assignment := by
  rw [digestCodec_encode_exact]
  simp only [List.map_cons, List.map_nil]
  rw [FPrimeFullHistoryOutputEncodingSound.outputDigest, rangeFour]
  rfl

/-- The physical-lane digest is accepted by the same decoder used by the
typed lowering profile. -/
theorem codec_roundtrip
    (assignment : Nat -> Nat)
    (encoding :
      FPrimeEncodingSound.Holds
        (FPrimeFullHistoryOutputEncodingSound.Pulled assignment)) :
    digestCodec.decode
        (digestCodec.encode (decodedDigest assignment encoding)) =
      some (decodedDigest assignment encoding) :=
  digestCodec_roundtrip (decodedDigest assignment encoding)

/-- The digest reconstructed for the selected codec is the same typed digest
used by the independently proved terminal logical-link theorem. The proof
components of the four bounded lanes are irrelevant; their physical values
and lane order are definitionally identical. -/
theorem decodedDigest_eq_logicalLinkDigest
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (encoding :
      FPrimeEncodingSound.Holds
        (FPrimeFullHistoryOutputEncodingSound.Pulled assignment)) :
    decodedDigest assignment encoding =
      FPrimeFullHistoryTerminalLogicalLinkSound.outputDigest
        assignment canonical := by
  funext lane
  apply Subtype.ext
  rfl

/-- The captured terminal logical public input is exactly the paper-owned
canonical public-input encoding of the selected typed production digest. -/
theorem terminalLogicalPublic_eq_encodePublicInput
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (encoding :
      FPrimeEncodingSound.Holds
        (FPrimeFullHistoryOutputEncodingSound.Pulled assignment))
    (terminalLink :
      FPrimeFullHistoryTerminalLinkSound.Holds assignment) :
    FPrimeFullHistoryTerminalLogicalLinkSound.terminalLogicalPublic
        assignment =
      Nightstream.Implementation.Encoding.FPrime.encodePublicInput
        (decodedDigest assignment encoding) := by
  have accepted :=
    FPrimeFullHistoryTerminalLogicalLinkSound.logicalCheck_of_holds
      canonical encoding terminalLink
  have equal :=
    (CanonicalPublicInputLink.check_eq_true_iff
      (FPrimeFullHistoryTerminalLogicalLinkSound.outputDigest
        assignment canonical)
      (FPrimeFullHistoryTerminalLogicalLinkSound.terminalLogicalPublic
        assignment)).mp accepted
  rw [← decodedDigest_eq_logicalLinkDigest assignment canonical encoding]
    at equal
  exact equal

/-- Satisfaction of the exact recursive-output owner yields one selected
typed digest whose codec coordinates are exactly the produced `x_out`
columns. -/
theorem rows_decode_exact_xOut
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment) :
    exists digest : ProductionDigest,
      (digestCodec.encode digest).map (fun field => field.val) =
          FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment /\
        digestCodec.decode (digestCodec.encode digest) = some digest := by
  let facts :=
    FPrimeFullHistoryRecursiveOutputSound.sound
      goldilocksPrime canonical one satisfies
  refine ⟨decodedDigest assignment facts.encoding, ?_, ?_⟩
  · rw [codec_values_eq_outputDigest assignment facts.encoding]
    exact
      FPrimeFullHistoryRecursiveOutputSound.outputDigest_eq_xOutColumns
        assignment
  · exact codec_roundtrip assignment facts.encoding

/-- The exact recursive producer and terminal delayed-link consumer identify
the same selected typed digest. Both physical boundaries use the codec's lane
order, and the decoder accepts that common representation. -/
theorem output_and_terminal_rows_decode_same_digest
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputSatisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (terminalSatisfies :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment) :
    exists digest : ProductionDigest,
      (digestCodec.encode digest).map (fun field => field.val) =
          FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment /\
        (digestCodec.encode digest).map (fun field => field.val) =
          FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment /\
        digestCodec.decode (digestCodec.encode digest) = some digest := by
  let facts :=
    FPrimeFullHistoryRecursiveOutputSound.sound
      goldilocksPrime canonical one outputSatisfies
  let digest := decodedDigest assignment facts.encoding
  have codecXOut :
      (digestCodec.encode digest).map (fun field => field.val) =
        FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment := by
    rw [codec_values_eq_outputDigest assignment facts.encoding]
    exact
      FPrimeFullHistoryRecursiveOutputSound.outputDigest_eq_xOutColumns
        assignment
  have terminalXOut :
      FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment =
        FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment :=
    FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut
      goldilocksPrime canonical one outputSatisfies terminalSatisfies
  exact
    ⟨digest, codecXOut.trans terminalXOut.symm, codecXOut,
      codec_roundtrip assignment facts.encoding⟩

/-- Exact captured-row composition: one selected typed digest simultaneously
decodes the recursive `x_out` columns, round-trips through the selected codec,
and canonically encodes the terminal paper public input. This theorem concerns
the captured 257-coordinate logical prefix; the thirteen current-production
plain-carrier padding rows remain outside this artifact. -/
theorem output_and_terminal_rows_decode_linked_digest
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputSatisfies :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (terminalSatisfies :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment) :
    exists digest : ProductionDigest,
      (digestCodec.encode digest).map (fun field => field.val) =
          FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment /\
        digestCodec.decode (digestCodec.encode digest) = some digest /\
        FPrimeFullHistoryTerminalLogicalLinkSound.terminalLogicalPublic
            assignment =
          Nightstream.Implementation.Encoding.FPrime.encodePublicInput
            digest := by
  let facts :=
    FPrimeFullHistoryRecursiveOutputSound.sound
      goldilocksPrime canonical one outputSatisfies
  let terminalLink :=
    FPrimeFullHistoryTerminalLinkSound.sound
      canonical one terminalSatisfies
  let digest := decodedDigest assignment facts.encoding
  refine ⟨digest, ?_, codec_roundtrip assignment facts.encoding, ?_⟩
  · rw [codec_values_eq_outputDigest assignment facts.encoding]
    exact
      FPrimeFullHistoryRecursiveOutputSound.outputDigest_eq_xOutColumns
        assignment
  · exact
      terminalLogicalPublic_eq_encodePublicInput
        canonical facts.encoding terminalLink

end Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec
