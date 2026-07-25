import Nightstream.Assurance.FPrimeFullHistoryCircuit
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCurrentTerminalLinkCompletion
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCurrentTerminalLinkPlacement

/-!
Contract: expose the selected typed production digest at the final-state
fresh-public boundary of the exact supported full-history artifact.

Assurance tier: artifact-checked.

Owns:
- the exact observation of fresh-public digests carried by a Construction-2
  proof state;
- refinement of the final state's singleton fresh digest into the selected
  four-lane production codec under full-row satisfaction.
- equality between the captured terminal logical public input and the
  paper-owned canonical public-input encoding of that same digest.
- construction of an honest assignment for the current isolated 270-row plain
  terminal-link owner, including exactly thirteen verifier-fixed zero padding
  coordinates.
- exact artifact-checked placement and paper refinement of the current
  270-row `terminal.latest_link` range when that bounded generated range is
  supplied alongside the captured aggregate.

Does not own: a codec for the complete state, compact optional/linked output,
a generated aggregate for every current full-history row, inclusion of the
current bounded range in the stale captured aggregate, equality with the
selected typed lowering program, compiled-Rust semantics, or Poseidon2
collision resistance.

Emits constraints: no.
-/

namespace Nightstream.Assurance.FPrimeFullHistoryProductionDigest

open Nightstream.HyperNova.Construction2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

abbrev ProductionDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

/-- Public `x_out` payloads in the state's ordered fresh batch. -/
def latestPublicXOuts
    (state :
      State FPrimeFullHistoryCircuit.Digest
        FPrimeFullHistoryCircuit.Accumulator
        FPrimeFullHistoryCircuit.Fresh Unit) :
    List (List Nat) :=
  match state.proof with
  | .initial => []
  | .active _ latest => latest.map (fun fresh => fresh.publicXOut)

/-- The exact final state carries the terminal-link decoder's singleton
fresh-public digest. -/
theorem finalState_latestPublicXOuts
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    latestPublicXOuts
        (FPrimeFullHistoryCircuit.finalState assignment canonical) =
      [FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment] :=
  rfl

/-- Satisfaction of the complete checked-in artifact exposes the final
state's singleton fresh-public payload as one selected typed digest, with an
exact decoder round trip. -/
theorem fullRows_finalState_latest_digest
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : Satisfies FPrimeFullHistoryRows.fullRows assignment) :
    exists digest : ProductionDigest,
      latestPublicXOuts
          (FPrimeFullHistoryCircuit.finalState assignment canonical) =
        [(digestCodec.encode digest).map (fun field => field.val)] /\
      digestCodec.decode (digestCodec.encode digest) = some digest := by
  let owners := FPrimeFullHistoryCircuit.ownerRows_of_satisfies rows
  rcases
      FPrimeFullHistoryProductionDigestCodec.output_and_terminal_rows_decode_same_digest
        goldilocksPrime canonical one owners.recursiveOutput
          owners.terminal.latestLink with
    ⟨digest, codecTerminal, _codecOutput, decoded⟩
  refine ⟨digest, ?_, decoded⟩
  rw [finalState_latestPublicXOuts assignment canonical, codecTerminal]

/-- Full-row artifact lift of the complete selected digest boundary: the
final state's singleton fresh payload and the captured terminal logical public
input are both canonical encodings of one typed digest. The statement does
not cover the thirteen padding rows absent from this captured artifact. -/
theorem fullRows_finalState_latest_digest_and_logical_public
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : Satisfies FPrimeFullHistoryRows.fullRows assignment) :
    exists digest : ProductionDigest,
      latestPublicXOuts
          (FPrimeFullHistoryCircuit.finalState assignment canonical) =
        [(digestCodec.encode digest).map (fun field => field.val)] /\
      FPrimeFullHistoryTerminalLogicalLinkSound.terminalLogicalPublic
          assignment =
        Nightstream.Implementation.Encoding.FPrime.encodePublicInput digest /\
      digestCodec.decode (digestCodec.encode digest) = some digest := by
  let owners := FPrimeFullHistoryCircuit.ownerRows_of_satisfies rows
  rcases
      FPrimeFullHistoryProductionDigestCodec.output_and_terminal_rows_decode_linked_digest
        goldilocksPrime canonical one owners.recursiveOutput
          owners.terminal.latestLink with
    ⟨digest, codecOutput, decoded, logicalPublic⟩
  have terminalOutput :
      FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment =
        FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment :=
    FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut
      goldilocksPrime canonical one owners.recursiveOutput
        owners.terminal.latestLink
  have codecTerminal :
      (digestCodec.encode digest).map (fun field => field.val) =
        FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment :=
    codecOutput.trans terminalOutput.symm
  refine ⟨digest, ?_, logicalPublic, decoded⟩
  rw [finalState_latestPublicXOuts assignment canonical, codecTerminal]

/-- Full-row honest completion of the captured/current terminal-link
boundary. One selected digest simultaneously owns the final Construction-2
payload, the paper public input, and an explicit satisfying assignment for the
current isolated 270-row owner. This does not assert a current full-history
column placement. -/
theorem fullRows_construct_currentPlainOwner
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : Satisfies FPrimeFullHistoryRows.fullRows assignment) :
    exists digest : ProductionDigest,
      Satisfies FPrimeTerminalLink.rows
          (FPrimeFullHistoryCurrentTerminalLinkCompletion.completedAssignment
            assignment) /\
        FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
            (FPrimeFullHistoryCurrentTerminalLinkCompletion.completedAssignment
              assignment) =
          CanonicalPlainCarrierLink.encodeClaim digest /\
        latestPublicXOuts
            (FPrimeFullHistoryCircuit.finalState assignment canonical) =
          [(digestCodec.encode digest).map (fun field => field.val)] /\
        FPrimeFullHistoryTerminalLogicalLinkSound.terminalLogicalPublic
            assignment =
          Nightstream.Implementation.Encoding.FPrime.encodePublicInput
            digest /\
        digestCodec.decode (digestCodec.encode digest) = some digest := by
  let owners := FPrimeFullHistoryCircuit.ownerRows_of_satisfies rows
  rcases
      FPrimeFullHistoryCurrentTerminalLinkCompletion.output_and_snapshot_rows_construct_currentPlainOwner
        goldilocksPrime canonical one owners.recursiveOutput
          owners.terminal.latestLink with
    ⟨digest, currentRows, currentClaim, codecOutput, decoded,
      logicalPublic⟩
  have terminalOutput :
      FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment =
        FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment :=
    FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut
      goldilocksPrime canonical one owners.recursiveOutput
        owners.terminal.latestLink
  have codecTerminal :
      (digestCodec.encode digest).map (fun field => field.val) =
        FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment :=
    codecOutput.trans terminalOutput.symm
  refine
    ⟨digest, currentRows, currentClaim, ?_, logicalPublic, decoded⟩
  rw [finalState_latestPublicXOuts assignment canonical, codecTerminal]

/-- Artifact-checked placement lift for the current production owner.

The two row premises are intentionally separate: `fullRows` is the captured
aggregate that owns the recursive producer and final-state decoder, while
`currentTerminalRows` is the bounded certificate exported from the live
current synthesis. No theorem identifies the stale aggregate with the whole
current program. -/
theorem fullRows_and_currentTerminalPlacement_construct_plainOwner
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : Satisfies FPrimeFullHistoryRows.fullRows assignment)
    (currentTerminalRows :
      Satisfies
        FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment) :
    exists digest : ProductionDigest,
      Satisfies FPrimeTerminalLink.rows
          (FPrimeFullHistoryCurrentTerminalLinkPlacementSound.Pulled
            assignment) /\
        FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
            (FPrimeFullHistoryCurrentTerminalLinkPlacementSound.Pulled
              assignment) =
          CanonicalPlainCarrierLink.encodeClaim digest /\
        latestPublicXOuts
            (FPrimeFullHistoryCircuit.finalState assignment canonical) =
          [(digestCodec.encode digest).map (fun field => field.val)] /\
        digestCodec.decode (digestCodec.encode digest) = some digest /\
        exists logical,
          CanonicalPublicInputLink.check digest logical = true /\
            FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
                (FPrimeFullHistoryCurrentTerminalLinkPlacementSound.Pulled
                  assignment) =
              CanonicalPlainCarrierLink.completeClaim logical := by
  let owners := FPrimeFullHistoryCircuit.ownerRows_of_satisfies rows
  rcases
      FPrimeFullHistoryCurrentTerminalLinkPlacementSound.output_and_generated_rows_construct_currentPlainOwner
        goldilocksPrime canonical one owners.recursiveOutput
          currentTerminalRows with
    ⟨digest, localRows, currentClaim, codecOutput, decoded, logical⟩
  have terminalOutput :
      FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment =
        FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment :=
    FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut
      goldilocksPrime canonical one owners.recursiveOutput
        owners.terminal.latestLink
  have codecTerminal :
      (digestCodec.encode digest).map (fun field => field.val) =
        FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment :=
    codecOutput.trans terminalOutput.symm
  refine
    ⟨digest, localRows, currentClaim, ?_, decoded, logical⟩
  rw [finalState_latestPublicXOuts assignment canonical, codecTerminal]

end Nightstream.Assurance.FPrimeFullHistoryProductionDigest
