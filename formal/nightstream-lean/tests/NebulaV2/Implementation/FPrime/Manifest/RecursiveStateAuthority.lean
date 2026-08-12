import Nightstream.Implementation.NebulaV2.FPrime.Manifest.RecursiveStateAuthority

/-! Focused regressions for row-derived recursive invocation state authority. -/

namespace NightstreamTests.NebulaV2RecursiveManifestStateAuthority

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall
open Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority

#check RecursiveManifestNifsCall.Call.CarryBlocks.priorAccepted
#check RecursiveManifestNifsCall.Call.CarryBlocks.intermediateAccepted
#check RecursiveManifestNifsCall.Call.CarryBlocks.outgoingAccepted

example {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityBoundaryRows.Invocation :=
  invocation carry satisfies

example {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    ∀ lane : Fin 4,
      (outgoingAuthority carry satisfies).digest lane =
        assignment
          (List.getD
            artifact.layouts.stateOutput.hash.stateOutput.trace.outputColumns
            lane.val 0) :=
  outgoingAuthority_digest_eq_columns carry satisfies

example {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (previous : StateAuthorityBoundaryRows.Authority)
    (previousPlaced : PreviousStatePlaced (artifact := artifact)
      (assignment := assignment) previous)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityBoundaryRows.Boundary previous
      (priorAuthority carry satisfies) :=
  boundaryFromPrevious carry previous previousPlaced satisfies

end NightstreamTests.NebulaV2RecursiveManifestStateAuthority
