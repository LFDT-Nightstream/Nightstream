import Nightstream.Implementation.Nebula.FPrime.Manifest.RecursiveStateAuthority

/-! Focused regressions for row-derived recursive invocation state authority. -/

namespace NightstreamTests.NebulaRecursiveManifestStateAuthority

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.Nebula.RecursiveManifestSchema
open Nightstream.Implementation.Nebula.RecursiveManifestNifsCall
open Nightstream.Implementation.Nebula.RecursiveManifestStateAuthority

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

end NightstreamTests.NebulaRecursiveManifestStateAuthority
