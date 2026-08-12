import Nightstream.Implementation.NebulaV2.TerminalManifestStateAuthority

/-! Focused regressions for row-derived terminal input-state authority. -/

namespace NightstreamTests.NebulaV2TerminalManifestStateAuthority

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.TerminalManifestSchema
open Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall
open Nightstream.Implementation.NebulaV2.TerminalManifestStateAuthority

example {widths : CompilerWidths}
    {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityBoundaryRows.Authority :=
  incomingAuthority carry satisfies

example {widths : CompilerWidths}
    {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    ∀ lane : Fin 4,
      (incomingAuthority carry satisfies).digest lane =
        assignment
          (List.getD
            artifact.layouts.priorStateLink.stateOutput.hash.stateOutput.trace.outputColumns
            lane.val 0) :=
  incomingAuthority_digest_eq_columns carry satisfies

example {widths : CompilerWidths}
    {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (previous : StateAuthorityBoundaryRows.Authority)
    (previousPlaced : PreviousStatePlaced (artifact := artifact)
      (assignment := assignment) previous)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityBoundaryRows.Boundary previous
      (incomingAuthority carry satisfies) :=
  boundaryFromPrevious carry previous previousPlaced satisfies

end NightstreamTests.NebulaV2TerminalManifestStateAuthority
