import Nightstream.Implementation.Nebula.FPrime.State.AuthorityFullClaim
import Nightstream.Implementation.Nebula.FPrime.Manifest.TerminalNifsCall

/-!
Contract: extraction of normalized incoming state authority from one
satisfying Nebula V2 terminal manifest.

Assurance tier: implementation schema and cryptographic boundary.

Owns the exact connection from the mandatory terminal prior-state rows to the
column-independent state value consumed by the terminal invocation. The carry
block is the same parsed block consumed by the trailing memory transition.

Does not own the preceding producer, the cross-invocation claim carrier, the
terminal NIFS arithmetic, Poseidon2 collision resistance, or Rust conformance.

Emits constraints: no new rows. It projects mandatory manifest rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.TerminalManifestStateAuthority

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.Nebula.TerminalManifestSchema
open Nightstream.Implementation.Nebula.TerminalManifestNifsCall

/-- Exact normalized state authenticated by the terminal trailing claim.
Canonicality is derived from the mandatory terminal prior-state rows. -/
def incomingAuthority
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityBoundaryRows.Authority where
  payload := StateOutputAuthorityRows.payload
    artifact.layouts.priorStateLink.stateOutput.authority assignment
  carryBlock := carry.priorBlock
  frameCanonical :=
    AuthoritativeStateOutputBinding.typedFrame_canonical_of_rows
      artifact.layoutsValid.priorStateLinkValid.stateOutputValid
      call.canonicalAssignment call.one
      (call.priorStateCarryPlaced carry)
      (PriorStateLinkRows.stateOutput_rows_hold
        (artifact.priorStateLink_satisfied satisfies))

/-- Typed interpretation of the prior public-state digest columns selected by
the generated terminal wrapper. Global proof extraction must connect these
columns to the preceding recursive producer. -/
def PreviousStatePlaced
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (previous : StateAuthorityBoundaryRows.Authority) : Prop :=
  ∀ lane,
    assignment (artifact.layouts.priorPublicStateDigestColumn lane) =
      previous.digest lane

/-- The normalized terminal-input digest is exactly the four columns linked
to the CCS public image of the same trailing full claim. -/
theorem incomingAuthority_digest_eq_columns
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape :
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
            lane.val 0) := by
  have output :=
    AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest
      artifact.layoutsValid.priorStateLinkValid.stateOutputValid
      call.canonicalAssignment call.one (call.priorStateCarryPlaced carry)
      (PriorStateLinkRows.stateOutput_rows_hold
        (artifact.priorStateLink_satisfied satisfies))
  intro lane
  simpa [incomingAuthority, StateAuthorityBoundaryRows.Authority.digest,
    AuthoritativeStateOutputBinding.typedDigest,
    AuthoritativeStateOutputBinding.typedFrame,
    StateOutputPoseidonBinding.outerHash] using (output lane).symm

/-- The mandatory terminal wrapper rows construct the final delayed boundary
from the prior public state to the recomputed trailing-claim state. -/
def boundaryFromPrevious
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape :
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
      (incomingAuthority carry satisfies) where
  layout := artifact.layouts.priorStateBoundary
  assignment := assignment
  canonicalAssignment := call.canonicalAssignment
  one := call.one
  placed := by
    intro lane
    constructor
    · rw [artifact.layoutsValid.priorBoundaryUsesPublicState]
      exact previousPlaced lane
    · rw [artifact.layoutsValid.priorBoundaryUsesRecomputedState]
      exact (incomingAuthority_digest_eq_columns carry satisfies lane).symm
  satisfies := artifact.priorStateBoundary_satisfied satisfies

/-- Exact producer/consumer state recovery makes the four terminal wrapper
rows place the producer digest in the named prior-public-state columns. The
placement is derived, not supplied. -/
theorem previousStatePlaced_of_same
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (previous : StateAuthorityBoundaryRows.Authority)
    (satisfies : Satisfies artifact.programRows assignment)
    (same : StateAuthorityBoundaryRows.Same previous
      (incomingAuthority carry satisfies)) :
    PreviousStatePlaced (artifact := artifact) (assignment := assignment)
      previous := by
  intro lane
  calc
    assignment (artifact.layouts.priorPublicStateDigestColumn lane) =
        assignment (artifact.layouts.priorStateBoundary.outgoingColumn lane) := by
      rw [artifact.layoutsValid.priorBoundaryUsesPublicState]
    _ = assignment
        (artifact.layouts.priorStateBoundary.incomingColumn lane) :=
      StateAuthorityBoundaryRows.columns_eq_of_rows call.canonicalAssignment
        call.one (artifact.priorStateBoundary_satisfied satisfies) lane
    _ = assignment
        (List.getD
          artifact.layouts.priorStateLink.stateOutput.hash.stateOutput.trace.outputColumns
          lane.val 0) := by
      rw [artifact.layoutsValid.priorBoundaryUsesRecomputedState]
    _ = (incomingAuthority carry satisfies).digest lane :=
      (incomingAuthority_digest_eq_columns carry satisfies lane).symm
    _ = previous.digest lane :=
      congrFun (StateAuthorityBoundaryRows.digest_eq_of_same same) lane |>.symm

/-- The terminal claim's row-derived digest is the canonical field view of
the same normalized incoming authority. -/
theorem priorOutputDigest_eq_canonicalDigest
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    PriorStateLinkRows.outputDigest artifact.layouts.priorStateLink assignment
        call.canonicalAssignment =
      StateAuthorityFullClaim.canonicalDigest
        (incomingAuthority carry satisfies) := by
  funext lane
  apply Subtype.ext
  exact (incomingAuthority_digest_eq_columns carry satisfies lane).symm

/-- Mandatory terminal rows make the trailing receipt carry the complete
normalized terminal input state. -/
theorem exactReceiptCarriesIncomingAuthority
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityFullClaim.Carries (incomingAuthority carry satisfies)
      (call.receiptOfRows satisfies).envelope := by
  exact StateAuthorityFullClaim.carries_of_ccsPublicExact
    (call.priorStateCcsPublicExact satisfies)
    (priorOutputDigest_eq_canonicalDigest carry satisfies)

end Nightstream.Implementation.Nebula.TerminalManifestStateAuthority
