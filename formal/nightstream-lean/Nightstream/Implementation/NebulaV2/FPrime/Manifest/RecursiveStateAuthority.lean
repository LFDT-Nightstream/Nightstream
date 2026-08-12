import Nightstream.Implementation.NebulaV2.FPrime.Manifest.RecursiveNifsCall
import Nightstream.Implementation.NebulaV2.FPrime.State.AuthorityFullClaim

/-!
Contract: extraction of normalized incoming and outgoing state authority from
one satisfying Nebula V2 recursive manifest.

Assurance tier: implementation schema and cryptographic boundary.

Owns the exact connection from mandatory local row families to the
column-independent state values consumed by the global delayed-boundary
theorem. Both normalized frames use the same carry blocks already validated
and consumed by the memory transition.

Does not own the generated cross-invocation wrapper boundary, the opaque
application/counter/control rows, NIFS arithmetic refinement, Poseidon2
collision resistance, or Rust conformance.

Emits constraints: no new rows. It projects mandatory manifest rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall

/-- Exact normalized state authenticated by the incoming 540-coordinate fresh
claim carrier. Its canonicality is derived from the mandatory prior-state
rows, not supplied as an abstract hash assumption. -/
def priorAuthority
    {widths : CompilerWidths} {artifact : Artifact widths}
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

/-- Exact normalized state exported by the current invocation. The carry bits
are the same outgoing bits selected by the memory continuation rows. -/
def outgoingAuthority
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityBoundaryRows.Authority where
  payload := StateOutputAuthorityRows.payload
    artifact.layouts.stateOutput.authority assignment
  carryBlock := carry.outgoingBlock
  frameCanonical :=
    AuthoritativeStateOutputBinding.typedFrame_canonical_of_rows
      artifact.layoutsValid.stateOutputValid call.canonicalAssignment call.one
      (call.outgoingStateCarryPlaced carry)
      (artifact.stateOutput_satisfied satisfies)

/-- One satisfying recursive invocation supplies both exact normalized states
needed by the global boundary chain. -/
def invocation
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
  StateAuthorityBoundaryRows.Invocation where
  incoming := priorAuthority carry satisfies
  outgoing := outgoingAuthority carry satisfies

/-- Typed interpretation of the prior public-state digest columns selected by
the generated recursive wrapper. This is placement data only; the global IVC
extraction theorem must prove that these columns are the preceding producer's
public output. -/
def PreviousStatePlaced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (previous : StateAuthorityBoundaryRows.Authority) : Prop :=
  ∀ lane,
    assignment (artifact.layouts.priorPublicStateDigestColumn lane) =
      previous.digest lane

/-- The incoming normalized digest is exactly the four row-derived output
columns already linked to the selected full claim's CCS public carrier. -/
theorem priorAuthority_digest_eq_columns
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    ∀ lane : Fin 4,
      (priorAuthority carry satisfies).digest lane =
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
  simpa [priorAuthority, StateAuthorityBoundaryRows.Authority.digest,
    AuthoritativeStateOutputBinding.typedDigest,
    AuthoritativeStateOutputBinding.typedFrame,
    StateOutputPoseidonBinding.outerHash] using (output lane).symm

/-- The mandatory wrapper layout and four equality rows construct the exact
boundary from the prior public state to the recomputed incoming typed state. -/
def boundaryFromPrevious
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (previous : StateAuthorityBoundaryRows.Authority)
    (previousPlaced : PreviousStatePlaced (artifact := artifact)
      (assignment := assignment) previous)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityBoundaryRows.Boundary previous
      (priorAuthority carry satisfies) where
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
      exact (priorAuthority_digest_eq_columns carry satisfies lane).symm
  satisfies := artifact.priorStateBoundary_satisfied satisfies

/-- If the complete producer-linked claim recovers the same typed state as
the consumer rows, the mandatory four wrapper rows also place the producer
digest in the named prior-public-state columns. No public-column placement is
an assumption of this theorem. -/
theorem previousStatePlaced_of_same
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (previous : StateAuthorityBoundaryRows.Authority)
    (satisfies : Satisfies artifact.programRows assignment)
    (same : StateAuthorityBoundaryRows.Same previous
      (priorAuthority carry satisfies)) :
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
    _ = (priorAuthority carry satisfies).digest lane :=
      (priorAuthority_digest_eq_columns carry satisfies lane).symm
    _ = previous.digest lane :=
      congrFun (StateAuthorityBoundaryRows.digest_eq_of_same same) lane |>.symm

/-- The exact digest function encoded in the selected claim is the canonical
field view of the same normalized incoming authority. -/
theorem priorOutputDigest_eq_canonicalDigest
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    PriorStateLinkRows.outputDigest artifact.layouts.priorStateLink assignment
        call.canonicalAssignment =
      StateAuthorityFullClaim.canonicalDigest
        (priorAuthority carry satisfies) := by
  funext lane
  apply Subtype.ext
  exact (priorAuthority_digest_eq_columns carry satisfies lane).symm

/-- Mandatory recursive rows make the exact selected receipt carry the
complete normalized prior state and exact memory suffix. The theorem uses the
complete 540-coordinate CCS word of the same full claim accepted by NIFS. -/
theorem exactReceiptCarriesPriorAuthority
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityFullClaim.Carries (priorAuthority carry satisfies)
      (call.receiptOfRows satisfies).envelope := by
  exact StateAuthorityFullClaim.carries_of_ccsPublicExact
    (call.priorStateCcsPublicExact satisfies)
    (priorOutputDigest_eq_canonicalDigest carry satisfies)

/-- The outgoing normalized digest is exactly the four mandatory state-output
columns exported by the current recursive invocation. -/
theorem outgoingAuthority_digest_eq_columns
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    ∀ lane : Fin 4,
      (outgoingAuthority carry satisfies).digest lane =
        assignment
          (List.getD artifact.layouts.stateOutput.hash.stateOutput.trace.outputColumns
            lane.val 0) := by
  have output :=
    AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest
      artifact.layoutsValid.stateOutputValid call.canonicalAssignment call.one
      (call.outgoingStateCarryPlaced carry)
      (artifact.stateOutput_satisfied satisfies)
  intro lane
  simpa [outgoingAuthority, StateAuthorityBoundaryRows.Authority.digest,
    AuthoritativeStateOutputBinding.typedDigest,
    AuthoritativeStateOutputBinding.typedFrame,
    StateOutputPoseidonBinding.outerHash] using (output lane).symm

end Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority
