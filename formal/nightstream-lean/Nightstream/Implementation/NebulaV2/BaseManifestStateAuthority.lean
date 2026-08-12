import Nightstream.Implementation.NebulaV2.BaseManifestSchema
import Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim

/-!
Contract: row-derived memory opening and outgoing state authority for one V2 base invocation.

Assurance tier: implementation schema and cryptographic boundary.

Owns parsing of the initial and outgoing memory carries, derivation of the
verifier-authoritative chain-start fields, exact segment opening with the
artifact-profile Poseidon2 transcript, and extraction of the normalized outgoing state
authority from the same satisfying base assignment.

The `Call.openingAuthority` value is only a typed value placed in the local
segment-open rows. This module does not prove that it is the canonical base
challenge authority. That equality is a separate generated-row refinement
obligation owned by the named `BaseManifestSchema.Owner.challengeAuthority`
family.

Does not own application and fresh-claim construction, accumulator or control
refinement, computation of `D_init` from the public statement, a generated
base artifact, the next delayed boundary, Poseidon2 collision resistance,
recursive-size closure, or Rust conformance.

Emits constraints: no new rows. It projects mandatory base-manifest rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.BaseManifestStateAuthority

open Nightstream.Implementation.NebulaV2.BaseManifestSchema
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime

/-- Typed values placed in one base assignment. The initial memory root is a
verifier-owned source. `openingAuthority` is not authority-bearing until the
separate base challenge-authority rows prove its exact derivation. -/
structure Call
    {widths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact widths) (assignment : Nat → Nat) where
  canonicalAssignment : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  headers : ChainHeaders Digest.Value
  initialMemoryRoot : Digest.Value
  openingAuthority : MemoryOpenSegment.Authority
  initialBlock : MemoryCarryParser.Block
  outgoingBlock : MemoryCarryParser.Block
  initialBitsPlaced :
    PublicBitBlock.Placed artifact.layouts.initialMemoryCarry.publicBits
      assignment initialBlock
  outgoingBitsPlaced :
    PublicBitBlock.Placed artifact.layouts.outgoingMemoryCarry.publicBits
      assignment outgoingBlock
  initialHeadersPlaced :
    MemoryCarryRows.HeadersPlaced artifact.layouts.initialMemoryCarry.carry
      assignment headers
  outgoingHeadersPlaced :
    MemoryCarryRows.HeadersPlaced artifact.layouts.outgoingMemoryCarry.carry
      assignment headers
  initialRootPlaced :
    InitialMemoryCarryRows.InitialMemoryRootPlaced
      artifact.layouts.initialAuthority assignment initialMemoryRoot
  openingAuthorityPlaced :
    MemoryOpenSegmentSound.AuthorityPlaced artifact.layouts.opening assignment
      openingAuthority

namespace Call

def initialValue
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment) : MemoryCarryCodec.Value :=
  MemoryCarryParser.decodedValue call.initialBlock

def outgoingValue
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment) : MemoryCarryCodec.Value :=
  MemoryCarryParser.decodedValue call.outgoingBlock

/-- The canonical base carry is parsed by the mandatory rows. Parser success
is not caller authority. -/
theorem initialAccepted
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryParser.parse call.headers call.initialBlock =
      some call.initialValue :=
  MemoryCarryPublicRows.rows_force_parse call.canonicalAssignment call.one
    call.initialBitsPlaced call.initialHeadersPlaced
    (artifact.initialCarry_satisfied satisfies)

/-- The opened base carry is parsed by the mandatory rows. Parser success is
not caller authority. -/
theorem outgoingAccepted
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryParser.parse call.headers call.outgoingBlock =
      some call.outgoingValue :=
  MemoryCarryPublicRows.rows_force_parse call.canonicalAssignment call.one
    call.outgoingBitsPlaced call.outgoingHeadersPlaced
    (artifact.outgoingCarry_satisfied satisfies)

def initialCarryColumnsMatch
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.initialMemoryCarry assignment call.headers
      call.initialValue := by
  exact MemoryCarryPublicRows.rows_force_parsed_columns_match
    call.canonicalAssignment call.one call.initialBitsPlaced
    call.initialHeadersPlaced (artifact.initialCarry_satisfied satisfies)

def outgoingCarryColumnsMatch
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.outgoingMemoryCarry assignment call.headers
      call.outgoingValue := by
  exact MemoryCarryPublicRows.rows_force_parsed_columns_match
    call.canonicalAssignment call.one call.outgoingBitsPlaced
    call.outgoingHeadersPlaced (artifact.outgoingCarry_satisfied satisfies)

/-- The compact header rows fix both chain headers to the profile and plan.
The typed `headers` object cannot supply a different value. -/
theorem headersExact
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    (∀ lane : Fin 4,
      (call.headers.operations.lanes lane).val =
        CompactChainPoseidonRows.pureHash
          (.header .operations artifact.seedManifest.profile
            artifact.seedManifest.plan) lane.val) ∧
      (∀ lane : Fin 4,
        (call.headers.memory.lanes lane).val =
          CompactChainPoseidonRows.pureHash
            (.header .memory artifact.seedManifest.profile
              artifact.seedManifest.plan) lane.val) := by
  have outputs := CompactChainHeaderRows.outputs_exact
    artifact.layoutsValid.compactHeadersValid call.canonicalAssignment
    call.one (artifact.compactHeaders_satisfied satisfies)
  constructor
  · intro lane
    calc
      (call.headers.operations.lanes lane).val =
          assignment
            (artifact.layouts.initialMemoryCarry.carry.headerColumn
              .operations lane) :=
        (call.initialHeadersPlaced .operations lane).symm
      _ = assignment
          (artifact.layouts.compactHeaders.operations.digestColumn lane) := by
        rw [artifact.layoutsValid.operationsHeaderUsesInitialCarry]
      _ = CompactChainPoseidonRows.pureHash
          (.header .operations artifact.seedManifest.profile
            artifact.seedManifest.plan) lane.val :=
        outputs.1 lane
  · intro lane
    calc
      (call.headers.memory.lanes lane).val =
          assignment
            (artifact.layouts.initialMemoryCarry.carry.headerColumn
              .initialSnapshot lane) :=
        (call.initialHeadersPlaced .initialSnapshot lane).symm
      _ = assignment
          (artifact.layouts.compactHeaders.memory.digestColumn lane) := by
        rw [artifact.layoutsValid.memoryHeaderUsesInitialCarryInitial]
      _ = CompactChainPoseidonRows.pureHash
          (.header .memory artifact.seedManifest.profile
            artifact.seedManifest.plan) lane.val :=
        outputs.2 lane

/-- Mandatory rows derive the canonical closed chain-start carry with the
exact verifier-owned initial memory root. -/
theorem initialExact
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    InitialMemoryCarryRows.Exact call.initialValue call.initialMemoryRoot := by
  have parsed := call.initialCarryColumnsMatch satisfies
  rw [← artifact.layoutsValid.initialAuthorityUsesInitialCarry] at parsed
  exact InitialMemoryCarryRows.sound call.canonicalAssignment call.one parsed
    call.initialRootPlaced (artifact.initialAuthority_satisfied satisfies)

/-- The base segment opens from the exact chain-start carry and produces the
exact active outgoing carry. All range and transcript facts are conclusions
of the local rows. -/
theorem opensExactInitialCarry
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    ∃ (canOpen :
        (MemoryOpenSegmentSound.closedOfWire call.initialValue).CanOpen)
      (activeCountInRange :
        call.outgoingValue.segmentActiveAccessCount < operationCountLimit)
      (endTimestampInRange :
        (MemoryOpenSegmentSound.closedOfWire call.initialValue).globalTimestamp +
            call.outgoingValue.segmentActiveAccessCount < timestampLimit)
      (stepBound :
        call.outgoingValue.stepIndex < Lifecycle.claimsPerSegment),
      call.initialValue.phase = .closed ∧
        call.outgoingValue.phase = .active ∧
        Carry.active
            (MemoryOpenSegmentSound.activeOfWire call.outgoingValue stepBound) =
          MemoryOpenSegment.openCarryFor artifact.profile call.openingAuthority
            call.headers
            call.outgoingValue.dPre
            call.outgoingValue.segmentActiveAccessCount
            (MemoryOpenSegmentSound.closedOfWire call.initialValue) canOpen
            activeCountInRange endTimestampInRange := by
  have beforeParsed := call.initialCarryColumnsMatch satisfies
  have afterParsed := call.outgoingCarryColumnsMatch satisfies
  rw [← artifact.layoutsValid.openingUsesInitialCarry] at beforeParsed
  rw [← artifact.layoutsValid.openingUsesOutgoingCarry] at afterParsed
  exact MemoryOpenSegmentBlockRows.ProfileIndexed.sound
    artifact.layoutsValid.profileCanonical artifact.layoutsValid.openingValid
    call.canonicalAssignment call.one beforeParsed afterParsed
    call.openingAuthorityPlaced (artifact.opening_satisfied satisfies)

/-- The same outgoing carry bits validated by the parser are absorbed by the
mandatory two-stage state hash. -/
theorem outgoingStateCarryPlaced
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment) :
    PublicBitBlock.Placed
      artifact.layouts.stateOutput.hash.carry.frame.packing.publicBits
      assignment call.outgoingBlock := by
  rw [← artifact.layoutsValid.carryBitsSharedWithStateOutput]
  exact call.outgoingBitsPlaced

/-- Normalized authority exported by this base invocation. -/
def outgoingAuthority
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    StateAuthorityBoundaryRows.Authority where
  payload := StateOutputAuthorityRows.payload
    artifact.layouts.stateOutput.authority assignment
  carryBlock := call.outgoingBlock
  frameCanonical :=
    AuthoritativeStateOutputBinding.typedFrame_canonical_of_rows
      artifact.layoutsValid.stateOutputValid call.canonicalAssignment call.one
      call.outgoingStateCarryPlaced (artifact.stateOutput_satisfied satisfies)

/-- The normalized digest is exactly the mandatory base state-output columns. -/
theorem outgoingAuthority_digest_eq_columns
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (call : Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    ∀ lane : Fin 4,
      (call.outgoingAuthority satisfies).digest lane =
        assignment
          (artifact.layouts.stateOutput.hash.stateOutput.trace.outputColumns.getD
            lane.val 0) := by
  have output :=
    AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest
      artifact.layoutsValid.stateOutputValid call.canonicalAssignment call.one
      call.outgoingStateCarryPlaced (artifact.stateOutput_satisfied satisfies)
  intro lane
  simpa [outgoingAuthority, StateAuthorityBoundaryRows.Authority.digest,
    AuthoritativeStateOutputBinding.typedDigest,
    AuthoritativeStateOutputBinding.typedFrame,
    StateOutputPoseidonBinding.outerHash] using (output lane).symm

end Call

end Nightstream.Implementation.NebulaV2.BaseManifestStateAuthority
