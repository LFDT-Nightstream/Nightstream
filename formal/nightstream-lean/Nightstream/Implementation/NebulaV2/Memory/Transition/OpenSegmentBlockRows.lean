import Nightstream.Implementation.NebulaV2.Memory.Transition.OpenSegmentSound

/-!
Contract: complete local row block for one production V2 segment opening.

Assurance tier: implementation model.

Owns the union of the exact 72 closed-to-active transition rows and the exact
11,472-row linked Poseidon2 memory transcript. It proves local soundness and
honest completeness for that union.

Does not own either carry parser, verifier-authority source rows, precommit
sequence extraction, branch selection, absolute generated columns, or Rust
conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryOpenSegmentBlockRows

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

abbrev Layout := MemoryOpenSegmentRows.Layout

def rows (layout : Layout) : List Row :=
  MemoryOpenSegmentRows.rows layout ++
    MemoryTranscriptPoseidonRows.rows layout.transcript

structure Layout.Valid (layout : Layout) : Prop where
  transcript : layout.transcript.Valid

theorem rows_length_exact
    {layout : Layout} (valid : layout.Valid) :
    (rows layout).length = 11544 := by
  simp [rows, MemoryOpenSegmentRows.rows_length_exact,
    MemoryTranscriptPoseidonRows.rows_length_exact valid.transcript]

private theorem local_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryOpenSegmentRows.rows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem transcript_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryTranscriptPoseidonRows.rows layout.transcript)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

/-- Satisfaction of the complete local block derives the exact production
segment opening. No challenge or transition conclusion is an assumption. -/
theorem sound
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (authorityPlaced :
      MemoryOpenSegmentSound.AuthorityPlaced layout assignment authority)
    (holds : Satisfies (rows layout) assignment) :
    ∃ (canOpen : (MemoryOpenSegmentSound.closedOfWire before).CanOpen)
      (activeCountInRange :
        after.segmentActiveAccessCount < operationCountLimit)
      (endTimestampInRange :
        (MemoryOpenSegmentSound.closedOfWire before).globalTimestamp +
            after.segmentActiveAccessCount < timestampLimit)
      (stepBound : after.stepIndex < Lifecycle.claimsPerSegment),
      before.phase = .closed ∧ after.phase = .active ∧
        Carry.active
            (MemoryOpenSegmentSound.activeOfWire after stepBound) =
          MemoryOpenSegment.openCarry authority headers after.dPre
            after.segmentActiveAccessCount
            (MemoryOpenSegmentSound.closedOfWire before) canOpen
            activeCountInRange endTimestampInRange := by
  exact MemoryOpenSegmentSound.rows_sound valid.transcript canonical one
    beforeParsed afterParsed authorityPlaced (local_rows_hold holds)
    (transcript_rows_hold holds)

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (input : MemoryTranscriptHashFrame.Input) : Prop where
  transition : MemoryOpenSegmentRows.Honest layout assignment
  transcript : MemoryTranscriptPoseidonRows.Honest layout.transcript
    assignment input

/-- Honest local transition and transcript witnesses satisfy the complete
11,544-row block. -/
theorem complete
    {layout : Layout} {assignment : Nat → Nat}
    {input : MemoryTranscriptHashFrame.Input}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment input) :
    Satisfies (rows layout) assignment := by
  have transitionHolds := MemoryOpenSegmentRows.rows_complete canonical one
    honest.transition
  have transcriptHolds := MemoryTranscriptPoseidonRows.rows_complete canonical one
    honest.transcript
  intro row member
  rcases List.mem_append.mp member with localMember | transcriptMember
  · exact transitionHolds row localMember
  · exact transcriptHolds row transcriptMember

/-! ## Verifier-selected successor profile -/

namespace ProfileIndexed

def rows (profile : Profile.Identity) (layout : Layout) : List Row :=
  MemoryOpenSegmentRows.rows layout ++
    MemoryTranscriptPoseidonRows.ProfileIndexed.rows profile layout.transcript

structure Valid (profile : Profile.Identity) (layout : Layout) : Prop where
  transcript :
    MemoryTranscriptPoseidonRows.ProfileIndexed.Valid profile layout.transcript

theorem rows_length_exact
    {profile : Profile.Identity} {layout : Layout}
    (valid : Valid profile layout) :
    (rows profile layout).length = 11544 := by
  simp [rows, MemoryOpenSegmentRows.rows_length_exact,
    MemoryTranscriptPoseidonRows.ProfileIndexed.rows_length_exact
      valid.transcript]

private theorem local_rows_hold
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows profile layout) assignment) :
    Satisfies (MemoryOpenSegmentRows.rows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem transcript_rows_hold
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows profile layout) assignment) :
    Satisfies
      (MemoryTranscriptPoseidonRows.ProfileIndexed.rows profile
        layout.transcript) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

/-- The complete local opening block binds its challenge to the exact selected
profile. No challenge value or transition result is an assumption. -/
theorem sound
    {profile : Profile.Identity} {layout : Layout}
    (profileCanonical : MemoryTranscriptHashFrame.ProfileCanonical profile)
    (valid : Valid profile layout)
    {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (authorityPlaced :
      MemoryOpenSegmentSound.AuthorityPlaced layout assignment authority)
    (holds : Satisfies (rows profile layout) assignment) :
    ∃ (canOpen : (MemoryOpenSegmentSound.closedOfWire before).CanOpen)
      (activeCountInRange :
        after.segmentActiveAccessCount < operationCountLimit)
      (endTimestampInRange :
        (MemoryOpenSegmentSound.closedOfWire before).globalTimestamp +
            after.segmentActiveAccessCount < timestampLimit)
      (stepBound : after.stepIndex < Lifecycle.claimsPerSegment),
      before.phase = .closed ∧ after.phase = .active ∧
        Carry.active
            (MemoryOpenSegmentSound.activeOfWire after stepBound) =
          MemoryOpenSegment.openCarryFor profile authority headers after.dPre
            after.segmentActiveAccessCount
            (MemoryOpenSegmentSound.closedOfWire before) canOpen
            activeCountInRange endTimestampInRange := by
  exact MemoryOpenSegmentSound.rows_sound_for profileCanonical valid.transcript
    canonical one beforeParsed afterParsed authorityPlaced
    (local_rows_hold holds) (transcript_rows_hold holds)

structure Honest (profile : Profile.Identity) (layout : Layout)
    (assignment : Nat → Nat)
    (input : MemoryTranscriptHashFrame.Input) : Prop where
  transition : MemoryOpenSegmentRows.Honest layout assignment
  transcript : MemoryTranscriptPoseidonRows.ProfileIndexed.Honest profile
    layout.transcript assignment input

theorem complete
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    {input : MemoryTranscriptHashFrame.Input}
    (profileCanonical : MemoryTranscriptHashFrame.ProfileCanonical profile)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest profile layout assignment input) :
    Satisfies (rows profile layout) assignment := by
  have transitionHolds := MemoryOpenSegmentRows.rows_complete canonical one
    honest.transition
  have transcriptHolds :=
    MemoryTranscriptPoseidonRows.ProfileIndexed.rows_complete profileCanonical
      canonical one honest.transcript
  intro row member
  rcases List.mem_append.mp member with localMember | transcriptMember
  · exact transitionHolds row localMember
  · exact transcriptHolds row transcriptMember

end ProfileIndexed

end Nightstream.Implementation.NebulaV2.MemoryOpenSegmentBlockRows
