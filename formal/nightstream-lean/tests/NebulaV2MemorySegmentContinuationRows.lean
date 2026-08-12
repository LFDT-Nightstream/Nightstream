import Nightstream.Implementation.NebulaV2.MemorySegmentContinuationRows

/-! Focused gates for the exact nonterminal segment continuation. -/

set_option autoImplicit false

namespace tests.NebulaV2MemorySegmentContinuationRows

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.SuperNeo.Concrete

local instance concreteKOne : One K := ⟨K.one⟩

theorem exact_row_count
    {layout : MemorySegmentContinuationRows.Layout}
    (valid : layout.Valid) :
    (MemorySegmentContinuationRows.rows layout).length = 38065 :=
  MemorySegmentContinuationRows.rows_length_exact valid

theorem satisfying_rows_select_exact_continuation
    {layout : MemorySegmentContinuationRows.Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {intermediateBlock outgoingBlock : MemoryCarryParser.Block}
    {intermediate outgoing : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (intermediateBits : PublicBitBlock.Placed
      layout.intermediate.publicBits assignment intermediateBlock)
    (outgoingBits : PublicBitBlock.Placed
      layout.outgoing.publicBits assignment outgoingBlock)
    (intermediateAccepted :
      MemoryCarryParser.parse headers intermediateBlock = some intermediate)
    (outgoingAccepted :
      MemoryCarryParser.parse headers outgoingBlock = some outgoing)
    (intermediateParsed :
      MemoryCarryPublicRows.ParsedColumnsMatch layout.intermediate assignment
        headers intermediate)
    (outgoingParsed :
      MemoryCarryPublicRows.ParsedColumnsMatch layout.outgoing assignment
        headers outgoing)
    (authorityPlaced :
      MemoryOpenSegmentSound.AuthorityPlaced layout.opening assignment
        authority)
    (holds : Satisfies (MemorySegmentContinuationRows.rows layout) assignment) :
    Continues
      (fun closed precommit activeAccessCount =>
        MemoryOpenSegment.derive authority closed precommit activeAccessCount)
      headers
      (MemoryCarryParser.semanticCarry intermediate
        intermediateParsed.parserCanonical.stepIndex)
      (MemoryCarryParser.semanticCarry outgoing
        outgoingParsed.parserCanonical.stepIndex) :=
  MemorySegmentContinuationRows.sound valid canonical one intermediateBits
    outgoingBits intermediateAccepted outgoingAccepted intermediateParsed
    outgoingParsed authorityPlaced holds

theorem closed_carry_cannot_be_copied
    {derive : ClosedCarry Digest.Value → Roots Digest.Value → Nat →
      ProductState.Challenges K}
    {headers : ChainHeaders Digest.Value}
    {closed : ClosedCarry Digest.Value}
    (continuation : Continues derive headers (.closed closed) (.closed closed)) :
    False := by
  rcases continuation.outgoing_active with ⟨active, impossible⟩
  cases impossible

end tests.NebulaV2MemorySegmentContinuationRows
