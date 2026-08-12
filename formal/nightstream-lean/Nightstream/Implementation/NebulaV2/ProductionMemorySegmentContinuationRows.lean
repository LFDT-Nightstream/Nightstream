import Nightstream.Implementation.NebulaV2.MemoryOpenSegmentBlockRows
import Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows
import Nightstream.Implementation.NebulaV2.ProductionMemoryTranscriptHashFrame
import Nightstream.Implementation.NebulaV2.SelectorGatedRows
import Nightstream.Protocol.NebulaV2.AugmentedLifecycle

/-!
Contract: exact field-native nonterminal memory continuation.

An active intermediate carry is copied through all 59 typed field-native
coordinates. A closed intermediate carry selects the complete segment-open
relation and produces the active carry for the next segment. The phase is
already Boolean in each production carry decoder.

This is the production-profile counterpart of the 3,433-bit reference
continuation. It does not use a carry digest or an independently supplied
typed transition.

Does not own either production carry decoder, the preceding checked batch,
verifier-authority source rows, absolute generated columns, terminal behavior,
Rust refinement, or Poseidon2 security.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionMemorySegmentContinuationRows

open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

local instance concreteKOne : One K := ⟨K.one⟩

structure Layout (candidate : Id) where
  intermediate : ProductionMemoryCarryRows.Layout
  outgoing : ProductionMemoryCarryRows.Layout
  opening : MemoryOpenSegmentBlockRows.Layout
  productColumn : Nat → Nat
  outputColumn : Nat → Nat

def Layout.gate {candidate : Id} (layout : Layout candidate) :
    SelectorGatedRows.Layout
      (MemoryOpenSegmentBlockRows.ProfileIndexed.rows (identity candidate)
        layout.opening) :=
  { selectorColumn := layout.intermediate.carry.fieldColumn .phase
    productColumn := layout.productColumn
    outputColumn := layout.outputColumn }

structure Layout.Valid {candidate : Id} (layout : Layout candidate) : Prop where
  opening : MemoryOpenSegmentBlockRows.ProfileIndexed.Valid
    (identity candidate) layout.opening
  openingBefore : layout.opening.before = layout.intermediate.reference
  openingAfter : layout.opening.after = layout.outgoing.reference

/-- The schema is the exact 7-counter, 8-challenge, 16-product, and 28-root
field order used by the production state hash. -/
def carryPairs {candidate : Id} (layout : Layout candidate) : List (Nat × Nat) :=
  MemoryCarryCodec.schema.map fun tag =>
    (layout.outgoing.carry.fieldColumn tag,
      layout.intermediate.carry.fieldColumn tag)

def copyRows {candidate : Id} (layout : Layout candidate) : List Row :=
  ConditionalEqualityOneRows.rows
    (layout.intermediate.carry.fieldColumn .phase) (carryPairs layout)

def openingRows {candidate : Id} (layout : Layout candidate) : List Row :=
  SelectorGatedRows.rows .zero layout.gate

def rows {candidate : Id} (layout : Layout candidate) : List Row :=
  copyRows layout ++ openingRows layout

def rowCount : Nat := 34691

theorem carryPairs_length {candidate : Id} (layout : Layout candidate) :
    (carryPairs layout).length = 59 := by
  rw [carryPairs, List.length_map]
  decide

theorem copyRows_length {candidate : Id} (layout : Layout candidate) :
    (copyRows layout).length = 59 := by
  rw [copyRows, ConditionalEqualityOneRows.rows_length,
    carryPairs_length]

theorem openingRows_length {candidate : Id} (layout : Layout candidate) :
    (openingRows layout).length =
      3 * (MemoryOpenSegmentBlockRows.ProfileIndexed.rows (identity candidate)
        layout.opening).length := by
  exact SelectorGatedRows.rows_length .zero layout.gate

theorem rows_length_exact {candidate : Id} {layout : Layout candidate}
    (valid : layout.Valid) :
    (rows layout).length = rowCount := by
  rw [rows, List.length_append, copyRows_length, openingRows_length,
    MemoryOpenSegmentBlockRows.ProfileIndexed.rows_length_exact valid.opening]
  rfl

private theorem copy_rows_hold
    {candidate : Id} {layout : Layout candidate} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (copyRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem opening_rows_hold
    {candidate : Id} {layout : Layout candidate} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (openingRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

private theorem selector_eq_phaseValue
    {candidate : Id} {layout : Layout candidate} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {intermediate : MemoryCarryCodec.Value}
    (parsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.intermediate.reference assignment headers intermediate) :
    assignment (layout.intermediate.carry.fieldColumn .phase) =
      phaseValue intermediate.phase := by
  simpa [Value.fieldValue] using parsed.placed .phase

/-- In the active branch, all 59 typed fields are equal. This is stronger
than equality of a carry digest. -/
private theorem active_copy_value
    {candidate : Id} {layout : Layout candidate} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {intermediate outgoing : MemoryCarryCodec.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (intermediateParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.intermediate.reference assignment headers intermediate)
    (outgoingParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.outgoing.reference assignment headers outgoing)
    (phaseActive : intermediate.phase = .active)
    (holds : Satisfies (rows layout) assignment) :
    outgoing = intermediate := by
  have selectorOne :
      assignment (layout.intermediate.carry.fieldColumn .phase) = 1 := by
    rw [selector_eq_phaseValue intermediateParsed, phaseActive]
    rfl
  have equalities := ConditionalEqualityOneRows.rows_sound_one canonical one
    selectorOne (copy_rows_hold holds)
  apply Value.fieldValue_injective
  funext tag
  have member :
      (layout.outgoing.carry.fieldColumn tag,
        layout.intermediate.carry.fieldColumn tag) ∈ carryPairs layout := by
    exact List.mem_map.mpr ⟨tag, tag.mem_schema, rfl⟩
  calc
    outgoing.fieldValue tag =
        assignment (layout.outgoing.carry.fieldColumn tag) :=
      (outgoingParsed.placed tag).symm
    _ = assignment (layout.intermediate.carry.fieldColumn tag) :=
      equalities _ member
    _ = intermediate.fieldValue tag := intermediateParsed.placed tag

/-- Every satisfying production continuation is exactly an active copy or
the complete closed-to-active segment opening. -/
theorem sound
    {candidate : Id} {layout : Layout candidate} (valid : layout.Valid)
    {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {intermediate outgoing : MemoryCarryCodec.Value}
    {authority : MemoryOpenSegment.Authority}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (intermediateParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.intermediate.reference assignment headers intermediate)
    (outgoingParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.outgoing.reference assignment headers outgoing)
    (authorityPlaced :
      MemoryOpenSegmentSound.AuthorityPlaced layout.opening assignment
        authority)
    (holds : Satisfies (rows layout) assignment) :
    Continues
      (fun closed precommit activeAccessCount =>
        MemoryOpenSegment.deriveFor (identity candidate) authority closed
          precommit activeAccessCount)
      headers
      (MemoryCarryParser.semanticCarry intermediate
        intermediateParsed.parserCanonical.stepIndex)
      (MemoryCarryParser.semanticCarry outgoing
        outgoingParsed.parserCanonical.stepIndex) := by
  cases phaseExact : intermediate.phase with
  | active =>
      have valueExact := active_copy_value canonical one intermediateParsed
        outgoingParsed phaseExact holds
      subst outgoing
      simpa [MemoryCarryParser.semanticCarry, phaseExact,
        MemoryOpenSegmentSound.activeOfWire] using
        (Continues.interior
          (derive := fun closed precommit activeAccessCount =>
            MemoryOpenSegment.deriveFor (identity candidate) authority closed precommit
              activeAccessCount)
          (headers := headers)
          (MemoryOpenSegmentSound.activeOfWire intermediate
            intermediateParsed.parserCanonical.stepIndex))
  | closed =>
      have selectorZero :
          assignment (layout.intermediate.carry.fieldColumn .phase) = 0 := by
        rw [selector_eq_phaseValue intermediateParsed, phaseExact]
        rfl
      have sourceHolds : Satisfies
          (MemoryOpenSegmentBlockRows.ProfileIndexed.rows (identity candidate)
            layout.opening) assignment :=
        SelectorGatedRows.rows_sound_selected canonical one selectorZero
          (opening_rows_hold holds)
      have openingBefore :
          MemoryCarryPublicRows.ParsedColumnsMatch layout.opening.before
            assignment headers intermediate := by
        simpa [valid.openingBefore] using intermediateParsed
      have openingAfter :
          MemoryCarryPublicRows.ParsedColumnsMatch layout.opening.after
            assignment headers outgoing := by
        simpa [valid.openingAfter] using outgoingParsed
      rcases MemoryOpenSegmentBlockRows.ProfileIndexed.sound
          (ProductionMemoryTranscriptHashFrame.candidateProfileCanonical
            candidate)
          valid.opening canonical one openingBefore openingAfter
          authorityPlaced sourceHolds with
        ⟨canOpen, activeCountInRange, endTimestampInRange, stepBound,
          _beforeClosed, afterActive, openingExact⟩
      have semanticBefore :
          MemoryCarryParser.semanticCarry intermediate
              intermediateParsed.parserCanonical.stepIndex =
            .closed (MemoryOpenSegmentSound.closedOfWire intermediate) := by
        simp [MemoryCarryParser.semanticCarry, phaseExact,
          MemoryOpenSegmentSound.closedOfWire]
      have semanticAfter :
          MemoryCarryParser.semanticCarry outgoing
              outgoingParsed.parserCanonical.stepIndex =
            MemoryOpenSegment.openCarryFor (identity candidate) authority
              headers outgoing.dPre
              outgoing.segmentActiveAccessCount
              (MemoryOpenSegmentSound.closedOfWire intermediate) canOpen
              activeCountInRange endTimestampInRange := by
        simpa [MemoryCarryParser.semanticCarry, afterActive] using openingExact
      rw [semanticBefore, semanticAfter]
      simpa [MemoryOpenSegment.openCarryFor] using
        (Continues.boundary
          (derive := fun closed precommit activeAccessCount =>
            MemoryOpenSegment.deriveFor (identity candidate) authority closed precommit
              activeAccessCount)
          (headers := headers)
          (MemoryOpenSegmentSound.closedOfWire intermediate)
          outgoing.dPre outgoing.segmentActiveAccessCount canOpen
          activeCountInRange endTimestampInRange)

/-- Honest active-copy witnesses satisfy the fixed production relation. -/
theorem rows_complete_active
    {candidate : Id} {layout : Layout candidate} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selectorOne :
      assignment (layout.intermediate.carry.fieldColumn .phase) = 1)
    (copies : ∀ pair ∈ carryPairs layout,
      assignment pair.1 = assignment pair.2)
    (auxiliaries : SelectorGatedRows.AuxiliariesPlaced layout.gate assignment) :
    Satisfies (rows layout) assignment := by
  have copyHolds := ConditionalEqualityOneRows.rows_complete_one canonical one
    selectorOne copies
  have gateUnselected : assignment layout.gate.selectorColumn = 1 := by
    simpa [Layout.gate] using selectorOne
  have openingHolds := SelectorGatedRows.rows_complete_unselected
    (when := .zero) canonical one gateUnselected auxiliaries
  intro row member
  rcases List.mem_append.mp member with copyMember | openingMember
  · exact copyHolds row copyMember
  · exact openingHolds row openingMember

/-- Honest closed-open witnesses satisfy the fixed production relation. -/
theorem rows_complete_closed
    {candidate : Id} {layout : Layout candidate} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selectorZero :
      assignment (layout.intermediate.carry.fieldColumn .phase) = 0)
    (openingHolds : Satisfies
      (MemoryOpenSegmentBlockRows.ProfileIndexed.rows (identity candidate)
        layout.opening) assignment)
    (auxiliaries : SelectorGatedRows.AuxiliariesPlaced layout.gate assignment) :
    Satisfies (rows layout) assignment := by
  have copyHolds := ConditionalEqualityOneRows.rows_complete_zero selectorZero
    (pairs := carryPairs layout)
  have gateSelected : assignment layout.gate.selectorColumn = 0 := by
    simpa [Layout.gate] using selectorZero
  have gatedHolds := SelectorGatedRows.rows_complete_selected
    (when := .zero) canonical one gateSelected openingHolds auxiliaries
  intro row member
  rcases List.mem_append.mp member with copyMember | openingMember
  · exact copyHolds row copyMember
  · exact gatedHolds row openingMember

end Nightstream.Implementation.NebulaV2.ProductionMemorySegmentContinuationRows
