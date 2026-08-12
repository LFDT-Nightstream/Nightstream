import Nightstream.Implementation.NebulaV2.MemoryOpenSegmentBlockRows
import Nightstream.Implementation.NebulaV2.SelectorGatedRows
import Nightstream.Protocol.NebulaV2.AugmentedLifecycle

/-!
Contract: exact nonterminal continuation after one delayed memory transition.

Assurance tier: implementation model.

Owns both fixed-shape branches. An active intermediate carry is copied in
full. A closed intermediate carry selects the complete segment-opening row
block and becomes the active carry for the next segment in the same augmented
invocation. It proves row soundness and honest completeness for both branches.

Does not own either carry parser, the preceding checked-step transition,
verifier-authority source rows, absolute generated columns, or terminal
behavior. A terminal invocation must omit this block and retain its closed
intermediate carry.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.MemorySegmentContinuationRows

open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.SuperNeo.Concrete

local instance concreteKOne : One K := ⟨K.one⟩

structure Layout where
  intermediate : MemoryCarryPublicRows.Layout
  outgoing : MemoryCarryPublicRows.Layout
  opening : MemoryOpenSegmentBlockRows.Layout
  productColumn : Nat → Nat
  outputColumn : Nat → Nat

def Layout.gate (layout : Layout) :
    SelectorGatedRows.Layout (MemoryOpenSegmentBlockRows.rows layout.opening) :=
  { selectorColumn := layout.intermediate.carry.fieldColumn .phase
    productColumn := layout.productColumn
    outputColumn := layout.outputColumn }

structure Layout.Valid (layout : Layout) : Prop where
  opening : layout.opening.Valid
  openingBefore : layout.opening.before = layout.intermediate
  openingAfter : layout.opening.after = layout.outgoing

def carryPairs (layout : Layout) : List (Nat × Nat) :=
  (List.range MemoryWireGeometry.carryBits).map fun index =>
    (layout.outgoing.carry.publicBitStart + index,
      layout.intermediate.carry.publicBitStart + index)

def copyRows (layout : Layout) : List Row :=
  ConditionalEqualityOneRows.rows
    (layout.intermediate.carry.fieldColumn .phase) (carryPairs layout)

def openingRows (layout : Layout) : List Row :=
  SelectorGatedRows.rows .zero layout.gate

def rows (layout : Layout) : List Row :=
  copyRows layout ++ openingRows layout

theorem carryPairs_length (layout : Layout) :
    (carryPairs layout).length = MemoryWireGeometry.carryBits := by
  simp [carryPairs]

theorem copyRows_length (layout : Layout) :
    (copyRows layout).length = MemoryWireGeometry.carryBits := by
  rw [copyRows, ConditionalEqualityOneRows.rows_length,
    carryPairs_length]

theorem openingRows_length (layout : Layout) :
    (openingRows layout).length =
      3 * (MemoryOpenSegmentBlockRows.rows layout.opening).length := by
  exact SelectorGatedRows.rows_length .zero layout.gate

theorem rows_length_exact {layout : Layout} (valid : layout.Valid) :
    (rows layout).length = 38065 := by
  rw [rows, List.length_append, copyRows_length, openingRows_length,
    MemoryOpenSegmentBlockRows.rows_length_exact valid.opening]
  rw [MemoryWireGeometry.carryBits_exact]

private theorem copy_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (copyRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem opening_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (openingRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

private theorem selector_eq_phaseValue
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {intermediate : MemoryCarryCodec.Value}
    (parsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.intermediate
      assignment headers intermediate) :
    assignment (layout.intermediate.carry.fieldColumn .phase) =
      phaseValue intermediate.phase := by
  simpa [Value.fieldValue] using parsed.placed .phase

private theorem active_copy_block
    {layout : Layout} {assignment : Nat → Nat}
    {intermediateBlock outgoingBlock : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (intermediateBits : PublicBitBlock.Placed
      layout.intermediate.publicBits assignment intermediateBlock)
    (outgoingBits : PublicBitBlock.Placed
      layout.outgoing.publicBits assignment outgoingBlock)
    (selectorOne :
      assignment (layout.intermediate.carry.fieldColumn .phase) = 1)
    (holds : Satisfies (rows layout) assignment) :
    outgoingBlock = intermediateBlock := by
  have equalities := ConditionalEqualityOneRows.rows_sound_one canonical one
    selectorOne (copy_rows_hold holds)
  apply Subtype.ext
  apply List.ext_getElem
  · exact outgoingBlock.property.1.trans intermediateBlock.property.1.symm
  · intro index outgoingBound intermediateBound
    have indexBound : index < MemoryWireGeometry.carryBits := by
      simpa [outgoingBlock.property.1] using outgoingBound
    have member :
        (layout.outgoing.carry.publicBitStart + index,
          layout.intermediate.carry.publicBitStart + index) ∈
            carryPairs layout := by
      exact List.mem_map.mpr ⟨index, by simp [indexBound], rfl⟩
    exact (outgoingBits index indexBound).symm |>.trans
      ((equalities _ member).trans (intermediateBits index indexBound))

private theorem active_copy_value
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {intermediateBlock outgoingBlock : MemoryCarryParser.Block}
    {intermediate outgoing : MemoryCarryCodec.Value}
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
    (phaseActive : intermediate.phase = .active)
    (holds : Satisfies (rows layout) assignment) :
    outgoing = intermediate := by
  have selectorOne :
      assignment (layout.intermediate.carry.fieldColumn .phase) = 1 := by
    rw [selector_eq_phaseValue intermediateParsed, phaseActive]
    rfl
  have blockExact := active_copy_block canonical one intermediateBits
    outgoingBits selectorOne holds
  subst outgoingBlock
  exact Option.some.inj (outgoingAccepted.symm.trans intermediateAccepted)

/-- Every satisfying nonterminal continuation implements exactly one of the
two independent augmented-lifecycle constructors. The desired continuation
is not an assumption. -/
theorem sound
    {layout : Layout} (valid : layout.Valid)
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
    (holds : Satisfies (rows layout) assignment) :
    Continues
      (fun closed precommit activeAccessCount =>
        MemoryOpenSegment.derive authority closed precommit activeAccessCount)
      headers
      (MemoryCarryParser.semanticCarry intermediate
        intermediateParsed.parserCanonical.stepIndex)
      (MemoryCarryParser.semanticCarry outgoing
        outgoingParsed.parserCanonical.stepIndex) := by
  cases phaseExact : intermediate.phase with
  | active =>
      have valueExact := active_copy_value canonical one intermediateBits
        outgoingBits intermediateAccepted outgoingAccepted intermediateParsed
        phaseExact holds
      subst outgoing
      simpa [MemoryCarryParser.semanticCarry, phaseExact,
        MemoryOpenSegmentSound.activeOfWire] using
        (Continues.interior
          (derive := fun closed precommit activeAccessCount =>
            MemoryOpenSegment.derive authority closed precommit
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
          (MemoryOpenSegmentBlockRows.rows layout.opening) assignment :=
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
      rcases MemoryOpenSegmentBlockRows.sound valid.opening canonical one
          openingBefore openingAfter authorityPlaced sourceHolds with
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
            MemoryOpenSegment.openCarry authority headers outgoing.dPre
              outgoing.segmentActiveAccessCount
              (MemoryOpenSegmentSound.closedOfWire intermediate) canOpen
              activeCountInRange endTimestampInRange := by
        simpa [MemoryCarryParser.semanticCarry, afterActive] using openingExact
      rw [semanticBefore, semanticAfter]
      exact Continues.boundary
        (derive := fun closed precommit activeAccessCount =>
          MemoryOpenSegment.derive authority closed precommit
            activeAccessCount)
        (headers := headers)
        (MemoryOpenSegmentSound.closedOfWire intermediate)
        outgoing.dPre outgoing.segmentActiveAccessCount canOpen
        activeCountInRange endTimestampInRange

/-- Honest active-branch witnesses satisfy the fixed continuation relation.
The opening source rows can be false in this branch; only their deterministic
auxiliary values remain constrained. -/
theorem rows_complete_active
    {layout : Layout} {assignment : Nat → Nat}
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

/-- Honest closed-branch witnesses satisfy the fixed continuation relation. -/
theorem rows_complete_closed
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selectorZero :
      assignment (layout.intermediate.carry.fieldColumn .phase) = 0)
    (openingHolds : Satisfies
      (MemoryOpenSegmentBlockRows.rows layout.opening) assignment)
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

end Nightstream.Implementation.NebulaV2.MemorySegmentContinuationRows
