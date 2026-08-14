import Nightstream.Implementation.Nebula.Memory.Product.BalanceRows
import Nightstream.Implementation.Nebula.Memory.Transition.TransitionRows

/-!
Contract: semantic soundness of the exact local Nebula V2 memory-transition
row program.

Assurance tier: implementation model.

Owns extraction of the active prior carry, all `MatchesActive` conditions,
the exact interior or close branch, and the final `FPrime.Consumes` result.

Does not own parser-bit placement, incoming state-hash authority, NIFS
acceptance, product-update rows, or the cryptographic product-balance event.

Emits constraints: no new rows. It interprets `MemoryTransitionRows.rows`.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.MemoryTransitionSound

open Nightstream.Implementation.Nebula.MemoryCarryCodec
open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.Nebula.MemoryTransitionRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.MemoryWireGeometry
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.SuperNeo.Concrete

private theorem subrows_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment)
    (subrows : List Row)
    (included : ∀ row ∈ subrows,
      row ∈ MemoryTransitionRows.rows layout) :
    Satisfies subrows assignment := by
  intro row member
  exact holds row (included row member)

private theorem pins_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (ConstantPins.rows layout.pins) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem matching_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (EqualityPins.rows (matchingPairs layout)) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem segment_limit_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (LessThanConstantLinkedRows.rows layout.segmentLimit)
      assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem segment_end_addition_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (UnsignedAdditionRows.rows layout.segmentEndAddition)
      assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem timestamp_addition_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (UnsignedAdditionRows.rows layout.timestampAddition)
      assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem step_addition_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (UnsignedAdditionRows.rows layout.stepAddition) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem segment_addition_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (UnsignedAdditionRows.rows layout.segmentAddition) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem start_global_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (UnsignedLessOrEqualRows.rows layout.startLeGlobal)
      assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem global_end_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (UnsignedLessOrEqualRows.rows layout.globalLeEnd)
      assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem output_end_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies (UnsignedLessOrEqualRows.rows layout.outputLeEnd)
      assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem interior_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies
      (ConditionalEqualityOneRows.rows (layout.afterColumn .phase)
        (interiorPairs layout)) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem close_hold
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    Satisfies
      (ConditionalEqualityRows.rows (layout.afterColumn .phase)
        (closePairs layout)) assignment :=
  subrows_hold holds _ (by
    intro row member
    simp [MemoryTransitionRows.rows, member])

private theorem pins_values_canonical (layout : MemoryTransitionRows.Layout) :
    ConstantPins.ValuesCanonical layout.pins := by
  intro pin member
  simp only [MemoryTransitionRows.Layout.pins, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;>
    norm_num [goldilocksP, claimsPerSegment]

private theorem rowsIncluded_self (program : List Row) :
    rowsIncluded program program = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true member

theorem pin_facts
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    assignment (layout.beforeColumn .phase) = 1 ∧
      assignment layout.claimsPerSegmentColumn = claimsPerSegment := by
  have facts := ConstantPins.sound (pins_values_canonical layout)
    (rowsIncluded_self (ConstantPins.rows layout.pins)) canonical one
    (pins_hold holds)
  exact
    ⟨facts (layout.beforeColumn .phase, 1)
        (by simp [MemoryTransitionRows.Layout.pins]),
      facts (layout.claimsPerSegmentColumn, claimsPerSegment)
        (by simp [MemoryTransitionRows.Layout.pins])⟩

theorem matching_facts
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    ∀ pair ∈ matchingPairs layout,
      assignment pair.1 = assignment pair.2 :=
  EqualityPins.rows_sound canonical one (matching_hold holds)

def activeOfWire (before : MemoryCarryCodec.Value)
    (stepBound : before.stepIndex < claimsPerSegment) :
    ActiveCarry Digest.Value (Challenges K) (State K) :=
  { segmentIndex := before.segmentIndex
    stepIndex := ⟨before.stepIndex, stepBound⟩
    globalTimestamp := before.globalTimestamp
    segmentStartTimestamp := before.segmentStartTimestamp
    segmentActiveAccessCount := before.segmentActiveAccessCount
    segmentEndTimestamp := before.segmentEndTimestamp
    challenge := before.challenges
    products := before.products
    dPre := before.dPre
    dSeen := before.dSeen
    memoryRoot := before.memoryRoot }

structure CoreEvidence
    (layout : MemoryTransitionRows.Layout) (assignment : Nat → Nat)
    (before : MemoryCarryCodec.Value) (claim : MemoryClaimCodec.Claim) : Prop where
  stepBound : before.stepIndex < claimsPerSegment
  priorActive : before.phase = .active
  activeWellFormed :
    (activeOfWire before stepBound).WellFormed
  agreement : MatchesActive (activeOfWire before stepBound) claim
  nextStep : assignment layout.nextStepColumn = before.stepIndex + 1
  nextSegment : assignment layout.nextSegmentColumn = before.segmentIndex + 1

private theorem activeCarry_ext
    {left right : ActiveCarry Digest.Value (Challenges K) (State K)}
    (segmentIndex : left.segmentIndex = right.segmentIndex)
    (stepIndex : left.stepIndex = right.stepIndex)
    (globalTimestamp : left.globalTimestamp = right.globalTimestamp)
    (segmentStartTimestamp :
      left.segmentStartTimestamp = right.segmentStartTimestamp)
    (segmentActiveAccessCount :
      left.segmentActiveAccessCount = right.segmentActiveAccessCount)
    (segmentEndTimestamp :
      left.segmentEndTimestamp = right.segmentEndTimestamp)
    (challenge : left.challenge = right.challenge)
    (products : left.products = right.products)
    (dPre : left.dPre = right.dPre)
    (dSeen : left.dSeen = right.dSeen)
    (memoryRoot : left.memoryRoot = right.memoryRoot) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem closedCarry_ext
    {left right : ClosedCarry Digest.Value}
    (segmentIndex : left.segmentIndex = right.segmentIndex)
    (globalTimestamp : left.globalTimestamp = right.globalTimestamp)
    (memoryRoot : left.memoryRoot = right.memoryRoot) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem matching_counter_value
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before : MemoryCarryCodec.Value} {claim : MemoryClaimCodec.Claim}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (equalities : ∀ pair ∈ matchingPairs layout,
      assignment pair.1 = assignment pair.2)
    (counter : MemoryClaimCounterRows.Counter)
    (beforeTag : MemoryCarryCodec.FieldTag)
    (member :
      (layout.claimCounterColumn counter, layout.beforeColumn beforeTag) ∈
        matchingPairs layout) :
    counter.claimValue claim = before.fieldValue beforeTag := by
  calc
    counter.claimValue claim =
        assignment (layout.claimCounterColumn counter) :=
      (claimParsed.counters counter).symm
    _ = assignment (layout.beforeColumn beforeTag) :=
      equalities _ member
    _ = before.fieldValue beforeTag := beforeParsed.placed beforeTag

private theorem matching_field_value
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before : MemoryCarryCodec.Value} {claim : MemoryClaimCodec.Claim}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (equalities : ∀ pair ∈ matchingPairs layout,
      assignment pair.1 = assignment pair.2)
    (slot : MemoryClaimFieldRows.Slot)
    (beforeTag : MemoryCarryCodec.FieldTag)
    (member :
      (layout.claimFieldColumn slot, layout.beforeColumn beforeTag) ∈
        matchingPairs layout) :
    claim.fieldValue slot.tag = before.fieldValue beforeTag := by
  calc
    claim.fieldValue slot.tag = assignment (layout.claimFieldColumn slot) :=
      (claimParsed.fields slot).symm
    _ = assignment (layout.beforeColumn beforeTag) :=
      equalities _ member
    _ = before.fieldValue beforeTag := beforeParsed.placed beforeTag

private theorem after_before_value
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {pairs : List (Nat × Nat)}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (equalities : ∀ pair ∈ pairs,
      assignment pair.1 = assignment pair.2)
    (afterTag beforeTag : MemoryCarryCodec.FieldTag)
    (member :
      (layout.afterColumn afterTag, layout.beforeColumn beforeTag) ∈ pairs) :
    after.fieldValue afterTag = before.fieldValue beforeTag := by
  calc
    after.fieldValue afterTag = assignment (layout.afterColumn afterTag) := by
      simpa [MemoryTransitionRows.Layout.afterColumn] using
        (afterParsed.placed afterTag).symm
    _ = assignment (layout.beforeColumn beforeTag) := equalities _ member
    _ = before.fieldValue beforeTag := by
      simpa [MemoryTransitionRows.Layout.beforeColumn] using
        beforeParsed.placed beforeTag

private theorem after_claim_field_value
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {after : MemoryCarryCodec.Value} {claim : MemoryClaimCodec.Claim}
    {pairs : List (Nat × Nat)}
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (equalities : ∀ pair ∈ pairs,
      assignment pair.1 = assignment pair.2)
    (afterTag : MemoryCarryCodec.FieldTag)
    (slot : MemoryClaimFieldRows.Slot)
    (member :
      (layout.afterColumn afterTag, layout.claimFieldColumn slot) ∈ pairs) :
    after.fieldValue afterTag = claim.fieldValue slot.tag := by
  calc
    after.fieldValue afterTag = assignment (layout.afterColumn afterTag) := by
      simpa [MemoryTransitionRows.Layout.afterColumn] using
        (afterParsed.placed afterTag).symm
    _ = assignment (layout.claimFieldColumn slot) := equalities _ member
    _ = claim.fieldValue slot.tag := claimParsed.fields slot

private theorem after_claim_counter_value
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {after : MemoryCarryCodec.Value} {claim : MemoryClaimCodec.Claim}
    {pairs : List (Nat × Nat)}
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (equalities : ∀ pair ∈ pairs,
      assignment pair.1 = assignment pair.2)
    (afterTag : MemoryCarryCodec.FieldTag)
    (counter : MemoryClaimCounterRows.Counter)
    (member :
      (layout.afterColumn afterTag, layout.claimCounterColumn counter) ∈ pairs) :
    after.fieldValue afterTag = counter.claimValue claim := by
  calc
    after.fieldValue afterTag = assignment (layout.afterColumn afterTag) := by
      simpa [MemoryTransitionRows.Layout.afterColumn] using
        (afterParsed.placed afterTag).symm
    _ = assignment (layout.claimCounterColumn counter) := equalities _ member
    _ = counter.claimValue claim := by
      simpa [MemoryTransitionRows.Layout.claimCounterColumn] using
        claimParsed.counters counter

private theorem claim_claim_field_value
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim} {pairs : List (Nat × Nat)}
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (equalities : ∀ pair ∈ pairs,
      assignment pair.1 = assignment pair.2)
    (left right : MemoryClaimFieldRows.Slot)
    (member :
      (layout.claimFieldColumn left, layout.claimFieldColumn right) ∈ pairs) :
    claim.fieldValue left.tag = claim.fieldValue right.tag := by
  calc
    claim.fieldValue left.tag = assignment (layout.claimFieldColumn left) :=
      (claimParsed.fields left).symm
    _ = assignment (layout.claimFieldColumn right) := equalities _ member
    _ = claim.fieldValue right.tag := claimParsed.fields right

private theorem claim_before_field_value
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before : MemoryCarryCodec.Value} {claim : MemoryClaimCodec.Claim}
    {pairs : List (Nat × Nat)}
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (equalities : ∀ pair ∈ pairs,
      assignment pair.1 = assignment pair.2)
    (slot : MemoryClaimFieldRows.Slot)
    (beforeTag : MemoryCarryCodec.FieldTag)
    (member :
      (layout.claimFieldColumn slot, layout.beforeColumn beforeTag) ∈ pairs) :
    claim.fieldValue slot.tag = before.fieldValue beforeTag := by
  calc
    claim.fieldValue slot.tag = assignment (layout.claimFieldColumn slot) :=
      (claimParsed.fields slot).symm
    _ = assignment (layout.beforeColumn beforeTag) := equalities _ member
    _ = before.fieldValue beforeTag := by
      simpa [MemoryTransitionRows.Layout.beforeColumn] using
        beforeParsed.placed beforeTag

theorem core_evidence
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before : MemoryCarryCodec.Value} {claim : MemoryClaimCodec.Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment) :
    CoreEvidence layout assignment before claim := by
  have pins := pin_facts canonical one holds
  have equalities := matching_facts canonical one holds
  have beforePlaced (tag : MemoryCarryCodec.FieldTag) :
      assignment (layout.beforeColumn tag) = before.fieldValue tag := by
    simpa [MemoryTransitionRows.Layout.beforeColumn] using
      beforeParsed.placed tag
  have claimCounterPlaced (counter : MemoryClaimCounterRows.Counter) :
      assignment (layout.claimCounterColumn counter) =
        counter.claimValue claim := by
    simpa [MemoryTransitionRows.Layout.claimCounterColumn] using
      claimParsed.counters counter
  have priorActive : before.phase = .active := by
    have phaseValueEq : before.fieldValue .phase = 1 :=
      (beforePlaced .phase).symm.trans pins.1
    cases phaseEq : before.phase with
    | closed => simp [MemoryCarryCodec.Value.fieldValue,
        MemoryCarryCodec.phaseValue, phaseEq] at phaseValueEq
    | active => rfl
  have segmentColumnBound :
      assignment (layout.beforeColumn .segmentIndex) <
        2 ^ segmentIndexBits := by
    rw [beforePlaced .segmentIndex]
    exact beforeParsed.parserCanonical.segmentIndex
  have segmentBoundColumn := LessThanConstantLinkedRows.value_lt_limit
    layout.segmentLimit_valid segmentColumnBound canonical one
    (segment_limit_hold holds)
  have segmentBound : before.segmentIndex < maximumSegments := by
    change assignment (layout.beforeColumn .segmentIndex) < maximumSegments at segmentBoundColumn
    rw [beforePlaced .segmentIndex] at segmentBoundColumn
    exact segmentBoundColumn
  have startBound : assignment (layout.beforeColumn .segmentStartTimestamp) <
      2 ^ MemoryWireGeometry.timestampBits := by
    rw [beforePlaced .segmentStartTimestamp]
    exact beforeParsed.parserCanonical.segmentStartTimestamp
  have activeCountBoundColumn :
      assignment (layout.beforeColumn .segmentActiveAccessCount) <
        2 ^ segmentActiveAccessCountBits := by
    rw [beforePlaced .segmentActiveAccessCount]
    exact beforeParsed.parserCanonical.segmentActiveAccessCount
  have segmentEndAddition := UnsignedAdditionRows.output_eq_add
    layout.segmentEndAddition_valid startBound activeCountBoundColumn
    canonical one (segment_end_addition_hold holds)
  have segmentEndExact :
      before.segmentEndTimestamp =
        before.segmentStartTimestamp + before.segmentActiveAccessCount := by
    change assignment (layout.beforeColumn .segmentEndTimestamp) =
      assignment (layout.beforeColumn .segmentStartTimestamp) +
        assignment (layout.beforeColumn .segmentActiveAccessCount) at segmentEndAddition
    simpa [beforePlaced] using segmentEndAddition
  have segmentEndRange : before.segmentEndTimestamp < timestampLimit := by
    simpa [timestampLimit, MemoryWireGeometry.timestampBits] using
      beforeParsed.parserCanonical.segmentEndTimestamp
  have startLeGlobalColumn := UnsignedLessOrEqualRows.left_le_right
    layout.startLeGlobal_valid startBound canonical one
    (start_global_hold holds)
  have startLeGlobal :
      before.segmentStartTimestamp ≤ before.globalTimestamp := by
    change assignment (layout.beforeColumn .segmentStartTimestamp) ≤
      assignment (layout.beforeColumn .globalTimestamp) at startLeGlobalColumn
    simpa [beforePlaced] using startLeGlobalColumn
  have globalBound : assignment (layout.beforeColumn .globalTimestamp) <
      2 ^ MemoryWireGeometry.timestampBits := by
    rw [beforePlaced .globalTimestamp]
    exact beforeParsed.parserCanonical.globalTimestamp
  have globalLeEndColumn := UnsignedLessOrEqualRows.left_le_right
    layout.globalLeEnd_valid globalBound canonical one
    (global_end_hold holds)
  have globalLeEnd : before.globalTimestamp ≤ before.segmentEndTimestamp := by
    change assignment (layout.beforeColumn .globalTimestamp) ≤
      assignment (layout.beforeColumn .segmentEndTimestamp) at globalLeEndColumn
    simpa [beforePlaced] using globalLeEndColumn
  have activeWellFormed :
      (activeOfWire before beforeParsed.parserCanonical.stepIndex).WellFormed :=
    ⟨segmentBound,
      by
        simpa [operationCountLimit, segmentActiveAccessCountBits] using
          beforeParsed.parserCanonical.segmentActiveAccessCount,
      segmentEndExact, segmentEndRange, startLeGlobal, globalLeEnd⟩
  have segmentIndexEq : claim.segmentIndex = before.segmentIndex :=
    matching_counter_value beforeParsed claimParsed equalities .segmentIndex
      .segmentIndex (by simp [matchingPairs])
  have stepIndexValEq : claim.stepIndex.val = before.stepIndex :=
    matching_counter_value beforeParsed claimParsed equalities .stepIndex
      .stepIndex (by simp [matchingPairs])
  have stepIndexEq :
      claim.stepIndex =
        (activeOfWire before beforeParsed.parserCanonical.stepIndex).stepIndex :=
    Fin.ext stepIndexValEq
  have timestampInEq : claim.timestampIn = before.globalTimestamp :=
    matching_counter_value beforeParsed claimParsed equalities .timestampIn
      .globalTimestamp (by simp [matchingPairs])
  have segmentStartEq :
      claim.segmentStartTimestamp = before.segmentStartTimestamp :=
    matching_counter_value beforeParsed claimParsed equalities
      .segmentStartTimestamp .segmentStartTimestamp (by simp [matchingPairs])
  have segmentEndEq :
      claim.segmentEndTimestamp = before.segmentEndTimestamp :=
    matching_counter_value beforeParsed claimParsed equalities
      .segmentEndTimestamp .segmentEndTimestamp (by simp [matchingPairs])
  have challengeEq : claim.challenge = before.challenges := by
    funext repetition
    apply challenge_eq_of_values
    intro coordinate limb
    exact matching_field_value beforeParsed claimParsed equalities
      (.challenge repetition coordinate limb)
      (.challenge repetition coordinate limb)
      (by
        fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
          simp [matchingPairs, challengePairs])
  have dPreEq : claim.dPre = before.dPre := by
    apply roots_eq_of_values
    intro role lane
    exact matching_field_value beforeParsed claimParsed equalities
      (.root .precommit role lane) (.root (.precommit role) lane)
      (by
        cases role <;> fin_cases lane <;>
          simp [matchingPairs, rootMatchPairs, rootRoles])
  have dSeenEq : claim.dSeenBefore = before.dSeen := by
    apply roots_eq_of_values
    intro role lane
    exact matching_field_value beforeParsed claimParsed equalities
      (.root .seenBefore role lane) (.root (.seen role) lane)
      (by
        cases role <;> fin_cases lane <;>
          simp [matchingPairs, rootMatchPairs, rootRoles])
  have productsEq : claim.productsBefore = before.products := by
    funext repetition
    apply product_eq_of_values
    intro role limb
    exact matching_field_value beforeParsed claimParsed equalities
      (.product 0 repetition role limb) (.product repetition role limb)
      (by
        fin_cases repetition <;> cases role <;> fin_cases limb <;>
          simp [matchingPairs, productBeforePairs, productRoles])
  have timestampInBound :
      assignment (layout.claimCounterColumn .timestampIn) <
        2 ^ MemoryWireGeometry.timestampBits := by
    rw [claimCounterPlaced .timestampIn]
    exact claimParsed.canonical.timestampIn
  have claimActiveCountBound :
      assignment (layout.claimCounterColumn .activeAccessCount) <
        2 ^ stepActiveAccessCountBits := by
    rw [claimCounterPlaced .activeAccessCount]
    exact claimParsed.canonical.activeAccessCount
  have timestampAddition := UnsignedAdditionRows.output_eq_add
    layout.timestampAddition_valid timestampInBound claimActiveCountBound
    canonical one (timestamp_addition_hold holds)
  have timestampAdvance :
      claim.timestampOut = claim.timestampIn + claim.activeAccessCount := by
    change assignment (layout.claimCounterColumn .timestampOut) =
      assignment (layout.claimCounterColumn .timestampIn) +
        assignment (layout.claimCounterColumn .activeAccessCount) at timestampAddition
    simpa [claimCounterPlaced] using timestampAddition
  have timestampOutBound :
      assignment (layout.claimCounterColumn .timestampOut) <
        2 ^ MemoryWireGeometry.timestampBits := by
    rw [claimCounterPlaced .timestampOut]
    exact claimParsed.canonical.timestampOut
  have timestampWithinColumn := UnsignedLessOrEqualRows.left_le_right
    layout.outputLeEnd_valid timestampOutBound canonical one
    (output_end_hold holds)
  have timestampWithin :
      claim.timestampOut ≤ before.segmentEndTimestamp := by
    change assignment (layout.claimCounterColumn .timestampOut) ≤
      assignment (layout.beforeColumn .segmentEndTimestamp) at timestampWithinColumn
    simpa [claimCounterPlaced, beforePlaced] using timestampWithinColumn
  have stepColumnBound : assignment (layout.beforeColumn .stepIndex) <
      2 ^ stepIndexBits := by
    rw [beforePlaced .stepIndex]
    exact beforeParsed.parserCanonical.stepIndex.trans (by decide)
  have oneBitBound : assignment 0 < 2 ^ 1 := by
    rw [one]
    decide
  have stepAddition := UnsignedAdditionRows.output_eq_add
    layout.stepAddition_valid stepColumnBound
    (by simpa [MemoryTransitionRows.Layout.stepAddition] using oneBitBound)
    canonical one
    (step_addition_hold holds)
  have nextStep : assignment layout.nextStepColumn = before.stepIndex + 1 := by
    change assignment layout.nextStepColumn =
      assignment (layout.beforeColumn .stepIndex) + assignment 0 at stepAddition
    simpa [beforePlaced, one] using stepAddition
  have segmentAddition := UnsignedAdditionRows.output_eq_add
    layout.segmentAddition_valid segmentColumnBound
    (by simpa [MemoryTransitionRows.Layout.segmentAddition] using oneBitBound)
    canonical one (segment_addition_hold holds)
  have nextSegment :
      assignment layout.nextSegmentColumn = before.segmentIndex + 1 := by
    change assignment layout.nextSegmentColumn =
      assignment (layout.beforeColumn .segmentIndex) + assignment 0 at segmentAddition
    simpa [beforePlaced, one] using segmentAddition
  exact
    { stepBound := beforeParsed.parserCanonical.stepIndex
      priorActive := priorActive
      activeWellFormed := activeWellFormed
      agreement :=
        { activeWellFormed := activeWellFormed
          segmentIndex := segmentIndexEq
          stepIndex := stepIndexEq
          timestampIn := timestampInEq
          segmentStartTimestamp := segmentStartEq
          segmentEndTimestamp := segmentEndEq
          challenge := challengeEq
          dPre := dPreEq
          dSeen := dSeenEq
          products := productsEq
          timestampAdvance := timestampAdvance
          activeCountBound := by
            have bound := claimParsed.canonical.activeAccessCount
            simp only [stepActiveAccessCountBits] at bound
            omega
          timestampWithinDeclaredEnd := timestampWithin
          timestampInRange := by
            simpa [timestampLimit, MemoryWireGeometry.timestampBits] using
              claimParsed.canonical.timestampIn
          timestampOutRange := by
            simpa [timestampLimit, MemoryWireGeometry.timestampBits] using
              claimParsed.canonical.timestampOut }
      nextStep := nextStep
      nextSegment := nextSegment }

/-- Satisfying parser, transition, and terminal-balance rows select exactly
one semantic F-prime transition. The balance premise is used only in the
closed branch; the recursive manifest derives it from the phase-gated
two-repetition product rows. -/
theorem consumes_of_rows
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {claim : MemoryClaimCodec.Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment)
    (balancedOnClose : after.phase = .closed →
      MemoryProductBalanceRows.ConcreteBalanced claim.productsAfter) :
    Consumes MemoryProductBalanceRows.ConcreteBalanced
      (MemoryCarryParser.semanticCarry before
        beforeParsed.parserCanonical.stepIndex)
      claim
      (MemoryCarryParser.semanticCarry after
        afterParsed.parserCanonical.stepIndex) := by
  have core := core_evidence canonical one beforeParsed claimParsed holds
  have pins := pin_facts canonical one holds
  have beforeSemantic :
      MemoryCarryParser.semanticCarry before
          beforeParsed.parserCanonical.stepIndex =
        .active (activeOfWire before core.stepBound) := by
    simp [MemoryCarryParser.semanticCarry, core.priorActive, activeOfWire]
  have afterPlaced (tag : MemoryCarryCodec.FieldTag) :
      assignment (layout.afterColumn tag) = after.fieldValue tag := by
    simpa [MemoryTransitionRows.Layout.afterColumn] using
      afterParsed.placed tag
  cases phaseEq : after.phase with
  | active =>
      have phaseOne : assignment (layout.afterColumn .phase) = 1 := by
        rw [afterPlaced .phase]
        simp [MemoryCarryCodec.Value.fieldValue, MemoryCarryCodec.phaseValue,
          phaseEq]
      have equalities := ConditionalEqualityOneRows.rows_sound_one canonical
        one phaseOne (interior_hold holds)
      have segmentEq : after.segmentIndex = before.segmentIndex :=
        after_before_value beforeParsed afterParsed equalities .segmentIndex
          .segmentIndex (by simp [interiorPairs])
      have stepColumnEq :
          assignment (layout.afterColumn .stepIndex) =
            assignment layout.nextStepColumn :=
        equalities
          (layout.afterColumn .stepIndex, layout.nextStepColumn)
          (by simp [interiorPairs])
      have stepEq : after.stepIndex = before.stepIndex + 1 := by
        calc
          after.stepIndex = assignment (layout.afterColumn .stepIndex) :=
            (afterPlaced .stepIndex).symm
          _ = assignment layout.nextStepColumn := stepColumnEq
          _ = before.stepIndex + 1 := core.nextStep
      have notLast : before.stepIndex + 1 < claimsPerSegment := by
        have bound := afterParsed.parserCanonical.stepIndex
        omega
      have globalEq : after.globalTimestamp = claim.timestampOut :=
        after_claim_counter_value afterParsed claimParsed equalities
          .globalTimestamp .timestampOut (by simp [interiorPairs])
      have startEq :
          after.segmentStartTimestamp = before.segmentStartTimestamp :=
        after_before_value beforeParsed afterParsed equalities
          .segmentStartTimestamp .segmentStartTimestamp
          (by simp [interiorPairs])
      have countEq : after.segmentActiveAccessCount =
          before.segmentActiveAccessCount :=
        after_before_value beforeParsed afterParsed equalities
          .segmentActiveAccessCount .segmentActiveAccessCount
          (by simp [interiorPairs])
      have endEq : after.segmentEndTimestamp = before.segmentEndTimestamp :=
        after_before_value beforeParsed afterParsed equalities
          .segmentEndTimestamp .segmentEndTimestamp
          (by simp [interiorPairs])
      have challengeEq : after.challenges = before.challenges := by
        funext repetition
        apply challenge_eq_of_values
        intro coordinate limb
        exact after_before_value beforeParsed afterParsed equalities
          (.challenge repetition coordinate limb)
          (.challenge repetition coordinate limb)
          (by
            fin_cases repetition <;> fin_cases coordinate <;>
              fin_cases limb <;>
                simp [interiorPairs, afterBeforeChallengePairs])
      have productsEq : after.products = claim.productsAfter := by
        funext repetition
        apply product_eq_of_values
        intro role limb
        exact after_claim_field_value afterParsed claimParsed equalities
          (.product repetition role limb)
          (.product 1 repetition role limb)
          (by
            fin_cases repetition <;> cases role <;> fin_cases limb <;>
              simp [interiorPairs, productAfterPairs, productRoles])
      have dPreEq : after.dPre = before.dPre := by
        apply roots_eq_of_values
        intro role lane
        exact after_before_value beforeParsed afterParsed equalities
          (.root (.precommit role) lane) (.root (.precommit role) lane)
          (by
            cases role <;> fin_cases lane <;>
              simp [interiorPairs, afterBeforeRootPairs, rootRoles])
      have dSeenEq : after.dSeen = claim.dSeenAfter := by
        apply roots_eq_of_values
        intro role lane
        exact after_claim_field_value afterParsed claimParsed equalities
          (.root (.seen role) lane) (.root .seenAfter role lane)
          (by
            cases role <;> fin_cases lane <;>
              simp [interiorPairs, afterClaimRootPairs, rootRoles])
      have memoryEq : after.memoryRoot = before.memoryRoot := by
        apply digest_eq_of_lane_values
        intro lane
        exact after_before_value beforeParsed afterParsed equalities
          (.root .memory lane) (.root .memory lane)
          (by
            have localMember :
                (layout.afterColumn (.root .memory lane),
                  layout.beforeColumn (.root .memory lane)) ∈
                    memoryRootPair layout :=
              List.mem_ofFn.mpr ⟨lane, rfl⟩
            simp [interiorPairs, localMember])
      let afterActive :
          ActiveCarry Digest.Value (Challenges K) (State K) :=
        { segmentIndex := after.segmentIndex
          stepIndex := ⟨after.stepIndex,
            afterParsed.parserCanonical.stepIndex⟩
          globalTimestamp := after.globalTimestamp
          segmentStartTimestamp := after.segmentStartTimestamp
          segmentActiveAccessCount := after.segmentActiveAccessCount
          segmentEndTimestamp := after.segmentEndTimestamp
          challenge := after.challenges
          products := after.products
          dPre := after.dPre
          dSeen := after.dSeen
          memoryRoot := after.memoryRoot }
      have afterSemantic :
          MemoryCarryParser.semanticCarry after
              afterParsed.parserCanonical.stepIndex = .active afterActive := by
        simp [MemoryCarryParser.semanticCarry, phaseEq, afterActive]
      have activeExact : afterActive =
          interiorCarry (activeOfWire before core.stepBound) claim notLast := by
        apply activeCarry_ext
        · exact segmentEq
        · apply Fin.ext
          exact stepEq
        · exact globalEq
        · exact startEq
        · exact countEq
        · exact endEq
        · exact challengeEq
        · exact productsEq
        · exact dPreEq
        · exact dSeenEq
        · exact memoryEq
      rw [beforeSemantic, afterSemantic, activeExact]
      exact .interior core.agreement notLast
  | closed =>
      have phaseClosed : assignment (layout.afterColumn .phase) = 0 := by
        rw [afterPlaced .phase]
        simp [MemoryCarryCodec.Value.fieldValue, MemoryCarryCodec.phaseValue,
          phaseEq]
      have equalities := ConditionalEqualityRows.rows_sound_closed canonical
        one phaseClosed (close_hold holds)
      have nextStepLast :
          assignment layout.nextStepColumn = claimsPerSegment := by
        calc
          assignment layout.nextStepColumn =
              assignment layout.claimsPerSegmentColumn :=
            equalities
              (layout.nextStepColumn, layout.claimsPerSegmentColumn)
              (by simp [closePairs])
          _ = claimsPerSegment := pins.2
      have last : before.stepIndex + 1 = claimsPerSegment := by
        rw [← core.nextStep]
        exact nextStepLast
      have segmentColumnEq :
          assignment (layout.afterColumn .segmentIndex) =
            assignment layout.nextSegmentColumn :=
        equalities
          (layout.afterColumn .segmentIndex, layout.nextSegmentColumn)
          (by simp [closePairs])
      have segmentEq : after.segmentIndex = before.segmentIndex + 1 := by
        calc
          after.segmentIndex = assignment (layout.afterColumn .segmentIndex) :=
            (afterPlaced .segmentIndex).symm
          _ = assignment layout.nextSegmentColumn := segmentColumnEq
          _ = before.segmentIndex + 1 := core.nextSegment
      have globalEq : after.globalTimestamp = claim.timestampOut :=
        after_claim_counter_value afterParsed claimParsed equalities
          .globalTimestamp .timestampOut (by simp [closePairs])
      have timestampEndColumn :
          assignment (layout.claimCounterColumn .timestampOut) =
            assignment (layout.beforeColumn .segmentEndTimestamp) :=
        equalities
          (layout.claimCounterColumn .timestampOut,
            layout.beforeColumn .segmentEndTimestamp)
          (by simp [closePairs])
      have timestampEndEq :
          claim.timestampOut = before.segmentEndTimestamp := by
        calc
          claim.timestampOut =
              assignment (layout.claimCounterColumn .timestampOut) := by
            simpa [MemoryTransitionRows.Layout.claimCounterColumn] using
              (claimParsed.counters .timestampOut).symm
          _ = assignment (layout.beforeColumn .segmentEndTimestamp) :=
            timestampEndColumn
          _ = before.segmentEndTimestamp := by
            simpa [MemoryTransitionRows.Layout.beforeColumn] using
              beforeParsed.placed .segmentEndTimestamp
      have seenEq : claim.dSeenAfter = claim.dPre := by
        apply roots_eq_of_values
        intro role lane
        exact claim_claim_field_value claimParsed equalities
          (.root .seenAfter role lane) (.root .precommit role lane)
          (by
            cases role <;> fin_cases lane <;>
              simp [closePairs, seenEqualsPrecommitPairs, rootRoles])
      have initialEq :
          claim.dSeenAfter.initialSnapshot = before.memoryRoot := by
        apply digest_eq_of_lane_values
        intro lane
        exact claim_before_field_value beforeParsed claimParsed equalities
          (.root .seenAfter .initialSnapshot lane) (.root .memory lane)
          (by
            have localMember :
                (layout.claimFieldColumn
                    (.root .seenAfter .initialSnapshot lane),
                  layout.beforeColumn (.root .memory lane)) ∈
                    initialEqualsMemoryPairs layout :=
              List.mem_ofFn.mpr ⟨lane, rfl⟩
            simp [closePairs, localMember])
      have memoryEq :
          after.memoryRoot = claim.dSeenAfter.finalSnapshot := by
        apply digest_eq_of_lane_values
        intro lane
        exact after_claim_field_value afterParsed claimParsed equalities
          (.root .memory lane) (.root .seenAfter .finalSnapshot lane)
          (by
            have localMember :
                (layout.afterColumn (.root .memory lane),
                  layout.claimFieldColumn
                    (.root .seenAfter .finalSnapshot lane)) ∈
                    closeMemoryRootPairs layout :=
              List.mem_ofFn.mpr ⟨lane, rfl⟩
            simp [closePairs, localMember])
      let afterClosed : ClosedCarry Digest.Value :=
        { segmentIndex := after.segmentIndex
          globalTimestamp := after.globalTimestamp
          memoryRoot := after.memoryRoot }
      have afterSemantic :
          MemoryCarryParser.semanticCarry after
              afterParsed.parserCanonical.stepIndex = .closed afterClosed := by
        simp [MemoryCarryParser.semanticCarry, phaseEq, afterClosed]
      have closedExact : afterClosed =
          closedCarryAfter (activeOfWire before core.stepBound) claim := by
        apply closedCarry_ext
        · exact segmentEq
        · exact globalEq
        · exact memoryEq
      have checks : CloseChecks MemoryProductBalanceRows.ConcreteBalanced
          (activeOfWire before core.stepBound) claim :=
        { seenEqualsPrecommit := seenEq.trans core.agreement.dPre
          initialEqualsMemory := initialEq
          productsBalanced := balancedOnClose phaseEq
          timestampEqualsDeclaredEnd := timestampEndEq }
      rw [beforeSemantic, afterSemantic, closedExact]
      exact .close core.agreement last checks

end Nightstream.Implementation.Nebula.MemoryTransitionSound
