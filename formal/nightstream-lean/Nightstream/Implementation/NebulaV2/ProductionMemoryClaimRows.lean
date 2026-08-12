import Nightstream.Implementation.NebulaV2.LessThanConstantLinkedRows
import Nightstream.Implementation.NebulaV2.MemoryClaimRows

/-!
Contract: exact mixed field-native decoder rows for one production memory
suffix.

The seven bounded counters use their 116 Boolean carrier coordinates. The 76
challenge, product, and root limbs use canonical native Goldilocks columns.
Satisfying rows construct one exact typed memory claim and the shared
`MemoryClaimRows.ParsedColumnsMatch` interface used by the memory gadgets.

No typed claim, counter bound, step bound, field equality, or decoder result
is a premise of `sound`.

Does not own the enclosing batch, absolute generated columns, record-source
rows, state hashing, NIFS verification, or Rust refinement.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryClaimRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

abbrev Slot := MemoryClaimFieldRows.Slot

/-- Reinterpret one canonical native field as one digest lane. -/
def digestLane (value : F) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  ⟨value.val, by
    simpa [ShiftedTernary41V1.modulus, goldilocksModulus] using value.isLt⟩

def decodedK (fields : Slot -> F) (low high : Slot) : K :=
  ⟨fields low, fields high⟩

def decodedChallenges (fields : Slot -> F) : Challenges K :=
  fun repetition =>
    { gamma1 := decodedK fields
        (.challenge repetition 0 0) (.challenge repetition 0 1)
      gamma2 := decodedK fields
        (.challenge repetition 1 0) (.challenge repetition 1 1) }

def decodedProduct (fields : Slot -> F) (side repetition : Fin 2) : Four K :=
  { initialSnapshot := decodedK fields
      (.product side repetition .initialSnapshot 0)
      (.product side repetition .initialSnapshot 1)
    writes := decodedK fields
      (.product side repetition .writes 0)
      (.product side repetition .writes 1)
    reads := decodedK fields
      (.product side repetition .reads 0)
      (.product side repetition .reads 1)
    finalSnapshot := decodedK fields
      (.product side repetition .finalSnapshot 0)
      (.product side repetition .finalSnapshot 1) }

def decodedProducts (fields : Slot -> F) (side : Fin 2) : State K :=
  fun repetition => decodedProduct fields side repetition

def decodedDigest (fields : Slot -> F) (stage : RootStage)
    (role : RootRole) : Digest.Value where
  lanes := fun lane => digestLane (fields (.root stage role lane))

def decodedRoots (fields : Slot -> F) (stage : RootStage) :
    Roots Digest.Value :=
  { operations := decodedDigest fields stage .operations
    initialSnapshot := decodedDigest fields stage .initialSnapshot
    finalSnapshot := decodedDigest fields stage .finalSnapshot }

/-- Total typed decode after the row-derived strict step bound. -/
def decodedClaim
    (counters : Counter -> Nat) (fields : Slot -> F)
    (stepBound : counters .stepIndex < Lifecycle.claimsPerSegment) : Claim :=
  { segmentIndex := counters .segmentIndex
    stepIndex := ⟨counters .stepIndex, stepBound⟩
    timestampIn := counters .timestampIn
    timestampOut := counters .timestampOut
    segmentStartTimestamp := counters .segmentStartTimestamp
    segmentEndTimestamp := counters .segmentEndTimestamp
    activeAccessCount := counters .activeAccessCount
    challenge := decodedChallenges fields
    dPre := decodedRoots fields .precommit
    dSeenBefore := decodedRoots fields .seenBefore
    dSeenAfter := decodedRoots fields .seenAfter
    productsBefore := decodedProducts fields 0
    productsAfter := decodedProducts fields 1 }

theorem decodedClaim_counterValue
    (counters : Counter -> Nat) (fields : Slot -> F)
    (stepBound : counters .stepIndex < Lifecycle.claimsPerSegment)
    (counter : Counter) :
    counter.claimValue (decodedClaim counters fields stepBound) =
      counters counter := by
  cases counter <;> rfl

/-- Every native slot is the exact matching field of the decoded claim. -/
theorem decodedClaim_fieldValue
    (counters : Counter -> Nat) (fields : Slot -> F)
    (stepBound : counters .stepIndex < Lifecycle.claimsPerSegment)
    (slot : Slot) :
    (decodedClaim counters fields stepBound).fieldValue slot.tag =
      (fields slot).val := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;> rfl
  | product side repetition role limb =>
      fin_cases side <;> cases role <;> fin_cases limb <;> rfl
  | root stage role lane =>
      cases stage <;> cases role <;> rfl

/-- Exact physical columns for one mixed suffix. -/
structure Layout where
  counterBitStart : Nat
  counterValueColumn : Counter -> Nat
  stepSlackColumn : Nat
  stepSlackBitStart : Nat
  nativeFieldColumn : Slot -> Nat

def nativeColumnMap (column : Nat) : List Nat := [0, column]

@[simp] theorem nativeColumnMap_zero (column : Nat) :
    Relabel.column (nativeColumnMap column) 0 = 0 := by
  rfl

@[simp] theorem nativeColumnMap_value (column : Nat) :
    Relabel.column (nativeColumnMap column) CanonicalU64.varCol = column := by
  rfl

/-- Compatibility view used by all existing memory semantic gadgets. The
unused 64-bit field-parser columns are not included in this module's rows. -/
def Layout.reference (layout : Layout) : MemoryClaimRows.Layout where
  publicBitStart := layout.counterBitStart
  counterValueColumn := layout.counterValueColumn
  stepSlackColumn := layout.stepSlackColumn
  stepSlackBitStart := layout.stepSlackBitStart
  fieldColumnMap := fun slot => nativeColumnMap (layout.nativeFieldColumn slot)
  fieldMapsConstantOne := by intro slot; simp

def Layout.counters (layout : Layout) : MemoryClaimCounterRows.Layout :=
  { publicBitStart := layout.counterBitStart
    valueColumn := layout.counterValueColumn }

def Layout.stepLimit (layout : Layout) : LessThanConstantLinkedRows.Layout :=
  { width := MemoryWireGeometry.stepIndexBits
    limit := Lifecycle.claimsPerSegment
    valueColumn := layout.counterValueColumn .stepIndex
    slackColumn := layout.stepSlackColumn
    slackBitStart := layout.stepSlackBitStart }

theorem Layout.stepLimit_valid (layout : Layout) : layout.stepLimit.Valid where
  limitPositive := by simp [Layout.stepLimit]; decide
  limitFits := by simp [Layout.stepLimit]; decide
  sumFits := by simp [Layout.stepLimit]; decide

/-- Only counter range rows and the strict step-index block are required. -/
def rows (layout : Layout) : List Row :=
  MemoryClaimCounterRows.rows layout.counters ++
    LessThanConstantLinkedRows.rows layout.stepLimit

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 136 := by
  rw [rows, List.length_append, MemoryClaimCounterRows.rows_length_exact,
    LessThanConstantLinkedRows.rows_length]
  simp [Layout.stepLimit, MemoryWireGeometry.stepIndexBits]

private theorem counterRows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (MemoryClaimCounterRows.rows layout.counters) assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem stepRows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (LessThanConstantLinkedRows.rows layout.stepLimit) assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

private theorem oneCounterRows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) (counter : Counter) :
    Satisfies (BoundedWordRows.rows (layout.counters.word counter))
      assignment := by
  have allCounters := counterRows_hold satisfied
  rw [MemoryClaimCounterRows.rows] at allCounters
  exact (satisfies_flatten_iff _ _).mp allCounters _
    (List.mem_map.mpr ⟨counter, counter.mem_all, rfl⟩)

def decodedCounters (layout : Layout) (assignment : Nat -> Nat) :
    Counter -> Nat :=
  fun counter => assignment (layout.counterValueColumn counter)

def decodedFields (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Slot -> F :=
  fun slot => ⟨assignment (layout.nativeFieldColumn slot), canonical _⟩

private theorem counterBound
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment)
    (counter : Counter) :
    decodedCounters layout assignment counter < 2 ^ counter.width := by
  exact BoundedWordRows.value_lt_twoPower counter.fitsGoldilocks canonical one
    (oneCounterRows_hold satisfied counter)

private theorem stepBound
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    decodedCounters layout assignment .stepIndex <
      Lifecycle.claimsPerSegment := by
  exact LessThanConstantLinkedRows.value_lt_limit layout.stepLimit_valid
    (counterBound canonical one satisfied .stepIndex) canonical one
    (stepRows_hold satisfied)

/-- Row-derived typed decode and compatibility bridge. -/
structure Sound (layout : Layout) (assignment : Nat -> Nat) where
  claim : Claim
  parsed : MemoryClaimRows.ParsedColumnsMatch layout.reference assignment claim

/-- Construct the unique decoder output selected by satisfying mixed rows. -/
def derive
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    Sound layout assignment := by
  let counters := decodedCounters layout assignment
  let fields := decodedFields layout assignment canonical
  have step := stepBound canonical one satisfied
  let claim := decodedClaim counters fields step
  refine { claim := claim, parsed := ?_ }
  refine
    { counters := ?_
      fields := ?_
      stepStrict := step
      canonical := ?_ }
  · intro counter
    change assignment (layout.counterValueColumn counter) =
      counter.claimValue claim
    exact (decodedClaim_counterValue counters fields step counter).symm
  · intro slot
    change assignment (layout.nativeFieldColumn slot) = claim.fieldValue slot.tag
    exact (decodedClaim_fieldValue counters fields step slot).symm
  · constructor
    · exact counterBound canonical one satisfied .segmentIndex
    · exact counterBound canonical one satisfied .timestampIn
    · exact counterBound canonical one satisfied .timestampOut
    · exact counterBound canonical one satisfied .segmentStartTimestamp
    · exact counterBound canonical one satisfied .segmentEndTimestamp
    · exact counterBound canonical one satisfied .activeAccessCount

/-- Every bounded counter word in the mixed carrier is the exact independent
codec word of the row-derived claim. -/
theorem counter_digits_eq_decoded
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment)
    (counter : Counter) :
    (layout.counters.word counter).digits assignment =
      WasmStateCodec.encodeWord counter.width
        (counter.claimValue (derive canonical one satisfied).claim) := by
  exact MemoryClaimCounterRows.counter_digits_eq_codec canonical one
    (derive canonical one satisfied).parsed.counters
    (counterRows_hold satisfied) counter

/-- Satisfying mixed rows determine an exact typed suffix. The theorem has no
decoder-result or claim-placement premise. -/
theorem rows_imply_exact_claim
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    exists claim,
      MemoryClaimRows.ParsedColumnsMatch layout.reference assignment claim := by
  let result := derive canonical one satisfied
  exact ⟨result.claim, result.parsed⟩

/-- Two typed claims cannot match the same mixed carrier columns. -/
theorem parsed_unique
    {layout : Layout} {assignment : Nat -> Nat} {left right : Claim}
    (leftParsed : MemoryClaimRows.ParsedColumnsMatch
      layout.reference assignment left)
    (rightParsed : MemoryClaimRows.ParsedColumnsMatch
      layout.reference assignment right) :
    left = right := by
  apply Claim.fieldValue_injective
  funext tag
  cases tag with
  | segmentIndex =>
      simpa using (leftParsed.counters .segmentIndex).symm.trans
        (rightParsed.counters .segmentIndex)
  | stepIndex =>
      simpa using (leftParsed.counters .stepIndex).symm.trans
        (rightParsed.counters .stepIndex)
  | timestampIn =>
      simpa using (leftParsed.counters .timestampIn).symm.trans
        (rightParsed.counters .timestampIn)
  | timestampOut =>
      simpa using (leftParsed.counters .timestampOut).symm.trans
        (rightParsed.counters .timestampOut)
  | segmentStartTimestamp =>
      simpa using (leftParsed.counters .segmentStartTimestamp).symm.trans
        (rightParsed.counters .segmentStartTimestamp)
  | segmentEndTimestamp =>
      simpa using (leftParsed.counters .segmentEndTimestamp).symm.trans
        (rightParsed.counters .segmentEndTimestamp)
  | activeAccessCount =>
      simpa using (leftParsed.counters .activeAccessCount).symm.trans
        (rightParsed.counters .activeAccessCount)
  | challenge repetition coordinate limb =>
      exact (leftParsed.fields (.challenge repetition coordinate limb)).symm.trans
        (rightParsed.fields (.challenge repetition coordinate limb))
  | product side repetition role limb =>
      exact (leftParsed.fields (.product side repetition role limb)).symm.trans
        (rightParsed.fields (.product side repetition role limb))
  | root stage role lane =>
      exact (leftParsed.fields (.root stage role lane)).symm.trans
        (rightParsed.fields (.root stage role lane))

/-- The computed decoder result is the only typed result compatible with the
satisfying carrier. -/
theorem derive_unique
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment)
    {claim : Claim}
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.reference assignment claim) :
    (derive canonical one satisfied).claim = claim :=
  parsed_unique (derive canonical one satisfied).parsed parsed

end Nightstream.Implementation.NebulaV2.ProductionMemoryClaimRows
