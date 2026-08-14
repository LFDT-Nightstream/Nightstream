import Nightstream.Implementation.Nebula.Core.ConditionalEqualityRows
import Nightstream.Implementation.Nebula.Core.LessThanConstantRows
import Nightstream.Implementation.Nebula.Memory.Carry.Codec
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: exact counter and closed-inactive-field rows for the V2 memory carry.

Assurance tier: implementation model.

Owns all seven carry counter words, the strict `step_index < 1088` block, a
pinned zero column, 52 closed-phase conditional equalities, and derivation of
the complete `MemoryCarryCodec.Value.Canonical` predicate from satisfying
rows.

Does not own canonical-u64 blocks for the 64-bit field limbs, the state-hash
permutation, absolute generated columns, or the F-prime transition rows.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryCarryRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.Nebula.MemoryCarryCodec
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.MemoryWireGeometry
open Nightstream.Protocol.Nebula.ProductState

abbrev CarryTag := MemoryCarryCodec.FieldTag

inductive RegularCounter where
  | phase
  | segmentIndex
  | globalTimestamp
  | segmentStartTimestamp
  | segmentActiveAccessCount
  | segmentEndTimestamp
deriving DecidableEq, Repr

def RegularCounter.all : List RegularCounter :=
  [.phase, .segmentIndex, .globalTimestamp, .segmentStartTimestamp,
    .segmentActiveAccessCount, .segmentEndTimestamp]

def RegularCounter.tag : RegularCounter → CarryTag
  | .phase => .phase
  | .segmentIndex => .segmentIndex
  | .globalTimestamp => .globalTimestamp
  | .segmentStartTimestamp => .segmentStartTimestamp
  | .segmentActiveAccessCount => .segmentActiveAccessCount
  | .segmentEndTimestamp => .segmentEndTimestamp

def RegularCounter.width : RegularCounter → Nat
  | .phase => phaseBits
  | .segmentIndex => segmentIndexBits
  | .globalTimestamp => MemoryWireGeometry.timestampBits
  | .segmentStartTimestamp => MemoryWireGeometry.timestampBits
  | .segmentActiveAccessCount => segmentActiveAccessCountBits
  | .segmentEndTimestamp => MemoryWireGeometry.timestampBits

def RegularCounter.bitOffset : RegularCounter → Nat
  | .phase => 0
  | .segmentIndex => 1
  | .globalTimestamp => 19
  | .segmentStartTimestamp => 42
  | .segmentActiveAccessCount => 65
  | .segmentEndTimestamp => 82

theorem RegularCounter.mem_all (counter : RegularCounter) :
    counter ∈ RegularCounter.all := by
  cases counter <;> simp [RegularCounter.all]

theorem RegularCounter.fitsGoldilocks (counter : RegularCounter) :
    2 ^ counter.width ≤ goldilocksP := by
  cases counter <;> decide

def closedRootSources : List RootSource :=
  [.precommit .operations, .precommit .initialSnapshot,
    .precommit .finalSnapshot, .seen .operations,
    .seen .initialSnapshot, .seen .finalSnapshot]

def closedRootTags : List CarryTag :=
  (closedRootSources.map fun source =>
      List.ofFn fun lane : Fin 4 =>
        MemoryCarryCodec.FieldTag.root source lane).flatten

/-- Exactly the fields that are inactive while `phase = closed`. -/
def inactiveTags : List CarryTag :=
  [.stepIndex, .segmentStartTimestamp, .segmentActiveAccessCount,
    .segmentEndTimestamp] ++ MemoryCarryCodec.challengeSchema ++
    MemoryCarryCodec.productSchema ++
    closedRootTags

theorem inactiveTags_length : inactiveTags.length = 52 := by
  decide

structure Layout where
  publicBitStart : Nat
  fieldColumn : CarryTag → Nat
  stepSlackColumn : Nat
  stepSlackBitStart : Nat
  zeroColumn : Nat
  headerColumn : MemoryClaimCodec.RootRole → Fin 4 → Nat

def Layout.regularWord (layout : Layout) (counter : RegularCounter) :
    BoundedWordRows.Layout :=
  { width := counter.width
    valueColumn := layout.fieldColumn counter.tag
    bitStart := layout.publicBitStart + counter.bitOffset }

def Layout.stepWord (layout : Layout) : LessThanConstantRows.Layout :=
  { width := stepIndexBits
    limit := Lifecycle.claimsPerSegment
    valueColumn := layout.fieldColumn .stepIndex
    valueBitStart := layout.publicBitStart + 8
    slackColumn := layout.stepSlackColumn
    slackBitStart := layout.stepSlackBitStart }

theorem Layout.stepWord_valid (layout : Layout) : layout.stepWord.Valid where
  limitPositive := by simp [Layout.stepWord]; decide
  limitFits := by simp [Layout.stepWord]; decide
  sumFits := by simp [Layout.stepWord]; decide

def Layout.expectedColumn (layout : Layout) : CarryTag → Nat
  | .product _ _ limb => if limb = 0 then 0 else layout.zeroColumn
  | .root (.precommit role) lane => layout.headerColumn role lane
  | .root (.seen role) lane => layout.headerColumn role lane
  | _ => layout.zeroColumn

def Layout.closedPairs (layout : Layout) : List (Nat × Nat) :=
  inactiveTags.map fun tag =>
    (layout.fieldColumn tag, layout.expectedColumn tag)

def regularRows (layout : Layout) : List Row :=
  (RegularCounter.all.map fun counter =>
    BoundedWordRows.rows (layout.regularWord counter)).flatten

def stepRows (layout : Layout) : List Row :=
  LessThanConstantRows.rows layout.stepWord

def zeroRows (layout : Layout) : List Row :=
  ConstantPins.rows [(layout.zeroColumn, 0)]

def closedRows (layout : Layout) : List Row :=
  ConditionalEqualityRows.rows (layout.fieldColumn .phase)
    layout.closedPairs

def rows (layout : Layout) : List Row :=
  regularRows layout ++ stepRows layout ++ zeroRows layout ++
    closedRows layout

private theorem regularRows_length (layout : Layout) :
    (regularRows layout).length = 100 := by
  simp [regularRows, RegularCounter.all, BoundedWordRows.rows_length,
    Layout.regularWord, RegularCounter.width, phaseBits, segmentIndexBits,
    MemoryWireGeometry.timestampBits, segmentActiveAccessCountBits]

private theorem stepRows_length (layout : Layout) :
    (stepRows layout).length = 25 := by
  rw [stepRows, LessThanConstantRows.rows_length]
  rfl

private theorem zeroRows_length (layout : Layout) :
    (zeroRows layout).length = 1 := by
  simp [zeroRows, ConstantPins.rows]

private theorem closedRows_length (layout : Layout) :
    (closedRows layout).length = 52 := by
  rw [closedRows, ConditionalEqualityRows.rows_length]
  simp [Layout.closedPairs, inactiveTags_length]

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 178 := by
  simp [rows, regularRows_length, stepRows_length, zeroRows_length,
    closedRows_length]

def Placed (layout : Layout) (assignment : Nat → Nat)
    (value : Value) : Prop :=
  ∀ tag, assignment (layout.fieldColumn tag) = value.fieldValue tag

def HeadersPlaced (layout : Layout) (assignment : Nat → Nat)
    (headers : ChainHeaders Digest.Value) : Prop :=
  ∀ role lane, assignment (layout.headerColumn role lane) =
    MemoryClaimCodec.rootValue headers.roots role lane

theorem regular_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    (counter : RegularCounter) :
    Satisfies (BoundedWordRows.rows (layout.regularWord counter))
      assignment := by
  intro row member
  apply holds row
  apply List.mem_append_left
  apply List.mem_append_left
  apply List.mem_append_left
  rw [regularRows]
  apply List.mem_flatten.mpr
  refine ⟨BoundedWordRows.rows (layout.regularWord counter), ?_, member⟩
  exact List.mem_map_of_mem counter.mem_all

theorem step_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (stepRows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem zero_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (zeroRows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem closed_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (closedRows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem regular_bound
    {layout : Layout} {assignment : Nat → Nat} {value : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment value)
    (holds : Satisfies (rows layout) assignment)
    (counter : RegularCounter) :
    value.fieldValue counter.tag < 2 ^ counter.width := by
  have bounded := BoundedWordRows.value_lt_twoPower
    counter.fitsGoldilocks canonical one
    (regular_rows_hold holds counter)
  simp only [Layout.regularWord] at bounded
  rw [placed counter.tag] at bounded
  exact bounded

private theorem strict_step_bound
    {layout : Layout} {assignment : Nat → Nat} {value : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment value)
    (holds : Satisfies (rows layout) assignment) :
    value.stepIndex < Lifecycle.claimsPerSegment := by
  have bounded := LessThanConstantRows.value_lt_limit
    layout.stepWord_valid canonical one (step_rows_hold holds)
  simp only [Layout.stepWord] at bounded
  change assignment (layout.fieldColumn .stepIndex) <
    Lifecycle.claimsPerSegment at bounded
  rw [placed .stepIndex] at bounded
  exact bounded

private theorem zero_column_eq_zero
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.zeroColumn = 0 := by
  have pinHolds := zero_rows_hold holds
    (ConstantPins.pinRow (layout.zeroColumn, 0))
    (by simp [zeroRows, ConstantPins.rows])
  have defined := builderLinearRow_sound canonical one layout.zeroColumn []
    (by simp [CanonicalTerms]) (by
      simpa [ConstantPins.pinRow] using pinHolds)
  simpa [lcEval] using defined

def expectedValue
    (headers : ChainHeaders Digest.Value) : CarryTag → Nat
  | .product _ _ limb => if limb = 0 then 1 else 0
  | .root (.precommit role) lane =>
      MemoryClaimCodec.rootValue headers.roots role lane
  | .root (.seen role) lane =>
      MemoryClaimCodec.rootValue headers.roots role lane
  | _ => 0

private theorem expected_column_placed
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    (one : assignment 0 = 1)
    (zero : assignment layout.zeroColumn = 0)
    (headersPlaced : HeadersPlaced layout assignment headers)
    (tag : CarryTag) (member : tag ∈ inactiveTags) :
    assignment (layout.expectedColumn tag) = expectedValue headers tag := by
  cases tag with
  | phase => exact False.elim ((by decide :
      MemoryCarryCodec.FieldTag.phase ∉ inactiveTags) member)
  | segmentIndex => exact False.elim ((by decide :
      MemoryCarryCodec.FieldTag.segmentIndex ∉ inactiveTags) member)
  | stepIndex => simpa [Layout.expectedColumn, expectedValue] using zero
  | globalTimestamp => exact False.elim ((by decide :
      MemoryCarryCodec.FieldTag.globalTimestamp ∉ inactiveTags) member)
  | segmentStartTimestamp =>
      simpa [Layout.expectedColumn, expectedValue] using zero
  | segmentActiveAccessCount =>
      simpa [Layout.expectedColumn, expectedValue] using zero
  | segmentEndTimestamp =>
      simpa [Layout.expectedColumn, expectedValue] using zero
  | challenge repetition coordinate limb =>
      simpa [Layout.expectedColumn, expectedValue] using zero
  | product repetition role limb =>
      fin_cases limb
      · simpa [Layout.expectedColumn, expectedValue] using one
      · simpa [Layout.expectedColumn, expectedValue] using zero
  | root source lane =>
      cases source with
      | memory =>
          fin_cases lane <;>
            exact False.elim ((by decide :
              MemoryCarryCodec.FieldTag.root .memory _ ∉ inactiveTags) member)
      | precommit role =>
          simpa [Layout.expectedColumn, expectedValue] using
            headersPlaced role lane
      | seen role =>
          simpa [Layout.expectedColumn, expectedValue] using
            headersPlaced role lane

private theorem inactive_field_equal
    {layout : Layout} {assignment : Nat → Nat} {value : Value}
    {headers : ChainHeaders Digest.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (phaseClosed : assignment (layout.fieldColumn .phase) = 0)
    (placed : Placed layout assignment value)
    (headersPlaced : HeadersPlaced layout assignment headers)
    (holds : Satisfies (rows layout) assignment)
    (tag : CarryTag) (member : tag ∈ inactiveTags) :
    value.fieldValue tag = expectedValue headers tag := by
  have zero := zero_column_eq_zero canonical one holds
  have gates := ConditionalEqualityRows.rows_sound_closed canonical one
    phaseClosed (by simpa [closedRows] using closed_rows_hold holds)
  have pairMember :
      (layout.fieldColumn tag, layout.expectedColumn tag) ∈
        layout.closedPairs :=
    List.mem_map.mpr ⟨tag, member, rfl⟩
  have equalColumns := gates _ pairMember
  rw [placed tag] at equalColumns
  rw [expected_column_placed one zero headersPlaced tag member] at equalColumns
  exact equalColumns

private theorem closed_fields_of_rows
    {layout : Layout} {assignment : Nat → Nat} {value : Value}
    {headers : ChainHeaders Digest.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment value)
    (headersPlaced : HeadersPlaced layout assignment headers)
    (holds : Satisfies (rows layout) assignment)
    (phaseClosed : value.phase = .closed) :
    ClosedFieldsCanonical headers value := by
  have phaseColumnClosed :
      assignment (layout.fieldColumn .phase) = 0 := by
    rw [placed .phase]
    simp [Value.fieldValue, phaseValue, phaseClosed]
  have fieldEqual := inactive_field_equal canonical one phaseColumnClosed
    placed headersPlaced holds
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [Value.fieldValue, expectedValue] using
      fieldEqual .stepIndex (by simp [inactiveTags])
  · simpa [Value.fieldValue, expectedValue] using
      fieldEqual .segmentStartTimestamp (by simp [inactiveTags])
  · simpa [Value.fieldValue, expectedValue] using
      fieldEqual .segmentActiveAccessCount (by simp [inactiveTags])
  · simpa [Value.fieldValue, expectedValue] using
      fieldEqual .segmentEndTimestamp (by simp [inactiveTags])
  · funext repetition
    apply MemoryClaimCodec.challenge_eq_of_values
    intro coordinate limb
    have equal := fieldEqual (.challenge repetition coordinate limb) (by
      fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
        simp [inactiveTags, MemoryCarryCodec.challengeSchema])
    have zeroValue :
        MemoryClaimCodec.challengeValue (zeroChallengesK repetition)
            coordinate limb = 0 := by
      fin_cases coordinate <;> fin_cases limb <;> rfl
    rw [zeroValue]
    simpa [Value.fieldValue, expectedValue] using equal
  · funext repetition
    apply MemoryClaimCodec.product_eq_of_values
    intro role limb
    have equal := fieldEqual (.product repetition role limb) (by
      fin_cases repetition <;> cases role <;> fin_cases limb <;>
        simp [inactiveTags, MemoryCarryCodec.productSchema,
          MemoryClaimCodec.productRoles])
    have oneValue :
        MemoryClaimCodec.productValue (oneProductsK repetition) role limb =
          if limb = 0 then 1 else 0 := by
      cases role <;> fin_cases limb <;> rfl
    rw [oneValue]
    simpa [Value.fieldValue, expectedValue] using equal
  · apply MemoryClaimCodec.roots_eq_of_values
    intro role lane
    have equal := fieldEqual (.root (.precommit role) lane) (by
      cases role <;> fin_cases lane <;>
        simp [inactiveTags, closedRootTags, closedRootSources])
    simpa [Value.fieldValue, rootSourceValue, expectedValue] using equal
  · apply MemoryClaimCodec.roots_eq_of_values
    intro role lane
    have equal := fieldEqual (.root (.seen role) lane) (by
      cases role <;> fin_cases lane <;>
        simp [inactiveTags, closedRootTags, closedRootSources])
    simpa [Value.fieldValue, rootSourceValue, expectedValue] using equal

/-- All carry canonicality, including closed inactive fields, is a conclusion
of concrete rows and verifier-header placement. -/
theorem value_canonical_of_rows
    {layout : Layout} {assignment : Nat → Nat} {value : Value}
    {headers : ChainHeaders Digest.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment value)
    (headersPlaced : HeadersPlaced layout assignment headers)
    (holds : Satisfies (rows layout) assignment) :
    value.Canonical headers where
  segmentIndex := by
    simpa [RegularCounter.tag, RegularCounter.width, Value.fieldValue] using
      regular_bound canonical one placed holds .segmentIndex
  stepIndex := strict_step_bound canonical one placed holds
  globalTimestamp := by
    simpa [RegularCounter.tag, RegularCounter.width, Value.fieldValue] using
      regular_bound canonical one placed holds .globalTimestamp
  segmentStartTimestamp := by
    simpa [RegularCounter.tag, RegularCounter.width, Value.fieldValue] using
      regular_bound canonical one placed holds .segmentStartTimestamp
  segmentActiveAccessCount := by
    simpa [RegularCounter.tag, RegularCounter.width, Value.fieldValue] using
      regular_bound canonical one placed holds .segmentActiveAccessCount
  segmentEndTimestamp := by
    simpa [RegularCounter.tag, RegularCounter.width, Value.fieldValue] using
      regular_bound canonical one placed holds .segmentEndTimestamp
  closedFields := closed_fields_of_rows canonical one placed headersPlaced holds

/-- Artifact-facing row certificate. It does not contain the resulting
canonicality predicate. -/
structure CallSite (programRows : List Row) (assignment : Nat → Nat)
    (headers : ChainHeaders Digest.Value) (value : Value) where
  layout : Layout
  rowsIncluded : rowsIncluded (rows layout) programRows = true
  canonicalAssignment : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  placed : Placed layout assignment value
  headersPlaced : HeadersPlaced layout assignment headers

theorem CallSite.sound
    {programRows : List Row} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value} {value : Value}
    (site : CallSite programRows assignment headers value)
    (satisfies : Satisfies programRows assignment) :
    value.Canonical headers := by
  apply value_canonical_of_rows site.canonicalAssignment site.one
    site.placed site.headersPlaced
  intro row member
  exact satisfies row (rowsIncluded_sound site.rowsIncluded row member)

end Nightstream.Implementation.Nebula.MemoryCarryRows
