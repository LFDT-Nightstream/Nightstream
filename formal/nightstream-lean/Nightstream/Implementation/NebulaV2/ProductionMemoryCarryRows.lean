import Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows

/-!
Contract: exact field-native decoder rows for one production memory carry.

The bounded carry counters remain Boolean words. The 52 challenge, product,
and root limbs are canonical native Goldilocks columns. The existing
`MemoryCarryRows` program still enforces the strict step bound and every
closed-state inactive-field rule.

Satisfying rows construct one exact typed carry and the shared
`MemoryCarryPublicRows.ParsedColumnsMatch` interface. No typed carry, phase,
counter bound, field equality, or decoder result is a premise of `derive`.

Does not own header authority, absolute generated columns, state hashing,
memory transitions, terminal verification, or Rust refinement.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.NebulaV2.MemoryCarryRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CarryEncoding
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

abbrev Slot := MemoryCarryFieldRows.Slot

def decodedPhase (value : Nat) : PhaseTag :=
  if value = 0 then .closed else .active

theorem phaseValue_decodedPhase {value : Nat} (bound : value < 2) :
    phaseValue (decodedPhase value) = value := by
  by_cases zero : value = 0
  · simp [decodedPhase, zero, phaseValue]
  · have one : value = 1 := by omega
    simp [decodedPhase, zero, one, phaseValue]

def digestLane (value : F) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  ⟨value.val, by
    simpa [ShiftedTernary41V1.modulus, goldilocksModulus] using value.isLt⟩

def decodedK (field : FieldTag -> F) (low high : FieldTag) : K :=
  ⟨field low, field high⟩

def decodedChallenges (field : FieldTag -> F) : Challenges K :=
  fun repetition =>
    { gamma1 := decodedK field
        (.challenge repetition 0 0) (.challenge repetition 0 1)
      gamma2 := decodedK field
        (.challenge repetition 1 0) (.challenge repetition 1 1) }

def decodedProduct (field : FieldTag -> F) (repetition : Fin 2) : Four K :=
  { initialSnapshot := decodedK field
      (.product repetition .initialSnapshot 0)
      (.product repetition .initialSnapshot 1)
    writes := decodedK field
      (.product repetition .writes 0)
      (.product repetition .writes 1)
    reads := decodedK field
      (.product repetition .reads 0)
      (.product repetition .reads 1)
    finalSnapshot := decodedK field
      (.product repetition .finalSnapshot 0)
      (.product repetition .finalSnapshot 1) }

def decodedProducts (field : FieldTag -> F) : State K :=
  fun repetition => decodedProduct field repetition

def decodedDigest (field : FieldTag -> F) (source : RootSource) :
    Digest.Value where
  lanes := fun lane => digestLane (field (.root source lane))

def decodedRoots (field : FieldTag -> F)
    (source : MemoryClaimCodec.RootRole -> RootSource) :
    Roots Digest.Value :=
  { operations := decodedDigest field (source .operations)
    initialSnapshot := decodedDigest field (source .initialSnapshot)
    finalSnapshot := decodedDigest field (source .finalSnapshot) }

def decodedValue (counter : FieldTag -> Nat) (field : FieldTag -> F) : Value :=
  { phase := decodedPhase (counter .phase)
    segmentIndex := counter .segmentIndex
    stepIndex := counter .stepIndex
    globalTimestamp := counter .globalTimestamp
    segmentStartTimestamp := counter .segmentStartTimestamp
    segmentActiveAccessCount := counter .segmentActiveAccessCount
    segmentEndTimestamp := counter .segmentEndTimestamp
    challenges := decodedChallenges field
    products := decodedProducts field
    dPre := decodedRoots field RootSource.precommit
    dSeen := decodedRoots field RootSource.seen
    memoryRoot := decodedDigest field .memory }

/-- Physical columns for one mixed carry. -/
structure Layout where
  carry : MemoryCarryRows.Layout

def nativeColumnMap (column : Nat) : List Nat := [0, column]

@[simp] theorem nativeColumnMap_zero (column : Nat) :
    Relabel.column (nativeColumnMap column) 0 = 0 := by
  rfl

@[simp] theorem nativeColumnMap_value (column : Nat) :
    Relabel.column (nativeColumnMap column) CanonicalU64.varCol = column := by
  rfl

/-- Compatibility view used by state and transition gadgets. The omitted
canonical-u64 field rows are not part of this production program. -/
def Layout.reference (layout : Layout) : MemoryCarryPublicRows.Layout where
  carry := layout.carry
  fieldColumnMap := fun slot =>
    nativeColumnMap (layout.carry.fieldColumn slot.tag)
  fieldMapsConstantOne := by intro slot; simp
  fieldValueColumn := by intro slot; simp

def rows (layout : Layout) : List Row := MemoryCarryRows.rows layout.carry

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 178 :=
  MemoryCarryRows.rows_length_exact layout.carry

def decodedCounters (layout : Layout) (assignment : Nat -> Nat) :
    FieldTag -> Nat :=
  fun tag => assignment (layout.carry.fieldColumn tag)

def decodedFields (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    FieldTag -> F :=
  fun tag => ⟨assignment (layout.carry.fieldColumn tag), canonical _⟩

private theorem phaseBound
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    decodedCounters layout assignment .phase < 2 := by
  exact BoundedWordRows.value_lt_twoPower
    (RegularCounter.fitsGoldilocks .phase) canonical one
    (MemoryCarryRows.regular_rows_hold satisfied .phase)

/-- Every column is the exact matching field of the deterministic decode. -/
theorem decodedValue_fieldValue
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (phaseIsBit : decodedCounters layout assignment .phase < 2)
    (tag : FieldTag) :
    (decodedValue (decodedCounters layout assignment)
      (decodedFields layout assignment canonical)).fieldValue tag =
        assignment (layout.carry.fieldColumn tag) := by
  cases tag with
  | phase => exact phaseValue_decodedPhase phaseIsBit
  | segmentIndex => rfl
  | stepIndex => rfl
  | globalTimestamp => rfl
  | segmentStartTimestamp => rfl
  | segmentActiveAccessCount => rfl
  | segmentEndTimestamp => rfl
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;> rfl
  | product repetition role limb =>
      cases role <;> fin_cases limb <;> rfl
  | root source lane =>
      cases source with
      | memory => rfl
      | precommit role => cases role <;> rfl
      | seen role => cases role <;> rfl

/-- Row-derived carry decoder output. -/
structure Sound
    (layout : Layout) (assignment : Nat -> Nat)
    (headers : ChainHeaders Digest.Value) where
  value : Value
  parsed : MemoryCarryPublicRows.ParsedColumnsMatch
    layout.reference assignment headers value

/-- Satisfying mixed rows determine one exact typed carry. Header placement
is verifier-owned input authority, not a prover-selected carry conclusion. -/
def derive
    {layout : Layout} {assignment : Nat -> Nat}
    (headers : ChainHeaders Digest.Value)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (headersPlaced : MemoryCarryRows.HeadersPlaced
      layout.carry assignment headers)
    (satisfied : Satisfies (rows layout) assignment) :
    Sound layout assignment headers := by
  let counters := decodedCounters layout assignment
  let fields := decodedFields layout assignment canonical
  have phaseIsBit := phaseBound canonical one satisfied
  let value := decodedValue counters fields
  have placed : MemoryCarryRows.Placed layout.carry assignment value := by
    intro tag
    exact (decodedValue_fieldValue layout assignment canonical phaseIsBit
      tag).symm
  have valueCanonical := MemoryCarryRows.value_canonical_of_rows canonical one
    placed headersPlaced satisfied
  exact
    { value := value
      parsed :=
        { placed := placed
          headersPlaced := headersPlaced
          rowCanonical := valueCanonical
          parserCanonical := valueCanonical } }

/-- Satisfying mixed rows imply an exact typed carry. -/
theorem rows_imply_exact_carry
    {layout : Layout} {assignment : Nat -> Nat}
    (headers : ChainHeaders Digest.Value)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (headersPlaced : MemoryCarryRows.HeadersPlaced
      layout.carry assignment headers)
    (satisfied : Satisfies (rows layout) assignment) :
    exists value,
      MemoryCarryPublicRows.ParsedColumnsMatch
        layout.reference assignment headers value := by
  let result := derive headers canonical one headersPlaced satisfied
  exact ⟨result.value, result.parsed⟩

/-- Two typed carries cannot match the same mixed carrier columns. -/
theorem parsed_unique
    {layout : Layout} {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value} {left right : Value}
    (leftParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.reference assignment headers left)
    (rightParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.reference assignment headers right) :
    left = right := by
  apply Value.fieldValue_injective
  funext tag
  exact (leftParsed.placed tag).symm.trans (rightParsed.placed tag)

/-- The deterministic decoder result is the only compatible typed carry. -/
theorem derive_unique
    {layout : Layout} {assignment : Nat -> Nat}
    (headers : ChainHeaders Digest.Value)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (headersPlaced : MemoryCarryRows.HeadersPlaced
      layout.carry assignment headers)
    (satisfied : Satisfies (rows layout) assignment)
    {value : Value}
    (parsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.reference assignment headers value) :
    (derive headers canonical one headersPlaced satisfied).value = value :=
  parsed_unique (derive headers canonical one headersPlaced satisfied).parsed
    parsed

end Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows
