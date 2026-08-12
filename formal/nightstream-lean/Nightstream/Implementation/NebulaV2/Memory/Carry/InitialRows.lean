import Nightstream.Implementation.NebulaV2.Memory.Carry.PublicRows
import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: verifier-authoritative canonical chain-start memory carry.

Assurance tier: implementation model.

Owns the exact base pins `phase = Closed`, `segment_index = 0`, and
`global_timestamp = 0`, plus four equality rows from the carry memory root to
the verifier-derived initial-memory root. Combined with the mandatory carry
parser rows, it derives the complete canonical closed carry at chain start.

Does not own computation of the initial-memory root from the public memory
plan, either carry parser, segment opening, application initialization,
absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.InitialMemoryCarryRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime

structure Layout where
  carry : MemoryCarryPublicRows.Layout
  initialMemoryRootColumn : Fin 4 → Nat

def pins (layout : Layout) : List (Nat × Nat) :=
  [ (layout.carry.carry.fieldColumn .phase, 0)
  , (layout.carry.carry.fieldColumn .segmentIndex, 0)
  , (layout.carry.carry.fieldColumn .globalTimestamp, 0)
  ]

def rootPairs (layout : Layout) : List (Nat × Nat) :=
  List.ofFn fun lane : Fin 4 =>
    (layout.carry.carry.fieldColumn (.root .memory lane),
      layout.initialMemoryRootColumn lane)

def rows (layout : Layout) : List Row :=
  ConstantPins.rows (pins layout) ++ EqualityPins.rows (rootPairs layout)

theorem pins_length (layout : Layout) : (pins layout).length = 3 := by
  simp [pins]

theorem rootPairs_length (layout : Layout) :
    (rootPairs layout).length = 4 := by
  simp [rootPairs]

theorem rows_length_exact (layout : Layout) : (rows layout).length = 7 := by
  simp [rows, ConstantPins.rows, EqualityPins.rows, pins_length,
    rootPairs_length]

private theorem pinValuesCanonical (layout : Layout) :
    ConstantPins.ValuesCanonical (pins layout) := by
  intro pin member
  simp only [pins, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl <;> norm_num [goldilocksP]

private theorem constantRowsHold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (ConstantPins.rows (pins layout)) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem equalityRowsHold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (EqualityPins.rows (rootPairs layout)) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

private theorem selfIncluded (program : List Row) :
    rowsIncluded program program = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true member

def InitialMemoryRootPlaced (layout : Layout) (assignment : Nat → Nat)
    (root : Digest.Value) : Prop :=
  ∀ lane,
    assignment (layout.initialMemoryRootColumn lane) =
      (root.lanes lane).val

/-- Exact semantic result of the seven base-authority rows. -/
structure Exact (value : MemoryCarryCodec.Value)
    (initialMemoryRoot : Digest.Value) : Prop where
  phase : value.phase = .closed
  segmentIndex : value.segmentIndex = 0
  globalTimestamp : value.globalTimestamp = 0
  memoryRoot : value.memoryRoot = initialMemoryRoot

/-- Unique complete field-native carry at the verifier-owned chain start. -/
def expectedValue
    (headers : ChainHeaders Digest.Value)
    (initialMemoryRoot : Digest.Value) : MemoryCarryCodec.Value :=
  { phase := .closed
    segmentIndex := 0
    stepIndex := 0
    globalTimestamp := 0
    segmentStartTimestamp := 0
    segmentActiveAccessCount := 0
    segmentEndTimestamp := 0
    challenges := MemoryCarryCodec.zeroChallengesK
    products := MemoryCarryCodec.oneProductsK
    dPre := headers.roots
    dSeen := headers.roots
    memoryRoot := initialMemoryRoot }

theorem expectedValue_canonical
    (headers : ChainHeaders Digest.Value)
    (initialMemoryRoot : Digest.Value) :
    (expectedValue headers initialMemoryRoot).Canonical headers := by
  refine
    { segmentIndex := by simp [expectedValue, MemoryWireGeometry.segmentIndexBits]
      stepIndex := by simp [expectedValue, Lifecycle.claimsPerSegment]
      globalTimestamp := by simp [expectedValue, MemoryWireGeometry.timestampBits]
      segmentStartTimestamp := by
        simp [expectedValue, MemoryWireGeometry.timestampBits]
      segmentActiveAccessCount := by
        simp [expectedValue, MemoryWireGeometry.segmentActiveAccessCountBits]
      segmentEndTimestamp := by
        simp [expectedValue, MemoryWireGeometry.timestampBits]
      closedFields := ?_ }
  intro _phase
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The four authoritative fields plus parser-derived closed-field
canonicality determine the complete chain-start carry. -/
theorem Exact.value_eq_expected
    {headers : ChainHeaders Digest.Value}
    {value : MemoryCarryCodec.Value}
    {initialMemoryRoot : Digest.Value}
    (exact : Exact value initialMemoryRoot)
    (canonical : value.Canonical headers) :
    value = expectedValue headers initialMemoryRoot := by
  rcases canonical.closedFields exact.phase with
    ⟨step, start, count, finish, challenges, products, dPre, dSeen⟩
  apply CarryEncoding.WireCarry.ext
  · exact exact.phase
  · exact exact.segmentIndex
  · exact step
  · exact exact.globalTimestamp
  · exact start
  · exact count
  · exact finish
  · exact challenges
  · exact products
  · exact dPre
  · exact dSeen
  · exact exact.memoryRoot

/-- The seven rows derive the authoritative base fields. Canonical inactive
fields come from `parsed.rowCanonical`; they are not repeated as assumptions. -/
theorem sound
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {value : MemoryCarryCodec.Value}
    {initialMemoryRoot : Digest.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.carry assignment
      headers value)
    (initialRootPlaced :
      InitialMemoryRootPlaced layout assignment initialMemoryRoot)
    (holds : Satisfies (rows layout) assignment) :
    Exact value initialMemoryRoot := by
  have pinFacts := ConstantPins.sound (pinValuesCanonical layout)
    (selfIncluded (ConstantPins.rows (pins layout))) canonical one
    (constantRowsHold holds)
  have equalityFacts := EqualityPins.rows_sound canonical one
    (equalityRowsHold holds)
  have phaseColumn :
      assignment (layout.carry.carry.fieldColumn .phase) = 0 :=
    pinFacts (layout.carry.carry.fieldColumn .phase, 0) (by simp [pins])
  have phaseExact : value.phase = .closed := by
    have encoded : phaseValue value.phase = 0 := by
      have placedPhase := parsed.placed .phase
      change assignment (layout.carry.carry.fieldColumn .phase) =
        phaseValue value.phase at placedPhase
      exact placedPhase.symm.trans phaseColumn
    cases phase : value.phase
    · rfl
    · rw [phase] at encoded
      contradiction
  have segmentExact : value.segmentIndex = 0 := by
    have placedSegment := parsed.placed .segmentIndex
    change assignment (layout.carry.carry.fieldColumn .segmentIndex) =
      value.segmentIndex at placedSegment
    exact placedSegment.symm.trans
      (pinFacts (layout.carry.carry.fieldColumn .segmentIndex, 0)
        (by simp [pins]))
  have timestampExact : value.globalTimestamp = 0 := by
    have placedTimestamp := parsed.placed .globalTimestamp
    change assignment (layout.carry.carry.fieldColumn .globalTimestamp) =
      value.globalTimestamp at placedTimestamp
    exact placedTimestamp.symm.trans
      (pinFacts (layout.carry.carry.fieldColumn .globalTimestamp, 0)
        (by simp [pins]))
  have memoryExact : value.memoryRoot = initialMemoryRoot := by
    apply MemoryClaimCodec.digest_eq_of_lane_values
    intro lane
    have member :
        (layout.carry.carry.fieldColumn (.root .memory lane),
          layout.initialMemoryRootColumn lane) ∈ rootPairs layout :=
      List.mem_ofFn.mpr ⟨lane, rfl⟩
    calc
      (value.memoryRoot.lanes lane).val =
          assignment
            (layout.carry.carry.fieldColumn (.root .memory lane)) := by
        exact (parsed.placed (.root .memory lane)).symm
      _ = assignment (layout.initialMemoryRootColumn lane) :=
        equalityFacts _ member
      _ = (initialMemoryRoot.lanes lane).val := initialRootPlaced lane
  exact ⟨phaseExact, segmentExact, timestampExact, memoryExact⟩

/-- Direct form used by the F-prime base branch: satisfying parser and
authority rows reconstruct the one complete canonical `z0` memory carry. -/
theorem sound_value_eq_expected
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {value : MemoryCarryCodec.Value}
    {initialMemoryRoot : Digest.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.carry assignment
      headers value)
    (initialRootPlaced :
      InitialMemoryRootPlaced layout assignment initialMemoryRoot)
    (holds : Satisfies (rows layout) assignment) :
    value = expectedValue headers initialMemoryRoot :=
  (sound canonical one parsed initialRootPlaced holds).value_eq_expected
    parsed.parserCanonical

structure Honest (layout : Layout) (assignment : Nat → Nat) : Prop where
  pins : ∀ pin ∈ InitialMemoryCarryRows.pins layout,
    assignment pin.1 = pin.2
  roots : ∀ pair ∈ rootPairs layout,
    assignment pair.1 = assignment pair.2

/-- Honest base authority placement satisfies all seven rows. -/
theorem complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment) :
    Satisfies (rows layout) assignment := by
  have pinsHold := ConstantPins.complete (pinValuesCanonical layout) one
    honest.pins
  have rootsHold := EqualityPins.rows_complete canonical one honest.roots
  intro row member
  rcases List.mem_append.mp member with pinMember | rootMember
  · exact pinsHold row pinMember
  · exact rootsHold row rootMember

end Nightstream.Implementation.NebulaV2.InitialMemoryCarryRows
