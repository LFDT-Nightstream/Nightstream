import Nightstream.Implementation.Nebula.Core.FixedBits
import Nightstream.Implementation.Nebula.Memory.Carry.FieldRows
import Nightstream.Implementation.Nebula.Memory.Carry.Rows

/-!
Contract: fail-closed parser from one exact 3,433-bit V2 recursive memory
carry to the concrete typed carry value.

Assurance tier: implementation model.

Owns safe counter and field slicing, exact phase decoding, strict
`step_index < 1088`, canonical Goldilocks rejection for all 52 field limbs,
and verifier-owned closed-state header/zero/one checks.

Does not own byte-container framing, state-hash rows, the enclosing recursive
state, or Rust conformance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryCarryParser

open Nightstream.Implementation.Nebula.MemoryCarryCodec
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CarryEncoding
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.MemoryWireGeometry
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.SuperNeo.Concrete

abbrev Block := FixedBits.Word carryBits

local instance concreteKZero : Zero K := ⟨K.zero⟩
local instance concreteKOne : One K := ⟨K.one⟩

inductive Counter where
  | phase
  | segmentIndex
  | stepIndex
  | globalTimestamp
  | segmentStartTimestamp
  | segmentActiveAccessCount
  | segmentEndTimestamp
deriving DecidableEq, Repr

def Counter.width : Counter → Nat
  | .phase => phaseBits
  | .segmentIndex => segmentIndexBits
  | .stepIndex => stepIndexBits
  | .globalTimestamp => MemoryWireGeometry.timestampBits
  | .segmentStartTimestamp => MemoryWireGeometry.timestampBits
  | .segmentActiveAccessCount => segmentActiveAccessCountBits
  | .segmentEndTimestamp => MemoryWireGeometry.timestampBits

def Counter.bitOffset : Counter → Nat
  | .phase => 0
  | .segmentIndex => 1
  | .stepIndex => 8
  | .globalTimestamp => 19
  | .segmentStartTimestamp => 42
  | .segmentActiveAccessCount => 65
  | .segmentEndTimestamp => 82

def Counter.tag : Counter → MemoryCarryCodec.FieldTag
  | .phase => .phase
  | .segmentIndex => .segmentIndex
  | .stepIndex => .stepIndex
  | .globalTimestamp => .globalTimestamp
  | .segmentStartTimestamp => .segmentStartTimestamp
  | .segmentActiveAccessCount => .segmentActiveAccessCount
  | .segmentEndTimestamp => .segmentEndTimestamp

def Counter.value (counter : Counter) (value : Value) : Nat :=
  value.fieldValue counter.tag

theorem Counter.width_eq_tag (counter : Counter) :
    counter.width = counter.tag.bitWidth := by
  cases counter <;> rfl

theorem Counter.bitOffset_eq_tag (counter : Counter) :
    counter.bitOffset = counter.tag.bitOffset := by
  cases counter <;> decide

def counterWord (block : Block) (counter : Counter) :
    FixedBits.Word counter.width :=
  FixedBits.slice block counter.bitOffset counter.width (by
    cases counter <;>
      norm_num [Counter.bitOffset, Counter.width, carryBits,
        carryCounterBits, carryChallengeBits, carryProductBits,
        carryRootBits, phaseBits, segmentIndexBits, stepIndexBits,
        MemoryWireGeometry.timestampBits, segmentActiveAccessCountBits,
        challengeBaseFieldLimbs, productStateBaseFieldLimbs,
        baseFieldBitCount, repetitionCount, challengeElementsPerRepetition,
        extensionLimbCount, productsPerRepetition, digestLimbCount])

def counterValue (block : Block) (counter : Counter) : Nat :=
  FixedBits.decode (counterWord block counter)

theorem counterValue_lt_width (block : Block) (counter : Counter) :
    counterValue block counter < 2 ^ counter.width :=
  FixedBits.decode_lt _

private theorem slot_position_lt (slot : MemoryCarryFieldRows.Slot) :
    slot.position < MemoryCarryFieldRows.Slot.all.length :=
  List.idxOf_lt_length_of_mem slot.mem_all

theorem field_slice_fits (slot : MemoryCarryFieldRows.Slot) :
    slot.bitOffset + CanonicalFieldBits.bitCount ≤ carryBits := by
  have positionBound := slot_position_lt slot
  rw [MemoryCarryFieldRows.Slot.all_length_exact] at positionBound
  rw [MemoryCarryFieldRows.Slot.bitOffset,
    MemoryCarryFieldRows.fieldBitStart_exact,
    MemoryWireGeometry.carryBits_exact]
  norm_num [CanonicalFieldBits.bitCount]
  omega

def fieldWord (block : Block) (slot : MemoryCarryFieldRows.Slot) :
    CanonicalFieldBits.Word :=
  let sliced := FixedBits.slice block slot.bitOffset
    CanonicalFieldBits.bitCount (field_slice_fits slot)
  ⟨sliced.val, sliced.property⟩

def rawWords (block : Block) : MemoryCarryFieldRows.RawWords :=
  fun slot => fieldWord block slot

def fieldsCanonical (block : Block) : Bool :=
  MemoryCarryFieldRows.Slot.all.all fun slot =>
    decide (CanonicalFieldBits.decode (fieldWord block slot) <
      ShiftedTernary41V1.modulus)

theorem field_canonical_of_all
    {block : Block} (allCanonical : fieldsCanonical block = true)
    (slot : MemoryCarryFieldRows.Slot) :
    CanonicalFieldBits.Canonical (fieldWord block slot) := by
  have every := List.all_eq_true.mp allCanonical slot slot.mem_all
  simpa [CanonicalFieldBits.Canonical] using of_decide_eq_true every

def decodedField (block : Block) (slot : MemoryCarryFieldRows.Slot) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  (FieldCodec.nativeDecode (fieldWord block slot)).getD
    CanonicalFieldBits.zero

theorem nativeDecode_field
    {block : Block} (allCanonical : fieldsCanonical block = true)
    (slot : MemoryCarryFieldRows.Slot) :
    FieldCodec.nativeDecode (fieldWord block slot) =
      some (decodedField block slot) := by
  have canonical := field_canonical_of_all allCanonical slot
  change CanonicalFieldBits.decode (fieldWord block slot) <
    ShiftedTernary41V1.modulus at canonical
  have decodedExact :
      FieldCodec.nativeDecode (fieldWord block slot) =
        some ⟨CanonicalFieldBits.decode (fieldWord block slot), canonical⟩ := by
    simp [FieldCodec.nativeDecode, canonical]
  rw [decodedField, decodedExact]
  rfl

def toF (value : ShiftedTernary41V1.CanonicalGoldilocks) : F :=
  ⟨value.val, by
    simpa [ShiftedTernary41V1.modulus, goldilocksModulus] using value.property⟩

def decodedK (block : Block)
    (slot0 slot1 : MemoryCarryFieldRows.Slot) : K :=
  ⟨toF (decodedField block slot0), toF (decodedField block slot1)⟩

def decodedChallenges (block : Block) : Challenges K :=
  fun repetition =>
    { gamma1 := decodedK block
        (.challenge repetition 0 0) (.challenge repetition 0 1)
      gamma2 := decodedK block
        (.challenge repetition 1 0) (.challenge repetition 1 1) }

def decodedProduct (block : Block) (repetition : Fin 2) : Four K :=
  { initialSnapshot := decodedK block
      (.product repetition .initialSnapshot 0)
      (.product repetition .initialSnapshot 1)
    writes := decodedK block
      (.product repetition .writes 0) (.product repetition .writes 1)
    reads := decodedK block
      (.product repetition .reads 0) (.product repetition .reads 1)
    finalSnapshot := decodedK block
      (.product repetition .finalSnapshot 0)
      (.product repetition .finalSnapshot 1) }

def decodedProducts (block : Block) : State K :=
  fun repetition => decodedProduct block repetition

def decodedDigest (block : Block) (source : RootSource) : Digest.Value where
  lanes := fun lane => decodedField block (.root source lane)

def decodedRoots (block : Block) (seen : Bool) : Roots Digest.Value :=
  if seen then
    { operations := decodedDigest block (.seen .operations)
      initialSnapshot := decodedDigest block (.seen .initialSnapshot)
      finalSnapshot := decodedDigest block (.seen .finalSnapshot) }
  else
    { operations := decodedDigest block (.precommit .operations)
      initialSnapshot := decodedDigest block (.precommit .initialSnapshot)
      finalSnapshot := decodedDigest block (.precommit .finalSnapshot) }

def decodedPhase (block : Block) : PhaseTag :=
  if counterValue block .phase = 0 then .closed else .active

def decodedValue (block : Block) : Value :=
  { phase := decodedPhase block
    segmentIndex := counterValue block .segmentIndex
    stepIndex := counterValue block .stepIndex
    globalTimestamp := counterValue block .globalTimestamp
    segmentStartTimestamp := counterValue block .segmentStartTimestamp
    segmentActiveAccessCount := counterValue block .segmentActiveAccessCount
    segmentEndTimestamp := counterValue block .segmentEndTimestamp
    challenges := decodedChallenges block
    products := decodedProducts block
    dPre := decodedRoots block false
    dSeen := decodedRoots block true
    memoryRoot := decodedDigest block .memory }

theorem decodedValue_counterValue (block : Block) (counter : Counter) :
    (decodedValue block).fieldValue counter.tag = counterValue block counter := by
  cases counter with
  | phase =>
      have bound : counterValue block .phase < 2 := by
        simpa [Counter.width, phaseBits] using
          counterValue_lt_width block .phase
      have exactValue : counterValue block .phase = 0 ∨
          counterValue block .phase = 1 := by omega
      rcases exactValue with exactValue | exactValue <;>
        simp [decodedValue, decodedPhase, Counter.tag, Value.fieldValue,
          phaseValue, exactValue]
  | segmentIndex => rfl
  | stepIndex => rfl
  | globalTimestamp => rfl
  | segmentStartTimestamp => rfl
  | segmentActiveAccessCount => rfl
  | segmentEndTimestamp => rfl

def closedCheck (headers : ChainHeaders Digest.Value) (value : Value) : Prop :=
  value.phase = .closed →
    MemoryCarryCodec.ClosedFieldsCanonical headers value

instance (headers : ChainHeaders Digest.Value) (value : Value) :
    Decidable (closedCheck headers value) := by
  unfold closedCheck MemoryCarryCodec.ClosedFieldsCanonical
  infer_instance

/-- The fail-closed logical-bit parser. -/
def parse (headers : ChainHeaders Digest.Value) (block : Block) :
    Option Value :=
  if stepBound : counterValue block .stepIndex < Lifecycle.claimsPerSegment then
    if _allCanonical : fieldsCanonical block = true then
      if _closed : closedCheck headers (decodedValue block) then
        some (decodedValue block)
      else none
    else none
  else none

theorem parse_some_checks
    {headers : ChainHeaders Digest.Value} {block : Block} {value : Value}
    (accepted : parse headers block = some value) :
    ∃ stepBound : counterValue block .stepIndex < Lifecycle.claimsPerSegment,
      fieldsCanonical block = true ∧
        closedCheck headers (decodedValue block) ∧
        value = decodedValue block := by
  unfold parse at accepted
  split at accepted
  next stepBound =>
    split at accepted
    next allCanonical =>
      split at accepted
      next closed =>
        exact ⟨stepBound, allCanonical, closed,
          Option.some.inj accepted.symm⟩
      next => simp at accepted
    next => simp at accepted
  next => simp at accepted

theorem parse_counterValue
    {headers : ChainHeaders Digest.Value} {block : Block} {value : Value}
    (accepted : parse headers block = some value) (counter : Counter) :
    value.fieldValue counter.tag = counterValue block counter := by
  rcases parse_some_checks accepted with
    ⟨stepBound, allCanonical, closed, valueEqual⟩
  subst value
  exact decodedValue_counterValue block counter

set_option maxHeartbeats 2000000 in
theorem decodedValue_canonicalValue
    (block : Block) (slot : MemoryCarryFieldRows.Slot) :
    slot.canonicalValue (decodedValue block) = decodedField block slot := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;>
        apply Subtype.ext <;> rfl
  | product repetition role limb =>
      cases role <;> fin_cases limb <;> apply Subtype.ext <;> rfl
  | root source lane =>
      cases source with
      | memory => rfl
      | precommit role => cases role <;> rfl
      | seen role => cases role <;> rfl

theorem parse_native_parses
    {headers : ChainHeaders Digest.Value} {block : Block} {value : Value}
    (accepted : parse headers block = some value) :
    MemoryCarryFieldRows.NativeParses (rawWords block) value := by
  rcases parse_some_checks accepted with
    ⟨stepBound, allCanonical, closed, valueEqual⟩
  subst value
  intro slot
  rw [rawWords, decodedValue_canonicalValue]
  exact nativeDecode_field allCanonical slot

theorem parse_value_canonical
    {headers : ChainHeaders Digest.Value} {block : Block} {value : Value}
    (accepted : parse headers block = some value) :
    value.Canonical headers := by
  rcases parse_some_checks accepted with
    ⟨stepBound, allCanonical, closed, valueEqual⟩
  subst value
  constructor
  · exact counterValue_lt_width block .segmentIndex
  · exact stepBound
  · exact counterValue_lt_width block .globalTimestamp
  · exact counterValue_lt_width block .segmentStartTimestamp
  · exact counterValue_lt_width block .segmentActiveAccessCount
  · exact counterValue_lt_width block .segmentEndTimestamp
  · exact closed

/-- The unique semantic carry selected by one canonical wire value. The step
bound is supplied by the parser and is used only in the active branch. -/
def semanticCarry (value : Value)
    (stepBound : value.stepIndex < Lifecycle.claimsPerSegment) :
    Carry Digest.Value (Challenges K) (State K) :=
  match value.phase with
  | .closed =>
      .closed
        { segmentIndex := value.segmentIndex
          globalTimestamp := value.globalTimestamp
          memoryRoot := value.memoryRoot }
  | .active =>
      .active
        { segmentIndex := value.segmentIndex
          stepIndex := ⟨value.stepIndex, stepBound⟩
          globalTimestamp := value.globalTimestamp
          segmentStartTimestamp := value.segmentStartTimestamp
          segmentActiveAccessCount := value.segmentActiveAccessCount
          segmentEndTimestamp := value.segmentEndTimestamp
          challenge := value.challenges
          products := value.products
          dPre := value.dPre
          dSeen := value.dSeen
          memoryRoot := value.memoryRoot }

/-- A canonical parsed wire value is not an opaque state. It decodes to the
exact semantic closed or active carry selected by its phase bit. -/
theorem canonical_decodes
    {headers : ChainHeaders Digest.Value} {value : Value}
    (canonical : value.Canonical headers) :
    CarryEncoding.Decodes headers value
      (semanticCarry value canonical.stepIndex) := by
  cases phaseEq : value.phase with
  | closed =>
      have closed := canonical.closedFields phaseEq
      have exactWire : value = CarryEncoding.encodeClosed headers
          { segmentIndex := value.segmentIndex
            globalTimestamp := value.globalTimestamp
            memoryRoot := value.memoryRoot } := by
        apply WireCarry.ext
        · exact phaseEq
        · rfl
        · exact closed.1
        · rfl
        · exact closed.2.1
        · exact closed.2.2.1
        · exact closed.2.2.2.1
        · simpa [zeroChallengesK, CarryEncoding.zeroChallenges] using
            closed.2.2.2.2.1
        · simpa [oneProductsK, ProductState.one] using
            closed.2.2.2.2.2.1
        · exact closed.2.2.2.2.2.2.1
        · exact closed.2.2.2.2.2.2.2
        · rfl
      simp only [semanticCarry, phaseEq]
      rw [exactWire]
      exact CarryEncoding.Decodes.closed _
  | active =>
      let active : ActiveCarry Digest.Value (Challenges K) (State K) :=
          { segmentIndex := value.segmentIndex
            stepIndex := ⟨value.stepIndex, canonical.stepIndex⟩
            globalTimestamp := value.globalTimestamp
            segmentStartTimestamp := value.segmentStartTimestamp
            segmentActiveAccessCount := value.segmentActiveAccessCount
            segmentEndTimestamp := value.segmentEndTimestamp
            challenge := value.challenges
            products := value.products
            dPre := value.dPre
            dSeen := value.dSeen
            memoryRoot := value.memoryRoot }
      have exactWire : value = CarryEncoding.encodeActive active := by
        apply WireCarry.ext
        · exact phaseEq
        · rfl
        · rfl
        · rfl
        · rfl
        · rfl
        · rfl
        · rfl
        · rfl
        · rfl
        · rfl
        · rfl
      have semanticExact :
          semanticCarry value canonical.stepIndex = .active active := by
        simp [semanticCarry, phaseEq, active]
      rw [semanticExact]
      rw [exactWire]
      exact CarryEncoding.Decodes.active active

/-- Successful parsing returns the exact wire and its exact semantic carry. -/
theorem parse_decodes
    {headers : ChainHeaders Digest.Value} {block : Block} {value : Value}
    (accepted : parse headers block = some value) :
    CarryEncoding.Decodes headers value
      (semanticCarry value (parse_value_canonical accepted).stepIndex) :=
  canonical_decodes (parse_value_canonical accepted)

theorem rejects_modulus_alias
    {headers : ChainHeaders Digest.Value} {block : Block}
    (slot : MemoryCarryFieldRows.Slot)
    (aliasEq : fieldWord block slot = CanonicalFieldBits.modulusWord) :
    parse headers block = none := by
  apply Option.eq_none_iff_forall_not_mem.mpr
  intro value accepted
  have parsed := parse_native_parses
    (show parse headers block = some value from accepted)
  have decoded := parsed slot
  rw [rawWords, aliasEq, FieldCodec.rejects_zero_modulus_alias.2] at decoded
  simp at decoded

def blockOfValue {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers) : Block :=
  ⟨MemoryCarryCodec.encode value,
    MemoryCarryCodec.encode_length value,
    fun digit member => by
      simp only [MemoryCarryCodec.encode,
        MemoryCarryCodec.encodeFor, List.mem_flatMap] at member
      obtain ⟨tag, _tagMember, digitMember⟩ := member
      exact WasmStateCodec.encodeWord_binary _ _ digit digitMember⟩

theorem counterWord_blockOfValue
    {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers) (counter : Counter) :
    (counterWord (blockOfValue value canonical) counter).val =
      MemoryCarryCodec.encodeFields value counter.tag := by
  change
    ((MemoryCarryCodec.encode value).drop counter.bitOffset).take
        counter.width = MemoryCarryCodec.encodeFields value counter.tag
  rw [counter.bitOffset_eq_tag, counter.width_eq_tag]
  exact MemoryCarryCodec.encode_slice value counter.tag

theorem counterValue_blockOfValue
    {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers) (counter : Counter) :
    counterValue (blockOfValue value canonical) counter = counter.value value := by
  rw [counterValue, FixedBits.decode, counterWord_blockOfValue]
  change Nat.ofDigits 2
      (WasmStateCodec.encodeWord counter.tag.bitWidth
        (value.fieldValue counter.tag)) = value.fieldValue counter.tag
  exact WasmStateCodec.ofDigits_encodeWord_of_bound
    (value.fieldValue_lt_width canonical counter.tag)

theorem fieldWord_blockOfValue
    {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers)
    (slot : MemoryCarryFieldRows.Slot) :
    fieldWord (blockOfValue value canonical) slot =
      CanonicalFieldBits.encode (slot.canonicalValue value) := by
  apply Subtype.ext
  change
    ((MemoryCarryCodec.encode value).drop slot.bitOffset).take
        CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (slot.canonicalValue value)).val
  rw [slot.bitOffset_eq_tag]
  have slicedAtTag :
      ((MemoryCarryCodec.encode value).drop slot.tag.bitOffset).take
          CanonicalFieldBits.bitCount =
        MemoryCarryCodec.encodeFields value slot.tag := by
    simpa only [slot.tag_width] using
      MemoryCarryCodec.encode_slice value slot.tag
  rw [slicedAtTag]
  have capacityBound : (slot.canonicalValue value).val <
      2 ^ CanonicalFieldBits.bitCount :=
    (slot.canonicalValue value).property.trans
      CanonicalFieldBits.modulus_lt_capacity
  unfold MemoryCarryCodec.encodeFields MemoryCarryCodec.encodeWord
  rw [slot.tag_width, ← slot.canonicalValue_val value]
  simp [WasmStateCodec.encodeWord, CanonicalFieldBits.encode,
    Nat.mod_eq_of_lt capacityBound]

theorem fieldsCanonical_blockOfValue
    {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers) :
    fieldsCanonical (blockOfValue value canonical) = true := by
  rw [fieldsCanonical, List.all_eq_true]
  intro slot member
  apply decide_eq_true
  rw [fieldWord_blockOfValue value canonical slot,
    CanonicalFieldBits.decode_encode]
  exact (slot.canonicalValue value).property

theorem decodedField_blockOfValue
    {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers)
    (slot : MemoryCarryFieldRows.Slot) :
    decodedField (blockOfValue value canonical) slot =
      slot.canonicalValue value := by
  have decoded :
      FieldCodec.nativeDecode
          (fieldWord (blockOfValue value canonical) slot) =
        some (slot.canonicalValue value) := by
    apply (FieldCodec.nativeDecode_some_iff
      (fieldWord (blockOfValue value canonical) slot)
      (slot.canonicalValue value)).2
    rw [fieldWord_blockOfValue value canonical slot]
    exact ⟨CanonicalFieldBits.encode_is_canonical _,
      CanonicalFieldBits.decode_encode _ |>.symm⟩
  unfold decodedField
  rw [decoded]
  rfl

private theorem decodedValue_fieldValue_at_slot
    {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers)
    (slot : MemoryCarryFieldRows.Slot) :
    (decodedValue (blockOfValue value canonical)).fieldValue slot.tag =
      value.fieldValue slot.tag := by
  calc
    (decodedValue (blockOfValue value canonical)).fieldValue slot.tag =
        (slot.canonicalValue
          (decodedValue (blockOfValue value canonical))).val :=
      (slot.canonicalValue_val _).symm
    _ = (decodedField (blockOfValue value canonical) slot).val :=
      congrArg Subtype.val (decodedValue_canonicalValue _ slot)
    _ = (slot.canonicalValue value).val :=
      congrArg Subtype.val (decodedField_blockOfValue value canonical slot)
    _ = value.fieldValue slot.tag := slot.canonicalValue_val value

theorem decodedValue_blockOfValue
    {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers) :
    decodedValue (blockOfValue value canonical) = value := by
  apply Value.fieldValue_injective
  funext tag
  cases tag with
  | phase =>
      have phaseCounter := counterValue_blockOfValue value canonical .phase
      cases phaseExact : value.phase <;>
        simp [Counter.value, Counter.tag, decodedValue, decodedPhase,
          Value.fieldValue, phaseValue, phaseExact, phaseCounter]
  | segmentIndex =>
      exact counterValue_blockOfValue value canonical .segmentIndex
  | stepIndex => exact counterValue_blockOfValue value canonical .stepIndex
  | globalTimestamp =>
      exact counterValue_blockOfValue value canonical .globalTimestamp
  | segmentStartTimestamp =>
      exact counterValue_blockOfValue value canonical .segmentStartTimestamp
  | segmentActiveAccessCount =>
      exact counterValue_blockOfValue value canonical
        .segmentActiveAccessCount
  | segmentEndTimestamp =>
      exact counterValue_blockOfValue value canonical .segmentEndTimestamp
  | challenge repetition coordinate limb =>
      exact decodedValue_fieldValue_at_slot value canonical
        (.challenge repetition coordinate limb)
  | product repetition role limb =>
      exact decodedValue_fieldValue_at_slot value canonical
        (.product repetition role limb)
  | root source lane =>
      exact decodedValue_fieldValue_at_slot value canonical (.root source lane)

theorem parse_blockOfValue
    {headers : ChainHeaders Digest.Value}
    (value : Value) (canonical : value.Canonical headers) :
    parse headers (blockOfValue value canonical) = some value := by
  have stepBound :
      counterValue (blockOfValue value canonical) .stepIndex <
        Lifecycle.claimsPerSegment := by
    rw [counterValue_blockOfValue value canonical .stepIndex]
    exact canonical.stepIndex
  have fieldsExact := fieldsCanonical_blockOfValue value canonical
  have valueExact := decodedValue_blockOfValue value canonical
  have closed : closedCheck headers
      (decodedValue (blockOfValue value canonical)) := by
    rw [valueExact]
    exact canonical.closedFields
  unfold parse
  rw [dif_pos stepBound, dif_pos fieldsExact, dif_pos closed, valueExact]

end Nightstream.Implementation.Nebula.MemoryCarryParser
