import Nightstream.Implementation.Nebula.Memory.Claim.Codec
import Nightstream.Protocol.Nebula.CarryEncoding

/-!
Contract: canonical 3,433-bit codec for the V2 recursive memory carry.

Assurance tier: implementation model.

Owns the exact phase, counter, challenge, product, and root word order; the
concrete SuperNeo extension-field coefficient order; closed-state inactive
field canonicality; total length; and encoding injectivity.

Does not own state-hash evaluation, generated Boolean rows, recursive public
column placement, native Rust parsing, or fixed-point compiler closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryCarryCodec

open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CarryEncoding
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.MemoryWireGeometry
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.SuperNeo.Concrete

abbrev Value := WireCarry Digest.Value K

inductive RootSource where
  | precommit (role : RootRole)
  | seen (role : RootRole)
  | memory
deriving DecidableEq, Repr

inductive FieldTag where
  | phase
  | segmentIndex
  | stepIndex
  | globalTimestamp
  | segmentStartTimestamp
  | segmentActiveAccessCount
  | segmentEndTimestamp
  | challenge (repetition coordinate limb : Fin 2)
  | product (repetition : Fin 2) (role : ProductRole) (limb : Fin 2)
  | root (source : RootSource) (lane : Fin 4)
deriving DecidableEq, Repr

def rootSources : List RootSource :=
  [.precommit .operations, .precommit .initialSnapshot,
    .precommit .finalSnapshot, .seen .operations,
    .seen .initialSnapshot, .seen .finalSnapshot, .memory]

def counterSchema : List FieldTag :=
  [.phase, .segmentIndex, .stepIndex, .globalTimestamp,
    .segmentStartTimestamp, .segmentActiveAccessCount,
    .segmentEndTimestamp]

def challengeSchema : List FieldTag :=
  (List.ofFn fun repetition : Fin 2 =>
    (List.ofFn fun coordinate : Fin 2 =>
      List.ofFn fun limb : Fin 2 =>
        FieldTag.challenge repetition coordinate limb).flatten).flatten

def productSchema : List FieldTag :=
  (List.ofFn fun repetition : Fin 2 =>
    (productRoles.map fun role =>
      List.ofFn fun limb : Fin 2 =>
        FieldTag.product repetition role limb).flatten).flatten

def rootSchema : List FieldTag :=
  (rootSources.map fun source =>
    List.ofFn fun lane : Fin 4 => FieldTag.root source lane).flatten

def schema : List FieldTag :=
  counterSchema ++ challengeSchema ++ productSchema ++ rootSchema

theorem schema_nodup : schema.Nodup := by decide

def FieldTag.bitWidth : FieldTag → Nat
  | .phase => phaseBits
  | .segmentIndex => segmentIndexBits
  | .stepIndex => stepIndexBits
  | .globalTimestamp => MemoryWireGeometry.timestampBits
  | .segmentStartTimestamp => MemoryWireGeometry.timestampBits
  | .segmentActiveAccessCount => segmentActiveAccessCountBits
  | .segmentEndTimestamp => MemoryWireGeometry.timestampBits
  | .challenge _ _ _ => baseFieldBitCount
  | .product _ _ _ => baseFieldBitCount
  | .root _ _ => baseFieldBitCount

def phaseValue : PhaseTag → Nat
  | .closed => 0
  | .active => 1

def rootSourceValue (value : Value) (source : RootSource)
    (lane : Fin 4) : Nat :=
  match source with
  | .precommit role => rootValue value.dPre role lane
  | .seen role => rootValue value.dSeen role lane
  | .memory => (value.memoryRoot.lanes lane).val

def Value.fieldValue (value : Value) : FieldTag → Nat
  | .phase => phaseValue value.phase
  | .segmentIndex => value.segmentIndex
  | .stepIndex => value.stepIndex
  | .globalTimestamp => value.globalTimestamp
  | .segmentStartTimestamp => value.segmentStartTimestamp
  | .segmentActiveAccessCount => value.segmentActiveAccessCount
  | .segmentEndTimestamp => value.segmentEndTimestamp
  | .challenge repetition coordinate limb =>
      challengeValue (value.challenges repetition) coordinate limb
  | .product repetition role limb =>
      productValue (value.products repetition) role limb
  | .root source lane => rootSourceValue value source lane

def zeroChallengesK : Challenges K :=
  fun _ => { gamma1 := K.zero, gamma2 := K.zero }

def oneProductsK : State K :=
  fun _ =>
    { initialSnapshot := K.one
      writes := K.one
      reads := K.one
      finalSnapshot := K.one }

/-- Concrete closed-state inactive fields. The active-state constraints are
owned by the F-prime transition relation. -/
def ClosedFieldsCanonical
    (headers : ChainHeaders Digest.Value) (value : Value) : Prop :=
  value.stepIndex = 0 ∧
    value.segmentStartTimestamp = 0 ∧
    value.segmentActiveAccessCount = 0 ∧
    value.segmentEndTimestamp = 0 ∧
    value.challenges = zeroChallengesK ∧
    value.products = oneProductsK ∧
    value.dPre = headers.roots ∧
    value.dSeen = headers.roots

structure Value.Canonical
    (headers : ChainHeaders Digest.Value) (value : Value) : Prop where
  segmentIndex : value.segmentIndex < 2 ^ segmentIndexBits
  stepIndex : value.stepIndex < Lifecycle.claimsPerSegment
  globalTimestamp :
    value.globalTimestamp < 2 ^ MemoryWireGeometry.timestampBits
  segmentStartTimestamp :
    value.segmentStartTimestamp < 2 ^ MemoryWireGeometry.timestampBits
  segmentActiveAccessCount :
    value.segmentActiveAccessCount < 2 ^ segmentActiveAccessCountBits
  segmentEndTimestamp :
    value.segmentEndTimestamp < 2 ^ MemoryWireGeometry.timestampBits
  closedFields : value.phase = .closed → ClosedFieldsCanonical headers value

private theorem fieldValue_lt_wordCapacity (value : F) :
    value.val < 2 ^ baseFieldBitCount := by
  exact value.isLt.trans (by
    norm_num [Nightstream.SuperNeo.Concrete.goldilocksModulus,
      baseFieldBitCount])

private theorem digestValue_lt_wordCapacity
    (value : ShiftedTernary41V1.CanonicalGoldilocks) :
    value.val < 2 ^ baseFieldBitCount := by
  exact value.property.trans (by
    norm_num [ShiftedTernary41V1.modulus, baseFieldBitCount])

theorem Value.fieldValue_lt_width
    {headers : ChainHeaders Digest.Value}
    {value : Value} (canonical : value.Canonical headers)
    (tag : FieldTag) :
    value.fieldValue tag < 2 ^ tag.bitWidth := by
  cases tag with
  | phase =>
      cases phaseExact : value.phase <;>
        simp [Value.fieldValue, phaseValue, FieldTag.bitWidth, phaseBits,
          phaseExact]
  | segmentIndex => exact canonical.segmentIndex
  | stepIndex =>
      exact canonical.stepIndex.trans (by
        show Lifecycle.claimsPerSegment < 2 ^ stepIndexBits
        decide)
  | globalTimestamp => exact canonical.globalTimestamp
  | segmentStartTimestamp => exact canonical.segmentStartTimestamp
  | segmentActiveAccessCount => exact canonical.segmentActiveAccessCount
  | segmentEndTimestamp => exact canonical.segmentEndTimestamp
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;>
        exact fieldValue_lt_wordCapacity _
  | product repetition role limb =>
      cases role <;> fin_cases limb <;>
        exact fieldValue_lt_wordCapacity _
  | root source lane =>
      cases source with
      | memory => exact digestValue_lt_wordCapacity _
      | precommit role =>
          cases role <;> exact digestValue_lt_wordCapacity _
      | seen role =>
          cases role <;> exact digestValue_lt_wordCapacity _

private theorem phaseValue_injective : Function.Injective phaseValue := by
  intro left right equal
  cases left <;> cases right <;> simp_all [phaseValue]

private theorem rootSource_values_determine
    {left right : Value}
    (equal : ∀ source lane,
      rootSourceValue left source lane = rootSourceValue right source lane) :
    left.dPre = right.dPre ∧ left.dSeen = right.dSeen ∧
      left.memoryRoot = right.memoryRoot := by
  refine ⟨roots_eq_of_values (fun role lane => ?_),
    roots_eq_of_values (fun role lane => ?_), ?_⟩
  · exact equal (.precommit role) lane
  · exact equal (.seen role) lane
  · apply digest_eq_of_lane_values
    intro lane
    exact equal .memory lane

theorem Value.fieldValue_injective :
    Function.Injective Value.fieldValue := by
  intro left right equal
  have rootsEqual :
      left.dPre = right.dPre ∧ left.dSeen = right.dSeen ∧
        left.memoryRoot = right.memoryRoot :=
    rootSource_values_determine (fun source lane =>
      congrFun equal (.root source lane))
  apply WireCarry.ext
  · exact phaseValue_injective (congrFun equal .phase)
  · exact congrFun equal .segmentIndex
  · exact congrFun equal .stepIndex
  · exact congrFun equal .globalTimestamp
  · exact congrFun equal .segmentStartTimestamp
  · exact congrFun equal .segmentActiveAccessCount
  · exact congrFun equal .segmentEndTimestamp
  · funext repetition
    apply challenge_eq_of_values
    intro coordinate limb
    exact congrFun equal (.challenge repetition coordinate limb)
  · funext repetition
    apply product_eq_of_values
    intro role limb
    exact congrFun equal (.product repetition role limb)
  · exact rootsEqual.1
  · exact rootsEqual.2.1
  · exact rootsEqual.2.2

def encodeWord :=
  Nightstream.Implementation.Nebula.WasmStateCodec.encodeWord

def encodeFields (value : Value) (tag : FieldTag) : List Nat :=
  encodeWord tag.bitWidth (value.fieldValue tag)

theorem encodeFields_length (value : Value) (tag : FieldTag) :
    (encodeFields value tag).length = tag.bitWidth :=
  Nightstream.Implementation.Nebula.WasmStateCodec.encodeWord_length _ _

def encodeFor (tags : List FieldTag) (value : Value) : List Nat :=
  tags.flatMap (encodeFields value)

def encode (value : Value) : List Nat :=
  encodeFor schema value

theorem encodeFor_length (tags : List FieldTag) (value : Value) :
    (encodeFor tags value).length =
      (tags.map FieldTag.bitWidth).sum := by
  induction tags with
  | nil => rfl
  | cons tag rest inductionHypothesis =>
      simp [encodeFor, encodeFields_length]

set_option maxRecDepth 10000 in
theorem schema_width_exact :
    (schema.map FieldTag.bitWidth).sum = carryBits := by
  decide

theorem encode_length (value : Value) :
    (encode value).length = carryBits := by
  rw [encode, encodeFor_length, schema_width_exact]

theorem encode_exact_length (value : Value) :
    (encode value).length = 3433 := by
  rw [encode_length, carryBits_exact]

theorem FieldTag.mem_schema (tag : FieldTag) : tag ∈ schema := by
  cases tag with
  | phase => simp [schema, counterSchema]
  | segmentIndex => simp [schema, counterSchema]
  | stepIndex => simp [schema, counterSchema]
  | globalTimestamp => simp [schema, counterSchema]
  | segmentStartTimestamp => simp [schema, counterSchema]
  | segmentActiveAccessCount => simp [schema, counterSchema]
  | segmentEndTimestamp => simp [schema, counterSchema]
  | challenge repetition coordinate limb =>
      fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
        simp [schema, challengeSchema]
  | product repetition role limb =>
      fin_cases repetition <;> cases role <;> fin_cases limb <;>
        simp [schema, productSchema, productRoles]
  | root source lane =>
      cases source with
      | memory =>
          fin_cases lane <;> simp [schema, rootSchema, rootSources]
      | precommit role =>
          cases role <;> fin_cases lane <;>
            simp [schema, rootSchema, rootSources]
      | seen role =>
          cases role <;> fin_cases lane <;>
            simp [schema, rootSchema, rootSources]

def FieldTag.bitOffset (tag : FieldTag) : Nat :=
  TaggedBitSlices.offsetAt FieldTag.bitWidth schema (schema.idxOf tag)

theorem encode_slice (value : Value) (tag : FieldTag) :
    ((encode value).drop tag.bitOffset).take tag.bitWidth =
      encodeFields value tag := by
  have bounded := List.idxOf_lt_length_of_mem tag.mem_schema
  have sliced := TaggedBitSlices.slice_flatten_at
    (encodeFields value) FieldTag.bitWidth (encodeFields_length value)
    schema (schema.idxOf tag) bounded
  simpa [encode, encodeFor, TaggedBitSlices.flatten, FieldTag.bitOffset] using
    sliced

private theorem encodeFor_equal_at_member
    {headers : ChainHeaders Digest.Value}
    {left right : Value}
    (leftCanonical : left.Canonical headers)
    (rightCanonical : right.Canonical headers)
    {tags : List FieldTag}
    (equal : encodeFor tags left = encodeFor tags right)
    {tag : FieldTag} (member : tag ∈ tags) :
    left.fieldValue tag = right.fieldValue tag := by
  induction tags with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with tagEqual | tailMember
      · subst tag
        have headEqual := congrArg (List.take head.bitWidth) equal
        have wordEqual :
            encodeFields left head = encodeFields right head := by
          simpa [encodeFor, encodeFields_length] using headEqual
        exact
          Nightstream.Implementation.Nebula.WasmStateCodec.encodeWord_injective_of_bound
            (left.fieldValue_lt_width leftCanonical head)
            (right.fieldValue_lt_width rightCanonical head)
            wordEqual
      · have tailEqual := congrArg (List.drop head.bitWidth) equal
        have exactTail : encodeFor tail left = encodeFor tail right := by
          simpa [encodeFor, encodeFields_length] using tailEqual
        exact inductionHypothesis exactTail tailMember

/-- One canonical recursive carry has one 3,433-bit encoding. -/
theorem encode_injective_on_canonical
    {headers : ChainHeaders Digest.Value}
    {left right : Value}
    (leftCanonical : left.Canonical headers)
    (rightCanonical : right.Canonical headers)
    (equal : encode left = encode right) :
    left = right := by
  apply Value.fieldValue_injective
  funext tag
  exact encodeFor_equal_at_member leftCanonical rightCanonical
    equal tag.mem_schema

end Nightstream.Implementation.Nebula.MemoryCarryCodec
