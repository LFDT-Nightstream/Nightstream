import Nightstream.Implementation.NebulaV2.Application.Wasm.StateCodec
import Nightstream.Implementation.NebulaV2.Core.TaggedBitSlices
import Nightstream.Protocol.NebulaV2.Digest
import Nightstream.Protocol.NebulaV2.FPrime
import Nightstream.Protocol.NebulaV2.MemoryWireGeometry
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: canonical 4,980-bit codec for the V2 fresh-claim memory block.

Assurance tier: implementation model.

Owns the exact field tags, field order, fixed little-endian widths, concrete
SuperNeo extension-field coefficient order, canonical counter bounds, total
length, and encoding injectivity.

Does not own byte-container framing, public-column placement, generated
Boolean rows, native Rust parsing, or recursive verifier refinement.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryClaimCodec

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.IdealFingerprint
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

abbrev Claim :=
  ClaimSuffix Digest.Value (Challenges K) (State K)

inductive ProductRole where
  | initialSnapshot
  | writes
  | reads
  | finalSnapshot
deriving DecidableEq, Fintype, Repr

inductive RootStage where
  | precommit
  | seenBefore
  | seenAfter
deriving DecidableEq, Fintype, Repr

inductive RootRole where
  | operations
  | initialSnapshot
  | finalSnapshot
deriving DecidableEq, Fintype, Repr

/-- A tag names one authority-bearing integer word before bit flattening.
Challenge coordinate zero is `gamma1`; coordinate one is `gamma2`. Extension
limb zero is `c0`; limb one is `c1`. -/
inductive FieldTag where
  | segmentIndex
  | stepIndex
  | timestampIn
  | timestampOut
  | segmentStartTimestamp
  | segmentEndTimestamp
  | activeAccessCount
  | challenge (repetition coordinate limb : Fin 2)
  | product (side repetition : Fin 2) (role : ProductRole) (limb : Fin 2)
  | root (stage : RootStage) (role : RootRole) (lane : Fin 4)
deriving DecidableEq, Repr

def productRoles : List ProductRole :=
  [.reads, .writes, .initialSnapshot, .finalSnapshot]

def rootStages : List RootStage :=
  [.precommit, .seenBefore, .seenAfter]

def rootRoles : List RootRole :=
  [.operations, .initialSnapshot, .finalSnapshot]

def counterSchema : List FieldTag :=
  [.segmentIndex, .stepIndex, .timestampIn, .timestampOut,
    .segmentStartTimestamp, .segmentEndTimestamp, .activeAccessCount]

def challengeSchema : List FieldTag :=
  (List.ofFn fun repetition : Fin 2 =>
    (List.ofFn fun coordinate : Fin 2 =>
      List.ofFn fun limb : Fin 2 =>
        FieldTag.challenge repetition coordinate limb).flatten).flatten

def productSchema : List FieldTag :=
  (List.ofFn fun side : Fin 2 =>
    (List.ofFn fun repetition : Fin 2 =>
      (productRoles.map fun role =>
        List.ofFn fun limb : Fin 2 =>
          FieldTag.product side repetition role limb).flatten).flatten).flatten

def rootSchema : List FieldTag :=
  (rootStages.map fun stage =>
    (rootRoles.map fun role =>
      List.ofFn fun lane : Fin 4 =>
        FieldTag.root stage role lane).flatten).flatten

/-- Exact public-memory word order from `SPEC.md`. -/
def schema : List FieldTag :=
  counterSchema ++ challengeSchema ++ productSchema ++ rootSchema

theorem schema_nodup : schema.Nodup := by decide

def FieldTag.bitWidth : FieldTag → Nat
  | .segmentIndex => segmentIndexBits
  | .stepIndex => stepIndexBits
  | .timestampIn => MemoryWireGeometry.timestampBits
  | .timestampOut => MemoryWireGeometry.timestampBits
  | .segmentStartTimestamp => MemoryWireGeometry.timestampBits
  | .segmentEndTimestamp => MemoryWireGeometry.timestampBits
  | .activeAccessCount => stepActiveAccessCountBits
  | .challenge _ _ _ => baseFieldBitCount
  | .product _ _ _ _ => baseFieldBitCount
  | .root _ _ _ => baseFieldBitCount

def kLimbValue (value : K) : Fin 2 → Nat
  | 0 => value.c0.val
  | _ => value.c1.val

def challengeValue (challenge : ChallengePair K)
    (coordinate limb : Fin 2) : Nat :=
  if coordinate = 0 then kLimbValue challenge.gamma1 limb
  else kLimbValue challenge.gamma2 limb

def productValue (products : Four K) (role : ProductRole)
    (limb : Fin 2) : Nat :=
  match role with
  | .initialSnapshot => kLimbValue products.initialSnapshot limb
  | .writes => kLimbValue products.writes limb
  | .reads => kLimbValue products.reads limb
  | .finalSnapshot => kLimbValue products.finalSnapshot limb

def rootValue (roots : Roots Digest.Value) (role : RootRole)
    (lane : Fin 4) : Nat :=
  match role with
  | .operations => (roots.operations.lanes lane).val
  | .initialSnapshot => (roots.initialSnapshot.lanes lane).val
  | .finalSnapshot => (roots.finalSnapshot.lanes lane).val

def Claim.fieldValue (claim : Claim) : FieldTag → Nat
  | .segmentIndex => claim.segmentIndex
  | .stepIndex => claim.stepIndex.val
  | .timestampIn => claim.timestampIn
  | .timestampOut => claim.timestampOut
  | .segmentStartTimestamp => claim.segmentStartTimestamp
  | .segmentEndTimestamp => claim.segmentEndTimestamp
  | .activeAccessCount => claim.activeAccessCount
  | .challenge repetition coordinate limb =>
      challengeValue (claim.challenge repetition) coordinate limb
  | .product side repetition role limb =>
      productValue
        (if side = 0 then claim.productsBefore repetition
         else claim.productsAfter repetition)
        role limb
  | .root stage role lane =>
      rootValue
        (match stage with
         | .precommit => claim.dPre
         | .seenBefore => claim.dSeenBefore
         | .seenAfter => claim.dSeenAfter)
        role lane

/-- Only the bounded integer fields need an additional canonical predicate.
Extension-field and digest coefficients are canonical by their concrete
`Fin q` and `CanonicalGoldilocks` types. -/
structure Claim.Canonical (claim : Claim) : Prop where
  segmentIndex : claim.segmentIndex < 2 ^ segmentIndexBits
  timestampIn : claim.timestampIn < 2 ^ MemoryWireGeometry.timestampBits
  timestampOut : claim.timestampOut < 2 ^ MemoryWireGeometry.timestampBits
  segmentStartTimestamp :
    claim.segmentStartTimestamp < 2 ^ MemoryWireGeometry.timestampBits
  segmentEndTimestamp :
    claim.segmentEndTimestamp < 2 ^ MemoryWireGeometry.timestampBits
  activeAccessCount :
    claim.activeAccessCount < 2 ^ stepActiveAccessCountBits

private theorem fieldValue_lt_wordCapacity (value : F) :
    value.val < 2 ^ baseFieldBitCount := by
  exact value.isLt.trans (by
    norm_num [Nightstream.SuperNeo.Concrete.goldilocksModulus,
      baseFieldBitCount])

private theorem digestValue_lt_wordCapacity
    (value : ShiftedTernary41V1.CanonicalGoldilocks) :
    value.val < 2 ^ baseFieldBitCount := by
  exact value.property.trans (by
    norm_num [Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.modulus,
      baseFieldBitCount])

theorem Claim.fieldValue_lt_width
    {claim : Claim} (canonical : claim.Canonical) (tag : FieldTag) :
    claim.fieldValue tag < 2 ^ tag.bitWidth := by
  cases tag with
  | segmentIndex => exact canonical.segmentIndex
  | stepIndex =>
      exact claim.stepIndex.isLt.trans (by
        show Lifecycle.claimsPerSegment < 2 ^ stepIndexBits
        decide)
  | timestampIn => exact canonical.timestampIn
  | timestampOut => exact canonical.timestampOut
  | segmentStartTimestamp => exact canonical.segmentStartTimestamp
  | segmentEndTimestamp => exact canonical.segmentEndTimestamp
  | activeAccessCount => exact canonical.activeAccessCount
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;>
        exact fieldValue_lt_wordCapacity _
  | product side repetition role limb =>
      fin_cases side <;> cases role <;> fin_cases limb <;>
        exact fieldValue_lt_wordCapacity _
  | root stage role lane =>
      cases stage <;> cases role <;>
        exact digestValue_lt_wordCapacity _

theorem k_eq_of_limb_values
    {left right : K}
    (equal : ∀ limb, kLimbValue left limb = kLimbValue right limb) :
    left = right := by
  cases left with
  | mk left0 left1 =>
    cases right with
    | mk right0 right1 =>
      have equal0 := equal (0 : Fin 2)
      have equal1 := equal (1 : Fin 2)
      simp only [kLimbValue] at equal0 equal1
      cases Fin.ext equal0
      cases Fin.ext equal1
      rfl

theorem challenge_eq_of_values
    {left right : ChallengePair K}
    (equal : ∀ coordinate limb,
      challengeValue left coordinate limb =
        challengeValue right coordinate limb) :
    left = right := by
  cases left with
  | mk leftGamma1 leftGamma2 =>
    cases right with
    | mk rightGamma1 rightGamma2 =>
      have gamma1 : leftGamma1 = rightGamma1 :=
        k_eq_of_limb_values (fun limb => by
          simpa [challengeValue] using equal (0 : Fin 2) limb)
      have gamma2 : leftGamma2 = rightGamma2 :=
        k_eq_of_limb_values (fun limb => by
          simpa [challengeValue] using equal (1 : Fin 2) limb)
      cases gamma1
      cases gamma2
      rfl

theorem product_eq_of_values
    {left right : Four K}
    (equal : ∀ role limb,
      productValue left role limb = productValue right role limb) :
    left = right := by
  apply Four.ext
  · exact k_eq_of_limb_values (fun limb => equal .initialSnapshot limb)
  · exact k_eq_of_limb_values (fun limb => equal .writes limb)
  · exact k_eq_of_limb_values (fun limb => equal .reads limb)
  · exact k_eq_of_limb_values (fun limb => equal .finalSnapshot limb)

theorem digest_eq_of_lane_values
    {left right : Digest.Value}
    (equal : ∀ lane, (left.lanes lane).val = (right.lanes lane).val) :
    left = right := by
  apply Digest.Value.ext
  funext lane
  exact Subtype.ext (equal lane)

theorem roots_eq_of_values
    {left right : Roots Digest.Value}
    (equal : ∀ role lane,
      rootValue left role lane = rootValue right role lane) :
    left = right := by
  apply Roots.ext
  · exact digest_eq_of_lane_values (fun lane => equal .operations lane)
  · exact digest_eq_of_lane_values (fun lane => equal .initialSnapshot lane)
  · exact digest_eq_of_lane_values (fun lane => equal .finalSnapshot lane)

/-- The complete tagged integer image retains every claim field. -/
theorem Claim.fieldValue_injective :
    Function.Injective Claim.fieldValue := by
  intro left right equal
  apply ClaimSuffix.ext
  · exact congrFun equal .segmentIndex
  · apply Fin.ext
    exact congrFun equal .stepIndex
  · exact congrFun equal .timestampIn
  · exact congrFun equal .timestampOut
  · exact congrFun equal .segmentStartTimestamp
  · exact congrFun equal .segmentEndTimestamp
  · exact congrFun equal .activeAccessCount
  · funext repetition
    apply challenge_eq_of_values
    intro coordinate limb
    exact congrFun equal (.challenge repetition coordinate limb)
  · apply roots_eq_of_values
    intro role lane
    exact congrFun equal (.root .precommit role lane)
  · apply roots_eq_of_values
    intro role lane
    exact congrFun equal (.root .seenBefore role lane)
  · apply roots_eq_of_values
    intro role lane
    exact congrFun equal (.root .seenAfter role lane)
  · funext repetition
    apply product_eq_of_values
    intro role limb
    exact congrFun equal (.product 0 repetition role limb)
  · funext repetition
    apply product_eq_of_values
    intro role limb
    exact congrFun equal (.product 1 repetition role limb)

def encodeWord :=
  Nightstream.Implementation.NebulaV2.WasmStateCodec.encodeWord

def encodeFields (claim : Claim) (tag : FieldTag) : List Nat :=
  encodeWord tag.bitWidth (claim.fieldValue tag)

theorem encodeFields_length (claim : Claim) (tag : FieldTag) :
    (encodeFields claim tag).length = tag.bitWidth :=
  Nightstream.Implementation.NebulaV2.WasmStateCodec.encodeWord_length _ _

theorem encodeFields_binary
    (claim : Claim) (tag : FieldTag) (digit : Nat)
    (member : digit ∈ encodeFields claim tag) :
    digit < 2 :=
  Nightstream.Implementation.NebulaV2.WasmStateCodec.encodeWord_binary
    _ _ _ member

def encodeFor (tags : List FieldTag) (claim : Claim) : List Nat :=
  tags.flatMap (encodeFields claim)

/-- Exact 4,980-bit public-memory block. -/
def encode (claim : Claim) : List Nat :=
  encodeFor schema claim

theorem encodeFor_length (tags : List FieldTag) (claim : Claim) :
    (encodeFor tags claim).length =
      (tags.map FieldTag.bitWidth).sum := by
  induction tags with
  | nil => rfl
  | cons tag rest inductionHypothesis =>
      simp [encodeFor, encodeFields_length]

set_option maxRecDepth 10000 in
theorem schema_width_exact :
    (schema.map FieldTag.bitWidth).sum = stepPublicBits := by
  decide

theorem encode_length (claim : Claim) :
    (encode claim).length = stepPublicBits := by
  rw [encode, encodeFor_length, schema_width_exact]

theorem encode_exact_length (claim : Claim) :
    (encode claim).length = 4980 := by
  rw [encode_length, stepPublicBits_exact]

theorem encode_binary (claim : Claim) (digit : Nat)
    (member : digit ∈ encode claim) :
    digit < 2 := by
  simp only [encode, encodeFor, List.mem_flatMap] at member
  obtain ⟨tag, _tagMember, digitMember⟩ := member
  exact encodeFields_binary claim tag digit digitMember

theorem FieldTag.mem_schema (tag : FieldTag) : tag ∈ schema := by
  cases tag with
  | segmentIndex => simp [schema, counterSchema]
  | stepIndex => simp [schema, counterSchema]
  | timestampIn => simp [schema, counterSchema]
  | timestampOut => simp [schema, counterSchema]
  | segmentStartTimestamp => simp [schema, counterSchema]
  | segmentEndTimestamp => simp [schema, counterSchema]
  | activeAccessCount => simp [schema, counterSchema]
  | challenge repetition coordinate limb =>
      fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
        simp [schema, challengeSchema]
  | product side repetition role limb =>
      fin_cases side <;> fin_cases repetition <;> cases role <;>
        fin_cases limb <;> simp [schema, productSchema, productRoles]
  | root stage role lane =>
      cases stage <;> cases role <;> fin_cases lane <;>
        simp [schema, rootSchema, rootStages, rootRoles]

/-- Exact offset of one tagged word in the flattened claim block. -/
def FieldTag.bitOffset (tag : FieldTag) : Nat :=
  TaggedBitSlices.offsetAt FieldTag.bitWidth schema (schema.idxOf tag)

/-- Slicing the flattened codec at the tag-owned offset recovers exactly that
tag's word. -/
theorem encode_slice (claim : Claim) (tag : FieldTag) :
    ((encode claim).drop tag.bitOffset).take tag.bitWidth =
      encodeFields claim tag := by
  have bounded := List.idxOf_lt_length_of_mem tag.mem_schema
  have sliced := TaggedBitSlices.slice_flatten_at
    (encodeFields claim) FieldTag.bitWidth (encodeFields_length claim)
    schema (schema.idxOf tag) bounded
  simpa [encode, encodeFor, TaggedBitSlices.flatten, FieldTag.bitOffset] using
    sliced

theorem encodeFor_equal_at_member
    {left right : Claim}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
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
          Nightstream.Implementation.NebulaV2.WasmStateCodec.encodeWord_injective_of_bound
            (left.fieldValue_lt_width leftCanonical head)
            (right.fieldValue_lt_width rightCanonical head)
            wordEqual
      · have tailEqual := congrArg (List.drop head.bitWidth) equal
        have exactTail : encodeFor tail left = encodeFor tail right := by
          simpa [encodeFor, encodeFields_length] using tailEqual
        exact inductionHypothesis exactTail tailMember

/-- No two canonical claims have the same 4,980-bit memory block. -/
theorem encode_injective_on_canonical
    {left right : Claim}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : encode left = encode right) :
    left = right := by
  apply Claim.fieldValue_injective
  funext tag
  exact encodeFor_equal_at_member leftCanonical rightCanonical
    equal tag.mem_schema

end Nightstream.Implementation.NebulaV2.MemoryClaimCodec
