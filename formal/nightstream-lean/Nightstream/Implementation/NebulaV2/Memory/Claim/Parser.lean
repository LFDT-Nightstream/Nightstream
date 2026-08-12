import Nightstream.Implementation.NebulaV2.Core.FixedBits
import Nightstream.Implementation.NebulaV2.Memory.Claim.CounterRows
import Nightstream.Implementation.NebulaV2.Memory.Claim.FieldRows

/-!
Contract: fail-closed parser from one exact 4,980-bit V2 memory-claim block
to the concrete typed claim suffix.

Assurance tier: implementation model.

Owns safe counter and field slicing, strict `step_index < 1088` rejection,
canonical Goldilocks rejection for all 76 field limbs, and construction of
the exact two-repetition challenge/product/root value.

Does not own byte-container framing, public-column placement, the enclosing
full CCS claim, or Rust conformance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryClaimParser

open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

abbrev Block := FixedBits.Word stepPublicBits

def counterWord (block : Block) (counter : Counter) :
    FixedBits.Word counter.width :=
  FixedBits.slice block counter.bitOffset counter.width (by
    cases counter <;>
      norm_num [Counter.bitOffset, Counter.width, stepPublicBits,
        stepCounterBits, stepChallengeBits, stepProductBits, stepRootBits,
        segmentIndexBits, stepIndexBits, MemoryWireGeometry.timestampBits,
        stepActiveAccessCountBits, challengeBaseFieldLimbs,
        productStateBaseFieldLimbs, rootsBaseFieldLimbs, baseFieldBitCount,
        repetitionCount, challengeElementsPerRepetition,
        extensionLimbCount, productsPerRepetition, digestLimbCount])

def counterValue (block : Block) (counter : Counter) : Nat :=
  FixedBits.decode (counterWord block counter)

theorem counterValue_lt_width (block : Block) (counter : Counter) :
    counterValue block counter < 2 ^ counter.width :=
  FixedBits.decode_lt _

private theorem slot_position_lt
    (slot : MemoryClaimFieldRows.Slot) :
    slot.position < MemoryClaimFieldRows.Slot.all.length :=
  List.idxOf_lt_length_of_mem slot.mem_all

theorem field_slice_fits (slot : MemoryClaimFieldRows.Slot) :
    slot.bitOffset + CanonicalFieldBits.bitCount ≤ stepPublicBits := by
  have positionBound := slot_position_lt slot
  rw [MemoryClaimFieldRows.Slot.all_length_exact] at positionBound
  rw [MemoryClaimFieldRows.Slot.bitOffset,
    MemoryClaimFieldRows.fieldBitStart_exact,
    MemoryWireGeometry.stepPublicBits_exact]
  norm_num [CanonicalFieldBits.bitCount]
  omega

def fieldWord (block : Block) (slot : MemoryClaimFieldRows.Slot) :
    CanonicalFieldBits.Word :=
  let sliced := FixedBits.slice block slot.bitOffset
    CanonicalFieldBits.bitCount (field_slice_fits slot)
  ⟨sliced.val, sliced.property⟩

def rawWords (block : Block) : MemoryClaimFieldRows.RawWords :=
  fun slot => fieldWord block slot

def fieldsCanonical (block : Block) : Bool :=
  MemoryClaimFieldRows.Slot.all.all fun slot =>
    decide (CanonicalFieldBits.decode (fieldWord block slot) <
      ShiftedTernary41V1.modulus)

theorem field_canonical_of_all
    {block : Block} (allCanonical : fieldsCanonical block = true)
    (slot : MemoryClaimFieldRows.Slot) :
    CanonicalFieldBits.Canonical (fieldWord block slot) := by
  have every := List.all_eq_true.mp allCanonical slot slot.mem_all
  simpa [CanonicalFieldBits.Canonical] using of_decide_eq_true every

def decodedField (block : Block) (slot : MemoryClaimFieldRows.Slot) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  (FieldCodec.nativeDecode (fieldWord block slot)).getD
    CanonicalFieldBits.zero

theorem nativeDecode_field
    {block : Block} (allCanonical : fieldsCanonical block = true)
    (slot : MemoryClaimFieldRows.Slot) :
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
    (slot0 slot1 : MemoryClaimFieldRows.Slot) : K :=
  ⟨toF (decodedField block slot0), toF (decodedField block slot1)⟩

def decodedChallenges (block : Block) : Challenges K :=
  fun repetition =>
    { gamma1 := decodedK block
        (.challenge repetition 0 0) (.challenge repetition 0 1)
      gamma2 := decodedK block
        (.challenge repetition 1 0) (.challenge repetition 1 1) }

def decodedProduct (block : Block) (side repetition : Fin 2) : Four K :=
  { initialSnapshot := decodedK block
      (.product side repetition .initialSnapshot 0)
      (.product side repetition .initialSnapshot 1)
    writes := decodedK block
      (.product side repetition .writes 0)
      (.product side repetition .writes 1)
    reads := decodedK block
      (.product side repetition .reads 0)
      (.product side repetition .reads 1)
    finalSnapshot := decodedK block
      (.product side repetition .finalSnapshot 0)
      (.product side repetition .finalSnapshot 1) }

def decodedProducts (block : Block) (side : Fin 2) : State K :=
  fun repetition => decodedProduct block side repetition

def decodedDigest (block : Block) (stage : RootStage)
    (role : RootRole) : Digest.Value where
  lanes := fun lane => decodedField block (.root stage role lane)

def decodedRoots (block : Block) (stage : RootStage) : Roots Digest.Value :=
  { operations := decodedDigest block stage .operations
    initialSnapshot := decodedDigest block stage .initialSnapshot
    finalSnapshot := decodedDigest block stage .finalSnapshot }

def decodedClaim (block : Block)
    (stepBound : counterValue block .stepIndex < Lifecycle.claimsPerSegment) :
    Claim :=
  { segmentIndex := counterValue block .segmentIndex
    stepIndex := ⟨counterValue block .stepIndex, stepBound⟩
    timestampIn := counterValue block .timestampIn
    timestampOut := counterValue block .timestampOut
    segmentStartTimestamp := counterValue block .segmentStartTimestamp
    segmentEndTimestamp := counterValue block .segmentEndTimestamp
    activeAccessCount := counterValue block .activeAccessCount
    challenge := decodedChallenges block
    dPre := decodedRoots block .precommit
    dSeenBefore := decodedRoots block .seenBefore
    dSeenAfter := decodedRoots block .seenAfter
    productsBefore := decodedProducts block 0
    productsAfter := decodedProducts block 1 }

/-- The fail-closed logical-bit parser. -/
def parse (block : Block) : Option Claim :=
  if stepBound : counterValue block .stepIndex < Lifecycle.claimsPerSegment then
    if _allCanonical : fieldsCanonical block = true then
      some (decodedClaim block stepBound)
    else none
  else none

theorem parse_some_bound_and_fields
    {block : Block} {claim : Claim} (accepted : parse block = some claim) :
    ∃ stepBound : counterValue block .stepIndex < Lifecycle.claimsPerSegment,
      fieldsCanonical block = true ∧ claim = decodedClaim block stepBound := by
  unfold parse at accepted
  split at accepted
  next stepBound =>
    split at accepted
    next allCanonical =>
      exact ⟨stepBound, allCanonical, Option.some.inj accepted.symm⟩
    next => simp at accepted
  next => simp at accepted

set_option maxHeartbeats 2000000 in
theorem decodedClaim_canonicalValue
    (block : Block)
    (stepBound : counterValue block .stepIndex < Lifecycle.claimsPerSegment)
    (slot : MemoryClaimFieldRows.Slot) :
    slot.canonicalValue (decodedClaim block stepBound) =
      decodedField block slot := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;>
        apply Subtype.ext <;> rfl
  | product side repetition role limb =>
      fin_cases side <;> cases role <;> fin_cases limb <;>
        apply Subtype.ext <;> rfl
  | root stage role lane =>
      cases stage <;> cases role <;> rfl

/-- Successful parsing constructs the exact native field interpretation used
by the schema-level R1CS bridge. No `NativeParses` assumption remains. -/
theorem parse_native_parses
    {block : Block} {claim : Claim} (accepted : parse block = some claim) :
    MemoryClaimFieldRows.NativeParses (rawWords block) claim := by
  rcases parse_some_bound_and_fields accepted with
    ⟨stepBound, allCanonical, claimEqual⟩
  subst claim
  intro slot
  rw [rawWords, decodedClaim_canonicalValue]
  exact nativeDecode_field allCanonical slot

/-- Successful parsing also proves every narrow counter bound in the typed
claim. -/
theorem parse_claim_canonical
    {block : Block} {claim : Claim} (accepted : parse block = some claim) :
    claim.Canonical := by
  rcases parse_some_bound_and_fields accepted with
    ⟨stepBound, allCanonical, claimEqual⟩
  subst claim
  constructor
  · exact counterValue_lt_width block .segmentIndex
  · exact counterValue_lt_width block .timestampIn
  · exact counterValue_lt_width block .timestampOut
  · exact counterValue_lt_width block .segmentStartTimestamp
  · exact counterValue_lt_width block .segmentEndTimestamp
  · exact counterValue_lt_width block .activeAccessCount

/-- A noncanonical `q` word in any field slot makes successful parsing
impossible. -/
theorem rejects_modulus_alias
    {block : Block} (slot : MemoryClaimFieldRows.Slot)
    (aliasEq : fieldWord block slot = CanonicalFieldBits.modulusWord) :
    parse block = none := by
  apply Option.eq_none_iff_forall_not_mem.mpr
  intro claim accepted
  have parsed := parse_native_parses (show parse block = some claim from accepted)
  have decoded := parsed slot
  rw [rawWords, aliasEq, FieldCodec.rejects_zero_modulus_alias.2] at decoded
  simp at decoded

/-- Canonical typed claims embed as exact parser inputs. -/
def blockOfClaim (claim : Claim) (canonical : claim.Canonical) : Block :=
  ⟨MemoryClaimCodec.encode claim,
    MemoryClaimCodec.encode_length claim,
    fun digit member => MemoryClaimCodec.encode_binary claim digit member⟩

theorem counterWord_blockOfClaim
    (claim : Claim) (canonical : claim.Canonical) (counter : Counter) :
    (counterWord (blockOfClaim claim canonical) counter).val =
      MemoryClaimCodec.encodeFields claim counter.tag := by
  change
    ((MemoryClaimCodec.encode claim).drop counter.bitOffset).take
        counter.width = MemoryClaimCodec.encodeFields claim counter.tag
  rw [counter.bitOffset_eq_tag, counter.width_eq_tag]
  exact MemoryClaimCodec.encode_slice claim counter.tag

theorem counterValue_blockOfClaim
    (claim : Claim) (canonical : claim.Canonical) (counter : Counter) :
    counterValue (blockOfClaim claim canonical) counter =
      counter.claimValue claim := by
  rw [counterValue, FixedBits.decode, counterWord_blockOfClaim]
  change Nat.ofDigits 2
      (WasmStateCodec.encodeWord counter.tag.bitWidth
        (claim.fieldValue counter.tag)) = counter.claimValue claim
  rw [WasmStateCodec.ofDigits_encodeWord_of_bound
    (claim.fieldValue_lt_width canonical counter.tag)]
  exact counter.claimValue_eq_tag claim |>.symm

theorem fieldWord_blockOfClaim
    (claim : Claim) (canonical : claim.Canonical)
    (slot : MemoryClaimFieldRows.Slot) :
    fieldWord (blockOfClaim claim canonical) slot =
      CanonicalFieldBits.encode (slot.canonicalValue claim) := by
  apply Subtype.ext
  change
    ((MemoryClaimCodec.encode claim).drop slot.bitOffset).take
        CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (slot.canonicalValue claim)).val
  rw [slot.bitOffset_eq_tag]
  have slicedAtTag :
      ((MemoryClaimCodec.encode claim).drop slot.tag.bitOffset).take
          CanonicalFieldBits.bitCount =
        MemoryClaimCodec.encodeFields claim slot.tag := by
    simpa only [slot.tag_width] using
      MemoryClaimCodec.encode_slice claim slot.tag
  rw [slicedAtTag]
  have capacityBound : (slot.canonicalValue claim).val <
      2 ^ CanonicalFieldBits.bitCount :=
    (slot.canonicalValue claim).property.trans
      CanonicalFieldBits.modulus_lt_capacity
  unfold MemoryClaimCodec.encodeFields MemoryClaimCodec.encodeWord
  rw [slot.tag_width, ← slot.canonicalValue_val claim]
  simp [WasmStateCodec.encodeWord, CanonicalFieldBits.encode,
    Nat.mod_eq_of_lt capacityBound]

theorem fieldsCanonical_blockOfClaim
    (claim : Claim) (canonical : claim.Canonical) :
    fieldsCanonical (blockOfClaim claim canonical) = true := by
  rw [fieldsCanonical, List.all_eq_true]
  intro slot member
  apply decide_eq_true
  rw [fieldWord_blockOfClaim claim canonical slot,
    CanonicalFieldBits.decode_encode]
  exact (slot.canonicalValue claim).property

theorem decodedField_blockOfClaim
    (claim : Claim) (canonical : claim.Canonical)
    (slot : MemoryClaimFieldRows.Slot) :
    decodedField (blockOfClaim claim canonical) slot =
      slot.canonicalValue claim := by
  have decoded :
      FieldCodec.nativeDecode
          (fieldWord (blockOfClaim claim canonical) slot) =
        some (slot.canonicalValue claim) := by
    apply (FieldCodec.nativeDecode_some_iff
      (fieldWord (blockOfClaim claim canonical) slot)
      (slot.canonicalValue claim)).2
    rw [fieldWord_blockOfClaim claim canonical slot]
    exact ⟨CanonicalFieldBits.encode_is_canonical _,
      CanonicalFieldBits.decode_encode _ |>.symm⟩
  unfold decodedField
  rw [decoded]
  rfl

private theorem decodedClaim_fieldValue_at_slot
    (claim : Claim) (canonical : claim.Canonical)
    (stepBound :
      counterValue (blockOfClaim claim canonical) .stepIndex <
        Lifecycle.claimsPerSegment)
    (slot : MemoryClaimFieldRows.Slot) :
    (decodedClaim (blockOfClaim claim canonical) stepBound).fieldValue
        slot.tag = claim.fieldValue slot.tag := by
  calc
    (decodedClaim (blockOfClaim claim canonical) stepBound).fieldValue
        slot.tag =
        (slot.canonicalValue
          (decodedClaim (blockOfClaim claim canonical) stepBound)).val :=
      (slot.canonicalValue_val _).symm
    _ = (decodedField (blockOfClaim claim canonical) slot).val :=
      congrArg Subtype.val
        (decodedClaim_canonicalValue _ stepBound slot)
    _ = (slot.canonicalValue claim).val :=
      congrArg Subtype.val (decodedField_blockOfClaim claim canonical slot)
    _ = claim.fieldValue slot.tag := slot.canonicalValue_val claim

theorem decodedClaim_blockOfClaim
    (claim : Claim) (canonical : claim.Canonical)
    (stepBound :
      counterValue (blockOfClaim claim canonical) .stepIndex <
        Lifecycle.claimsPerSegment) :
    decodedClaim (blockOfClaim claim canonical) stepBound = claim := by
  apply Claim.fieldValue_injective
  funext tag
  cases tag with
  | segmentIndex =>
      exact counterValue_blockOfClaim claim canonical .segmentIndex
  | stepIndex =>
      exact counterValue_blockOfClaim claim canonical .stepIndex
  | timestampIn =>
      exact counterValue_blockOfClaim claim canonical .timestampIn
  | timestampOut =>
      exact counterValue_blockOfClaim claim canonical .timestampOut
  | segmentStartTimestamp =>
      exact counterValue_blockOfClaim claim canonical .segmentStartTimestamp
  | segmentEndTimestamp =>
      exact counterValue_blockOfClaim claim canonical .segmentEndTimestamp
  | activeAccessCount =>
      exact counterValue_blockOfClaim claim canonical .activeAccessCount
  | challenge repetition coordinate limb =>
      exact decodedClaim_fieldValue_at_slot claim canonical stepBound
        (.challenge repetition coordinate limb)
  | product side repetition role limb =>
      exact decodedClaim_fieldValue_at_slot claim canonical stepBound
        (.product side repetition role limb)
  | root stage role lane =>
      exact decodedClaim_fieldValue_at_slot claim canonical stepBound
        (.root stage role lane)

/-- Parser completeness: every canonical typed claim round-trips through its
exact 4,980-bit encoding. -/
theorem parse_blockOfClaim
    (claim : Claim) (canonical : claim.Canonical) :
    parse (blockOfClaim claim canonical) = some claim := by
  have stepBound :
      counterValue (blockOfClaim claim canonical) .stepIndex <
        Lifecycle.claimsPerSegment := by
    rw [counterValue_blockOfClaim claim canonical .stepIndex]
    exact claim.stepIndex.isLt
  have fieldsExact := fieldsCanonical_blockOfClaim claim canonical
  unfold parse
  rw [dif_pos stepBound, dif_pos fieldsExact]
  rw [decodedClaim_blockOfClaim claim canonical stepBound]

end Nightstream.Implementation.NebulaV2.MemoryClaimParser
