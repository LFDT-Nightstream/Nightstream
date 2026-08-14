import Nightstream.Implementation.Nebula.Memory.Claim.PoseidonBinding
import Nightstream.Implementation.Nebula.Production.Memory.BatchPoseidonBinding
import Nightstream.Implementation.Nebula.Application.Wasm.ResultCodec

/-!
Contract: mixed field-native carrier for one successor memory-suffix batch.

Each checked step carries 116 bounded counter bits and 76 native Goldilocks
coordinates. Challenge, product, and root coordinates are not expanded into
64 Boolean coordinates. The complete ordered batch image is injective on
canonical claims.

Does not own generated columns, range-check rows, state hashing, NIFS
verification, external bytes, or a selected production candidate.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionMemorySuffixCarrier

open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

abbrev Claim := MemoryClaimCodec.Claim
abbrev Batch := ProductionMemoryBatchPoseidonBinding.Batch

def nativeSchema : List FieldTag :=
  challengeSchema ++ productSchema ++ rootSchema

theorem counterSchema_width_exact :
    (counterSchema.map FieldTag.bitWidth).sum = 116 := by
  decide

theorem nativeSchema_length : nativeSchema.length = 76 := by
  decide

def counterBits (claim : Claim) : List Nat :=
  MemoryClaimCodec.encodeFor counterSchema claim

theorem counterBits_length (claim : Claim) :
    (counterBits claim).length = 116 := by
  rw [counterBits, MemoryClaimCodec.encodeFor_length,
    counterSchema_width_exact]

theorem counterBits_binary
    (claim : Claim) (digit : Nat) (member : digit ∈ counterBits claim) :
    digit < 2 := by
  simp only [counterBits, MemoryClaimCodec.encodeFor,
    List.mem_flatMap] at member
  rcases member with ⟨tag, _tagMember, digitMember⟩
  exact MemoryClaimCodec.encodeFields_binary claim tag digit digitMember

/-- Native conversion is total. Canonicality later proves that the modulo
operation is the identity. -/
def nativeField (claim : Claim) (tag : FieldTag) : F :=
  ⟨claim.fieldValue tag % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

def nativeFields (claim : Claim) : List F :=
  nativeSchema.map (nativeField claim)

theorem nativeFields_length (claim : Claim) :
    (nativeFields claim).length = 76 := by
  simp [nativeFields, nativeSchema_length]

private theorem mapped_value_eq_at
    {Alpha Beta : Type} {tags : List Alpha} {left right : Alpha -> Beta}
    (equal : tags.map left = tags.map right)
    {tag : Alpha} (member : tag ∈ tags) : left tag = right tag := by
  induction tags with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq] at equal
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact equal.1
      · exact inductionHypothesis equal.2 tailMember

private theorem fieldValue_lt_modulus
    {claim : Claim} (canonical : claim.Canonical) (tag : FieldTag) :
    claim.fieldValue tag < goldilocksModulus := by
  apply MemoryClaimPoseidonBinding.frame_fields_canonical canonical
  rw [MemoryClaimPoseidonBinding.frame, List.mem_append]
  exact Or.inr (List.mem_map.mpr ⟨tag, tag.mem_schema, rfl⟩)

private theorem counter_value_eq_of_bits
    {left right : Claim}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : counterBits left = counterBits right)
    {tag : FieldTag} (member : tag ∈ counterSchema) :
    left.fieldValue tag = right.fieldValue tag := by
  exact MemoryClaimCodec.encodeFor_equal_at_member
    leftCanonical rightCanonical equal member

private theorem native_value_eq_of_fields
    {left right : Claim}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : nativeFields left = nativeFields right)
    {tag : FieldTag} (member : tag ∈ nativeSchema) :
    left.fieldValue tag = right.fieldValue tag := by
  have fieldEqual : nativeField left tag = nativeField right tag :=
    mapped_value_eq_at equal member
  have valueEqual := congrArg Fin.val fieldEqual
  have leftBound := fieldValue_lt_modulus leftCanonical tag
  have rightBound := fieldValue_lt_modulus rightCanonical tag
  simpa [nativeField, Nat.mod_eq_of_lt leftBound,
    Nat.mod_eq_of_lt rightBound] using valueEqual

def stepImage (claim : Claim) : List Nat × List F :=
  (counterBits claim, nativeFields claim)

/-- The mixed representation retains the complete typed suffix. -/
theorem stepImage_injective_on_canonical
    {left right : Claim}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : stepImage left = stepImage right) : left = right := by
  apply Claim.fieldValue_injective
  funext tag
  by_cases counter : tag ∈ counterSchema
  · exact counter_value_eq_of_bits leftCanonical rightCanonical
      (congrArg Prod.fst equal) counter
  · have native : tag ∈ nativeSchema := by
      have all : tag ∈ counterSchema ++ nativeSchema := by
        simpa [MemoryClaimCodec.schema, nativeSchema, List.append_assoc]
          using tag.mem_schema
      exact (List.mem_append.mp all).resolve_left counter
    exact native_value_eq_of_fields leftCanonical rightCanonical
      (congrArg Prod.snd equal) native

def counterBlocks {candidate : Id} (batch : Batch candidate) :
    List (List Nat) :=
  batch.suffixes.map counterBits

def nativeBlocks {candidate : Id} (batch : Batch candidate) :
    List (List F) :=
  batch.suffixes.map nativeFields

theorem counterBlocks_lengths
    {candidate : Id} (batch : Batch candidate) :
    (counterBlocks batch).map List.length =
      List.replicate (checkedStepsPerFreshClaim candidate) 116 := by
  apply List.eq_replicate_iff.mpr
  constructor
  · simp [counterBlocks, batch.length_exact]
  · intro width member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_map.mp blockMember with ⟨claim, _claimMember, rfl⟩
    exact counterBits_length claim

theorem nativeBlocks_lengths
    {candidate : Id} (batch : Batch candidate) :
    (nativeBlocks batch).map List.length =
      List.replicate (checkedStepsPerFreshClaim candidate) 76 := by
  apply List.eq_replicate_iff.mpr
  constructor
  · simp [nativeBlocks, batch.length_exact]
  · intro width member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_map.mp blockMember with ⟨claim, _claimMember, rfl⟩
    exact nativeFields_length claim

def batchCounterBits {candidate : Id} (batch : Batch candidate) : List Nat :=
  (counterBlocks batch).flatten

def batchNativeFields {candidate : Id} (batch : Batch candidate) : List F :=
  (nativeBlocks batch).flatten

theorem batchCounterBits_length
    {candidate : Id} (batch : Batch candidate) :
    (batchCounterBits batch).length =
      checkedStepsPerFreshClaim candidate * 116 := by
  rw [batchCounterBits, List.length_flatten, counterBlocks_lengths]
  simp

theorem batchNativeFields_length
    {candidate : Id} (batch : Batch candidate) :
    (batchNativeFields batch).length =
      checkedStepsPerFreshClaim candidate * 76 := by
  rw [batchNativeFields, List.length_flatten, nativeBlocks_lengths]
  simp

def batchImage {candidate : Id} (batch : Batch candidate) :
    List Nat × List F :=
  (batchCounterBits batch, batchNativeFields batch)

private theorem claim_lists_eq_of_block_maps
    {left right : List Claim}
    (leftCanonical : ∀ claim, claim ∈ left ->
      MemoryClaimCodec.Claim.Canonical claim)
    (rightCanonical : ∀ claim, claim ∈ right ->
      MemoryClaimCodec.Claim.Canonical claim)
    (counterEqual : left.map counterBits = right.map counterBits)
    (nativeEqual : left.map nativeFields = right.map nativeFields) :
    left = right := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons head tail => simp at counterEqual
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp at counterEqual
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at counterEqual
          simp only [List.map_cons, List.cons.injEq] at nativeEqual
          have headEqual : leftHead = rightHead :=
            stepImage_injective_on_canonical
              (leftCanonical leftHead (by simp))
              (rightCanonical rightHead (by simp))
              (Prod.ext counterEqual.1 nativeEqual.1)
          have tailEqual : leftTail = rightTail :=
            inductionHypothesis
              (fun claim member => leftCanonical claim (by simp [member]))
              (fun claim member => rightCanonical claim (by simp [member]))
              counterEqual.2 nativeEqual.2
          rw [headEqual, tailEqual]

/-- Equal complete mixed-coordinate batch images recover the exact ordered
batch. No digest assumption is used here. -/
theorem batchImage_injective_on_canonical
    {candidate : Id} {left right : Batch candidate}
    (leftCanonical : ∀ claim, claim ∈ left.suffixes ->
      MemoryClaimCodec.Claim.Canonical claim)
    (rightCanonical : ∀ claim, claim ∈ right.suffixes ->
      MemoryClaimCodec.Claim.Canonical claim)
    (equal : batchImage left = batchImage right) : left = right := by
  have counters : counterBlocks left = counterBlocks right :=
    WasmResultCodec.flatten_injective_of_lengths
      (counterBlocks_lengths left) (counterBlocks_lengths right)
      (congrArg Prod.fst equal)
  have fields : nativeBlocks left = nativeBlocks right :=
    WasmResultCodec.flatten_injective_of_lengths
      (nativeBlocks_lengths left) (nativeBlocks_lengths right)
      (congrArg Prod.snd equal)
  apply SuffixBatch.ext
  exact claim_lists_eq_of_block_maps leftCanonical rightCanonical
    counters fields

theorem batch_coordinate_count
    {candidate : Id} (batch : Batch candidate) :
    (batchImage batch).1.length + (batchImage batch).2.length =
      checkedStepsPerFreshClaim candidate * 192 := by
  rw [batchImage, batchCounterBits_length, batchNativeFields_length]
  omega

end Nightstream.Implementation.Nebula.ProductionMemorySuffixCarrier
