import Nightstream.Implementation.Nebula.Memory.Claim.BoundCcsPublic
import Nightstream.Implementation.Nebula.Production.Memory.BatchPoseidonBinding

/-!
Contract: exact 540-coordinate CCS public carrier for one successor batch.

The carrier contains one affine coordinate, the 256-bit state digest, the
256-bit digest of the complete candidate-specific memory batch, and 27 zero
coordinates. The consumer recomputes the batch digest from all `E` suffixes.

Does not own state-digest semantics, generated rows, NIFS placement, Poseidon2
collision security, Rust conformance, candidate selection, or a verifier key.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic

open Nightstream.Implementation.Nebula.MemoryBoundCcsPublic
open Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionProfileCandidates

abbrev CanonicalDigest := MemoryBoundCcsPublic.CanonicalDigest
abbrev Batch := ProductionMemoryBatchPoseidonBinding.Batch

def memoryDigest {candidate : Id} (batch : Batch candidate) :
    CanonicalDigest :=
  fun lane => (canonicalDigest batch).lanes lane

def encode {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate) : List Nat :=
  [1] ++ digestBits stateDigest ++ digestBits (memoryDigest batch) ++
    List.replicate paddingBitCount 0

theorem encode_length
    {candidate : Id} (stateDigest : CanonicalDigest)
    (batch : Batch candidate) :
    (encode stateDigest batch).length = coordinateCount := by
  norm_num [encode, digestBits_length, digestBitCount,
    paddingBitCount, coordinateCount]

theorem encode_binary
    {candidate : Id} (stateDigest : CanonicalDigest)
    (batch : Batch candidate) (digit : Nat)
    (member : digit ∈ encode stateDigest batch) : digit < 2 := by
  simp only [encode, List.mem_append, List.mem_cons, List.not_mem_nil,
    or_false, List.mem_replicate] at member
  rcases member with ((one | stateMember) | memoryMember) | paddingMember
  · subst digit
    decide
  · exact digestBits_binary stateDigest digit stateMember
  · exact digestBits_binary (memoryDigest batch) digit memoryMember
  · exact paddingMember.2 ▸ (by decide)

def word {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate) :
    FixedBits.Word coordinateCount :=
  ⟨encode stateDigest batch, encode_length stateDigest batch,
    encode_binary stateDigest batch⟩

@[simp] theorem word_zero
    {candidate : Id} (stateDigest : CanonicalDigest)
    (batch : Batch candidate) :
    (word stateDigest batch).val.getD 0 0 = 1 := by
  rfl

/-- Consumer-side binding of a separately decoded batch to the fresh CCS
public coordinates. -/
def MemoryMatches
    {candidate : Id} (carrier : FixedBits.Word coordinateCount)
    (batch : Batch candidate) : Prop :=
  (carrier.val.drop 257).take 256 = digestBits (memoryDigest batch)

instance memoryMatchesDecidable
    {candidate : Id} (carrier : FixedBits.Word coordinateCount)
    (batch : Batch candidate) : Decidable (MemoryMatches carrier batch) := by
  unfold MemoryMatches
  infer_instance

/-- Consumer-side binding of one canonical four-lane state digest to the
state half of the fresh CCS public coordinates. This predicate does not
claim that the digest was computed from an authoritative recursive state. -/
def StateMatches
    (carrier : FixedBits.Word coordinateCount)
    (stateDigest : CanonicalDigest) : Prop :=
  (carrier.val.drop 1).take 256 = digestBits stateDigest

instance stateMatchesDecidable
    (carrier : FixedBits.Word coordinateCount)
    (stateDigest : CanonicalDigest) : Decidable (StateMatches carrier stateDigest) := by
  unfold StateMatches
  infer_instance

/-- Exact authority relation for all 540 coordinates. Unlike `MemoryMatches`,
this also fixes the affine coordinate, the complete canonical state digest,
and all padding coordinates. -/
def FullMatches
    {candidate : Id} (carrier : FixedBits.Word coordinateCount)
    (stateDigest : CanonicalDigest) (batch : Batch candidate) : Prop :=
  carrier.val = encode stateDigest batch

instance fullMatchesDecidable
    {candidate : Id} (carrier : FixedBits.Word coordinateCount)
    (stateDigest : CanonicalDigest) (batch : Batch candidate) :
    Decidable (FullMatches carrier stateDigest batch) := by
  unfold FullMatches
  infer_instance

theorem word_memoryMatches
    {candidate : Id} (stateDigest : CanonicalDigest)
    (batch : Batch candidate) :
    MemoryMatches (word stateDigest batch) batch := by
  simp [MemoryMatches, word, encode, digestBits_length, digestBitCount]

theorem word_stateMatches
    {candidate : Id} (stateDigest : CanonicalDigest)
    (batch : Batch candidate) :
    StateMatches (word stateDigest batch) stateDigest := by
  simp [StateMatches, word, encode, digestBits_length, digestBitCount]

theorem word_fullMatches
    {candidate : Id} (stateDigest : CanonicalDigest)
    (batch : Batch candidate) :
    FullMatches (word stateDigest batch) stateDigest batch := by
  rfl

def stateLaneWord {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate)
    (lane : Fin 4) : CanonicalFieldBits.Word :=
  FixedBits.slice (word stateDigest batch)
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [coordinateCount, CanonicalFieldBits.bitCount] at *
      omega)

theorem stateLaneWord_eq {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate)
    (lane : Fin 4) :
    stateLaneWord stateDigest batch lane =
      CanonicalFieldBits.encode (stateDigest lane) := by
  apply Subtype.ext
  change
    ((encode stateDigest batch).drop
      (1 + CanonicalFieldBits.bitCount * lane.val)).take
        CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (stateDigest lane)).val
  simp only [encode, List.append_assoc, List.singleton_append]
  rw [show 1 + CanonicalFieldBits.bitCount * lane.val =
      Nat.succ (CanonicalFieldBits.bitCount * lane.val) by omega]
  change
    ((digestBits stateDigest ++
      (digestBits (memoryDigest batch) ++
        List.replicate paddingBitCount 0)).drop
        (CanonicalFieldBits.bitCount * lane.val)).take
          CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (stateDigest lane)).val
  have offsetLe : CanonicalFieldBits.bitCount * lane.val ≤
      (digestBits stateDigest).length := by
    rw [digestBits_length]
    have laneBound := lane.isLt
    norm_num [digestBitCount, CanonicalFieldBits.bitCount] at *
    omega
  rw [List.drop_append_of_le_length offsetLe]
  have takeLe : CanonicalFieldBits.bitCount ≤
      (digestBits stateDigest).length -
        CanonicalFieldBits.bitCount * lane.val := by
    rw [digestBits_length]
    have laneBound := lane.isLt
    norm_num [digestBitCount, CanonicalFieldBits.bitCount] at *
    omega
  rw [List.take_append_of_le_length (by
    simpa [List.length_drop] using takeLe)]
  exact digestBits_slice_lane stateDigest lane

def memoryLaneWord {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate)
    (lane : Fin 4) : CanonicalFieldBits.Word :=
  FixedBits.slice (word stateDigest batch)
    (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [coordinateCount, digestBitCount,
        CanonicalFieldBits.bitCount] at *
      omega)

theorem memoryLaneWord_eq {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate)
    (lane : Fin 4) :
    memoryLaneWord stateDigest batch lane =
      CanonicalFieldBits.encode (memoryDigest batch lane) := by
  apply Subtype.ext
  change
    ((encode stateDigest batch).drop
      (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)).take
        CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (memoryDigest batch lane)).val
  simp only [encode, List.append_assoc, List.singleton_append]
  rw [show 1 + digestBitCount +
      CanonicalFieldBits.bitCount * lane.val =
        Nat.succ (digestBitCount +
          CanonicalFieldBits.bitCount * lane.val) by omega]
  change
    ((digestBits stateDigest ++
      (digestBits (memoryDigest batch) ++
        List.replicate paddingBitCount 0)).drop
        (digestBitCount +
          CanonicalFieldBits.bitCount * lane.val)).take
          CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (memoryDigest batch lane)).val
  have offsetShape :
      digestBitCount + CanonicalFieldBits.bitCount * lane.val =
        (digestBits stateDigest).length +
          CanonicalFieldBits.bitCount * lane.val := by
    rw [digestBits_length]
  rw [offsetShape, List.drop_length_add_append]
  have offsetLe : CanonicalFieldBits.bitCount * lane.val ≤
      (digestBits (memoryDigest batch)).length := by
    rw [digestBits_length]
    have laneBound := lane.isLt
    norm_num [digestBitCount, CanonicalFieldBits.bitCount] at *
    omega
  rw [List.drop_append_of_le_length offsetLe]
  have takeLe : CanonicalFieldBits.bitCount ≤
      (digestBits (memoryDigest batch)).length -
        CanonicalFieldBits.bitCount * lane.val := by
    rw [digestBits_length]
    have laneBound := lane.isLt
    norm_num [digestBitCount, CanonicalFieldBits.bitCount] at *
    omega
  rw [List.take_append_of_le_length (by
    simpa [List.length_drop] using takeLe)]
  exact digestBits_slice_lane (memoryDigest batch) lane

theorem encode_get_stateDigest {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    (encode stateDigest batch).getD
        (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode (stateDigest lane)).val.getD bit.val 0 := by
  have selected := FixedBits.slice_getD (word stateDigest batch)
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [coordinateCount, CanonicalFieldBits.bitCount] at *
      omega) bit.val bit.isLt
  change (stateLaneWord stateDigest batch lane).val.getD bit.val 0 =
    (word stateDigest batch).val.getD
      (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 at selected
  rw [stateLaneWord_eq stateDigest batch lane] at selected
  simpa [word] using selected.symm

theorem encode_get_memoryDigest {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    (encode stateDigest batch).getD
        (1 + digestBitCount +
          CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode (memoryDigest batch lane)).val.getD
        bit.val 0 := by
  have selected := FixedBits.slice_getD (word stateDigest batch)
    (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [coordinateCount, digestBitCount,
        CanonicalFieldBits.bitCount] at *
      omega) bit.val bit.isLt
  change (memoryLaneWord stateDigest batch lane).val.getD bit.val 0 =
    (word stateDigest batch).val.getD
      (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val +
        bit.val) 0 at selected
  rw [memoryLaneWord_eq stateDigest batch lane] at selected
  simpa [word] using selected.symm

theorem encode_get_padding {candidate : Id}
    (stateDigest : CanonicalDigest) (batch : Batch candidate)
    (padding : Nat) (paddingBound : padding < paddingBitCount) :
    (encode stateDigest batch).getD
      (1 + digestBitCount + digestBitCount + padding) 0 = 0 := by
  change
    ([1] ++ digestBits stateDigest ++ digestBits (memoryDigest batch) ++
      List.replicate paddingBitCount 0).getD
        (1 + digestBitCount + digestBitCount + padding) 0 = 0
  let headValues := [1] ++ digestBits stateDigest ++
    digestBits (memoryDigest batch)
  have totalShape :
      [1] ++ digestBits stateDigest ++ digestBits (memoryDigest batch) ++
          List.replicate paddingBitCount 0 =
        headValues ++ List.replicate paddingBitCount 0 := by
    simp [headValues, List.append_assoc]
  rw [totalShape]
  have indexShape :
      1 + digestBitCount + digestBitCount + padding =
        headValues.length + padding := by
    simp [headValues, digestBits_length]
    omega
  rw [indexShape]
  simp [paddingBound]

theorem FullMatches.stateMatches
    {candidate : Id} {carrier : FixedBits.Word coordinateCount}
    {stateDigest : CanonicalDigest} {batch : Batch candidate}
    (bound : FullMatches carrier stateDigest batch) :
    StateMatches carrier stateDigest := by
  unfold FullMatches at bound
  unfold StateMatches
  rw [bound]
  simp [encode, digestBits_length, digestBitCount]

theorem FullMatches.memoryMatches
    {candidate : Id} {carrier : FixedBits.Word coordinateCount}
    {stateDigest : CanonicalDigest} {batch : Batch candidate}
    (bound : FullMatches carrier stateDigest batch) :
    MemoryMatches carrier batch := by
  unfold FullMatches at bound
  unfold MemoryMatches
  rw [bound]
  simp [encode, digestBits_length, digestBitCount]

theorem matched_state_eq
    {carrier : FixedBits.Word coordinateCount}
    {left right : CanonicalDigest}
    (leftMatches : StateMatches carrier left)
    (rightMatches : StateMatches carrier right) :
    left = right := by
  apply digestBits_injective
  exact leftMatches.symm.trans rightMatches

theorem matched_batch_digest_eq
    {candidate : Id} {carrier : FixedBits.Word coordinateCount}
    {left right : Batch candidate}
    (leftMatches : MemoryMatches carrier left)
    (rightMatches : MemoryMatches carrier right) :
    ProductionMemoryBatchPoseidonBinding.digest left =
      ProductionMemoryBatchPoseidonBinding.digest right := by
  have bitsEqual : digestBits (memoryDigest left) =
      digestBits (memoryDigest right) := leftMatches.symm.trans rightMatches
  have canonicalEqual := digestBits_injective bitsEqual
  funext lane
  exact congrArg Subtype.val (congrFun canonicalEqual lane)

theorem matched_batch_eq_or_collision
    {candidate : Id} {carrier : FixedBits.Word coordinateCount}
    {left right : Batch candidate}
    (leftCanonical : ∀ claim ∈ left.suffixes,
      MemoryClaimCodec.Claim.Canonical claim)
    (rightCanonical : ∀ claim ∈ right.suffixes,
      MemoryClaimCodec.Claim.Canonical claim)
    (leftMatches : MemoryMatches carrier left)
    (rightMatches : MemoryMatches carrier right) :
    left = right ∨
      ProductionMemoryBatchPoseidonBinding.PoseidonCollision candidate := by
  exact batch_eq_or_poseidon_collision leftCanonical rightCanonical
    (matched_batch_digest_eq leftMatches rightMatches)

/-- One exact 540-coordinate carrier recovers both typed authority inputs, or
exposes the one candidate-specific memory-batch collision. -/
theorem full_matches_unique_or_collision
    {candidate : Id} {carrier : FixedBits.Word coordinateCount}
    {leftState rightState : CanonicalDigest}
    {leftBatch rightBatch : Batch candidate}
    (leftCanonical : ∀ claim ∈ leftBatch.suffixes,
      MemoryClaimCodec.Claim.Canonical claim)
    (rightCanonical : ∀ claim ∈ rightBatch.suffixes,
      MemoryClaimCodec.Claim.Canonical claim)
    (leftMatches : FullMatches carrier leftState leftBatch)
    (rightMatches : FullMatches carrier rightState rightBatch) :
    (leftState = rightState ∧ leftBatch = rightBatch) ∨
      ProductionMemoryBatchPoseidonBinding.PoseidonCollision candidate := by
  have stateEqual := matched_state_eq
    leftMatches.stateMatches rightMatches.stateMatches
  rcases matched_batch_eq_or_collision leftCanonical rightCanonical
      leftMatches.memoryMatches rightMatches.memoryMatches with
    batchEqual | collision
  · exact Or.inl ⟨stateEqual, batchEqual⟩
  · exact Or.inr collision

theorem stateDigestBits_of_encode_eq
    {candidate : Id}
    {leftState rightState : CanonicalDigest}
    {leftBatch rightBatch : Batch candidate}
    (equal : encode leftState leftBatch = encode rightState rightBatch) :
    digestBits leftState = digestBits rightState := by
  have selected := congrArg (fun values => (values.drop 1).take 256) equal
  simpa [encode, digestBits_length, digestBitCount] using selected

theorem memoryDigestBits_of_encode_eq
    {candidate : Id}
    {leftState rightState : CanonicalDigest}
    {leftBatch rightBatch : Batch candidate}
    (equal : encode leftState leftBatch = encode rightState rightBatch) :
    digestBits (memoryDigest leftBatch) =
      digestBits (memoryDigest rightBatch) := by
  have selected := congrArg (fun values => (values.drop 257).take 256) equal
  simpa [encode, digestBits_length, digestBitCount] using selected

theorem stateDigest_eq_of_encode_eq
    {candidate : Id}
    {leftState rightState : CanonicalDigest}
    {leftBatch rightBatch : Batch candidate}
    (equal : encode leftState leftBatch = encode rightState rightBatch) :
    leftState = rightState :=
  digestBits_injective (stateDigestBits_of_encode_eq equal)

theorem batchDigest_eq_of_encode_eq
    {candidate : Id}
    {leftState rightState : CanonicalDigest}
    {leftBatch rightBatch : Batch candidate}
    (equal : encode leftState leftBatch = encode rightState rightBatch) :
    ProductionMemoryBatchPoseidonBinding.digest leftBatch =
      ProductionMemoryBatchPoseidonBinding.digest rightBatch := by
  have canonicalEqual := digestBits_injective
    (memoryDigestBits_of_encode_eq equal)
  funext lane
  exact congrArg Subtype.val (congrFun canonicalEqual lane)

/-- Equal carriers recover the state digest and complete batch, or expose the
one exact candidate-specific batch collision. -/
theorem authority_eq_or_memory_collision
    {candidate : Id}
    {leftState rightState : CanonicalDigest}
    {leftBatch rightBatch : Batch candidate}
    (leftCanonical : ∀ claim ∈ leftBatch.suffixes,
      MemoryClaimCodec.Claim.Canonical claim)
    (rightCanonical : ∀ claim ∈ rightBatch.suffixes,
      MemoryClaimCodec.Claim.Canonical claim)
    (equal : encode leftState leftBatch = encode rightState rightBatch) :
    (leftState = rightState ∧ leftBatch = rightBatch) ∨
      ProductionMemoryBatchPoseidonBinding.PoseidonCollision candidate := by
  have stateEqual := stateDigest_eq_of_encode_eq equal
  have batchDigestEqual := batchDigest_eq_of_encode_eq equal
  rcases batch_eq_or_poseidon_collision leftCanonical rightCanonical
      batchDigestEqual with batchEqual | collision
  · exact Or.inl ⟨stateEqual, batchEqual⟩
  · exact Or.inr collision

end Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic
