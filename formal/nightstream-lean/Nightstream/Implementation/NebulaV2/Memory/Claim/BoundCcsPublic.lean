import Nightstream.Implementation.NebulaV2.Memory.Claim.PoseidonBinding
import Nightstream.Implementation.NebulaV2.Core.FixedBits
import Nightstream.Implementation.NebulaV2.Core.TaggedBitSlices
import Nightstream.Implementation.NebulaV2.Application.Wasm.ResultCodec
import Nightstream.Protocol.NebulaV2.CanonicalFieldBits

/-!
Contract: exact memory-bound CCS public carrier for Nebula V2.

The 540 coordinates are one affine coordinate, 256 state-output digest bits,
256 memory-suffix digest bits, and 27 zero padding coordinates. The carrier
occupies ten complete 54-coordinate ring columns. It binds the complete
canonical memory suffix through the domain-separated Poseidon2 reduction in
`MemoryClaimPoseidonBinding`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryBoundCcsPublic

open Nightstream.Protocol.NebulaV2
open Nightstream.Implementation.NebulaV2.MemoryClaimPoseidonBinding

abbrev CanonicalDigest :=
  Fin 4 → ShiftedTernary41V1.CanonicalGoldilocks

def digestBitCount : Nat := 256
def paddingBitCount : Nat := 27
def coordinateCount : Nat := 540
def ringColumnCount : Nat := 10

theorem exactGeometry :
    coordinateCount = 1 + digestBitCount + digestBitCount + paddingBitCount ∧
      coordinateCount = ringColumnCount * 54 := by
  decide

def digestBlocks (digest : CanonicalDigest) : List (List Nat) :=
  List.ofFn fun lane => (CanonicalFieldBits.encode (digest lane)).val

def digestBits (digest : CanonicalDigest) : List Nat :=
  (digestBlocks digest).flatten

theorem digestBlocks_lengths (digest : CanonicalDigest) :
    (digestBlocks digest).map List.length = List.replicate 4 64 := by
  simp [digestBlocks, CanonicalFieldBits.encode,
    CanonicalFieldBits.bitCount]
  exact ⟨(CanonicalFieldBits.encode (digest 0)).property.1,
    (CanonicalFieldBits.encode (digest 1)).property.1,
    (CanonicalFieldBits.encode (digest 2)).property.1,
    (CanonicalFieldBits.encode (digest 3)).property.1⟩

theorem digestBits_length (digest : CanonicalDigest) :
    (digestBits digest).length = digestBitCount := by
  rw [digestBits, List.length_flatten, digestBlocks_lengths]
  decide

theorem digestBits_slice_lane (digest : CanonicalDigest) (lane : Fin 4) :
    ((digestBits digest).drop
        (CanonicalFieldBits.bitCount * lane.val)).take
        CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (digest lane)).val := by
  have selected := TaggedBitSlices.slice_flatten_at
    (fun position : Fin 4 =>
      (CanonicalFieldBits.encode (digest position)).val)
    (fun _ : Fin 4 => CanonicalFieldBits.bitCount)
    (fun position => (CanonicalFieldBits.encode (digest position)).property.1)
    (List.ofFn id) lane.val (by simpa using lane.isLt)
  fin_cases lane <;>
    simpa [digestBits, digestBlocks, TaggedBitSlices.flatten,
      TaggedBitSlices.offsetAt, CanonicalFieldBits.bitCount] using selected

private theorem getD_append_right
    {Alpha : Type} (left right : List Alpha) (index : Nat)
    (default : Alpha) :
    (left ++ right).getD (left.length + index) default =
      right.getD index default := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by omega)]
  simp only [Nat.add_sub_cancel_left]
  rfl

theorem digestBits_binary (digest : CanonicalDigest)
    (digit : Nat) (member : digit ∈ digestBits digest) : digit < 2 := by
  rcases List.mem_flatten.mp member with ⟨block, blockMember, digitMember⟩
  rcases List.mem_ofFn.mp blockMember with ⟨lane, rfl⟩
  exact (CanonicalFieldBits.encode (digest lane)).property.2 digit digitMember

theorem digestBits_injective : Function.Injective digestBits := by
  intro left right equal
  have blocksEqual : digestBlocks left = digestBlocks right := by
    apply WasmResultCodec.flatten_injective_of_lengths
      (digestBlocks_lengths left) (digestBlocks_lengths right) equal
  have laneBlocksEqual :
      (fun lane => (CanonicalFieldBits.encode (left lane)).val) =
        fun lane => (CanonicalFieldBits.encode (right lane)).val :=
    List.ofFn_injective blocksEqual
  funext lane
  apply Subtype.ext
  have wordsEqual :
      CanonicalFieldBits.encode (left lane) =
        CanonicalFieldBits.encode (right lane) := by
    apply Subtype.ext
    exact congrFun laneBlocksEqual lane
  have decoded := congrArg CanonicalFieldBits.decode wordsEqual
  simpa [CanonicalFieldBits.decode_encode] using decoded

def memoryDigest (claim : Claim) : CanonicalDigest :=
  fun lane => (canonicalDigest claim).lanes lane

def encode (stateDigest : CanonicalDigest) (memory : Claim) : List Nat :=
  [1] ++ digestBits stateDigest ++ digestBits (memoryDigest memory) ++
    List.replicate paddingBitCount 0

theorem encode_length (stateDigest : CanonicalDigest) (memory : Claim) :
    (encode stateDigest memory).length = coordinateCount := by
  norm_num [encode, digestBits_length, digestBitCount,
    paddingBitCount, coordinateCount]

theorem encode_binary (stateDigest : CanonicalDigest) (memory : Claim)
    (digit : Nat) (member : digit ∈ encode stateDigest memory) : digit < 2 := by
  simp only [encode, List.mem_append, List.mem_cons, List.not_mem_nil,
    or_false, List.mem_replicate] at member
  rcases member with ((one | stateMember) | memoryMember) | paddingMember
  · subst digit
    decide
  · exact digestBits_binary stateDigest digit stateMember
  · exact digestBits_binary (memoryDigest memory) digit memoryMember
  · exact paddingMember.2 ▸ (by decide)

def word (stateDigest : CanonicalDigest) (memory : Claim) :
    FixedBits.Word coordinateCount :=
  ⟨encode stateDigest memory, encode_length stateDigest memory,
    encode_binary stateDigest memory⟩

def stateLaneWord (stateDigest : CanonicalDigest) (memory : Claim)
    (lane : Fin 4) : CanonicalFieldBits.Word :=
  FixedBits.slice (word stateDigest memory)
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [coordinateCount, CanonicalFieldBits.bitCount] at *
      omega)

theorem stateLaneWord_eq (stateDigest : CanonicalDigest) (memory : Claim)
    (lane : Fin 4) :
    stateLaneWord stateDigest memory lane =
      CanonicalFieldBits.encode (stateDigest lane) := by
  apply Subtype.ext
  change
    ((encode stateDigest memory).drop
      (1 + CanonicalFieldBits.bitCount * lane.val)).take
        CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (stateDigest lane)).val
  simp only [encode, List.append_assoc, List.singleton_append]
  rw [show 1 + CanonicalFieldBits.bitCount * lane.val =
      Nat.succ (CanonicalFieldBits.bitCount * lane.val) by omega]
  change
    ((digestBits stateDigest ++
      (digestBits (memoryDigest memory) ++
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

def memoryLaneWord (stateDigest : CanonicalDigest) (memory : Claim)
    (lane : Fin 4) : CanonicalFieldBits.Word :=
  FixedBits.slice (word stateDigest memory)
    (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [coordinateCount, digestBitCount,
        CanonicalFieldBits.bitCount] at *
      omega)

theorem memoryLaneWord_eq (stateDigest : CanonicalDigest) (memory : Claim)
    (lane : Fin 4) :
    memoryLaneWord stateDigest memory lane =
      CanonicalFieldBits.encode (memoryDigest memory lane) := by
  apply Subtype.ext
  change
    ((encode stateDigest memory).drop
      (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)).take
        CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (memoryDigest memory lane)).val
  simp only [encode, List.append_assoc, List.singleton_append]
  rw [show 1 + digestBitCount +
      CanonicalFieldBits.bitCount * lane.val =
        Nat.succ (digestBitCount +
          CanonicalFieldBits.bitCount * lane.val) by omega]
  change
    ((digestBits stateDigest ++
      (digestBits (memoryDigest memory) ++
        List.replicate paddingBitCount 0)).drop
        (digestBitCount +
          CanonicalFieldBits.bitCount * lane.val)).take
          CanonicalFieldBits.bitCount =
      (CanonicalFieldBits.encode (memoryDigest memory lane)).val
  have offsetShape :
      digestBitCount + CanonicalFieldBits.bitCount * lane.val =
        (digestBits stateDigest).length +
          CanonicalFieldBits.bitCount * lane.val := by
    rw [digestBits_length]
  rw [offsetShape, List.drop_length_add_append]
  have offsetLe : CanonicalFieldBits.bitCount * lane.val ≤
      (digestBits (memoryDigest memory)).length := by
    rw [digestBits_length]
    have laneBound := lane.isLt
    norm_num [digestBitCount, CanonicalFieldBits.bitCount] at *
    omega
  rw [List.drop_append_of_le_length offsetLe]
  have takeLe : CanonicalFieldBits.bitCount ≤
      (digestBits (memoryDigest memory)).length -
        CanonicalFieldBits.bitCount * lane.val := by
    rw [digestBits_length]
    have laneBound := lane.isLt
    norm_num [digestBitCount, CanonicalFieldBits.bitCount] at *
    omega
  rw [List.take_append_of_le_length (by
    simpa [List.length_drop] using takeLe)]
  exact digestBits_slice_lane (memoryDigest memory) lane

theorem encode_get_stateDigest
    (stateDigest : CanonicalDigest) (memory : Claim)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    (encode stateDigest memory).getD
        (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode (stateDigest lane)).val.getD bit.val 0 := by
  have selected := FixedBits.slice_getD (word stateDigest memory)
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [coordinateCount, CanonicalFieldBits.bitCount] at *
      omega) bit.val bit.isLt
  change (stateLaneWord stateDigest memory lane).val.getD bit.val 0 =
    (word stateDigest memory).val.getD
      (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 at selected
  rw [stateLaneWord_eq stateDigest memory lane] at selected
  simpa [word] using selected.symm

theorem encode_get_memoryDigest
    (stateDigest : CanonicalDigest) (memory : Claim)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    (encode stateDigest memory).getD
        (1 + digestBitCount +
          CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode (memoryDigest memory lane)).val.getD
        bit.val 0 := by
  have selected := FixedBits.slice_getD (word stateDigest memory)
    (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [coordinateCount, digestBitCount,
        CanonicalFieldBits.bitCount] at *
      omega) bit.val bit.isLt
  change (memoryLaneWord stateDigest memory lane).val.getD bit.val 0 =
    (word stateDigest memory).val.getD
      (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val +
        bit.val) 0 at selected
  rw [memoryLaneWord_eq stateDigest memory lane] at selected
  simpa [word] using selected.symm

theorem encode_get_padding
    (stateDigest : CanonicalDigest) (memory : Claim)
    (padding : Nat) (paddingBound : padding < paddingBitCount) :
    (encode stateDigest memory).getD
      (1 + digestBitCount + digestBitCount + padding) 0 = 0 := by
  change
    ([1] ++ digestBits stateDigest ++ digestBits (memoryDigest memory) ++
      List.replicate paddingBitCount 0).getD
        (1 + digestBitCount + digestBitCount + padding) 0 = 0
  let headValues := [1] ++ digestBits stateDigest ++
    digestBits (memoryDigest memory)
  have totalShape :
      [1] ++ digestBits stateDigest ++ digestBits (memoryDigest memory) ++
          List.replicate paddingBitCount 0 =
        headValues ++ List.replicate paddingBitCount 0 := by
    simp [headValues, List.append_assoc]
  rw [totalShape]
  have indexShape :
      1 + digestBitCount + digestBitCount + padding =
        headValues.length + padding := by
    simp [headValues, digestBits_length]
    omega
  rw [indexShape, getD_append_right]
  simp [paddingBound]

/-- The consumer-side check that binds the separately decoded memory suffix
to the memory-digest coordinates accepted by paper NIFS. -/
def MemoryMatches (carrier : FixedBits.Word coordinateCount)
    (memory : Claim) : Prop :=
  (carrier.val.drop 257).take 256 = digestBits (memoryDigest memory)

instance memoryMatchesDecidable
    (carrier : FixedBits.Word coordinateCount) (memory : Claim) :
    Decidable (MemoryMatches carrier memory) := by
  unfold MemoryMatches
  infer_instance

theorem word_memoryMatches (stateDigest : CanonicalDigest) (memory : Claim) :
    MemoryMatches (word stateDigest memory) memory := by
  simp [MemoryMatches, word, encode, digestBits_length, digestBitCount]

theorem matched_memory_digest_eq
    {carrier : FixedBits.Word coordinateCount}
    {left right : Claim}
    (leftMatches : MemoryMatches carrier left)
    (rightMatches : MemoryMatches carrier right) :
    MemoryClaimPoseidonBinding.digest left =
      MemoryClaimPoseidonBinding.digest right := by
  have bitsEqual : digestBits (memoryDigest left) =
      digestBits (memoryDigest right) := leftMatches.symm.trans rightMatches
  have canonicalEqual := digestBits_injective bitsEqual
  funext lane
  exact congrArg Subtype.val (congrFun canonicalEqual lane)

theorem matched_memory_eq_or_collision
    {carrier : FixedBits.Word coordinateCount}
    {left right : Claim}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (leftMatches : MemoryMatches carrier left)
    (rightMatches : MemoryMatches carrier right) :
    left = right ∨ MemoryClaimPoseidonBinding.PoseidonCollision := by
  exact claim_eq_or_poseidon_collision leftCanonical rightCanonical
    (matched_memory_digest_eq leftMatches rightMatches)

theorem stateDigestBits_of_encode_eq
    {leftState rightState : CanonicalDigest}
    {leftMemory rightMemory : Claim}
    (equal : encode leftState leftMemory = encode rightState rightMemory) :
    digestBits leftState = digestBits rightState := by
  have selected := congrArg (fun values => (values.drop 1).take 256) equal
  simpa [encode, digestBits_length, digestBitCount] using selected

theorem memoryDigestBits_of_encode_eq
    {leftState rightState : CanonicalDigest}
    {leftMemory rightMemory : Claim}
    (equal : encode leftState leftMemory = encode rightState rightMemory) :
    digestBits (memoryDigest leftMemory) =
      digestBits (memoryDigest rightMemory) := by
  have selected := congrArg (fun values => (values.drop 257).take 256) equal
  simpa [encode, digestBits_length, digestBitCount] using selected

theorem stateDigest_eq_of_encode_eq
    {leftState rightState : CanonicalDigest}
    {leftMemory rightMemory : Claim}
    (equal : encode leftState leftMemory = encode rightState rightMemory) :
    leftState = rightState :=
  digestBits_injective (stateDigestBits_of_encode_eq equal)

theorem memoryDigest_eq_of_encode_eq
    {leftState rightState : CanonicalDigest}
    {leftMemory rightMemory : Claim}
    (equal : encode leftState leftMemory = encode rightState rightMemory) :
    MemoryClaimPoseidonBinding.digest leftMemory =
      MemoryClaimPoseidonBinding.digest rightMemory := by
  have canonicalEqual := digestBits_injective
    (memoryDigestBits_of_encode_eq equal)
  funext lane
  exact congrArg Subtype.val (congrFun canonicalEqual lane)

/-- Equal carriers recover the state digest and the complete memory suffix,
or expose one exact memory-suffix Poseidon2 collision. -/
theorem authority_eq_or_memory_collision
    {leftState rightState : CanonicalDigest}
    {leftMemory rightMemory : Claim}
    (leftCanonical : leftMemory.Canonical)
    (rightCanonical : rightMemory.Canonical)
    (equal : encode leftState leftMemory = encode rightState rightMemory) :
    (leftState = rightState ∧ leftMemory = rightMemory) ∨
      MemoryClaimPoseidonBinding.PoseidonCollision := by
  have stateEqual := stateDigest_eq_of_encode_eq equal
  have memoryDigestEqual := memoryDigest_eq_of_encode_eq equal
  rcases claim_eq_or_poseidon_collision leftCanonical rightCanonical
      memoryDigestEqual with memoryEqual | collision
  · exact Or.inl ⟨stateEqual, memoryEqual⟩
  · exact Or.inr collision

end Nightstream.Implementation.NebulaV2.MemoryBoundCcsPublic
