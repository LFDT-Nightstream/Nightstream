import Mathlib.Data.BitVec
import Batteries.Data.BitVec.Lemmas
import Nightstream.Implementation.NebulaV2.MemoryCarryParser
import Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec

/-!
Contract: exact Poseidon2 input frame for the V2 memory-carry digest.

Assurance tier: implementation model and cryptographic boundary.

Owns the fixed domain and frame version, exact V2 profile fields, lossless
packing of all 3,433 carry bits into 108 little-endian 32-bit field words,
23 canonical zero high-padding bits, frame length, canonical field bounds,
and frame injectivity.

Does not own Poseidon2 permutation rows, collision resistance, the outer
F-prime state-output frame, or Rust conformance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryCarryHashFrame

open Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

def domainTag : Nat := 0x4e534d43
def frameVersion : Nat := 1
def wordBitCount : Nat := 32
def packedWordCount : Nat := 108
def paddedBitCount : Nat := packedWordCount * wordBitCount
def highPaddingBitCount : Nat := paddedBitCount - carryBits

theorem exact_geometry :
    carryBits = 3433 ∧ paddedBitCount = 3456 ∧
      highPaddingBitCount = 23 := by
  decide

def bitBool (bit : Nat) : Bool := bit = 1

theorem bitBool_toNat_of_below_two
    {bit : Nat} (bound : bit < 2) : (bitBool bit).toNat = bit := by
  interval_cases bit <;> rfl

theorem bitBool_injective_below_two
    {left right : Nat} (leftBound : left < 2) (rightBound : right < 2)
    (equal : bitBool left = bitBool right) : left = right := by
  interval_cases left <;> interval_cases right <;> simp_all [bitBool]

private theorem ofBits_eq_ofDigits_ofFn
    {width : Nat} (bits : Fin width → Bool) :
    Nat.ofBits bits =
      Nat.ofDigits 2 (List.ofFn fun index => (bits index).toNat) := by
  induction width with
  | zero => simp
  | succ width inductionHypothesis =>
      rw [Nat.ofBits_succ, List.ofFn_succ, Nat.ofDigits_cons,
        inductionHypothesis (bits := bits ∘ Fin.succ)]
      simp only [Function.comp_apply]
      omega

def fixedBitsVector {width : Nat} (word : FixedBits.Word width) :
    BitVec width :=
  BitVec.ofFnLE fun index =>
    bitBool (word.val.get
      ⟨index.val, by simpa [word.property.1] using index.isLt⟩)

theorem fixedBitsVector_toNat
    {width : Nat} (word : FixedBits.Word width) :
    (fixedBitsVector word).toNat = FixedBits.decode word := by
  rw [fixedBitsVector, BitVec.toNat_ofFnLE,
    ofBits_eq_ofDigits_ofFn]
  change
    Nat.ofDigits 2
        (List.ofFn fun index : Fin width =>
          (bitBool
            (word.val.get
              ⟨index.val, by simpa [word.property.1] using index.isLt⟩)).toNat) =
      Nat.ofDigits 2 word.val
  congr 1
  apply List.ext_get
  · simp [word.property.1]
  · intro index leftBound rightBound
    simp only [List.get_ofFn]
    apply bitBool_toNat_of_below_two
    exact word.property.2 _ (List.get_mem _ ⟨index, rightBound⟩)

def logicalWord (block : MemoryCarryParser.Block) : BitVec carryBits :=
  fixedBitsVector block

def paddedWord (block : MemoryCarryParser.Block) : BitVec paddedBitCount :=
  (logicalWord block).setWidth paddedBitCount

def splitWords (word : BitVec paddedBitCount) :
    Fin packedWordCount → BitVec wordBitCount :=
  fun index => word.extractLsb' (wordBitCount * index.val) wordBitCount

def packedWords (block : MemoryCarryParser.Block) :
    Fin packedWordCount → BitVec wordBitCount :=
  splitWords (paddedWord block)

def encodePacked (block : MemoryCarryParser.Block) : List Nat :=
  List.ofFn fun index => (packedWords block index).toNat

theorem encodePacked_length (block : MemoryCarryParser.Block) :
    (encodePacked block).length = packedWordCount := by
  simp [encodePacked]

theorem packedWord_lt (block : MemoryCarryParser.Block)
    (index : Fin packedWordCount) :
    (packedWords block index).toNat < 2 ^ wordBitCount :=
  BitVec.toNat_lt_twoPow_of_le (x := packedWords block index) (Nat.le_refl _)

theorem packedWord_canonical (block : MemoryCarryParser.Block)
    (index : Fin packedWordCount) :
    (packedWords block index).toNat <
      Nightstream.Implementation.R1CS.goldilocksP := by
  exact (packedWord_lt block index).trans_le (by decide)

theorem padded_high_bit_zero
    (block : MemoryCarryParser.Block) (index : Nat)
    (logicalEnd : carryBits ≤ index) :
    (paddedWord block).getLsbD index = false := by
  simp only [paddedWord, BitVec.getLsbD_setWidth]
  by_cases inPadded : index < paddedBitCount
  · simp only [inPadded, decide_true, Bool.true_and]
    exact BitVec.getLsbD_of_ge _ _ logicalEnd
  · simp [inPadded]

def joinWords
    (words : Fin packedWordCount → BitVec wordBitCount) :
    BitVec paddedBitCount :=
  BitVec.ofFnLE fun index =>
    (words ⟨index.val / wordBitCount, by
      have bound := index.isLt
      simp only [paddedBitCount] at bound
      exact Nat.div_lt_of_lt_mul (by
        simpa only [Nat.mul_comm] using bound)⟩).getLsb
      ⟨index.val % wordBitCount, Nat.mod_lt _ (by decide)⟩

theorem join_split (word : BitVec paddedBitCount) :
    joinWords (splitWords word) = word := by
  apply BitVec.eq_of_getLsbD_eq
  intro index indexBound
  simp only [joinWords, BitVec.getLsbD_ofFnLE, splitWords]
  rw [dif_pos indexBound]
  change
    (word.extractLsb' (wordBitCount * (index / wordBitCount))
      wordBitCount).getLsbD (index % wordBitCount) = word.getLsbD index
  rw [BitVec.getLsbD_extractLsb']
  have remainderBound : index % wordBitCount < wordBitCount :=
    Nat.mod_lt index (by decide)
  simp only [remainderBound, decide_true, Bool.true_and]
  have division := Nat.mod_add_div index wordBitCount
  have recombine :
      wordBitCount * (index / wordBitCount) + index % wordBitCount = index := by
    omega
  rw [recombine]

theorem splitWords_injective : Function.Injective splitWords := by
  intro left right equal
  rw [← join_split left, ← join_split right, equal]

theorem logicalWord_injective : Function.Injective logicalWord := by
  intro left right equal
  apply Subtype.ext
  apply List.ext_get
  · rw [left.property.1, right.property.1]
  · intro index leftBound rightBound
    have wordBitEqual := congrArg (fun word => word.getLsbD index) equal
    simp only [logicalWord, fixedBitsVector,
      BitVec.getLsbD_ofFnLE] at wordBitEqual
    have indexBound : index < carryBits := by
      simpa [left.property.1] using leftBound
    simp only [indexBound, dite_true] at wordBitEqual
    apply bitBool_injective_below_two
    · exact left.property.2 _ (List.get_mem _ ⟨index, leftBound⟩)
    · exact right.property.2 _ (List.get_mem _ ⟨index, rightBound⟩)
    · simpa using wordBitEqual

theorem encodePacked_injective : Function.Injective encodePacked := by
  intro left right equal
  have wordValueFunctions :
      (fun index => (packedWords left index).toNat) =
        fun index => (packedWords right index).toNat :=
    List.ofFn_injective equal
  have wordFunctions : packedWords left = packedWords right := by
    funext index
    apply BitVec.eq_of_toNat_eq
    exact congrFun wordValueFunctions index
  have paddedEqual : paddedWord left = paddedWord right :=
    splitWords_injective wordFunctions
  have logicalEqual := congrArg (BitVec.setWidth carryBits) paddedEqual
  have exactLogical : logicalWord left = logicalWord right := by
    simpa [paddedWord, Nat.le_of_lt (by decide : carryBits < paddedBitCount)]
      using logicalEqual
  exact logicalWord_injective exactLogical

def profileFields : List Nat :=
  [ profileNameValue Profile.v2.name
  , Profile.v2.version
  , Profile.v2.checkedStepsPerFreshClaim
  , commitmentEncodingValue Profile.v2.commitmentEncoding
  ]

theorem profileFields_exact : profileFields = [2, 2, 1, 1] := rfl

def framePrefix : List Nat :=
  [domainTag, frameVersion] ++ profileFields ++
    [carryBits, wordBitCount, packedWordCount]

theorem framePrefix_exact :
    framePrefix = [0x4e534d43, 1, 2, 2, 1, 1, 3433, 32, 108] := by
  decide

def frame (block : MemoryCarryParser.Block) : List Nat :=
  framePrefix ++ encodePacked block

theorem frame_length (block : MemoryCarryParser.Block) :
    (frame block).length = 117 := by
  simp [frame, framePrefix, profileFields, encodePacked_length,
    packedWordCount]

/-- The exact Poseidon2 field frame is lossless for all accepted 3,433-bit
blocks. -/
theorem frame_injective : Function.Injective frame := by
  intro left right equal
  apply encodePacked_injective
  have tails := congrArg (List.drop framePrefix.length) equal
  simpa [frame] using tails

theorem frame_fields_canonical (block : MemoryCarryParser.Block) :
    ∀ value ∈ frame block,
      value < Nightstream.Implementation.R1CS.goldilocksP := by
  intro value member
  rw [frame, List.mem_append] at member
  rcases member with fixed | packed
  · rw [framePrefix_exact] at fixed
    simp only [List.mem_cons, List.not_mem_nil, or_false] at fixed
    rcases fixed with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      decide
  · rcases List.mem_ofFn.mp packed with ⟨index, rfl⟩
    exact packedWord_canonical block index

abbrev Hash (Digest : Type) := List Nat → Digest

def digest {Digest : Type} (hash : Hash Digest)
    (block : MemoryCarryParser.Block) : Digest :=
  hash (frame block)

def Collision {Digest : Type} (hash : Hash Digest) : Prop :=
  ∃ left right : MemoryCarryParser.Block,
    frame left ≠ frame right ∧ digest hash left = digest hash right

/-- Equal carry digests imply equal complete carry blocks or expose the exact
collision event that the cryptographic reduction must price. -/
theorem block_eq_or_collision
    {Digest : Type} [DecidableEq MemoryCarryParser.Block]
    (hash : Hash Digest) (left right : MemoryCarryParser.Block)
    (equal : digest hash left = digest hash right) :
    left = right ∨ Collision hash := by
  by_cases same : left = right
  · exact Or.inl same
  · have frameDifferent : frame left ≠ frame right :=
      fun frameEqual => same (frame_injective frameEqual)
    exact Or.inr ⟨left, right, frameDifferent, equal⟩

end Nightstream.Implementation.NebulaV2.MemoryCarryHashFrame
