import Mathlib.Data.List.GetD
import Mathlib.Data.Fintype.Fin
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum
import NightstreamFPrime.Spec.AjtaiSetupV1

/-! Exact little-endian seed bytes and bounded nonce/counter indices for the
selected ChaCha20 setup. No statement about pseudorandomness is made. -/

namespace NightstreamFPrime.Spec.AjtaiSetupV1.ChaCha20

private theorem recover_four_bytes (a b c d : Nat)
    (ha : a < 256) (hb : b < 256) (hc : c < 256) (hd : d < 256)
    (byte : Fin 4) :
    ((a + 256 * b + 65536 * c + 16777216 * d) % wordModulus /
      256 ^ byte.val) % 256 = [a, b, c, d].getD byte.val 0 := by
  have wordBound : a + 256 * b + 65536 * c + 16777216 * d < wordModulus := by
    unfold wordModulus
    omega
  rw [Nat.mod_eq_of_lt wordBound]
  fin_cases byte <;> norm_num <;> omega

private theorem seedByte_lt (seed : Seed) (index : Nat) :
    seed.bytes.getD index 0 < 256 := by
  by_cases bounded : index < seed.bytes.length
  · rw [List.getD_eq_getElem (l := seed.bytes) (d := 0) bounded]
    exact seed.canonical _ (List.getElem_mem bounded)
  · have zero : seed.bytes.getD index 0 = 0 := by
      apply List.getD_eq_default
      omega
    rw [zero]
    decide

/-- Every byte of an encoded seed word is recovered in little-endian order.
Canonical seed bytes ensure the modular word conversion loses no information. -/
theorem littleEndian32_byte (seed : Seed) (offset : Nat) (byte : Fin 4) :
    (littleEndian32 seed.bytes offset / 256 ^ byte.val) % 256 =
      seed.bytes.getD (offset + byte.val) 0 := by
  have recovered := recover_four_bytes
    (seed.bytes.getD offset 0) (seed.bytes.getD (offset + 1) 0)
    (seed.bytes.getD (offset + 2) 0) (seed.bytes.getD (offset + 3) 0)
    (seedByte_lt seed offset) (seedByte_lt seed (offset + 1))
    (seedByte_lt seed (offset + 2)) (seedByte_lt seed (offset + 3)) byte
  change (littleEndian32 seed.bytes offset / 256 ^ byte.val) % 256 =
    [seed.bytes.getD offset 0, seed.bytes.getD (offset + 1) 0,
      seed.bytes.getD (offset + 2) 0, seed.bytes.getD (offset + 3) 0].getD byte.val 0
    at recovered
  rw [recovered]
  fin_cases byte <;> simp

/-- The eight seed words occupy positions 4 through 11 in their stated order. -/
theorem initialState_seed_word (seed : List Nat) (row block lane : Nat)
    (word : Fin 8) :
    getWord (initialState seed row block lane) (4 + word.val) =
      littleEndian32 seed (4 * word.val) := by
  fin_cases word <;> rfl

/-- Equality of the initial state preserves all 32 canonical seed bytes. -/
theorem initialState_seed_injective (left right : Seed)
    (leftRow leftBlock leftLane rightRow rightBlock rightLane : Nat)
    (same : initialState left.bytes leftRow leftBlock leftLane =
      initialState right.bytes rightRow rightBlock rightLane) : left = right := by
  have sameBytes : left.bytes = right.bytes := by
    apply List.ext_getElem
    · rw [left.length_eq, right.length_eq]
    · intro index leftBound rightBound
      have indexBound : index < 32 := by simpa [left.length_eq] using leftBound
      let word : Fin 8 := ⟨index / 4, by omega⟩
      let byte : Fin 4 := ⟨index % 4, Nat.mod_lt _ (by decide)⟩
      have sameWord :
          getWord (initialState left.bytes leftRow leftBlock leftLane) (4 + word.val) =
          getWord (initialState right.bytes rightRow rightBlock rightLane) (4 + word.val) :=
        congrArg (fun state => getWord state (4 + word.val)) same
      rw [initialState_seed_word left.bytes leftRow leftBlock leftLane word,
        initialState_seed_word right.bytes rightRow rightBlock rightLane word] at sameWord
      have recovered :
          (littleEndian32 left.bytes (4 * word.val) / 256 ^ byte.val) % 256 =
          (littleEndian32 right.bytes (4 * word.val) / 256 ^ byte.val) % 256 :=
        congrArg (fun value => value / 256 ^ byte.val % 256) sameWord
      rw [littleEndian32_byte left (4 * word.val) byte,
        littleEndian32_byte right (4 * word.val) byte] at recovered
      have position : 4 * word.val + byte.val = index := by
        dsimp [word, byte]
        omega
      rw [position] at recovered
      simpa only [List.getD_eq_getElem (l := left.bytes) (d := 0) leftBound,
        List.getD_eq_getElem (l := right.bytes) (d := 0) rightBound] using recovered
  cases left
  cases right
  cases sameBytes
  rfl

/-- Distinct in-range key coordinates cannot alias the nonce/counter frame,
even when the seed words differ. The range premises are essential. -/
theorem initialState_index_injective
    (leftSeed rightSeed : List Nat)
    (leftRow leftBlock leftLane rightRow rightBlock rightLane : Nat)
    (leftRowBound : leftRow < wordModulus)
    (rightRowBound : rightRow < wordModulus)
    (leftBlockBound : leftBlock < wordModulus ^ 2)
    (rightBlockBound : rightBlock < wordModulus ^ 2)
    (leftLaneBound : leftLane < wordModulus)
    (rightLaneBound : rightLane < wordModulus)
    (same : initialState leftSeed leftRow leftBlock leftLane =
      initialState rightSeed rightRow rightBlock rightLane) :
    leftRow = rightRow ∧ leftBlock = rightBlock ∧ leftLane = rightLane := by
  have counter := congrArg (fun state => getWord state 12) same
  have row := congrArg (fun state => getWord state 13) same
  have low := congrArg (fun state => getWord state 14) same
  have high := congrArg (fun state => getWord state 15) same
  change leftLane % wordModulus = rightLane % wordModulus at counter
  change leftRow % wordModulus = rightRow % wordModulus at row
  change leftBlock % wordModulus = rightBlock % wordModulus at low
  change (leftBlock / wordModulus) % wordModulus =
    (rightBlock / wordModulus) % wordModulus at high
  norm_num [wordModulus] at *
  omega

end NightstreamFPrime.Spec.AjtaiSetupV1.ChaCha20
