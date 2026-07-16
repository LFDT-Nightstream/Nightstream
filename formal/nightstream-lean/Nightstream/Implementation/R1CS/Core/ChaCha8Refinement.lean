import Nightstream.Implementation.R1CS.Core.ChaCha8
import Nightstream.Implementation.R1CS.Core.ChaCha8Fast

/-!
Refinement of the machine-efficient ChaCha8 stream to the pure word model.

Assurance tier: implementation correspondence. This file proves equality for
every seed, counter, stream offset, and finite output length; it does not rely
on generated vectors, native evaluation, or a bounded production fixture.

Owns: word-operation refinement; state-transition refinement; block-stream
refinement; and exact `u32`/little-endian-`u64` stream equality.

Does not own: conformance of Rust `rand_chacha` to the pure model; seeded-Phi81
rejection semantics; verifier-owned seed selection; SIS security; R1CS rows;
Poseidon2; transcript authority; row removal; or cost totals.

Emits constraints: no.

Authority boundary: `ChaCha8` is the arithmetic specification and
`ChaCha8Fast` is the optimized implementation. Only the specification may be
used as the protocol-side meaning of a stream; this file permits replacing it
with the fast implementation without changing that meaning.

| Protocol | Phase | Mathematical branch | Theorem | Exact guarantee |
|---|---|---|---|---|
| seeded SIS | seed expansion | 32-bit ARX | `quarterRound_refines` | one fast quarter round equals the pure modular round |
| seeded SIS | seed expansion | block function | `blockWords_eq` | every 16-word fast block equals the pure block |
| seeded SIS | coefficient stream | finite word slice | `words_eq` | arbitrary finite `u32` slices are equal |
| seeded SIS | coefficient stream | little-endian pairing | `u64s_eq` | arbitrary finite `u64` slices are equal |
-/

namespace Nightstream.Implementation.R1CS.ChaCha8Refinement

def stateView (state : Array UInt32) : Array Nat :=
  state.map UInt32.toNat

theorem getWord_refines (state : Array UInt32) (index : Nat) :
    (ChaCha8Fast.getWord state index).toNat =
      ChaCha8.getWord (stateView state) index := by
  simp only [ChaCha8Fast.getWord, ChaCha8.getWord, stateView,
    Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases state[index]? <;> simp

theorem stateView_set (state : Array UInt32) (index : Nat)
    (value : UInt32) :
    stateView (state.set! index value) =
      (stateView state).set! index value.toNat := by
  simp [stateView]

theorem add32_refines (left right : UInt32) :
    (left + right).toNat = ChaCha8.add32 left.toNat right.toNat := by
  simp [ChaCha8.add32, ChaCha8.wordModulus, UInt32.toNat_add]

theorem xor32_refines (left right : UInt32) :
    (left ^^^ right).toNat = ChaCha8.xor32 left.toNat right.toNat := by
  rw [UInt32.toNat_xor]
  change Nat.xor left.toNat right.toNat =
    Nat.xor left.toNat right.toNat % ChaCha8.wordModulus
  unfold ChaCha8.wordModulus
  rw [Nat.mod_eq_of_lt]
  exact Nat.xor_lt_two_pow left.toNat_lt right.toNat_lt

theorem rotateLeft32_refines (value : UInt32) (amount : Nat) :
    (ChaCha8Fast.rotateLeft32 value amount).toNat =
      ChaCha8.rotateLeft32 value.toNat amount := by
  unfold ChaCha8Fast.rotateLeft32 ChaCha8.rotateLeft32
  let normalized := amount % 32
  change
    (if normalized = 0 then value
      else (value <<< UInt32.ofNat normalized) +
        (value >>> UInt32.ofNat (32 - normalized))).toNat =
      if normalized = 0 then value.toNat % ChaCha8.wordModulus
      else
        ((Nat.shiftLeft value.toNat normalized) % ChaCha8.wordModulus +
          Nat.shiftRight value.toNat (32 - normalized)) %
            ChaCha8.wordModulus
  have normalizedLt : normalized < 32 := Nat.mod_lt _ (by decide)
  by_cases normalizedZero : normalized = 0
  · simp [normalizedZero, ChaCha8.wordModulus,
      Nat.mod_eq_of_lt value.toNat_lt]
  · have normalizedLtSize : normalized < UInt32.size :=
      Nat.lt_trans normalizedLt (by decide)
    have subtractLt32 : 32 - normalized < 32 := by omega
    have subtractLtSize : 32 - normalized < UInt32.size :=
      Nat.lt_trans subtractLt32 (by decide)
    simp only [normalizedZero, if_false, UInt32.toNat_add,
      UInt32.toNat_shiftLeft, UInt32.toNat_shiftRight,
      UInt32.toNat_ofNat']
    rw [Nat.mod_eq_of_lt normalizedLtSize,
      Nat.mod_eq_of_lt normalizedLt,
      Nat.mod_eq_of_lt subtractLtSize,
      Nat.mod_eq_of_lt subtractLt32]
    rfl

theorem littleEndian32_refines (bytes : List Nat) (offset : Nat) :
    (ChaCha8Fast.littleEndian32 bytes offset).toNat =
      ChaCha8.littleEndian32 bytes offset := by
  simp [ChaCha8Fast.littleEndian32, ChaCha8.littleEndian32,
    ChaCha8.wordModulus, UInt32.toNat_ofNat']

theorem initialState_refines (seed : List Nat) (block : Nat) :
    stateView (ChaCha8Fast.initialState seed block) =
      ChaCha8.initialState seed block := by
  simp [stateView, ChaCha8Fast.initialState, ChaCha8.initialState,
    littleEndian32_refines, ChaCha8.wordModulus,
    UInt32.toNat_ofNat']

theorem quarterRound_refines (state : Array UInt32)
    (ai bi ci di : Nat) :
    stateView (ChaCha8Fast.quarterRound state ai bi ci di) =
      ChaCha8.quarterRound (stateView state) ai bi ci di := by
  simp only [ChaCha8Fast.quarterRound, ChaCha8.quarterRound,
    stateView_set, getWord_refines, add32_refines, xor32_refines,
    rotateLeft32_refines]

theorem doubleRound_refines (state : Array UInt32) :
    stateView (ChaCha8Fast.doubleRound state) =
      ChaCha8.doubleRound (stateView state) := by
  simp [ChaCha8Fast.doubleRound, ChaCha8.doubleRound,
    quarterRound_refines]

theorem runDoubleRounds_refines (rounds : Nat) (state : Array UInt32) :
    stateView (ChaCha8Fast.runDoubleRounds rounds state) =
      ChaCha8.runDoubleRounds rounds (stateView state) := by
  induction rounds generalizing state with
  | zero => rfl
  | succ rounds ih =>
      simp [ChaCha8Fast.runDoubleRounds, ChaCha8.runDoubleRounds,
        ih, doubleRound_refines]

theorem blockWord32s_refines (seed : List Nat) (block : Nat) :
    (ChaCha8Fast.blockWord32s seed block).map UInt32.toNat =
      ChaCha8.blockWords seed block := by
  simp [ChaCha8Fast.blockWord32s, ChaCha8.blockWords, List.map_map,
    initialState_refines, runDoubleRounds_refines, getWord_refines,
    ChaCha8.add32, ChaCha8.wordModulus]

theorem blockWords_eq (seed : List Nat) (block : Nat) :
    ChaCha8Fast.blockWords seed block = ChaCha8.blockWords seed block := by
  exact blockWord32s_refines seed block

theorem words_eq (seed : List Nat) (wordStart count : Nat) :
    ChaCha8Fast.words seed wordStart count =
      ChaCha8.words seed wordStart count := by
  simp [ChaCha8Fast.words, ChaCha8Fast.word32s, ChaCha8.words,
    List.map_flatMap, blockWord32s_refines]

theorem u64s_eq (seed : List Nat) (wordStart count : Nat) :
    ChaCha8Fast.u64s seed wordStart count =
      ChaCha8.u64s seed wordStart count := by
  simp [ChaCha8Fast.u64s, ChaCha8.u64s, words_eq,
    ChaCha8.wordModulus]

end Nightstream.Implementation.R1CS.ChaCha8Refinement
