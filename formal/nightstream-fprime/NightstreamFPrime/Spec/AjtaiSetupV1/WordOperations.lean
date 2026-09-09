import Init.Data.BitVec.Lemmas
import Mathlib.Logic.Function.Iterate
import NightstreamFPrime.Spec.AjtaiSetupV1.ChaCha20

/-! RFC 8439 sections 2.1–2.3: 32-bit word operations, initialization,
the quarter-round order, ten double rounds and modular feed-forward.
These are deterministic properties of the existing block function. -/

namespace NightstreamFPrime.Spec.AjtaiSetupV1.ChaCha20

private def rfcQuarter (a b c d : BitVec 32) :
    BitVec 32 × BitVec 32 × BitVec 32 × BitVec 32 :=
  let a1 := a + b
  let d1 := (d ^^^ a1).rotateLeft 16
  let c1 := c + d1
  let b1 := (b ^^^ c1).rotateLeft 12
  let a2 := a1 + b1
  let d2 := (d1 ^^^ a2).rotateLeft 8
  let c2 := c1 + d2
  let b2 := (b1 ^^^ c2).rotateLeft 7
  (a2, b2, c2, d2)

theorem add32_eq_bitvec (left right : BitVec 32) :
    add32 left.toNat right.toNat = (left + right).toNat := by
  rfl

theorem xor32_eq_bitvec (left right : BitVec 32) :
    xor32 left.toNat right.toNat = (left ^^^ right).toNat := by
  change (left ^^^ right).toNat % wordModulus = (left ^^^ right).toNat
  exact Nat.mod_eq_of_lt (left ^^^ right).isLt

/-- Rotation agrees with 32-bit rotation for every canonical input word and
every rotation amount, including zero and amounts exceeding the word width. -/
theorem rotateLeft32_eq_bitvec (value : BitVec 32) (amount : Nat) :
    rotateLeft32 value.toNat amount = (value.rotateLeft amount).toNat := by
  have amountBound : amount % 32 ≤ 32 :=
    Nat.le_of_lt (Nat.mod_lt _ (by decide))
  have powerSplit : 2 ^ (32 - amount % 32) * 2 ^ (amount % 32) = 2 ^ 32 := by
    rw [← Nat.pow_add, Nat.sub_add_cancel amountBound]
  have lowBound : value.toNat >>> (32 - amount % 32) < 2 ^ (amount % 32) := by
    rw [Nat.shiftRight_eq_div_pow]
    apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr
    rw [Nat.mul_comm, powerSplit]
    exact value.isLt
  have highForm : (value.toNat <<< (amount % 32)) % 2 ^ 32 =
      2 ^ (amount % 32) * (value.toNat % 2 ^ (32 - amount % 32)) := by
    rw [Nat.shiftLeft_eq, ← powerSplit, Nat.mul_mod_mul_right]
    exact Nat.mul_comm _ _
  have joined :
      (value.toNat <<< (amount % 32)) % 2 ^ 32 + (value.toNat >>> (32 - amount % 32)) =
      (value.toNat <<< (amount % 32)) % 2 ^ 32 ||| (value.toNat >>> (32 - amount % 32)) := by
    rw [highForm]
    exact Nat.two_pow_add_eq_or_of_lt lowBound _
  by_cases zero : amount % 32 = 0
  · simp [rotateLeft32, zero, wordModulus, BitVec.toNat_rotateLeft,
      Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt value.isLt,
      Nat.mod_eq_of_lt value.isLt]
  · simp only [rotateLeft32, if_neg zero]
    change ((value.toNat <<< (amount % 32)) % 2 ^ 32 +
      (value.toNat >>> (32 - amount % 32))) % 2 ^ 32 = _
    rw [joined, ← BitVec.toNat_rotateLeft]
    exact Nat.mod_eq_of_lt (value.rotateLeft amount).isLt

/-- The actual array update agrees with the four-word RFC calculation using
bounded bit-vector addition, XOR and rotation. -/
theorem quarterRound_eq_bitvec (state : Array Nat) (ai bi ci di : Nat)
    (a b c d : BitVec 32)
    (readA : getWord state ai = a.toNat) (readB : getWord state bi = b.toNat)
    (readC : getWord state ci = c.toNat) (readD : getWord state di = d.toNat) :
    quarterRound state ai bi ci di =
      let words := rfcQuarter a b c d
      (((state.set! ai words.1.toNat).set! bi words.2.1.toNat).set!
        ci words.2.2.1.toNat).set! di words.2.2.2.toNat := by
  simp only [quarterRound, rfcQuarter, readA, readB, readC, readD,
    add32_eq_bitvec, xor32_eq_bitvec, rotateLeft32_eq_bitvec]

private theorem add32_lt (left right : Nat) : add32 left right < 2 ^ 32 :=
  Nat.mod_lt _ (by decide)

private theorem rotateLeft32_lt (value amount : Nat) : rotateLeft32 value amount < 2 ^ 32 := by
  dsimp only [rotateLeft32]
  split <;> exact Nat.mod_lt _ (by decide)

private theorem set_canonical (state : Array Nat) (index value : Nat)
    (canonical : ∀ word ∈ state, word < 2 ^ 32) (valueBound : value < 2 ^ 32) :
    ∀ word ∈ state.setIfInBounds index value, word < 2 ^ 32 := by
  intro word membership
  rcases Array.mem_or_eq_of_mem_setIfInBounds membership with old | rfl
  · exact canonical word old
  · exact valueBound

theorem quarterRound_canonical (state : Array Nat) (ai bi ci di : Nat)
    (canonical : ∀ word ∈ state, word < 2 ^ 32) :
    ∀ word ∈ quarterRound state ai bi ci di, word < 2 ^ 32 := by
  dsimp only [quarterRound]
  apply set_canonical
  · apply set_canonical
    · apply set_canonical
      · apply set_canonical
        · exact canonical
        · exact add32_lt _ _
      · exact rotateLeft32_lt _ _
    · exact add32_lt _ _
  · exact rotateLeft32_lt _ _

theorem quarterRound_size (state : Array Nat) (ai bi ci di : Nat) :
    (quarterRound state ai bi ci di).size = state.size := by
  simp [quarterRound]

private theorem doubleRound_canonical (state : Array Nat)
    (canonical : ∀ word ∈ state, word < 2 ^ 32) :
    ∀ word ∈ doubleRound state, word < 2 ^ 32 := by
  dsimp only [doubleRound]
  repeat' apply quarterRound_canonical
  exact canonical

private theorem doubleRound_size (state : Array Nat) :
    (doubleRound state).size = state.size := by
  simp only [doubleRound, quarterRound_size]

theorem runDoubleRounds_canonical (rounds : Nat) (state : Array Nat)
    (canonical : ∀ word ∈ state, word < 2 ^ 32) :
    ∀ word ∈ runDoubleRounds rounds state, word < 2 ^ 32 := by
  induction rounds generalizing state with
  | zero => exact canonical
  | succ rounds inductionHypothesis =>
    exact inductionHypothesis (doubleRound state) (doubleRound_canonical state canonical)

theorem runDoubleRounds_size (rounds : Nat) (state : Array Nat) :
    (runDoubleRounds rounds state).size = state.size := by
  induction rounds generalizing state with
  | zero => rfl
  | succ rounds inductionHypothesis =>
    rw [runDoubleRounds, inductionHypothesis, doubleRound_size]

theorem initialState_canonical (seed : List Nat) (row block lane : Nat) :
    ∀ word ∈ initialState seed row block lane, word < 2 ^ 32 := by
  intro word membership
  have listed := Array.mem_toList_iff.mpr membership
  simp [initialState] at listed
  rcases listed with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    first | exact Nat.mod_lt _ (by decide) | decide

theorem initialState_size (seed : List Nat) (row block lane : Nat) :
    (initialState seed row block lane).size = 16 := by rfl

theorem getWord_canonical (state : Array Nat)
    (canonical : ∀ word ∈ state, word < 2 ^ 32) (index : Nat) :
    getWord state index < 2 ^ 32 := by
  by_cases bounded : index < state.size
  · have valueBound := canonical state[index] (Array.getElem_mem bounded)
    simpa [getWord, Array.getD, bounded] using valueBound
  · simp [getWord, Array.getD, bounded]

/-- The fixed constants and nonce/counter positions are those in RFC 8439.
The eight intervening seed words are covered by `IndexEncoding`. -/
theorem initialState_rfc_framing (seed : List Nat) (row block lane : Nat) :
    (initialState seed row block lane).toList.take 4 =
      [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574] ∧
    (initialState seed row block lane).toList.drop 12 =
      [lane % wordModulus, row % wordModulus, block % wordModulus,
        (block / wordModulus) % wordModulus] := by
  constructor <;> rfl

/-- The column rounds precede the diagonal rounds in the exact stated order. -/
theorem doubleRound_eq_schedule (state : Array Nat) :
    doubleRound state =
      ([(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
        (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)] :
        List (Nat × Nat × Nat × Nat)).foldl
          (fun current indices => quarterRound current
            indices.1 indices.2.1 indices.2.2.1 indices.2.2.2) state := by
  simp only [List.foldl_cons, List.foldl_nil]
  rfl

theorem runDoubleRounds_eq_iterate (rounds : Nat) (state : Array Nat) :
    runDoubleRounds rounds state = (doubleRound^[rounds]) state := by
  induction rounds generalizing state with
  | zero => rfl
  | succ rounds inductionHypothesis =>
      rw [runDoubleRounds, Function.iterate_succ_apply, inductionHypothesis]

/-- The block uses ten double rounds and adds each corresponding original
word modulo 2^32, in coefficient order. -/
theorem blockWords_eq_feedForward (seed : List Nat) (row block lane : Nat) :
    blockWords seed row block lane =
      (List.range 16).map (fun index =>
        (getWord ((doubleRound^[10]) (initialState seed row block lane)) index +
          getWord (initialState seed row block lane) index) % 2 ^ 32) := by
  dsimp only [blockWords]
  rw [runDoubleRounds_eq_iterate]
  rfl

theorem blockWords_length (seed : List Nat) (row block lane : Nat) :
    (blockWords seed row block lane).length = 16 := by
  simp [blockWords]

theorem blockWords_canonical (seed : List Nat) (row block lane : Nat) :
    ∀ word ∈ blockWords seed row block lane, word < 2 ^ 32 := by
  intro word membership
  unfold blockWords at membership
  obtain ⟨index, _, rfl⟩ := List.mem_map.mp membership
  exact Nat.mod_lt _ (by decide)

end NightstreamFPrime.Spec.AjtaiSetupV1.ChaCha20
