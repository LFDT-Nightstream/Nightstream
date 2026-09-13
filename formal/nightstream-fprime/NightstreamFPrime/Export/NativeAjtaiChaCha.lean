import Init.Data.UInt.Bitwise
import Init.Data.Array.Lemmas
import NightstreamFPrime.Spec.AjtaiSetupV1.WordOperations
import NightstreamFPrime.Spec.AjtaiSetupV1

/-!
Native-word execution of the existing indexed Ajtai ChaCha20 generator.
The proofs preserve the exact seed/nonce/counter, round order, feed-forward
and wide field reduction. This module adds no pseudorandomness assumption.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.NativeAjtaiChaCha

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.AjtaiSetupV1

@[inline] private def rotateLeft (value amount : UInt32) : UInt32 :=
  (value <<< amount) ||| (value >>> ((32 : UInt32) - amount))

private theorem rotateLeft_toNat (value amount : UInt32)
    (positive : 0 < amount.toNat) (below : amount.toNat < 32) :
    (rotateLeft value amount).toNat =
      ChaCha20.rotateLeft32 value.toNat amount.toNat := by
  have amountLe : amount ≤ (32 : UInt32) :=
    UInt32.le_iff_toNat_le.2 (by change amount.toNat ≤ 32; omega)
  have complement : ((32 : UInt32) - amount).toNat = 32 - amount.toNat := by
    rw [UInt32.toNat_sub_of_le 32 amount amountLe]
    rfl
  have complementBelow : 32 - amount.toNat < 32 := by omega
  rw [← UInt32.toNat_toBitVec value, ChaCha20.rotateLeft32_eq_bitvec]
  simp only [rotateLeft, UInt32.toNat_or, UInt32.toNat_shiftLeft,
    UInt32.toNat_shiftRight, complement, BitVec.toNat_rotateLeft,
    UInt32.toNat_toBitVec, Nat.mod_eq_of_lt below,
    Nat.mod_eq_of_lt complementBelow]

private theorem add_toNat (left right : UInt32) :
    (left + right).toNat = ChaCha20.add32 left.toNat right.toNat := by
  exact (ChaCha20.add32_eq_bitvec left.toBitVec right.toBitVec).symm

private theorem xor_toNat (left right : UInt32) :
    (left ^^^ right).toNat = ChaCha20.xor32 left.toNat right.toNat := by
  exact (ChaCha20.xor32_eq_bitvec left.toBitVec right.toBitVec).symm

/-- The existing 16/12/8/7 quarter round using machine-word operations. -/
@[inline] def quarter (a b c d : UInt32) : UInt32 × UInt32 × UInt32 × UInt32 :=
  let a1 := a + b
  let d1 := rotateLeft (d ^^^ a1) 16
  let c1 := c + d1
  let b1 := rotateLeft (b ^^^ c1) 12
  let a2 := a1 + b1
  let d2 := rotateLeft (d1 ^^^ a2) 8
  let c2 := c1 + d2
  let b2 := rotateLeft (b1 ^^^ c2) 7
  (a2, b2, c2, d2)

/-- Native word outputs give the exact existing Nat-array quarter round.
The read equations are the representation boundary. No distinct-index,
cryptographic, expected-output, or counter assumption is added. -/
theorem quarterRound_eq (state : Array Nat) (ai bi ci di : Nat)
    (a b c d : UInt32)
    (readA : ChaCha20.getWord state ai = a.toNat)
    (readB : ChaCha20.getWord state bi = b.toNat)
    (readC : ChaCha20.getWord state ci = c.toNat)
    (readD : ChaCha20.getWord state di = d.toNat) :
    ChaCha20.quarterRound state ai bi ci di =
      let words := quarter a b c d
      (((state.set! ai words.1.toNat).set! bi words.2.1.toNat).set!
        ci words.2.2.1.toNat).set! di words.2.2.2.toNat := by
  have rotate16 (value : UInt32) :
      (rotateLeft value 16).toNat = ChaCha20.rotateLeft32 value.toNat 16 :=
    rotateLeft_toNat value 16 (by decide) (by decide)
  have rotate12 (value : UInt32) :
      (rotateLeft value 12).toNat = ChaCha20.rotateLeft32 value.toNat 12 :=
    rotateLeft_toNat value 12 (by decide) (by decide)
  have rotate8 (value : UInt32) :
      (rotateLeft value 8).toNat = ChaCha20.rotateLeft32 value.toNat 8 :=
    rotateLeft_toNat value 8 (by decide) (by decide)
  have rotate7 (value : UInt32) :
      (rotateLeft value 7).toNat = ChaCha20.rotateLeft32 value.toNat 7 :=
    rotateLeft_toNat value 7 (by decide) (by decide)
  simp only [ChaCha20.quarterRound, quarter, readA, readB, readC, readD,
    add_toNat, xor_toNat, rotate16, rotate12, rotate8, rotate7]


/-- Lift a native array read to the existing Nat-array representation.
The default is zero on both sides, including out-of-bounds indices. -/
private theorem getWord_map (state : Array UInt32) (index : Nat) :
    ChaCha20.getWord (state.map UInt32.toNat) index =
      (state.getD index (0 : UInt32)).toNat := by
  simp only [ChaCha20.getWord, Array.getD_eq_getD_getElem?, Array.getElem?_map]
  exact Option.getD_map UInt32.toNat (0 : UInt32) state[index]?

/-- Read all four original words before the existing ordered array writes. -/
@[inline] def nativeStep (state : Array UInt32) (ai bi ci di : Nat) : Array UInt32 :=
  let words := quarter
    (state.getD ai 0) (state.getD bi 0) (state.getD ci 0) (state.getD di 0)
  (((state.set! ai words.1).set! bi words.2.1).set! ci words.2.2.1).set!
    di words.2.2.2

/-- Native array updates commute with the existing Nat-array quarter round.
Canonical reads follow from UInt32 representation; no array-size or index
premise is required by the total getD/set! operations. -/
theorem nativeStep_map (state : Array UInt32) (ai bi ci di : Nat) :
    (nativeStep state ai bi ci di).map UInt32.toNat =
      ChaCha20.quarterRound (state.map UInt32.toNat) ai bi ci di := by
  have quarterEq := quarterRound_eq (state.map UInt32.toNat) ai bi ci di
    (state.getD ai 0) (state.getD bi 0) (state.getD ci 0) (state.getD di 0)
    (getWord_map state ai) (getWord_map state bi)
    (getWord_map state ci) (getWord_map state di)
  rw [quarterEq]
  simp only [nativeStep, Array.set!_eq_setIfInBounds, Array.map_setIfInBounds]

/-- The exact four column quarters followed by the four diagonal quarters. -/
def doubleRound (state : Array UInt32) : Array UInt32 :=
  let state := nativeStep state 0 4 8 12
  let state := nativeStep state 1 5 9 13
  let state := nativeStep state 2 6 10 14
  let state := nativeStep state 3 7 11 15
  let state := nativeStep state 0 5 10 15
  let state := nativeStep state 1 6 11 12
  let state := nativeStep state 2 7 8 13
  nativeStep state 3 4 9 14

theorem doubleRound_map (state : Array UInt32) :
    (doubleRound state).map UInt32.toNat =
      ChaCha20.doubleRound (state.map UInt32.toNat) := by
  simp only [doubleRound, ChaCha20.doubleRound, nativeStep_map]

/-- The existing recursion order, including the identity at zero rounds. -/
def runDoubleRounds : Nat → Array UInt32 → Array UInt32
  | 0, state => state
  | rounds + 1, state => runDoubleRounds rounds (doubleRound state)

theorem runDoubleRounds_map (rounds : Nat) (state : Array UInt32) :
    (runDoubleRounds rounds state).map UInt32.toNat =
      ChaCha20.runDoubleRounds rounds (state.map UInt32.toNat) := by
  induction rounds generalizing state with
  | zero => rfl
  | succ rounds ih =>
      rw [runDoubleRounds, ChaCha20.runDoubleRounds, ih, doubleRound_map]


/-- Preserve the existing constants, seed words, nonce, and lane counter.
Every initial Nat word is already canonical before conversion to UInt32. -/
def initialState (seed : List Nat) (row block lane : Nat) : Array UInt32 :=
  (ChaCha20.initialState seed row block lane).map UInt32.ofNat

theorem initialState_map (seed : List Nat) (row block lane : Nat) :
    (initialState seed row block lane).map UInt32.toNat =
      ChaCha20.initialState seed row block lane := by
  rw [initialState, Array.map_map]
  calc
    _ = (ChaCha20.initialState seed row block lane).map id := by
      apply Array.map_congr_left
      intro word membership
      exact UInt32.toNat_ofNat_of_lt'
        (ChaCha20.initialState_canonical seed row block lane word membership)
    _ = _ := Array.map_id _

/-- Ten native double rounds followed by the same ordered modular
feed-forward additions. The output remains the existing sixteen Nat words. -/
def blockWords (seed : List Nat) (row block lane : Nat) : List Nat :=
  let initial := initialState seed row block lane
  let permuted := runDoubleRounds 10 initial
  (List.range 16).map fun index =>
    (permuted.getD index 0 + initial.getD index 0).toNat

theorem blockWords_eq (seed : List Nat) (row block lane : Nat) :
    blockWords seed row block lane = ChaCha20.blockWords seed row block lane := by
  dsimp only [blockWords, ChaCha20.blockWords]
  apply List.map_congr_left
  intro index _membership
  rw [add_toNat, ← getWord_map, ← getWord_map]
  simp only [runDoubleRounds_map, initialState_map]

/-- Preserve the existing first-eight-word selection and little-endian
256-bit assembly exactly. -/
def first256Nat (seed : List Nat) (row block lane : Nat) : Nat :=
  ((blockWords seed row block lane).take 8).reverse.foldl
    (fun value word => value * ChaCha20.wordModulus + word) 0

theorem first256Nat_eq (seed : List Nat) (row block lane : Nat) :
    first256Nat seed row block lane = ChaCha20.first256Nat seed row block lane := by
  rw [first256Nat, blockWords_eq]
  rfl

/-- The existing Goldilocks wide-reduction rule, applied to native rounds. -/
def wideCoefficientNat (seed : List Nat) (row block lane : Nat) : Nat :=
  first256Nat seed row block lane % goldilocksModulus

theorem wideCoefficientNat_eq (seed : List Nat) (row block lane : Nat) :
    wideCoefficientNat seed row block lane =
      AjtaiSetupV1.wideCoefficientNat seed row block lane := by
  rw [wideCoefficientNat, first256Nat_eq]
  rfl

/-- Compute the coefficient from the same typed verifier-owned setup.
No supplied key value or equality premise enters the executable. -/
def coefficient {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) (lane : Fin ringDegree) : F :=
  ⟨wideCoefficientNat setup.seed.bytes row.val block.val lane.val,
    by unfold wideCoefficientNat; exact Nat.mod_lt _ (by decide)⟩

/-- The native coefficient is the exact existing indexed key coefficient.
Instantiate setup with the existing productionSetup for the selected key. -/
theorem coefficient_eq {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) (lane : Fin ringDegree) :
    coefficient setup row block lane = setup.verifierKey row block lane := by
  apply Fin.ext
  exact wideCoefficientNat_eq setup.seed.bytes row.val block.val lane.val

end NightstreamFPrime.Export.NativeAjtaiChaCha
