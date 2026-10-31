import NeoVerified.Funs
import CompressionLean.Fields
import CompressionLean.Ports.Replay.Field
import CompressionLeanRust.Shape

/-! The translated field of `neo_verified::field` equals the copy in
`CompressionLean/Ports/Replay/Field.lean`, and each operation is that of `ZMod p` (`Gl`) or of
`AdjoinRoot (X^3 - X - 1)` (`Cubic`).

The Rust word functions are `add64`, `sub64` and `mul64` of `NativeGoldilocks.lean` step for step:
`toU` of a translated result is that function of `toU` of its inputs, on every input, so the
lemmas of that file apply. Every operation reduces its inputs first, so a copy reads an input word
`w` as `Word.ofNat w`. No operation can fail, so each `Complete` premise is `True`. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open NightstreamFPrime.Spec (goldilocksModulus)
open NightstreamFPrime.Export.NativePoseidon2 (add64 sub64 mul64 ofNat64 low64 high64 add64_canonical
  sub64_canonical mul64_canonical ofNat64_canonical add64_denote sub64_denote mul64_denote ofNat64_denote
  low64_toNat high64_toNat)

/-! ## Words as `UInt64` -/

/-- A translated word as Lean's `UInt64`, the word type of `NativeGoldilocks.lean`. -/
def toU (x : U64) : UInt64 := ⟨x.bv⟩

@[simp] theorem toU_toNat (x : U64) : (toU x).toNat = x.val := rfl

theorem toU_add (x y : U64) : toU (core.num.U64.wrapping_add x y) = toU x + toU y := rfl

theorem toU_sub (x y : U64) : toU (core.num.U64.wrapping_sub x y) = toU x - toU y := rfl

theorem toU_mul (x y : U64) : toU (core.num.U64.wrapping_mul x y) = toU x * toU y := rfl

theorem toU_lt (x y : U64) : x < y ↔ toU x < toU y := by
  rw [UScalar.lt_equiv, UInt64.lt_iff_toNat_lt, toU_toNat, toU_toNat]

theorem toU_le (x y : U64) : x ≤ y ↔ toU x ≤ toU y := by
  rw [UScalar.le_equiv, UInt64.le_iff_toNat_le, toU_toNat, toU_toNat]

theorem modulus_val : field.MODULUS.val = goldilocksModulus := by
  unfold field.MODULUS
  rfl

theorem toU_modulus : toU field.MODULUS = 0xffffffff00000001 :=
  UInt64.toNat_inj.mp (by rw [toU_toNat, modulus_val]; rfl)

theorem toU_shr (x : U64) : toU (core.num.U64.wrapping_shr x 32#u32) = high64 (toU x) := by
  apply UInt64.toNat_inj.mp
  rw [high64_toNat, toU_toNat, toU_toNat]
  simp only [UScalar.val, core.num.U64.wrapping_shr_bv_eq]
  rw [show (32#u32).bv.toNat % 64 = 32 from rfl, BitVec.ushiftRight_eq, BitVec.toNat_ushiftRight,
    Nat.shiftRight_eq_div_pow]
  rfl

theorem toU_shl (x : U64) : toU (core.num.U64.wrapping_shl x 32#u32) = toU x <<< 32 := by
  apply UInt64.toNat_inj.mp
  rw [toU_toNat, UInt64.toNat_shiftLeft, toU_toNat]
  simp only [UScalar.val, core.num.U64.wrapping_shl_bv_eq]
  rfl

/-- The low 32 bits, as `low_half`. -/
def lowW (w : U64) : U64 := UScalar.cast .U64 (UScalar.cast .U32 w)

theorem toU_low (w : U64) : toU (lowW w) = low64 (toU w) := by
  apply UInt64.toNat_inj.mp
  rw [low64_toNat, toU_toNat, toU_toNat, lowW, UScalar.cast_val_eq, UScalar.cast_val_eq]
  simp only [UScalarTy.U32_numBits_eq, UScalarTy.U64_numBits_eq, NightstreamFPrime.Export.NativePoseidon2.radix]
  omega

/-! ## The word functions

Each private word function of `field.rs` returns a pure value (`…W`), and `toU` of that value is
the function of `NativeGoldilocks.lean` of the same name. -/

/-- `NativeGoldilocks.lean`'s `canonicalize64`, which is private there. -/
def canon64 (v : UInt64) : UInt64 := if v < 0xffffffff00000001 then v else v - 0xffffffff00000001

/-- `NativeGoldilocks.lean`'s `foldHigh64`, which is private there. -/
def foldHigh64 (v : UInt64) : UInt64 := sub64 ((low64 v <<< 32) - low64 v) (high64 v)

theorem add64_eq (x y : UInt64) :
    add64 x y = if x < 0xffffffff00000001 - y then x + y else x - (0xffffffff00000001 - y) := rfl

theorem sub64_eq (x y : UInt64) : sub64 x y = if y ≤ x then x - y else 0xffffffff00000001 - (y - x) := rfl

theorem mul64_eq (a b : UInt64) : mul64 a b =
    let a0 := low64 a
    let a1 := high64 a
    let b0 := low64 b
    let b1 := high64 b
    let p00 := a0 * b0
    let p10 := a1 * b0
    let t0 := p10 + high64 p00
    let p01 := a0 * b1
    let t1 := p01 + low64 t0
    let low := (low64 t1 <<< 32) + low64 p00
    let p11 := a1 * b1
    let high := p11 + high64 t0 + high64 t1
    add64 (canon64 low) (foldHigh64 high) := rfl

def redW (w : U64) : U64 := if w < field.MODULUS then w else core.num.U64.wrapping_sub w field.MODULUS

def addW (a b : U64) : U64 :=
  if a < core.num.U64.wrapping_sub field.MODULUS b then core.num.U64.wrapping_add a b
  else core.num.U64.wrapping_sub a (core.num.U64.wrapping_sub field.MODULUS b)

def subW (a b : U64) : U64 :=
  if b ≤ a then core.num.U64.wrapping_sub a b
  else core.num.U64.wrapping_sub field.MODULUS (core.num.U64.wrapping_sub b a)

def foldW (w : U64) : U64 :=
  subW (core.num.U64.wrapping_sub (core.num.U64.wrapping_shl (lowW w) 32#u32) (lowW w))
    (core.num.U64.wrapping_shr w 32#u32)

def mulW (a b : U64) : U64 :=
  let a0 := lowW a
  let a1 := core.num.U64.wrapping_shr a 32#u32
  let b0 := lowW b
  let b1 := core.num.U64.wrapping_shr b 32#u32
  let p00 := core.num.U64.wrapping_mul a0 b0
  let p10 := core.num.U64.wrapping_mul a1 b0
  let t0 := core.num.U64.wrapping_add p10 (core.num.U64.wrapping_shr p00 32#u32)
  let p01 := core.num.U64.wrapping_mul a0 b1
  let t1 := core.num.U64.wrapping_add p01 (lowW t0)
  let low := core.num.U64.wrapping_add (core.num.U64.wrapping_shl (lowW t1) 32#u32) (lowW p00)
  let p11 := core.num.U64.wrapping_mul a1 b1
  let high := core.num.U64.wrapping_add (core.num.U64.wrapping_add p11 (core.num.U64.wrapping_shr t0 32#u32))
    (core.num.U64.wrapping_shr t1 32#u32)
  addW (redW low) (foldW high)

theorem reduce_word_eq (w : U64) : field.reduce_word w = ok (redW w) := by
  unfold field.reduce_word redW
  split <;> rfl

theorem add_words_eq (a b : U64) : field.add_words a b = ok (addW a b) := by
  unfold field.add_words addW
  simp only [lift, bind_ok]
  split <;> rfl

theorem sub_words_eq (a b : U64) : field.sub_words a b = ok (subW a b) := by
  unfold field.sub_words subW
  simp only [lift, bind_ok]
  split <;> rfl

theorem low_half_eq (w : U64) : field.low_half w = ok (lowW w) := by
  simp only [field.low_half, lift, bind_ok, lowW]

theorem high_half_eq (w : U64) : field.high_half w = ok (core.num.U64.wrapping_shr w 32#u32) := by
  simp only [field.high_half]

theorem fold_high_eq (w : U64) : field.fold_high w = ok (foldW w) := by
  simp only [field.fold_high, low_half_eq, high_half_eq, sub_words_eq, lift, bind_ok, foldW]

theorem mul_words_eq (a b : U64) : field.mul_words a b = ok (mulW a b) := by
  simp only [field.mul_words, low_half_eq, high_half_eq, reduce_word_eq, fold_high_eq, add_words_eq, lift,
    bind_ok, mulW]

theorem toU_red (w : U64) : toU (redW w) = canon64 (toU w) := by
  unfold redW canon64
  rw [← toU_modulus]
  by_cases h : w < field.MODULUS
  · rw [ite_eq_left h, ite_eq_left ((toU_lt _ _).mp h)]
  · rw [ite_eq_right h, ite_eq_right (fun h' => h ((toU_lt _ _).mpr h')), toU_sub]

theorem toU_addW (a b : U64) : toU (addW a b) = add64 (toU a) (toU b) := by
  unfold addW
  rw [add64_eq, ← toU_modulus, ← toU_sub]
  by_cases h : a < core.num.U64.wrapping_sub field.MODULUS b
  · rw [ite_eq_left h, ite_eq_left ((toU_lt _ _).mp h), toU_add]
  · rw [ite_eq_right h, ite_eq_right (fun h' => h ((toU_lt _ _).mpr h')), toU_sub]

theorem toU_subW (a b : U64) : toU (subW a b) = sub64 (toU a) (toU b) := by
  unfold subW
  rw [sub64_eq, ← toU_modulus]
  by_cases h : b ≤ a
  · rw [ite_eq_left h, ite_eq_left ((toU_le _ _).mp h), toU_sub]
  · rw [ite_eq_right h, ite_eq_right (fun h' => h ((toU_le _ _).mpr h')), toU_sub, toU_sub]

theorem toU_foldW (w : U64) : toU (foldW w) = foldHigh64 (toU w) := by
  rw [foldW, toU_subW, toU_sub, toU_shl, toU_low, toU_shr, foldHigh64]

theorem toU_mulW (a b : U64) : toU (mulW a b) = mul64 (toU a) (toU b) := by
  simp only [mulW, toU_addW, toU_red, toU_foldW, toU_add, toU_mul, toU_shl, toU_shr, toU_low, mul64_eq]

/-! ## Copies read a translated word mod `p` -/

theorem ofNat64_toNat (n : ℕ) : (ofNat64 n).toNat = n % goldilocksModulus :=
  (Nat.mod_eq_of_lt (ofNat64_canonical n)).symm.trans (congrArg Fin.val (ofNat64_denote n))

theorem canon_ofNat64 (w : U64) : canon64 (toU w) = ofNat64 w.val := by
  apply UInt64.toNat_inj.mp
  have bound := (toU w).toNat_lt
  rw [toU_toNat] at bound
  rw [ofNat64_toNat]
  unfold canon64
  split
  · rename_i h
    rw [UInt64.lt_iff_toNat_lt, toU_toNat] at h
    rw [toU_toNat, Nat.mod_eq_of_lt (by unfold goldilocksModulus; simpa using h)]
  · rename_i h
    rw [UInt64.lt_iff_toNat_lt, toU_toNat] at h
    have le : (0xffffffff00000001 : UInt64) ≤ toU w := UInt64.le_iff_toNat_le.mpr (by simpa using h)
    rw [UInt64.toNat_sub_of_le _ _ le, toU_toNat]
    simp only [UInt64.reduceToNat, goldilocksModulus] at h ⊢
    omega

/-- The copy's reading of a translated `Gl`: its word mod `p`. -/
def wordOf (g : field.Gl) : Replay.Word := Replay.Word.ofNat g.val

theorem toU_red_word (w : U64) : toU (redW w) = (wordOf w).word := by
  rw [toU_red, canon_ofNat64]
  rfl

/-- The translated word `c` is the copy's word `x`. -/
def Exact (c : U64) (x : Replay.Word) : Prop := toU c = x.word

theorem Exact.val {c : U64} {x : Replay.Word} (h : Exact c x) : c.val = x.toNat := by
  rw [← toU_toNat, h]
  rfl

theorem Exact.wordOf {c : U64} {x : Replay.Word} (h : Exact c x) : wordOf c = x := by
  have value := h.val
  have small : c.val < goldilocksModulus := value ▸ x.canonical
  obtain ⟨w, hw⟩ := x
  unfold CompressionLeanRust.wordOf Replay.Word.ofNat
  congr 1
  apply UInt64.toNat_inj.mp
  rw [ofNat64_toNat, Nat.mod_eq_of_lt small]
  exact value

theorem word_add (x y : Replay.Word) : (x + y).word = add64 x.word y.word := by
  simp only [HAdd.hAdd, Add.add]

theorem word_sub (x y : Replay.Word) : (x - y).word = sub64 x.word y.word := rfl

theorem word_mul (x y : Replay.Word) : (x * y).word = mul64 x.word y.word := by
  simp only [HMul.hMul, Mul.mul]

/-! ## `Gl` -/

def glAdd (a b : field.Gl) : U64 := addW (redW a) (redW b)

def glSub (a b : field.Gl) : U64 := subW (redW a) (redW b)

def glMul (a b : field.Gl) : U64 := mulW (redW a) (redW b)

theorem reduce_eq (w : U64) : field.reduce w = ok (redW w) := by
  simp only [field.reduce, reduce_word_eq, bind_ok]

theorem gl_add_eq (a b : field.Gl) : field.gl_add a b = ok (glAdd a b) := by
  simp only [field.gl_add, reduce_word_eq, add_words_eq, bind_ok, glAdd]

theorem gl_sub_eq (a b : field.Gl) : field.gl_sub a b = ok (glSub a b) := by
  simp only [field.gl_sub, reduce_word_eq, sub_words_eq, bind_ok, glSub]

theorem gl_neg_eq (a : field.Gl) : field.gl_neg a = ok (glSub field.GL_ZERO a) := by
  simp only [field.gl_neg, gl_sub_eq]

theorem gl_mul_eq (a b : field.Gl) : field.gl_mul a b = ok (glMul a b) := by
  simp only [field.gl_mul, reduce_word_eq, mul_words_eq, bind_ok, glMul]

theorem glAdd_exact (a b : field.Gl) : Exact (glAdd a b) (wordOf a + wordOf b) := by
  unfold Exact glAdd
  rw [toU_addW, toU_red_word, toU_red_word, word_add]

theorem glSub_exact (a b : field.Gl) : Exact (glSub a b) (wordOf a - wordOf b) := by
  unfold Exact glSub
  rw [toU_subW, toU_red_word, toU_red_word, word_sub]

theorem glMul_exact (a b : field.Gl) : Exact (glMul a b) (wordOf a * wordOf b) := by
  unfold Exact glMul
  rw [toU_mulW, toU_red_word, toU_red_word, word_mul]

theorem wordOf_zero : wordOf field.GL_ZERO = 0 := by
  unfold wordOf field.GL_ZERO
  rfl

theorem glNeg_exact (a : field.Gl) : Exact (glSub field.GL_ZERO a) (-wordOf a) := by
  have h := glSub_exact field.GL_ZERO a
  rwa [wordOf_zero] at h

theorem one_exact : Exact field.GL_ONE 1 := by
  unfold Exact field.GL_ONE
  rfl

theorem shr_one_val (e : U64) : (core.num.U64.wrapping_shr e 1#u32).val = e.val / 2 := by
  simp only [UScalar.val, core.num.U64.wrapping_shr_bv_eq]
  rw [show (1#u32).bv.toNat % 64 = 1 from rfl, BitVec.ushiftRight_eq, BitVec.toNat_ushiftRight,
    Nat.shiftRight_eq_div_pow]

theorem is_multiple_of_two (e : U64) : core.num.U64.is_multiple_of e 2#u64 = ok (e.val % 2 == 0) := rfl

theorem u64_eq_zero (e : U64) : e = 0#u64 ↔ e.val = 0 := by
  constructor
  · rintro rfl
    rfl
  · intro h
    exact UScalar.val_eq_imp _ _ (by rw [h]; rfl)

theorem gl_pow_exact (n : ℕ) : ∀ (a : field.Gl) (e : U64), e.val = n →
    ∃ c, field.gl_pow a e = ok c ∧ Exact c ((wordOf a).pow n) := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro a e he
    rw [field.gl_pow, Replay.Word.pow]
    by_cases zero : n = 0
    · rw [ite_eq_left ((u64_eq_zero e).mpr (he.trans zero)), ite_eq_left zero]
      exact ⟨_, rfl, one_exact⟩
    · rw [ite_eq_right (fun h => zero (he.symm.trans ((u64_eq_zero e).mp h))), ite_eq_right zero]
      obtain ⟨half, run, exact⟩ := ih (n / 2) (by omega) (glMul a a) (core.num.U64.wrapping_shr e 1#u32)
        (by rw [shr_one_val, he])
      rw [(glMul_exact a a).wordOf] at exact
      simp only [gl_mul_eq, lift, bind_ok, run, is_multiple_of_two, he]
      by_cases even : n % 2 = 0
      · rw [show (n % 2 == 0) = true from beq_iff_eq.mpr even, ite_eq_left rfl,
          ite_eq_right (by omega)]
        exact ⟨half, rfl, exact⟩
      · rw [show (n % 2 == 0) = false from beq_eq_false_iff_ne.mpr even, ite_eq_right (by simp),
          ite_eq_left (by omega)]
        refine ⟨_, rfl, ?_⟩
        have h := glMul_exact half a
        rwa [exact.wordOf] at h

theorem gl_inv_exact (a : field.Gl) : ∃ c, field.gl_inv a = ok c ∧ Exact c (wordOf a).inv := by
  simp only [field.gl_inv, lift, bind_ok]
  exact gl_pow_exact _ a _ (by
    simp only [UScalar.val, core.num.U64.wrapping_sub_bv_eq, field.MODULUS]
    rfl)

/-! ## `Gl` is `ZMod p` -/

/-- A translated word as an element of `ZMod p`. -/
def glOf (c : U64) : Gl := (c.val : Gl)

theorem toNat_add64 {x y : UInt64} (hx : x.toNat < goldilocksModulus) (hy : y.toNat < goldilocksModulus) :
    (add64 x y).toNat = (x.toNat + y.toNat) % goldilocksModulus := by
  have h := congrArg Fin.val (add64_denote x y hx hy)
  rw [Fin.val_add] at h
  exact (Nat.mod_eq_of_lt (add64_canonical x y hx hy)).symm.trans
    (h.trans (by rw [show (x.denote).val = x.toNat % goldilocksModulus from rfl,
      show (y.denote).val = y.toNat % goldilocksModulus from rfl, Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy]))

theorem toNat_sub64 {x y : UInt64} (hx : x.toNat < goldilocksModulus) (hy : y.toNat < goldilocksModulus) :
    (sub64 x y).toNat = (goldilocksModulus - y.toNat + x.toNat) % goldilocksModulus := by
  have h := congrArg Fin.val (sub64_denote x y hx hy)
  rw [Fin.val_sub] at h
  exact (Nat.mod_eq_of_lt (sub64_canonical x y hx hy)).symm.trans
    (h.trans (by rw [show (x.denote).val = x.toNat % goldilocksModulus from rfl,
      show (y.denote).val = y.toNat % goldilocksModulus from rfl, Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy]))

theorem toNat_mul64 {x y : UInt64} (hx : x.toNat < goldilocksModulus) (hy : y.toNat < goldilocksModulus) :
    (mul64 x y).toNat = x.toNat * y.toNat % goldilocksModulus := by
  have h := congrArg Fin.val (mul64_denote x y hx hy)
  rw [Fin.val_mul] at h
  exact (Nat.mod_eq_of_lt (mul64_canonical x y)).symm.trans
    (h.trans (by rw [show (x.denote).val = x.toNat % goldilocksModulus from rfl,
      show (y.denote).val = y.toNat % goldilocksModulus from rfl, Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy]))

theorem cast_wordOf (g : field.Gl) : ((wordOf g).toNat : Gl) = glOf g := by
  rw [Replay.Word.toNat, wordOf, Replay.Word.ofNat, ofNat64_toNat, ZMod.natCast_mod, glOf]

theorem cast_add (x y : Replay.Word) : ((x + y).toNat : Gl) = (x.toNat : Gl) + (y.toNat : Gl) := by
  rw [Replay.Word.toNat, word_add, toNat_add64 x.canonical y.canonical, ZMod.natCast_mod, Nat.cast_add]
  rfl

theorem cast_sub (x y : Replay.Word) : ((x - y).toNat : Gl) = (x.toNat : Gl) - (y.toNat : Gl) := by
  rw [Replay.Word.toNat, word_sub, toNat_sub64 x.canonical y.canonical, ZMod.natCast_mod, Nat.cast_add,
    Nat.cast_sub y.canonical.le, ZMod.natCast_self]
  simp only [Replay.Word.toNat]
  ring

theorem cast_mul (x y : Replay.Word) : ((x * y).toNat : Gl) = (x.toNat : Gl) * (y.toNat : Gl) := by
  rw [Replay.Word.toNat, word_mul, toNat_mul64 x.canonical y.canonical, ZMod.natCast_mod, Nat.cast_mul]
  rfl

theorem cast_one : ((1 : Replay.Word).toNat : Gl) = 1 := by
  rw [Replay.Word.toNat, show (1 : Replay.Word).word = ofNat64 1 from rfl, ofNat64_toNat]
  rfl

theorem cast_pow (x : Replay.Word) (n : ℕ) : ((x.pow n).toNat : Gl) = (x.toNat : Gl) ^ n := by
  induction n using Nat.strong_induction_on generalizing x with
  | _ n ih =>
    rw [Replay.Word.pow]
    by_cases zero : n = 0
    · rw [ite_eq_left zero, zero, cast_one, pow_zero]
    · rw [ite_eq_right zero]
      dsimp only
      have half := ih (n / 2) (by omega) (x * x)
      rw [cast_mul] at half
      have split := Nat.div_add_mod n 2
      split_ifs with odd
      · rw [cast_mul, half, ← pow_two, ← pow_mul, ← pow_succ]
        congr 1
        omega
      · rw [half, ← pow_two, ← pow_mul]
        congr 1
        omega

theorem Exact.glOf {c : U64} {x : Replay.Word} (h : Exact c x) : CompressionLeanRust.glOf c = x.toNat := by
  rw [CompressionLeanRust.glOf, h.val]

theorem glAdd_field (a b : field.Gl) : glOf (glAdd a b) = glOf a + glOf b := by
  rw [(glAdd_exact a b).glOf, cast_add, cast_wordOf, cast_wordOf]

theorem glSub_field (a b : field.Gl) : glOf (glSub a b) = glOf a - glOf b := by
  rw [(glSub_exact a b).glOf, cast_sub, cast_wordOf, cast_wordOf]

theorem glMul_field (a b : field.Gl) : glOf (glMul a b) = glOf a * glOf b := by
  rw [(glMul_exact a b).glOf, cast_mul, cast_wordOf, cast_wordOf]

theorem glOf_zero : glOf field.GL_ZERO = 0 := by
  unfold glOf field.GL_ZERO
  rfl

theorem glOf_one : glOf field.GL_ONE = 1 := by
  unfold glOf field.GL_ONE
  rfl

theorem glOf_red (w : U64) : glOf (redW w) = glOf w := by
  rw [(show Exact (redW w) (wordOf w) from toU_red_word w).glOf, cast_wordOf]

/-! ## Each `Gl` function equals its copy and its `ZMod p` operation -/

theorem canonical_sound : Sound field.canonical optionValue
    (fun x => (if x.val < goldilocksModulus then some (Replay.Word.ofNat x.val) else none).map Replay.Word.toNat) := by
  intro w y h
  dsimp only
  unfold field.canonical at h
  have small : w < field.MODULUS ↔ w.val < goldilocksModulus := by rw [UScalar.lt_equiv, modulus_val]
  by_cases lt : w.val < goldilocksModulus
  · rw [ite_eq_left (small.mpr lt), Result.ok.injEq] at h
    subst h
    rw [ite_eq_left lt]
    simp only [optionValue, glValue, Option.map_some, Replay.Word.toNat, Replay.Word.ofNat, ofNat64_toNat,
      Nat.mod_eq_of_lt lt]
  · rw [ite_eq_right (fun h' => lt (small.mp h')), Result.ok.injEq] at h
    subst h
    rw [ite_eq_right lt]
    rfl

theorem canonical_complete : Complete (fun _ => True) field.canonical optionValue
    (fun x => (if x.val < goldilocksModulus then some (Replay.Word.ofNat x.val) else none).map Replay.Word.toNat) :=
  complete_of canonical_sound ⟨0#u64, trivial⟩ fun w _ => by
    unfold field.canonical
    split
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩

theorem canonical_field (w : U64) (g : field.Gl) (h : field.canonical w = ok (some g)) :
    g.val = w.val ∧ w.val < goldilocksModulus := by
  unfold field.canonical at h
  split at h
  · rename_i lt
    rw [Result.ok.injEq, Option.some.injEq] at h
    subst h
    exact ⟨rfl, by rwa [UScalar.lt_equiv, modulus_val] at lt⟩
  · rw [Result.ok.injEq] at h
    cases h

theorem reduce_sound : Sound field.reduce glValue (fun x => (Replay.Word.ofNat x.val).toNat) := by
  intro w y h
  rw [reduce_eq, Result.ok.injEq] at h
  subst h
  exact (show Exact (redW w) (wordOf w) from toU_red_word w).val

theorem reduce_complete : Complete (fun _ => True) field.reduce glValue (fun x => (Replay.Word.ofNat x.val).toNat) :=
  complete_of reduce_sound ⟨0#u64, trivial⟩ fun w _ => ⟨_, reduce_eq w⟩

theorem reduce_field (w : U64) (g : field.Gl) (h : field.reduce w = ok g) : glOf g = (w.val : Gl) := by
  rw [reduce_eq, Result.ok.injEq] at h
  subst h
  exact glOf_red w

theorem gl_add_sound : Sound (fun x : field.Gl × field.Gl => field.gl_add x.1 x.2) glValue
    (fun x => (wordOf x.1 + wordOf x.2).toNat) := by
  rintro ⟨a, b⟩ y h
  dsimp only at h ⊢
  rw [gl_add_eq, Result.ok.injEq] at h
  subst h
  exact (glAdd_exact a b).val

theorem gl_add_complete : Complete (fun _ => True) (fun x : field.Gl × field.Gl => field.gl_add x.1 x.2) glValue
    (fun x => (wordOf x.1 + wordOf x.2).toNat) :=
  complete_of gl_add_sound ⟨(0#u64, 0#u64), trivial⟩ fun x _ => ⟨_, gl_add_eq x.1 x.2⟩

theorem gl_add_field (a b c : field.Gl) (h : field.gl_add a b = ok c) : glOf c = glOf a + glOf b := by
  rw [gl_add_eq, Result.ok.injEq] at h
  subst h
  exact glAdd_field a b

theorem gl_sub_sound : Sound (fun x : field.Gl × field.Gl => field.gl_sub x.1 x.2) glValue
    (fun x => (wordOf x.1 - wordOf x.2).toNat) := by
  rintro ⟨a, b⟩ y h
  dsimp only at h ⊢
  rw [gl_sub_eq, Result.ok.injEq] at h
  subst h
  exact (glSub_exact a b).val

theorem gl_sub_complete : Complete (fun _ => True) (fun x : field.Gl × field.Gl => field.gl_sub x.1 x.2) glValue
    (fun x => (wordOf x.1 - wordOf x.2).toNat) :=
  complete_of gl_sub_sound ⟨(0#u64, 0#u64), trivial⟩ fun x _ => ⟨_, gl_sub_eq x.1 x.2⟩

theorem gl_sub_field (a b c : field.Gl) (h : field.gl_sub a b = ok c) : glOf c = glOf a - glOf b := by
  rw [gl_sub_eq, Result.ok.injEq] at h
  subst h
  exact glSub_field a b

theorem gl_neg_sound : Sound field.gl_neg glValue (fun x => (-wordOf x).toNat) := by
  intro a y h
  rw [gl_neg_eq, Result.ok.injEq] at h
  subst h
  exact (glNeg_exact a).val

theorem gl_neg_complete : Complete (fun _ => True) field.gl_neg glValue (fun x => (-wordOf x).toNat) :=
  complete_of gl_neg_sound ⟨0#u64, trivial⟩ fun a _ => ⟨_, gl_neg_eq a⟩

theorem gl_neg_field (a c : field.Gl) (h : field.gl_neg a = ok c) : glOf c = -glOf a := by
  rw [gl_neg_eq, Result.ok.injEq] at h
  subst h
  rw [glSub_field, glOf_zero, zero_sub]

theorem gl_mul_sound : Sound (fun x : field.Gl × field.Gl => field.gl_mul x.1 x.2) glValue
    (fun x => (wordOf x.1 * wordOf x.2).toNat) := by
  rintro ⟨a, b⟩ y h
  dsimp only at h ⊢
  rw [gl_mul_eq, Result.ok.injEq] at h
  subst h
  exact (glMul_exact a b).val

theorem gl_mul_complete : Complete (fun _ => True) (fun x : field.Gl × field.Gl => field.gl_mul x.1 x.2) glValue
    (fun x => (wordOf x.1 * wordOf x.2).toNat) :=
  complete_of gl_mul_sound ⟨(0#u64, 0#u64), trivial⟩ fun x _ => ⟨_, gl_mul_eq x.1 x.2⟩

theorem gl_mul_field (a b c : field.Gl) (h : field.gl_mul a b = ok c) : glOf c = glOf a * glOf b := by
  rw [gl_mul_eq, Result.ok.injEq] at h
  subst h
  exact glMul_field a b

theorem gl_pow_sound : Sound (fun x : field.Gl × U64 => field.gl_pow x.1 x.2) glValue
    (fun x => ((wordOf x.1).pow x.2.val).toNat) := by
  rintro ⟨a, e⟩ y h
  dsimp only at h ⊢
  obtain ⟨c, run, exact⟩ := gl_pow_exact e.val a e rfl
  rw [run, Result.ok.injEq] at h
  subst h
  exact exact.val

theorem gl_pow_complete : Complete (fun _ => True) (fun x : field.Gl × U64 => field.gl_pow x.1 x.2) glValue
    (fun x => ((wordOf x.1).pow x.2.val).toNat) :=
  complete_of gl_pow_sound ⟨(0#u64, 0#u64), trivial⟩ fun x _ =>
    let ⟨c, run, _⟩ := gl_pow_exact x.2.val x.1 x.2 rfl
    ⟨c, run⟩

theorem gl_pow_field (a : field.Gl) (e : U64) (c : field.Gl) (h : field.gl_pow a e = ok c) :
    glOf c = glOf a ^ e.val := by
  obtain ⟨c', run, exact⟩ := gl_pow_exact e.val a e rfl
  rw [run, Result.ok.injEq] at h
  subst h
  rw [exact.glOf, cast_pow, cast_wordOf]

theorem gl_inv_sound : Sound field.gl_inv glValue (fun x => (wordOf x).inv.toNat) := by
  intro a y h
  obtain ⟨c, run, exact⟩ := gl_inv_exact a
  rw [run, Result.ok.injEq] at h
  subst h
  exact exact.val

theorem gl_inv_complete : Complete (fun _ => True) field.gl_inv glValue (fun x => (wordOf x).inv.toNat) :=
  complete_of gl_inv_sound ⟨0#u64, trivial⟩ fun a _ =>
    let ⟨c, run, _⟩ := gl_inv_exact a
    ⟨c, run⟩

theorem gl_inv_field (a c : field.Gl) (h : field.gl_inv a = ok c) : glOf c = (glOf a)⁻¹ := by
  obtain ⟨c', run, exact⟩ := gl_inv_exact a
  rw [run, Result.ok.injEq] at h
  subst h
  rw [exact.glOf, Replay.Word.inv, cast_pow, cast_wordOf]
  exact pow_card_sub_two goldilocksModulus three_le_modulus _ ZMod.pow_card_sub_one_eq_one

/-! ## `Ext` -/

/-- The copy's reading of a translated `Ext`: each coordinate mod `p`. -/
def extOf (e : field.Ext) : Replay.Ext := ⟨wordOf e.c0, wordOf e.c1, wordOf e.c2⟩

/-- The copy's words, as `extValue` reads a translated `Ext`. -/
def extWords (x : Replay.Ext) : ℕ × ℕ × ℕ := (x.c0.toNat, x.c1.toNat, x.c2.toNat)

/-- Each coordinate of the translated `c` is the copy's word. -/
def ExtExact (c : field.Ext) (x : Replay.Ext) : Prop := Exact c.c0 x.c0 ∧ Exact c.c1 x.c1 ∧ Exact c.c2 x.c2

theorem ExtExact.val {c : field.Ext} {x : Replay.Ext} (h : ExtExact c x) : extValue c = extWords x := by
  rw [extValue, extWords, h.1.val, h.2.1.val, h.2.2.val]

theorem ExtExact.extOf {c : field.Ext} {x : Replay.Ext} (h : ExtExact c x) : CompressionLeanRust.extOf c = x := by
  obtain ⟨x0, x1, x2⟩ := x
  rw [CompressionLeanRust.extOf, h.1.wordOf, h.2.1.wordOf, h.2.2.wordOf]

theorem wordOf_glAdd (a b : field.Gl) : wordOf (glAdd a b) = wordOf a + wordOf b := (glAdd_exact a b).wordOf

theorem wordOf_glSub (a b : field.Gl) : wordOf (glSub a b) = wordOf a - wordOf b := (glSub_exact a b).wordOf

theorem wordOf_glMul (a b : field.Gl) : wordOf (glMul a b) = wordOf a * wordOf b := (glMul_exact a b).wordOf

def extAdd (a b : field.Ext) : field.Ext := ⟨glAdd a.c0 b.c0, glAdd a.c1 b.c1, glAdd a.c2 b.c2⟩

def extSub (a b : field.Ext) : field.Ext := ⟨glSub a.c0 b.c0, glSub a.c1 b.c1, glSub a.c2 b.c2⟩

def extMul (a b : field.Ext) : field.Ext :=
  let d3 := glAdd (glMul a.c1 b.c2) (glMul a.c2 b.c1)
  let d4 := glMul a.c2 b.c2
  ⟨glAdd (glMul a.c0 b.c0) d3, glAdd (glAdd (glAdd (glMul a.c0 b.c1) (glMul a.c1 b.c0)) d3) d4,
    glAdd (glAdd (glAdd (glMul a.c0 b.c2) (glMul a.c1 b.c1)) (glMul a.c2 b.c0)) d4⟩

def extScale (a : field.Ext) (g : field.Gl) : field.Ext := ⟨glMul a.c0 g, glMul a.c1 g, glMul a.c2 g⟩

theorem ext_of_gl_eq (a : field.Gl) : field.ext_of_gl a = ok ⟨redW a, field.GL_ZERO, field.GL_ZERO⟩ := by
  simp only [field.ext_of_gl, reduce_word_eq, bind_ok]

theorem ext_add_eq (a b : field.Ext) : field.ext_add a b = ok (extAdd a b) := by
  simp only [field.ext_add, gl_add_eq, bind_ok, extAdd]

theorem ext_sub_eq (a b : field.Ext) : field.ext_sub a b = ok (extSub a b) := by
  simp only [field.ext_sub, gl_sub_eq, bind_ok, extSub]

theorem ext_neg_eq (a : field.Ext) : field.ext_neg a = ok (extSub field.EXT_ZERO a) := by
  simp only [field.ext_neg, ext_sub_eq]

theorem ext_mul_eq (a b : field.Ext) : field.ext_mul a b = ok (extMul a b) := by
  simp only [field.ext_mul, gl_add_eq, gl_mul_eq, bind_ok, extMul]

theorem ext_scale_eq (a : field.Ext) (g : field.Gl) : field.ext_scale a g = ok (extScale a g) := by
  simp only [field.ext_scale, gl_mul_eq, bind_ok, extScale]

theorem zero_exact : Exact field.GL_ZERO 0 := by
  unfold Exact field.GL_ZERO
  rfl

theorem extOf_zero : extOf field.EXT_ZERO = 0 := by
  unfold extOf field.EXT_ZERO
  rw [wordOf_zero]
  rfl

theorem extOne_exact : ExtExact field.EXT_ONE 1 := by
  unfold field.EXT_ONE
  exact ⟨one_exact, zero_exact, zero_exact⟩

theorem extOfGl_exact (a : field.Gl) :
    ExtExact ⟨redW a, field.GL_ZERO, field.GL_ZERO⟩ (Replay.Ext.ofGl (wordOf a)) :=
  ⟨toU_red_word a, zero_exact, zero_exact⟩

theorem extAdd_exact (a b : field.Ext) : ExtExact (extAdd a b) (extOf a + extOf b) :=
  ⟨glAdd_exact _ _, glAdd_exact _ _, glAdd_exact _ _⟩

theorem extSub_exact (a b : field.Ext) : ExtExact (extSub a b) (extOf a - extOf b) :=
  ⟨glSub_exact _ _, glSub_exact _ _, glSub_exact _ _⟩

theorem extNeg_exact (a : field.Ext) : ExtExact (extSub field.EXT_ZERO a) (-extOf a) := by
  have h := extSub_exact field.EXT_ZERO a
  rwa [extOf_zero] at h

theorem extMul_exact (a b : field.Ext) : ExtExact (extMul a b) (extOf a * extOf b) := by
  refine ⟨?_, ?_, ?_⟩
  · have h := glAdd_exact (glMul a.c0 b.c0) (glAdd (glMul a.c1 b.c2) (glMul a.c2 b.c1))
    rwa [wordOf_glMul, wordOf_glAdd, wordOf_glMul, wordOf_glMul] at h
  · have h := glAdd_exact (glAdd (glAdd (glMul a.c0 b.c1) (glMul a.c1 b.c0)) (glAdd (glMul a.c1 b.c2)
      (glMul a.c2 b.c1))) (glMul a.c2 b.c2)
    rwa [wordOf_glAdd, wordOf_glAdd, wordOf_glAdd, wordOf_glMul, wordOf_glMul, wordOf_glMul, wordOf_glMul,
      wordOf_glMul] at h
  · have h := glAdd_exact (glAdd (glAdd (glMul a.c0 b.c2) (glMul a.c1 b.c1)) (glMul a.c2 b.c0))
      (glMul a.c2 b.c2)
    rwa [wordOf_glAdd, wordOf_glAdd, wordOf_glMul, wordOf_glMul, wordOf_glMul, wordOf_glMul] at h

theorem extScale_exact (a : field.Ext) (g : field.Gl) : ExtExact (extScale a g) ((extOf a).scale (wordOf g)) :=
  ⟨glMul_exact _ _, glMul_exact _ _, glMul_exact _ _⟩

theorem ext_pow_exact (n : ℕ) : ∀ (a : field.Ext) (e : U64), e.val = n →
    ∃ c, field.ext_pow a e = ok c ∧ ExtExact c ((extOf a).pow n) := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro a e he
    rw [field.ext_pow, Replay.Ext.pow]
    by_cases zero : n = 0
    · rw [ite_eq_left ((u64_eq_zero e).mpr (he.trans zero)), ite_eq_left zero]
      exact ⟨_, rfl, extOne_exact⟩
    · rw [ite_eq_right (fun h => zero (he.symm.trans ((u64_eq_zero e).mp h))), ite_eq_right zero]
      obtain ⟨half, run, exact⟩ := ih (n / 2) (by omega) (extMul a a) (core.num.U64.wrapping_shr e 1#u32)
        (by rw [shr_one_val, he])
      rw [(extMul_exact a a).extOf] at exact
      simp only [ext_mul_eq, lift, bind_ok, run, is_multiple_of_two, he]
      by_cases even : n % 2 = 0
      · rw [show (n % 2 == 0) = true from beq_iff_eq.mpr even, ite_eq_left rfl,
          ite_eq_right (by omega)]
        exact ⟨half, rfl, exact⟩
      · rw [show (n % 2 == 0) = false from beq_eq_false_iff_ne.mpr even, ite_eq_right (by simp),
          ite_eq_left (by omega)]
        refine ⟨_, rfl, ?_⟩
        have h := extMul_exact half a
        rwa [exact.extOf] at h

/-! ## `Ext` is `AdjoinRoot (X^3 - X - 1)` -/

local notation "ι" => algebraMap Gl Cubic
local notation "x₀" => ChallengeField.root (K := Cubic)

/-- A translated `Ext` as an element of `Cubic`: `c0 + c1·x₀ + c2·x₀²`, each coordinate mod `p`. -/
noncomputable def cubicOf (e : field.Ext) : Cubic := ι (glOf e.c0) + ι (glOf e.c1) * x₀ + ι (glOf e.c2) * x₀ ^ 2

theorem cubicOf_ofGl (a : field.Gl) : cubicOf ⟨redW a, field.GL_ZERO, field.GL_ZERO⟩ = ι (glOf a) := by
  simp only [cubicOf, glOf_red, glOf_zero, map_zero, zero_mul, add_zero]

theorem extAdd_cubic (a b : field.Ext) : cubicOf (extAdd a b) = cubicOf a + cubicOf b := by
  simp only [cubicOf, extAdd, glAdd_field, map_add]
  ring

theorem extSub_cubic (a b : field.Ext) : cubicOf (extSub a b) = cubicOf a - cubicOf b := by
  simp only [cubicOf, extSub, glSub_field, map_sub]
  ring

theorem cubicOf_zero : cubicOf field.EXT_ZERO = 0 := by
  unfold field.EXT_ZERO
  simp only [cubicOf, glOf_zero, map_zero, zero_mul, add_zero]

theorem cubicOf_one : cubicOf field.EXT_ONE = 1 := by
  unfold field.EXT_ONE
  simp only [cubicOf, glOf_one, glOf_zero, map_one, map_zero, zero_mul, add_zero]

theorem extMul_cubic (a b : field.Ext) : cubicOf (extMul a b) = cubicOf a * cubicOf b := by
  simp only [cubicOf, extMul, glAdd_field, glMul_field, map_add, map_mul]
  linear_combination (-(ι (glOf a.c1) * ι (glOf b.c2) + ι (glOf a.c2) * ι (glOf b.c1) +
    ι (glOf a.c2) * ι (glOf b.c2) * x₀)) * ChallengeField.cubic (K := Cubic)

theorem extScale_cubic (a : field.Ext) (g : field.Gl) : cubicOf (extScale a g) = cubicOf a * ι (glOf g) := by
  simp only [cubicOf, extScale, glMul_field, map_mul]
  ring

theorem ext_pow_cubic (n : ℕ) : ∀ (a : field.Ext) (e : U64) (c : field.Ext), e.val = n →
    field.ext_pow a e = ok c → cubicOf c = cubicOf a ^ n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro a e c he h
    rw [field.ext_pow] at h
    by_cases zero : n = 0
    · rw [ite_eq_left ((u64_eq_zero e).mpr (he.trans zero)), Result.ok.injEq] at h
      subst h
      rw [zero, cubicOf_one, pow_zero]
    · rw [ite_eq_right (fun h => zero (he.symm.trans ((u64_eq_zero e).mp h)))] at h
      obtain ⟨half, run, -⟩ := ext_pow_exact (n / 2) (extMul a a) (core.num.U64.wrapping_shr e 1#u32)
        (by rw [shr_one_val, he])
      have hhalf := ih (n / 2) (by omega) (extMul a a) _ half (by rw [shr_one_val, he]) run
      rw [extMul_cubic, ← pow_two, ← pow_mul] at hhalf
      simp only [ext_mul_eq, lift, bind_ok, run, is_multiple_of_two, he] at h
      have split := Nat.div_add_mod n 2
      by_cases even : n % 2 = 0
      · rw [show (n % 2 == 0) = true from beq_iff_eq.mpr even, ite_eq_left rfl, Result.ok.injEq] at h
        subst h
        rw [hhalf]
        congr 1
        omega
      · rw [show (n % 2 == 0) = false from beq_eq_false_iff_ne.mpr even, ite_eq_right (by simp),
          Result.ok.injEq] at h
        subst h
        rw [extMul_cubic, hhalf, ← pow_succ]
        congr 1
        omega

/-! ## Each `Ext` function equals its copy and its `AdjoinRoot` operation -/

theorem ext_of_gl_sound : Sound field.ext_of_gl extValue (fun x => extWords (Replay.Ext.ofGl (wordOf x))) := by
  intro a y h
  rw [ext_of_gl_eq, Result.ok.injEq] at h
  subst h
  exact (extOfGl_exact a).val

theorem ext_of_gl_complete :
    Complete (fun _ => True) field.ext_of_gl extValue (fun x => extWords (Replay.Ext.ofGl (wordOf x))) :=
  complete_of ext_of_gl_sound ⟨0#u64, trivial⟩ fun a _ => ⟨_, ext_of_gl_eq a⟩

theorem ext_of_gl_field (a : field.Gl) (c : field.Ext) (h : field.ext_of_gl a = ok c) : cubicOf c = ι (glOf a) := by
  rw [ext_of_gl_eq, Result.ok.injEq] at h
  subst h
  exact cubicOf_ofGl a

theorem ext_add_sound : Sound (fun x : field.Ext × field.Ext => field.ext_add x.1 x.2) extValue
    (fun x => extWords (extOf x.1 + extOf x.2)) := by
  rintro ⟨a, b⟩ y h
  dsimp only at h ⊢
  rw [ext_add_eq, Result.ok.injEq] at h
  subst h
  exact (extAdd_exact a b).val

theorem ext_add_complete : Complete (fun _ => True) (fun x : field.Ext × field.Ext => field.ext_add x.1 x.2)
    extValue (fun x => extWords (extOf x.1 + extOf x.2)) :=
  complete_of ext_add_sound ⟨(field.EXT_ZERO, field.EXT_ZERO), trivial⟩ fun x _ => ⟨_, ext_add_eq x.1 x.2⟩

theorem ext_add_field (a b c : field.Ext) (h : field.ext_add a b = ok c) : cubicOf c = cubicOf a + cubicOf b := by
  rw [ext_add_eq, Result.ok.injEq] at h
  subst h
  exact extAdd_cubic a b

theorem ext_sub_sound : Sound (fun x : field.Ext × field.Ext => field.ext_sub x.1 x.2) extValue
    (fun x => extWords (extOf x.1 - extOf x.2)) := by
  rintro ⟨a, b⟩ y h
  dsimp only at h ⊢
  rw [ext_sub_eq, Result.ok.injEq] at h
  subst h
  exact (extSub_exact a b).val

theorem ext_sub_complete : Complete (fun _ => True) (fun x : field.Ext × field.Ext => field.ext_sub x.1 x.2)
    extValue (fun x => extWords (extOf x.1 - extOf x.2)) :=
  complete_of ext_sub_sound ⟨(field.EXT_ZERO, field.EXT_ZERO), trivial⟩ fun x _ => ⟨_, ext_sub_eq x.1 x.2⟩

theorem ext_sub_field (a b c : field.Ext) (h : field.ext_sub a b = ok c) : cubicOf c = cubicOf a - cubicOf b := by
  rw [ext_sub_eq, Result.ok.injEq] at h
  subst h
  exact extSub_cubic a b

theorem ext_neg_sound : Sound field.ext_neg extValue (fun x => extWords (-extOf x)) := by
  intro a y h
  rw [ext_neg_eq, Result.ok.injEq] at h
  subst h
  exact (extNeg_exact a).val

theorem ext_neg_complete : Complete (fun _ => True) field.ext_neg extValue (fun x => extWords (-extOf x)) :=
  complete_of ext_neg_sound ⟨field.EXT_ZERO, trivial⟩ fun a _ => ⟨_, ext_neg_eq a⟩

theorem ext_neg_field (a c : field.Ext) (h : field.ext_neg a = ok c) : cubicOf c = -cubicOf a := by
  rw [ext_neg_eq, Result.ok.injEq] at h
  subst h
  rw [extSub_cubic, cubicOf_zero, zero_sub]

theorem ext_mul_sound : Sound (fun x : field.Ext × field.Ext => field.ext_mul x.1 x.2) extValue
    (fun x => extWords (extOf x.1 * extOf x.2)) := by
  rintro ⟨a, b⟩ y h
  dsimp only at h ⊢
  rw [ext_mul_eq, Result.ok.injEq] at h
  subst h
  exact (extMul_exact a b).val

theorem ext_mul_complete : Complete (fun _ => True) (fun x : field.Ext × field.Ext => field.ext_mul x.1 x.2)
    extValue (fun x => extWords (extOf x.1 * extOf x.2)) :=
  complete_of ext_mul_sound ⟨(field.EXT_ZERO, field.EXT_ZERO), trivial⟩ fun x _ => ⟨_, ext_mul_eq x.1 x.2⟩

theorem ext_mul_field (a b c : field.Ext) (h : field.ext_mul a b = ok c) : cubicOf c = cubicOf a * cubicOf b := by
  rw [ext_mul_eq, Result.ok.injEq] at h
  subst h
  exact extMul_cubic a b

theorem ext_scale_sound : Sound (fun x : field.Ext × field.Gl => field.ext_scale x.1 x.2) extValue
    (fun x => extWords ((extOf x.1).scale (wordOf x.2))) := by
  rintro ⟨a, g⟩ y h
  dsimp only at h ⊢
  rw [ext_scale_eq, Result.ok.injEq] at h
  subst h
  exact (extScale_exact a g).val

theorem ext_scale_complete : Complete (fun _ => True) (fun x : field.Ext × field.Gl => field.ext_scale x.1 x.2)
    extValue (fun x => extWords ((extOf x.1).scale (wordOf x.2))) :=
  complete_of ext_scale_sound ⟨(field.EXT_ZERO, 0#u64), trivial⟩ fun x _ => ⟨_, ext_scale_eq x.1 x.2⟩

theorem ext_scale_field (a : field.Ext) (g : field.Gl) (c : field.Ext) (h : field.ext_scale a g = ok c) :
    cubicOf c = cubicOf a * ι (glOf g) := by
  rw [ext_scale_eq, Result.ok.injEq] at h
  subst h
  exact extScale_cubic a g

theorem ext_pow_sound : Sound (fun x : field.Ext × U64 => field.ext_pow x.1 x.2) extValue
    (fun x => extWords ((extOf x.1).pow x.2.val)) := by
  rintro ⟨a, e⟩ y h
  dsimp only at h ⊢
  obtain ⟨c, run, exact⟩ := ext_pow_exact e.val a e rfl
  rw [run, Result.ok.injEq] at h
  subst h
  exact exact.val

theorem ext_pow_complete : Complete (fun _ => True) (fun x : field.Ext × U64 => field.ext_pow x.1 x.2) extValue
    (fun x => extWords ((extOf x.1).pow x.2.val)) :=
  complete_of ext_pow_sound ⟨(field.EXT_ZERO, 0#u64), trivial⟩ fun x _ =>
    let ⟨c, run, _⟩ := ext_pow_exact x.2.val x.1 x.2 rfl
    ⟨c, run⟩

theorem ext_pow_field (a : field.Ext) (e : U64) (c : field.Ext) (h : field.ext_pow a e = ok c) :
    cubicOf c = cubicOf a ^ e.val :=
  ext_pow_cubic e.val a e c rfl h

end CompressionLeanRust
