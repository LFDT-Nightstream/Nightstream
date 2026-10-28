import CompressionLean.Replay.Stages
import CompressionLean.Layer1.CubicField
import CompressionLean.Layer1.ClosedForms

namespace CompressionLean.Bridge

open Polynomial
open NightstreamFPrime.Spec
open CompressionLean.Layer1

abbrev Fp := ZMod goldilocksModulus

instance : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩

theorem three_le_modulus : 3 ≤ goldilocksModulus := by decide

theorem pow_card_sub_two {L : Type} [Field L] (q : ℕ) (three : 3 ≤ q) (a : L)
    (fermat : a ≠ 0 → a ^ (q - 1) = 1) : a ^ (q - 2) = a⁻¹ := by
  by_cases zero : a = 0
  · rw [zero, zero_pow (by omega), inv_zero]
  · have one := fermat zero
    rw [show q - 1 = q - 2 + 1 by omega, pow_succ] at one
    exact eq_inv_of_mul_eq_one_left one

class ChallengeField (K : Type) [Field K] [Algebra Fp K] [Fintype K] where
  root : K
  cubic : root ^ 3 = root + 1
  card : Fintype.card K = goldilocksModulus ^ 3

end CompressionLean.Bridge

namespace CompressionLean.Replay

open Polynomial
open NightstreamFPrime.Spec NightstreamFPrime.Export.NativePoseidon2
open CompressionLean.Bridge CompressionLean.Layer1

namespace Gl

def toF (a : Gl) : Fp := (a.word.toNat : Fp)

theorem toF_eq (a : Gl) : a.toF = ((a.word.denote).val : Fp) := by
  simp [toF, UInt64.denote, Poseidon2.ofNat, Nat.mod_eq_of_lt a.canonical]

theorem add_mk (a b : Gl) : a + b = ⟨add64 a.word b.word, add64_canonical _ _ a.canonical b.canonical⟩ := by
  rfl

theorem mul_mk (a b : Gl) : a * b = ⟨mul64 a.word b.word, mul64_canonical _ _⟩ := by
  rfl

theorem sub_mk (a b : Gl) : a - b = ⟨sub64 a.word b.word, sub64_canonical _ _ a.canonical b.canonical⟩ := by
  rfl

theorem toF_add (a b : Gl) : (a + b).toF = a.toF + b.toF := by
  rw [add_mk, toF_eq, toF_eq a, toF_eq b]
  dsimp only
  rw [add64_denote _ _ a.canonical b.canonical, Fin.val_add, ZMod.natCast_mod, Nat.cast_add]

theorem toF_mul (a b : Gl) : (a * b).toF = a.toF * b.toF := by
  rw [mul_mk, toF_eq, toF_eq a, toF_eq b]
  dsimp only
  rw [mul64_denote _ _ a.canonical b.canonical, Fin.val_mul, ZMod.natCast_mod, Nat.cast_mul]

theorem toF_sub (a b : Gl) : (a - b).toF = a.toF - b.toF := by
  rw [sub_mk, toF_eq, toF_eq a, toF_eq b]
  dsimp only
  rw [sub64_denote _ _ a.canonical b.canonical, Fin.sub_def]
  dsimp only
  rw [ZMod.natCast_mod, Nat.cast_add, Nat.cast_sub (b.word.denote).isLt.le, ZMod.natCast_self]
  ring

theorem toF_ofNat (n : ℕ) : (Gl.ofNat n).toF = (n : Fp) := by
  rw [toF_eq]
  show (((ofNat64 n).denote).val : Fp) = _
  rw [ofNat64_denote]
  simp [Poseidon2.ofNat]

@[simp] theorem toF_lit (n : ℕ) [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n : Gl)).toF = (OfNat.ofNat n : Fp) := by
  rw [show (OfNat.ofNat n : Gl) = Gl.ofNat n from rfl, toF_ofNat]
  rfl

@[simp] theorem toF_zero : (0 : Gl).toF = 0 := by
  rw [show (0 : Gl) = Gl.ofNat 0 from rfl, toF_ofNat, Nat.cast_zero]

@[simp] theorem toF_one : (1 : Gl).toF = 1 := by
  rw [show (1 : Gl) = Gl.ofNat 1 from rfl, toF_ofNat, Nat.cast_one]

theorem toF_neg (a : Gl) : (-a).toF = -a.toF := by
  rw [show -a = 0 - a from rfl, toF_sub, toF_zero, zero_sub]

theorem toF_pow (a : Gl) (e : ℕ) : (a.pow e).toF = a.toF ^ e := by
  induction e using Nat.strong_induction_on generalizing a with
  | _ e ih =>
    rw [Gl.pow]
    by_cases zero : e = 0
    · rw [ite_eq_left zero, zero, toF_one, pow_zero]
    · rw [ite_eq_right zero]
      dsimp only
      have half := ih (e / 2) (by omega) (a * a)
      rw [toF_mul] at half
      have split := Nat.div_add_mod e 2
      split_ifs with odd
      · rw [toF_mul, half, ← pow_two, ← pow_mul, ← pow_succ]
        congr 1
        omega
      · rw [half, ← pow_two, ← pow_mul]
        congr 1
        omega

theorem toF_inv (a : Gl) : a.inv.toF = a.toF⁻¹ := by
  rw [Gl.inv, toF_pow]
  exact pow_card_sub_two goldilocksModulus three_le_modulus _ ZMod.pow_card_sub_one_eq_one

theorem eq_of_word {a b : Gl} (words : a.word = b.word) : a = b := by
  obtain ⟨wa, ca⟩ := a
  obtain ⟨wb, cb⟩ := b
  cases words
  rfl

theorem eq_of_toF {a b : Gl} (equal : a.toF = b.toF) : a = b := by
  have same := (ZMod.natCast_eq_natCast_iff' _ _ _).mp equal
  rw [Nat.mod_eq_of_lt a.canonical, Nat.mod_eq_of_lt b.canonical] at same
  exact eq_of_word (UInt64.toNat_inj.mp same)

theorem eq_of_beq {a b : Gl} (equal : (a == b) = true) : a = b :=
  eq_of_word (beq_iff_eq.mp equal)

end Gl

section Challenge

variable {K : Type} [Field K] [Algebra Fp K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Fp K
local notation "x₀" => ChallengeField.root (K := K)

namespace Ext

def toK (a : Ext) : K := ι a.c0.toF + ι a.c1.toF * x₀ + ι a.c2.toF * x₀ ^ 2

theorem toK_add (a b : Ext) : (a + b).toK (K := K) = a.toK + b.toK := by
  show toK ⟨a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2⟩ = _
  simp only [toK, Gl.toF_add, map_add]
  ring

theorem toK_sub (a b : Ext) : (a - b).toK (K := K) = a.toK - b.toK := by
  show toK ⟨a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2⟩ = _
  simp only [toK, Gl.toF_sub, map_sub]
  ring

theorem toK_mul (a b : Ext) : (a * b).toK (K := K) = a.toK * b.toK := by
  show toK ⟨a.c0 * b.c0 + (a.c1 * b.c2 + a.c2 * b.c1),
    a.c0 * b.c1 + a.c1 * b.c0 + (a.c1 * b.c2 + a.c2 * b.c1) + a.c2 * b.c2,
    a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0 + a.c2 * b.c2⟩ = _
  simp only [toK, Gl.toF_add, Gl.toF_mul, map_add, map_mul]
  linear_combination (-(ι a.c1.toF * ι b.c2.toF + ι a.c2.toF * ι b.c1.toF + ι a.c2.toF * ι b.c2.toF * x₀)) *
    ChallengeField.cubic (K := K)

theorem toK_ofGl (a : Gl) : (Ext.ofGl a).toK (K := K) = ι a.toF := by
  simp [toK, ofGl]

@[simp] theorem toK_zero : (0 : Ext).toK (K := K) = 0 := by
  rw [show (0 : Ext) = Ext.ofGl 0 from rfl, toK_ofGl, Gl.toF_zero, map_zero]

@[simp] theorem toK_one : (1 : Ext).toK (K := K) = 1 := by
  rw [show (1 : Ext) = Ext.ofGl 1 from rfl, toK_ofGl, Gl.toF_one, map_one]

theorem toK_neg (a : Ext) : (-a).toK (K := K) = -a.toK := by
  rw [show -a = 0 - a from rfl, toK_sub, toK_zero, zero_sub]

theorem toK_scale (a : Ext) (g : Gl) : (a.scale g).toK (K := K) = a.toK * ι g.toF := by
  simp only [scale, toK, Gl.toF_mul, map_mul]
  ring

theorem toK_pow (a : Ext) (e : ℕ) : (a.pow e).toK (K := K) = a.toK ^ e := by
  induction e using Nat.strong_induction_on generalizing a with
  | _ e ih =>
    rw [Ext.pow]
    by_cases zero : e = 0
    · rw [ite_eq_left zero, zero, toK_one, pow_zero]
    · rw [ite_eq_right zero]
      dsimp only
      have half := ih (e / 2) (by omega) (a * a)
      rw [toK_mul] at half
      have split := Nat.div_add_mod e 2
      split_ifs with odd
      · rw [toK_mul, half, ← pow_two, ← pow_mul, ← pow_succ]
        congr 1
        omega
      · rw [half, ← pow_two, ← pow_mul]
        congr 1
        omega

theorem toK_inv (a : Ext) : a.inv.toK (K := K) = (a.toK)⁻¹ := by
  rw [Ext.inv, toK_pow]
  apply pow_card_sub_two _ (three_le_modulus.trans (Nat.le_self_pow (by norm_num) _))
  intro nonzero
  have := FiniteField.pow_card_sub_one_eq_one (toK (K := K) a) nonzero
  rwa [ChallengeField.card] at this

theorem toK_basis (c : ℕ) (small : c < 3) : (Ext.basis c).toK (K := K) = x₀ ^ c := by
  rcases (show c = 0 ∨ c = 1 ∨ c = 2 by omega) with rfl | rfl | rfl <;> simp [basis, toK]

theorem minpoly_root : (X ^ 3 - X - 1 : Fp[X]) = minpoly Fp (x₀) := by
  apply minpoly.eq_of_irreducible_of_monic cubic_irreducible
  · simp [ChallengeField.cubic (K := K)]
  · have : (X ^ 3 - X - 1 : Fp[X]) = X ^ 3 - (X + 1) := by ring
    rw [this]
    exact monic_X_pow_sub (by compute_degree!)

theorem coords_of_toK {a b c : Fp} (zero : ι a + ι b * x₀ + ι c * x₀ ^ 2 = 0) : a = 0 ∧ b = 0 ∧ c = 0 := by
  set g : Fp[X] := C a + C b * X + C c * X ^ 2
  by_contra nonzero
  have g_ne : g ≠ 0 := by
    intro gz
    apply nonzero
    have c0 := congrArg (fun p => p.coeff 0) gz
    have c1 := congrArg (fun p => p.coeff 1) gz
    have c2 := congrArg (fun p => p.coeff 2) gz
    simp [g, coeff_X, coeff_X_pow] at c0 c1 c2
    exact ⟨c0, c1, c2⟩
  have root : aeval (x₀) g = 0 := by simpa [g, Algebra.algebraMap_eq_smul_one] using zero
  have degree := minpoly.degree_le_of_ne_zero Fp (x₀) g_ne root
  rw [← minpoly_root] at degree
  have three : (X ^ 3 - X - 1 : Fp[X]).natDegree = 3 := by compute_degree!
  have two : g.natDegree ≤ 2 := by simp only [g]; compute_degree
  have := natDegree_le_natDegree degree
  omega

theorem eq_zero_of_toK {a : Ext} (zero : a.toK (K := K) = 0) : a = 0 := by
  obtain ⟨h0, h1, h2⟩ := coords_of_toK zero
  obtain ⟨a0, a1, a2⟩ := a
  have e0 : a0 = 0 := Gl.eq_of_toF (by rw [h0, Gl.toF_zero])
  have e1 : a1 = 0 := Gl.eq_of_toF (by rw [h1, Gl.toF_zero])
  have e2 : a2 = 0 := Gl.eq_of_toF (by rw [h2, Gl.toF_zero])
  subst e0 e1 e2
  rfl

theorem eq_of_beq {a b : Ext} (equal : (a == b) = true) : a = b := by
  obtain ⟨a0, a1, a2⟩ := a
  obtain ⟨b0, b1, b2⟩ := b
  have parts : (a0.word = b0.word ∧ a1.word = b1.word) ∧ a2.word = b2.word := by
    simpa [BEq.beq, Bool.and_eq_true] using equal
  rw [Gl.eq_of_word parts.1.1, Gl.eq_of_word parts.1.2, Gl.eq_of_word parts.2]

theorem toK_of_beq {a b : Ext} (equal : (a == b) = true) : a.toK (K := K) = b.toK := by
  rw [eq_of_beq equal]

theorem toK_ne_zero {a : Ext} (nonzero : (!a.isZero) = true) : a.toK (K := K) ≠ 0 := by
  intro zero
  have := eq_zero_of_toK zero
  subst this
  revert nonzero
  decide

end Ext

namespace Kx

def toQ (a : Kx) : QuadX K := ⟨a.re.toK, a.im.toK⟩

@[simp] theorem toQ_re (a : Kx) : (a.toQ (K := K)).re = a.re.toK := rfl
@[simp] theorem toQ_im (a : Kx) : (a.toQ (K := K)).im = a.im.toK := rfl

theorem toQ_add (a b : Kx) : (a.add b).toQ (K := K) = a.toQ + b.toQ := by
  ext <;> simp [toQ, add, Ext.toK_add]

theorem toQ_mul (a b : Kx) : (a.mul b).toQ (K := K) = a.toQ * b.toQ := by
  ext
  · simp only [mul, toQ_re, Ext.toK_add, Ext.toK_mul, Ext.toK_scale, QuadraticAlgebra.re_mul, Gl.toF_lit,
      map_ofNat, toQ_im]
    ring
  · simp only [mul, toQ_im, Ext.toK_add, Ext.toK_mul, QuadraticAlgebra.im_mul, toQ_re]
    ring

theorem toQ_scale (a : Kx) (e : Ext) : (a.scale e).toQ (K := K) = a.toQ * algebraMap K (QuadX K) e.toK := by
  ext <;> simp [toQ, scale, Ext.toK_mul]

@[simp] theorem toQ_zero : zero.toQ (K := K) = 0 := by
  ext <;> simp [toQ, zero]

@[simp] theorem toQ_one : one.toQ (K := K) = 1 := by
  ext <;> simp [toQ, one, QuadraticAlgebra.re_one, QuadraticAlgebra.im_one]

theorem toQ_oneMinus (a : Kx) : a.oneMinus.toQ (K := K) = 1 - a.toQ := by
  ext <;> simp [toQ, oneMinus, Ext.toK_sub, QuadraticAlgebra.re_one, QuadraticAlgebra.im_one]

theorem toQ_ofK (value : Gl × Gl) :
    (Kx.ofK value).toQ (K := K) = fromK (K := K) (⟨value.1.toF, value.2.toF⟩ : Quad Fp) := by
  ext <;> simp [toQ, ofK, Ext.toK_ofGl]

end Kx

theorem toK_sumRange (n : ℕ) (f : ℕ → Ext) :
    (sumRange n f).toK (K := K) = ∑ i ∈ Finset.range n, (f i).toK := by
  unfold sumRange
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Ext.toK_add, ih,
      Finset.sum_range_succ]

theorem toK_prodRange (n : ℕ) (f : ℕ → Ext) :
    (prodRange n f).toK (K := K) = ∏ i ∈ Finset.range n, (f i).toK := by
  unfold prodRange
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Ext.toK_mul, ih,
      Finset.prod_range_succ]

theorem toF_prodRangeGl (n : ℕ) (f : ℕ → Gl) :
    (prodRangeGl n f).toF = ∏ i ∈ Finset.range n, (f i).toF := by
  unfold prodRangeGl
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Gl.toF_mul, ih,
      Finset.prod_range_succ]

theorem toQ_sumKx (n : ℕ) (f : ℕ → Kx) :
    (sumKx n f).toQ (K := K) = ∑ i ∈ Finset.range n, (f i).toQ := by
  unfold sumKx
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Kx.toQ_add, ih,
      Finset.sum_range_succ]

end Challenge

end CompressionLean.Replay
