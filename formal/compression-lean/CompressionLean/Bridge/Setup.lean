import CompressionLean.Bridge.Honest
import CompressionLean.Spec.Openings

/-!
Owns the bridge of the setup queries (`Replay.setupCheck`) to the model's
honest fold. Plonky3's two-adic generator of order `8` has order `8`
(`omega_order`, a kernel computation), so the replay's leaf weights are the
inverse DFT of `HonestFold.leaf_formula`. Under the Merkle assumption
(`MerkleBinds`: a proof leaf whose path reaches the key root at an index
below the leaf count holds the honest setup codeword on that index's coset),
each folded value the replay computes
is the model's honest fold at the leaf's query point (`setupCheck_value`).
-/

namespace CompressionLean.Bridge

open CompressionLean.Spec
open CompressionLean.Lib
open Polynomial
open NightstreamFPrime.Spec
open CompressionLean.Replay CompressionLean.Layer1

/-! ## Two-adic generators -/

/-- `n` squarings of `x` mod `p`. -/
def squarings : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => squarings n (x * x % goldilocksModulus)

theorem squarings_cast (n x : ℕ) : ((squarings n x : ℕ) : Gl) = (x : Gl) ^ (2 ^ n) := by
  induction n generalizing x with
  | zero => simp [squarings]
  | succ n ih => rw [squarings, ih, ZMod.natCast_mod, Nat.cast_mul, ← sq, ← pow_mul, pow_succ']

theorem toF_twoAdicGenerator (bits : ℕ) : (twoAdicGenerator bits).toF = twoAdic bits := by
  unfold twoAdicGenerator twoAdic
  have squares : ∀ (l : List ℕ) (g : Word), (l.foldl (fun g _ => g * g) g).toF = g.toF ^ (2 ^ l.length) := by
    intro l
    induction l with
    | nil => intro g; simp
    | cons _ rest ih =>
        intro g
        rw [List.foldl_cons, ih, Word.toF_mul, ← sq, ← pow_mul, List.length_cons, pow_succ']
  rw [squares, List.length_range, Word.toF_ofNat]
  norm_num

theorem generator_pow_31 : (0x185629dcda58878c : Gl) ^ (2 ^ 31) = -1 := by
  have computed : squarings 31 0x185629dcda58878c = goldilocksModulus - 1 := by decide
  have := squarings_cast 31 0x185629dcda58878c
  rw [computed, Nat.cast_sub (by decide), ZMod.natCast_self, Nat.cast_one, zero_sub, Nat.cast_ofNat] at this
  exact this.symm

theorem generator_pow_32 : (0x185629dcda58878c : Gl) ^ (2 ^ 32) = 1 := by
  rw [pow_succ, pow_mul, generator_pow_31]
  norm_num

theorem neg_one_ne_one : (-1 : Gl) ≠ 1 := by
  intro same
  have two : (2 : Gl) = 0 := by linear_combination -same
  have := (ZMod.natCast_eq_zero_iff 2 goldilocksModulus).mp (by exact_mod_cast two)
  have := Nat.le_of_dvd (by norm_num) this
  have := three_le_modulus
  omega

/-- The generator of the coset of a leaf has order `8`. -/
theorem omega_order : orderOf (twoAdic 3) = 2 ^ 3 := by
  apply orderOf_eq_prime_pow (p := 2) (n := 2)
  · rw [twoAdic, ← pow_mul, show 2 ^ (32 - 3) * 2 ^ 2 = 2 ^ 31 by norm_num, generator_pow_31]
    exact neg_one_ne_one
  · rw [twoAdic, ← pow_mul, show 2 ^ (32 - 3) * 2 ^ (2 + 1) = 2 ^ 32 by norm_num, generator_pow_32]

theorem omega_pow_eight : twoAdic 3 ^ 8 = 1 := by
  rw [show (8 : ℕ) = 2 ^ 3 by norm_num, ← omega_order]
  exact pow_orderOf_eq_one _

theorem generator_ne_zero : (0x185629dcda58878c : Gl) ≠ 0 := by
  intro zero
  have := generator_pow_32
  rw [zero, zero_pow (by norm_num)] at this
  exact zero_ne_one this

theorem twoAdic_ne_zero (bits : ℕ) : twoAdic bits ≠ 0 := pow_ne_zero _ generator_ne_zero

/-- Eight steps of order `2^(n+1)` are one step of order `2^(n-2)`. -/
theorem twoAdic_pow_eight (n : ℕ) (low : 2 ≤ n) (high : n + 1 ≤ 32) : twoAdic (n + 1) ^ 8 = twoAdic (n - 2) := by
  rw [twoAdic, twoAdic, ← pow_mul, show (8 : ℕ) = 2 ^ 3 by norm_num, ← pow_add]
  congr 2
  omega

theorem seven_ne_zero : (7 : Gl) ≠ 0 := by
  intro zero
  have := (ZMod.natCast_eq_zero_iff 7 goldilocksModulus).mp (by exact_mod_cast zero)
  have := Nat.le_of_dvd (by norm_num) this
  have : (8 : ℕ) ≤ goldilocksModulus := by decide
  omega

theorem leafY_ne_zero (n i : ℕ) : leafY n i ≠ 0 :=
  mul_ne_zero seven_ne_zero (pow_ne_zero _ (twoAdic_ne_zero _))

theorem leafY_pow_eight (n i : ℕ) (low : 2 ≤ n) (high : n + 1 ≤ 32) : leafY n i ^ 8 = queryPoint n i := by
  rw [leafY, queryPoint, mul_pow, ← pow_mul, mul_comm i 8, pow_mul, twoAdic_pow_eight n low high]

/-! ## Powers and monomials -/

theorem toF_powerFromBits (bits : ℕ) : ∀ (g : Word) (index : ℕ),
    (powerFromBits g index bits).toF = g.toF ^ (index % 2 ^ bits) := by
  induction bits with
  | zero => intro g index; simp [powerFromBits, Nat.mod_one]
  | succ bits ih =>
      intro g index
      rw [powerFromBits, Word.toF_mul, ih, Word.toF_mul, ← sq, ← pow_mul, pow_succ', Nat.mod_mul, pow_add, pow_mul]
      rcases Nat.mod_two_eq_zero_or_one index with even | odd
      · rw [if_neg (by omega), Word.toF_one, even, pow_zero, one_mul]
      · rw [if_pos odd, odd, pow_one]

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Gl K

theorem toK_monomial : ∀ (alpha : List Ext) (u : ℕ),
    (Replay.monomial alpha u).toK (K := K) = Layer1.monomial (alpha.map Ext.toK) u
  | [], u => by simp [Replay.monomial, Layer1.monomial]
  | a :: rest, u => by
      simp only [Replay.monomial, Layer1.monomial, List.map_cons, Ext.toK_mul, toK_monomial rest (u / 2)]
      split_ifs <;> simp

omit [Fintype K] [ChallengeField K] in
theorem powerPoint_map (x : Gl) : ∀ n, (powerPoint x n).map (algebraMap Gl K) = powerPoint (ι x) n := by
  intro n
  induction n generalizing x with
  | zero => rfl
  | succ n ih => simp [powerPoint, ih, map_pow]

omit [Fintype K] [ChallengeField K] in
theorem mle_map (p : List Gl) (t : ℕ → Gl) : ι (mle p t) = mle (p.map ι) fun i => ι (t i) := by
  rw [mle_eq_cubeSum, mle_eq_cubeSum, cubeSum_eq_sum_range, cubeSum_eq_sum_range, List.length_map, map_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [map_mul, eqAt_map]

omit [Fintype K] [ChallengeField K] in
/-- A codeword value in `Gl` read in `K`. -/
theorem univ_map (n : ℕ) (t : ℕ → Gl) (x : Gl) : ι ((univ n t).eval x) = (univ n fun i => ι (t i)).eval (ι x) := by
  rw [univ_eval, univ_eval, mle_map, powerPoint_map]

omit [Algebra Gl K] [Fintype K] [ChallengeField K] in
theorem univ_sum (n m : ℕ) (f : ℕ → ℕ → K) :
    univ n (fun x => ∑ k ∈ Finset.range m, f k x) = ∑ k ∈ Finset.range m, univ n (f k) := by
  induction m with
  | zero =>
      have := univ_mul_left n 0 (fun _ => (0 : K))
      simpa using this
  | succ m ih =>
      simp only [Finset.sum_range_succ]
      rw [univ_add, ih]

/-! ## The leaf check -/

omit [Algebra Gl K] [Fintype K] [ChallengeField K] in
theorem pow_mod_eight {w : K} (root : w ^ 8 = 1) (n : ℕ) : w ^ (n % 8) = w ^ n := by
  conv_rhs => rw [← Nat.div_add_mod n 8, pow_add, pow_mul, root, one_pow, one_mul]

omit [Algebra Gl K] [Fintype K] [ChallengeField K] in
/-- `ω^((8 - u)·v mod 8) = ω^-uv` for `ω^8 = 1`. -/
theorem pow_dft {w : K} (root : w ^ 8 = 1) {u : ℕ} (small : u < 8) (v : ℕ) :
    w ^ ((8 - u) * v % 8) = w⁻¹ ^ (u * v) := by
  have nonzero : w ≠ 0 := fun zero => by rw [zero, zero_pow (by norm_num)] at root; exact zero_ne_one root
  rw [pow_mod_eight root, inv_pow]
  apply eq_inv_of_mul_eq_one_left
  rw [← pow_add, show (8 - u) * v + u * v = 8 * v by rw [← add_mul]; congr 1; omega, pow_mul, root, one_pow]

omit [Fintype K] [ChallengeField K] in
theorem omega_K : (ι (twoAdic 3)) ^ 8 = 1 := by rw [← map_pow, omega_pow_eight, map_one]

/-- The replay's weight of coset value `v` is `leaf_formula`'s. -/
theorem toK_leafWeight (alpha : Array Ext) (y : Gl) (yInverse : Word) (hy : yInverse.toF = y⁻¹) (v : ℕ) :
    (leafWeight alpha yInverse (twoAdicGenerator foldVars) v).toK (K := K) =
      ∑ u ∈ Finset.range (2 ^ foldVars), Layer1.monomial (pointK alpha) u * ((2 ^ foldVars : ℕ) : K)⁻¹ *
        (ι y)⁻¹ ^ u * (ι (twoAdic 3))⁻¹ ^ (u * v) := by
  rw [leafWeight, toK_sumRange]
  apply Finset.sum_congr rfl
  intro u inside
  have small : u < 8 := Finset.mem_range.mp inside
  have eight : (2 : ℕ) ^ foldVars = 8 := rfl
  rw [Ext.toK_scale, Ext.toK_scale, Word.toF_mul, Word.toF_pow, Word.toF_pow, Word.toF_inv, Word.toF_ofNat, hy,
    toF_twoAdicGenerator, map_mul, map_pow, map_pow, map_inv₀, map_inv₀, map_natCast, eight,
    show foldVars = 3 from rfl, pow_dft omega_K small, toK_monomial]
  simp only [pointK, inv_pow]
  ring

omit [Fintype K] [ChallengeField K] in
/-- `ω_8` in `K` has order `8`. -/
theorem omega_K_primitive : ∀ k, 0 < k → k < 2 ^ 3 → (ι (twoAdic 3)) ^ k ≠ 1 := by
  intro k positive small one
  apply pow_ne_one_of_lt_orderOf (x := twoAdic 3) (n := k) (by omega) (by rw [omega_order]; exact small)
  apply (algebraMap Gl K).injective
  rw [map_pow, one, map_one]

omit [Fintype K] [ChallengeField K] in
theorem eight_ne_zero : ((2 ^ 3 : ℕ) : K) ≠ 0 := by
  rw [← map_natCast (algebraMap Gl K)]
  refine (map_ne_zero (algebraMap Gl K)).mpr ?_
  intro zero
  have := (ZMod.natCast_eq_zero_iff (2 ^ 3) goldilocksModulus).mp (by exact_mod_cast zero)
  have := Nat.le_of_dvd (by norm_num) this
  have : (9 : ℕ) ≤ goldilocksModulus := by decide
  omega

/-- One group's value at a leaf, `Σ_v w_v·Σ_k weight_k·Ô_{offset+k}(y·ω^v)`, is
the honest fold of the group at the leaf's query point. -/
theorem toK_groupValue (rel : Replay.Relation) (S : Setup Gl) (leaf : Leaf) (index : ℕ)
    (codeword : ∀ o v, o < rel.kappa + runOracles → v < 2 ^ foldVars →
      (leaf.values[o * 2 ^ foldVars + v]!).toF =
        (univ rel.setupVars (setupOracle rel S o)).eval (leafY rel.setupVars index * twoAdic 3 ^ v))
    (alpha : Array Ext) (alphaSize : alpha.size = foldVars) (yInverse : Word)
    (hy : yInverse.toF = (leafY rel.setupVars index)⁻¹) (weights : Array Ext) (offset : ℕ)
    (fit : offset + weights.size ≤ rel.kappa + runOracles) (G : ℕ → K)
    (group : ∀ x, G x = ∑ k ∈ Finset.range weights.size, (weights[k]!).toK * ι (setupOracle rel S (offset + k) x))
    (setup : 3 ≤ rel.setupVars) (high : rel.setupVars + 1 ≤ 32) :
    (sumRange (2 ^ foldVars) fun v => leafWeight alpha yInverse (twoAdicGenerator foldVars) v *
        sumRange weights.size fun k => (weights[k]!).scale leaf.values[(offset + k) * 2 ^ foldVars + v]!).toK
        (K := K) =
      (univ (rel.setupVars - 3) (fold (pointK alpha) G)).eval (ι (queryPoint rel.setupVars index)) := by
  have inner : ∀ v, v < 2 ^ foldVars →
      (sumRange weights.size fun k => (weights[k]!).scale leaf.values[(offset + k) * 2 ^ foldVars + v]!).toK
          (K := K) =
        (univ rel.setupVars G).eval (ι (leafY rel.setupVars index) * ι (twoAdic 3) ^ v) := by
    intro v small
    rw [toK_sumRange, show G = fun x => ∑ k ∈ Finset.range weights.size,
        (weights[k]!).toK * ι (setupOracle rel S (offset + k) x) from funext group, univ_sum, eval_finsetSum]
    apply Finset.sum_congr rfl
    intro k inside
    rw [Ext.toK_scale, codeword _ _ (by have := Finset.mem_range.mp inside; omega) small, univ_map,
      univ_mul_left, eval_mul, eval_C, map_mul, map_pow]
  have length : (pointK (K := K) alpha).length = 3 := by rw [pointK_length, alphaSize]; rfl
  have formula := leaf_formula (rel.setupVars - 3) G (pointK alpha) (ι (leafY rel.setupVars index))
    (ι (twoAdic 3)) ((map_ne_zero (algebraMap Gl K)).mpr (leafY_ne_zero _ _)) (by rw [length]; exact omega_K)
    (by rw [length]; exact omega_K_primitive) (by rw [length]; exact eight_ne_zero)
  have target : (ι (leafY rel.setupVars index)) ^ 2 ^ (pointK (K := K) alpha).length =
      ι (queryPoint rel.setupVars index) := by
    rw [length, ← map_pow, show (2 : ℕ) ^ 3 = 8 by norm_num, leafY_pow_eight _ _ (by omega) high]
  have total : rel.setupVars - 3 + (pointK (K := K) alpha).length = rel.setupVars := by rw [length]; omega
  rw [target, total, length] at formula
  rw [← formula, toK_sumRange]
  apply Finset.sum_congr rfl
  intro v small
  rw [Ext.toK_mul, toK_leafWeight alpha _ yInverse hy v, inner v (Finset.mem_range.mp small)]
  rfl

theorem toF_yInverse (n index : ℕ) (inside : index < 2 ^ (n - 2)) (low : 2 ≤ n) :
    (powerFromBits (twoAdicGenerator (n + 1)).inv index (n + 1 - foldVars) * generator.inv).toF = (leafY n index)⁻¹ := by
  have seven : generator.toF = 7 := Word.toF_lit 7
  rw [Word.toF_mul, toF_powerFromBits, Word.toF_inv, Word.toF_inv, toF_twoAdicGenerator, seven,
    show n + 1 - foldVars = n - 2 by simp only [foldVars]; omega, Nat.mod_eq_of_lt inside, leafY, mul_inv, inv_pow,
    mul_comm]

theorem two_groups (a b : Array Ext) (f : Array Ext × ℕ → Ext) :
    (#[a, b].zip (#[a, b].foldl (fun offsets weights => offsets.push (offsets.back! + weights.size)) #[0])).map f =
      #[f (a, 0), f (b, a.size)] := by
  simp

/-- The replay's two folded values at a queried leaf are the honest folds of
the key group and of the run group at the leaf's query point. -/
theorem setupCheck_value (rel : Replay.Relation) (S : Setup Gl) {leaves : Array Leaf} (merkle : MerkleBinds rel S leaves)
    {index : ℕ} (inside : index < 2 ^ (rel.setupVars - 2)) {leaf : Leaf} (member : leaf ∈ leaves.toList) (lam mu : Ext)
    {alpha : Array Ext} (alphaSize : alpha.size = foldVars) (setup : 3 ≤ rel.setupVars) (high : rel.setupVars + 1 ≤ 32)
    {vals : Array Ext}
    (ok : setupCheck rel.key.root rel.setupVars index leaf
      #[(Array.range rel.kappa).map (Mixing.commitment ⟨lam, rel.kappa, rel.matrices⟩),
        (Array.range runOracles).map mu.pow] alpha = .ok vals) :
    vals.size = 2 ∧
      (vals[0]!).toK (K := K) = (univ (rel.setupVars - 3)
        (fold (pointK alpha) (keyGroup (shapeOf rel) S lam.toK))).eval (ι (queryPoint rel.setupVars index)) ∧
      (vals[1]!).toK (K := K) = (univ (rel.setupVars - 3)
        (fold (pointK alpha) (runGroup S mu.toK))).eval (ι (queryPoint rel.setupVars index)) := by
  unfold setupCheck at ok
  obtain ⟨_, path, ok⟩ := Except.bind_ok ok
  have codeword := merkle leaf member index inside (Except.ensure_ok path)
  have yInverse := toF_yInverse rel.setupVars index inside (by omega)
  dsimp only at ok
  rw [two_groups] at ok
  obtain rfl := (Except.pure_eq_ok ok).symm
  refine ⟨rfl, ?_, ?_⟩
  · rw [getElem!_pos _ 0 (by simp)]
    apply toK_groupValue rel S leaf index codeword alpha alphaSize _ yInverse _ 0 (by simp) _ _ setup high
    intro x
    unfold keyGroup
    rw [Array.size_map, Array.size_range]
    apply Finset.sum_congr rfl
    intro k inside
    have lt : k < rel.kappa := Finset.mem_range.mp inside
    rw [getElem!_map_range _ _ lt, toK_commitment, setupOracle, zero_add, if_pos lt]
  · rw [getElem!_pos _ 1 (by simp)]
    apply toK_groupValue rel S leaf index codeword alpha alphaSize _ yInverse _ _ (by simp [runOracles]) _ _ setup high
    intro x
    rw [Array.size_map, Array.size_range, Array.size_map, Array.size_range]
    simp only [runGroup, rowColumn, slotColumn, coefColumn, blockColumn, runOracles, Finset.sum_range_succ,
      Finset.sum_range_zero, zero_add]
    rw [getElem!_map_range _ _ (by norm_num), getElem!_map_range _ _ (by norm_num),
      getElem!_map_range _ _ (by norm_num), getElem!_map_range _ _ (by norm_num)]
    simp only [setupOracle, Ext.toK_pow, pow_zero, pow_one, lt_self_iff_false, if_false, if_true,
      show ¬ (rel.kappa + 1 < rel.kappa) by omega, show ¬ (rel.kappa + 2 < rel.kappa) by omega,
      show ¬ (rel.kappa + 3 < rel.kappa) by omega, show rel.kappa + 1 ≠ rel.kappa by omega,
      show rel.kappa + 2 ≠ rel.kappa by omega, show rel.kappa + 2 ≠ rel.kappa + 1 by omega,
      show rel.kappa + 3 ≠ rel.kappa by omega, show rel.kappa + 3 ≠ rel.kappa + 1 by omega,
      show rel.kappa + 3 ≠ rel.kappa + 2 by omega, add_zero, map_natCast, apply_ite (algebraMap Gl K), map_zero]
    ring

end CompressionLean.Bridge
