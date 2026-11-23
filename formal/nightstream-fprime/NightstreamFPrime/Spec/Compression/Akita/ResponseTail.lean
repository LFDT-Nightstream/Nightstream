import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Series
import NightstreamFPrime.Spec.RandomOracle
import NightstreamFPrime.Spec.Compression.Akita.ChallengeFamily
import NightstreamFPrime.Spec.Compression.Akita.Euclid
import NightstreamFPrime.Spec.Compression.Akita.Prover

/-! The tail bound of an honest fold response of Akita (ePrint 2026/1983, revision of 2026-11-23,
Appendix K: Definition K.1, Lemmas K.2 and K.3, and Corollary K.10 with (313)), and the sizing
rule of the Akita code (revision `d50f700f`, `crates/akita-params/src/sis/fold_linf_cap.rs`).

- `mass` (`Spec/RandomOracle.lean`) is the probability of a set under the uniform distribution.
  `mass_exists_le` is the union bound, `mass_preimage` states that a bijection keeps `mass`, and
  `mass_eq_expect_mass` averages `mass` over a family of bijections.
- `rademacher_tail` is Akita Lemma K.2: `Pr[|X| > t] ≤ 2·exp(−t²/(2V))` for the Rademacher sum
  `X = Σ_j ε_j·w_j` with `Σ_j w_j² ≤ V`. `expect_exp_le` is the moment bound of its proof.
- `signed ε c` is `c` with the sign `ε j` on coefficient `j`, and `finSigns` extends a sign vector
  on `Fin n` by `false`. `Shifts φ` states that `φ(Y^j)·x` has at most the coefficient magnitude
  of `x`. `shifts_id` and `shifts_embed` give it for the identity and for `ι` of Akita (38).
- `coordinate_tail` is (302) of Akita Lemma K.3 for one coefficient of `z = Σ_i φ(c_i)·s_i` with
  uniform signs on the challenges, and `response_tail` is (303) with `V = |Q|·κ_2²·σ_∞²`.
- `SignFair decode` states that the sign flips of a challenge act on the slots of its decoder,
  and `signFair_signed` is an instance. `decoded_tail` is the bound of `response_tail` for the
  challenges that such decoders read from a uniform view.
- `normSq_of_mem` is `‖c‖_2² = n₁ + 4·n₂` for the members of `SparseFamily`
  (`challenge_l2_sq_max` of the code), and `encodable_of_magnitude` gives `Enc` of Akita (147)
  from a bound on each coefficient magnitude.
- `sized_tail` is the sizing rule of the code: for `16·N ≤ 7·2^k`, `71·k ≤ 100·ℓ` and
  `2·V·ℓ ≤ T²`, the bound `2N·exp(−T²/(2V))` of (303) is at most `1 − p = 7/8` for `p = 1/8`.
  `two_pow_le_exp` is the bound `ln 2 ≤ 71/100` of the code. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass mass_mono)

section Mass

theorem mass_le_expect {Ω : Type} [Fintype Ω] (S : Set Ω) (f : Ω → ℝ)
    (nonnegative : ∀ ω, 0 ≤ f ω) (cover : ∀ ω ∈ S, 1 ≤ f ω) : mass S ≤ 𝔼 ω, f ω := by
  unfold mass
  refine Finset.expect_le_expect fun ω _ => ?_
  split_ifs with inside
  · exact cover ω inside
  · exact nonnegative ω

theorem mass_preimage {Ω Ω' : Type} [Fintype Ω] [Fintype Ω'] (e : Ω ≃ Ω') (S : Set Ω') :
    mass (e ⁻¹' S) = mass S :=
  Fintype.expect_equiv e _ _ fun _ => rfl

theorem mass_exists_le {Ω Index : Type} [Fintype Ω] (s : Finset Index) (P : Index → Ω → Prop) :
    mass {ω | ∃ i ∈ s, P i ω} ≤ ∑ i ∈ s, mass {ω | P i ω} := by
  unfold mass
  rw [← Finset.expect_sum_comm]
  refine Finset.expect_le_expect fun ω _ => ?_
  split_ifs with inside
  · obtain ⟨i, member, holds⟩ := inside
    refine le_trans ?_ (Finset.single_le_sum (fun j _ => ?_) member)
    · split_ifs with member'
      · exact le_rfl
      · exact absurd holds member'
    · split_ifs <;> norm_num
  · exact Finset.sum_nonneg fun j _ => by split_ifs <;> norm_num

theorem mass_eq_expect_mass {Ω Ω' : Type} [Fintype Ω] [Fintype Ω'] [Nonempty Ω']
    (move : Ω' → Ω ≃ Ω) (S : Set Ω) : mass S = 𝔼 ω, mass {e : Ω' | move e ω ∈ S} := by
  calc mass S = 𝔼 e : Ω', mass (move e ⁻¹' S) := by
        simp only [mass_preimage, Fintype.expect_const]
    _ = 𝔼 ω, mass {e : Ω' | move e ω ∈ S} := by
        unfold mass
        exact Finset.expect_comm _ _ _

end Mass

section Rademacher

theorem expect_prod_bool {ι : Type} [Fintype ι] [DecidableEq ι] (f : ι → Bool → ℝ) :
    𝔼 ε : ι → Bool, ∏ j, f j (ε j) = ∏ j, 𝔼 b, f j b := by
  rw [Fintype.expect_eq_sum_div_card, ← Fintype.prod_sum, Fintype.card_fun, Fintype.card_bool]
  simp only [Fintype.expect_eq_sum_div_card, Fintype.card_bool, Finset.prod_div_distrib,
    Finset.prod_const, Finset.card_univ]
  push_cast
  rfl

/-- The moment bound of the proof of Akita Lemma K.2: `E[exp(λ·X)] ≤ exp(λ²·V/2)`. -/
theorem expect_exp_le {ι : Type} [Fintype ι] [DecidableEq ι] (w : ι → ℝ) {V : ℝ}
    (proxy : ∑ j, w j ^ 2 ≤ V) (t : ℝ) :
    𝔼 ε : ι → Bool, Real.exp (t * ∑ j, if ε j then -w j else w j) ≤
      Real.exp (t ^ 2 * V / 2) := by
  have product (ε : ι → Bool) : Real.exp (t * ∑ j, if ε j then -w j else w j) =
      ∏ j, Real.exp (t * if ε j then -w j else w j) := by
    rw [Finset.mul_sum, Real.exp_sum]
  have each (j : ι) :
      𝔼 b : Bool, Real.exp (t * if b then -w j else w j) = Real.cosh (t * w j) := by
    rw [Fintype.expect_eq_sum_div_card, Fintype.sum_bool, Fintype.card_bool, Real.cosh_eq]
    simp only [Bool.false_eq_true, ↓reduceIte]
    push_cast
    ring_nf
  simp only [product]
  rw [expect_prod_bool (fun j b => Real.exp (t * if b then -w j else w j))]
  simp only [each]
  calc ∏ j, Real.cosh (t * w j) ≤ ∏ j, Real.exp ((t * w j) ^ 2 / 2) :=
        Finset.prod_le_prod₀ (fun j _ => (Real.cosh_pos _).le)
          fun j _ => Real.cosh_le_exp_half_sq _
    _ = Real.exp (t ^ 2 * (∑ j, w j ^ 2) / 2) := by
        rw [← Real.exp_sum, Finset.mul_sum, Finset.sum_div]
        exact congrArg Real.exp (Finset.sum_congr rfl fun j _ => by ring)
    _ ≤ Real.exp (t ^ 2 * V / 2) := Real.exp_le_exp.mpr (by gcongr)

/-- Akita Lemma K.2 (Hoeffding bound for Rademacher sums): `Pr[|X| > t] ≤ 2·exp(−t²/(2V))` for
`X = Σ_j ε_j·w_j` with uniform signs `ε_j` and `Σ_j w_j² ≤ V`. -/
theorem rademacher_tail {ι : Type} [Fintype ι] [DecidableEq ι] (w : ι → ℝ) {t V : ℝ}
    (nonnegative : 0 ≤ t) (positive : 0 < V) (proxy : ∑ j, w j ^ 2 ≤ V) :
    mass {ε : ι → Bool | t < |∑ j, if ε j then -w j else w j|} ≤
      2 * Real.exp (-(t ^ 2) / (2 * V)) := by
  have scale : 0 ≤ t / V := div_nonneg nonnegative positive.le
  have upper := expect_exp_le w proxy (t / V)
  have lower := expect_exp_le w proxy (-(t / V))
  refine (mass_le_expect _ (fun ε =>
    Real.exp (t / V * ∑ j, if ε j then -w j else w j) * Real.exp (-(t / V * t)) +
      Real.exp (-(t / V) * ∑ j, if ε j then -w j else w j) * Real.exp (-(t / V * t)))
    (fun ε => by positivity) fun ε large => ?_).trans ?_
  · rcases lt_abs.mp (show t < |_| from large) with above | below
    · refine le_add_of_le_of_nonneg ?_ (by positivity)
      rw [← Real.exp_add]
      exact Real.one_le_exp (by nlinarith)
    · refine le_add_of_nonneg_of_le (by positivity) ?_
      rw [← Real.exp_add]
      exact Real.one_le_exp (by nlinarith)
  · rw [Finset.expect_add_distrib, ← Finset.expect_mul, ← Finset.expect_mul]
    have weight := (Real.exp_pos (-(t / V * t))).le
    calc _ ≤ Real.exp ((t / V) ^ 2 * V / 2) * Real.exp (-(t / V * t)) +
          Real.exp ((-(t / V)) ^ 2 * V / 2) * Real.exp (-(t / V * t)) :=
          add_le_add (mul_le_mul_of_nonneg_right upper weight)
            (mul_le_mul_of_nonneg_right lower weight)
      _ = 2 * Real.exp (-(t ^ 2) / (2 * V)) := by
          rw [neg_sq, ← two_mul, ← Real.exp_add]
          congr 2
          field_simp
          ring

end Rademacher

section Response

variable {d dC : ℕ} [NeZero d] [NeZero dC]

/-- The sign `ε j` for `j < n`, and `false` for each other `j`. -/
def finSigns {n : ℕ} (ε : Fin n → Bool) (j : ℕ) : Bool := if h : j < n then ε ⟨j, h⟩ else false

/-- The element `c` with the sign `ε j` on coefficient `j`: `−c_j` for `true`. The Rademacher
signs of Akita Lemma K.3 act on a challenge by this map. -/
noncomputable def signed (ε : ℕ → Bool) (c : Rq d) : Rq d :=
  ofCoeffs fun j => if ε j then -coeff c j else coeff c j

theorem coeff_signed (ε : ℕ → Bool) (c : Rq d) {j : ℕ} (small : j < d) :
    coeff (signed ε c) j = if ε j then -coeff c j else coeff c j :=
  coeff_ofCoeffs _ small

theorem signed_zero (ε : ℕ → Bool) : signed ε (0 : Rq d) = 0 :=
  ext_coeff fun j small => by rw [coeff_signed _ _ small]; simp

theorem normSq_signed (ε : ℕ → Bool) (c : Rq d) : normSq (signed ε c) = normSq c :=
  Finset.sum_congr rfl fun j member => by
    rw [coeff_signed _ _ (Finset.mem_range.mp member)]
    split <;> simp [magnitude_neg]

/-- `‖φ(Y^j)·x‖_∞ ≤ ‖x‖_∞` for each `j` and `x`. The identity and `ι` of Akita (38) satisfy it
(`shifts_id`, `shifts_embed`). -/
def Shifts (φ : Rq dC →ₐ[Zq] Rq d) : Prop :=
  ∀ (j : ℕ) (x : Rq d), normInf (φ (AdjoinRoot.root (modulus dC) ^ j) * x) ≤ normInf x

theorem shifts_id : Shifts (AlgHom.id Zq (Rq d)) :=
  fun j x => normInf_root_pow_mul_le j x

omit [NeZero dC] in
theorem shifts_embed {stride : ℕ} (tower : stride * dC = d) : Shifts (embed stride tower) :=
  fun j x => by
    rw [map_pow, embed_root, ← pow_mul]
    exact normInf_root_pow_mul_le _ x

theorem coeff_signed_mul (φ : Rq dC →ₐ[Zq] Rq d) (ε : ℕ → Bool) (c : Rq dC) (x : Rq d)
    (k : ℕ) : coeff (φ (signed ε c) * x) k = ∑ j ∈ Finset.range dC, (if ε j then -1 else 1) *
      (coeff c j * coeff (φ (AdjoinRoot.root (modulus dC) ^ j) * x) k) := by
  rw [signed, ofCoeffs_eq_sum, map_sum, Finset.sum_mul, coeff_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [map_smul, smul_mul_assoc, coeff_smul]
  split <;> ring

variable {B : ℕ} {ι : Type}

/-- The weight of the sign of coefficient `j` of `c_i` in coefficient `k` of entry `col` of
`Σ_i φ(c_i)·s_i`. Its square is the term `|c_{i,j}|²·s_{i,k⊖j}²` of (301) of Akita Lemma K.3. -/
noncomputable def signWeight (φ : Rq dC →ₐ[Zq] Rq d) (c : Fin B → Rq dC)
    (s : Fin B → ι → Rq d) (col : ι) (k : ℕ) (p : Fin B × Fin dC) : ℤ :=
  (coeff (c p.1) p.2).valMinAbs *
    (coeff (φ (AdjoinRoot.root (modulus dC) ^ (p.2 : ℕ)) * s p.1 col) k).valMinAbs

theorem coeff_response (φ : Rq dC →ₐ[Zq] Rq d) (c : Fin B → Rq dC) (s : Fin B → ι → Rq d)
    (ε : Fin B → Fin dC → Bool) (col : ι) (k : ℕ) :
    coeff ((∑ i, φ (signed (finSigns (ε i)) (c i)) • s i) col) k =
      ((∑ p : Fin B × Fin dC, if ε p.1 p.2 then -signWeight φ c s col k p
        else signWeight φ c s col k p : ℤ) : Zq) := by
  rw [Finset.sum_apply, coeff_sum]
  simp only [Pi.smul_apply, smul_eq_mul, coeff_signed_mul]
  rw [Int.cast_sum, ← Finset.univ_product_univ, Finset.sum_product]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_range]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [finSigns, Fin.is_lt, ↓reduceDIte, Fin.eta, signWeight]
  split <;> simp [Int.cast_mul, ZMod.coe_valMinAbs]

theorem signWeight_sq_le {φ : Rq dC →ₐ[Zq] Rq d} (shifts : Shifts φ) {c : Fin B → Rq dC}
    {s : Fin B → ι → Rq d} {κ β : ℕ} (bounded : ∀ i col, normInf (s i col) ≤ β)
    (energy : ∀ i, normSq (c i) ≤ κ) (col : ι) {k : ℕ} (small : k < d) :
    ∑ p, (signWeight φ c s col k p).natAbs ^ 2 ≤ B * κ * β ^ 2 := by
  have each (p : Fin B × Fin dC) :
      (signWeight φ c s col k p).natAbs ≤ magnitude (coeff (c p.1) p.2) * β := by
    rw [signWeight, Int.natAbs_mul]
    exact Nat.mul_le_mul_left _
      ((normInf_le_iff _ _).mp ((shifts _ _).trans (bounded _ _)) k small)
  calc ∑ p, (signWeight φ c s col k p).natAbs ^ 2
      ≤ ∑ p : Fin B × Fin dC, (magnitude (coeff (c p.1) p.2) * β) ^ 2 :=
        Finset.sum_le_sum fun p _ => Nat.pow_le_pow_left (each p) 2
    _ = ∑ i, normSq (c i) * β ^ 2 := by
        rw [← Finset.univ_product_univ, Finset.sum_product]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [normSq, Finset.sum_mul, Finset.sum_range]
        exact Finset.sum_congr rfl fun j _ => by ring
    _ ≤ ∑ _i : Fin B, κ * β ^ 2 :=
        Finset.sum_le_sum fun i _ => Nat.mul_le_mul_right _ (energy i)
    _ = B * κ * β ^ 2 := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul, mul_assoc]

/-- Akita Lemma K.3, (302), for coefficient `k` of entry `col` of `z = Σ_i φ(c_i)·s_i`, with
uniform signs on the coefficients of the challenges and `V = |Q|·κ_2²·σ_∞²`. -/
theorem coordinate_tail {φ : Rq dC →ₐ[Zq] Rq d} (shifts : Shifts φ) {c : Fin B → Rq dC}
    {s : Fin B → ι → Rq d} {κ β T : ℕ} (bounded : ∀ i col, normInf (s i col) ≤ β)
    (energy : ∀ i, normSq (c i) ≤ κ) (positive : 0 < B * κ * β ^ 2) (col : ι) {k : ℕ}
    (small : k < d) :
    mass {ε : Fin B → Fin dC → Bool | T < magnitude (coeff
      ((∑ i, φ (signed (finSigns (ε i)) (c i)) • s i) col) k)} ≤
      2 * Real.exp (-((T : ℝ) ^ 2) / (2 * (B * κ * β ^ 2))) := by
  have proxy : ∑ p, ((signWeight φ c s col k p : ℤ) : ℝ) ^ 2 ≤ ((B * κ * β ^ 2 : ℕ) : ℝ) := by
    have bound := signWeight_sq_le shifts bounded energy col small
    have cast (p : Fin B × Fin dC) : ((signWeight φ c s col k p : ℤ) : ℝ) ^ 2 =
        (((signWeight φ c s col k p).natAbs ^ 2 : ℕ) : ℝ) := by
      rw [Nat.cast_pow, Nat.cast_natAbs, Int.cast_abs, sq_abs]
    simp only [cast]
    exact_mod_cast bound
  have tail := rademacher_tail (fun p => ((signWeight φ c s col k p : ℤ) : ℝ))
    (Nat.cast_nonneg T) (by exact_mod_cast positive) proxy
  rw [← mass_preimage (Equiv.curry (Fin B) (Fin dC) Bool)]
  refine (mass_mono fun ε large => ?_).trans (tail.trans_eq (by push_cast; ring_nf))
  have large : T < magnitude (coeff ((∑ i, φ (signed (finSigns (Equiv.curry _ _ _ ε i)) (c i)) •
      s i) col) k) := large
  rw [coeff_response] at large
  have exceeds := large.trans_le (magnitude_intCast_le _)
  show (T : ℝ) < |∑ p, if ε p then -((signWeight φ c s col k p : ℤ) : ℝ)
    else ((signWeight φ c s col k p : ℤ) : ℝ)|
  have real : ((∑ p : Fin B × Fin dC, if Equiv.curry _ _ _ ε p.1 p.2 then
      -signWeight φ c s col k p else signWeight φ c s col k p : ℤ) : ℝ) =
      ∑ p, if ε p then -((signWeight φ c s col k p : ℤ) : ℝ)
        else ((signWeight φ c s col k p : ℤ) : ℝ) := by
    push_cast
    rfl
  rw [← real, ← Int.cast_abs, ← Nat.cast_natAbs]
  exact_mod_cast exceeds

/-- Akita Lemma K.3 (dense-source response bound), (303): `Pr[‖z‖_∞ > T] ≤ 2N·exp(−T²/(2V))`
for `z = Σ_i φ(c_i)·s_i` with uniform signs on the coefficients of the challenges, `N` the count
of the coefficients of `z`, `‖s_i‖_∞ ≤ σ_∞`, `‖c_i‖_2² ≤ κ_2²` and `V = |Q|·κ_2²·σ_∞²`. -/
theorem response_tail [Fintype ι] (φ : Rq dC →ₐ[Zq] Rq d) (shifts : Shifts φ)
    (c : Fin B → Rq dC) (s : Fin B → ι → Rq d) {κ β T : ℕ}
    (bounded : ∀ i col, normInf (s i col) ≤ β)
    (energy : ∀ i, normSq (c i) ≤ κ) (positive : 0 < B * κ * β ^ 2) :
    mass {ε : Fin B → Fin dC → Bool | ∃ col, ∃ k < d, T < magnitude (coeff
      ((∑ i, φ (signed (finSigns (ε i)) (c i)) • s i) col) k)} ≤
      2 * (Fintype.card ι * d) * Real.exp (-((T : ℝ) ^ 2) / (2 * (B * κ * β ^ 2))) := by
  refine (mass_mono fun ε inside => ?_).trans ((mass_exists_le (Finset.univ ×ˢ Finset.range d)
    fun (p : ι × ℕ) (ε : Fin B → Fin dC → Bool) => T < magnitude (coeff
      ((∑ i, φ (signed (finSigns (ε i)) (c i)) • s i) p.1) p.2)).trans
    ((Finset.sum_le_sum fun p member => coordinate_tail shifts bounded energy positive p.1
      (Finset.mem_range.mp (Finset.mem_product.mp member).2)).trans_eq ?_))
  · obtain ⟨col, k, small, large⟩ := inside
    exact ⟨(col, k), Finset.mem_product.mpr ⟨Finset.mem_univ _, Finset.mem_range.mpr small⟩,
      large⟩
  · rw [Finset.sum_const, Finset.card_product, Finset.card_univ, Finset.card_range, nsmul_eq_mul]
    push_cast
    ring

/-- The sign flips of a challenge act on the slots of the decoder `decode`: for each sign vector
`ε`, a bijection of the slots changes each decoded challenge `c` to `signed ε c`. A uniform slot
then gives uniform signs to the coefficients of the challenge (Akita Lemma K.3, premise). -/
def SignFair {Slot : Type} (decode : Slot → Option (Rq dC)) : Prop :=
  ∃ flip : (Fin dC → Bool) → Slot ≃ Slot, ∀ ε slot,
    decode (flip ε slot) = (decode slot).map (signed (finSigns ε))

/-- The flip of a sign vector by the signs `ε`. -/
def xorFlip {n : ℕ} (ε : Fin n → Bool) : (Fin n → Bool) ≃ (Fin n → Bool) where
  toFun slot j := xor (ε j) (slot j)
  invFun slot j := xor (ε j) (slot j)
  left_inv slot := funext fun j => by
    show xor (ε j) (xor (ε j) (slot j)) = slot j
    cases ε j <;> cases slot j <;> rfl
  right_inv slot := funext fun j => by
    show xor (ε j) (xor (ε j) (slot j)) = slot j
    cases ε j <;> cases slot j <;> rfl

theorem signFair_signed (c : Rq dC) :
    SignFair fun slot : Fin dC → Bool => some (signed (finSigns slot) c) := by
  refine ⟨xorFlip, fun ε slot => ?_⟩
  simp only [Option.map_some]
  refine congrArg some (ext_coeff fun j small => ?_)
  simp only [coeff_signed _ _ small, finSigns, small, ↓reduceDIte, xorFlip, Equiv.coe_fn_mk]
  cases ε ⟨j, small⟩ <;> cases slot ⟨j, small⟩ <;> simp

theorem getD_map_signed (ε : ℕ → Bool) (o : Option (Rq dC)) :
    (o.map (signed ε)).getD 0 = signed ε (o.getD 0) := by
  cases o <;> simp [signed_zero]

/-- Akita Lemma K.3, (303), for the challenges that sign-fair decoders read from distinct
coordinates of a uniform view. Block `i` reads its challenge from coordinate `block i`, and a
slot that does not decode gives the challenge 0. -/
theorem decoded_tail [Fintype ι] {Slot Coord : Type} [Fintype Slot] [Nonempty Slot]
    [Fintype Coord] [DecidableEq Coord] {κ β T : ℕ} (block : Fin B → Coord)
    (injective : Function.Injective block) (decode : Fin B → Slot → Option (Rq dC))
    (fair : ∀ i, SignFair (decode i))
    (sparse : ∀ i slot c, decode i slot = some c → normSq c ≤ κ) (φ : Rq dC →ₐ[Zq] Rq d)
    (shifts : Shifts φ) (s : Fin B → ι → Rq d) (bounded : ∀ i col, normInf (s i col) ≤ β)
    (positive : 0 < B * κ * β ^ 2) :
    mass {view : Coord → Slot | ∃ col, ∃ k < d, T < magnitude (coeff
      ((∑ i, φ ((decode i (view (block i))).getD 0) • s i) col) k)} ≤
      2 * (Fintype.card ι * d) * Real.exp (-((T : ℝ) ^ 2) / (2 * (B * κ * β ^ 2))) := by
  classical
  choose flip flips using fair
  let owner : Coord → Option (Fin B) := fun x =>
    if h : ∃ i, block i = x then some h.choose else none
  have owner_block (i : Fin B) : owner (block i) = some i := by
    have h : ∃ i', block i' = block i := ⟨i, rfl⟩
    simp only [owner, h, ↓reduceDIte]
    exact congrArg some (injective h.choose_spec)
  let move : (Fin B → Fin dC → Bool) → (Coord → Slot) ≃ (Coord → Slot) := fun ε =>
    Equiv.piCongrRight fun x => (owner x).elim (Equiv.refl Slot) fun i => flip i (ε i)
  have move_block (ε : Fin B → Fin dC → Bool) (view : Coord → Slot) (i : Fin B) :
      move ε view (block i) = flip i (ε i) (view (block i)) := by
    show ((owner (block i)).elim (Equiv.refl Slot) fun i => flip i (ε i)) (view (block i)) = _
    rw [owner_block]
    rfl
  rw [mass_eq_expect_mass move]
  refine (Finset.expect_le_expect fun view _ => ?_).trans_eq (Fintype.expect_const _)
  refine (mass_mono fun ε inside => ?_).trans (response_tail φ shifts
    (fun i => (decode i (view (block i))).getD 0) s bounded (fun i => ?_) positive)
  · obtain ⟨col, k, small, large⟩ := inside
    refine ⟨col, k, small, ?_⟩
    simpa only [move_block, flips, getD_map_signed] using large
  · cases h : decode i (view (block i)) with
    | none => simp [normSq]
    | some c => exact sparse i _ c h

end Response

section Family

variable {d : ℕ} [NeZero d] {ones twos : ℕ} {c : Rq d}

/-- `‖c‖_2² = n₁ + 4·n₂` for each member of the family (`challenge_l2_sq_max` of the Akita code,
`crates/akita-challenges/src/config.rs`). -/
theorem normSq_of_mem (inside : c ∈ SparseFamily d ones twos) :
    normSq c = ones + 4 * twos := by
  obtain ⟨short, rfl, rfl⟩ := inside
  rw [normSq, Finset.card_filter, Finset.card_filter, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i at_ => ?_
  have small := (normInf_le_iff c 2).mp short i (Finset.mem_range.mp at_)
  generalize magnitude (coeff c i) = m at small ⊢
  interval_cases m <;> rfl

/-- A vector with each coefficient magnitude at most `T = (b/2 − 1)·Σ_{u<δ} b^u` is in `Enc` of
Akita (147). -/
theorem encodable_of_magnitude {ι : Type} {b δ : ℕ} {z : ι → Rq d} (two : 2 ≤ b)
    (bound : ∀ col, ∀ k < d,
      magnitude (coeff (z col) k) ≤ (b / 2 - 1) * ∑ u ∈ Finset.range δ, b ^ u) :
    Encodable b δ z := by
  intro col k small
  have size := bound col k small
  have half : 1 ≤ b / 2 := (Nat.le_div_iff_mul_le (by decide)).mpr two
  have cast : (((b / 2 - 1) * ∑ u ∈ Finset.range δ, b ^ u : ℕ) : ℤ) =
      (((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u := by
    push_cast [Nat.cast_sub half]
    rfl
  have total : (0 : ℤ) ≤ ∑ u ∈ Finset.range δ, (b : ℤ) ^ u := by positivity
  have upper : |(coeff (z col) k).valMinAbs| ≤
      (((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u := by
    rw [← Int.natCast_natAbs, ← cast]
    exact_mod_cast size
  obtain ⟨low, high⟩ := abs_le.mp upper
  exact ⟨by linarith, high⟩

end Family

section Sizing

/-- `2^k ≤ e^ℓ` for `71·k ≤ 100·ℓ`, the bound `ln x ≤ ⌈log_2 x⌉·71/100` of `ceil_natural_log`
of the Akita code (`crates/akita-params/src/sis/fold_linf_cap.rs`). -/
theorem two_pow_le_exp {ℓ k : ℕ} (enough : 71 * k ≤ 100 * ℓ) : (2 : ℝ) ^ k ≤ Real.exp ℓ := by
  have log := Real.log_two_lt_d9
  have scaled : (71 : ℝ) * k ≤ 100 * ℓ := by exact_mod_cast enough
  have nonnegative : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  rw [← Real.exp_log (by norm_num : (0 : ℝ) < 2), ← Real.exp_nat_mul]
  exact Real.exp_le_exp.mpr (by nlinarith)

/-- The sizing rule of `fold_linf_cap.rs` of the Akita code, which is Akita (313) with
`ε^enc = 1 − p = 7/8` for `p = 1/8`: for `t*² = 2·V·ln_term` with `ln_term ≥ ln(2N/(1 − p))`,
the bound (303) of Akita Lemma K.3 is at most `7/8`. -/
theorem sized_tail {N k ℓ T V : ℕ} (union : 16 * N ≤ 7 * 2 ^ k) (enough : 71 * k ≤ 100 * ℓ)
    (positive : 0 < V) (sized : 2 * V * ℓ ≤ T ^ 2) :
    2 * (N : ℝ) * Real.exp (-((T : ℝ) ^ 2) / (2 * V)) ≤ 7 / 8 := by
  have power := two_pow_le_exp enough
  have volume : (0 : ℝ) < 2 * V := by positivity
  have exponent : -((T : ℝ) ^ 2) / (2 * V) ≤ -ℓ := by
    rw [div_le_iff₀ volume]
    have : (2 * V * ℓ : ℝ) ≤ (T : ℝ) ^ 2 := by exact_mod_cast sized
    nlinarith
  have tail := Real.exp_le_exp.mpr exponent
  have small : Real.exp (-((T : ℝ) ^ 2) / (2 * V)) * 2 ^ k ≤ 1 := by
    calc _ ≤ Real.exp (-ℓ) * Real.exp ℓ :=
          mul_le_mul tail power (by positivity) (Real.exp_pos _).le
      _ = 1 := by rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  have count : (16 * N : ℝ) ≤ 7 * 2 ^ k := by exact_mod_cast union
  have weight := (Real.exp_pos (-((T : ℝ) ^ 2) / (2 * V))).le
  nlinarith [mul_le_mul_of_nonneg_right count weight]

end Sizing

end NightstreamFPrime.Spec.Compression.Akita
