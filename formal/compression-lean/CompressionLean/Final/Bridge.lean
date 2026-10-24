import CompressionLean.Fields
import CompressionLean.Layer1.Weights
import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial
import CompressionLean.Lib.List

/-!
Owns the maps from F′'s executable algebra to the layer-1 model's Mathlib
algebra over `Gl`.

- `toQuad`: an F′ `K` value `c0 + c1·u` as a layer-1 `K` value; it keeps
  `K.add`, `K.sub` and `K.mul` (`u² = 7` on both sides).
- `evaluate_tabulate`: F′'s Boolean-table extension is the layer-1 eq sum,
  with vertex `i` in F′'s little-endian order (`vertexAt`).
- `matrixVectorAt_eq`, `sumRange_eq`: F′'s folds are finite sums.
- `ringSum_eq_of_dvd`: when `Φ81` divides a ring row's residual over `Gl`,
  F′'s ring sum of the row is its target.
- `ringFMul_barRing`: one transformed ring product, coefficient by
  coefficient, is the bilinear sum of the Φ81 kernel.
-/

namespace CompressionLean.Final

open CompressionLean.Lib
open Polynomial NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial

/-! ## The extension field -/

/-- An F′ `K` value as a layer-1 `K` value. -/
def toQuad (k : K) : Layer1.Quad Gl := ⟨k.c0, k.c1⟩

theorem toQuad_add (a b : K) : toQuad (K.add a b) = toQuad a + toQuad b := rfl

theorem toQuad_sub (a b : K) : toQuad (K.sub a b) = toQuad a - toQuad b := rfl

theorem toQuad_zero : toQuad K.zero = 0 := rfl

theorem toQuad_mul (a b : K) : toQuad (K.mul a b) = toQuad a * toQuad b := by
  ext
  · simp only [toQuad, K.mul, QuadraticAlgebra.re_mul]
    rfl
  · simp only [toQuad, K.mul, QuadraticAlgebra.im_mul, zero_mul, add_zero]
    rfl

/-- The extension operations of F′'s Boolean tables. -/
theorem toQuad_ops_add (a b : K) :
    toQuad (ConcreteCarrier.extensionOps.add a b) = toQuad a + toQuad b := rfl

theorem toQuad_ops_mul (a b : K) :
    toQuad (ConcreteCarrier.extensionOps.mul a b) = toQuad a * toQuad b := toQuad_mul a b

theorem toQuad_ops_sub (a b : K) :
    toQuad (ConcreteCarrier.extensionOps.sub a b) = toQuad a - toQuad b := by
  change toQuad (K.add a (K.sub K.zero b)) = _
  rw [toQuad_add, toQuad_sub, toQuad_zero, zero_sub, sub_eq_add_neg]

/-! ## Boolean tables -/

/-- Vertex `i mod 2^n` of the Boolean cube, in F′'s little-endian order. -/
def vertexAt (n i : ℕ) : BooleanVertex n :=
  NumericBooleanDomain.vertex n ⟨i % 2 ^ n, Nat.mod_lt _ (by positivity)⟩

theorem index_vertexAt (n i : ℕ) : NumericBooleanDomain.index (vertexAt n i) = i % 2 ^ n :=
  NumericBooleanDomain.index_vertex n _

theorem vertexAt_index {n : ℕ} (v : BooleanVertex n) : vertexAt n (NumericBooleanDomain.index v) = v := by
  calc vertexAt n (NumericBooleanDomain.index v) =
        NumericBooleanDomain.vertex n ⟨_, NumericBooleanDomain.index_lt_twoPow v⟩ := by
          unfold vertexAt
          congr 1
          exact Fin.ext (Nat.mod_eq_of_lt (NumericBooleanDomain.index_lt_twoPow v))
    _ = v := NumericBooleanDomain.vertex_index v

theorem vertexAt_eq_iff {n i : ℕ} (small : i < 2 ^ n) (v : BooleanVertex n) :
    vertexAt n i = v ↔ NumericBooleanDomain.index v = i := by
  constructor
  · rintro rfl
    rw [index_vertexAt, Nat.mod_eq_of_lt small]
  · rintro rfl
    exact vertexAt_index v

theorem vertexAt_bit (n i : ℕ) (bit : Bool) (small : i < 2 ^ n) :
    vertexAt (n + 1) (2 * i + bit.toNat) = .cons bit (vertexAt n i) := by
  rw [vertexAt_eq_iff (by rw [pow_succ]; cases bit <;> simp <;> omega)]
  have unfold : NumericBooleanDomain.index (BooleanVertex.cons bit (vertexAt n i)) =
      bit.toNat + 2 * NumericBooleanDomain.index (vertexAt n i) := by
    cases bit <;> rfl
  rw [unfold, index_vertexAt, Nat.mod_eq_of_lt small]
  omega

/-- F′'s extension of a tabulated function at a point is the eq-weighted sum
of its values over the cube, low coordinate first. -/
theorem evaluate_tabulate (n : ℕ) (f : BooleanVertex n → K) (point : List K) (length : point.length = n) :
    toQuad ((BooleanTable.tabulate f).evaluateCoordinates ConcreteCarrier.extensionOps point) =
      ∑ i ∈ Finset.range (2 ^ n), Lib.eqAt (point.map toQuad) i * toQuad (f (vertexAt n i)) := by
  induction n generalizing point with
  | zero =>
      obtain rfl : point = [] := List.length_eq_zero_iff.mp length
      simp only [BooleanTable.tabulate, BooleanTable.evaluateCoordinates, pow_zero,
        Finset.sum_range_one, List.map_nil, Lib.eqAt, one_mul]
      rfl
  | succ n ih =>
      obtain ⟨c, rest, rfl⟩ := List.exists_cons_of_length_eq_add_one length
      have restLength : rest.length = n := by simpa using length
      simp only [BooleanTable.tabulate, BooleanTable.evaluateCoordinates]
      rw [toQuad_ops_add, toQuad_ops_mul, toQuad_ops_sub,
        ih (fun tail => f (.cons false tail)) rest restLength,
        ih (fun tail => f (.cons true tail)) rest restLength,
        show 2 ^ (n + 1) = 2 * 2 ^ n by ring, Layer1.sum_range_pairs]
      have even (j : ℕ) (small : j < 2 ^ n) : vertexAt (n + 1) (2 * j) = .cons false (vertexAt n j) := by
        simpa using vertexAt_bit n j false small
      have odd (j : ℕ) (small : j < 2 ^ n) : vertexAt (n + 1) (2 * j + 1) = .cons true (vertexAt n j) := by
        simpa using vertexAt_bit n j true small
      simp only [List.map_cons, Lib.eqAt]
      have halfEven (j : ℕ) : 2 * j / 2 = j := by omega
      have halfOdd (j : ℕ) : (2 * j + 1) / 2 = j := by omega
      have modEven (j : ℕ) : ¬ (2 * j % 2 = 1) := by omega
      have modOdd (j : ℕ) : (2 * j + 1) % 2 = 1 := by omega
      simp only [halfEven, halfOdd, modEven, modOdd, if_false, if_true]
      rw [← Finset.sum_sub_distrib, Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro j member
      have small := Finset.mem_range.mp member
      rw [even j small, odd j small]
      ring

/-! ## Finite folds -/

/-- An F′ field value as a `Gl` value. `Spec.F` and `Gl` are the same type;
`gl` fixes the type that the arithmetic around it uses. -/
abbrev gl (x : F) : Gl := x

/-- F′'s matrix-vector row is the finite sum over the columns. -/
theorem matrixVectorAt_eq {n columns : ℕ} (M : PaperLinearAlgebra.BooleanMatrix F n columns)
    (a : PaperLinearAlgebra.Assignment F columns) (v : BooleanVertex n) :
    gl (PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps M a v) =
      ∑ c : Fin columns, gl (M v c) * gl (a c) := by
  unfold PaperLinearAlgebra.matrixVectorAt canonicalFinIndices
  have := foldl_add_eq_sum (M := Gl) (fun c : Fin columns => gl (M v c) * gl (a c)) (List.ofFn id) 0
  rw [zero_add, List.map_ofFn, List.sum_ofFn] at this
  exact this

/-- F′'s `sumRange` over the base operations is the range sum. -/
theorem sumRange_eq (count : ℕ) (term : ℕ → F) :
    gl (MatrixCoefficientSource.sumRange ConcreteCarrier.baseOps count term) =
      ∑ i ∈ Finset.range count, gl (term i) := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [Finset.sum_range_succ, ← ih]
      rfl

/-! ## Ring rows -/

theorem toPolynomial_add' (a b : RingF) :
    toPolynomial (ringFAdd a b) = toPolynomial a + toPolynomial b := by
  simp only [toPolynomial, ← Finset.sum_add_distrib, ← add_mul, ← C_add]
  rfl

theorem toPolynomial_ringFSum {n : ℕ} (terms : Fin n → RingF) :
    toPolynomial (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.ringFSum terms) =
      ∑ b, toPolynomial (terms b) := by
  induction n with
  | zero => exact toPolynomial_zero
  | succ n ih =>
      rw [NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.ringFSum, toPolynomial_add', ih,
        Fin.sum_univ_succ]

theorem degree_toPolynomial' (value : RingF) : (toPolynomial value).degree < 54 :=
  degree_sum_fin_lt (R := Gl) value

/-- Two F′ ring values that agree modulo `Φ81` are equal. -/
theorem eq_of_dvd {x y : RingF} (holds : modulus ∣ toPolynomial x - toPolynomial y) : x = y := by
  apply toPolynomial_injective
  have small : (toPolynomial x - toPolynomial y).degree < modulus.degree := by
    rw [degree_modulus]
    exact (degree_sub_le _ _).trans_lt (max_lt (degree_toPolynomial' x) (degree_toPolynomial' y))
  exact sub_eq_zero.mp (eq_zero_of_dvd_of_degree_lt holds small)

/-- A ring row of F′ values. When `Φ81` divides `Σ_b W_b·A_b - y` over `Gl`,
F′'s ring sum `Σ_b W_b ⋆ A_b` is `y`. -/
theorem ringSum_eq_of_dvd {n : ℕ} (W A : Fin n → RingF) (y : RingF)
    (holds : modulus ∣ ∑ b, toPolynomial (W b) * toPolynomial (A b) - toPolynomial y) :
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.ringFSum (fun b => ringFMul (W b) (A b)) = y := by
  apply eq_of_dvd
  have reduce : modulus ∣ ∑ b, toPolynomial (W b) * toPolynomial (A b) -
      toPolynomial (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.ringFSum
        (fun b => ringFMul (W b) (A b))) := by
    rw [toPolynomial_ringFSum]
    simp only [toPolynomial_ringFMul_mod]
    rw [← Finset.sum_sub_distrib]
    apply Finset.dvd_sum
    intro b _
    refine ⟨(toPolynomial (W b) * toPolynomial (A b)) /ₘ modulus, ?_⟩
    have := modByMonic_add_div (toPolynomial (W b) * toPolynomial (A b)) modulus
    linear_combination -this
  have := dvd_sub holds reduce
  rwa [sub_sub_sub_cancel_left] at this

/-- Coefficient `p` of F′'s ring sum is the sum of coefficients `p`. -/
theorem ringFSum_apply {n : ℕ} (terms : Fin n → RingF) (p : Fin ringDegree) :
    gl (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.ringFSum terms p) = ∑ b, gl (terms b p) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.ringFSum, Fin.sum_univ_succ, ← ih]
      rfl

theorem coeff_toPolynomial' (v : RingF) (q : ℕ) :
    (toPolynomial v).coeff q = if h : q < ringDegree then gl (v ⟨q, h⟩) else 0 := by
  split_ifs with h
  · exact coeff_toPolynomial v ⟨q, h⟩
  · exact coeff_eq_zero_of_degree_lt ((degree_toPolynomial' v).trans_le (by
      exact_mod_cast (show 54 ≤ q by simp [ringDegree] at h; omega)))

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel in
/-- `bar(u)` as an F′ ring value: coefficient `p` is `Σ_l bar(p, l)·u_l`. -/
def barRing (u : RingF) : RingF := fun p => ∑ l, gl (nativeBarEntry p l) * gl (u l)

theorem toPolynomial_monomial (e : Fin ringDegree) : toPolynomial (ringFMonomial e.val 1) = X ^ e.val := by
  classical
  unfold toPolynomial ringFMonomial
  rw [Finset.sum_eq_single e]
  · rw [if_pos rfl]
    show C (1 : Gl) * X ^ e.val = X ^ e.val
    rw [C_1, one_mul]
  · intro other _ different
    rw [if_neg (fun same => different (Fin.ext same))]
    show C (0 : Gl) * X ^ other.val = 0
    rw [C_0, zero_mul]
  · simp

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel in
theorem toPolynomial_barRing (u : RingF) :
    toPolynomial (barRing u) = ∑ l, C (gl (u l)) * toPolynomial (barBasis l) := by
  ext q
  rw [coeff_toPolynomial', finsetSum_coeff]
  simp only [coeff_C_mul, coeff_toPolynomial']
  split_ifs with h
  · simp only [barRing, barBasis]
    exact Finset.sum_congr rfl fun l _ => mul_comm _ _
  · simp

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel in
/-- One transformed ring product, coefficient `p`: the bilinear sum of the
Φ81 kernel `W[p][l][e]`, the coefficient of `bar(e_l)·X^e` (SuperNeo v1.2
Definition 15, Theorem 8). -/
theorem ringFMul_barRing (u A : RingF) (p : Fin ringDegree) :
    gl (ringFMul (barRing u) A p) = ∑ l, ∑ e, gl (u l) * gl (A e) * gl (phi81Kernel.weight p l e) := by
  have product : toPolynomial (barRing u) * toPolynomial A =
      ∑ l, ∑ e, C (gl (u l) * gl (A e)) * (toPolynomial (barBasis l) * toPolynomial (ringFMonomial e.val 1)) := by
    rw [toPolynomial_barRing, Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro l _
    rw [show toPolynomial A = ∑ e : Fin ringDegree, C (gl (A e)) * X ^ e.val from rfl, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro e _
    rw [toPolynomial_monomial, C_mul]
    ring
  have weight (l e : Fin ringDegree) : gl (phi81Kernel.weight p l e) =
      ((toPolynomial (barBasis l) * toPolynomial (ringFMonomial e.val 1)) %ₘ modulus).coeff p := by
    rw [← toPolynomial_ringFMul_mod, coeff_toPolynomial]
    rfl
  rw [show gl (ringFMul (barRing u) A p) = (toPolynomial (ringFMul (barRing u) A)).coeff p from
    (coeff_toPolynomial _ p).symm, toPolynomial_ringFMul_mod, product]
  have linear := map_sum (modByMonicHom modulus) (fun l => ∑ e, C (gl (u l) * gl (A e)) *
    (toPolynomial (barBasis l) * toPolynomial (ringFMonomial e.val 1))) Finset.univ
  simp only [modByMonicHom_apply] at linear
  rw [linear, finsetSum_coeff]
  apply Finset.sum_congr rfl
  intro l _
  have inner := map_sum (modByMonicHom modulus) (fun e => C (gl (u l) * gl (A e)) *
    (toPolynomial (barBasis l) * toPolynomial (ringFMonomial e.val 1))) Finset.univ
  simp only [modByMonicHom_apply] at inner
  rw [inner, finsetSum_coeff]
  apply Finset.sum_congr rfl
  intro e _
  rw [← smul_eq_C_mul, smul_modByMonic, coeff_smul, smul_eq_mul, weight]

end CompressionLean.Final
