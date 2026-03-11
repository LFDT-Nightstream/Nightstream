import SuperNeo.PolynomialBridge
import Mathlib.Algebra.Polynomial.OfFn
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Algebra.Polynomial.Degree.Lemmas

/-!
Constructive interpolation over the concrete Goldilocks field.

This module formalizes:
- coefficient-array polynomial evaluation,
- Lagrange interpolation on pairwise-distinct sample points,
- interpolation correctness on the sample points,
- uniqueness of the resulting coefficient array among degree-`< n` arrays.
-/

namespace SuperNeo

open scoped BigOperators

/-- Distinct interpolation nodes, indexed by the concrete array positions. -/
def interpolationNodesDistinct (xs : Array F) : Prop :=
  ∀ ⦃i j : Fin xs.size⦄, i ≠ j → xs[i.1]! ≠ xs[j.1]!

/-- Coefficient-array view as a univariate polynomial over `ZMod q`. -/
def coeffArrayPolynomial (coeffs : Array F) : Polynomial Fq :=
  Polynomial.ofFn coeffs.size (fun i => fToZMod (coeffs[i.1]!))

/-- Truncate/pad a `ZMod q[X]` polynomial into exactly `n` coefficients. -/
noncomputable def zmodPolyToCoeffArray (n : Nat) (p : Polynomial Fq) : Array F :=
  Array.ofFn (fun i : Fin n => zmodToF ((Polynomial.toFn n p) i))

@[simp] theorem zmodPolyToCoeffArray_size (n : Nat) (p : Polynomial Fq) :
    (zmodPolyToCoeffArray n p).size = n := by
  simp [zmodPolyToCoeffArray]

theorem coeffArrayPolynomial_natDegree_lt_size
    {coeffs : Array F}
    (hSizePos : 0 < coeffs.size) :
    (coeffArrayPolynomial coeffs).natDegree < coeffs.size := by
  unfold coeffArrayPolynomial
  have hOneLe : 1 ≤ coeffs.size := Nat.succ_le_of_lt hSizePos
  simpa using
    (Polynomial.ofFn_natDegree_lt (R := Fq) hOneLe
      (fun i => fToZMod (coeffs[i.1]!)))

theorem coeffArrayPolynomial_zmodPolyToCoeffArray
    (n : Nat)
    (p : Polynomial Fq)
    (hDeg : p.natDegree < n) :
    coeffArrayPolynomial (zmodPolyToCoeffArray n p) = p := by
  let arr : Array F := zmodPolyToCoeffArray n p
  have hSize : arr.size = n := by
    simp [arr, zmodPolyToCoeffArray]
  unfold coeffArrayPolynomial
  rw [hSize]
  have hFun :
      (fun i : Fin n => fToZMod (arr[i.1]!)) = Polynomial.toFn n p := by
    funext i
    simp [arr, zmodPolyToCoeffArray, fToZMod_zmodToF]
  calc
    Polynomial.ofFn n (fun i : Fin n => fToZMod (arr[i.1]!))
      = Polynomial.ofFn n (Polynomial.toFn n p) := by
          simp [hFun]
    _ = p := Polynomial.ofFn_comp_toFn_eq_id_of_natDegree_lt hDeg

theorem coeffArrayPolynomial_injective_of_size
    {n : Nat}
    {a b : Array F}
    (ha : a.size = n)
    (hb : b.size = n)
    (hEq : coeffArrayPolynomial a = coeffArrayPolynomial b) :
    a = b := by
  unfold coeffArrayPolynomial at hEq
  rw [ha, hb] at hEq
  apply Array.ext
  · simp [ha, hb]
  · intro i hiA hiB
    have hi : i < n := by
      simpa [ha] using hiA
    have hCoeff :
        fToZMod (a[i]!) = fToZMod (b[i]!) := by
      simpa [hi] using congrArg (fun p : Polynomial Fq => p.coeff i) hEq
    have hAt : a[i]! = b[i]! := fToZMod_injective hCoeff
    simpa [hiA, hiB] using hAt

/-- Polynomial evaluation for coefficient arrays. -/
noncomputable def polyEval (coeffs : Array F) (x : F) : F :=
  zmodToF ((coeffArrayPolynomial coeffs).eval (fToZMod x))

@[simp] theorem fToZMod_polyEval (coeffs : Array F) (x : F) :
    fToZMod (polyEval coeffs x) = (coeffArrayPolynomial coeffs).eval (fToZMod x) := by
  simp [polyEval]

/-- Denominator of the `i`-th Lagrange basis term. -/
noncomputable def lagrangeDenomZMod (xs : Array F) (i : Fin xs.size) : Fq :=
  ∏ j ∈ Finset.univ.erase i, (fToZMod (xs[i.1]!) - fToZMod (xs[j.1]!))

/-- Numerator polynomial of the `i`-th Lagrange basis term. -/
noncomputable def lagrangeNumeratorZMod (xs : Array F) (i : Fin xs.size) : Polynomial Fq :=
  ∏ j ∈ Finset.univ.erase i, (Polynomial.X - Polynomial.C (fToZMod (xs[j.1]!)))

/-- Scaled `i`-th Lagrange interpolation term. -/
noncomputable def lagrangeTermZMod
    (xs ys : Array F)
    (hSize : xs.size = ys.size)
    (i : Fin xs.size) : Polynomial Fq :=
  let yi := fToZMod (ys[i.1]'(by simpa [hSize] using i.2))
  Polynomial.C (yi * (lagrangeDenomZMod xs i)⁻¹) * lagrangeNumeratorZMod xs i

/-- Constructive interpolation polynomial from sample points and values. -/
noncomputable def interpolationPolynomialZMod (xs ys : Array F) : Polynomial Fq :=
  if hSize : xs.size = ys.size then
    ∑ i : Fin xs.size, lagrangeTermZMod xs ys hSize i
  else
    0

theorem coeffArrayPolynomial_eq_zero_of_size_zero
    {coeffs : Array F}
    (hZero : coeffs.size = 0) :
    coeffArrayPolynomial coeffs = 0 := by
  unfold coeffArrayPolynomial
  rw [hZero]
  simp

theorem interpolationPolynomialZMod_eq_zero_of_size_zero
    {xs ys : Array F}
    (hSize : xs.size = ys.size)
    (hZero : xs.size = 0) :
    interpolationPolynomialZMod xs ys = 0 := by
  rw [interpolationPolynomialZMod, dif_pos hSize]
  haveI : IsEmpty (Fin xs.size) := by
    refine ⟨?_⟩
    intro i
    have : i.1 < 0 := by
      simpa [hZero] using i.2
    exact (Nat.not_lt_zero _ this).elim
  simp

theorem lagrangeDenomZMod_ne_zero
    {xs : Array F}
    (hDistinct : interpolationNodesDistinct xs)
    (i : Fin xs.size) :
    lagrangeDenomZMod xs i ≠ 0 := by
  classical
  unfold lagrangeDenomZMod
  refine Finset.prod_ne_zero_iff.mpr ?_
  intro j hj
  have hjNe : j ≠ i := (Finset.mem_erase.mp hj).1
  have hNodesNe : xs[i.1]! ≠ xs[j.1]! := hDistinct (by
    intro hij
    exact hjNe hij.symm)
  intro hSub
  apply hNodesNe
  exact fToZMod_injective (sub_eq_zero.mp hSub)

theorem lagrangeNumeratorZMod_eval_self
    (xs : Array F)
    (i : Fin xs.size) :
    (lagrangeNumeratorZMod xs i).eval (fToZMod (xs[i.1]!)) = lagrangeDenomZMod xs i := by
  classical
  simp [lagrangeNumeratorZMod, lagrangeDenomZMod, Polynomial.eval_prod]

theorem lagrangeNumeratorZMod_eval_other
    {xs : Array F}
    {i k : Fin xs.size}
    (hik : k ≠ i) :
    (lagrangeNumeratorZMod xs i).eval (fToZMod (xs[k.1]!)) = 0 := by
  classical
  rw [lagrangeNumeratorZMod, Polynomial.eval_prod]
  refine Finset.prod_eq_zero (i := k) ?_ ?_
  · exact Finset.mem_erase.mpr ⟨hik, Finset.mem_univ k⟩
  · simp

theorem lagrangeTermZMod_eval_self
    {xs ys : Array F}
    (hSize : xs.size = ys.size)
    (hDistinct : interpolationNodesDistinct xs)
    (i : Fin xs.size) :
    (lagrangeTermZMod xs ys hSize i).eval (fToZMod (xs[i.1]!)) =
      fToZMod (ys[i.1]'(by simpa [hSize] using i.2)) := by
  have hDenom : lagrangeDenomZMod xs i ≠ 0 := lagrangeDenomZMod_ne_zero hDistinct i
  let yi : Fq := fToZMod (ys[i.1]'(by simpa [hSize] using i.2))
  calc
    (lagrangeTermZMod xs ys hSize i).eval (fToZMod (xs[i.1]!))
        = (yi * (lagrangeDenomZMod xs i)⁻¹) * lagrangeDenomZMod xs i := by
            rw [lagrangeTermZMod, Polynomial.eval_mul, Polynomial.eval_C,
              lagrangeNumeratorZMod_eval_self]
    _ = yi * ((lagrangeDenomZMod xs i)⁻¹ * lagrangeDenomZMod xs i) := by
          rw [mul_assoc]
    _ = yi := by simp [hDenom]

theorem lagrangeTermZMod_eval_other
    {xs ys : Array F}
    (hSize : xs.size = ys.size)
    {i k : Fin xs.size}
    (hik : k ≠ i) :
    (lagrangeTermZMod xs ys hSize i).eval (fToZMod (xs[k.1]!)) = 0 := by
  rw [lagrangeTermZMod, Polynomial.eval_mul, Polynomial.eval_C,
    lagrangeNumeratorZMod_eval_other hik]
  simp

theorem interpolationPolynomialZMod_eval_node
    {xs ys : Array F}
    (hSize : xs.size = ys.size)
    (hDistinct : interpolationNodesDistinct xs)
    (i : Fin xs.size) :
    (interpolationPolynomialZMod xs ys).eval (fToZMod (xs[i.1]!)) =
      fToZMod (ys[i.1]'(by simpa [hSize] using i.2)) := by
  classical
  dsimp [interpolationPolynomialZMod]
  rw [dif_pos hSize, Polynomial.eval_finset_sum]
  rw [Finset.sum_eq_single i]
  · simpa using lagrangeTermZMod_eval_self hSize hDistinct i
  · intro j _hj hji
    have hij : i ≠ j := by
      exact fun hijEq => hji hijEq.symm
    simpa using lagrangeTermZMod_eval_other (xs := xs) (ys := ys) hSize (i := j) (k := i) hij
  · intro hiNotMem
    exact False.elim (hiNotMem (Finset.mem_univ i))

  theorem lagrangeNumeratorZMod_natDegree
    (xs : Array F)
    (i : Fin xs.size) :
    (lagrangeNumeratorZMod xs i).natDegree = xs.size - 1 := by
  classical
  unfold lagrangeNumeratorZMod
  simpa [Finset.card_erase_of_mem (Finset.mem_univ i)] using
    (Polynomial.natDegree_finset_prod_X_sub_C_eq_card
      (s := Finset.univ.erase i)
      (f := fun j : Fin xs.size => fToZMod (xs[j.1]!)))

theorem interpolationPolynomialZMod_natDegree_lt_size
    {xs ys : Array F}
    (hSize : xs.size = ys.size)
    (hSizePos : 0 < xs.size) :
    (interpolationPolynomialZMod xs ys).natDegree < xs.size := by
  classical
  dsimp [interpolationPolynomialZMod]
  rw [dif_pos hSize]
  have hBound :
      ∀ i ∈ (Finset.univ : Finset (Fin xs.size)),
        (lagrangeTermZMod xs ys hSize i).natDegree ≤ xs.size - 1 := by
    intro i _hi
    exact (Polynomial.natDegree_C_mul_le _ _).trans
      (le_of_eq (lagrangeNumeratorZMod_natDegree xs i))
  have hLe :
      (∑ i : Fin xs.size, lagrangeTermZMod xs ys hSize i).natDegree ≤ xs.size - 1 :=
    Polynomial.natDegree_sum_le_of_forall_le _ _ hBound
  have hPred : xs.size - 1 < xs.size := by
    exact Nat.sub_lt hSizePos (Nat.succ_pos 0)
  exact lt_of_le_of_lt hLe hPred

/-- Executable coefficient-array interpolation algorithm. -/
noncomputable def interpolateCoeffs (xs ys : Array F) : Array F :=
  zmodPolyToCoeffArray xs.size (interpolationPolynomialZMod xs ys)

@[simp] theorem interpolateCoeffs_size (xs ys : Array F) :
    (interpolateCoeffs xs ys).size = xs.size := by
  simp [interpolateCoeffs]

theorem coeffArrayPolynomial_interpolateCoeffs
    {xs ys : Array F}
    (hSize : xs.size = ys.size) :
    coeffArrayPolynomial (interpolateCoeffs xs ys) = interpolationPolynomialZMod xs ys := by
  by_cases hZero : xs.size = 0
  · have hPolyZero : interpolationPolynomialZMod xs ys = 0 := by
      exact interpolationPolynomialZMod_eq_zero_of_size_zero hSize hZero
    have hInterpZero : (interpolateCoeffs xs ys).size = 0 := by
      simp [interpolateCoeffs, hZero]
    calc
      coeffArrayPolynomial (interpolateCoeffs xs ys) = 0 := by
        exact coeffArrayPolynomial_eq_zero_of_size_zero hInterpZero
      _ = interpolationPolynomialZMod xs ys := hPolyZero.symm
  · exact coeffArrayPolynomial_zmodPolyToCoeffArray xs.size
      (interpolationPolynomialZMod xs ys)
      (interpolationPolynomialZMod_natDegree_lt_size hSize (Nat.pos_of_ne_zero hZero))

/-- Mathematical interpolation relation on the sample set. -/
noncomputable def interpolatesOn (xs ys coeffs : Array F) : Prop :=
  ∃ hSize : xs.size = ys.size,
    coeffs.size = xs.size ∧
    ∀ i : Fin xs.size,
      polyEval coeffs (xs[i.1]!) = ys[i.1]'(by simpa [hSize] using i.2)

/-- Theorem-facing interpolation proposition. -/
noncomputable def interpolationProp
    (xs ys coeffs : Array F)
    (evalPoint expectedEval : F) : Prop :=
  interpolationNodesDistinct xs ∧
  interpolatesOn xs ys coeffs ∧
  polyEval coeffs evalPoint = expectedEval

/-- Legacy universal interpolation boundary retained as an explicit refuted surface. -/
noncomputable def interpolationAssumption : Prop :=
  ∀ xs ys coeffs : Array F, ∀ evalPoint expectedEval : F,
    interpolationProp xs ys coeffs evalPoint expectedEval

theorem interpolationProp_intro
    {xs ys coeffs : Array F}
    {evalPoint expectedEval : F}
    (hDistinct : interpolationNodesDistinct xs)
    (hInterp : interpolatesOn xs ys coeffs)
    (hEval : polyEval coeffs evalPoint = expectedEval) :
    interpolationProp xs ys coeffs evalPoint expectedEval := by
  exact ⟨hDistinct, hInterp, hEval⟩

theorem interpolationProp_sizes
    {xs ys coeffs : Array F}
    {evalPoint expectedEval : F}
    (hProp : interpolationProp xs ys coeffs evalPoint expectedEval) :
    xs.size = ys.size ∧ coeffs.size = xs.size := by
  rcases hProp.2.1 with ⟨hSize, hCoeffSize, _⟩
  exact ⟨hSize, hCoeffSize⟩

theorem interpolationProp_eval_eq
    {xs ys coeffs : Array F}
    {evalPoint expectedEval : F}
    (hProp : interpolationProp xs ys coeffs evalPoint expectedEval) :
    polyEval coeffs evalPoint = expectedEval := by
  exact hProp.2.2

noncomputable instance interpolationProp_decidable
    (xs ys coeffs : Array F)
    (evalPoint expectedEval : F) :
    Decidable (interpolationProp xs ys coeffs evalPoint expectedEval) := by
  unfold interpolationProp interpolatesOn interpolationNodesDistinct
  infer_instance

instance interpolationNodesDistinct_decidable
    (xs : Array F) :
    Decidable (interpolationNodesDistinct xs) := by
  unfold interpolationNodesDistinct
  infer_instance

theorem interpolateCoeffs_interpolatesOn
    {xs ys : Array F}
    (hSize : xs.size = ys.size)
    (hDistinct : interpolationNodesDistinct xs) :
    interpolatesOn xs ys (interpolateCoeffs xs ys) := by
  refine ⟨hSize, by simp, ?_⟩
  intro i
  apply fToZMod_injective
  have hEval :
      (interpolationPolynomialZMod xs ys).eval (fToZMod (xs[i.1]!)) =
        fToZMod (ys[i.1]'(by simpa [hSize] using i.2)) :=
    interpolationPolynomialZMod_eval_node hSize hDistinct i
  simpa [polyEval, coeffArrayPolynomial_interpolateCoeffs hSize] using hEval

theorem interpolateCoeffs_unique
    {xs ys coeffs : Array F}
    (hDistinct : interpolationNodesDistinct xs)
    (hInterp : interpolatesOn xs ys coeffs) :
    coeffs = interpolateCoeffs xs ys := by
  rcases hInterp with ⟨hSize, hCoeffSize, hEval⟩
  have hPolyEq : coeffArrayPolynomial coeffs = interpolationPolynomialZMod xs ys := by
    by_cases hZero : xs.size = 0
    · have hCoeffZero : coeffs.size = 0 := by
        simpa [hZero] using hCoeffSize
      have hPolyZero : interpolationPolynomialZMod xs ys = 0 := by
        exact interpolationPolynomialZMod_eq_zero_of_size_zero hSize hZero
      calc
        coeffArrayPolynomial coeffs = 0 := by
          exact coeffArrayPolynomial_eq_zero_of_size_zero hCoeffZero
        _ = interpolationPolynomialZMod xs ys := hPolyZero.symm
    · have hCoeffPos : 0 < coeffs.size := by
        simpa [hCoeffSize] using Nat.pos_of_ne_zero hZero
      have hCoeffDeg : (coeffArrayPolynomial coeffs).natDegree < xs.size := by
        simpa [hCoeffSize] using coeffArrayPolynomial_natDegree_lt_size (coeffs := coeffs) hCoeffPos
      have hInterpDeg : (interpolationPolynomialZMod xs ys).natDegree < xs.size :=
        interpolationPolynomialZMod_natDegree_lt_size hSize (Nat.pos_of_ne_zero hZero)
      have hNodeInj :
          Function.Injective (fun i : Fin xs.size => fToZMod (xs[i.1]!)) := by
        intro i j hij
        by_contra hNe
        have hNodesNe : xs[i.1]! ≠ xs[j.1]! := hDistinct hNe
        exact hNodesNe (fToZMod_injective hij)
      have hEvalEq :
          ∀ i : Fin xs.size,
            (coeffArrayPolynomial coeffs).eval (fToZMod (xs[i.1]!)) =
              (interpolationPolynomialZMod xs ys).eval (fToZMod (xs[i.1]!)) := by
        intro i
        have hCoeffEval :
            (coeffArrayPolynomial coeffs).eval (fToZMod (xs[i.1]!)) =
              fToZMod (ys[i.1]'(by simpa [hSize] using i.2)) := by
          simpa [polyEval] using congrArg fToZMod (hEval i)
        have hInterpEval :
            (interpolationPolynomialZMod xs ys).eval (fToZMod (xs[i.1]!)) =
              fToZMod (ys[i.1]'(by simpa [hSize] using i.2)) :=
          interpolationPolynomialZMod_eval_node hSize hDistinct i
        exact hCoeffEval.trans hInterpEval.symm
      exact Polynomial.eq_of_natDegree_lt_card_of_eval_eq
        (coeffArrayPolynomial coeffs)
        (interpolationPolynomialZMod xs ys)
        hNodeInj
        hEvalEq
        (by
          simpa using (max_lt_iff.mpr ⟨hCoeffDeg, hInterpDeg⟩))
  exact coeffArrayPolynomial_injective_of_size
    hCoeffSize
    (by simp [interpolateCoeffs])
    (by simpa [coeffArrayPolynomial_interpolateCoeffs hSize] using hPolyEq)

theorem interpolateCoeffs_interpolationProp
    {xs ys : Array F}
    {evalPoint : F}
    (hSize : xs.size = ys.size)
    (hDistinct : interpolationNodesDistinct xs) :
    interpolationProp xs ys (interpolateCoeffs xs ys) evalPoint
      (polyEval (interpolateCoeffs xs ys) evalPoint) := by
  exact ⟨hDistinct, interpolateCoeffs_interpolatesOn hSize hDistinct, rfl⟩

/-- Executable interpolation checker. -/
noncomputable def interpolationCase
    (xs ys expectedCoeffs : Array F)
    (evalPoint expectedEval : F) : Bool :=
  if _hSize : xs.size = ys.size then
    if _hDistinct : interpolationNodesDistinct xs then
      decide
        (expectedCoeffs = interpolateCoeffs xs ys ∧
          polyEval (interpolateCoeffs xs ys) evalPoint = expectedEval)
    else
      false
  else
    false

theorem interpolationCase_sound
    {xs ys expectedCoeffs : Array F}
    {evalPoint expectedEval : F}
    (hOk : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
    interpolationProp xs ys expectedCoeffs evalPoint expectedEval := by
  unfold interpolationCase at hOk
  by_cases hSize : xs.size = ys.size
  · by_cases hDistinct : interpolationNodesDistinct xs
    · have hDec :
        decide
          (expectedCoeffs = interpolateCoeffs xs ys ∧
            polyEval (interpolateCoeffs xs ys) evalPoint = expectedEval) = true := by
          simpa [hSize, hDistinct] using hOk
      rcases decide_eq_true_eq.mp hDec with ⟨hCoeffEq, hEvalEq⟩
      have hInterp : interpolatesOn xs ys (interpolateCoeffs xs ys) :=
        interpolateCoeffs_interpolatesOn hSize hDistinct
      exact ⟨hDistinct, by simpa [hCoeffEq] using hInterp, by simpa [hCoeffEq] using hEvalEq⟩
    · simp [hSize, hDistinct] at hOk
  · simp [hSize] at hOk

theorem interpolationCase_complete
    {xs ys expectedCoeffs : Array F}
    {evalPoint expectedEval : F}
    (hProp : interpolationProp xs ys expectedCoeffs evalPoint expectedEval) :
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  rcases hProp with ⟨hDistinct, hInterp, hEval⟩
  rcases hInterp with ⟨hSize, hCoeffSize, hNodes⟩
  have hInterpAll : interpolatesOn xs ys expectedCoeffs := ⟨hSize, hCoeffSize, hNodes⟩
  have hCoeffEq : expectedCoeffs = interpolateCoeffs xs ys :=
    interpolateCoeffs_unique hDistinct hInterpAll
  have hEvalEq : polyEval (interpolateCoeffs xs ys) evalPoint = expectedEval := by
    simpa [hCoeffEq] using hEval
  have hDec :
      decide
        (expectedCoeffs = interpolateCoeffs xs ys ∧
          polyEval (interpolateCoeffs xs ys) evalPoint = expectedEval) = true := by
    exact decide_eq_true ⟨hCoeffEq, hEvalEq⟩
  unfold interpolationCase
  simp [hSize, hDistinct, hDec]

theorem interpolationCase_eq_true_iff
    {xs ys expectedCoeffs : Array F}
    {evalPoint expectedEval : F} :
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true ↔
      interpolationProp xs ys expectedCoeffs evalPoint expectedEval := by
  constructor
  · exact interpolationCase_sound
  · exact interpolationCase_complete

theorem not_interpolationAssumption : ¬ interpolationAssumption := by
  intro h
  let xs : Array F := #[0, 0]
  let ys : Array F := #[0, 0]
  let coeffs : Array F := #[0, 0]
  have hBad := h xs ys coeffs (0 : F) (0 : F)
  let i : Fin xs.size := ⟨0, by simp [xs]⟩
  let j : Fin xs.size := ⟨1, by simp [xs]⟩
  have hij : i ≠ j := by decide
  have hNe : xs[i.1]! ≠ xs[j.1]! := hBad.1 hij
  simp [xs, i, j] at hNe

end SuperNeo
