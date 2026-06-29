import SuperNeo.Primitives.Field
import SuperNeo.Primitives.Parameters
import Mathlib.Algebra.QuadraticAlgebra.Basic
import Mathlib.NumberTheory.LegendreSymbol.Basic

namespace SuperNeo

/-!
Concrete quadratic extension-field carrier `K = F[u]/(u^2 - 7)`.

This module owns the concrete degree-2 extension carrier used by the
opening-convergence formalization. It reuses mathlib's
`QuadraticAlgebra F a b` at the paper-faithful specialization `a = 7`, `b = 0`,
then freezes the repo-facing names and coefficient views on top.
-/

open F

/-- The binomial relation constant: `u^2 = 7`. -/
def KExt_w : F := F.ofNat 7

/-- Concrete coefficient-pair carrier for the quadratic extension field
`K = F[u]/(u^2 - 7)`. -/
abbrev KExt : Type := QuadraticAlgebra F KExt_w 0

namespace KExt

noncomputable instance : Fintype KExt :=
  Fintype.ofEquiv (F × F) (QuadraticAlgebra.equivProd KExt_w 0).symm

/-- Curated local name for the quadratic relation constant. -/
def w : F := KExt_w

private theorem w_pow_div_two_eq_neg_one :
    ((7 : SuperNeo.FBridge) ^ (Goldilocks.q / 2)) = (-1 : SuperNeo.FBridge) := by
  native_decide

/-- `7` is a quadratic non-residue in the Goldilocks base field. -/
theorem w_not_square_zmod : ¬ IsSquare (7 : SuperNeo.FBridge) := by
  have hSevenNe : (7 : SuperNeo.FBridge) ≠ 0 := by
    native_decide
  intro hSq
  have hEuler :
      ((7 : SuperNeo.FBridge) ^ (Goldilocks.q / 2)) = (1 : SuperNeo.FBridge) :=
    (ZMod.euler_criterion (p := Goldilocks.q) hSevenNe).mp hSq
  rw [w_pow_div_two_eq_neg_one] at hEuler
  have hNe : (-1 : SuperNeo.FBridge) ≠ (1 : SuperNeo.FBridge) := by
    native_decide
  exact hNe hEuler

/-- No base-field square equals the quadratic relation constant `w = 7`. -/
theorem w_not_square (a : F) : a * a ≠ w := by
  intro hSq
  have hzSq : IsSquare (7 : SuperNeo.FBridge) := by
    refine ⟨SuperNeo.toZMod a, ?_⟩
    rw [← SuperNeo.toZMod_mul, hSq, w, KExt_w, SuperNeo.toZMod_ofNat]
    norm_num
  exact w_not_square_zmod hzSq

/-- Embed one base-field element as `a + 0*u`. -/
def ofF (x : F) : KExt :=
  QuadraticAlgebra.C x

/-- Explicit constructor from coefficients `(a, b)`. -/
def ofCoeffs (a b : F) : KExt :=
  ⟨a, b⟩

/-- Canonical coefficient view `(real, imag)`. -/
def coeffs (x : KExt) : Fin 2 → F
  | ⟨0, _⟩ => x.re
  | ⟨1, _⟩ => x.im

/-- Scale both coefficients by one base-field scalar. -/
def scaleBase (a : F) (x : KExt) : KExt :=
  ⟨a * x.re, a * x.im⟩

/-- Exponentiation in the multiplicative monoid of `KExt`. -/
def pow (x : KExt) : Nat → KExt
  | 0 => 1
  | n + 1 => pow x n * x

@[simp] theorem ofF_re (x : F) : (ofF x).re = x := by
  simp [ofF]

@[simp] theorem ofF_im (x : F) : (ofF x).im = 0 := by
  simp [ofF]

@[simp] theorem ofCoeffs_re (a b : F) : (ofCoeffs a b).re = a := rfl
@[simp] theorem ofCoeffs_im (a b : F) : (ofCoeffs a b).im = b := rfl

@[simp] theorem add_re (x y : KExt) : (x + y).re = x.re + y.re := by
  simp

@[simp] theorem add_im (x y : KExt) : (x + y).im = x.im + y.im := by
  simp

@[simp] theorem neg_re (x : KExt) : (-x).re = -x.re := by
  simp

@[simp] theorem neg_im (x : KExt) : (-x).im = -x.im := by
  simp

@[simp] theorem sub_re (x y : KExt) : (x - y).re = x.re - y.re := by
  simp

@[simp] theorem sub_im (x y : KExt) : (x - y).im = x.im - y.im := by
  simp

@[simp] theorem mul_re (x y : KExt) :
    (x * y).re = x.re * y.re + w * (x.im * y.im) := by
  simpa [w, KExt_w, mul_assoc, mul_left_comm, mul_comm] using
    (QuadraticAlgebra.re_mul (a := KExt_w) (b := (0 : F)) x y)

@[simp] theorem mul_im (x y : KExt) :
    (x * y).im = x.re * y.im + x.im * y.re := by
  simpa [w, KExt_w, mul_assoc, mul_left_comm, mul_comm] using
    (QuadraticAlgebra.im_mul (a := KExt_w) (b := (0 : F)) x y)

@[simp] theorem scaleBase_re (a : F) (x : KExt) :
    (scaleBase a x).re = a * x.re := by
  rfl

@[simp] theorem scaleBase_im (a : F) (x : KExt) :
    (scaleBase a x).im = a * x.im := by
  rfl

@[simp] theorem pow_zero (x : KExt) : pow x 0 = 1 := by
  rfl

@[simp] theorem pow_succ (x : KExt) (n : Nat) :
    pow x (n + 1) = pow x n * x := by
  rfl

theorem ext
    {x y : KExt}
    (hre : x.re = y.re)
    (him : x.im = y.im) : x = y := by
  exact QuadraticAlgebra.ext hre him

private theorem no_quadratic_root (r : F) :
    r ^ 2 ≠ KExt_w + (0 : F) * r := by
  simpa [pow_two, KExt_w, w] using w_not_square r

noncomputable instance : Field KExt := by
  letI : Fact (∀ r : F, r ^ 2 ≠ KExt_w + (0 : F) * r) := ⟨no_quadratic_root⟩
  infer_instance

end KExt

theorem extDegreeK_eq_two : Parameters.Goldilocks.extDegreeK = 2 :=
  Parameters.Goldilocks.extDegreeK_eq_2

end SuperNeo
