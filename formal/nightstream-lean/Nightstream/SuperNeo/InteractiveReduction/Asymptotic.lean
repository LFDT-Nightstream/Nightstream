import Init.Data.Rat
import Nightstream.SuperNeo.InteractiveReduction.Paper
import Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition

/-!
Security-parameter-indexed probability weights and an explicit polynomial
work predicate.

Owns: the pointwise rational probability scale used by asymptotic games and
the elementary closure facts needed to derive an extractor work bound from a
one-run work bound and an inverse-success-floor bound.

Does not own: a protocol, a sampler, a success floor, a random oracle, Rust,
R1CS, or constraints.

`PolynomiallyBounded` is intentionally arithmetic rather than an opaque
`Prop`: a witness is one nonnegative rational coefficient and one natural
degree bounding the work at every security parameter.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.Asymptotic

open Nightstream.SuperNeo.InteractiveReduction.Paper

/-- Probability or work quantities indexed by the security parameter. -/
abbrev Weight := Nat -> Rat

/-- Pointwise rational probability arithmetic. -/
def scale : ProbabilityScale Weight where
  zero := fun _ => 0
  one := fun _ => 1
  add := fun left right securityParameter =>
    left securityParameter + right securityParameter
  subtract := fun left right securityParameter =>
    left securityParameter - right securityParameter
  le := fun left right =>
    forall securityParameter, left securityParameter <= right securityParameter
  le_refl := fun _ _ => Rat.le_refl
  le_trans := by
    intro left middle right leftMiddle middleRight securityParameter
    exact Rat.le_trans
      (leftMiddle securityParameter) (middleRight securityParameter)
  subtract_zero := by
    intro weight
    funext securityParameter
    rw [Rat.sub_eq_add_neg, Rat.neg_zero, Rat.add_zero]

/-- The standard ordered-additive laws hold pointwise. -/
def scaleLaws :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.ScaleLaws
      scale where
  subtract_mono_left := by
    intro left right error bound securityParameter
    change
      left securityParameter - error securityParameter <=
        right securityParameter - error securityParameter
    rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg]
    exact (Rat.add_le_add_right
      (c := -error securityParameter)).mpr (bound securityParameter)
  subtract_subtract := by
    intro probability first second
    funext securityParameter
    change
      (probability securityParameter - first securityParameter) -
          second securityParameter =
        probability securityParameter -
          (first securityParameter + second securityParameter)
    rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, Rat.sub_eq_add_neg,
      Rat.neg_add, Rat.add_assoc]

/-- The canonical monomial used for a polynomial work bound. -/
def monomial (coefficient : Rat) (degree securityParameter : Nat) : Rat :=
  coefficient * ((securityParameter + 1 : Nat) : Rat) ^ degree

/-- Explicit polynomial upper bound on a nonnegative operational quantity. -/
def PolynomiallyBounded (work : Weight) : Prop :=
  exists coefficient : Rat, exists degree : Nat,
    0 <= coefficient /\
    forall securityParameter,
      work securityParameter <=
        monomial coefficient degree securityParameter

private theorem ratPow_add
    (base : Rat)
    (left right : Nat) :
    base ^ (left + right) = base ^ left * base ^ right := by
  induction right with
  | zero => simp
  | succ smaller inductionHypothesis =>
      rw [Nat.add_succ, Rat.pow_succ, Rat.pow_succ, inductionHypothesis,
        Rat.mul_assoc]

private theorem one_le_ratPow
    {base : Rat}
    (oneLe : 1 <= base) :
    forall degree : Nat, 1 <= base ^ degree := by
  intro degree
  induction degree with
  | zero => simp
  | succ smaller inductionHypothesis =>
      rw [Rat.pow_succ]
      calc
        (1 : Rat) <= base := oneLe
        _ = 1 * base := (Rat.one_mul base).symm
        _ <= base ^ smaller * base :=
          Rat.mul_le_mul_of_nonneg_right inductionHypothesis
            (Rat.le_trans (by decide : (0 : Rat) <= 1) oneLe)

private theorem ratPow_mono_exponent
    {base : Rat}
    (oneLe : 1 <= base)
    {lower upper : Nat}
    (degreeLe : lower <= upper) :
    base ^ lower <= base ^ upper := by
  rcases Nat.exists_eq_add_of_le degreeLe with ⟨extra, rfl⟩
  rw [ratPow_add]
  have lowerNonnegative : 0 <= base ^ lower :=
    Rat.pow_nonneg (Rat.le_trans (by decide : (0 : Rat) <= 1) oneLe)
  simpa using
    (Rat.mul_le_mul_of_nonneg_left
      (a := 1) (b := base ^ extra)
      (one_le_ratPow oneLe extra) lowerNonnegative)

private theorem securityBase_one_le
    (securityParameter : Nat) :
    (1 : Rat) <= ((securityParameter + 1 : Nat) : Rat) := by
  exact Rat.natCast_le_natCast.mpr (Nat.succ_le_succ (Nat.zero_le _))

/-- The constant-one function is polynomially bounded. -/
theorem polynomiallyBounded_one :
    PolynomiallyBounded (fun _ => 1) := by
  refine ⟨1, 0, by decide, ?_⟩
  intro securityParameter
  simp [monomial]

/-- Pointwise smaller work inherits an explicit polynomial bound. -/
theorem PolynomiallyBounded.mono
    {work bound : Weight}
    (workLe : forall securityParameter,
      work securityParameter <= bound securityParameter)
    (bounded : PolynomiallyBounded bound) :
    PolynomiallyBounded work := by
  rcases bounded with ⟨coefficient, degree, coefficientNonnegative, boundLe⟩
  exact ⟨coefficient, degree, coefficientNonnegative, fun securityParameter =>
    Rat.le_trans (workLe securityParameter) (boundLe securityParameter)⟩

/-- Addition preserves explicit polynomial boundedness. -/
theorem PolynomiallyBounded.add
    {left right : Weight}
    (leftBounded : PolynomiallyBounded left)
    (rightBounded : PolynomiallyBounded right) :
    PolynomiallyBounded
      (fun securityParameter =>
        left securityParameter + right securityParameter) := by
  rcases leftBounded with
    ⟨leftCoefficient, leftDegree, leftCoefficientNonnegative, leftBound⟩
  rcases rightBounded with
    ⟨rightCoefficient, rightDegree, rightCoefficientNonnegative, rightBound⟩
  refine ⟨leftCoefficient + rightCoefficient,
    leftDegree + rightDegree,
    Rat.add_nonneg leftCoefficientNonnegative rightCoefficientNonnegative, ?_⟩
  intro securityParameter
  let base : Rat := ((securityParameter + 1 : Nat) : Rat)
  have baseOneLe : 1 <= base := securityBase_one_le securityParameter
  have commonLeft :
      leftCoefficient * base ^ leftDegree <=
        leftCoefficient * base ^ (leftDegree + rightDegree) :=
    Rat.mul_le_mul_of_nonneg_left
      (ratPow_mono_exponent baseOneLe (Nat.le_add_right _ _))
      leftCoefficientNonnegative
  have commonRight :
      rightCoefficient * base ^ rightDegree <=
        rightCoefficient * base ^ (leftDegree + rightDegree) :=
    Rat.mul_le_mul_of_nonneg_left
      (ratPow_mono_exponent baseOneLe (Nat.le_add_left _ _))
      rightCoefficientNonnegative
  calc
    left securityParameter + right securityParameter <=
        leftCoefficient * base ^ leftDegree +
          rightCoefficient * base ^ rightDegree :=
      Rat.le_trans
        ((Rat.add_le_add_right
          (c := right securityParameter)).mpr
            (leftBound securityParameter))
        ((Rat.add_le_add_left
          (c := leftCoefficient * base ^ leftDegree)).mpr
            (rightBound securityParameter))
    _ <=
        leftCoefficient * base ^ (leftDegree + rightDegree) +
          rightCoefficient * base ^ (leftDegree + rightDegree) :=
      Rat.le_trans
        ((Rat.add_le_add_right
          (c := rightCoefficient * base ^ rightDegree)).mpr commonLeft)
        ((Rat.add_le_add_left
          (c := leftCoefficient * base ^ (leftDegree + rightDegree))).mpr
            commonRight)
    _ =
        monomial (leftCoefficient + rightCoefficient)
          (leftDegree + rightDegree) securityParameter := by
      simp only [monomial, base]
      rw [Rat.add_mul]

/-- Multiplication preserves explicit polynomial boundedness for nonnegative
quantities. -/
theorem PolynomiallyBounded.mul_of_nonnegative
    {left right : Weight}
    (rightNonnegative : forall securityParameter,
      0 <= right securityParameter)
    (leftBounded : PolynomiallyBounded left)
    (rightBounded : PolynomiallyBounded right) :
    PolynomiallyBounded
      (fun securityParameter =>
        left securityParameter * right securityParameter) := by
  rcases leftBounded with
    ⟨leftCoefficient, leftDegree, leftCoefficientNonnegative, leftBound⟩
  rcases rightBounded with
    ⟨rightCoefficient, rightDegree, rightCoefficientNonnegative, rightBound⟩
  refine ⟨leftCoefficient * rightCoefficient,
    leftDegree + rightDegree,
    Rat.mul_nonneg leftCoefficientNonnegative rightCoefficientNonnegative, ?_⟩
  intro securityParameter
  let base : Rat := ((securityParameter + 1 : Nat) : Rat)
  have productBound :
      left securityParameter * right securityParameter <=
        (leftCoefficient * base ^ leftDegree) *
          (rightCoefficient * base ^ rightDegree) := by
    exact Rat.le_trans
      (Rat.mul_le_mul_of_nonneg_right
        (leftBound securityParameter)
        (rightNonnegative securityParameter))
      (Rat.mul_le_mul_of_nonneg_left
        (rightBound securityParameter)
        (Rat.mul_nonneg leftCoefficientNonnegative
          (Rat.pow_nonneg
            (Rat.le_trans (by decide : (0 : Rat) <= 1)
              (securityBase_one_le securityParameter)))))
  refine Rat.le_trans productBound ?_
  simp only [monomial, base]
  rw [ratPow_add]
  let leftPower : Rat :=
    (↑(securityParameter + 1) : Rat) ^ leftDegree
  let rightPower : Rat :=
    (↑(securityParameter + 1) : Rat) ^ rightDegree
  change
    (leftCoefficient * leftPower) *
        (rightCoefficient * rightPower) <=
      (leftCoefficient * rightCoefficient) *
        (leftPower * rightPower)
  calc
    (leftCoefficient * leftPower) * (rightCoefficient * rightPower) =
        leftCoefficient * (leftPower * (rightCoefficient * rightPower)) := by
      rw [Rat.mul_assoc]
    _ = leftCoefficient * ((leftPower * rightCoefficient) * rightPower) := by
      rw [← Rat.mul_assoc leftPower rightCoefficient rightPower]
    _ = leftCoefficient * ((rightCoefficient * leftPower) * rightPower) := by
      rw [Rat.mul_comm leftPower rightCoefficient]
    _ = leftCoefficient * (rightCoefficient * (leftPower * rightPower)) := by
      rw [Rat.mul_assoc rightCoefficient leftPower rightPower]
    _ = (leftCoefficient * rightCoefficient) *
        (leftPower * rightPower) := by
      rw [Rat.mul_assoc]
    _ <= (leftCoefficient * rightCoefficient) *
        (leftPower * rightPower) := Rat.le_refl

end Nightstream.SuperNeo.InteractiveReduction.Asymptotic
