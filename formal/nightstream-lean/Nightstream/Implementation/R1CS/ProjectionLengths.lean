import Nightstream.Implementation.R1CS.ProjectionProgram

/-!
Contract: representation-width lemmas for the bounded polynomial language
used by PiRLC projection checks.
-/

namespace Nightstream.Implementation.R1CS.ProjectionProgram

open Nightstream.SuperNeo.ProjectionCheck

namespace Polynomial

theorem length_add (left right : List K) :
    (add left right).length = max left.length right.length := by
  induction left generalizing right with
  | nil => simp [add]
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp [add]
      | cons rightHead rightTail =>
          simp only [add, List.length_cons, Nat.succ_max_succ,
            inductionHypothesis]

@[simp] theorem length_scale (scalar : K) (coefficients : List K) :
    (scale scalar coefficients).length = coefficients.length := by
  induction coefficients with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [scale, inductionHypothesis]

theorem length_mul {left right : List K}
    (leftNonempty : left ≠ []) (rightNonempty : right ≠ []) :
    (mul left right).length = left.length + right.length - 1 := by
  induction left with
  | nil => exact False.elim (leftNonempty rfl)
  | cons head tail inductionHypothesis =>
      cases tail with
      | nil =>
          change (add (scale head right) [K.zero]).length =
            (head :: []).length + right.length - 1
          rw [length_add, length_scale]
          simp only [List.length_cons, List.length_nil]
          have rightPositive : 0 < right.length :=
            List.length_pos_iff.mpr rightNonempty
          omega
      | cons tailHead tailTail =>
          have tailNonempty : tailHead :: tailTail ≠ [] := by simp
          have tailLength := inductionHypothesis tailNonempty
          change (add (scale head right)
              (K.zero :: mul (tailHead :: tailTail) right)).length =
            (head :: tailHead :: tailTail).length + right.length - 1
          rw [length_add, length_scale]
          simp only [List.length_cons]
          rw [tailLength]
          have rightPositive : 0 < right.length :=
            List.length_pos_iff.mpr rightNonempty
          have selectRight : right.length ≤
              (tailHead :: tailTail).length + right.length - 1 + 1 := by
            simp only [List.length_cons]
            omega
          rw [Nat.max_eq_right selectRight]
          simp only [List.length_cons]
          omega

theorem length_sum_eq {polynomials : List (List K)} {width : Nat}
    (nonempty : polynomials ≠ [])
    (sameWidth : ∀ polynomial ∈ polynomials,
      polynomial.length = width) :
    (sum polynomials).length = width := by
  induction polynomials with
  | nil => exact False.elim (nonempty rfl)
  | cons head tail inductionHypothesis =>
      cases tail with
      | nil =>
          change (add head []).length = width
          rw [length_add]
          simp only [List.length_nil, Nat.max_zero]
          exact sameWidth head (by simp)
      | cons tailHead tailTail =>
          have headLength := sameWidth head (by simp)
          have tailLength := inductionHypothesis (by simp)
            (by
              intro polynomial member
              exact sameWidth polynomial (by simp [member]))
          change (add head (sum (tailHead :: tailTail))).length = width
          rw [length_add, headLength, tailLength, Nat.max_self]

theorem length_padRight {width : Nat} {coefficients : List K}
    (within : coefficients.length ≤ width) :
    (padRight width coefficients).length = width := by
  simp [padRight]
  omega

end Polynomial

@[simp] theorem basePolynomial_length (assignment : Nat → Nat)
    (columns : List Nat) :
    (basePolynomial assignment columns).length = columns.length := by
  simp [basePolynomial]

theorem PairTrace.productPolynomial_length (trace : PairTrace)
    (assignment : Nat → Nat) {rhoWidth inputWidth : Nat}
    (rhoLength : trace.rhoColumns.length = rhoWidth)
    (inputLength : trace.inputColumns.length = inputWidth)
    (rhoPositive : 0 < rhoWidth) (inputPositive : 0 < inputWidth) :
    (trace.productPolynomial assignment).length =
      rhoWidth + inputWidth - 1 := by
  unfold PairTrace.productPolynomial
  rw [Polynomial.length_mul]
  · simp [rhoLength, inputLength]
  · intro empty
    have := congrArg List.length empty
    simp [rhoLength] at this
    omega
  · intro empty
    have := congrArg List.length empty
    simp [inputLength] at this
    omega

/-- Fixed-width representation is a structural consequence, independent of
coefficient values. -/
theorem ProjectionTrace.identity_wellFormed_of_widths
    (trace : ProjectionTrace) (assignment : Nat → Nat)
    (layout : trace.LayoutValid) (pairsNonempty : trace.pairs ≠ [])
    (pairWidths : ∀ pair ∈ trace.pairs,
      pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54) :
    (trace.identity assignment).WellFormed := by
  rcases layout with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, outputLength,
      quotientLength, maxDegree⟩
  have mappedNonempty :
      (trace.pairs.map fun pair => pair.productPolynomial assignment) ≠ [] := by
    simpa using pairsNonempty
  have productWidths : ∀ polynomial ∈
      (trace.pairs.map fun pair => pair.productPolynomial assignment),
      polynomial.length = 107 := by
    intro polynomial member
    rcases List.mem_map.mp member with ⟨pair, pairMember, rfl⟩
    rcases pairWidths pair pairMember with ⟨rhoLength, inputLength⟩
    simpa using pair.productPolynomial_length assignment rhoLength inputLength
      (by decide) (by decide)
  have lhsLength : (Polynomial.sum (trace.pairs.map fun pair =>
      pair.productPolynomial assignment)).length = 107 :=
    Polynomial.length_sum_eq mappedNonempty productWidths
  have phiLength : Polynomial.phi81.length = 55 := by decide
  have quotientNonempty :
      basePolynomial assignment trace.quotientColumns ≠ [] := by
    intro empty
    have := congrArg List.length empty
    simp [quotientLength] at this
  have phiNonempty : Polynomial.phi81 ≠ [] := by decide
  have quotientProductLength :
      (Polynomial.mul
        (basePolynomial assignment trace.quotientColumns)
        Polynomial.phi81).length = 107 := by
    rw [Polynomial.length_mul quotientNonempty phiNonempty,
      basePolynomial_length, quotientLength, phiLength]
  have outputPadLength :
      (Polynomial.padRight (trace.maxDegree + 1)
        (basePolynomial assignment trace.outputColumns)).length = 107 := by
    have within :
        (basePolynomial assignment trace.outputColumns).length ≤
          trace.maxDegree + 1 := by
      simp [outputLength, maxDegree]
    rw [Polynomial.length_padRight within, maxDegree]
  unfold ProjectionTrace.identity Identity.WellFormed
  simp only
  rw [lhsLength, Polynomial.length_add, quotientProductLength,
    outputPadLength, Nat.max_self, maxDegree]
  decide

end Nightstream.Implementation.R1CS.ProjectionProgram
