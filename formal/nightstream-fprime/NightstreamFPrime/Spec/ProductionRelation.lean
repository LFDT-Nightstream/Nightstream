import NightstreamFPrime.Spec.ProductionRelation.GatePolynomial
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Owns the one production CCS polynomial for the Nightstream F-prime relation.

The relation uses the Lean-owned low-norm gate `a * b - c + sboxInput ^ 7`.
Its 4 matrix slots are the gate ports. SuperNeo Pad is not a CCS matrix and
remains the separate `Eval_K` family.
-/

namespace NightstreamFPrime.Spec.ProductionRelation

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Fixed SuperNeo v1.1 `Eval_A` arity. -/
def matrixCount : Nat := GatePolynomial.matrixCount

/-- Number of matrix slots used by the gate compiler. -/
def meaningfulPortCount : Nat := GatePolynomial.meaningfulPortCount

/-- The sole production CCS polynomial. -/
def polynomial : ConstraintPolynomial F matrixCount :=
  GatePolynomial.polynomial

@[simp] theorem matrixCount_eq : matrixCount = 4 := by
  rfl

@[simp] theorem meaningfulPortCount_eq : meaningfulPortCount = 4 := by
  rfl

@[simp] theorem polynomial_terms :
    polynomial.terms = GatePolynomial.terms := by
  rfl

@[simp] theorem polynomial_degreeBound : polynomial.degreeBound = 8 := by
  rfl

theorem polynomial_canonicalEqualityGatedDegreeBound :
    polynomial.canonicalEqualityGatedDegreeBound = 8 :=
  GatePolynomial.polynomial_canonicalEqualityGatedDegreeBound

private theorem exists_pos_of_sum_pos :
    ∀ values : List Nat, 0 < values.sum →
      ∃ value ∈ values, 0 < value
  | [], positive => by simp at positive
  | value :: values, positive => by
      by_cases valuePositive : 0 < value
      · exact ⟨value, by simp, valuePositive⟩
      · have valueZero : value = 0 := Nat.eq_zero_of_not_pos valuePositive
        have tailPositive : 0 < values.sum := by
          simpa [valueZero] using positive
        rcases exists_pos_of_sum_pos values tailPositive with
          ⟨tail, member, tailPositive⟩
        exact ⟨tail, by simp [member], tailPositive⟩

private theorem foldl_mul_zero
    {Index : Type} (factor : Index → F) :
    ∀ items : List Index,
      items.foldl (fun product index => product * factor index) 0 = 0
  | [] => rfl
  | _ :: items => by
      simp only [List.foldl_cons, zero_mul]
      exact foldl_mul_zero factor items

private theorem foldl_mul_eq_zero_of_mem_zero
    {Index : Type} (factor : Index → F) :
    ∀ (items : List Index) (initial : F),
      (∃ index ∈ items, factor index = 0) →
      items.foldl (fun product index => product * factor index) initial = 0
  | [], _, witness => by simp at witness
  | head :: items, initial, witness => by
      rcases witness with ⟨index, member, zero⟩
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · simp only [List.foldl_cons, zero, mul_zero]
        exact foldl_mul_zero factor items
      · simp only [List.foldl_cons]
        exact foldl_mul_eq_zero_of_mem_zero factor items
          (initial * factor head) ⟨index, member, zero⟩

private theorem pow_zero_of_pos (exponent : Nat) (positive : 0 < exponent) :
    pow baseOps (0 : F) exponent = 0 := by
  cases exponent with
  | zero => omega
  | succ exponent => simp [pow, baseOps]

private theorem evaluateMonomial_zero_of_totalDegree_pos
    (monomial : Monomial F matrixCount)
    (positive : 0 < monomial.totalDegree) :
    evaluateMonomial baseOps monomial (fun _ => 0) = 0 := by
  unfold Monomial.totalDegree at positive
  rcases exists_pos_of_sum_pos _ positive with
    ⟨degree, degreeMember, degreePositive⟩
  rcases List.mem_map.mp degreeMember with
    ⟨index, indexMember, indexDegree⟩
  have exponentPositive : 0 < monomial.exponents index := by
    rw [indexDegree]
    exact degreePositive
  have factorZero :
      pow baseOps (0 : F) (monomial.exponents index) = 0 :=
    pow_zero_of_pos _ exponentPositive
  unfold evaluateMonomial
  change (canonicalFinIndices matrixCount).foldl
      (fun product current =>
        product * pow baseOps 0 (monomial.exponents current))
      monomial.coefficient = 0
  exact foldl_mul_eq_zero_of_mem_zero
    (fun current => pow baseOps 0 (monomial.exponents current))
    (canonicalFinIndices matrixCount) monomial.coefficient
    ⟨index, indexMember, factorZero⟩

private theorem foldl_add_eq_zero_of_all_zero
    {Item : Type} (value : Item → F) :
    ∀ items : List Item,
      (∀ item ∈ items, value item = 0) →
      items.foldl (fun total item => total + value item) 0 = 0
  | [], _ => rfl
  | item :: items, allZero => by
      simp only [List.foldl_cons, allZero item (by simp), add_zero]
      apply foldl_add_eq_zero_of_all_zero value items
      intro current member
      exact allZero current (by simp [member])

/-- Canonical padding rows are valid: with every matrix image zero, the fixed
gate polynomial has value zero. -/
theorem polynomial_zeroImages :
    evaluatePolynomial baseOps polynomial (fun _ => 0) = 0 := by
  unfold evaluatePolynomial polynomial GatePolynomial.polynomial
  change GatePolynomial.terms.foldl
      (fun total monomial =>
        total + evaluateMonomial baseOps monomial (fun _ => 0)) 0 = 0
  apply foldl_add_eq_zero_of_all_zero
  intro monomial member
  exact evaluateMonomial_zero_of_totalDegree_pos monomial
    (GatePolynomial.term_totalDegree_pos monomial member)

end NightstreamFPrime.Spec.ProductionRelation
