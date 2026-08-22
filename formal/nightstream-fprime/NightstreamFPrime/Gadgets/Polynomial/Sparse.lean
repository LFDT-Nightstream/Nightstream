import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Owns circuit evaluation of one explicit sparse constraint polynomial over the
production quadratic extension.

The polynomial is static relation data. Its point values are symbolic parent
inputs. The gadget mirrors `CCSResidualTable.evaluatePolynomial`, emits only
the two output-component assertions, and allocates no witness variables.
-/

namespace NightstreamFPrime.Gadgets.Polynomial.Sparse

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

def constant (value : K) : KExpr :=
  ⟨Expr.const value.c0, Expr.const value.c1⟩

@[simp] theorem eval_constant (env : Env) (value : K) :
    (constant value).eval env = value := by
  cases value
  rfl

def pow (value : KExpr) : Nat → KExpr
  | 0 => KExpr.one
  | exponent + 1 => KExpr.mul (pow value exponent) value

theorem eval_pow (env : Env) (value : KExpr) : ∀ exponent,
    (pow value exponent).eval env =
      CCSResidualTable.pow extensionOps (value.eval env) exponent
  | 0 => by rfl
  | exponent + 1 => by
      simp only [pow, CCSResidualTable.pow, KExpr.eval_mul]
      rw [eval_pow env value exponent]
      rfl

theorem pow_varsBelow (value : KExpr) (bound : Nat)
    (below : value.VarsBelow bound) : ∀ exponent,
    (pow value exponent).VarsBelow bound
  | 0 => ⟨trivial, trivial⟩
  | exponent + 1 =>
      KExpr.mul_varsBelow _ _ bound (pow_varsBelow value bound below exponent)
        below

def evaluateMonomial {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) : KExpr :=
  (canonicalFinIndices matrixCount).foldl
    (fun accumulated index =>
      KExpr.mul accumulated (pow (point index) (monomial.exponents index)))
    (constant monomial.coefficient)

private theorem eval_monomialFold {matrixCount : Nat}
    (env : Env) (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    ∀ (indices : List (Fin matrixCount)) (initial : KExpr),
      (indices.foldl
          (fun accumulated index => KExpr.mul accumulated
            (pow (point index) (monomial.exponents index))) initial).eval env =
        indices.foldl
          (fun accumulated index => extensionOps.mul accumulated
            (CCSResidualTable.pow extensionOps
              ((point index).eval env) (monomial.exponents index)))
          (initial.eval env)
  | [], _ => rfl
  | index :: indices, initial => by
      simp only [List.foldl_cons]
      rw [eval_monomialFold env monomial point indices]
      simp only [KExpr.eval_mul, eval_pow]
      rfl

theorem eval_evaluateMonomial {matrixCount : Nat} (env : Env)
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    (evaluateMonomial monomial point).eval env =
      CCSResidualTable.evaluateMonomial extensionOps monomial
        (fun index => (point index).eval env) := by
  unfold evaluateMonomial CCSResidualTable.evaluateMonomial
  rw [eval_monomialFold]
  simp [eval_constant]

private theorem monomialFold_varsBelow {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) (bound : Nat)
    (pointBelow : ∀ index, (point index).VarsBelow bound) :
    ∀ (indices : List (Fin matrixCount)) (initial : KExpr),
      initial.VarsBelow bound →
      (indices.foldl
        (fun accumulated index => KExpr.mul accumulated
          (pow (point index) (monomial.exponents index))) initial).VarsBelow bound
  | [], _, initialBelow => initialBelow
  | index :: indices, initial, initialBelow => by
      apply monomialFold_varsBelow monomial point bound pointBelow indices
      exact KExpr.mul_varsBelow _ _ bound initialBelow
        (pow_varsBelow (point index) bound (pointBelow index)
          (monomial.exponents index))

theorem evaluateMonomial_varsBelow {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) (bound : Nat)
    (pointBelow : ∀ index, (point index).VarsBelow bound) :
    (evaluateMonomial monomial point).VarsBelow bound := by
  apply monomialFold_varsBelow monomial point bound pointBelow
  exact ⟨trivial, trivial⟩

def evaluate {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (point : Fin matrixCount → KExpr) : KExpr :=
  polynomial.terms.foldl
    (fun accumulated monomial =>
      KExpr.add accumulated (evaluateMonomial monomial point))
    KExpr.zero

private theorem eval_polynomialFold {matrixCount : Nat}
    (env : Env) (point : Fin matrixCount → KExpr) :
    ∀ (terms : List (Monomial K matrixCount)) (initial : KExpr),
      (terms.foldl
          (fun accumulated monomial =>
            KExpr.add accumulated (evaluateMonomial monomial point))
          initial).eval env =
        terms.foldl
          (fun accumulated monomial => extensionOps.add accumulated
            (CCSResidualTable.evaluateMonomial extensionOps monomial
              (fun index => (point index).eval env)))
          (initial.eval env)
  | [], _ => rfl
  | monomial :: terms, initial => by
      simp only [List.foldl_cons]
      rw [eval_polynomialFold env point terms]
      simp only [KExpr.eval_add, eval_evaluateMonomial]
      rfl

theorem eval_evaluate {matrixCount : Nat} (env : Env)
    (polynomial : ConstraintPolynomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    (evaluate polynomial point).eval env =
      CCSResidualTable.evaluatePolynomial extensionOps polynomial
        (fun index => (point index).eval env) := by
  unfold evaluate CCSResidualTable.evaluatePolynomial
  rw [eval_polynomialFold]
  rfl

private theorem polynomialFold_varsBelow {matrixCount : Nat}
    (point : Fin matrixCount → KExpr) (bound : Nat)
    (pointBelow : ∀ index, (point index).VarsBelow bound) :
    ∀ (terms : List (Monomial K matrixCount)) (initial : KExpr),
      initial.VarsBelow bound →
      (terms.foldl
        (fun accumulated monomial =>
          KExpr.add accumulated (evaluateMonomial monomial point))
        initial).VarsBelow bound
  | [], _, initialBelow => initialBelow
  | monomial :: terms, initial, initialBelow => by
      apply polynomialFold_varsBelow point bound pointBelow terms
      exact KExpr.add_varsBelow _ _ bound initialBelow
        (evaluateMonomial_varsBelow monomial point bound pointBelow)

theorem evaluate_varsBelow {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (point : Fin matrixCount → KExpr) (bound : Nat)
    (pointBelow : ∀ index, (point index).VarsBelow bound) :
    (evaluate polynomial point).VarsBelow bound := by
  apply polynomialFold_varsBelow point bound pointBelow
  exact ⟨trivial, trivial⟩

structure Interface (matrixCount : Nat) where
  point : Nat → Fin matrixCount → KExpr
  expected : Nat → KExpr

def Interface.VarsBelow {matrixCount : Nat}
    (interface : Interface matrixCount) (offset : Nat) : Prop :=
  (∀ index, (interface.point offset index).VarsBelow offset) ∧
    (interface.expected offset).VarsBelow offset

def expression {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) : KExpr :=
  evaluate polynomial (interface.point offset)

def constraints {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) : List Expr :=
  KExpr.equalities (interface.expected offset)
    (expression polynomial interface offset)

def main {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) : Circuit Unit :=
  fun offset => ((), offset, (constraints polynomial interface offset).map
    Op.assertZero)

def Assumptions {matrixCount : Nat}
    (_polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) (_env : Env) : Prop :=
  interface.VarsBelow offset

def SpecHolds {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) (env : Env) : Prop :=
  (interface.expected offset).eval env =
    CCSResidualTable.evaluatePolynomial extensionOps polynomial
      (fun index => (interface.point offset index).eval env)

private theorem holds_assertions_iff (env : Env) (expressions : List Expr) :
    holds env (expressions.map Op.assertZero) ↔
      ConstraintsHold env expressions := by
  induction expressions with
  | nil => simp [ConstraintsHold]
  | cons expression expressions inductionHypothesis =>
      simp only [List.map_cons, holds_cons, Op.holds_assertZero,
        inductionHypothesis]
      constructor
      · rintro ⟨head, tail⟩ current member
        rcases List.mem_cons.mp member with rfl | member
        · exact head
        · exact tail current member
      · intro all
        exact ⟨all expression (by simp), fun current member =>
          all current (by simp [member])⟩

private theorem flatConstraints_assertions_eq (expressions : List Expr) :
    flatConstraints (expressions.map Op.assertZero) = expressions := by
  induction expressions with
  | nil => rfl
  | cons expression expressions inductionHypothesis =>
      change expression :: flatConstraints (expressions.map Op.assertZero) =
        expression :: expressions
      rw [inductionHypothesis]

def circuit {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) : FormalCircuit where
  main := main polynomial interface
  assumptions := Assumptions polynomial interface
  spec := SpecHolds polynomial interface
  soundness := by
    intro env offset _ rows
    have equalities :
        (interface.expected offset).eval env =
          (expression polynomial interface offset).eval env :=
      (KExpr.equalities_hold_iff env _ _).mp <|
        (holds_assertions_iff env _).mp rows
    exact equalities.trans <| by
      unfold expression
      exact eval_evaluate env polynomial (interface.point offset)
  completeness := by
    intro env offset _ specification
    refine ⟨env, ?_, ?_⟩
    · intro index outside
      rfl
    · unfold holdsFlat
      change ConstraintsHold env (flatConstraints
        ((constraints polynomial interface offset).map Op.assertZero))
      rw [flatConstraints_assertions_eq]
      apply (KExpr.equalities_hold_iff env _ _).mpr
      exact specification.trans <| by
        unfold expression
        exact (eval_evaluate env polynomial (interface.point offset)).symm

theorem soundness {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset env)
    (rows : holds env (Circuit.ops (circuit polynomial interface).main offset)) :
    SpecHolds polynomial interface offset env :=
  (circuit polynomial interface).soundness env offset assumptions rows

theorem completeness {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset env)
    (specification : SpecHolds polynomial interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit polynomial interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit polynomial interface).main offset) :=
  (circuit polynomial interface).completeness env offset assumptions
    specification

/-- The sparse polynomial specification is stable when every external input
wire is unchanged. -/
theorem specHolds_of_agree_below {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (before after : Env)
    (assumptions : Assumptions polynomial interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds polynomial interface offset before) :
    SpecHolds polynomial interface offset after := by
  have expectedEq : (interface.expected offset).eval after =
      (interface.expected offset).eval before :=
    (interface.expected offset).eval_eq_of_agree_below offset after before
      assumptions.2 agrees
  have pointEq :
      (fun index => (interface.point offset index).eval after) =
        fun index => (interface.point offset index).eval before := by
    funext index
    exact (interface.point offset index).eval_eq_of_agree_below offset
      after before (assumptions.1 index) agrees
  unfold SpecHolds at specification ⊢
  rw [expectedEq, pointEq]
  exact specification

theorem localLength_eq {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    localLength (Circuit.ops (circuit polynomial interface).main offset) = 0 := by
  change (List.map Op.localLength
    ((constraints polynomial interface offset).map Op.assertZero)).sum = 0
  rw [List.map_map]
  simp [Function.comp_def, Op.localLength]

theorem operations_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (Circuit.ops (circuit polynomial interface).main offset).length = 2 := by
  change ((constraints polynomial interface offset).map Op.assertZero).length = 2
  simp [constraints, KExpr.equalities]

theorem flatConstraints_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit polynomial interface).main offset)).length = 2 := by
  change (flatConstraints
    ((constraints polynomial interface offset).map Op.assertZero)).length = 2
  rw [flatConstraints_assertions_eq]
  simp [constraints, KExpr.equalities]

theorem flatConstraints_varsBelow {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (assumptions : interface.VarsBelow offset) :
    ∀ constraint ∈ flatConstraints
      (Circuit.ops (circuit polynomial interface).main offset),
      constraint.VarsBelow offset := by
  change ∀ constraint ∈ flatConstraints
    ((constraints polynomial interface offset).map Op.assertZero),
      constraint.VarsBelow offset
  rw [flatConstraints_assertions_eq]
  apply KExpr.equalities_varsBelow
  · exact assumptions.2
  · unfold expression
    exact evaluate_varsBelow polynomial (interface.point offset) offset
      assumptions.1

/-! ## Child-owned output variant -/

namespace Owned

structure Interface (matrixCount : Nat) where
  point : Nat → Fin matrixCount → KExpr

def output {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) : KExpr :=
  evaluate polynomial (interface.point offset)

def Assumptions {matrixCount : Nat}
    (_polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) (_env : Env) : Prop :=
  ∀ index, (interface.point offset index).VarsBelow offset

def SpecHolds {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) (env : Env) : Prop :=
  (output polynomial interface offset).eval env =
    CCSResidualTable.evaluatePolynomial extensionOps polynomial
      (fun index => (interface.point offset index).eval env)

def main {matrixCount : Nat}
    (_polynomial : ConstraintPolynomial K matrixCount)
    (_interface : Interface matrixCount) : Circuit Unit :=
  fun offset => ((), offset, [])

def circuit {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) : FormalCircuit where
  main := main polynomial interface
  assumptions := Assumptions polynomial interface
  spec := SpecHolds polynomial interface
  soundness := by
    intro env offset assumptions rows
    exact eval_evaluate env polynomial (interface.point offset)
  completeness := by
    intro env offset assumptions specification
    refine ⟨env, fun _ _ => rfl, ?_⟩
    change ConstraintsHold env []
    intro expression member
    simp at member

theorem soundness {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset env)
    (rows : holds env (Circuit.ops (circuit polynomial interface).main offset)) :
    SpecHolds polynomial interface offset env :=
  (circuit polynomial interface).soundness env offset assumptions rows

theorem build {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit polynomial interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit polynomial interface).main offset) := by
  apply (circuit polynomial interface).completeness env offset assumptions
  exact eval_evaluate env polynomial (interface.point offset)

theorem completeness {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset env)
    (_specification : SpecHolds polynomial interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit polynomial interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit polynomial interface).main offset) :=
  build polynomial interface env offset assumptions

theorem localLength_eq {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    localLength (Circuit.ops (circuit polynomial interface).main offset) = 0 := by
  rfl

theorem operations_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (Circuit.ops (circuit polynomial interface).main offset).length = 0 := by
  rfl

theorem flatConstraints_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit polynomial interface).main offset)).length = 0 := by
  rfl

theorem flatConstraints_varsBelow {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    ∀ constraint ∈ flatConstraints
      (Circuit.ops (circuit polynomial interface).main offset),
      constraint.VarsBelow offset := by
  change ∀ constraint ∈ ([] : List Expr), constraint.VarsBelow offset
  intro constraint member
  simp at member

end Owned

end NightstreamFPrime.Gadgets.Polynomial.Sparse
