import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Owns circuit evaluation of one explicit sparse constraint polynomial over the
production quadratic extension.

The polynomial is static relation data. Its point values are symbolic parent
inputs. The gadget mirrors `CCSResidualTable.evaluatePolynomial`. The owned
variant materializes one two-component result with two checked recipes.
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

/-- Skip zero powers instead of materializing multiplication by one. -/
def multiplyPower (accumulated value : KExpr) (exponent : Nat) : KExpr :=
  if exponent = 0 then accumulated else KExpr.mul accumulated (pow value exponent)

theorem eval_multiplyPower (env : Env) (accumulated value : KExpr)
    (exponent : Nat) :
    (multiplyPower accumulated value exponent).eval env =
      extensionOps.mul (accumulated.eval env)
        (CCSResidualTable.pow extensionOps (value.eval env) exponent) := by
  by_cases zero : exponent = 0
  · subst exponent
    simp [multiplyPower, CCSResidualTable.pow, extensionLaws.mul_one]
  · simp only [multiplyPower, zero, if_false, KExpr.eval_mul]
    rw [eval_pow]
    rfl

theorem multiplyPower_varsBelow (accumulated value : KExpr) (exponent bound : Nat)
    (accumulatedBelow : accumulated.VarsBelow bound)
    (valueBelow : value.VarsBelow bound) :
    (multiplyPower accumulated value exponent).VarsBelow bound := by
  by_cases zero : exponent = 0
  · simp [multiplyPower, zero, accumulatedBelow]
  · simp only [multiplyPower, zero, if_false]
    exact KExpr.mul_varsBelow _ _ bound accumulatedBelow
      (pow_varsBelow value bound valueBelow exponent)

def evaluateMonomial {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) : KExpr :=
  (canonicalFinIndices matrixCount).foldl
    (fun accumulated index =>
      multiplyPower accumulated (point index) (monomial.exponents index))
    (constant monomial.coefficient)

private theorem eval_monomialFold {matrixCount : Nat}
    (env : Env) (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    ∀ (indices : List (Fin matrixCount)) (initial : KExpr),
      (indices.foldl
          (fun accumulated index => multiplyPower accumulated (point index)
            (monomial.exponents index)) initial).eval env =
        indices.foldl
          (fun accumulated index => extensionOps.mul accumulated
            (CCSResidualTable.pow extensionOps
              ((point index).eval env) (monomial.exponents index)))
          (initial.eval env)
  | [], _ => rfl
  | index :: indices, initial => by
      simp only [List.foldl_cons]
      rw [eval_monomialFold env monomial point indices]
      rw [eval_multiplyPower]

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
        (fun accumulated index => multiplyPower accumulated (point index)
          (monomial.exponents index)) initial).VarsBelow bound
  | [], _, initialBelow => initialBelow
  | index :: indices, initial, initialBelow => by
      apply monomialFold_varsBelow monomial point bound pointBelow indices
      exact multiplyPower_varsBelow initial (point index)
        (monomial.exponents index) bound initialBelow (pointBelow index)

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

def expression {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) : KExpr :=
  evaluate polynomial (interface.point offset)

def recipes {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) : List Expr :=
  [(expression polynomial interface offset).c0,
    (expression polynomial interface offset).c1]

def output {matrixCount : Nat}
    (_polynomial : ConstraintPolynomial K matrixCount)
    (_interface : Interface matrixCount) (offset : Nat) : KExpr :=
  ⟨Expr.var offset, Expr.var (offset + 1)⟩

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
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) : Circuit Unit :=
  fun offset =>
    ((), offset + 2,
      [Op.witness (WitnessBatch.arithmetic offset
        (recipes polynomial interface offset))])

private theorem expression_varsBelow {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset (fun _ => 0)) :
    (expression polynomial interface offset).VarsBelow offset := by
  unfold expression
  exact evaluate_varsBelow polynomial (interface.point offset) offset
    assumptions

private theorem recipes_causal {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset (fun _ => 0)) :
    RecipesCausal offset (recipes polynomial interface offset) := by
  apply recipesCausal_of_all_below
  intro recipe member
  have below := expression_varsBelow polynomial interface offset assumptions
  simp only [recipes, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact below.1
  · exact below.2

private theorem output_eq_expression_of_rows {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) (env : Env)
    (rows : ConstraintsHold env
      (recipeConstraints offset (recipes polynomial interface offset))) :
    (output polynomial interface offset).eval env =
      (expression polynomial interface offset).eval env := by
  have c0Row := rows
    (Expr.var offset - (expression polynomial interface offset).c0) (by
      simp [recipes, recipeConstraints])
  have c1Row := rows
    (Expr.var (offset + 1) - (expression polynomial interface offset).c1) (by
      simp [recipes, recipeConstraints])
  have c0Eq : env offset =
      (expression polynomial interface offset).c0.eval env :=
    sub_eq_zero.mp (by simpa using c0Row)
  have c1Eq : env (offset + 1) =
      (expression polynomial interface offset).c1.eval env :=
    sub_eq_zero.mp (by simpa using c1Row)
  change K.mk (env offset) (env (offset + 1)) = K.mk _ _
  rw [c0Eq, c1Eq]

def circuit {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) : FormalCircuit where
  main := main polynomial interface
  assumptions := Assumptions polynomial interface
  spec := SpecHolds polynomial interface
  soundness := by
    intro env offset _assumptions rows
    have recipeRows : ConstraintsHold env
        (recipeConstraints offset (recipes polynomial interface offset)) :=
      rows (Op.witness (WitnessBatch.arithmetic offset
        (recipes polynomial interface offset)))
        (by
          change Op.witness (WitnessBatch.arithmetic offset
              (recipes polynomial interface offset)) ∈
            [Op.witness (WitnessBatch.arithmetic offset
              (recipes polynomial interface offset))]
          simp)
    exact (output_eq_expression_of_rows polynomial interface offset env
      recipeRows).trans (by
        unfold expression
        exact eval_evaluate env polynomial (interface.point offset))
  completeness := by
    intro env offset assumptions _specification
    let completed := executeRecipes env offset
      (recipes polynomial interface offset)
    have assumptionsAtZero :
        Assumptions polynomial interface offset (fun _ => 0) := assumptions
    have causal := recipes_causal polynomial interface offset assumptionsAtZero
    have recipeRows := executeRecipes_holds_recipeConstraints env offset
      (recipes polynomial interface offset) causal
    refine ⟨completed, ?_, ?_⟩
    · simpa [completed] using executeRecipes_agreesOutside env offset
        (recipes polynomial interface offset)
    · change ConstraintsHold completed
        (recipeConstraints offset (recipes polynomial interface offset))
      exact recipeRows

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
  let completed := executeRecipes env offset
    (recipes polynomial interface offset)
  have assumptionsAtZero :
      Assumptions polynomial interface offset (fun _ => 0) := assumptions
  have causal := recipes_causal polynomial interface offset assumptionsAtZero
  have recipeRows := executeRecipes_holds_recipeConstraints env offset
    (recipes polynomial interface offset) causal
  refine ⟨completed, ?_, ?_⟩
  · simpa [completed] using executeRecipes_agreesOutside env offset
      (recipes polynomial interface offset)
  · change ConstraintsHold completed
      (recipeConstraints offset (recipes polynomial interface offset))
    exact recipeRows

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
    localLength (Circuit.ops (circuit polynomial interface).main offset) = 2 := by
  rfl

theorem operations_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (Circuit.ops (circuit polynomial interface).main offset).length = 1 := by
  rfl

theorem flatConstraints_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit polynomial interface).main offset)).length = 2 := by
  change (recipeConstraints offset
    (recipes polynomial interface offset)).length = 2
  rw [recipeConstraints_length]
  rfl

theorem flatConstraints_varsBelow {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset (fun _ => 0)) :
    ∀ constraint ∈ flatConstraints
      (Circuit.ops (circuit polynomial interface).main offset),
      constraint.VarsBelow (offset + 2) := by
  change ∀ constraint ∈ recipeConstraints offset
      (recipes polynomial interface offset),
    constraint.VarsBelow (offset + 2)
  exact recipeConstraints_varsBelow_of_causal offset _
    (recipes_causal polynomial interface offset assumptions)

end Owned

end NightstreamFPrime.Gadgets.Polynomial.Sparse
