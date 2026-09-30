import NightstreamFPrime.Gadgets.Polynomial.Horner
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Owns circuit evaluation of one explicit sparse constraint polynomial over the
production quadratic extension.

The polynomial is static relation data. Its point values are symbolic parent
inputs. The gadget mirrors `CCSResidualTable.evaluatePolynomial`. The owned
variant stores each extension product in three rank-one cells and the result
in two cells.
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

/-!
Each monomial multiplies its factors in the specification's index order. The
coefficient scales the first factor, so a monomial of total degree `d > 0`
stores `d - 1` products in `3 * (d - 1)` cells. The last two cells hold the
sum of all terms.
-/

structure Interface (matrixCount : Nat) where
  point : Nat → Fin matrixCount → KExpr

/-- Multiplication by a fixed extension constant. Each coordinate stays
affine in `value`. -/
def scale (coefficient : K) (value : KExpr) : KExpr :=
  ⟨Expr.const coefficient.c0 * value.c0 +
      Expr.const (7 * coefficient.c1) * value.c1,
    Expr.const coefficient.c0 * value.c1 +
      Expr.const coefficient.c1 * value.c0⟩

@[simp] theorem eval_scale (env : Env) (coefficient : K) (value : KExpr) :
    (scale coefficient value).eval env =
      K.mul coefficient (value.eval env) := by
  rfl

/-- The factors of one monomial in the specification's index order. -/
def factors {matrixCount : Nat} (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) : List KExpr :=
  (canonicalFinIndices matrixCount).flatMap fun index =>
    List.replicate (monomial.exponents index) (point index)

/-- Multiply `accumulated` by each factor. Each product owns three cells. -/
def storeProducts (start : Nat) (accumulated : KExpr) :
    List KExpr → Horner.Program
  | [] => ⟨[], accumulated⟩
  | factor :: rest =>
      let tail := storeProducts (start + 3) (Horner.productAt start) rest
      ⟨Horner.mulRecipes start accumulated factor ++ tail.recipes,
        tail.output⟩

/-- One monomial. The coefficient scales the first factor, so a monomial of
total degree `d > 0` stores `d - 1` products. -/
def compileMonomial {matrixCount : Nat} (start : Nat)
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) : Horner.Program :=
  match factors monomial point with
  | [] => ⟨[], constant monomial.coefficient⟩
  | first :: rest =>
      storeProducts start (scale monomial.coefficient first) rest

/-- The terms in specification order, added to `sum`. -/
def compileTerms {matrixCount : Nat} (point : Fin matrixCount → KExpr) :
    Nat → KExpr → List (Monomial K matrixCount) → Horner.Program
  | _, sum, [] => ⟨[], sum⟩
  | start, sum, monomial :: rest =>
      let term := compileMonomial start monomial point
      let tail := compileTerms point (start + term.recipes.length)
        (KExpr.add sum term.output) rest
      ⟨term.recipes ++ tail.recipes, tail.output⟩

/-- Number of stored product cells for the whole polynomial. -/
def productCount {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount) : Nat :=
  (polynomial.terms.map fun monomial => 3 * (monomial.totalDegree - 1)).sum

/-! ### Lengths -/

theorem storeProducts_length (start : Nat) (accumulated : KExpr)
    (rest : List KExpr) :
    (storeProducts start accumulated rest).recipes.length =
      3 * rest.length := by
  induction rest generalizing start accumulated with
  | nil => rfl
  | cons factor rest inductionHypothesis =>
      simp only [storeProducts, List.length_append, Horner.mulRecipes_length,
        inductionHypothesis, List.length_cons]
      omega

theorem factors_length {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    (factors monomial point).length = monomial.totalDegree := by
  simp [factors, List.length_flatMap, Monomial.totalDegree]

theorem compileMonomial_length {matrixCount : Nat} (start : Nat)
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    (compileMonomial start monomial point).recipes.length =
      3 * (monomial.totalDegree - 1) := by
  rw [← factors_length monomial point]
  unfold compileMonomial
  cases factors monomial point with
  | nil => rfl
  | cons first rest =>
      rw [storeProducts_length]
      simp

theorem compileTerms_length {matrixCount : Nat}
    (point : Fin matrixCount → KExpr) :
    ∀ (start : Nat) (sum : KExpr) (terms : List (Monomial K matrixCount)),
      (compileTerms point start sum terms).recipes.length =
        (terms.map fun monomial => 3 * (monomial.totalDegree - 1)).sum
  | _, _, [] => rfl
  | start, sum, monomial :: rest => by
      simp only [compileTerms, List.length_append, compileMonomial_length,
        compileTerms_length point _ _ rest, List.map_cons, List.sum_cons]

/-! ### Causality -/

private theorem storeProducts_causal (start : Nat) (accumulated : KExpr)
    (rest : List KExpr)
    (accumulatedBelow : accumulated.VarsBelow start)
    (restBelow : ∀ factor ∈ rest, factor.VarsBelow start) :
    RecipesCausal start (storeProducts start accumulated rest).recipes ∧
      (storeProducts start accumulated rest).output.VarsBelow
        (start + 3 * rest.length) := by
  induction rest generalizing start accumulated with
  | nil => exact ⟨trivial, by simpa [storeProducts] using accumulatedBelow⟩
  | cons factor rest inductionHypothesis =>
      have factorBelow := restBelow factor (by simp)
      have tailBelow : ∀ current ∈ rest, current.VarsBelow (start + 3) :=
        fun current member =>
          (current.varsBelow_mono (restBelow current (by simp [member]))
            (by omega))
      have productBelow : (Horner.productAt start).VarsBelow (start + 3) := by
        unfold Horner.productAt KExpr.VarsBelow Expr.VarsBelow
        omega
      have tail := inductionHypothesis (start + 3) (Horner.productAt start)
        productBelow tailBelow
      refine ⟨?_, ?_⟩
      · exact Horner.recipesCausal_concat start _ _
          (Horner.mulRecipes_causal start accumulated factor accumulatedBelow
            factorBelow)
          (by simpa using tail.1)
      · have bound : start + 3 + 3 * rest.length =
            start + 3 * (factor :: rest).length := by
          simp only [List.length_cons]
          omega
        simpa [storeProducts, bound] using tail.2

private theorem constant_varsBelow (value : K) (bound : Nat) :
    (constant value).VarsBelow bound :=
  ⟨trivial, trivial⟩

private theorem scale_varsBelow (coefficient : K) (value : KExpr)
    (bound : Nat) (below : value.VarsBelow bound) :
    (scale coefficient value).VarsBelow bound :=
  ⟨⟨⟨trivial, below.1⟩, ⟨trivial, below.2⟩⟩,
    ⟨⟨trivial, below.2⟩, ⟨trivial, below.1⟩⟩⟩

private theorem factors_varsBelow {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) (bound : Nat)
    (pointBelow : ∀ index, (point index).VarsBelow bound) :
    ∀ factor ∈ factors monomial point, factor.VarsBelow bound := by
  intro factor member
  simp only [factors, List.mem_flatMap, List.mem_replicate] at member
  rcases member with ⟨index, _, _, rfl⟩
  exact pointBelow index

private theorem compileMonomial_causal {matrixCount : Nat} (start : Nat)
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr)
    (pointBelow : ∀ index, (point index).VarsBelow start) :
    RecipesCausal start (compileMonomial start monomial point).recipes ∧
      (compileMonomial start monomial point).output.VarsBelow
        (start + (compileMonomial start monomial point).recipes.length) := by
  have factorsBelow := factors_varsBelow monomial point start pointBelow
  unfold compileMonomial
  cases equals : factors monomial point with
  | nil => exact ⟨trivial, constant_varsBelow _ _⟩
  | cons first rest =>
      rw [equals] at factorsBelow
      have chain := storeProducts_causal start
        (scale monomial.coefficient first) rest
        (scale_varsBelow _ _ _ (factorsBelow first (by simp)))
        (fun factor member => factorsBelow factor (by simp [member]))
      rw [storeProducts_length]
      exact chain

private theorem compileTerms_causal {matrixCount : Nat}
    (point : Fin matrixCount → KExpr) (offset : Nat)
    (pointBelow : ∀ index, (point index).VarsBelow offset) :
    ∀ (start : Nat) (sum : KExpr) (terms : List (Monomial K matrixCount)),
      offset ≤ start → sum.VarsBelow start →
      RecipesCausal start (compileTerms point start sum terms).recipes ∧
        (compileTerms point start sum terms).output.VarsBelow
          (start + (compileTerms point start sum terms).recipes.length)
  | start, sum, [], _, sumBelow => ⟨trivial, by simpa [compileTerms] using sumBelow⟩
  | start, sum, monomial :: rest, offsetLe, sumBelow => by
      let term := compileMonomial start monomial point
      have termProof := compileMonomial_causal start monomial point
        (fun index => (point index).varsBelow_mono (pointBelow index) offsetLe)
      have nextSum : (KExpr.add sum term.output).VarsBelow
          (start + term.recipes.length) :=
        ⟨⟨(sum.varsBelow_mono sumBelow (by omega)).1, termProof.2.1⟩,
          ⟨(sum.varsBelow_mono sumBelow (by omega)).2, termProof.2.2⟩⟩
      have tail := compileTerms_causal point offset pointBelow
        (start + term.recipes.length) (KExpr.add sum term.output) rest
        (by omega) nextSum
      refine ⟨?_, ?_⟩
      · exact Horner.recipesCausal_concat start _ _ termProof.1 tail.1
      · have bound : start + term.recipes.length +
            (compileTerms point (start + term.recipes.length)
              (KExpr.add sum term.output) rest).recipes.length =
            start + (term.recipes ++ (compileTerms point
              (start + term.recipes.length)
              (KExpr.add sum term.output) rest).recipes).length := by
          simp only [List.length_append]
          omega
        simpa [compileTerms, term, bound] using tail.2

/-! ### Soundness -/

theorem storeProducts_sound (env : Env) :
    ∀ (start : Nat) (accumulated : KExpr) (rest : List KExpr),
      ConstraintsHold env
        (recipeConstraints start (storeProducts start accumulated rest).recipes) →
      (storeProducts start accumulated rest).output.eval env =
        rest.foldl (fun value factor => K.mul value (factor.eval env))
          (accumulated.eval env)
  | _, _, [], _ => rfl
  | start, accumulated, factor :: rest, rows => by
      have split :
          ConstraintsHold env (recipeConstraints start
            (Horner.mulRecipes start accumulated factor)) ∧
          ConstraintsHold env (recipeConstraints (start + 3)
            (storeProducts (start + 3) (Horner.productAt start) rest).recipes) := by
        have parts := (constraintsHold_append env _ _).mp (by
          rw [← recipeConstraints_append]
          exact rows)
        simpa using parts
      have product := Horner.productAt_sound env start accumulated factor split.1
      have tail := storeProducts_sound env (start + 3) (Horner.productAt start)
        rest split.2
      simp only [storeProducts, List.foldl_cons]
      rw [tail, product]

private theorem foldl_replicate (env : Env) (value : KExpr) :
    ∀ (exponent : Nat) (initial : K),
      (List.replicate exponent value).foldl
          (fun accumulated factor => K.mul accumulated (factor.eval env))
          initial =
        extensionOps.mul initial
          (CCSResidualTable.pow extensionOps (value.eval env) exponent)
  | 0, initial => (extensionLaws.mul_one initial).symm
  | exponent + 1, initial => by
      rw [List.replicate_succ, List.foldl_cons,
        foldl_replicate env value exponent]
      simp only [CCSResidualTable.pow]
      change extensionOps.mul (extensionOps.mul initial (value.eval env))
          (CCSResidualTable.pow extensionOps (value.eval env) exponent) =
        extensionOps.mul initial
          (extensionOps.mul
            (CCSResidualTable.pow extensionOps (value.eval env) exponent)
            (value.eval env))
      rw [extensionLaws.mul_assoc, extensionLaws.mul_comm (value.eval env)]

private theorem foldl_factors {matrixCount : Nat} (env : Env)
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    ∀ (indices : List (Fin matrixCount)) (initial : K),
      (indices.flatMap fun index =>
          List.replicate (monomial.exponents index) (point index)).foldl
          (fun accumulated factor => K.mul accumulated (factor.eval env))
          initial =
        indices.foldl
          (fun accumulated index => extensionOps.mul accumulated
            (CCSResidualTable.pow extensionOps ((point index).eval env)
              (monomial.exponents index)))
          initial
  | [], _ => rfl
  | index :: indices, initial => by
      rw [List.flatMap_cons, List.foldl_append, foldl_replicate,
        List.foldl_cons]
      exact foldl_factors env monomial point indices _

theorem compileMonomial_sound {matrixCount : Nat} (env : Env) (start : Nat)
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr)
    (rows : ConstraintsHold env
      (recipeConstraints start (compileMonomial start monomial point).recipes)) :
    (compileMonomial start monomial point).output.eval env =
      CCSResidualTable.evaluateMonomial extensionOps monomial
        (fun index => (point index).eval env) := by
  have folded : (factors monomial point).foldl
      (fun accumulated factor => K.mul accumulated (factor.eval env))
      monomial.coefficient =
    CCSResidualTable.evaluateMonomial extensionOps monomial
      (fun index => (point index).eval env) :=
    foldl_factors env monomial point _ _
  rw [← folded]
  unfold compileMonomial at rows ⊢
  cases equals : factors monomial point with
  | nil => exact eval_constant env _
  | cons first rest =>
      rw [equals] at rows
      rw [storeProducts_sound env start _ rest rows, eval_scale]
      rfl

theorem compileTerms_sound {matrixCount : Nat} (env : Env)
    (point : Fin matrixCount → KExpr) :
    ∀ (start : Nat) (sum : KExpr) (terms : List (Monomial K matrixCount)),
      ConstraintsHold env
        (recipeConstraints start (compileTerms point start sum terms).recipes) →
      (compileTerms point start sum terms).output.eval env =
        terms.foldl
          (fun accumulated monomial => extensionOps.add accumulated
            (CCSResidualTable.evaluateMonomial extensionOps monomial
              (fun index => (point index).eval env)))
          (sum.eval env)
  | _, _, [], _ => rfl
  | start, sum, monomial :: rest, rows => by
      let term := compileMonomial start monomial point
      have split := (constraintsHold_append env _ _).mp (by
        rw [← recipeConstraints_append]
        exact rows)
      have termSound := compileMonomial_sound env start monomial point split.1
      have tail := compileTerms_sound env point (start + term.recipes.length)
        (KExpr.add sum term.output) rest split.2
      simp only [compileTerms, List.foldl_cons]
      rw [tail, KExpr.eval_add, termSound]
      rfl

/-! ### Owned circuit -/

def program {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) : Horner.Program :=
  compileTerms (interface.point offset) offset KExpr.zero polynomial.terms

theorem program_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (program polynomial interface offset).recipes.length =
      productCount polynomial :=
  compileTerms_length _ _ _ _

/-- Stored products, then the two result cells. -/
def recipes {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) : List Expr :=
  (program polynomial interface offset).recipes ++
    [(program polynomial interface offset).output.c0,
      (program polynomial interface offset).output.c1]

@[simp] theorem recipes_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (recipes polynomial interface offset).length =
      productCount polynomial + 2 := by
  simp [recipes, program_length]

def output {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (_interface : Interface matrixCount) (offset : Nat) : KExpr :=
  ⟨Expr.var (offset + productCount polynomial),
    Expr.var (offset + productCount polynomial + 1)⟩

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

def opsAt {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset
    (recipes polynomial interface offset))]

def main {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) : Circuit Unit :=
  fun offset =>
    ((), offset + (productCount polynomial + 2),
      opsAt polynomial interface offset)

theorem flatConstraints_opsAt {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    flatConstraints (opsAt polynomial interface offset) =
      recipeConstraints offset (recipes polynomial interface offset) := by
  simp [flatConstraints, opsAt, Op.flatConstraints]

private theorem recipes_causal {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset (fun _ => 0)) :
    RecipesCausal offset (recipes polynomial interface offset) := by
  have terms : RecipesCausal offset
        (program polynomial interface offset).recipes ∧
      (program polynomial interface offset).output.VarsBelow
        (offset + (program polynomial interface offset).recipes.length) :=
    compileTerms_causal (interface.point offset) offset assumptions offset
      KExpr.zero polynomial.terms le_rfl ⟨trivial, trivial⟩
  apply Horner.recipesCausal_concat offset _ _ terms.1
  exact ⟨terms.2.1,
    ⟨Expr.VarsBelow.mono _ terms.2.2 (Nat.le_succ _), trivial⟩⟩

private theorem output_eq_program_of_rows {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) (env : Env)
    (rows : ConstraintsHold env
      (recipeConstraints (offset + productCount polynomial)
        [(program polynomial interface offset).output.c0,
          (program polynomial interface offset).output.c1])) :
    (output polynomial interface offset).eval env =
      (program polynomial interface offset).output.eval env := by
  have c0Row := rows
    (Expr.var (offset + productCount polynomial) -
      (program polynomial interface offset).output.c0) (by
      simp [recipeConstraints])
  have c1Row := rows
    (Expr.var (offset + productCount polynomial + 1) -
      (program polynomial interface offset).output.c1) (by
      simp [recipeConstraints])
  have c0Eq : env (offset + productCount polynomial) =
      (program polynomial interface offset).output.c0.eval env :=
    sub_eq_zero.mp (by simpa using c0Row)
  have c1Eq : env (offset + productCount polynomial + 1) =
      (program polynomial interface offset).output.c1.eval env :=
    sub_eq_zero.mp (by simpa using c1Row)
  change K.mk (env (offset + productCount polynomial))
    (env (offset + productCount polynomial + 1)) = K.mk _ _
  rw [c0Eq, c1Eq]

private theorem split_rows {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) (env : Env)
    (rows : ConstraintsHold env
      (recipeConstraints offset (recipes polynomial interface offset))) :
    ConstraintsHold env (recipeConstraints offset
        (program polynomial interface offset).recipes) ∧
      ConstraintsHold env
        (recipeConstraints (offset + productCount polynomial)
          [(program polynomial interface offset).output.c0,
            (program polynomial interface offset).output.c1]) := by
  have parts := (constraintsHold_append env _ _).mp (by
    rw [← recipeConstraints_append]
    exact rows)
  rw [program_length] at parts
  exact parts

private theorem specHolds_of_recipeRows {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (rows : ConstraintsHold env
      (recipeConstraints offset (recipes polynomial interface offset))) :
    SpecHolds polynomial interface offset env := by
  have split := split_rows polynomial interface offset env rows
  rw [SpecHolds, output_eq_program_of_rows polynomial interface offset env
    split.2]
  exact compileTerms_sound env (interface.point offset) offset KExpr.zero
    polynomial.terms split.1

private theorem execute {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (opsAt polynomial interface offset)) ∧
      holdsFlat completed (opsAt polynomial interface offset) := by
  let completed := executeRecipes env offset
    (recipes polynomial interface offset)
  have causal := recipes_causal polynomial interface offset assumptions
  refine ⟨completed, ?_, ?_⟩
  · change AgreesOutside env completed offset
      (recipes polynomial interface offset).length
    exact executeRecipes_agreesOutside env offset _
  · change ConstraintsHold completed
      (flatConstraints (opsAt polynomial interface offset))
    rw [flatConstraints_opsAt]
    exact executeRecipes_holds_recipeConstraints env offset _ causal

def circuit {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) : FormalCircuit where
  main := main polynomial interface
  assumptions := Assumptions polynomial interface
  spec := SpecHolds polynomial interface
  soundness := by
    intro env offset _assumptions rows
    exact specHolds_of_recipeRows polynomial interface env offset
      (rows (Op.witness (WitnessBatch.arithmetic offset
        (recipes polynomial interface offset))) (by
          change Op.witness (WitnessBatch.arithmetic offset
              (recipes polynomial interface offset)) ∈
            [Op.witness (WitnessBatch.arithmetic offset
              (recipes polynomial interface offset))]
          simp))
  completeness := fun env offset assumptions _specification =>
    execute polynomial interface env offset assumptions

theorem soundness {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset env)
    (rows : holds env (Circuit.ops (circuit polynomial interface).main offset)) :
    SpecHolds polynomial interface offset env :=
  (circuit polynomial interface).soundness env offset assumptions rows

/-- Honest execution constructs the owned result with no semantic premise. -/
theorem build {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit polynomial interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit polynomial interface).main offset) :=
  execute polynomial interface env offset assumptions

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
    localLength (Circuit.ops (circuit polynomial interface).main offset) =
      productCount polynomial + 2 := by
  change (recipes polynomial interface offset).length + 0 = _
  simp

theorem operations_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (Circuit.ops (circuit polynomial interface).main offset).length = 1 := by
  rfl

theorem flatConstraints_length {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit polynomial interface).main offset)).length =
      productCount polynomial + 2 := by
  change (flatConstraints (opsAt polynomial interface offset)).length = _
  rw [flatConstraints_opsAt, recipeConstraints_length, recipes_length]

theorem flatConstraints_varsBelow {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (assumptions : Assumptions polynomial interface offset (fun _ => 0)) :
    ∀ constraint ∈ flatConstraints
      (Circuit.ops (circuit polynomial interface).main offset),
      constraint.VarsBelow (offset + (productCount polynomial + 2)) := by
  change ∀ constraint ∈ flatConstraints (opsAt polynomial interface offset),
    constraint.VarsBelow (offset + (productCount polynomial + 2))
  rw [flatConstraints_opsAt, ← recipes_length polynomial interface offset]
  exact recipeConstraints_varsBelow_of_causal offset _
    (recipes_causal polynomial interface offset assumptions)

end Owned

end NightstreamFPrime.Gadgets.Polynomial.Sparse
