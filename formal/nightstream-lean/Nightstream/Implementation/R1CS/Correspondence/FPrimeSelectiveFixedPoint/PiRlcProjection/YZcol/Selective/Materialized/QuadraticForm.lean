import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm

/-!
Sparse degree-two forms used to compare compact rewrite recurrences with the
source projection equations.

Owns: field evaluation, commutative monomial construction, coefficient
normalization, and evaluation-preserving linear/product embeddings.

Does not own: generated data, source schedules, row satisfaction, selector
truth, protocol authority, security events, or permission to remove rows.

Emits constraints: no.

| Normal-form leaf | Mathematical obligation | Authority class |
|---|---|---|
| monomial | canonical commutative degree-one/two key | derived |
| product embedding | multiply two sparse linear forms exactly | derived |
| equivalence | normalized quadratic forms have equal evaluation | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.QuadraticForm

open Nightstream.SuperNeo.Concrete

abbrev LinearTerm :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.Term

abbrev InputForm := List LinearTerm

def linearEval (assignment : Nat → F) (form : InputForm) : F :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.eval
    assignment form

inductive Monomial where
  | linear (column : Nat)
  | quadratic (left right : Nat)
deriving DecidableEq, Repr

/-- Multiplication is commutative, so quadratic keys are oriented once at
construction time. -/
def Monomial.product (left right : Nat) : Monomial :=
  if left ≤ right then .quadratic left right else .quadratic right left

def Monomial.key : Monomial → Nat × Nat × Nat
  | .linear column => (0, column, 0)
  | .quadratic left right => (1, left, right)

def Monomial.Before (left right : Monomial) : Prop :=
  left.key.1 < right.key.1 ∨
    (left.key.1 = right.key.1 ∧
      (left.key.2.1 < right.key.2.1 ∨
        (left.key.2.1 = right.key.2.1 ∧
          left.key.2.2 < right.key.2.2)))

instance (left right : Monomial) : Decidable (left.Before right) := by
  unfold Monomial.Before
  infer_instance

abbrev Term := Monomial × F
abbrev Form := List Term

def monomialValue (assignment : Nat → F) : Monomial → F
  | .linear column => assignment column
  | .quadratic left right => assignment left * assignment right

def termValue (assignment : Nat → F) (term : Term) : F :=
  term.2 * monomialValue assignment term.1

def eval (assignment : Nat → F) : Form → F
  | [] => 0
  | term :: rest => termValue assignment term + eval assignment rest

private theorem fadd_assoc (a b c : F) :
    (a + b) + c = a + (b + c) := Lean.Grind.Fin.add_assoc _ _ _

private theorem fadd_comm (a b : F) : a + b = b + a :=
  Lean.Grind.Fin.add_comm _ _

private theorem fadd_left_comm (a b c : F) :
    a + (b + c) = b + (a + c) := by
  rw [← fadd_assoc, fadd_comm a b, fadd_assoc]

private theorem fadd_mul (a b c : F) :
    (a + b) * c = a * c + b * c := by
  calc
    (a + b) * c = c * (a + b) := Fin.mul_comm _ _
    _ = c * a + c * b := Lean.Grind.Fin.left_distrib _ _ _
    _ = a * c + b * c := by
      rw [Fin.mul_comm c a, Fin.mul_comm c b]

private theorem fmul_add (a b c : F) :
    a * (b + c) = a * b + a * c :=
  Lean.Grind.Fin.left_distrib _ _ _

def insert (term : Term) : Form → Form
  | [] => if term.2 = 0 then [] else [term]
  | head :: rest =>
      if term.2 = 0 then
        head :: rest
      else if term.1.Before head.1 then
        term :: head :: rest
      else if term.1 = head.1 then
        let coefficient := term.2 + head.2
        if coefficient = 0 then rest else (head.1, coefficient) :: rest
      else
        head :: insert term rest

theorem eval_insert (assignment : Nat → F) (term : Term) :
    ∀ terms, eval assignment (insert term terms) =
      termValue assignment term + eval assignment terms := by
  rcases term with ⟨termMonomial, termCoefficient⟩
  intro terms
  induction terms with
  | nil =>
      by_cases coefficientZero : termCoefficient = 0
      · simp [insert, coefficientZero, termValue, eval, Fin.zero_mul]
      · simp [insert, coefficientZero, termValue, eval]
  | cons head rest inductionHypothesis =>
      rcases head with ⟨headMonomial, headCoefficient⟩
      by_cases coefficientZero : termCoefficient = 0
      · simp [insert, coefficientZero, termValue, eval, Fin.zero_mul]
      · simp only [insert, coefficientZero, ↓reduceIte]
        by_cases before : termMonomial.Before headMonomial
        · simp only [before, ↓reduceIte, eval, termValue]
        · simp only [before, ↓reduceIte]
          by_cases same : termMonomial = headMonomial
          · subst headMonomial
            simp only [↓reduceIte, eval, termValue]
            by_cases sumZero : termCoefficient + headCoefficient = 0
            · simp only [sumZero, ↓reduceIte]
              have multiplied := congrArg
                (fun value : F => value * monomialValue assignment termMonomial)
                sumZero
              change
                (termCoefficient + headCoefficient) *
                    monomialValue assignment termMonomial =
                  0 * monomialValue assignment termMonomial at multiplied
              rw [fadd_mul, Fin.zero_mul] at multiplied
              rw [← fadd_assoc, multiplied, Fin.zero_add]
            · simp only [sumZero, ↓reduceIte, eval, termValue]
              rw [fadd_mul, fadd_assoc]
          · simp only [same, ↓reduceIte, eval, inductionHypothesis,
              termValue]
            exact fadd_left_comm _ _ _

def normalize (form : Form) : Form :=
  form.foldr insert []

theorem eval_normalize (assignment : Nat → F) (form : Form) :
    eval assignment (normalize form) = eval assignment form := by
  induction form with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [normalize, List.foldr_cons, eval_insert, eval]
      unfold normalize at inductionHypothesis
      rw [inductionHypothesis]

def Equivalent (left right : Form) : Prop :=
  normalize left = normalize right

instance (left right : Form) : Decidable (Equivalent left right) := by
  unfold Equivalent
  infer_instance

theorem eval_eq_of_equivalent {left right : Form}
    (equivalent : Equivalent left right) (assignment : Nat → F) :
    eval assignment left = eval assignment right := by
  rw [← eval_normalize assignment left,
    ← eval_normalize assignment right, equivalent]

def ofLinear (form : InputForm) : Form :=
  form.map fun term => (.linear term.1, term.2)

def scale (coefficient : F) (form : Form) : Form :=
  form.map fun term => (term.1, coefficient * term.2)

def mulLinear (coefficient : F)
    (left right : InputForm) : Form :=
  left.flatMap fun leftTerm =>
    right.map fun rightTerm =>
      (Monomial.product leftTerm.1 rightTerm.1,
        coefficient * leftTerm.2 * rightTerm.2)

theorem eval_append (assignment : Nat → F) (left right : Form) :
    eval assignment (left ++ right) =
      eval assignment left + eval assignment right := by
  induction left with
  | nil => simp [eval]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, eval, inductionHypothesis]
      exact (fadd_assoc _ _ _).symm

theorem eval_scale (assignment : Nat → F) (coefficient : F)
    (form : Form) :
    eval assignment (scale coefficient form) =
      coefficient * eval assignment form := by
  induction form with
  | nil => exact (Fin.mul_zero coefficient).symm
  | cons head tail inductionHypothesis =>
      simp only [scale, List.map_cons, eval, termValue]
      unfold scale at inductionHypothesis
      rw [inductionHypothesis, Fin.mul_assoc, fmul_add]

theorem eval_ofLinear (assignment : Nat → F) (form : InputForm) :
    eval assignment (ofLinear form) = linearEval assignment form := by
  unfold linearEval
  induction form with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [ofLinear, List.map_cons, eval, termValue, monomialValue,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.eval,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.termValue]
      unfold ofLinear at inductionHypothesis
      rw [inductionHypothesis]

private theorem monomialValue_product (assignment : Nat → F)
    (left right : Nat) :
    monomialValue assignment (Monomial.product left right) =
      assignment left * assignment right := by
  unfold Monomial.product
  split
  · rfl
  · simp only [monomialValue]
    exact Fin.mul_comm _ _

private theorem eval_mulLinear_one (assignment : Nat → F)
    (coefficient : F) (leftTerm : LinearTerm) :
    ∀ right,
      eval assignment
          (right.map fun rightTerm =>
            (Monomial.product leftTerm.1 rightTerm.1,
              coefficient * leftTerm.2 * rightTerm.2)) =
        coefficient * (leftTerm.2 * assignment leftTerm.1) *
          linearEval assignment right := by
  intro right
  induction right with
  | nil => simp [eval, linearEval, Fin.mul_zero,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.eval]
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, eval, termValue, monomialValue_product,
        linearEval,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.eval,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.termValue]
      rw [inductionHypothesis]
      calc
        (coefficient * leftTerm.2 * head.2) *
              (assignment leftTerm.1 * assignment head.1) +
            coefficient * (leftTerm.2 * assignment leftTerm.1) *
              linearEval assignment tail =
            coefficient * (leftTerm.2 * assignment leftTerm.1) *
              (head.2 * assignment head.1) +
            coefficient * (leftTerm.2 * assignment leftTerm.1) *
              linearEval assignment tail := by
          congr 1
          ac_rfl
        _ = coefficient * (leftTerm.2 * assignment leftTerm.1) *
              ((head.2 * assignment head.1) +
                linearEval assignment tail) := by
          rw [← fmul_add]

theorem eval_mulLinear (assignment : Nat → F) (coefficient : F)
    (left right : InputForm) :
    eval assignment (mulLinear coefficient left right) =
      coefficient * linearEval assignment left * linearEval assignment right := by
  induction left with
  | nil => simp [mulLinear, eval, linearEval, Fin.zero_mul, Fin.mul_zero,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.eval]
  | cons head tail inductionHypothesis =>
      simp only [mulLinear, List.flatMap_cons, eval_append, linearEval,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.eval,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm.termValue]
      unfold mulLinear at inductionHypothesis
      rw [eval_mulLinear_one, inductionHypothesis]
      calc
        coefficient * (head.2 * assignment head.1) *
              linearEval assignment right +
            coefficient * linearEval assignment tail *
              linearEval assignment right =
            coefficient *
              ((head.2 * assignment head.1) +
                linearEval assignment tail) *
              linearEval assignment right := by
          rw [fmul_add, fadd_mul]
        _ = _ := rfl

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.QuadraticForm
