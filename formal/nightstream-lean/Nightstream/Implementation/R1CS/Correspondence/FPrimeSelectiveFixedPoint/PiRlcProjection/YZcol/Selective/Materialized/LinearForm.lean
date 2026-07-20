import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics

/-!
Small sparse linear-form normalizer for coefficient-level artifact checks.

Owns: semantic evaluation, sorted insertion with duplicate coalescing, and the
proof that normalization preserves every field assignment.

Does not own: generated coefficients, source-column decoding, compiler
substitution, row-family meaning, selector truth, or permission to remove
rows.

Emits constraints: no.

This is deliberately a field-level utility. It lets a later executable check
compare actual compact port streams with independently expanded source linear
combinations without treating term order or cancelled duplicates as evidence.

| Normal-form leaf | Mathematical obligation | Authority class |
|---|---|---|
| insertion | preserve evaluation while coalescing one coefficient | derived |
| normalization | preserve evaluation for every assignment | derived |
| equivalence | normalized sparse forms are exactly equal | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics

abbrev Term := Nat × F

def termValue (assignment : Nat → F) (term : Term) : F :=
  term.2 * assignment term.1

def eval (assignment : Nat → F) : List Term → F
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

/-- Insert one term into a column-sorted form, coalescing an equal-column
coefficient and deleting a zero coefficient. -/
def insert (term : Term) : List Term → List Term
  | [] => if term.2 = 0 then [] else [term]
  | head :: rest =>
      if term.2 = 0 then
        head :: rest
      else if term.1 < head.1 then
        term :: head :: rest
      else if term.1 = head.1 then
        let coefficient := term.2 + head.2
        if coefficient = 0 then rest else (head.1, coefficient) :: rest
      else
        head :: insert term rest

theorem eval_insert (assignment : Nat → F) (term : Term) :
    ∀ terms, eval assignment (insert term terms) =
      termValue assignment term + eval assignment terms := by
  rcases term with ⟨termColumn, termCoefficient⟩
  intro terms
  induction terms with
  | nil =>
      by_cases coefficientZero : termCoefficient = 0
      · simp [insert, coefficientZero, termValue, eval, Fin.zero_mul]
      · simp [insert, coefficientZero, termValue, eval]
  | cons head rest inductionHypothesis =>
      rcases head with ⟨headColumn, headCoefficient⟩
      by_cases coefficientZero : termCoefficient = 0
      · simp [insert, coefficientZero, termValue, eval, Fin.zero_mul]
      · simp only [insert, coefficientZero, ↓reduceIte]
        by_cases before : termColumn < headColumn
        · simp only [before, ↓reduceIte, eval, termValue]
        · simp only [before, ↓reduceIte]
          by_cases same : termColumn = headColumn
          · subst headColumn
            simp only [↓reduceIte, eval, termValue]
            by_cases sumZero : termCoefficient + headCoefficient = 0
            · simp only [sumZero, ↓reduceIte]
              have multiplied := congrArg (fun value : F => value * assignment termColumn)
                sumZero
              change
                (termCoefficient + headCoefficient) * assignment termColumn =
                  0 * assignment termColumn at multiplied
              rw [fadd_mul, Fin.zero_mul] at multiplied
              rw [← fadd_assoc, multiplied, Fin.zero_add]
            · simp only [sumZero, ↓reduceIte, eval, termValue]
              rw [fadd_mul, fadd_assoc]
          · simp only [same, ↓reduceIte, eval, inductionHypothesis,
              termValue]
            exact fadd_left_comm _ _ _

/-- Canonical sparse form: strictly ordered nonzero columns after all
duplicates have been added in the field. -/
def normalize (terms : List Term) : List Term :=
  terms.foldr insert []

theorem eval_normalize (assignment : Nat → F) (terms : List Term) :
    eval assignment (normalize terms) = eval assignment terms := by
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [normalize, List.foldr_cons, eval_insert, eval]
      unfold normalize at inductionHypothesis
      rw [inductionHypothesis]

def Equivalent (left right : List Term) : Prop :=
  normalize left = normalize right

instance (left right : List Term) : Decidable (Equivalent left right) := by
  unfold Equivalent
  infer_instance

theorem eval_eq_of_equivalent {left right : List Term}
    (equivalent : Equivalent left right) (assignment : Nat → F) :
    eval assignment left = eval assignment right := by
  rw [← eval_normalize assignment left,
    ← eval_normalize assignment right, equivalent]

def scale (coefficient : F) (terms : List Term) : List Term :=
  terms.map fun term => (term.1, coefficient * term.2)

theorem eval_append (assignment : Nat → F) (left right : List Term) :
    eval assignment (left ++ right) =
      eval assignment left + eval assignment right := by
  induction left with
  | nil => simp [eval]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, eval, inductionHypothesis]
      exact (fadd_assoc _ _ _).symm

theorem eval_scale (assignment : Nat → F) (coefficient : F)
    (terms : List Term) :
    eval assignment (scale coefficient terms) =
      coefficient * eval assignment terms := by
  induction terms with
  | nil => exact (Fin.mul_zero coefficient).symm
  | cons head tail inductionHypothesis =>
      simp only [scale, List.map_cons, eval, termValue]
      unfold scale at inductionHypothesis
      rw [inductionHypothesis, Fin.mul_assoc, fmul_add]

def portTerms {columns : Nat} (port : DecodedPort columns) : List Term :=
  (expandedFieldTerms port).map fun term => (term.1.val, term.2)

private theorem foldl_eq_initial_add_eval (assignment : Nat → F) :
    ∀ (terms : List Term) (initial : F),
      terms.foldl
          (fun total term => total + term.2 * assignment term.1) initial =
        initial + eval assignment terms := by
  intro terms
  induction terms with
  | nil => intro initial; simp [eval]
  | cons head tail inductionHypothesis =>
      intro initial
      simp only [List.foldl_cons]
      rw [inductionHypothesis]
      simp only [eval, termValue]
      exact fadd_assoc _ _ _

private theorem foldl_mapped_terms {columns : Nat}
    (assignment : Nat → F) :
    ∀ (terms : List (Fin columns × F)) (initial : F),
      (terms.map fun term => (term.1.val, term.2)).foldl
          (fun total term => total + term.2 * assignment term.1) initial =
        terms.foldl
          (fun total term => total + term.2 * assignment term.1.val) initial := by
  intro terms
  induction terms with
  | nil => intro initial; rfl
  | cons head tail inductionHypothesis =>
      intro initial
      simp only [List.map_cons, List.foldl_cons]
      exact inductionHypothesis _

/-- The compact port action is exactly evaluation of its sparse form. -/
theorem action_eq_eval {columns : Nat} (port : DecodedPort columns)
    (assignment : Nat → F) :
    action port (fun column => assignment column.val) =
      eval assignment (portTerms port) := by
  unfold action portTerms
  rw [← foldl_mapped_terms assignment]
  rw [foldl_eq_initial_add_eval assignment]
  exact Fin.zero_add _

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm
