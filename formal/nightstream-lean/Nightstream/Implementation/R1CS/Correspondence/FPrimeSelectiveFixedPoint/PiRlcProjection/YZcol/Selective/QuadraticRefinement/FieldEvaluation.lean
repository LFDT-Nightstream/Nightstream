import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Core

/-!
Field interpretation of the focused compact `y_zcol` symbolic forms.

Owns: evaluation of decoded linear forms, product forms, and factor sums over
the concrete protocol field.

Does not own: certificate aggregation, symbolic state execution, terminal
matching, selected-row materialization, or protocol authority.

Emits constraints: no.

| Equation leaf | Mathematical obligation | Authority class |
|---|---|---|
| linear form | evaluating decoded terms agrees with the source linear combination modulo the field | derived |
| product form | a symbolic product evaluates to the product of its decoded factor values | derived |
| factor fold | linear and product factor expressions evaluate to the corresponding source values and sums | direct dataflow |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

theorem fieldResidue_eq_residue (value : Nat) :
    Materialized.Semantics.fieldResidue value = residue value := by
  rfl

theorem fieldResidue_lcEval_unit (assignment : Nat → Nat) (column : Nat) :
    Materialized.Semantics.fieldResidue (lcEval assignment [(column, 1)]) =
      Materialized.Semantics.fieldResidue (assignment column) := by
  apply Fin.ext
  simp only [Materialized.Semantics.fieldResidue, lcEval,
    List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]
  rw [← Materialized.Semantics.modulus_eq]
  exact Nat.mod_mod _ _

theorem residue_lcEval_unit (assignment : Nat → Nat) (column : Nat) :
    residue (lcEval assignment [(column, 1)]) =
      residue (assignment column) := by
  apply Fin.ext
  simp only [residue, lcEval, List.foldl_cons, List.foldl_nil,
    Nat.zero_add, Nat.one_mul]
  exact Nat.mod_mod _ _

theorem evalTermsLinearForm (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    evalLinearForm (compilerAssignment assignment) (termsLinearForm terms) =
      Materialized.Semantics.fieldResidue
        (lcEval (compilerAssignment assignment) terms) := by
  exact evalNatTermsLinearForm (compilerAssignment assignment) terms

theorem evalTermsExpression (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (fun column => Materialized.Semantics.fieldResidue
      (compilerAssignment assignment column)) (termsExpression terms) =
      Materialized.Semantics.fieldResidue
        (lcEval (compilerAssignment assignment) terms) := by
  unfold termsExpression
  rw [Materialized.QuadraticForm.eval_ofLinear]
  exact evalTermsLinearForm assignment terms

theorem evalProductExpression (assignment : Nat → Nat)
    (coefficient : Nat) (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (fun column => Materialized.Semantics.fieldResidue
        (compilerAssignment assignment column))
        (productExpression coefficient left right) =
      Materialized.Semantics.fieldResidue coefficient *
        Materialized.Semantics.fieldResidue
          (lcEval (compilerAssignment assignment) left) *
        Materialized.Semantics.fieldResidue
          (lcEval (compilerAssignment assignment) right) := by
  unfold productExpression
  rw [Materialized.QuadraticForm.eval_mulLinear]
  change Materialized.Semantics.fieldResidue coefficient *
      evalLinearForm (compilerAssignment assignment) (termsLinearForm left) *
      evalLinearForm (compilerAssignment assignment) (termsLinearForm right) = _
  rw [evalTermsLinearForm, evalTermsLinearForm]

def sourceFieldAssignment (assignment : Nat → Nat) :
    Nat → Nightstream.SuperNeo.Concrete.F :=
  fun column => Materialized.Semantics.fieldResidue
    (compilerAssignment assignment column)

theorem evalTermsExpression_sourceField (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (termsExpression terms) =
      Materialized.Semantics.fieldResidue
        (lcEval (compilerAssignment assignment) terms) := by
  exact evalTermsExpression assignment terms

theorem evalProductExpression_sourceField (assignment : Nat → Nat)
    (coefficient : Nat) (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productExpression coefficient left right) =
      Materialized.Semantics.fieldResidue coefficient *
        Materialized.Semantics.fieldResidue
          (lcEval (compilerAssignment assignment) left) *
        Materialized.Semantics.fieldResidue
          (lcEval (compilerAssignment assignment) right) := by
  exact evalProductExpression assignment coefficient left right

theorem evalProductExpression_projection (assignment : Nat → Nat)
    (coefficient : Nat) (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productExpression coefficient left right) =
      residue coefficient *
        residue (lcEval (compilerAssignment assignment) left) *
        residue (lcEval (compilerAssignment assignment) right) := by
  rw [evalProductExpression_sourceField,
    fieldResidue_eq_residue, fieldResidue_eq_residue,
    fieldResidue_eq_residue]
  rfl

theorem evalProductExpression_one (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productExpression 1 left right) =
      residue (lcEval (compilerAssignment assignment) left) *
        residue (lcEval (compilerAssignment assignment) right) := by
  rw [evalProductExpression_projection, residue_one, Fin.one_mul]

theorem evalProductExpression_seven (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
        (productExpression 7 left right) =
      residue 7 * (residue (lcEval (compilerAssignment assignment) left) *
        residue (lcEval (compilerAssignment assignment) right)) := by
  rw [evalProductExpression_projection, Fin.mul_assoc]

theorem sameDerivedSlot_eq {left right : DecodedDerivedSlot}
    (same : SameDerivedSlot left right) : left = right := by
  rcases left with ⟨leftIndex, leftStart, leftWidth,
    leftPositive, leftBalanced, leftBound⟩
  rcases right with ⟨rightIndex, rightStart, rightWidth,
    rightPositive, rightBalanced, rightBound⟩
  simp only [SameDerivedSlot] at same
  rcases same with ⟨indexEq, startEq, widthEq⟩
  subst rightIndex
  subst rightStart
  subst rightWidth
  congr

theorem evalLinearExpression (assignment : Nat → Nat)
    (linear : DecodedSourceLinearCombination) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) (linearExpression linear) =
      sourceValue assignment linear := by
  unfold linearExpression
  rw [Materialized.QuadraticForm.eval_ofLinear]
  exact evalNatTermsLinearForm (compilerAssignment assignment)
    linear.programTerms

theorem evalFactorExpression (assignment : Nat → Nat)
    (factor : DecodedProductFactor) :
    Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) (factorExpression factor) =
      factorValue assignment factor := by
  unfold factorExpression
  rw [Materialized.QuadraticForm.eval_mulLinear]
  change Materialized.Semantics.fieldResidue factor.coefficient *
      evalLinearForm (compilerAssignment assignment)
        (natTermsLinearForm factor.left.programTerms) *
      evalLinearForm (compilerAssignment assignment)
        (natTermsLinearForm factor.right.programTerms) = _
  rw [
    evalNatTermsLinearForm (compilerAssignment assignment),
    evalNatTermsLinearForm (compilerAssignment assignment)]
  rfl

theorem evalFactorsExpression (assignment : Nat → Nat) :
    ∀ factors,
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment)
          (factorsExpression factors) =
        factors.foldr
          (fun factor suffix => factorValue assignment factor + suffix) 0 := by
  intro factors
  induction factors with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [factorsExpression, List.flatMap_cons, Materialized.QuadraticForm.eval_append,
        List.foldr_cons]
      unfold factorsExpression at inductionHypothesis
      rw [evalFactorExpression, inductionHypothesis]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
