import SuperNeo.FPrimeRecursiveVerifier.Plan
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
Owns: compositional row, column, and nonzero accounting for modular verifier
plans.

Does not own: measured production counts, semantic soundness, or constraint
refinement.

Emits constraints: no.

Authority boundary: costs are lowering-supplied metadata and never establish
acceptance or justify removing a check.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Block cost | `R1csCost` | Records rows, columns, and nonzeros |
| Plan aggregation | `planCost` | Sums selected block costs componentwise |
| Candidate budget | `CostedCandidate` | Couples a certified plan with declared costs |
-/

namespace SuperNeo.FPrimeRecursiveVerifier

open scoped BigOperators

universe u

/-- Backend-facing R1CS dimensions for one independent check block. -/
structure R1csCost where
  rows : Nat
  columns : Nat
  nonzeros : Nat
deriving Repr, DecidableEq

namespace R1csCost

/-- The zero-cost block. -/
def zero : R1csCost :=
  { rows := 0, columns := 0, nonzeros := 0 }

/-- Componentwise addition of independent block costs. -/
def add (lhs rhs : R1csCost) : R1csCost :=
  { rows := lhs.rows + rhs.rows
    columns := lhs.columns + rhs.columns
    nonzeros := lhs.nonzeros + rhs.nonzeros }

/-- Componentwise comparison used for backend-budget gates. -/
def Le (lhs rhs : R1csCost) : Prop :=
  lhs.rows ≤ rhs.rows ∧
    lhs.columns ≤ rhs.columns ∧
    lhs.nonzeros ≤ rhs.nonzeros

@[simp] theorem zero_rows : zero.rows = 0 := rfl
@[simp] theorem zero_columns : zero.columns = 0 := rfl
@[simp] theorem zero_nonzeros : zero.nonzeros = 0 := rfl

end R1csCost

/-- Sum the certified costs of all selected check blocks. -/
def planCost
    {Check : Type u} [DecidableEq Check]
    (cost : Check → R1csCost)
    (checks : Finset Check) : R1csCost :=
  { rows := ∑ check ∈ checks, (cost check).rows
    columns := ∑ check ∈ checks, (cost check).columns
    nonzeros := ∑ check ∈ checks, (cost check).nonzeros }

@[simp] theorem planCost_empty
    {Check : Type u} [DecidableEq Check]
    (cost : Check → R1csCost) :
    planCost cost ∅ = R1csCost.zero := by
  rfl

theorem planCost_erase_rows
    {Check : Type u} [DecidableEq Check]
    (cost : Check → R1csCost)
    (checks : Finset Check)
    (check : Check)
    (hMember : check ∈ checks) :
    (planCost cost (checks.erase check)).rows + (cost check).rows =
      (planCost cost checks).rows := by
  exact Finset.sum_erase_add _ _ hMember

theorem planCost_erase_columns
    {Check : Type u} [DecidableEq Check]
    (cost : Check → R1csCost)
    (checks : Finset Check)
    (check : Check)
    (hMember : check ∈ checks) :
    (planCost cost (checks.erase check)).columns + (cost check).columns =
      (planCost cost checks).columns := by
  exact Finset.sum_erase_add _ _ hMember

theorem planCost_erase_nonzeros
    {Check : Type u} [DecidableEq Check]
    (cost : Check → R1csCost)
    (checks : Finset Check)
    (check : Check)
    (hMember : check ∈ checks) :
    (planCost cost (checks.erase check)).nonzeros + (cost check).nonzeros =
      (planCost cost checks).nonzeros := by
  exact Finset.sum_erase_add _ _ hMember

/-- A complete candidate for comparison: theorem certificate plus backend cost. -/
structure CostedCandidate
    {Input : Type u} {Check : Type u} [DecidableEq Check]
    (semantics : Check → Input → Prop)
    (target : Input → Prop) where
  certificate : CertifiedPlan semantics target
  costOf : Check → R1csCost

namespace CostedCandidate

/-- Total certified cost of this candidate's selected blocks. -/
def totalCost
    {Input : Type u} {Check : Type u} [DecidableEq Check]
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    (candidate : CostedCandidate semantics target) : R1csCost :=
  planCost candidate.costOf candidate.certificate.checks

end CostedCandidate

end SuperNeo.FPrimeRecursiveVerifier
