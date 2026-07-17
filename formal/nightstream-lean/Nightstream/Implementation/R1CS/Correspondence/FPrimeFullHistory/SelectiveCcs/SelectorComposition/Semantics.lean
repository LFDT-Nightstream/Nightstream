import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: minimal algebraic semantics for composing finitely many F' branch
relations with selector-gated residual rows.

Owns: indexed branch-residual families, the sum-to-one selector obligation,
per-branch residual gating, the Goldilocks no-zero-product bridge, semantic
soundness, honest completeness, and the exact existential characterization of
the selected branch relation.

Does not own: any branch's residual equations, the correspondence from those
rows to `Paper.BaseHolds` or `Paper.RecursiveHolds`, selector-row matrix
coefficients, Rust arm ordering, witness storage, or permission to delete a
production row. Those are separate refinement obligations.

Emits constraints: no. This file identifies the minimal model-level selector
obligations against which an executable emitter can be refined.

Authority boundary: a selector vector is not authoritative merely because a
caller labels it one-hot. Soundness uses only the checked sum equation and the
checked gated residuals. Selector Booleanity and inactive-advice zeroing are
not assumptions of the semantic theorem.

| Stage path | Mathematical obligation | Authority class | Lean owner | Rust owner | Removal status |
|---|---|---|---|---|---|
| `f_prime.selective_ccs.branch.total` | `sum_i selector[i] = 1` | checked | `SelectorTotal` | selective one-hot row | retained at this level |
| `f_prime.selective_ccs.branch.gate[i]` | every residual of arm `i` is multiplied by `selector[i]` | checked | `GatedRowsZero` | arm-local emitted rows | retained at this level |
| `f_prime.selective_ccs.branch.sound` | some branch has every residual equal to zero | derived | `accepts_sound` | none | model-level proved |
| `f_prime.selective_ccs.branch.complete` | an honest branch has a canonical unit selector witness | computed | `accepts_complete` | selective encoder | model-level proved |
| `f_prime.selective_ccs.branch.selector_domain` | each selector is Boolean | canonicalization | not assumed | selector-domain rows | candidate elimination; concrete refinement open |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete

universe uRow

/-- One branch's residuals indexed by their physical or semantic row type.
The index may be a subtype of final row numbers, so no list proportional to a
production circuit is required. -/
structure ResidualFamily where
  Row : Type uRow
  residual : Row → F

/-- Finite adapter retained for small countermodels and executable tests. It is
not the production interface. -/
def ResidualFamily.ofList (values : List F) : ResidualFamily where
  Row := Fin values.length
  residual row := values.get row

/-- Every indexed residual in one branch is zero. -/
def RowsZero (family : ResidualFamily) : Prop :=
  ∀ row, family.residual row = 0

/-- Every residual in one branch is zero after multiplication by its selector
weight. -/
def GatedRowsZero (weight : F) (family : ResidualFamily) : Prop :=
  ∀ row, weight * family.residual row = 0

/-- Finite selector sum, defined locally so the protocol layer does not acquire
a Mathlib dependency merely for finite notation. -/
def selectorSum : {armCount : Nat} → (Fin armCount → F) → F
  | 0, _ => 0
  | _ + 1, weights =>
      weights 0 + selectorSum (fun arm => weights arm.succ)

/-- Exact selector sum checked by the executable multi-arm selective
compiler. No Booleanity premise is hidden in this definition. -/
def SelectorTotal {armCount : Nat} (weights : Fin armCount → F) : Prop :=
  selectorSum weights = 1

private theorem selectorSum_zero_of_pointwise_zero
    {armCount : Nat} {weights : Fin armCount → F}
    (everyZero : ∀ arm, weights arm = 0) :
    selectorSum weights = 0 := by
  induction armCount with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [selectorSum]
      rw [everyZero 0, Fin.zero_add]
      apply inductionHypothesis
      intro arm
      exact everyZero arm.succ

/-- Minimal multi-arm selector composition: the weights sum to one and every
branch's residual rows are gated by its own weight. -/
structure Accepts {armCount : Nat}
    (weights : Fin armCount → F)
    (families : Fin armCount → ResidualFamily) : Prop where
  total : SelectorTotal weights
  gated : ∀ arm, GatedRowsZero (weights arm) (families arm)

/-- Complete composition shell once always-on rows are added. `common` is
explicit because selector-gated branch rows never discharge constant-one,
padding, shared-domain, context, or authority obligations by themselves. -/
structure ComposedAccepts {armCount : Nat}
    (common : Prop)
    (weights : Fin armCount → F)
    (families : Fin armCount → ResidualFamily) : Prop where
  commonHolds : common
  selectorHolds : Accepts weights families

/-- Semantic result of selector composition, before assigning a meaning to
each branch: at least one complete branch residual list vanishes. -/
def SelectedBranch {armCount : Nat}
    (families : Fin armCount → ResidualFamily) : Prop :=
  ∃ arm, RowsZero (families arm)

/-- The one algebraic property needed from the concrete field. It is kept
explicit so selector soundness never silently relies on a type-class instance
that is unavailable for arbitrary `Fin n` arithmetic. -/
def NoZeroProducts : Prop :=
  ∀ left right : F, left * right = 0 → left = 0 ∨ right = 0

/-- The named Euclid property of the production Goldilocks modulus supplies
the exact no-zero-product fact used by selector soundness. -/
theorem goldilocks_noZeroProducts
    (prime : EuclidPrime goldilocksP) :
    NoZeroProducts := by
  intro left right productZero
  have modularZero : left.val * right.val % goldilocksP = 0 := by
    have values := congrArg Fin.val productZero
    simpa [Fin.val_mul, goldilocksP, goldilocksModulus] using values
  rcases prime left.val right.val modularZero with leftZero | rightZero
  · left
    apply Fin.eq_of_val_eq
    have leftLt : left.val < goldilocksP := by
      exact left.isLt
    simpa [Nat.mod_eq_of_lt leftLt] using leftZero
  · right
    apply Fin.eq_of_val_eq
    have rightLt : right.val < goldilocksP := by
      exact right.isLt
    simpa [Nat.mod_eq_of_lt rightLt] using rightZero

theorem rowsZero_of_gated
    (noZeroProducts : NoZeroProducts)
    {weight : F} {family : ResidualFamily}
    (weightNonzero : weight ≠ 0)
    (gated : GatedRowsZero weight family) :
    RowsZero family := by
  intro row
  rcases noZeroProducts weight (family.residual row) (gated row) with
    weightZero | residualZero
  · exact (weightNonzero weightZero).elim
  · exact residualZero

theorem nonzero_selector_of_total
    {armCount : Nat} {weights : Fin armCount → F}
    (total : SelectorTotal weights) :
    ∃ arm, weights arm ≠ 0 := by
  induction armCount with
  | zero =>
      have zeroOne : (0 : F) = 1 := total
      exact ((by decide : (0 : F) ≠ 1) zeroOne).elim
  | succ count inductionHypothesis =>
      by_cases headZero : weights 0 = 0
      · have tailTotal :
            SelectorTotal (fun arm : Fin count => weights arm.succ) := by
          unfold SelectorTotal at total ⊢
          simp only [selectorSum] at total
          rw [headZero, Fin.zero_add] at total
          exact total
        rcases inductionHypothesis tailTotal with ⟨arm, nonzero⟩
        exact ⟨arm.succ, nonzero⟩
      · exact ⟨0, headZero⟩

/-- Sum-to-one plus gated rows is sound without selector Booleanity: at least
one weight is nonzero, and that branch's residuals therefore vanish. -/
theorem accepts_sound
    (noZeroProducts : NoZeroProducts)
    {armCount : Nat}
    {weights : Fin armCount → F}
    {families : Fin armCount → ResidualFamily}
    (accepted : Accepts weights families) :
    SelectedBranch families := by
  rcases nonzero_selector_of_total accepted.total with ⟨arm, active⟩
  exact ⟨arm, rowsZero_of_gated noZeroProducts active (accepted.gated arm)⟩

/-- Canonical unit-vector selectors used by an honest branch witness. -/
def unitWeights {armCount : Nat} (selected : Fin armCount) :
    Fin armCount → F :=
  fun arm => if arm = selected then 1 else 0

theorem unitWeights_total
    {armCount : Nat} (selected : Fin armCount) :
    SelectorTotal (unitWeights selected) := by
  induction armCount with
  | zero => exact Fin.elim0 selected
  | succ count inductionHypothesis =>
      exact Fin.cases
        (by
          unfold SelectorTotal
          simp only [selectorSum]
          have headOne :
              unitWeights (0 : Fin (count + 1)) 0 = 1 := by
            unfold unitWeights
            rw [if_pos rfl]
          have tailZero :
              selectorSum (fun arm : Fin count =>
                unitWeights (0 : Fin (count + 1)) arm.succ) = 0 := by
            apply selectorSum_zero_of_pointwise_zero
            intro arm
            unfold unitWeights
            rw [if_neg (Fin.succ_ne_zero arm)]
          rw [headOne, tailZero, Fin.add_zero])
        (fun tail => by
          unfold SelectorTotal
          simp only [selectorSum]
          have headZero :
              unitWeights tail.succ (0 : Fin (count + 1)) = 0 := by
            unfold unitWeights
            rw [if_neg (Fin.succ_ne_zero tail).symm]
          have tailWeights :
              (fun arm : Fin count => unitWeights tail.succ arm.succ) =
                unitWeights tail := by
            funext arm
            unfold unitWeights
            by_cases equal : arm = tail
            · subst arm
              rw [if_pos rfl, if_pos rfl]
            · have succDifferent : arm.succ ≠ tail.succ := by
                intro succEqual
                exact equal (Fin.succ_inj.mp succEqual)
              rw [if_neg succDifferent, if_neg equal]
          rw [headZero, Fin.zero_add, tailWeights]
          exact inductionHypothesis tail)
        selected

/-- Honest completeness: a branch whose residuals vanish is accepted with
its unit selector, while every inactive gate vanishes independently of its
branch residuals. -/
theorem accepts_complete
    {armCount : Nat}
    (families : Fin armCount → ResidualFamily)
    (selected : Fin armCount)
    (selectedZero : RowsZero (families selected)) :
    Accepts (unitWeights selected) families := by
  refine ⟨unitWeights_total selected, ?_⟩
  intro arm row
  by_cases active : arm = selected
  · subst arm
    unfold unitWeights
    rw [if_pos rfl, Fin.one_mul]
    exact selectedZero row
  · unfold unitWeights
    rw [if_neg active, Fin.zero_mul]

/-- Exact minimal selector contract: there exists an accepted selector vector
if and only if at least one branch's residual list vanishes. -/
theorem exists_accepts_iff_selectedBranch
    (noZeroProducts : NoZeroProducts)
    {armCount : Nat}
    (families : Fin armCount → ResidualFamily) :
    (∃ weights, Accepts weights families) ↔ SelectedBranch families := by
  constructor
  · rintro ⟨weights, accepted⟩
    exact accepts_sound noZeroProducts accepted
  · rintro ⟨selected, selectedZero⟩
    exact ⟨unitWeights selected,
      accepts_complete families selected selectedZero⟩

theorem exists_composedAccepts_iff
    (noZeroProducts : NoZeroProducts)
    {armCount : Nat}
    (common : Prop)
    (families : Fin armCount → ResidualFamily) :
    (∃ weights, ComposedAccepts common weights families) ↔
      common ∧ SelectedBranch families := by
  constructor
  · rintro ⟨weights, accepted⟩
    exact ⟨accepted.commonHolds,
      accepts_sound noZeroProducts accepted.selectorHolds⟩
  · rintro ⟨commonHolds, selected, selectedZero⟩
    exact ⟨unitWeights selected, commonHolds,
      accepts_complete families selected selectedZero⟩

/-- Independent branch-local refinement contract. The selector proof consumes
this interface rather than re-expressing the base, bootstrap-recursive, or
steady-recursive verifier inside the selector module. -/
structure ExactBranchRefinement {armCount : Nat}
    (families : Fin armCount → ResidualFamily)
    (semantics : Fin armCount → Prop) : Prop where
  sound : ∀ arm, RowsZero (families arm) → semantics arm
  complete : ∀ arm, semantics arm → RowsZero (families arm)

theorem selectedBranch_iff_semantics
    {armCount : Nat}
    {families : Fin armCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactBranchRefinement families semantics) :
    SelectedBranch families ↔ ∃ arm, semantics arm := by
  constructor
  · rintro ⟨arm, zero⟩
    exact ⟨arm, refinement.sound arm zero⟩
  · rintro ⟨arm, holds⟩
    exact ⟨arm, refinement.complete arm holds⟩

/-- Once every branch independently refines its paper semantics, the minimal
selector composition is sound and complete for the disjunction of those
semantics. This theorem does not manufacture that branch refinement. -/
theorem exists_accepts_iff_semantics
    (noZeroProducts : NoZeroProducts)
    {armCount : Nat}
    {families : Fin armCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactBranchRefinement families semantics) :
    (∃ weights, Accepts weights families) ↔ ∃ arm, semantics arm := by
  rw [exists_accepts_iff_selectedBranch noZeroProducts families,
    selectedBranch_iff_semantics refinement]

/-- Direct soundness adapter for a later `Paper.Holds` target. Each branch
owner supplies only its own implication; selector composition supplies the
disjunction. -/
theorem accepts_target_sound
    (noZeroProducts : NoZeroProducts)
    {armCount : Nat}
    {weights : Fin armCount → F}
    {families : Fin armCount → ResidualFamily}
    {Target : Prop}
    (accepted : Accepts weights families)
    (branchSound : ∀ arm, RowsZero (families arm) → Target) :
    Target := by
  rcases accepts_sound noZeroProducts accepted with ⟨arm, zero⟩
  exact branchSound arm zero

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
