import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Semantics

/-!
Contract: inclusion-minimality witnesses for the selector-composition
obligations, separated from canonical witness restrictions.

Owns: a bad two-arm residual fixture proving necessity of the selector-total
equation and of every branch gate, plus a nonzero inactive-advice witness
showing that inactive zeroing is not part of semantic selector soundness.

Does not own: isolation of any concrete Rust row, production arm semantics,
matrix coefficients, low-norm commitment soundness, or permission to remove
selector-domain/inactive-binding rows. Those require artifact refinement.

Emits constraints: no.

| Stage path | Omitted obligation | Accepted invalid witness | Lean owner | Theorem classification |
|---|---|---|---|---|
| `f_prime.selective_ccs.branch.total` | `sum_i selector[i] = 1` | all weights zero, both residuals one | `selectorTotal_necessary` | inclusion-necessary |
| `f_prime.selective_ccs.branch.gate[i]` | gate for arm `i` | only arm `i` has weight one, every residual is one | `eachBranchGate_necessary` | inclusion-necessary |
| `f_prime.selective_ccs.branch.selector_domain` | selector Booleanity | no countermodel; `exists_accepts_iff_selectedBranch` omits it | `selectorBitness_not_required` | derived/eliminable at model level |
| `f_prime.selector_gated.inactive_binding` | inactive advice equals zero | base valid, inactive advice one | `inactiveAdviceZero_not_required` | canonicalization only at model level |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Necessity

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics

/-- Both toy arms contain one indexed nonzero residual, so neither branch is
valid. The finite adapter is used only for this countermodel. -/
def badRows : Fin 2 → ResidualFamily :=
  fun _ => .ofList [1]

theorem no_bad_branch_selected :
    ¬ SelectedBranch badRows := by
  rintro ⟨arm, rowsZero⟩
  have oneZero := rowsZero (0 : Fin 1)
  exact (by decide : (1 : F) ≠ 0) oneZero

/-- What remains if the selector-total row is omitted. -/
def AcceptsWithoutTotal {armCount : Nat}
    (weights : Fin armCount → F)
    (families : Fin armCount → ResidualFamily) : Prop :=
  ∀ arm, GatedRowsZero (weights arm) (families arm)

/-- The total equation is necessary: zero weights disable every invalid arm. -/
theorem selectorTotal_necessary :
    ∃ weights : Fin 2 → F,
      AcceptsWithoutTotal weights badRows ∧
        ¬ SelectedBranch badRows := by
  let weights : Fin 2 → F := fun _ => 0
  refine ⟨weights, ?_, no_bad_branch_selected⟩
  intro arm row
  change (0 : F) * (badRows arm).residual row = 0
  rw [Fin.zero_mul]

/-- What remains if exactly one arm's gated row family is omitted. -/
def AcceptsWithoutBranchGate {armCount : Nat}
    (omitted : Fin armCount)
    (weights : Fin armCount → F)
    (families : Fin armCount → ResidualFamily) : Prop :=
  SelectorTotal weights ∧
    ∀ arm, arm ≠ omitted → GatedRowsZero (weights arm) (families arm)

/-- Every branch gate family is independently necessary. Selecting the
omitted arm with weight one makes all retained gates inactive while neither
branch relation holds. -/
theorem eachBranchGate_necessary (omitted : Fin 2) :
    ∃ weights : Fin 2 → F,
      AcceptsWithoutBranchGate omitted weights badRows ∧
        ¬ SelectedBranch badRows := by
  refine ⟨unitWeights omitted, ?_, no_bad_branch_selected⟩
  constructor
  · exact unitWeights_total omitted
  · intro arm different row
    unfold unitWeights
    rw [if_neg different, Fin.zero_mul]

/-- Model-level elimination theorem for selector Booleanity. The minimal
accepted relation is already exact for branch disjunction under the explicit
Goldilocks no-zero-product hypothesis. -/
theorem selectorBitness_not_required
    (noZeroProducts : NoZeroProducts)
    {armCount : Nat}
    (families : Fin armCount → ResidualFamily) :
    (∃ weights, Accepts weights families) ↔ SelectedBranch families :=
  exists_accepts_iff_selectedBranch noZeroProducts families

/-- Canonical-witness condition sometimes imposed on branch-private advice.
It is deliberately outside `Accepts`. -/
def InactiveAdviceZero {armCount : Nat}
    (weights : Fin armCount → F)
    (advice : Fin armCount → ResidualFamily) : Prop :=
  ∀ arm, weights arm = 0 → RowsZero (advice arm)

def baseSelectedRows : Fin 2 → ResidualFamily
  | ⟨0, _⟩ => .ofList [0]
  | ⟨1, _⟩ => .ofList [1]

def nonzeroInactiveAdvice : Fin 2 → ResidualFamily
  | ⟨0, _⟩ => .ofList []
  | ⟨1, _⟩ => .ofList [1]

theorem baseSelectedRows_zero :
    RowsZero (baseSelectedRows 0) := by
  change ∀ row : Fin 1, ([0].get row : F) = 0
  intro row
  have rowZero : row = (0 : Fin 1) := by
    apply Fin.eq_of_val_eq
    omega
  subst row
  rfl

theorem inactiveAdviceZero_fails :
    ¬ InactiveAdviceZero (unitWeights (0 : Fin 2)) nonzeroInactiveAdvice := by
  intro inactive
  have inactiveWeight : unitWeights (0 : Fin 2) 1 = 0 := by
    decide
  have rowsZero := inactive 1 inactiveWeight
  have oneZero := rowsZero (0 : Fin 1)
  exact (by decide : (1 : F) ≠ 0) oneZero

/-- An accepted branch composition can carry nonzero inactive advice. This
proves that inactive zeroing is canonicalization, not a premise of the
selector soundness theorem. Concrete storage aliasing and commitment rules
still have to be refined before deleting any row. -/
theorem inactiveAdviceZero_not_required :
    ∃ weights : Fin 2 → F,
      Accepts weights baseSelectedRows ∧
        ¬ InactiveAdviceZero weights nonzeroInactiveAdvice := by
  exact ⟨unitWeights 0,
    accepts_complete baseSelectedRows 0 baseSelectedRows_zero,
    inactiveAdviceZero_fails⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Necessity
