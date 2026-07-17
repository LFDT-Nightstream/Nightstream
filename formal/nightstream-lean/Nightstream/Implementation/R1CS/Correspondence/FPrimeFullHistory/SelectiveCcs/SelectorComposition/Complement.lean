import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Semantics

/-!
Contract: exact model-level selector convention of the executable fixed
two-arm compiler.

Owns: the `s` / `1-s` complement gates, their disjunctive soundness, honest
completeness for either arm, and the exact existential characterization.

Does not own: the active three-arm selective compiler, selector coefficients,
branch-to-paper refinement, concrete rows, or permission to remove rows.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.fixed_ccs.branch.base` | `s * baseResidual = 0` | checked |
| `f_prime.fixed_ccs.branch.recursive` | `(1-s) * recursiveResidual = 0` | checked |
| `f_prime.fixed_ccs.branch.selected` | base or recursive residual family vanishes | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Complement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics

/-- Exact two-arm convention: `selector = 1` activates base, while
`selector = 0` activates recursive for honest witnesses. Soundness does not
assume the selector is Boolean. -/
structure ComplementAccepts
    (selector : F) (base recursive : ResidualFamily) : Prop where
  baseGated : GatedRowsZero selector base
  recursiveGated : GatedRowsZero (1 - selector) recursive

theorem complementAccepts_sound
    (noZeroProducts : NoZeroProducts)
    {selector : F} {base recursive : ResidualFamily}
    (accepted : ComplementAccepts selector base recursive) :
    RowsZero base ∨ RowsZero recursive := by
  by_cases baseInactive : selector = 0
  · subst selector
    right
    exact rowsZero_of_gated noZeroProducts
      (weight := (1 : F) - 0) (by decide) accepted.recursiveGated
  · left
    exact rowsZero_of_gated noZeroProducts baseInactive accepted.baseGated

theorem complementAccepts_complete_base
    {base recursive : ResidualFamily}
    (baseZero : RowsZero base) :
    ComplementAccepts 1 base recursive := by
  constructor
  · intro row
    rw [Fin.one_mul]
    exact baseZero row
  · intro row
    have weightZero : (1 : F) - 1 = 0 := by decide
    rw [weightZero, Fin.zero_mul]

theorem complementAccepts_complete_recursive
    {base recursive : ResidualFamily}
    (recursiveZero : RowsZero recursive) :
    ComplementAccepts 0 base recursive := by
  constructor
  · intro row
    rw [Fin.zero_mul]
  · intro row
    have weightOne : (1 : F) - 0 = 1 := by decide
    rw [weightOne, Fin.one_mul]
    exact recursiveZero row

/-- The fixed complement compiler is exact for branch disjunction without a
selector-Booleanity premise. -/
theorem exists_complementAccepts_iff
    (noZeroProducts : NoZeroProducts)
    (base recursive : ResidualFamily) :
    (∃ selector, ComplementAccepts selector base recursive) ↔
      RowsZero base ∨ RowsZero recursive := by
  constructor
  · rintro ⟨selector, accepted⟩
    exact complementAccepts_sound noZeroProducts accepted
  · intro selected
    rcases selected with baseZero | recursiveZero
    · exact ⟨1, complementAccepts_complete_base baseZero⟩
    · exact ⟨0, complementAccepts_complete_recursive recursiveZero⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Complement
