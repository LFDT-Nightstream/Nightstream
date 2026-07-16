import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCompiler

/-!
Exact semantic counterexample for production strict-`Pi_DEC` digit authority.

Owns: two canonical Goldilocks assignments over the actual
`PiDecStrictCompiler.Recomposes`, `radixPowers`, `lcEval`, and `CenteredUnit`
predicates. Both open parent residue one at radix two, but their child vectors
are `[1,0]` and `[p-1,1]`.

Does not own: a full `PiDecStrictCompiler.Layout`, complete `Accepted`, NIFS
output authority, protocol unsoundness, Rust artifact equality, row removal, or
constraint counts.

Emits constraints: no.

Authority boundary: this is no longer an analogous integer toy—the witness
uses the exact current family-level implementation semantics. It proves that
strict recomposition plus the current centered alphabet is non-functional.
The complete child vector must therefore remain authoritative unless another
proved binding family fixes it.

| Protocol | Phase | Constraint family | Exact production fact |
|---|---|---|---|
| `Pi_DEC` | recomposition | radix powers | `radixPowers 2 2 = [1,2]` |
| `Pi_DEC` | field equation | `Recomposes` / `lcEval` | both child vectors yield parent residue one modulo Goldilocks |
| `Pi_DEC` | child alphabet | `CenteredUnit` | `1`, `0`, and `p-1` all pass the current predicate |
| NIFS | output authority | complete child vector | the two accepted child vectors are different |
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictNecessity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

def parentColumn : Nat := 1
def childColumns : List Nat := [2, 3]
def powers : List Nat := radixPowers 2 childColumns.length

/-- Canonical binary-looking opening `[1,0]`. -/
def canonicalAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column = parentColumn then 1
  else if column = 2 then 1
  else 0

/-- Signed alias `[-1,1]`, represented canonically as `[p-1,1]`. -/
def signedAliasAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column = parentColumn then 1
  else if column = 2 then goldilocksP - 1
  else if column = 3 then 1
  else 0

theorem powers_eq : powers = [1, 2] := by
  decide

theorem canonicalAssignment_canonical (column : Nat) :
    canonicalAssignment column < goldilocksP := by
  by_cases zero : column = 0
  · subst column
    decide
  by_cases parent : column = parentColumn
  · subst column
    decide
  by_cases firstChild : column = 2
  · subst column
    decide
  · simp [canonicalAssignment, zero, parent, firstChild, goldilocksP]

theorem signedAliasAssignment_canonical (column : Nat) :
    signedAliasAssignment column < goldilocksP := by
  by_cases zero : column = 0
  · subst column
    decide
  by_cases parent : column = parentColumn
  · subst column
    decide
  by_cases firstChild : column = 2
  · subst column
    decide
  by_cases secondChild : column = 3
  · subst column
    decide
  · simp [signedAliasAssignment, zero, parent, firstChild, secondChild,
      goldilocksP]

theorem canonical_recomposes :
    Recomposes canonicalAssignment parentColumn childColumns powers := by
  decide

theorem signedAlias_recomposes :
    Recomposes signedAliasAssignment parentColumn childColumns powers := by
  decide

theorem canonical_children_centered :
    forall child, child ∈ childColumns ->
      CenteredUnit (canonicalAssignment child) := by
  intro child member
  have equal : child = 2 ∨ child = 3 := by
    simpa [childColumns] using member
  rcases equal with first | second
  · subst child
    decide
  · subst child
    decide

theorem signedAlias_children_centered :
    forall child, child ∈ childColumns ->
      CenteredUnit (signedAliasAssignment child) := by
  intro child member
  have equal : child = 2 ∨ child = 3 := by
    simpa [childColumns] using member
  rcases equal with first | second
  · subst child
    decide
  · subst child
    decide

def childValues (assignment : Nat -> Nat) : List Nat :=
  childColumns.map assignment

theorem canonical_childValues :
    childValues canonicalAssignment = [1, 0] := by
  decide

theorem signedAlias_childValues :
    childValues signedAliasAssignment = [goldilocksP - 1, 1] := by
  decide

theorem childValues_different :
    childValues canonicalAssignment ≠ childValues signedAliasAssignment := by
  rw [canonical_childValues, signedAlias_childValues]
  decide

/-- Exact current implementation-family ambiguity: recomposition and centered
alphabet checks accept two different child vectors for the same parent residue. -/
theorem recomposition_and_centered_alphabet_not_functional :
    Recomposes canonicalAssignment parentColumn childColumns powers /\
      Recomposes signedAliasAssignment parentColumn childColumns powers /\
      (forall child, child ∈ childColumns ->
        CenteredUnit (canonicalAssignment child)) /\
      (forall child, child ∈ childColumns ->
        CenteredUnit (signedAliasAssignment child)) /\
      canonicalAssignment parentColumn =
        signedAliasAssignment parentColumn /\
      childValues canonicalAssignment ≠ childValues signedAliasAssignment :=
  ⟨canonical_recomposes, signedAlias_recomposes,
    canonical_children_centered, signedAlias_children_centered,
    rfl, childValues_different⟩

end Nightstream.Implementation.R1CS.PiDecStrictNecessity
