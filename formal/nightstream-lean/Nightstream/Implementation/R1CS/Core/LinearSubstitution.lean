import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: exact transport of R1CS rows through sparse linear column substitution.

Owns: expansion of one source column into an arbitrary sparse linear
combination, the induced decoded assignment, and equivalence of LC, row, and
row-list satisfaction before and after expansion.

Does not own: any concrete column layout, inclusion in a production artifact,
or witness-generation claim.

Emits constraints: no. It transforms already-owned rows.

Assurance tier: model-level. A separate correspondence module must instantiate
the expansion with verifier-owned production layout evidence.

| Predicate/theorem | Mathematical obligation | Guarantee |
|---|---|---|
| `lcEval_terms` | linear substitution | Expanded LC equals source LC on the decoded assignment |
| `rowHolds_iff` | bilinear row transport | Expanded and source rows have identical truth values |
| `satisfies_mapped_iff` | block transport | Every expanded row holds iff every source row holds |
-/

namespace Nightstream.Implementation.R1CS.LinearSubstitution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

/-- Sparse encoded-column representation of every source column. -/
abbrev ColumnExpansion := Nat → List (Nat × Nat)

/-- Canonical coefficient multiplication for substituting one source term. -/
def scaleTerms (coefficient : Nat) (source : List (Nat × Nat)) :
    List (Nat × Nat) :=
  source.map fun term =>
    (term.1, coefficient * term.2 % goldilocksP)

/-- Expand every source term and preserve its outer coefficient. -/
def terms (expansion : ColumnExpansion) (source : List (Nat × Nat)) :
    List (Nat × Nat) :=
  source.flatMap fun term => scaleTerms term.2 (expansion term.1)

/-- Source-column view decoded from an encoded assignment. -/
def assignment (expansion : ColumnExpansion) (encoded : Nat → Nat) : Nat → Nat :=
  fun source => lcEval encoded (expansion source)

/-- Expand all three linear combinations in one R1CS row. -/
def row (expansion : ColumnExpansion) (source : Row) : Row where
  a := terms expansion source.a
  b := terms expansion source.b
  c := terms expansion source.c

private theorem rawLcEval_scaleTerms_mod
    (encoded : Nat → Nat) (coefficient : Nat) (source : List (Nat × Nat)) :
    rawLcEval encoded (scaleTerms coefficient source) % goldilocksP =
      coefficient * (rawLcEval encoded source % goldilocksP) % goldilocksP := by
  induction source with
  | nil => simp [scaleTerms, rawLcEval]
  | cons head tail inductionHypothesis =>
      simp only [scaleTerms, List.map_cons, rawLcEval]
      unfold scaleTerms at inductionHypothesis
      rw [Nat.add_mod, inductionHypothesis]
      simp only [Nat.mod_mul_mod, Nat.mul_mod_mod, Nat.mod_add_mod,
        Nat.add_mod_mod, Nat.mul_add, Nat.mul_assoc]

private theorem rawLcEval_append (encoded : Nat → Nat)
    (left right : List (Nat × Nat)) :
    rawLcEval encoded (left ++ right) =
      rawLcEval encoded left + rawLcEval encoded right := by
  induction left with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis, Nat.add_assoc]

theorem lcEval_terms (expansion : ColumnExpansion) (encoded : Nat → Nat)
    (source : List (Nat × Nat)) :
    lcEval encoded (terms expansion source) =
      lcEval (assignment expansion encoded) source := by
  induction source with
  | nil => simp [terms, lcEval]
  | cons head tail inductionHypothesis =>
      rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod] at inductionHypothesis
      unfold terms at inductionHypothesis
      rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod]
      simp only [terms, List.flatMap_cons, rawLcEval_append, rawLcEval]
      rw [Nat.add_mod, rawLcEval_scaleTerms_mod, inductionHypothesis]
      simp only [assignment]
      rw [lcEval_eq_raw_mod]
      rw [Nat.mul_mod]
      simp only [Nat.mod_mod]
      simp only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod]

theorem rowHolds_iff (expansion : ColumnExpansion) (encoded : Nat → Nat)
    (source : Row) :
    RowHolds encoded (row expansion source) ↔
      RowHolds (assignment expansion encoded) source := by
  simp only [RowHolds, row, lcEval_terms]

theorem satisfies_mapped_iff
    (sourceRows : List Row) (expansion : ColumnExpansion)
    (encoded : Nat → Nat) :
    Satisfies (sourceRows.map (row expansion)) encoded ↔
      Satisfies sourceRows (assignment expansion encoded) := by
  constructor
  · intro satisfies source sourceMember
    apply (rowHolds_iff expansion encoded source).mp
    apply satisfies
    exact List.mem_map.mpr ⟨source, sourceMember, rfl⟩
  · intro satisfies expanded expandedMember
    rcases List.mem_map.mp expandedMember with ⟨source, sourceMember, rfl⟩
    exact (rowHolds_iff expansion encoded source).mpr
      (satisfies source sourceMember)

end Nightstream.Implementation.R1CS.LinearSubstitution
