import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonest

/-!
Contract: positional receipt ownership for the Lean-owned canonical `Pi_RLC`
sampler suffix.

The three constructors name the physical row family; the carried index names
the position inside that family's emitted list.  Ownership is positional, not
an assertion that distinct receipts must emit structurally distinct `Row`
values.

This file does not own transcript rows.  Its program is exactly
`PiRlcCanonicalSamplerHonest.suffixRows`.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerOwnership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonest

/-- One receipt family for each physical family in the suffix. -/
inductive RowOwner where
  | u64 (index : Nat)
  | candidate (index : Nat)
  | selector (index : Nat)
deriving DecidableEq, Repr

private def blank : Row := ⟨[], [], []⟩

/-- The row emitted by one positional receipt. -/
def ownedRow
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) : RowOwner → Row
  | .u64 index =>
      (PiRlcCanonicalU64.rows duplexBase u64Base count initialBuilder).getD
        index blank
  | .candidate index =>
      (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
        initialBuilder).getD index blank
  | .selector index =>
      (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
        selectorBase count initialBuilder).getD index blank

/-- Every receipt, in the exact family-major emission order. -/
def owners
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) : List RowOwner :=
  (List.range
      (PiRlcCanonicalU64.rows duplexBase u64Base count
        initialBuilder).length).map RowOwner.u64 ++
    (List.range
      (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
        initialBuilder).length).map RowOwner.candidate ++
    (List.range
      (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
        selectorBase count initialBuilder).length).map RowOwner.selector

private theorem map_getD_range {α : Type} (list : List α) (fallback : α) :
    (List.range list.length).map (fun index => list.getD index fallback) =
      list := by
  induction list with
  | nil => rfl
  | cons head tail hypothesis =>
      rw [List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map]
      exact congrArg (head :: ·) hypothesis

/-- The emitted suffix is exactly the receipt list's image. -/
theorem rows_eq_map_owners
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) :
    suffixRows duplexBase u64Base candidateBase selectorBase count
        initialBuilder =
      (owners duplexBase u64Base candidateBase selectorBase count
        initialBuilder).map
        (ownedRow duplexBase u64Base candidateBase selectorBase count
          initialBuilder) := by
  unfold suffixRows owners
  simp only [List.map_append, List.map_map, Function.comp_def, ownedRow,
    map_getD_range]

private theorem mappedRange_nodup
    (constructor : Nat → RowOwner)
    (injective :
      ∀ first second, constructor first = constructor second → first = second)
    (count : Nat) :
    ((List.range count).map constructor).Nodup :=
  nodup_map (List.range count) constructor injective List.nodup_range

/-- No receipt occurs twice, even if two positions happen to carry equal row
values. -/
theorem owners_nodup
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) :
    (owners duplexBase u64Base candidateBase selectorBase count
      initialBuilder).Nodup := by
  unfold owners
  rw [List.nodup_append, List.nodup_append]
  refine
    ⟨⟨mappedRange_nodup RowOwner.u64
          (fun _ _ equal => by cases equal; rfl) _,
        mappedRange_nodup RowOwner.candidate
          (fun _ _ equal => by cases equal; rfl) _,
        ?_⟩,
      mappedRange_nodup RowOwner.selector
        (fun _ _ equal => by cases equal; rfl) _,
      ?_⟩
  · intro left leftMember right rightMember equal
    subst right
    simp only [List.mem_map] at leftMember rightMember
    rcases rightMember with ⟨index, _, rfl⟩
    simp at leftMember
  · intro left leftMember right rightMember equal
    subst right
    simp only [List.mem_append, List.mem_map] at leftMember rightMember
    rcases rightMember with ⟨index, _, rfl⟩
    simp at leftMember

/-- Exactly one structured receipt for every emitted row position. -/
theorem ownership_is_positional
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) :
    (suffixRows duplexBase u64Base candidateBase selectorBase count
        initialBuilder).length =
        (owners duplexBase u64Base candidateBase selectorBase count
          initialBuilder).length
      ∧
        (owners duplexBase u64Base candidateBase selectorBase count
          initialBuilder).Nodup
      ∧
        suffixRows duplexBase u64Base candidateBase selectorBase count
            initialBuilder =
          (owners duplexBase u64Base candidateBase selectorBase count
            initialBuilder).map
            (ownedRow duplexBase u64Base candidateBase selectorBase count
              initialBuilder) := by
  refine
    ⟨?_,
      owners_nodup duplexBase u64Base candidateBase selectorBase count
        initialBuilder,
      rows_eq_map_owners duplexBase u64Base candidateBase selectorBase count
        initialBuilder⟩
  rw [rows_eq_map_owners, List.length_map]

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerOwnership
