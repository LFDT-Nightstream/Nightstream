import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the fold-digest checks that Π_DEC and Π_RLC both run.

Owns: the emitted equality program for `validate_fold_digest_consistency`, its
derived count, soundness, honest completeness, conservation and cost — and the
proof that `validate_fold_digest_canonical` has **no** row program.

## Neither check needs a hash

Both were carried in the ledger as "needs Poseidon2".  Reading them shows
otherwise, and the correction matters because it changes which of them is a
constraint at all.

`validate_fold_digest_consistency` (`pi_dec.rs`, and the same function in
`pi_rlc.rs`) is equality: every child's `fold_digest` must equal the parent's.
No hash is recomputed. That is this module's row program.

`validate_fold_digest_canonical` reads each eight-byte lane little-endian as a
`u64` and rejects it when `value >= F::ORDER_U64`. That is a **range check**, not
a hash either.

## The canonicality check has no row, and that is a theorem

`lcEval` reduces modulo the prime, so **every** value this row layer can carry
is already a canonical residue.  `carried_is_canonical` proves it. There is no
assignment, honest or adversarial, under which a carried lane fails the check.

Emitting rows for it would therefore fabricate a constraint the encoding cannot
violate — the same defect as emitting rows for a list-length assertion, and the
trap prompt section 3 names. What the check does in Rust is guard the **decoder**
against a byte string that does not decode to field elements. That is decoding
work, and it belongs to whatever turns bytes into lanes.

## One row per lane, not two

A fold-digest lane is a **base-field** value, not an extension element.  So a
lane equality is one row, where a `K` equality is two.  `KEquality.rows` is the
`K`-valued pair; this module uses `KEquality.equalityRow` directly, once per
lane, and derives the count from the emitted list.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-! ## Canonicality is unrepresentable to violate -/

/-- **Every carried value is a canonical residue.**

`lcEval` reduces modulo the prime, so the value of any combination under any
assignment is already below it. -/
theorem carried_is_canonical (z : Nat → Nat) (comb : LinComb) :
    lcEval z comb < goldilocksP :=
  Nat.mod_lt _ (by decide)

/-- **So the canonicality check is unfalsifiable here**, and a row program for
it would constrain nothing.

Stated as the non-existence of a violating assignment rather than as a comment,
because "this check needs no rows" is a claim about every assignment and should
be checkable as one. -/
theorem no_canonicality_violation :
    ¬ ∃ (z : Nat → Nat) (comb : LinComb), goldilocksP ≤ lcEval z comb := by
  rintro ⟨z, comb, violation⟩
  exact Nat.not_lt.2 violation (carried_is_canonical z comb)

/-! ## The consistency program

Every child's fold digest equals the parent's, lane by lane. -/

/-- **The emitted fold-digest consistency program.**  One row per lane pair. -/
def digestRows (pairs : List (LinComb × LinComb)) : List Row :=
  pairs.map (fun pair => KEquality.equalityRow pair.1 pair.2)

/-- **The derived row count**, from the emitted list.  One per lane — a
base-field equality, not the two a `K` equality costs. -/
theorem digestRows_length (pairs : List (LinComb × LinComb)) :
    (digestRows pairs).length = pairs.length :=
  List.length_map _

/-- The check allocates nothing. -/
def digestColumns : List Nat := []

theorem digestColumns_length : digestColumns.length = 0 := rfl

theorem digestColumns_nodup : digestColumns.Nodup := List.nodup_nil

/-- **Satisfaction forces every lane to agree.** -/
theorem digestRows_sound
    (z : Nat → Nat) (pairs : List (LinComb × LinComb)) (constantWire : z 0 = 1)
    (satisfied : Satisfies (digestRows pairs) z)
    (pair : LinComb × LinComb) (member : pair ∈ pairs) :
    lcEval z pair.1 = lcEval z pair.2 :=
  (KEquality.equalityRow_iff z pair.1 pair.2 constantWire).1
    (satisfied _ (List.mem_map.2 ⟨pair, member, rfl⟩))

/-- **Agreeing lanes satisfy the check**, under the caller's own assignment.
Nothing is allocated, so there is no witness to extend. -/
theorem digestRows_honest
    (z : Nat → Nat) (pairs : List (LinComb × LinComb)) (constantWire : z 0 = 1)
    (agree : ∀ pair ∈ pairs, lcEval z pair.1 = lcEval z pair.2) :
    Satisfies (digestRows pairs) z := by
  intro row member
  rcases List.mem_map.1 member with ⟨pair, pairMember, rfl⟩
  exact (KEquality.equalityRow_iff z pair.1 pair.2 constantWire).2
    (agree pair pairMember)

/-- **Every column is a compared lane's or the constant wire.** -/
theorem digestRows_conservation
    (pairs : List (LinComb × LinComb)) (row : Row)
    (member : row ∈ digestRows pairs) (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ pair ∈ pairs,
      Mentions pair.1 column ∨ Mentions pair.2 column ∨ column = 0 := by
  rcases List.mem_map.1 member with ⟨pair, pairMember, rfl⟩
  refine ⟨pair, pairMember, ?_⟩
  simp only [KEquality.equalityRow] at mentioned
  rcases mentioned with inA | inB | inC
  · exact Or.inl inA
  · exact Or.inr (Or.inr (by
      simpa only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] using inB))
  · exact Or.inr (Or.inl inC)

/-! ## Cost -/

/-- **The check's cost**, folded over lanes.  Nothing is allocated at any lane,
so the auxiliary component stays zero however many lanes there are. -/
def digestCost (pairs : List (LinComb × LinComb)) : Lowering.Typed.Cost where
  recurringRows := pairs.length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem digestCost_rows (pairs : List (LinComb × LinComb)) :
    (digestRows pairs).length = (digestCost pairs).recurringRows :=
  digestRows_length pairs

theorem digestCost_columns (pairs : List (LinComb × LinComb)) :
    digestColumns.length = (digestCost pairs).auxiliaryColumns :=
  digestColumns_length

/-- **The canonicality check's cost.**  Zero, and derived rather than declared:
`no_canonicality_violation` is why there is nothing to emit. -/
def canonicalityCost : Lowering.Typed.Cost where
  recurringRows := 0
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem canonicalityCost_rows :
    ([] : List Row).length = canonicalityCost.recurringRows := rfl

/-! ## The digest is compression, not authority

This recipe forces `child.fold_digest = parent.fold_digest`, lane by lane.  It
does **not** establish that either digest determines what it compresses, and the
distinction is the project's standing rule: digests are fine as compression,
never as authority.

Where that rule is discharged is `AccumulatorBinding`, whose ownership table
carries the row and whose header states the principle — "Neither digest is
authority unless recomputed from its carrier and reduced through the
corresponding failure partition."  Its
`parent_children_eq_or_commitmentFailure` is the named-event form: two accepted
transitions with the same handle are the same transition, or a
`CommitmentFamilyFailure` occurred.

That reduction is **two-transition**.  It takes two accepted Π_DEC transitions
and a digest collision between them.  The rows below are about **one**
transition and one check, so they are not an input to it and cannot be phrased
in its disjunction: a single-transition arithmetic fact fed into a collision
reduction is a category error, not a stronger theorem.

What the rows are is exactly what a collision reduction needs *underneath* it —
the arithmetic that makes an accepted transition accepted.  Soundness to the
frozen relation is the right form at this scope; the named event belongs where a
digest is being trusted, and here nothing is. -/

/-- **Lane agreement is all that is claimed.**

Restated as its own theorem so the boundary is checkable rather than a comment:
satisfaction gives equality of the compared lanes and nothing about what those
lanes compress. -/
theorem digestRows_claim_is_lane_equality
    (z : Nat → Nat) (pairs : List (LinComb × LinComb)) (constantWire : z 0 = 1)
    (satisfied : Satisfies (digestRows pairs) z) :
    ∀ pair ∈ pairs, lcEval z pair.1 = lcEval z pair.2 :=
  fun pair member => digestRows_sound z pairs constantWire satisfied pair member

/-! ## Row ownership

Section 2 item 3.  Existence is `List.mem_map`.  Uniqueness needs the receipts
distinguishable, and here they are: `equalityRow left right` is
`⟨left, [(0,1)], right⟩`, so two lane pairs emit the same row **only if they are
the same pair**.

**No hypothesis is needed.**  The first draft carried `pairs.Nodup`; the linter
found it unused, and it is: `equalityRow_injective` gives uniqueness outright,
because the row *is* the pair.  A premise nothing consumes is the section-3 trap
in reverse, and dropping it makes the theorem stronger. -/

/-- Two lane pairs emit the same row only if they are the same pair. -/
theorem equalityRow_injective
    (first second : LinComb × LinComb)
    (sameRow : KEquality.equalityRow first.1 first.2
      = KEquality.equalityRow second.1 second.2) :
    first = second := by
  simp only [KEquality.equalityRow, Row.mk.injEq] at sameRow
  exact Prod.ext sameRow.1 sameRow.2.2

/-- **Every emitted row belongs to exactly one lane pair.** -/
theorem digestRows_owned
    (pairs : List (LinComb × LinComb))
    (row : Row) (member : row ∈ digestRows pairs) :
    ∃ pair, pair ∈ pairs
      ∧ row = KEquality.equalityRow pair.1 pair.2
      ∧ ∀ other ∈ pairs,
          row = KEquality.equalityRow other.1 other.2 → other = pair := by
  rcases List.mem_map.1 member with ⟨pair, pairMember, rfl⟩
  exact ⟨pair, pairMember, rfl, fun other _ sameRow =>
    equalityRow_injective other pair sameRow.symm⟩

end Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe
