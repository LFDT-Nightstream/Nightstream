import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Contract: canonical aggregation for sparse linear combinations.

A combination built by concatenation retains duplicate column entries, so its
*syntactic* length grows without bound across composed linear layers even
though its *mathematical* support does not.  This module supplies the
aggregating normal form and proves it semantics-preserving, which is what makes
a never-materialize encoding implementable rather than merely countable.

Owns: `insertTerm`, `normalize`, the raw-sum invariant, semantic preservation
under `lcEval`, and the support projection.

Does not own: any Poseidon2-specific round structure; the support *recurrence*
across rounds is `POSEIDON2-SUPPORT-BOUND` and lives with the schedule.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.LinCombNormal

open Nightstream.Implementation.R1CS

/-- A sparse linear combination over columns. -/
abbrev LinComb := List (Nat × Nat)

/-! ## Raw sum

`lcEval` reduces modulo the prime only at the end, so the invariant is easier
to state on the unreduced sum. -/

def rawSum (z : Nat → Nat) (terms : LinComb) : Nat :=
  terms.foldl (fun acc term => acc + term.2 * z term.1) 0

theorem lcEval_eq_rawSum (z : Nat → Nat) (terms : LinComb) :
    lcEval z terms = rawSum z terms % goldilocksP := rfl

/-- `foldl` with an accumulator offset, needed to peel a head term. -/
theorem rawSum_cons (z : Nat → Nat) (term : Nat × Nat) (terms : LinComb) :
    rawSum z (term :: terms) = term.2 * z term.1 + rawSum z terms := by
  unfold rawSum
  simp only [List.foldl_cons, Nat.zero_add]
  generalize term.2 * z term.1 = head
  induction terms generalizing head with
  | nil => simp
  | cons next rest hypothesis =>
      simp only [List.foldl_cons, Nat.zero_add]
      rw [hypothesis, hypothesis (next.2 * z next.1)]
      omega

/-- The raw sum is additive over concatenation. -/
theorem rawSum_append (z : Nat → Nat) (left right : LinComb) :
    rawSum z (left ++ right) = rawSum z left + rawSum z right := by
  induction left with
  | nil => simp [rawSum]
  | cons term rest hypothesis =>
      rw [List.cons_append, rawSum_cons, rawSum_cons, hypothesis]
      omega

/-- A concatenating combination evaluates to the sum of its pieces.  This is
what lets a matrix application — which is a `flatMap` — be evaluated blockwise
instead of unfolded. -/
theorem rawSum_flatMap {α : Type} (z : Nat → Nat) (list : List α)
    (f : α → LinComb) :
    rawSum z (list.flatMap f) = ((list.map (fun x => rawSum z (f x))).sum) := by
  induction list with
  | nil => simp [rawSum]
  | cons head tail hypothesis =>
      rw [List.flatMap_cons, rawSum_append, hypothesis]
      simp

/-! ## Aggregating insertion -/

/-- Insert one term, merging into an existing entry for the same column. -/
def insertTerm (term : Nat × Nat) : LinComb → LinComb
  | [] => [term]
  | entry :: rest =>
      if entry.1 = term.1 then (entry.1, entry.2 + term.2) :: rest
      else entry :: insertTerm term rest

/-- **Insertion is semantics-preserving on the raw sum.** -/
theorem rawSum_insertTerm (z : Nat → Nat) (term : Nat × Nat) (terms : LinComb) :
    rawSum z (insertTerm term terms) = term.2 * z term.1 + rawSum z terms := by
  induction terms with
  | nil => simp [insertTerm, rawSum_cons, rawSum]
  | cons entry rest hypothesis =>
      unfold insertTerm
      by_cases same : entry.1 = term.1
      · rw [if_pos same, rawSum_cons, rawSum_cons, same, Nat.add_mul]
        omega
      · rw [if_neg same, rawSum_cons, hypothesis, rawSum_cons]
        omega

/-! ## Normalization -/

/-- Canonical aggregating normal form: every column appears at most once. -/
def normalize (comb : LinComb) : LinComb :=
  comb.foldr insertTerm []

/-- **Normalization preserves the raw sum.** -/
theorem rawSum_normalize (z : Nat → Nat) (comb : LinComb) :
    rawSum z (normalize comb) = rawSum z comb := by
  unfold normalize
  induction comb with
  | nil => rfl
  | cons term rest hypothesis =>
      simp only [List.foldr_cons]
      rw [rawSum_insertTerm, hypothesis, rawSum_cons]

/-- **Normalization is semantics-preserving.**  This is what licenses carrying
combinations symbolically instead of materializing them: the aggregated form
evaluates identically, so no row is needed to re-establish equality. -/
theorem lcEval_normalize (z : Nat → Nat) (comb : LinComb) :
    lcEval z (normalize comb) = lcEval z comb := by
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum, rawSum_normalize]

/-! ## Support

The mathematical support is the set of columns actually referenced.  It is what
stays bounded across rounds, unlike syntactic length. -/

def support (comb : LinComb) : List Nat :=
  (normalize comb).map Prod.fst

theorem support_length_le (comb : LinComb) :
    (support comb).length = (normalize comb).length := by
  simp [support]

/-- Insertion grows the normalized length by at most one: merging an existing
column keeps it fixed. -/
theorem insertTerm_length_le (term : Nat × Nat) (terms : LinComb) :
    (insertTerm term terms).length ≤ terms.length + 1 := by
  induction terms with
  | nil => simp [insertTerm]
  | cons entry rest hypothesis =>
      unfold insertTerm
      by_cases same : entry.1 = term.1
      · rw [if_pos same]; simp
      · rw [if_neg same]; simp; omega

/-- **Support never exceeds the number of terms**, so aggregation can only
shrink a combination. -/
theorem normalize_length_le (comb : LinComb) :
    (normalize comb).length ≤ comb.length := by
  unfold normalize
  induction comb with
  | nil => simp
  | cons term rest hypothesis =>
      simp only [List.foldr_cons, List.length_cons]
      exact Nat.le_trans (insertTerm_length_le term _)
        (Nat.succ_le_succ hypothesis)

/-! ## Support algebra

The recurrence that bounds a never-materialize encoding rests on three facts:
a single-column combination has support one; concatenation cannot introduce a
column absent from both sides; and scaling changes no column.  An S-box output
is a fresh single column, which is why S-boxing *resets* support rather than
growing it. -/

/-- Membership in the support is membership among the columns. -/
def Mentions (comb : LinComb) (column : Nat) : Prop :=
  column ∈ comb.map Prod.fst

instance (comb : LinComb) (column : Nat) : Decidable (Mentions comb column) := by
  unfold Mentions; infer_instance

/-- A single-column combination mentions exactly that column. -/
theorem mentions_single (column target : Nat) (coefficient : Nat) :
    Mentions [(column, coefficient)] target ↔ target = column := by
  simp [Mentions]

/-- Scaling changes no column. -/
theorem mentions_map_scale
    (factor : Nat) (comb : LinComb) (target : Nat) :
    Mentions (comb.map (fun term => (term.1, factor * term.2 % goldilocksP))) target
      ↔ Mentions comb target := by
  simp [Mentions]

/-- Concatenation introduces no new column. -/
theorem mentions_append (left right : LinComb) (target : Nat) :
    Mentions (left ++ right) target ↔ Mentions left target ∨ Mentions right target := by
  simp [Mentions]

/-- Insertion mentions exactly the old columns plus the inserted one. -/
theorem mentions_insertTerm (term : Nat × Nat) (terms : LinComb) (target : Nat) :
    Mentions (insertTerm term terms) target ↔
      Mentions terms target ∨ target = term.1 := by
  induction terms with
  | nil => simp [insertTerm, Mentions]
  | cons entry rest hypothesis =>
      unfold insertTerm
      by_cases same : entry.1 = term.1
      · rw [if_pos same]
        simp only [Mentions, List.map_cons, List.mem_cons]
        constructor
        · intro left; exact Or.inl left
        · rintro (left | rfl)
          · exact left
          · exact Or.inl same.symm
      · rw [if_neg same]
        simp only [Mentions, List.map_cons, List.mem_cons] at hypothesis ⊢
        constructor
        · rintro (head | tail)
          · exact Or.inl (Or.inl head)
          · rcases hypothesis.1 tail with old | fresh
            · exact Or.inl (Or.inr old)
            · exact Or.inr fresh
        · rintro ((head | old) | fresh)
          · exact Or.inl head
          · exact Or.inr (hypothesis.2 (Or.inl old))
          · exact Or.inr (hypothesis.2 (Or.inr fresh))

/-! ## Normalization produces a duplicate-free form

The coefficient count of a carried combination is the length of its normal
form, so that length has to be pinned exactly rather than bounded.  These are
what turn `normalize` from "at most as long" into "exactly one entry per
referenced column". -/

/-- Inserting a column not already present grows the list by exactly one. -/
theorem insertTerm_length_of_fresh
    (term : Nat × Nat) (terms : LinComb) (fresh : ¬ Mentions terms term.1) :
    (insertTerm term terms).length = terms.length + 1 := by
  induction terms with
  | nil => simp [insertTerm]
  | cons entry rest hypothesis =>
      have notHead : entry.1 ≠ term.1 := by
        intro same
        apply fresh
        simp only [Mentions, List.map_cons, List.mem_cons]
        exact Or.inl same.symm
      have notRest : ¬ Mentions rest term.1 := by
        intro member
        apply fresh
        simp only [Mentions, List.map_cons, List.mem_cons]
        exact Or.inr member
      unfold insertTerm
      rw [if_neg notHead]
      simp [hypothesis notRest]

/-- Insertion preserves duplicate-freeness: it either merges into an existing
entry or appends a genuinely new column. -/
theorem insertTerm_nodup
    (term : Nat × Nat) (terms : LinComb)
    (nodup : (terms.map Prod.fst).Nodup) :
    ((insertTerm term terms).map Prod.fst).Nodup := by
  induction terms with
  | nil => simp [insertTerm]
  | cons entry rest hypothesis =>
      rw [List.map_cons, List.nodup_cons] at nodup
      unfold insertTerm
      by_cases same : entry.1 = term.1
      · rw [if_pos same]
        simp only [List.map_cons, List.nodup_cons]
        exact ⟨nodup.1, nodup.2⟩
      · rw [if_neg same]
        simp only [List.map_cons, List.nodup_cons]
        refine ⟨?_, hypothesis nodup.2⟩
        intro member
        rcases (mentions_insertTerm term rest entry.1).1 member with old | isTerm
        · exact nodup.1 old
        · exact same isTerm

/-- **The normal form has exactly one entry per referenced column.** -/
theorem normalize_nodup (comb : LinComb) :
    ((normalize comb).map Prod.fst).Nodup := by
  unfold normalize
  induction comb with
  | nil => simp
  | cons term rest hypothesis =>
      simp only [List.foldr_cons]
      exact insertTerm_nodup term _ hypothesis

/-- **Normalization preserves support.**  Aggregation merges duplicate entries
and introduces nothing, so the bound may be reasoned about before or after
normalizing. -/
theorem mentions_normalize (comb : LinComb) (target : Nat) :
    Mentions (normalize comb) target ↔ Mentions comb target := by
  unfold normalize
  induction comb with
  | nil => simp [Mentions]
  | cons term rest hypothesis =>
      simp only [List.foldr_cons]
      rw [mentions_insertTerm, hypothesis]
      simp only [Mentions, List.map_cons, List.mem_cons]
      constructor
      · rintro (old | rfl)
        · exact Or.inr old
        · exact Or.inl rfl
      · rintro (rfl | old)
        · exact Or.inr rfl
        · exact Or.inl old

/-- **Normalization is length-preserving on a duplicate-free combination.** -/
theorem normalize_length_of_nodup
    (comb : LinComb) (nodup : (comb.map Prod.fst).Nodup) :
    (normalize comb).length = comb.length := by
  unfold normalize
  induction comb with
  | nil => simp
  | cons term rest hypothesis =>
      rw [List.map_cons, List.nodup_cons] at nodup
      simp only [List.foldr_cons, List.length_cons]
      rw [insertTerm_length_of_fresh term _ ?fresh, hypothesis nodup.2]
      case fresh =>
        intro member
        exact nodup.1 ((mentions_normalize rest term.1).1 member)


/-! ## Counting a normal form against a witness list

The normal form's length is the number of distinct columns referenced, so it can
be read off any duplicate-free list with the same membership.  Core supplies the
`erase` lemmas but no "two duplicate-free lists with equal membership have equal
length", so it is proved here. -/

/-- `List.Nodup.map` is not available without Mathlib. -/
theorem nodup_map {α β : Type} (list : List α) (f : α → β)
    (inj : ∀ a b, f a = f b → a = b) (nodup : list.Nodup) :
    (list.map f).Nodup := by
  induction list with
  | nil => simp
  | cons head tail hypothesis =>
      rw [List.nodup_cons] at nodup
      simp only [List.map_cons, List.nodup_cons]
      refine ⟨?_, hypothesis nodup.2⟩
      intro member
      rcases List.mem_map.1 member with ⟨other, memberOther, image⟩
      exact nodup.1 (inj other head image ▸ memberOther)

theorem nodup_length_eq {α : Type} [BEq α] [LawfulBEq α] (left : List α) :
    ∀ right : List α, left.Nodup → right.Nodup →
      (∀ x, x ∈ left ↔ x ∈ right) → left.length = right.length := by
  induction left with
  | nil =>
      intro right _ _ same
      have empty : right = [] :=
        List.eq_nil_iff_forall_not_mem.2 (fun a member => by
          simpa using (same a).2 member)
      simp [empty]
  | cons head tail hypothesis =>
      intro right nodupLeft nodupRight same
      rw [List.nodup_cons] at nodupLeft
      have headMem : head ∈ right := (same head).1 List.mem_cons_self
      have tailSame : ∀ x, x ∈ tail ↔ x ∈ right.erase head := by
        intro x
        constructor
        · intro member
          have distinct : x ≠ head := fun equal => nodupLeft.1 (equal ▸ member)
          exact (List.mem_erase_of_ne distinct).2
            ((same x).1 (List.mem_cons_of_mem _ member))
        · intro member
          have distinct : x ≠ head := by
            intro equal
            subst equal
            exact List.Nodup.not_mem_erase nodupRight member
          rcases List.mem_cons.1 ((same x).2 (List.mem_of_mem_erase member)) with
            isHead | inTail
          · exact absurd isHead distinct
          · exact inTail
      have lengths := hypothesis (right.erase head) nodupLeft.2
        (List.Nodup.erase head nodupRight) tailSame
      have nonempty : 1 ≤ right.length := by
        cases right with
        | nil => simp at headMem
        | cons _ _ => simp
      rw [List.length_cons, lengths, List.length_erase_of_mem headMem]
      omega

/-- **The normal form's length is the size of any duplicate-free witness list
for its support.**  This is what turns a support characterization into an exact
coefficient count. -/
theorem normalize_length_eq_witness
    (comb : LinComb) (witness : List Nat)
    (witnessNodup : witness.Nodup)
    (agree : ∀ column, Mentions comb column ↔ column ∈ witness) :
    (normalize comb).length = witness.length := by
  have columns : ((normalize comb).map Prod.fst).length = witness.length := by
    refine nodup_length_eq _ witness (normalize_nodup comb) witnessNodup ?_
    intro column
    calc column ∈ (normalize comb).map Prod.fst
        ↔ Mentions (normalize comb) column := Iff.rfl
      _ ↔ Mentions comb column := mentions_normalize comb column
      _ ↔ column ∈ witness := agree column
  simpa using columns

/-! ## Reading one normalized coefficient

The compact Poseidon2 cancellation certificate reasons about one basis column
at a time.  Evaluating a combination under the Kronecker assignment for that
column reads exactly its normalized coefficient. -/

def basisAssignment (target : Nat) : Nat → Nat :=
  fun column => if column = target then 1 else 0

theorem lcEval_basis_singleton
    (target column coefficient : Nat) :
    lcEval (basisAssignment target) [(column, coefficient)] =
      if column = target then coefficient % goldilocksP else 0 := by
  by_cases same : column = target
  · simp [lcEval, rawSum, basisAssignment, same]
  · simp [lcEval, rawSum, basisAssignment, same]

theorem rawSum_basis_not_mentions
    (comb : LinComb) (target : Nat) (absent : ¬ Mentions comb target) :
    rawSum (basisAssignment target) comb = 0 := by
  induction comb with
  | nil => rfl
  | cons term rest hypothesis =>
      have headNe : term.1 ≠ target := by
        intro equal
        apply absent
        simp only [Mentions, List.map_cons, List.mem_cons]
        exact Or.inl equal.symm
      have tailAbsent : ¬ Mentions rest target := by
        intro member
        apply absent
        simp only [Mentions, List.map_cons, List.mem_cons]
        exact Or.inr member
      rw [rawSum_cons, basisAssignment, if_neg headNe, Nat.mul_zero,
        Nat.zero_add, hypothesis tailAbsent]

theorem lcEval_basis_not_mentions
    (comb : LinComb) (target : Nat) (absent : ¬ Mentions comb target) :
    lcEval (basisAssignment target) comb = 0 := by
  rw [lcEval_eq_rawSum, rawSum_basis_not_mentions comb target absent]
  exact Nat.zero_mod _

theorem lcEval_basis_of_nodup
    (comb : LinComb) (entry : Nat × Nat)
    (nodup : (comb.map Prod.fst).Nodup) (member : entry ∈ comb) :
    lcEval (basisAssignment entry.1) comb = entry.2 % goldilocksP := by
  induction comb generalizing entry with
  | nil => simp at member
  | cons head rest hypothesis =>
      rw [List.map_cons, List.nodup_cons] at nodup
      rcases List.mem_cons.1 member with same | inRest
      · subst entry
        rw [lcEval_eq_rawSum, rawSum_cons, basisAssignment, if_pos rfl,
          Nat.mul_one, rawSum_basis_not_mentions rest head.1 nodup.1,
          Nat.add_zero]
      · have headNe : head.1 ≠ entry.1 := by
          intro equal
          apply nodup.1
          have : entry.1 ∈ rest.map Prod.fst := by
            exact List.mem_map.2 ⟨entry, inRest, rfl⟩
          simpa [equal] using this
        rw [lcEval_eq_rawSum, rawSum_cons, basisAssignment, if_neg headNe,
          Nat.mul_zero, Nat.zero_add, ← lcEval_eq_rawSum]
        exact hypothesis entry nodup.2 inRest

/-- Evaluating the original combination on an entry's basis column reads the
coefficient carried by its normal form. -/
theorem lcEval_basis_normalized_entry
    (comb : LinComb) (entry : Nat × Nat) (member : entry ∈ normalize comb) :
    lcEval (basisAssignment entry.1) comb = entry.2 % goldilocksP := by
  rw [← lcEval_normalize]
  exact lcEval_basis_of_nodup (normalize comb) entry
    (normalize_nodup comb) member


/-! ## Field-canonical normal form

`normalize` merges duplicate columns but adds coefficients as unbounded
naturals.  Two consequences, neither affecting `lcEval`:

  * a merged coefficient can exceed the prime, so a stored entry need not be a
    canonical residue — and Rust stores canonical residues, so row-level
    equality against production could fail on representation alone;
  * a merged coefficient congruent to zero is retained, so the entry count is
    an upper bound on the number of nonzero field coefficients, not the count.

`fieldNormalize` reduces every coefficient and drops the zeros.  Its length is
therefore the exact nonzero coefficient count, not a bound, and every entry it
carries is a canonical residue. -/

def reduceTerm (term : Nat × Nat) : Option (Nat × Nat) :=
  if term.2 % goldilocksP = 0 then none else some (term.1, term.2 % goldilocksP)

def fieldNormalize (comb : LinComb) : LinComb :=
  (normalize comb).filterMap reduceTerm

/-- Dropping zero-valued terms and reducing the rest preserves the value. -/
theorem rawSum_filterMap_reduceTerm (z : Nat → Nat) (comb : LinComb) :
    rawSum z (comb.filterMap reduceTerm) % goldilocksP
      = rawSum z comb % goldilocksP := by
  induction comb with
  | nil => simp [rawSum]
  | cons term rest hypothesis =>
      by_cases vanishes : term.2 % goldilocksP = 0
      · have dropped : reduceTerm term = none := by simp [reduceTerm, vanishes]
        rw [List.filterMap_cons_none dropped, hypothesis, rawSum_cons,
          Nat.add_mod]
        have zeroed : term.2 * z term.1 % goldilocksP = 0 := by
          rw [← Nat.mod_mul_mod, vanishes, Nat.zero_mul, Nat.zero_mod]
        rw [zeroed, Nat.zero_add, Nat.mod_mod]
      · have kept : reduceTerm term
            = some (term.1, term.2 % goldilocksP) := by
          simp [reduceTerm, vanishes]
        rw [List.filterMap_cons_some kept, rawSum_cons, rawSum_cons,
          Nat.add_mod, hypothesis, ← Nat.add_mod, Nat.add_mod]
        rw [Nat.mod_mul_mod]
        rw [← Nat.add_mod]

/-- **Field normalization is semantics-preserving.** -/
theorem lcEval_fieldNormalize (z : Nat → Nat) (comb : LinComb) :
    lcEval z (fieldNormalize comb) = lcEval z comb := by
  unfold fieldNormalize
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum, rawSum_filterMap_reduceTerm,
    rawSum_normalize]

/-- **Every stored coefficient is a canonical residue.**  This is what a
row-level comparison against Rust needs. -/
theorem fieldNormalize_canonical (comb : LinComb) :
    ∀ term ∈ fieldNormalize comb, term.2 < goldilocksP := by
  intro term member
  unfold fieldNormalize at member
  rcases List.mem_filterMap.1 member with ⟨source, _, image⟩
  unfold reduceTerm at image
  split at image
  · exact absurd image (by simp)
  · rw [← Option.some_inj.1 image]
    exact Nat.mod_lt _ (by decide)

/-- **No stored coefficient vanishes.**  Together with
`fieldNormalize_canonical` this is what makes the length an exact nonzero
count rather than an upper bound. -/
theorem fieldNormalize_nonzero (comb : LinComb) :
    ∀ term ∈ fieldNormalize comb, term.2 ≠ 0 := by
  intro term member
  unfold fieldNormalize at member
  rcases List.mem_filterMap.1 member with ⟨source, _, image⟩
  unfold reduceTerm at image
  split at image
  · exact absurd image (by simp)
  · rename_i notZero
    rw [← Option.some_inj.1 image]
    exact notZero

/-- Field normalization introduces no column.  Only this direction holds: a
column whose coefficient vanishes is dropped. -/
theorem mentions_fieldNormalize_subset (comb : LinComb) (column : Nat)
    (mentioned : Mentions (fieldNormalize comb) column) :
    Mentions (normalize comb) column := by
  unfold fieldNormalize at mentioned
  unfold Mentions at mentioned ⊢
  rcases List.mem_map.1 mentioned with ⟨term, member, rfl⟩
  rcases List.mem_filterMap.1 member with ⟨source, sourceMember, image⟩
  unfold reduceTerm at image
  split at image
  · exact absurd image (by simp)
  · rw [← Option.some_inj.1 image]
    exact List.mem_map.2 ⟨source, sourceMember, rfl⟩

/-- Field normalization can only shrink the entry count, so every bound proved
for `normalize` transfers. -/
theorem fieldNormalize_length_le (comb : LinComb) :
    (fieldNormalize comb).length ≤ (normalize comb).length := by
  unfold fieldNormalize
  exact List.length_filterMap_le _ _


/-! ## Coefficients survive normalization when columns are distinct

If no two entries share a column, `insertTerm` never merges, so every entry of
the normal form is an entry of the original with its coefficient intact.  That
is what lets matrix density rule out cancellation without any computation: a
combination built as one entry per column keeps exactly those coefficients. -/

theorem insertTerm_entries
    (term : Nat × Nat) (terms : LinComb) (fresh : ¬ Mentions terms term.1) :
    ∀ entry ∈ insertTerm term terms, entry = term ∨ entry ∈ terms := by
  induction terms with
  | nil => intro entry member; simp [insertTerm] at member; exact Or.inl member
  | cons head rest hypothesis =>
      have notHead : head.1 ≠ term.1 := by
        intro same
        apply fresh
        simp only [Mentions, List.map_cons, List.mem_cons]
        exact Or.inl same.symm
      have notRest : ¬ Mentions rest term.1 := by
        intro member
        apply fresh
        simp only [Mentions, List.map_cons, List.mem_cons]
        exact Or.inr member
      intro entry member
      rw [insertTerm, if_neg notHead, List.mem_cons] at member
      rcases member with rfl | inTail
      · exact Or.inr List.mem_cons_self
      · rcases hypothesis notRest entry inTail with isTerm | inRest
        · exact Or.inl isTerm
        · exact Or.inr (List.mem_cons_of_mem _ inRest)

/-- **Normalization preserves entries on a duplicate-free combination.** -/
theorem normalize_entries_of_nodup
    (comb : LinComb) (nodup : (comb.map Prod.fst).Nodup) :
    ∀ entry ∈ normalize comb, entry ∈ comb := by
  unfold normalize
  induction comb with
  | nil => simp
  | cons term rest hypothesis =>
      rw [List.map_cons, List.nodup_cons] at nodup
      intro entry member
      simp only [List.foldr_cons] at member
      have fresh : ¬ Mentions (rest.foldr insertTerm []) term.1 := by
        intro mentioned
        exact nodup.1 ((mentions_normalize rest term.1).1 mentioned)
      rcases insertTerm_entries term _ fresh entry member with rfl | inRest
      · exact List.mem_cons_self
      · exact List.mem_cons_of_mem _ (hypothesis nodup.2 entry inRest)

/-- Nothing is dropped when every normalized coefficient is a nonzero
residue. -/
theorem filterMap_reduceTerm_length
    (comb : LinComb)
    (allNonzero : ∀ entry ∈ comb, entry.2 % goldilocksP ≠ 0) :
    (comb.filterMap reduceTerm).length = comb.length := by
  induction comb with
  | nil => simp
  | cons entry rest hypothesis =>
      have kept : reduceTerm entry = some (entry.1, entry.2 % goldilocksP) := by
        simp [reduceTerm, allNonzero entry List.mem_cons_self]
      rw [List.filterMap_cons_some kept, List.length_cons, List.length_cons,
        hypothesis (fun other member =>
          allNonzero other (List.mem_cons_of_mem _ member))]

theorem fieldNormalize_length_of_nonzero
    (comb : LinComb)
    (allNonzero : ∀ entry ∈ normalize comb, entry.2 % goldilocksP ≠ 0) :
    (fieldNormalize comb).length = (normalize comb).length :=
  filterMap_reduceTerm_length (normalize comb) allNonzero

end Nightstream.Implementation.R1CS.Canonical.LinCombNormal
