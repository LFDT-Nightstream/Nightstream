import Lean.Elab.Tactic.Omega
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: finite normal-form selection for independently justified lowering
candidates.

Owns:
- total and transitive use of the fixed `Cost.LexLe` optimization order;
- deterministic selection from an explicitly nonempty finite candidate list;
- membership, semantic preservation, and least-cost certificates for the
  selected candidate.

Does not own: the candidate semantics, a lowering vocabulary, physical rows,
Rust behavior, generated artifacts, or a claim of global arithmetization
minimality.  The caller supplies the semantic correctness relation independently
of every candidate's cost.

The proved minimum is relative only to the supplied finite list and the fixed
order `(recurring rows, committed columns, public columns, auxiliary columns)`.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks.NormalForm

open Nightstream.Implementation.Lowering.Typed

universe u v

/-! ## Fixed cost order -/

instance (left right : Cost) : Decidable (Cost.LexLe left right) := by
  unfold Cost.LexLe
  infer_instance

/-- Executable view of the project's fixed lexicographic cost relation. -/
def lexLeBool (left right : Cost) : Bool :=
  decide (Cost.LexLe left right)

@[simp] theorem lexLeBool_eq_true_iff (left right : Cost) :
    lexLeBool left right = true ↔ Cost.LexLe left right := by
  simp [lexLeBool]

theorem lexLe_refl (cost : Cost) :
    Cost.LexLe cost cost := by
  simp [Cost.LexLe]

theorem lexLe_total (left right : Cost) :
    Cost.LexLe left right ∨ Cost.LexLe right left := by
  unfold Cost.LexLe
  omega

theorem lexLe_trans {first second third : Cost}
    (firstSecond : Cost.LexLe first second)
    (secondThird : Cost.LexLe second third) :
    Cost.LexLe first third := by
  unfold Cost.LexLe at firstSecond secondThird ⊢
  omega

/-! ## Selection kernel -/

/-- Keep the left candidate on a cost tie, making selection deterministic. -/
def prefer {Candidate : Type u}
    (cost : Candidate -> Cost)
    (left right : Candidate) : Candidate :=
  if Cost.LexLe (cost left) (cost right) then left else right

theorem prefer_eq_left_or_right {Candidate : Type u}
    (cost : Candidate -> Cost)
    (left right : Candidate) :
    prefer cost left right = left ∨ prefer cost left right = right := by
  by_cases leftLe : Cost.LexLe (cost left) (cost right)
  · exact Or.inl (by simp [prefer, leftLe])
  · exact Or.inr (by simp [prefer, leftLe])

theorem prefer_le_left {Candidate : Type u}
    (cost : Candidate -> Cost)
    (left right : Candidate) :
    Cost.LexLe (cost (prefer cost left right)) (cost left) := by
  by_cases leftLe : Cost.LexLe (cost left) (cost right)
  · simpa [prefer, leftLe] using lexLe_refl (cost left)
  · rcases lexLe_total (cost left) (cost right) with forward | reverse
    · exact False.elim (leftLe forward)
    · simpa [prefer, leftLe] using reverse

theorem prefer_le_right {Candidate : Type u}
    (cost : Candidate -> Cost)
    (left right : Candidate) :
    Cost.LexLe (cost (prefer cost left right)) (cost right) := by
  by_cases leftLe : Cost.LexLe (cost left) (cost right)
  · simpa [prefer, leftLe] using leftLe
  · simpa [prefer, leftLe] using lexLe_refl (cost right)

/-- Select the least-cost member of `head :: tail`.

The separate head makes emptiness unrepresentable at this boundary. -/
def least {Candidate : Type u}
    (cost : Candidate -> Cost) :
    Candidate -> List Candidate -> Candidate
  | head, [] => head
  | head, next :: tail => prefer cost head (least cost next tail)

theorem least_mem {Candidate : Type u}
    (cost : Candidate -> Cost)
    (head : Candidate)
    (tail : List Candidate) :
    least cost head tail ∈ head :: tail := by
  induction tail generalizing head with
  | nil =>
      simp [least]
  | cons next tail inductionHypothesis =>
      simp only [least]
      rcases prefer_eq_left_or_right cost head (least cost next tail) with
        chosenHead | chosenTail
      · rw [chosenHead]
        exact List.mem_cons_self
      · rw [chosenTail]
        exact List.mem_cons_of_mem head (inductionHypothesis next)

theorem least_le_every_member {Candidate : Type u}
    (cost : Candidate -> Cost)
    (head : Candidate)
    (tail : List Candidate)
    {candidate : Candidate}
    (member : candidate ∈ head :: tail) :
    Cost.LexLe (cost (least cost head tail)) (cost candidate) := by
  induction tail generalizing head candidate with
  | nil =>
      have equal : candidate = head := by
        simpa only [List.mem_singleton] using member
      subst candidate
      exact lexLe_refl (cost head)
  | cons next tail inductionHypothesis =>
      rcases List.mem_cons.mp member with equal | tailMember
      · subst candidate
        simpa only [least] using
          prefer_le_left cost head (least cost next tail)
      · have selectedLeTail :
            Cost.LexLe
              (cost (prefer cost head (least cost next tail)))
              (cost (least cost next tail)) :=
          prefer_le_right cost head (least cost next tail)
        have tailLeCandidate :
            Cost.LexLe (cost (least cost next tail)) (cost candidate) :=
          inductionHypothesis next tailMember
        simpa only [least] using
          lexLe_trans selectedLeTail tailLeCandidate

/-! ## Semantically justified finite candidates -/

/-- A nonempty finite set of candidates for one specification.

`Implements` is an explicit caller-owned semantic relation.  It does not
mention `Cost`, and this structure stores no cost-derived correctness claim. -/
structure FiniteCandidates
    (Candidate : Type u)
    (Specification : Type v)
    (Implements : Candidate -> Specification -> Prop)
    (specification : Specification) where
  head : Candidate
  tail : List Candidate
  correct :
    ∀ candidate, candidate ∈ head :: tail ->
      Implements candidate specification

namespace FiniteCandidates

/-- The exact finite list over which canonicality is proved. -/
def members
    {Candidate : Type u}
    {Specification : Type v}
    {Implements : Candidate -> Specification -> Prop}
    {specification : Specification}
    (candidates :
      FiniteCandidates Candidate Specification Implements specification) :
    List Candidate :=
  candidates.head :: candidates.tail

theorem members_ne_nil
    {Candidate : Type u}
    {Specification : Type v}
    {Implements : Candidate -> Specification -> Prop}
    {specification : Specification}
    (candidates :
      FiniteCandidates Candidate Specification Implements specification) :
    candidates.members ≠ [] := by
  simp [members]

/-- Deterministic least-cost selection from the declared candidate list. -/
def canonical
    {Candidate : Type u}
    {Specification : Type v}
    {Implements : Candidate -> Specification -> Prop}
    {specification : Specification}
    (candidates :
      FiniteCandidates Candidate Specification Implements specification)
    (cost : Candidate -> Cost) : Candidate :=
  least cost candidates.head candidates.tail

theorem canonical_mem
    {Candidate : Type u}
    {Specification : Type v}
    {Implements : Candidate -> Specification -> Prop}
    {specification : Specification}
    (candidates :
      FiniteCandidates Candidate Specification Implements specification)
    (cost : Candidate -> Cost) :
    candidates.canonical cost ∈ candidates.members := by
  exact least_mem cost candidates.head candidates.tail

/-- Selection preserves the independently supplied candidate semantics. -/
theorem canonical_correct
    {Candidate : Type u}
    {Specification : Type v}
    {Implements : Candidate -> Specification -> Prop}
    {specification : Specification}
    (candidates :
      FiniteCandidates Candidate Specification Implements specification)
    (cost : Candidate -> Cost) :
    Implements (candidates.canonical cost) specification := by
  exact candidates.correct _ (candidates.canonical_mem cost)

/-- The selected candidate is no more expensive than every declared member.

This is inclusion-minimality only inside `candidates.members`; it says nothing
about encodings outside that explicit finite set. -/
theorem canonical_minimum
    {Candidate : Type u}
    {Specification : Type v}
    {Implements : Candidate -> Specification -> Prop}
    {specification : Specification}
    (candidates :
      FiniteCandidates Candidate Specification Implements specification)
    (cost : Candidate -> Cost)
    {candidate : Candidate}
    (member : candidate ∈ candidates.members) :
    Cost.LexLe
      (cost (candidates.canonical cost))
      (cost candidate) := by
  exact least_le_every_member
    cost candidates.head candidates.tail member

/-- Compact certificate combining semantic preservation and finite-list
minimality. -/
theorem canonical_certificate
    {Candidate : Type u}
    {Specification : Type v}
    {Implements : Candidate -> Specification -> Prop}
    {specification : Specification}
    (candidates :
      FiniteCandidates Candidate Specification Implements specification)
    (cost : Candidate -> Cost) :
    Implements (candidates.canonical cost) specification ∧
      ∀ candidate, candidate ∈ candidates.members ->
        Cost.LexLe
          (cost (candidates.canonical cost))
          (cost candidate) := by
  exact ⟨candidates.canonical_correct cost,
    fun _ member => candidates.canonical_minimum cost member⟩

end FiniteCandidates

end Nightstream.Implementation.Lowering.Goldilocks.NormalForm
