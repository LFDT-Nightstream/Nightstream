import Mathlib.Data.Finset.Basic

/-!
Owns: generic selection, exactness, redundancy, and necessity calculus for
finite verifier-check plans.

Does not own: SuperNeo predicates, Fiat-Shamir semantics, R1CS blocks, or
concrete counterexamples.

Emits constraints: no.

Authority boundary: callers supply both check semantics and the target
relation; a selected check set is authoritative only through proved soundness
and completeness.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Plan language | `Accepts`, `Sound`, `Complete`, `Exact` | Defines semantic plan correctness |
| Safe pruning | `exact_erase_of_redundant` | Preserves exactness for a proved redundant check |
| Necessity | `not_sound_erase_of_necessary` | Rejects removal when a counterexample exists |
-/

namespace SuperNeo.FPrimeRecursiveVerifier

universe u v

/-- A plan accepts an input exactly when every selected check accepts it. -/
def Accepts
    {Input : Type u} {Check : Type v}
    (semantics : Check → Input → Prop)
    (checks : Finset Check)
    (input : Input) : Prop :=
  ∀ check, check ∈ checks → semantics check input

/-- Selected checks are sound when acceptance implies the target relation. -/
def Sound
    {Input : Type u} {Check : Type v}
    (semantics : Check → Input → Prop)
    (target : Input → Prop)
    (checks : Finset Check) : Prop :=
  ∀ input, Accepts semantics checks input → target input

/-- Selected checks are complete when every target execution is accepted. -/
def Complete
    {Input : Type u} {Check : Type v}
    (semantics : Check → Input → Prop)
    (target : Input → Prop)
    (checks : Finset Check) : Prop :=
  ∀ input, target input → Accepts semantics checks input

/-- Exactness is pointwise equality of the selected and target languages. -/
def Exact
    {Input : Type u} {Check : Type v}
    (semantics : Check → Input → Prop)
    (target : Input → Prop)
    (checks : Finset Check) : Prop :=
  ∀ input, Accepts semantics checks input ↔ target input

/-- A check is redundant when all the other selected checks imply it. -/
def Redundant
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    (semantics : Check → Input → Prop)
    (checks : Finset Check)
    (check : Check) : Prop :=
  ∀ input, Accepts semantics (checks.erase check) input → semantics check input

/--
A concrete necessity witness demonstrates unsoundness after one selected check
is removed.
-/
def NecessaryForSoundness
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    (semantics : Check → Input → Prop)
    (target : Input → Prop)
    (checks : Finset Check)
    (check : Check) : Prop :=
  ∃ input, Accepts semantics (checks.erase check) input ∧ ¬ target input

/-- Soundness together with a removal attack for every retained check. -/
def InclusionMinimalSound
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    (semantics : Check → Input → Prop)
    (target : Input → Prop)
    (checks : Finset Check) : Prop :=
  Sound semantics target checks ∧
    ∀ check, check ∈ checks →
      NecessaryForSoundness semantics target checks check

/-- A certified candidate packages both directions of language equivalence. -/
structure CertifiedPlan
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    (semantics : Check → Input → Prop)
    (target : Input → Prop) where
  checks : Finset Check
  sound : Sound semantics target checks
  complete : Complete semantics target checks

theorem accepts_of_superset
    {Input : Type u} {Check : Type v}
    {semantics : Check → Input → Prop}
    {small large : Finset Check}
    {input : Input}
    (hSubset : small ⊆ large)
    (hLarge : Accepts semantics large input) :
    Accepts semantics small input := by
  intro check hCheck
  exact hLarge check (hSubset hCheck)

/-- Adding checks preserves soundness. -/
theorem sound_of_superset
    {Input : Type u} {Check : Type v}
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    {small large : Finset Check}
    (hSubset : small ⊆ large)
    (hSound : Sound semantics target small) :
    Sound semantics target large := by
  intro input hLarge
  exact hSound input (accepts_of_superset hSubset hLarge)

/-- Removing checks preserves completeness. -/
theorem complete_of_subset
    {Input : Type u} {Check : Type v}
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    {small large : Finset Check}
    (hSubset : small ⊆ large)
    (hComplete : Complete semantics target large) :
    Complete semantics target small := by
  intro input hTarget
  exact accepts_of_superset hSubset (hComplete input hTarget)

theorem exact_iff_sound_and_complete
    {Input : Type u} {Check : Type v}
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    {checks : Finset Check} :
    Exact semantics target checks ↔
      Sound semantics target checks ∧ Complete semantics target checks := by
  constructor
  · intro hExact
    constructor
    · intro input hAccepts
      exact (hExact input).mp hAccepts
    · intro input hTarget
      exact (hExact input).mpr hTarget
  · rintro ⟨hSound, hComplete⟩ input
    exact ⟨hSound input, hComplete input⟩

theorem accepts_erase_iff_of_redundant
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    {semantics : Check → Input → Prop}
    {checks : Finset Check}
    {check : Check}
    (hRedundant : Redundant semantics checks check)
    (input : Input) :
    Accepts semantics checks input ↔
      Accepts semantics (checks.erase check) input := by
  constructor
  · exact accepts_of_superset (Finset.erase_subset check checks)
  · intro hErased candidate hCandidate
    by_cases hEq : candidate = check
    · subst candidate
      exact hRedundant input hErased
    · exact hErased candidate (Finset.mem_erase.mpr ⟨hEq, hCandidate⟩)

/-- Removing a proved-redundant check preserves exactness. -/
theorem exact_erase_of_redundant
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    {checks : Finset Check}
    {check : Check}
    (hExact : Exact semantics target checks)
    (hRedundant : Redundant semantics checks check) :
    Exact semantics target (checks.erase check) := by
  intro input
  rw [← accepts_erase_iff_of_redundant hRedundant input]
  exact hExact input

/-- A necessity witness refutes soundness of the plan with that check removed. -/
theorem not_sound_erase_of_necessary
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    {checks : Finset Check}
    {check : Check}
    (hNecessary : NecessaryForSoundness semantics target checks check) :
    ¬ Sound semantics target (checks.erase check) := by
  rintro hSound
  rcases hNecessary with ⟨input, hAccepts, hNotTarget⟩
  exact hNotTarget (hSound input hAccepts)

theorem inclusionMinimalSound_of_witnesses
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    {checks : Finset Check}
    (hSound : Sound semantics target checks)
    (hWitnesses :
      ∀ check, check ∈ checks →
        NecessaryForSoundness semantics target checks check) :
    InclusionMinimalSound semantics target checks :=
  ⟨hSound, hWitnesses⟩

namespace CertifiedPlan

theorem exact
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    (plan : CertifiedPlan semantics target) :
    Exact semantics target plan.checks :=
  (exact_iff_sound_and_complete).2 ⟨plan.sound, plan.complete⟩

/-- Produce a smaller certified plan from a proof of redundancy. -/
def eraseRedundant
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    (plan : CertifiedPlan semantics target)
    (check : Check)
    (hRedundant : Redundant semantics plan.checks check) :
    CertifiedPlan semantics target where
  checks := plan.checks.erase check
  sound := by
    intro input hErased
    exact plan.sound input
      ((accepts_erase_iff_of_redundant hRedundant input).mpr hErased)
  complete := by
    intro input hTarget
    exact (accepts_erase_iff_of_redundant hRedundant input).mp
      (plan.complete input hTarget)

end CertifiedPlan

end SuperNeo.FPrimeRecursiveVerifier
