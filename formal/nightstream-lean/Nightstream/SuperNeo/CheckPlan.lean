/-!
Generic inclusion-minimality calculus for finite verifier-check plans.

Owns: semantic plan acceptance, soundness, completeness, exactness, proved
redundancy, concrete removal witnesses, and the corresponding certified-removal
and soundness-loss theorems.

Does not own: any protocol predicate, Fiat--Shamir schedule, R1CS row, cost
model, concrete counterexample, or claim of global gate-count minimality.

Emits constraints: no.

Authority boundary: callers supply both each named check's independent
semantics and the target relation. A check may be removed only through a
`Redundant` proof; a `NecessaryForSoundness` witness proves only
inclusion-necessity relative to the supplied plan and semantics.

Plans are lists so order remains inspectable. `without` removes every duplicate
occurrence, ensuring an omitted family cannot remain accidentally active.

| Semantic phase | Family | Mathematical obligation | Main theorem |
|---|---|---|---|
| specification | selected checks | acceptance is conjunction of the named leaves | `Accepts` |
| equivalence | soundness / completeness | plan language equals target language | `Exact` |
| minimization | redundancy | all retained checks imply the removed check | `exact_without_of_redundant` |
| red team | necessity | removing one family admits a non-target input | `not_sound_without_of_necessary` |
| certification | inclusion minimality | full plan is sound and every leaf has a removal witness | `InclusionMinimalSound` |
-/

namespace Nightstream.SuperNeo.CheckPlan

universe uInput uCheck

/-- A plan accepts exactly when every selected check accepts the input. -/
def Accepts
    {Input : Type uInput} {Check : Type uCheck}
    (semantics : Check -> Input -> Prop)
    (checks : List Check) (input : Input) : Prop :=
  forall check, check ∈ checks -> semantics check input

/-- Selected checks are sound when acceptance implies the target relation. -/
def Sound
    {Input : Type uInput} {Check : Type uCheck}
    (semantics : Check -> Input -> Prop)
    (target : Input -> Prop) (checks : List Check) : Prop :=
  forall input, Accepts semantics checks input -> target input

/-- Selected checks are complete when every target input is accepted. -/
def Complete
    {Input : Type uInput} {Check : Type uCheck}
    (semantics : Check -> Input -> Prop)
    (target : Input -> Prop) (checks : List Check) : Prop :=
  forall input, target input -> Accepts semantics checks input

/-- Exactness is pointwise equality of the plan and target languages. -/
def Exact
    {Input : Type uInput} {Check : Type uCheck}
    (semantics : Check -> Input -> Prop)
    (target : Input -> Prop) (checks : List Check) : Prop :=
  forall input, Accepts semantics checks input <-> target input

/-- Membership inclusion between two check plans. -/
def Included {Check : Type uCheck} (small large : List Check) : Prop :=
  forall check, check ∈ small -> check ∈ large

/-- Remove every occurrence of one named family from a plan. -/
def without {Check : Type uCheck} [DecidableEq Check]
    (checks : List Check) (removed : Check) : List Check :=
  checks.filter fun check => decide (check ≠ removed)

@[simp] theorem mem_without_iff
    {Check : Type uCheck} [DecidableEq Check]
    {checks : List Check} {removed candidate : Check} :
    candidate ∈ without checks removed <->
      candidate ∈ checks /\ candidate ≠ removed := by
  simp [without]

/-- One family is redundant when every input accepted by the remaining plan
also satisfies that family's predicate. -/
def Redundant
    {Input : Type uInput} {Check : Type uCheck} [DecidableEq Check]
    (semantics : Check -> Input -> Prop)
    (checks : List Check) (check : Check) : Prop :=
  forall input, Accepts semantics (without checks check) input ->
    semantics check input

/-- A concrete removal witness: the weakened plan accepts a non-target input. -/
def NecessaryForSoundness
    {Input : Type uInput} {Check : Type uCheck} [DecidableEq Check]
    (semantics : Check -> Input -> Prop)
    (target : Input -> Prop) (checks : List Check) (check : Check) : Prop :=
  exists input,
    Accepts semantics (without checks check) input /\ ¬ target input

/-- The plan is sound and every retained family has a concrete removal
counterexample. This is inclusion minimality, not a global circuit-size lower
bound. -/
def InclusionMinimalSound
    {Input : Type uInput} {Check : Type uCheck} [DecidableEq Check]
    (semantics : Check -> Input -> Prop)
    (target : Input -> Prop) (checks : List Check) : Prop :=
  Sound semantics target checks /\
    forall check, check ∈ checks ->
      NecessaryForSoundness semantics target checks check

/-- A candidate plan certified in both directions. -/
structure CertifiedPlan
    {Input : Type uInput} {Check : Type uCheck}
    (semantics : Check -> Input -> Prop)
    (target : Input -> Prop) where
  checks : List Check
  sound : Sound semantics target checks
  complete : Complete semantics target checks

theorem accepts_of_included
    {Input : Type uInput} {Check : Type uCheck}
    {semantics : Check -> Input -> Prop}
    {small large : List Check} {input : Input}
    (included : Included small large)
    (accepted : Accepts semantics large input) :
    Accepts semantics small input := by
  intro check member
  exact accepted check (included check member)

/-- Adding checks preserves soundness. -/
theorem sound_of_included
    {Input : Type uInput} {Check : Type uCheck}
    {semantics : Check -> Input -> Prop} {target : Input -> Prop}
    {small large : List Check}
    (included : Included small large)
    (sound : Sound semantics target small) :
    Sound semantics target large := by
  intro input accepted
  exact sound input (accepts_of_included included accepted)

/-- Removing checks preserves completeness. -/
theorem complete_of_included
    {Input : Type uInput} {Check : Type uCheck}
    {semantics : Check -> Input -> Prop} {target : Input -> Prop}
    {small large : List Check}
    (included : Included small large)
    (complete : Complete semantics target large) :
    Complete semantics target small := by
  intro input targetHolds
  exact accepts_of_included included (complete input targetHolds)

theorem exact_iff_sound_and_complete
    {Input : Type uInput} {Check : Type uCheck}
    {semantics : Check -> Input -> Prop} {target : Input -> Prop}
    {checks : List Check} :
    Exact semantics target checks <->
      Sound semantics target checks /\ Complete semantics target checks := by
  constructor
  · intro exact
    exact ⟨fun input accepted => (exact input).mp accepted,
      fun input targetHolds => (exact input).mpr targetHolds⟩
  · rintro ⟨sound, complete⟩ input
    exact ⟨sound input, complete input⟩

theorem accepts_without_iff_of_redundant
    {Input : Type uInput} {Check : Type uCheck} [DecidableEq Check]
    {semantics : Check -> Input -> Prop} {checks : List Check}
    {check : Check}
    (redundant : Redundant semantics checks check) (input : Input) :
    Accepts semantics checks input <->
      Accepts semantics (without checks check) input := by
  constructor
  · intro accepted candidate member
    exact accepted candidate (mem_without_iff.mp member).1
  · intro accepted candidate member
    by_cases equal : candidate = check
    · subst candidate
      exact redundant input accepted
    · exact accepted candidate (mem_without_iff.mpr ⟨member, equal⟩)

/-- Removing a proved-redundant family preserves exactness. -/
theorem exact_without_of_redundant
    {Input : Type uInput} {Check : Type uCheck} [DecidableEq Check]
    {semantics : Check -> Input -> Prop} {target : Input -> Prop}
    {checks : List Check} {check : Check}
    (exact : Exact semantics target checks)
    (redundant : Redundant semantics checks check) :
    Exact semantics target (without checks check) := by
  intro input
  rw [← accepts_without_iff_of_redundant redundant input]
  exact exact input

/-- A concrete removal witness refutes soundness of the weakened plan. -/
theorem not_sound_without_of_necessary
    {Input : Type uInput} {Check : Type uCheck} [DecidableEq Check]
    {semantics : Check -> Input -> Prop} {target : Input -> Prop}
    {checks : List Check} {check : Check}
    (necessary : NecessaryForSoundness semantics target checks check) :
    ¬ Sound semantics target (without checks check) := by
  rintro sound
  rcases necessary with ⟨input, accepted, notTarget⟩
  exact notTarget (sound input accepted)

theorem inclusionMinimalSound_of_witnesses
    {Input : Type uInput} {Check : Type uCheck} [DecidableEq Check]
    {semantics : Check -> Input -> Prop} {target : Input -> Prop}
    {checks : List Check}
    (sound : Sound semantics target checks)
    (witnesses : forall check, check ∈ checks ->
      NecessaryForSoundness semantics target checks check) :
    InclusionMinimalSound semantics target checks :=
  ⟨sound, witnesses⟩

namespace CertifiedPlan

theorem exact
    {Input : Type uInput} {Check : Type uCheck}
    {semantics : Check -> Input -> Prop} {target : Input -> Prop}
    (plan : CertifiedPlan semantics target) :
    Exact semantics target plan.checks :=
  exact_iff_sound_and_complete.mpr ⟨plan.sound, plan.complete⟩

/-- Produce a smaller certified plan from a proof of semantic redundancy. -/
def withoutRedundant
    {Input : Type uInput} {Check : Type uCheck} [DecidableEq Check]
    {semantics : Check -> Input -> Prop} {target : Input -> Prop}
    (plan : CertifiedPlan semantics target) (check : Check)
    (redundant : Redundant semantics plan.checks check) :
    CertifiedPlan semantics target where
  checks := without plan.checks check
  sound := by
    intro input accepted
    exact plan.sound input
      ((accepts_without_iff_of_redundant redundant input).mpr accepted)
  complete := by
    intro input targetHolds
    exact (accepts_without_iff_of_redundant redundant input).mp
      (plan.complete input targetHolds)

end CertifiedPlan

end Nightstream.SuperNeo.CheckPlan
