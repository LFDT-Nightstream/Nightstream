import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompletionProtection

/-!
Contract: aggregate exact occurrence-level separation into the cross-arm
certificate consumed by constructive branch completion.

Owns:
- lifting pairwise temporary/visible separation through both occurrence
  lists;
- no Step- or Terminal-specific path facts.

Does not own: within-arm separation, semantic witnesses, assignment
construction, branch rows, or production artifacts.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

universe u

namespace PrimitivePlan

/-- Exact bidirectional visible separation and temporary separation for two
primitive plans in different arms. -/
theorem crossPairwiseSeparated
    {parameters : Parameters}
    {profile : Profile parameters}
    {firstInput firstOutput secondInput secondOutput :
      Schema (typeSystem parameters)}
    {firstPrimitive :
      Primitive (SelectedSignature parameters)
        firstInput firstOutput}
    {secondPrimitive :
      Primitive (SelectedSignature parameters)
        secondInput secondOutput}
    {firstPath secondPath : OwnerPath}
    {firstInputColumns : Columns firstInput}
    {secondInputColumns : Columns secondInput}
    {one firstActive secondActive : ColumnId}
    (first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns one firstActive)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns one secondActive)
    (oneExcludesFirst :
      one.owner ≠ .typed (.instruction firstPath))
    (secondActiveExcludesFirst :
      secondActive.owner ≠ .typed (.instruction firstPath))
    (oneExcludesSecond :
      one.owner ≠ .typed (.instruction secondPath))
    (firstActiveExcludesSecond :
      firstActive.owner ≠ .typed (.instruction secondPath))
    (different : firstPath ≠ secondPath)
    (firstInputExcludesSecond :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction secondPath)) firstInputColumns)
    (secondInputExcludesFirst :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction firstPath)) secondInputColumns) :
    IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        first.occurrence.visibleIds ∧
      IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.temporaryIds := by
  exact ⟨
    first.occurrenceTemporariesDisjointOtherVisible second
      oneExcludesFirst secondActiveExcludesFirst
      (Ne.symm different) secondInputExcludesFirst,
    second.occurrenceTemporariesDisjointOtherVisible first
      oneExcludesSecond firstActiveExcludesSecond
      different firstInputExcludesSecond,
    first.occurrenceTemporariesDisjointOtherTemporaries
      second different⟩

/-- If the first occurrence has no temporary witnesses, only exclusion of the
second instruction owner from the first visible context remains substantive. -/
theorem crossPairwiseSeparated_of_first_no_temporaries
    {parameters : Parameters}
    {profile : Profile parameters}
    {firstInput firstOutput secondInput secondOutput :
      Schema (typeSystem parameters)}
    {firstPrimitive :
      Primitive (SelectedSignature parameters)
        firstInput firstOutput}
    {secondPrimitive :
      Primitive (SelectedSignature parameters)
        secondInput secondOutput}
    {firstPath secondPath : OwnerPath}
    {firstInputColumns : Columns firstInput}
    {secondInputColumns : Columns secondInput}
    {one firstActive secondActive : ColumnId}
    (first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns one firstActive)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns one secondActive)
    (noTemporaries : first.occurrence.temporaryIds = [])
    (oneExcludesSecond :
      one.owner ≠ .typed (.instruction secondPath))
    (firstActiveExcludesSecond :
      firstActive.owner ≠ .typed (.instruction secondPath))
    (different : firstPath ≠ secondPath)
    (firstInputExcludesSecond :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction secondPath)) firstInputColumns) :
    IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        first.occurrence.visibleIds ∧
      IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.temporaryIds := by
  constructor
  · intro id member
    rw [noTemporaries] at member
    simp at member
  · constructor
    · exact
        second.occurrenceTemporariesDisjointOtherVisible first
          oneExcludesSecond firstActiveExcludesSecond
          different firstInputExcludesSecond
    · intro id member
      rw [noTemporaries] at member
      simp at member

/-- An occurrence's instruction-owned temporaries cannot alias two controls
whose owners exclude that instruction path. -/
theorem occurrenceTemporariesDisjointControls
    {parameters : Parameters}
    {profile : Profile parameters}
    {input output : Schema (typeSystem parameters)}
    {primitive :
      Primitive (SelectedSignature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    (left right : ColumnId)
    (leftExcludes :
      left.owner ≠ .typed (.instruction path))
    (rightExcludes :
      right.owner ≠ .typed (.instruction path)) :
    IdsDisjoint plan.occurrence.temporaryIds [left, right] := by
  intro id temporaryMember controlMember
  have ownerExact :=
    plan.occurrenceTemporaryOwner id temporaryMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at controlMember
  rcases controlMember with equal | equal
  · subst id
    exact leftExcludes ownerExact
  · subst id
    exact rightExcludes ownerExact

end PrimitivePlan

namespace PrimitivePlan.ProtectedExtension

/-- A protected SSA extension also supplies cross-group separation when the
two occurrences use different activation controls. -/
theorem crossPairwiseSeparated
    {parameters : Parameters}
    {profile : Profile parameters}
    {firstInput firstOutput secondInput secondOutput :
      Schema (typeSystem parameters)}
    {firstPrimitive :
      Primitive (SelectedSignature parameters)
        firstInput firstOutput}
    {secondPrimitive :
      Primitive (SelectedSignature parameters)
        secondInput secondOutput}
    {firstPath secondPath : OwnerPath}
    {firstInputColumns : Columns firstInput}
    {secondInputColumns : Columns secondInput}
    {one firstActive secondActive : ColumnId}
    {first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns one firstActive}
    (protection : first.ProtectedExtension secondInputColumns)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns one secondActive)
    (oneExcludesFirst :
      one.owner ≠ .typed (.instruction firstPath))
    (secondActiveExcludesFirst :
      secondActive.owner ≠ .typed (.instruction firstPath))
    (oneExcludesSecond :
      one.owner ≠ .typed (.instruction secondPath))
    (firstActiveExcludesSecond :
      firstActive.owner ≠ .typed (.instruction secondPath))
    (different : firstPath ≠ secondPath)
    (firstInputExcludesSecond :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction secondPath)) firstInputColumns) :
    IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        first.occurrence.visibleIds ∧
      IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.temporaryIds := by
  exact ⟨
    first.occurrenceTemporariesDisjointOtherVisibleOfInput second
      oneExcludesFirst secondActiveExcludesFirst
      (Ne.symm different) protection.temporariesDisjoint,
    second.occurrenceTemporariesDisjointOtherVisible first
      oneExcludesSecond firstActiveExcludesSecond
      different firstInputExcludesSecond,
    first.occurrenceTemporariesDisjointOtherTemporaries
      second different⟩

end PrimitivePlan.ProtectedExtension

namespace CompletionSeparation

/-- Pairwise cross-arm separation plus explicit protection of the other
arm's controls is exactly the aggregate `PlansSeparated` contract. -/
theorem plansSeparated_of_pairwise
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one firstActive secondActive : ColumnId}
    (first : ArmPlan signature family one firstActive)
    (second : ArmPlan signature family one secondActive)
    (pairwise :
      ∀ firstOccurrence,
        firstOccurrence ∈ first.occurrences ->
      ∀ secondOccurrence,
        secondOccurrence ∈ second.occurrences ->
        IdsDisjoint firstOccurrence.temporaryIds
            secondOccurrence.visibleIds ∧
          IdsDisjoint secondOccurrence.temporaryIds
            firstOccurrence.visibleIds ∧
          IdsDisjoint firstOccurrence.temporaryIds
            secondOccurrence.temporaryIds)
    (control :
      IdsDisjoint first.temporaryIds [one, secondActive]) :
    ArmPlan.PlansSeparated first second := by
  constructor
  · intro id firstMember secondMember
    rcases List.mem_flatMap.mp firstMember with
      ⟨firstOccurrence, firstOccurrenceMember, firstTemporary⟩
    rcases List.mem_flatMap.mp secondMember with
      ⟨secondOccurrence, secondOccurrenceMember, secondVisible⟩
    exact
      (pairwise firstOccurrence firstOccurrenceMember
        secondOccurrence secondOccurrenceMember).1
        id firstTemporary secondVisible
  · intro id secondMember firstMember
    rcases List.mem_flatMap.mp secondMember with
      ⟨secondOccurrence, secondOccurrenceMember, secondTemporary⟩
    rcases List.mem_flatMap.mp firstMember with
      ⟨firstOccurrence, firstOccurrenceMember, firstVisible⟩
    exact
      (pairwise firstOccurrence firstOccurrenceMember
        secondOccurrence secondOccurrenceMember).2.1
        id secondTemporary firstVisible
  · intro id firstMember secondMember
    rcases List.mem_flatMap.mp firstMember with
      ⟨firstOccurrence, firstOccurrenceMember, firstTemporary⟩
    rcases List.mem_flatMap.mp secondMember with
      ⟨secondOccurrence, secondOccurrenceMember, secondTemporary⟩
    exact
      (pairwise firstOccurrence firstOccurrenceMember
        secondOccurrence secondOccurrenceMember).2.2
        id firstTemporary secondTemporary
  · exact control

/-- The same pairwise facts can be consumed in the reverse completion order
once the reverse arm's controls are protected. -/
theorem plansSeparated_reverse_of_pairwise
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one firstActive secondActive : ColumnId}
    (first : ArmPlan signature family one firstActive)
    (second : ArmPlan signature family one secondActive)
    (pairwise :
      ∀ firstOccurrence,
        firstOccurrence ∈ first.occurrences ->
      ∀ secondOccurrence,
        secondOccurrence ∈ second.occurrences ->
        IdsDisjoint firstOccurrence.temporaryIds
            secondOccurrence.visibleIds ∧
          IdsDisjoint secondOccurrence.temporaryIds
            firstOccurrence.visibleIds ∧
          IdsDisjoint firstOccurrence.temporaryIds
            secondOccurrence.temporaryIds)
    (control :
      IdsDisjoint second.temporaryIds [one, firstActive]) :
    ArmPlan.PlansSeparated second first := by
  apply plansSeparated_of_pairwise second first
  · intro secondOccurrence secondMember
      firstOccurrence firstMember
    have separated :=
      pairwise firstOccurrence firstMember
        secondOccurrence secondMember
    exact ⟨
      separated.2.1,
      separated.1,
      by
        intro id secondTemporary firstTemporary
        exact separated.2.2 id firstTemporary secondTemporary⟩
  · exact control

end CompletionSeparation

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
