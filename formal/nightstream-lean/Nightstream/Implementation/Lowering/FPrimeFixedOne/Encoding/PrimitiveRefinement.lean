import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPrimitivePlan
import Nightstream.Implementation.Lowering.Goldilocks.SelectedBranch

/-!
Contract: semantic refinement for one selected fixed-one primitive receipt.

Owns:
- the exact conversion from a `PrimitivePlan` to its structural physical
  occurrence;
- the canonical result-column context for that occurrence;
- active row soundness against the independently stated typed primitive
  relation;
- active and inactive honest temporary completion.

Does not own: whole-block traversal, branch activation or joining,
whole-program row satisfaction, production Rust behavior, generated rows, or
numeric R1CS column indices.

Every theorem is artifact-independent and ranges over certified call recipes.
No caller-supplied acceptance proposition appears in this interface.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

universe u

namespace PrimitivePlan

/-- The structural occurrence represented by the plan's nonoptional
instruction receipt. -/
def occurrence
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
        inputColumns one active) :
    ArmOccurrence (SelectedSignature parameters)
      (SelectedFamily parameters profile) one active :=
  match plan with
  | .invoke invokePlan =>
      .call _ invokePlan.recipe invokePlan.frame
        invokePlan.oneExact invokePlan.activeExact
  | .literal literalPlan =>
      .literal _ literalPlan.recipe literalPlan.oneExact
  | .assertTrue assertPlan =>
      .assertion assertPlan.recipe
        assertPlan.oneExact assertPlan.activeExact

/-- Canonical physical columns for the complete typed result context. -/
def resultColumns
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
        inputColumns one active) :
    Columns output :=
  match plan with
  | @invoke _ _ _ call operands path inputColumns one active
      invokePlan =>
      (instructionColumns path
        ((SelectedSignature parameters).callOutputs call)).append
          inputColumns
  | @literal _ _ _ port value path inputColumns one active
      literalPlan =>
      (instructionColumns path [port]).append inputColumns
  | @assertTrue _ _ _ condition path inputColumns one active
      assertPlan =>
      inputColumns

/-- Extending a canonical SSA context at a different instruction path
preserves exclusion of an earlier instruction owner. -/
theorem resultExcludesInstruction
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
    (target : OwnerPath)
    (different : path ≠ target)
    (inputExcludes :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction target)) inputColumns) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target)) plan.resultColumns := by
  cases plan with
  | invoke invokePlan =>
      exact CanonicalPrimitivePlan.ContextExcludesOwner.append
        (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
          path target _ different)
        inputExcludes
  | literal literalPlan =>
      exact CanonicalPrimitivePlan.ContextExcludesOwner.append
        (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
          path target _ different)
        inputExcludes
  | assertTrue assertPlan =>
      exact inputExcludes

/-- Every row dependency of the structural occurrence is either visible or
one of that occurrence's explicitly declared temporary coordinates. -/
theorem occurrenceRowsSupported
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
        inputColumns one active) :
    ∀ id, id ∈ rowsColumns plan.occurrence.rows ->
      id ∈ plan.occurrence.visibleIds ++
        plan.occurrence.temporaryIds := by
  intro id member
  rcases List.mem_flatMap.mp member with
    ⟨row, rowMember, columnMember⟩
  have ownedColumnMember : id ∈ row.columnIds := by
    simpa [rowColumns, OwnedRow.columnIds, Row.columnIds,
      List.map_append] using columnMember
  cases plan with
  | invoke invokePlan =>
      exact invokePlan.recipe.rowsSupported invokePlan.frame row
        rowMember id ownedColumnMember
  | literal literalPlan =>
      have supported :=
        literalPlan.rowsSupported row rowMember id ownedColumnMember
      rw [literalPlan.oneExact] at supported
      change
        id ∈ ([one, active] ++ literalPlan.recipe.output.ids) ++ []
      simp only [List.append_nil]
      rcases List.mem_append.mp supported with
        oneMember | outputMember
      · exact List.mem_append.mpr <| Or.inl <| by
          simpa using Or.inl oneMember
      · exact List.mem_append.mpr <| Or.inr outputMember
  | assertTrue assertPlan =>
      have support :
          id = assertPlan.recipe.active ∨
            id = assertPlan.recipe.one ∨
              id = assertPlan.recipe.condition := by
        simpa [occurrence, ArmOccurrence.rows, BoolAssertRecipe.rows,
          rowsColumns, rowColumns, CanonicalRow.row,
          Goldilocks.singleton, Goldilocks.oneMinus] using member
      change
        id ∈
          [assertPlan.recipe.one, assertPlan.recipe.active,
            assertPlan.recipe.condition] ++ []
      simp only [List.append_nil]
      simp [assertPlan.oneExact, assertPlan.activeExact] at support ⊢
      rcases support with
        activeMember | oneMember | conditionMember
      · exact Or.inr <| Or.inl activeMember
      · exact Or.inl oneMember
      · exact Or.inr <| Or.inr conditionMember

/-- Temporary witnesses are disjoint from every visible coordinate of the
same canonical primitive occurrence. -/
theorem occurrenceTemporariesDisjointVisible
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
        inputColumns one active) :
    IdsDisjoint plan.occurrence.temporaryIds
      plan.occurrence.visibleIds := by
  cases plan with
  | invoke invokePlan =>
      exact invokePlan.frame.temporariesDisjointVisible
  | literal literalPlan =>
      intro id member
      simp [occurrence, ArmOccurrence.temporaryIds] at member
  | assertTrue assertPlan =>
      intro id member
      simp [occurrence, ArmOccurrence.temporaryIds] at member

/-- Every temporary coordinate retains the exact source-instruction owner.
This is the cross-occurrence separation key used by whole-program honest
completion. -/
theorem occurrenceTemporaryOwner
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
        inputColumns one active) :
    ∀ id, id ∈ plan.occurrence.temporaryIds ->
      id.owner = .typed (.instruction path) := by
  intro id member
  cases plan with
  | invoke invokePlan =>
      change id ∈ invokePlan.frame.temporaries.ids at member
      unfold LayoutBundles.ids at member
      rcases List.mem_map.mp member with
        ⟨column, columnMember, rfl⟩
      rw [← invokePlan.ownerExact]
      exact invokePlan.frame.allocationsOwned column
        (List.mem_append_right _ columnMember)
  | literal literalPlan =>
      simp [occurrence, ArmOccurrence.temporaryIds] at member
  | assertTrue assertPlan =>
      simp [occurrence, ArmOccurrence.temporaryIds] at member

/-- Completion temporaries cannot change any coordinate in the primitive's
complete typed result context. -/
theorem occurrenceTemporariesDisjointResultColumns
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
        inputColumns one active) :
    IdsDisjoint plan.occurrence.temporaryIds
      plan.resultColumns.toSchemaBundles.ids := by
  intro id temporaryMember resultMember
  cases plan with
  | invoke invokePlan =>
      change
        id ∈
          (Columns.toSchemaBundles
            (HVec.append
              (instructionColumns path
                ((SelectedSignature parameters).callOutputs _))
              inputColumns)).ids at resultMember
      rw [Columns.append_ids] at resultMember
      apply invokePlan.frame.temporariesDisjointVisible
        id temporaryMember
      rcases List.mem_append.mp resultMember with
        outputMember | contextMember
      · have outputMember' :
            id ∈ invokePlan.frame.outputs.ids := by
          rw [invokePlan.outputsExact]
          exact outputMember
        simp [CallFrame.visibleIds, outputMember']
      · have contextMember' :
            id ∈ invokePlan.frame.contextBundles.ids := by
          rw [invokePlan.contextExact]
          exact contextMember
        simp [CallFrame.visibleIds, contextMember']
  | literal literalPlan =>
      simp [occurrence, ArmOccurrence.temporaryIds] at temporaryMember
  | assertTrue assertPlan =>
      simp [occurrence, ArmOccurrence.temporaryIds] at temporaryMember

/-- Every occurrence-visible coordinate is either a shared control or belongs
to the primitive's complete typed result context. -/
theorem occurrenceVisibleSubsetControlsAndResult
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
        inputColumns one active) :
    ∀ id, id ∈ plan.occurrence.visibleIds ->
      id ∈ [one, active] ++
        plan.resultColumns.toSchemaBundles.ids := by
  intro id member
  cases plan with
  | invoke invokePlan =>
      change
        id ∈ [one, active] ++
          (Columns.toSchemaBundles
            (HVec.append
              (instructionColumns path
                ((SelectedSignature parameters).callOutputs _))
              inputColumns)).ids
      rw [Columns.append_ids]
      change
        id ∈
          [invokePlan.frame.one, invokePlan.frame.active] ++
            invokePlan.frame.contextBundles.ids ++
              invokePlan.frame.outputs.ids at member
      rw [invokePlan.oneExact, invokePlan.activeExact,
        invokePlan.contextExact, invokePlan.outputsExact] at member
      rcases List.mem_append.mp member with
        controlOrContext | outputMember
      · rcases List.mem_append.mp controlOrContext with
          controlMember | contextMember
        · exact List.mem_append_left _ controlMember
        · exact List.mem_append_right _ <|
            List.mem_append_right _ contextMember
      · exact List.mem_append_right _ <|
          List.mem_append_left _ outputMember
  | @literal port value path inputColumns one active literalPlan =>
      change
        id ∈ [one, active] ++
          (Columns.toSchemaBundles
            (HVec.append
              (instructionColumns path [port])
              inputColumns)).ids
      rw [Columns.append_ids]
      simp only [occurrence, ArmOccurrence.visibleIds] at member
      rcases List.mem_append.mp member with
        controlMember | outputMember
      · exact List.mem_append_left _ controlMember
      · exact List.mem_append_right _ <|
          List.mem_append_left _ <| by
            apply SchemaBundles.get_ids_subset
              (.here port)
              (instructionColumns path [port]).toSchemaBundles
            rw [Columns.toSchemaBundles_get]
            change
              id ∈
                (HVec.head
                  (instructionColumns path [port])).toColumnBundle.ids
            rw [← literalPlan.outputExact]
            exact outputMember
  | assertTrue assertPlan =>
      change
        id ∈
          [assertPlan.recipe.one, assertPlan.recipe.active,
            assertPlan.recipe.condition] at member
      change id ∈ [one, active] ++ inputColumns.toSchemaBundles.ids
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with oneMember | activeMember | conditionMember
      · apply List.mem_append_left
        simpa [assertPlan.oneExact] using Or.inl oneMember
      · apply List.mem_append_left
        simpa [assertPlan.activeExact] using Or.inr activeMember
      · apply List.mem_append_right
        apply SchemaBundles.get_ids_subset _ _
        rw [assertPlan.conditionIdsExact]
        simpa using conditionMember

/-- If the shared controls and complete input context exclude an instruction
owner, then so does every visible coordinate of a different occurrence. -/
theorem occurrenceVisibleExcludesInstruction
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
    (target : OwnerPath)
    (oneExcludes :
      one.owner ≠ .typed (.instruction target))
    (activeExcludes :
      active.owner ≠ .typed (.instruction target))
    (different : path ≠ target)
    (inputExcludes :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction target)) inputColumns) :
    ∀ id, id ∈ plan.occurrence.visibleIds ->
      id.owner ≠ .typed (.instruction target) := by
  intro id member
  have covered :=
    plan.occurrenceVisibleSubsetControlsAndResult id member
  rcases List.mem_append.mp covered with
    controlMember | resultMember
  · simp only [List.mem_cons, List.not_mem_nil, or_false]
      at controlMember
    rcases controlMember with equal | equal
    · subst id
      exact oneExcludes
    · subst id
      exact activeExcludes
  · exact
      CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
        (plan.resultExcludesInstruction target different inputExcludes)
        id resultMember

/-- Extending an already protected typed context at a distinct instruction
path preserves disjointness from an earlier occurrence's temporaries. -/
theorem occurrenceTemporariesDisjointOtherResult
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
    (different : secondPath ≠ firstPath)
    (inputDisjoint :
      IdsDisjoint first.occurrence.temporaryIds
        secondInputColumns.toSchemaBundles.ids) :
    IdsDisjoint first.occurrence.temporaryIds
      second.resultColumns.toSchemaBundles.ids := by
  intro id temporaryMember resultMember
  cases second with
  | invoke invokePlan =>
      change
        id ∈
          (Columns.toSchemaBundles
            (HVec.append
              (instructionColumns secondPath
                ((SelectedSignature parameters).callOutputs _))
              secondInputColumns)).ids at resultMember
      rw [Columns.append_ids] at resultMember
      rcases List.mem_append.mp resultMember with
        outputMember | inputMember
      · exact
          (CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
            (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
              secondPath firstPath _ different)
            id outputMember)
          (first.occurrenceTemporaryOwner id temporaryMember)
      · exact inputDisjoint id temporaryMember inputMember
  | literal literalPlan =>
      change
        id ∈
          (Columns.toSchemaBundles
            (HVec.append
              (instructionColumns secondPath [_])
              secondInputColumns)).ids at resultMember
      rw [Columns.append_ids] at resultMember
      rcases List.mem_append.mp resultMember with
        outputMember | inputMember
      · exact
          (CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
            (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
              secondPath firstPath _ different)
            id outputMember)
          (first.occurrenceTemporaryOwner id temporaryMember)
      · exact inputDisjoint id temporaryMember inputMember
  | assertTrue assertPlan =>
      exact inputDisjoint id temporaryMember resultMember

/-- An occurrence whose input context is protected from earlier temporaries
also has a protected visible interface, provided its controls and structural
path are distinct from the earlier instruction owner. -/
theorem occurrenceTemporariesDisjointOtherVisibleOfInput
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
    (oneExcludes :
      one.owner ≠ .typed (.instruction firstPath))
    (secondActiveExcludes :
      secondActive.owner ≠ .typed (.instruction firstPath))
    (different : secondPath ≠ firstPath)
    (inputDisjoint :
      IdsDisjoint first.occurrence.temporaryIds
        secondInputColumns.toSchemaBundles.ids) :
    IdsDisjoint first.occurrence.temporaryIds
      second.occurrence.visibleIds := by
  intro id temporaryMember visibleMember
  have covered :=
    second.occurrenceVisibleSubsetControlsAndResult id visibleMember
  rcases List.mem_append.mp covered with
    controlMember | resultMember
  · have ownerExact :=
      first.occurrenceTemporaryOwner id temporaryMember
    simp only [List.mem_cons, List.not_mem_nil, or_false]
        at controlMember
    rcases controlMember with equal | equal
    · subst id
      exact oneExcludes ownerExact
    · subst id
      exact secondActiveExcludes ownerExact
  · exact
      (first.occurrenceTemporariesDisjointOtherResult
        second different inputDisjoint)
        id temporaryMember resultMember

/-- When two occurrences share their arm controls and the earlier result
context is included in the later input context, the later temporaries cannot
alias the earlier visible interface. -/
theorem laterTemporariesDisjointEarlierVisible
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
    {one active : ColumnId}
    (first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns one active)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns one active)
    (resultIncluded :
      ∀ id, id ∈ first.resultColumns.toSchemaBundles.ids ->
        id ∈ secondInputColumns.toSchemaBundles.ids) :
    IdsDisjoint second.occurrence.temporaryIds
      first.occurrence.visibleIds := by
  intro id temporaryMember visibleMember
  cases second with
  | invoke invokePlan =>
      apply invokePlan.frame.temporariesDisjointVisible
        id temporaryMember
      have covered :=
        first.occurrenceVisibleSubsetControlsAndResult id visibleMember
      rcases List.mem_append.mp covered with
        controlMember | resultMember
      · apply List.mem_append_left
        apply List.mem_append_left
        simpa [invokePlan.oneExact, invokePlan.activeExact] using
          controlMember
      · apply List.mem_append_left
        apply List.mem_append_right
        rw [invokePlan.contextExact]
        exact resultIncluded id resultMember
  | literal literalPlan =>
      simp [occurrence, ArmOccurrence.temporaryIds] at temporaryMember
  | assertTrue assertPlan =>
      simp [occurrence, ArmOccurrence.temporaryIds] at temporaryMember

/-- Complete ordered pair certificate for two distinct primitive plans in
one SSA arm. -/
theorem occurrencePairwiseSeparatedOfInput
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
    {one active : ColumnId}
    (first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns one active)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns one active)
    (oneExcludes :
      one.owner ≠ .typed (.instruction firstPath))
    (activeExcludes :
      active.owner ≠ .typed (.instruction firstPath))
    (different : secondPath ≠ firstPath)
    (inputDisjoint :
      IdsDisjoint first.occurrence.temporaryIds
        secondInputColumns.toSchemaBundles.ids)
    (resultIncluded :
      ∀ id, id ∈ first.resultColumns.toSchemaBundles.ids ->
        id ∈ secondInputColumns.toSchemaBundles.ids) :
    IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        first.occurrence.visibleIds ∧
      IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.temporaryIds := by
  exact ⟨
    first.occurrenceTemporariesDisjointOtherVisibleOfInput second
      oneExcludes activeExcludes different inputDisjoint,
    first.laterTemporariesDisjointEarlierVisible second resultIncluded,
    by
      intro id firstMember secondMember
      apply different
      exact Owner.instruction.inj <|
        PhysicalOwner.typed.inj <|
          (second.occurrenceTemporaryOwner id secondMember).symm.trans
            (first.occurrenceTemporaryOwner id firstMember)⟩

/-- Temporaries of one canonical occurrence cannot alias visible coordinates
of a distinct occurrence whose exact input context excludes the first
instruction owner. -/
theorem occurrenceTemporariesDisjointOtherVisible
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
    (oneExcludes :
      one.owner ≠ .typed (.instruction firstPath))
    (secondActiveExcludes :
      secondActive.owner ≠ .typed (.instruction firstPath))
    (different : secondPath ≠ firstPath)
    (secondInputExcludes :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction firstPath)) secondInputColumns) :
    IdsDisjoint first.occurrence.temporaryIds
      second.occurrence.visibleIds := by
  intro id temporaryMember visibleMember
  exact
    (second.occurrenceVisibleExcludesInstruction firstPath
      oneExcludes secondActiveExcludes different secondInputExcludes
      id visibleMember)
    (first.occurrenceTemporaryOwner id temporaryMember)

/-- Distinct structural instruction paths have disjoint canonical temporary
witness coordinates. -/
theorem occurrenceTemporariesDisjointOtherTemporaries
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
    (different : firstPath ≠ secondPath) :
    IdsDisjoint first.occurrence.temporaryIds
      second.occurrence.temporaryIds := by
  intro id firstMember secondMember
  have firstOwner :=
    first.occurrenceTemporaryOwner id firstMember
  have secondOwner :=
    second.occurrenceTemporaryOwner id secondMember
  apply different
  exact Owner.instruction.inj <|
    PhysicalOwner.typed.inj <| firstOwner.symm.trans secondOwner

/-- Any coordinate list that excludes the occurrence's structural owner is
disjoint from its temporary witnesses. -/
theorem occurrenceTemporariesDisjointOwnerExcluded
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
    (ids : List ColumnId)
    (excluded :
      ∀ id, id ∈ ids ->
        id.owner ≠ .typed (.instruction path)) :
    IdsDisjoint plan.occurrence.temporaryIds ids := by
  intro id temporaryMember preservedMember
  exact excluded id preservedMember
    (plan.occurrenceTemporaryOwner id temporaryMember)

@[simp] theorem occurrence_rows
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
        inputColumns one active) :
    plan.occurrence.rows = plan.receipt.rows := by
  cases plan <;>
    rfl

@[simp] theorem occurrence_allocations
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
        inputColumns one active) :
    plan.occurrence.allocations = plan.receipt.allocations := by
  cases plan <;>
    rfl

private theorem schemaBundles_get_decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {schema : Schema types}
    {kind : types.Kind}
    (reference : Ref types schema kind)
    (bundles : SchemaBundles schema)
    (values : Schema.Values types schema)
    (decoded : bundles.Decodes family assignment values) :
    (bundles.get reference).Decodes family kind assignment
      (reference.get values) := by
  induction reference with
  | here =>
      cases bundles
      cases values
      exact decoded.1
  | there reference inductionHypothesis =>
      cases bundles
      cases values
      exact inductionHypothesis _ _ decoded.2

private theorem refBundles_fromSchema_decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context : Schema types}
    {sorts : List types.Kind}
    (references : Refs types context sorts)
    (bundles : SchemaBundles context)
    (values : Schema.Values types context)
    (decoded : bundles.Decodes family assignment values) :
    (RefBundles.fromSchema references bundles).Decodes
      family assignment (references.get values) := by
  induction references with
  | nil =>
      trivial
  | cons reference tail inductionHypothesis =>
      exact ⟨
        schemaBundles_get_decodes
          family assignment reference bundles values decoded,
        inductionHypothesis⟩

private theorem schemaBundles_get_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {schema : Schema types}
    {kind : types.Kind}
    (reference : Ref types schema kind)
    (bundles : SchemaBundles schema)
    (values : Schema.Values types schema)
    (encoded : bundles.Encodes family assignment values) :
    (bundles.get reference).Encodes family kind assignment
      (reference.get values) := by
  induction reference with
  | here =>
      cases bundles
      cases values
      exact encoded.1
  | there reference inductionHypothesis =>
      cases bundles
      cases values
      exact inductionHypothesis _ _ encoded.2

private theorem refBundles_fromSchema_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context : Schema types}
    {sorts : List types.Kind}
    (references : Refs types context sorts)
    (bundles : SchemaBundles context)
    (values : Schema.Values types context)
    (encoded : bundles.Encodes family assignment values) :
    (RefBundles.fromSchema references bundles).Encodes
      family assignment (references.get values) := by
  induction references with
  | nil =>
      trivial
  | cons reference tail inductionHypothesis =>
      exact ⟨
        schemaBundles_get_encodes
          family assignment reference bundles values encoded,
        inductionHypothesis⟩

private theorem columns_append_decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {left right : Schema types}
    (leftColumns : Columns left)
    (rightColumns : Columns right)
    (leftValues : Schema.Values types left)
    (rightValues : Schema.Values types right)
    (leftDecoded :
      Columns.Decodes family leftColumns assignment leftValues)
    (rightDecoded :
      Columns.Decodes family rightColumns assignment rightValues) :
    Columns.Decodes family (leftColumns.append rightColumns) assignment
      (leftValues.append rightValues) := by
  induction leftColumns with
  | nil =>
      cases leftValues
      exact rightDecoded
  | cons head tail inductionHypothesis =>
      cases leftValues with
      | cons value values =>
          exact ⟨leftDecoded.1,
            inductionHypothesis values leftDecoded.2⟩

private theorem columns_left_encodes_of_append
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {left right : Schema types}
    (leftColumns : Columns left)
    (rightColumns : Columns right)
    (leftValues : Schema.Values types left)
    (rightValues : Schema.Values types right)
    (encoded :
      Columns.Encodes family (leftColumns.append rightColumns) assignment
        (leftValues.append rightValues)) :
    Columns.Encodes family leftColumns assignment leftValues := by
  induction leftColumns with
  | nil =>
      cases leftValues
      trivial
  | cons head tail inductionHypothesis =>
      cases leftValues with
      | cons value values =>
          exact ⟨encoded.1,
            inductionHypothesis values encoded.2⟩

private theorem columns_right_encodes_of_append
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {left right : Schema types}
    (leftColumns : Columns left)
    (rightColumns : Columns right)
    (leftValues : Schema.Values types left)
    (rightValues : Schema.Values types right)
    (encoded :
      Columns.Encodes family (leftColumns.append rightColumns) assignment
        (leftValues.append rightValues)) :
    Columns.Encodes family rightColumns assignment rightValues := by
  induction leftColumns with
  | nil =>
      cases leftValues
      exact encoded
  | cons head tail inductionHypothesis =>
      cases leftValues with
      | cons value values =>
          exact inductionHypothesis values encoded.2

private theorem bundle_values_eq_ids_map
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field) :
    bundle.values assignment = bundle.ids.map assignment := by
  simp [ColumnBundle.values, ColumnBundle.ids, List.map_map]

/-- Satisfied active rows force the independently defined typed primitive
relation and decode the complete canonical result context. -/
theorem activeSound
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
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1)
    (source : Schema.Values (typeSystem parameters) input)
    (sourceDecoded :
      Columns.Decodes (SelectedFamily parameters profile)
        inputColumns assignment source)
    (rowsHold : Satisfies plan.receipt.rows assignment) :
    ∃ result : Schema.Values (typeSystem parameters) output,
      primitive.Holds source result ∧
        Columns.Decodes (SelectedFamily parameters profile)
          plan.resultColumns assignment result := by
  cases plan with
  | @invoke call operands path inputColumns one active invokePlan =>
      have operandsDecoded :
          invokePlan.frame.operands.Decodes
            (SelectedFamily parameters profile) assignment
            (operands.get source) := by
        rw [CallFrame.operands, invokePlan.contextExact]
        exact refBundles_fromSchema_decodes
          (SelectedFamily parameters profile) assignment
          operands inputColumns.toSchemaBundles source sourceDecoded
      have frameOne : assignment invokePlan.frame.one = 1 := by
        rw [invokePlan.oneExact]
        exact constantOne
      have frameActive : assignment invokePlan.frame.active = 1 := by
        rw [invokePlan.activeExact]
        exact activeOne
      have activeResult :=
        invokePlan.recipe.activeSoundness invokePlan.frame assignment
          (operands.get source)
          frameOne frameActive
          operandsDecoded
          (by simpa [occurrence_rows] using rowsHold)
      rcases activeResult with ⟨callOutputs, evaluated, outputsDecoded⟩
      refine ⟨callOutputs.append source, ?_, ?_⟩
      · exact ⟨callOutputs, evaluated, rfl⟩
      · have canonicalOutputsDecoded :
            (instructionColumns path
                ((SelectedSignature parameters).callOutputs call)
              ).toSchemaBundles.Decodes
                (SelectedFamily parameters profile) assignment
                callOutputs := by
            rw [← invokePlan.outputsExact]
            exact outputsDecoded
        change
          Columns.Decodes (SelectedFamily parameters profile)
            ((instructionColumns path
                ((SelectedSignature parameters).callOutputs call)).append
              inputColumns)
            assignment (callOutputs.append source)
        exact columns_append_decodes
          (SelectedFamily parameters profile) assignment
          (instructionColumns path
            ((SelectedSignature parameters).callOutputs call))
          inputColumns callOutputs source
          canonicalOutputsDecoded sourceDecoded
  | @literal port value path inputColumns one active literalPlan =>
      have recipeOne : assignment literalPlan.recipe.one = 1 := by
        rw [literalPlan.oneExact]
        exact constantOne
      have literalDecoded :
          literalPlan.recipe.output.Decodes
            (SelectedFamily parameters profile) port.kind
            assignment literalPlan.recipe.value :=
        literalPlan.recipe.decode_of_satisfies
          assignment recipeOne
          (by
            rw [literalPlan.valueExact]
            exact literalPlan.admissible)
          (by simpa [occurrence_rows] using rowsHold)
      refine ⟨.cons value source, rfl, ?_⟩
      have canonicalLiteralDecoded :
          (HVec.head
              (instructionColumns path [port])).toColumnBundle.Decodes
            (SelectedFamily parameters profile) port.kind
            assignment value := by
        have decodedValue :
            literalPlan.recipe.output.Decodes
              (SelectedFamily parameters profile) port.kind
              assignment value := by
          simpa only [literalPlan.valueExact] using literalDecoded
        rw [literalPlan.outputExact] at decodedValue
        exact decodedValue
      change
        Columns.Decodes (SelectedFamily parameters profile)
          ((instructionColumns path [port]).append inputColumns)
          assignment (.cons value source)
      exact ⟨canonicalLiteralDecoded, sourceDecoded⟩
  | @assertTrue condition path inputColumns one active assertPlan =>
      have recipeOne : assignment assertPlan.recipe.one = 1 := by
        rw [assertPlan.oneExact]
        exact constantOne
      have recipeActive : assignment assertPlan.recipe.active = 1 := by
        rw [assertPlan.activeExact]
        exact activeOne
      have asserted :
          boolCodec.decode [assignment assertPlan.recipe.condition] =
            some true :=
        (assertPlan.recipe.active_iff_decode_true
          laws assignment recipeOne recipeActive).mp
          (by simpa [occurrence_rows] using rowsHold)
      have sourceConditionDecoded :
          (inputColumns.toSchemaBundles.get condition).Decodes
            (SelectedFamily parameters profile) .bit assignment
            (condition.get source) :=
        schemaBundles_get_decodes
          (SelectedFamily parameters profile) assignment
          condition inputColumns.toSchemaBundles source sourceDecoded
      have sourceConditionDecoded' :
          boolCodec.decode [assignment assertPlan.recipe.condition] =
            some (condition.get source) := by
        unfold ColumnBundle.Decodes at sourceConditionDecoded
        change
          boolCodec.decode
              ((inputColumns.toSchemaBundles.get condition).values
                assignment) =
            some (condition.get source) at sourceConditionDecoded
        rw [bundle_values_eq_ids_map,
          assertPlan.conditionIdsExact] at sourceConditionDecoded
        exact sourceConditionDecoded
      have conditionTrue : condition.get source = true :=
        boolCodec.decoded_value_unique sourceConditionDecoded' asserted
      refine ⟨source, ⟨conditionTrue, rfl⟩, ?_⟩
      change
        Columns.Decodes (SelectedFamily parameters profile)
          inputColumns assignment source
      exact sourceDecoded

/-- Honest semantic execution and canonical visible encodings induce the
exact occurrence-level witness consumed by compositional completion. -/
theorem honestActive
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
    (assignment : ColumnId -> Field)
    (source : Schema.Values (typeSystem parameters) input)
    (result : Schema.Values (typeSystem parameters) output)
    (sourceEncoded :
      Columns.Encodes (SelectedFamily parameters profile)
        inputColumns assignment source)
    (resultEncoded :
      Columns.Encodes (SelectedFamily parameters profile)
        plan.resultColumns assignment result)
    (semantic : primitive.Holds source result) :
    plan.occurrence.HonestActive assignment := by
  cases plan with
  | @invoke call operands path inputColumns one active invokePlan =>
      rcases semantic with ⟨callOutputs, evaluated, resultExact⟩
      subst result
      have operandsEncoded :
          invokePlan.frame.operands.Encodes
            (SelectedFamily parameters profile) assignment
            (operands.get source) := by
        rw [CallFrame.operands, invokePlan.contextExact]
        exact refBundles_fromSchema_encodes
          (SelectedFamily parameters profile) assignment
          operands inputColumns.toSchemaBundles source sourceEncoded
      have canonicalOutputsEncoded :
          Columns.Encodes (SelectedFamily parameters profile)
            (instructionColumns path
              ((SelectedSignature parameters).callOutputs call))
            assignment callOutputs := by
        apply columns_left_encodes_of_append
          (SelectedFamily parameters profile) assignment
          (instructionColumns path
            ((SelectedSignature parameters).callOutputs call))
          inputColumns callOutputs source
        change
          Columns.Encodes (SelectedFamily parameters profile)
            ((instructionColumns path
                ((SelectedSignature parameters).callOutputs call)).append
              inputColumns)
            assignment (callOutputs.append source) at resultEncoded
        exact resultEncoded
      have outputsEncoded :
          invokePlan.frame.outputs.Encodes
            (SelectedFamily parameters profile) assignment callOutputs := by
        rw [invokePlan.outputsExact]
        exact canonicalOutputsEncoded
      exact ArmOccurrence.HonestActive.call
        (operands.get source) callOutputs
        operandsEncoded outputsEncoded evaluated
  | @literal port value path inputColumns one active literalPlan =>
      subst result
      have canonicalLiteralEncoded :
          (HVec.head
              (instructionColumns path [port])).toColumnBundle.Encodes
            (SelectedFamily parameters profile) port.kind assignment
            value := by
        exact
          (columns_left_encodes_of_append
            (SelectedFamily parameters profile) assignment
            (instructionColumns path [port]) inputColumns
            (.cons value .nil) source
            (by
              change
                Columns.Encodes (SelectedFamily parameters profile)
                  ((instructionColumns path [port]).append inputColumns)
                  assignment (.cons value source) at resultEncoded
              exact resultEncoded)).1
      have recipeLiteralEncoded :
          literalPlan.recipe.output.Encodes
            (SelectedFamily parameters profile) port.kind assignment
            literalPlan.recipe.value := by
        rw [literalPlan.outputExact, literalPlan.valueExact]
        exact canonicalLiteralEncoded
      exact ArmOccurrence.HonestActive.literal
        (literalPlan.recipe.output.decodes_of_encodes
          (SelectedFamily parameters profile) port.kind assignment
          literalPlan.recipe.value recipeLiteralEncoded)
  | @assertTrue condition path inputColumns one active assertPlan =>
      rcases semantic with ⟨conditionTrue, resultExact⟩
      subst result
      change condition.get source = true at conditionTrue
      have conditionEncoded :
          (inputColumns.toSchemaBundles.get condition).Encodes
            (SelectedFamily parameters profile) .bit assignment true := by
        have sourceConditionEncoded :=
          schemaBundles_get_encodes
            (SelectedFamily parameters profile) assignment
            condition inputColumns.toSchemaBundles source sourceEncoded
        rw [conditionTrue] at sourceConditionEncoded
        exact sourceConditionEncoded
      have conditionDecoded :
          boolCodec.decode [assignment assertPlan.recipe.condition] =
            some true := by
        have decoded :=
          (inputColumns.toSchemaBundles.get condition).decodes_of_encodes
            (SelectedFamily parameters profile) .bit assignment true
            conditionEncoded
        unfold ColumnBundle.Decodes at decoded
        change
          boolCodec.decode
              ((inputColumns.toSchemaBundles.get condition).values
                assignment) = some true at decoded
        rw [bundle_values_eq_ids_map,
          assertPlan.conditionIdsExact] at decoded
        exact decoded
      exact ArmOccurrence.HonestActive.assertion conditionDecoded

/-- Honest semantic execution and canonical visible encodings can fill only
the occurrence's declared temporary coordinates and satisfy every emitted
active row. -/
theorem activeComplete
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
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1)
    (source : Schema.Values (typeSystem parameters) input)
    (result : Schema.Values (typeSystem parameters) output)
    (sourceEncoded :
      Columns.Encodes (SelectedFamily parameters profile)
        inputColumns assignment source)
    (resultEncoded :
      Columns.Encodes (SelectedFamily parameters profile)
        plan.resultColumns assignment result)
    (semantic : primitive.Holds source result) :
    ∃ completed : ColumnId -> Field,
      AgreesOn plan.occurrence.visibleIds assignment completed ∧
        ChangesOnly plan.occurrence.temporaryIds assignment completed ∧
        Satisfies plan.receipt.rows completed := by
  have honest :=
    plan.honestActive assignment source result
      sourceEncoded resultEncoded semantic
  have completed :=
    plan.occurrence.completeActive
      laws assignment constantOne activeOne honest
  simpa only [occurrence_rows] using completed

/-- The exact visible condition needed by an inactive occurrence.  Calls and
assertions need none; a verifier-owned literal remains pinned in both arms. -/
def InactiveVisible
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
    (assignment : ColumnId -> Field) : Prop :=
  match plan with
  | .invoke _ => True
  | .literal literalPlan =>
      literalPlan.recipe.output.Decodes
        (SelectedFamily parameters profile) _
        assignment literalPlan.recipe.value
  | .assertTrue _ => True

/-- The inactive visible contract induces the exact occurrence-level witness
used by branch completion. -/
theorem honestInactive
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
    (assignment : ColumnId -> Field)
    (visible : plan.InactiveVisible assignment) :
    plan.occurrence.HonestInactive assignment := by
  cases plan with
  | invoke invokePlan =>
      exact ArmOccurrence.HonestInactive.call
  | literal literalPlan =>
      exact ArmOccurrence.HonestInactive.literal visible
  | assertTrue assertPlan =>
      exact ArmOccurrence.HonestInactive.assertion

/-- An inactive selected-plan receipt is satisfiable without constraining call
semantics or assertion inputs; only static literal pins remain visible. -/
theorem inactiveComplete
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
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeZero : assignment active = 0)
    (visible : plan.InactiveVisible assignment) :
    ∃ completed : ColumnId -> Field,
      AgreesOn plan.occurrence.visibleIds assignment completed ∧
        ChangesOnly plan.occurrence.temporaryIds assignment completed ∧
        Satisfies plan.receipt.rows completed := by
  have honest := plan.honestInactive assignment visible
  have completed :=
    plan.occurrence.completeInactive
      assignment constantOne activeZero honest
  simpa only [occurrence_rows] using completed

end PrimitivePlan

namespace CompletionSeparation

/-- Local support plus ordered pairwise separation is sufficient for the
recursive occurrence-separation contract used by honest completion. -/
theorem separatedOccurrences_of_pairwise
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrences :
      List (ArmOccurrence signature family one active))
    (rowsSupported :
      ∀ occurrence, occurrence ∈ occurrences ->
        ∀ id, id ∈ rowsColumns occurrence.rows ->
          id ∈ occurrence.visibleIds ++ occurrence.temporaryIds)
    (localDisjoint :
      ∀ occurrence, occurrence ∈ occurrences ->
        IdsDisjoint occurrence.temporaryIds occurrence.visibleIds)
    (pairwise :
      occurrences.Pairwise fun first second =>
        IdsDisjoint first.temporaryIds second.visibleIds ∧
          IdsDisjoint second.temporaryIds first.visibleIds ∧
          IdsDisjoint first.temporaryIds second.temporaryIds) :
    ArmPlan.SeparatedOccurrences occurrences := by
  induction occurrences with
  | nil =>
      trivial
  | cons head tail inductionHypothesis =>
      have relations := (List.pairwise_cons.mp pairwise).1
      have tailPairwise := (List.pairwise_cons.mp pairwise).2
      refine ⟨
        rowsSupported head (by simp),
        localDisjoint head (by simp),
        ?_, ?_, ?_, ?_⟩
      · intro id temporaryMember visibleMember
        rcases List.mem_flatMap.mp visibleMember with
          ⟨occurrence, occurrenceMember, occurrenceVisible⟩
        exact (relations occurrence occurrenceMember).1
          id temporaryMember occurrenceVisible
      · intro id temporaryMember laterTemporary
        rcases List.mem_flatMap.mp laterTemporary with
          ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
        exact (relations occurrence occurrenceMember).2.2
          id temporaryMember occurrenceTemporary
      · intro id laterTemporary visibleMember
        rcases List.mem_flatMap.mp laterTemporary with
          ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
        exact (relations occurrence occurrenceMember).2.1
          id occurrenceTemporary visibleMember
      · apply inductionHypothesis
        · intro occurrence occurrenceMember
          exact rowsSupported occurrence (by simp [occurrenceMember])
        · intro occurrence occurrenceMember
          exact localDisjoint occurrence (by simp [occurrenceMember])
        · exact tailPairwise

end CompletionSeparation

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
