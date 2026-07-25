import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.HonestAssignment
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitiveRefinement

/-!
Contract: construct the visible SSA result of one canonical call or literal
without assigning its temporary witnesses.

Owns:
- writing exact canonical output coordinates for calls and literals;
- preservation of the complete earlier typed context and shared controls;
- composition into the primitive's exact result-column context.

Does not own: temporary completion, branch selection, whole-program
assignment construction, production codecs, or generated artifacts.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

universe u

namespace HonestAssignment

/-- A write to a disjoint canonical schema preserves an earlier schema
encoding exactly. -/
theorem encodes_preserved_by_encodeInto
    {types : TypeSystem.{u}}
    (family : Family types)
    {oldSchema newSchema : Schema types}
    (oldColumns : Columns oldSchema)
    (newColumns : Columns newSchema)
    (oldValues : Schema.Values types oldSchema)
    (newValues : Schema.Values types newSchema)
    (assignment : ColumnId -> Field)
    (disjoint :
      IdsDisjoint newColumns.toSchemaBundles.ids
        oldColumns.toSchemaBundles.ids)
    (oldEncoded :
      Columns.Encodes family oldColumns assignment oldValues) :
    Columns.Encodes family oldColumns
      (encodeInto family newColumns newValues assignment) oldValues := by
  apply oldColumns.toSchemaBundles.encodes_of_agrees
    family assignment
      (encodeInto family newColumns newValues assignment)
      oldValues
  · exact agreesOn_of_changesOnly disjoint
      (encodeInto_changesOnly
        family newColumns newValues assignment)
  · exact oldEncoded

/-- A disjoint schema write preserves any explicit coordinate list. -/
theorem agreesOn_encodeInto
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (values : Schema.Values types schema)
    (assignment : ColumnId -> Field)
    (protectedIds : List ColumnId)
    (disjoint :
      IdsDisjoint columns.toSchemaBundles.ids protectedIds) :
    AgreesOn protectedIds assignment
      (encodeInto family columns values assignment) :=
  agreesOn_of_changesOnly disjoint
    (encodeInto_changesOnly family columns values assignment)

end HonestAssignment

namespace InvokePlan

/-- The canonical verifier-one control is visible to the call occurrence. -/
theorem occurrenceOneMemVisible
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active) :
    one ∈ (PrimitivePlan.invoke plan).occurrence.visibleIds := by
  simp [PrimitivePlan.occurrence, ArmOccurrence.visibleIds,
    CallFrame.visibleIds, plan.oneExact]

/-- The branch-activation control is visible to the call occurrence. -/
theorem occurrenceActiveMemVisible
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active) :
    active ∈ (PrimitivePlan.invoke plan).occurrence.visibleIds := by
  simp [PrimitivePlan.occurrence, ArmOccurrence.visibleIds,
    CallFrame.visibleIds, plan.activeExact]

/-- Every coordinate of the complete typed call input is occurrence-visible. -/
theorem occurrenceInputIdsSubsetVisible
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active) :
    ∀ id, id ∈ inputColumns.toSchemaBundles.ids ->
      id ∈ (PrimitivePlan.invoke plan).occurrence.visibleIds := by
  intro id member
  change id ∈
    [plan.frame.one, plan.frame.active] ++
      plan.frame.contextBundles.ids ++ plan.frame.outputs.ids
  apply List.mem_append_left plan.frame.outputs.ids
  apply List.mem_append_right [plan.frame.one, plan.frame.active]
  rw [plan.contextExact]
  exact member

/-- Every coordinate of the complete typed call result is occurrence-visible.
This includes both the newly allocated output prefix and the retained input
suffix. -/
theorem occurrenceResultIdsSubsetVisible
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active) :
    ∀ id,
      id ∈
          (PrimitivePlan.invoke plan).resultColumns.toSchemaBundles.ids ->
        id ∈ (PrimitivePlan.invoke plan).occurrence.visibleIds := by
  intro id member
  change id ∈
    (Columns.toSchemaBundles
      (HVec.append
        (instructionColumns path
          ((SelectedSignature parameters).callOutputs call))
        inputColumns)).ids at member
  rw [Columns.append_ids] at member
  rcases List.mem_append.mp member with outputMember | inputMember
  · change id ∈
      [plan.frame.one, plan.frame.active] ++
        plan.frame.contextBundles.ids ++ plan.frame.outputs.ids
    apply List.mem_append_right
    rw [plan.outputsExact]
    exact outputMember
  · exact plan.occurrenceInputIdsSubsetVisible id inputMember

/-- Canonical call outputs are disjoint from the complete earlier typed
context. -/
theorem outputIdsDisjointInput
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active) :
    IdsDisjoint
      (instructionColumns path
        ((SelectedSignature parameters).callOutputs call)
        ).toSchemaBundles.ids
      inputColumns.toSchemaBundles.ids := by
  intro id outputMember inputMember
  apply plan.frame.outputsDisjointPreexisting id
  · rw [plan.outputsExact]
    exact outputMember
  · apply List.mem_append_right [plan.frame.one, plan.frame.active]
    rw [plan.contextExact]
    exact inputMember

/-- Canonical call outputs are disjoint from the verifier one coordinate and
the enclosing activation coordinate. -/
theorem outputIdsDisjointControls
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active) :
    IdsDisjoint
      (instructionColumns path
        ((SelectedSignature parameters).callOutputs call)
        ).toSchemaBundles.ids
      [one, active] := by
  intro id outputMember controlMember
  apply plan.frame.outputsDisjointPreexisting id
  · rw [plan.outputsExact]
    exact outputMember
  · rw [plan.oneExact, plan.activeExact]
    exact List.mem_append_left _ controlMember

/-- Writing one honest call output extends the encoded SSA context and
preserves both controls.  Temporary witnesses remain untouched. -/
theorem extendEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active)
    (assignment : ColumnId -> Field)
    (source :
      Schema.Values (typeSystem parameters) context)
    (outputs :
      Schema.Values (typeSystem parameters)
        ((SelectedSignature parameters).callOutputs call))
    (sourceEncoded :
      Columns.Encodes (SelectedFamily parameters profile)
        inputColumns assignment source)
    (outputsAdmissible :
      HonestAssignment.Admissible
        (SelectedFamily parameters profile) outputs) :
    ∃ completed : ColumnId -> Field,
      Columns.Encodes (SelectedFamily parameters profile)
        ((instructionColumns path
          ((SelectedSignature parameters).callOutputs call)).append
            inputColumns)
        completed (outputs.append source) ∧
      ChangesOnly
        (instructionColumns path
          ((SelectedSignature parameters).callOutputs call)
          ).toSchemaBundles.ids
        assignment completed ∧
      AgreesOn [one, active] assignment completed := by
  let outputColumns :=
    instructionColumns path
      ((SelectedSignature parameters).callOutputs call)
  let completed :=
    HonestAssignment.encodeInto
      (SelectedFamily parameters profile)
      outputColumns outputs assignment
  have outputNodup :
      outputColumns.toSchemaBundles.ids.Nodup := by
    have allocations := plan.frame.allocationsNodup
    rw [List.nodup_append] at allocations
    rw [plan.outputsExact] at allocations
    exact allocations.1
  have outputEncoded :
      Columns.Encodes (SelectedFamily parameters profile)
        outputColumns completed outputs := by
    exact HonestAssignment.encodeInto_encodes
      (SelectedFamily parameters profile)
      outputColumns outputs assignment
      plan.frame.outputWidthsAgree outputsAdmissible outputNodup
  have sourcePreserved :
      Columns.Encodes (SelectedFamily parameters profile)
        inputColumns completed source := by
    exact HonestAssignment.encodes_preserved_by_encodeInto
      (SelectedFamily parameters profile)
      inputColumns outputColumns source outputs assignment
      plan.outputIdsDisjointInput sourceEncoded
  exact ⟨
    completed,
    Columns.append_encodes
      (SelectedFamily parameters profile) completed
      outputColumns inputColumns outputs source
      outputEncoded sourcePreserved,
    HonestAssignment.encodeInto_changesOnly
      (SelectedFamily parameters profile)
      outputColumns outputs assignment,
    HonestAssignment.agreesOn_encodeInto
      (SelectedFamily parameters profile)
      outputColumns outputs assignment [one, active]
      plan.outputIdsDisjointControls⟩

end InvokePlan

namespace LiteralPlan

/-- Every verifier-owned literal output coordinate is visible to its
occurrence even though the literal does not consume its surrounding context. -/
theorem occurrenceOutputIdsSubsetVisible
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {port : Port (typeSystem parameters)}
    {value : (typeSystem parameters).Value port.kind}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      LiteralPlan parameters profile port value path
        inputColumns one active) :
    ∀ id, id ∈ plan.recipe.output.ids ->
      id ∈ (PrimitivePlan.literal plan).occurrence.visibleIds := by
  intro id member
  simp [PrimitivePlan.occurrence, ArmOccurrence.visibleIds, member]

/-- Writing a canonical literal output extends the encoded SSA context when
the exact earlier context excludes this literal's instruction owner. -/
theorem extendEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {port : Port (typeSystem parameters)}
    {value : (typeSystem parameters).Value port.kind}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      LiteralPlan parameters profile port value path
        inputColumns one active)
    (assignment : ColumnId -> Field)
    (source :
      Schema.Values (typeSystem parameters) context)
    (sourceEncoded :
      Columns.Encodes (SelectedFamily parameters profile)
        inputColumns assignment source)
    (oneExcludes :
      one.owner ≠ .typed (.instruction path))
    (activeExcludes :
      active.owner ≠ .typed (.instruction path))
    (inputExcludes :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction path)) inputColumns) :
    ∃ completed : ColumnId -> Field,
      Columns.Encodes (SelectedFamily parameters profile)
        ((instructionColumns path [port]).append inputColumns)
        completed (.cons value source) ∧
      ChangesOnly
        (instructionColumns path [port]).toSchemaBundles.ids
        assignment completed ∧
      AgreesOn [one, active] assignment completed := by
  let outputColumns := instructionColumns path [port]
  let outputValues :
      Schema.Values (typeSystem parameters) [port] :=
    .cons value .nil
  let completed :=
    HonestAssignment.encodeInto
      (SelectedFamily parameters profile)
      outputColumns outputValues assignment
  have outputNodup :
      outputColumns.toSchemaBundles.ids.Nodup := by
    have canonical :=
      ColumnPlan.allocateSchemaFrom_ids_nodup
        (fun _ => PhysicalOwner.typed (.instruction path))
        0 [port]
    simpa [outputColumns, ColumnPlan.schemaColumnIds,
      SchemaBundles.ids, Columns.toSchemaBundles_columns] using canonical
  have outputWidths :
      SchemaWidthAgrees
        (SelectedFamily parameters profile) [port] := by
    intro item member
    simp only [List.mem_singleton] at member
    subst item
    exact plan.recipe.widthAgrees
  have outputAdmissible :
      HonestAssignment.Admissible
        (SelectedFamily parameters profile) outputValues :=
    ⟨plan.admissible, trivial⟩
  have outputEncoded :
      Columns.Encodes (SelectedFamily parameters profile)
        outputColumns completed outputValues := by
    exact HonestAssignment.encodeInto_encodes
      (SelectedFamily parameters profile)
      outputColumns outputValues assignment
      outputWidths outputAdmissible outputNodup
  have outputIdsOwned :
      ∀ id, id ∈ outputColumns.toSchemaBundles.ids ->
        id.owner = .typed (.instruction path) := by
    intro id member
    have canonicalMember :
        id ∈ ColumnPlan.schemaColumnIds outputColumns := by
      have idsExact :
          outputColumns.toSchemaBundles.ids =
            ColumnPlan.schemaColumnIds outputColumns := by
        unfold SchemaBundles.ids ColumnPlan.schemaColumnIds
        rw [Columns.toSchemaBundles_columns]
      rw [← idsExact]
      exact member
    have exactOwner :=
      ColumnPlan.mem_allocateSchemaFrom
        (fun _ => PhysicalOwner.typed (.instruction path))
        0 [port] id canonicalMember
    exact exactOwner.1
  have outputInputDisjoint :
      IdsDisjoint outputColumns.toSchemaBundles.ids
        inputColumns.toSchemaBundles.ids := by
    intro id outputMember inputMember
    exact
      (CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
        inputExcludes id inputMember)
      (outputIdsOwned id outputMember)
  have outputControlDisjoint :
      IdsDisjoint outputColumns.toSchemaBundles.ids [one, active] := by
    intro id outputMember controlMember
    have ownerExact := outputIdsOwned id outputMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at controlMember
    rcases controlMember with equal | equal
    · subst id
      exact oneExcludes ownerExact
    · subst id
      exact activeExcludes ownerExact
  have sourcePreserved :
      Columns.Encodes (SelectedFamily parameters profile)
        inputColumns completed source := by
    exact HonestAssignment.encodes_preserved_by_encodeInto
      (SelectedFamily parameters profile)
      inputColumns outputColumns source outputValues assignment
      outputInputDisjoint sourceEncoded
  exact ⟨
    completed,
    Columns.append_encodes
      (SelectedFamily parameters profile) completed
      outputColumns inputColumns outputValues source
      outputEncoded sourcePreserved,
    HonestAssignment.encodeInto_changesOnly
      (SelectedFamily parameters profile)
      outputColumns outputValues assignment,
    HonestAssignment.agreesOn_encodeInto
      (SelectedFamily parameters profile)
      outputColumns outputValues assignment [one, active]
      outputControlDisjoint⟩

end LiteralPlan

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
