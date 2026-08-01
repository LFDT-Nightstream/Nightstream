import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan
import Nightstream.Implementation.Lowering.Goldilocks.ColumnPlan.Uniqueness

/-!
Contract: canonical local physical plans for fixed-one primitive occurrences.

Owns:
- canonical output and temporary allocations at one structural path;
- construction of a call frame from those allocations and one certified
  closed-call recipe;
- derivation of allocation uniqueness, owner separation, and width agreement
  from the column plan and semantic codec profile.

Does not own: Step/Terminal traversal, branch receipts, source-owner order,
whole-program scoping, Rust artifacts, or call-recipe internals.

The only occurrence-specific separation premise is that the exact earlier
context excludes the new instruction owner.  No row, allocation, or cost list
is caller supplied.

Emits constraints: exactly the receipt of the selected `CallRecipe`.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

universe u

namespace CanonicalPrimitivePlan

/-- A typed context excludes one physical owner when none of its exact
canonical coordinate identities uses that owner. -/
def ContextExcludesOwner
    {types : TypeSystem.{u}}
    (owner : PhysicalOwner) :
    {schema : Schema types} -> Columns schema -> Prop
  | [], .nil => True
  | _ :: _, .cons head tail =>
      (∀ coordinate, (head.column coordinate).owner ≠ owner) ∧
        ContextExcludesOwner owner tail

private theorem schemaIds_eq
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (columns : Columns schema) :
    columns.toSchemaBundles.ids =
      ColumnPlan.schemaColumnIds columns := by
  unfold SchemaBundles.ids ColumnPlan.schemaColumnIds
  rw [Columns.toSchemaBundles_columns]

private theorem layoutIds_eq
    {types : TypeSystem.{u}}
    {layouts : List Layout}
    (columns :
      Columns (layouts.map fun layout =>
        { kind := (TypeSystem.Kind.field : types.Kind)
          layout := layout })) :
    columns.toLayoutBundles.ids =
      ColumnPlan.schemaColumnIds columns := by
  unfold LayoutBundles.ids ColumnPlan.schemaColumnIds
  rw [Columns.toLayoutBundles_columns]

namespace ContextExcludesOwner

theorem id_excludes
    {types : TypeSystem.{u}}
    {schema : Schema types}
    {columns : Columns schema}
    {owner : PhysicalOwner}
    (excludes : ContextExcludesOwner owner columns) :
    ∀ id, id ∈ columns.toSchemaBundles.ids -> id.owner ≠ owner := by
  induction columns with
  | nil =>
      intro id member
      simp [Columns.toSchemaBundles, SchemaBundles.ids,
        SchemaBundles.columns, SchemaBundles.portColumns] at member
  | @cons port tail head rest inductionHypothesis =>
      intro id member
      rw [Columns.toSchemaBundles,
        SchemaBundles.ids_cons] at member
      rcases List.mem_append.mp member with headMember | tailMember
      · unfold ColumnBundle.ids at headMember
        rw [Bundle.toColumnBundle_columns,
          bundleOwnedColumns, List.map_ofFn,
          List.mem_ofFn] at headMember
        rcases headMember with ⟨coordinate, equal⟩
        subst id
        exact excludes.1 coordinate
      · exact inductionHypothesis excludes.2 id tailMember

theorem of_ids
    {types : TypeSystem.{u}}
    {schema : Schema types}
    {columns : Columns schema}
    {owner : PhysicalOwner}
    (excluded :
      ∀ id, id ∈ columns.toSchemaBundles.ids -> id.owner ≠ owner) :
    ContextExcludesOwner owner columns := by
  induction columns with
  | nil =>
      trivial
  | @cons port tail head rest inductionHypothesis =>
      constructor
      · intro coordinate
        apply excluded (head.column coordinate)
        rw [Columns.toSchemaBundles, SchemaBundles.ids_cons,
          List.mem_append]
        left
        unfold ColumnBundle.ids
        rw [Bundle.toColumnBundle_columns,
          bundleOwnedColumns, List.map_ofFn, List.mem_ofFn]
        exact ⟨coordinate, rfl⟩
      · apply inductionHypothesis
        intro id member
        apply excluded id
        rw [Columns.toSchemaBundles, SchemaBundles.ids_cons,
          List.mem_append]
        exact Or.inr member

theorem append
    {types : TypeSystem.{u}}
    {left right : Schema types}
    {leftColumns : Columns left}
    {rightColumns : Columns right}
    {owner : PhysicalOwner}
    (leftExcludes : ContextExcludesOwner owner leftColumns)
    (rightExcludes : ContextExcludesOwner owner rightColumns) :
    ContextExcludesOwner owner (leftColumns.append rightColumns) := by
  induction leftColumns with
  | nil =>
      exact rightExcludes
  | cons head rest inductionHypothesis =>
      exact ⟨leftExcludes.1,
        inductionHypothesis leftExcludes.2⟩

theorem input
    {types : TypeSystem.{u}}
    (schema : Schema types)
    (path : OwnerPath) :
    ContextExcludesOwner (.typed (.instruction path))
      (inputColumns schema) := by
  apply of_ids
  intro id member equal
  have canonicalMember :
      id ∈
        ColumnPlan.schemaColumnIds (inputColumns schema) := by
    rw [← schemaIds_eq]
    exact member
  have ownerExact :=
    (ColumnPlan.mem_allocateSchemaFrom
      (fun slot => PhysicalOwner.typed (.input slot))
      0 schema id canonicalMember).1
  rw [equal] at ownerExact
  exact Owner.noConfusion (PhysicalOwner.typed.inj ownerExact)

theorem instruction
    {types : TypeSystem.{u}}
    (source target : OwnerPath)
    (schema : Schema types)
    (different : source ≠ target) :
    ContextExcludesOwner (.typed (.instruction target))
      (instructionColumns source schema) := by
  apply of_ids
  intro id member equal
  have canonicalMember :
      id ∈
        ColumnPlan.schemaColumnIds
          (instructionColumns source schema) := by
    rw [← schemaIds_eq]
    exact member
  have ownerExact :=
    (ColumnPlan.mem_allocateSchemaFrom
      (fun _ => PhysicalOwner.typed (.instruction source))
      0 schema id canonicalMember).1
  rw [equal] at ownerExact
  exact different
    (Owner.instruction.inj
      (PhysicalOwner.typed.inj ownerExact.symm))

theorem branch
    {types : TypeSystem.{u}}
    (branchPath target : OwnerPath)
    (schema : Schema types) :
    ContextExcludesOwner (.typed (.instruction target))
      (branchJoinColumns branchPath schema) := by
  apply of_ids
  intro id member equal
  have canonicalMember :
      id ∈
        ColumnPlan.schemaColumnIds
          (branchJoinColumns branchPath schema) := by
    rw [← schemaIds_eq]
    exact member
  have ownerExact :=
    (ColumnPlan.mem_allocateSchemaFrom
      (fun _ => PhysicalOwner.typed (.branch branchPath))
      0 schema id canonicalMember).1
  rw [equal] at ownerExact
  exact Owner.noConfusion (PhysicalOwner.typed.inj ownerExact)

end ContextExcludesOwner

/-- Every coordinate of a canonical instruction output has the exact
instruction owner used to allocate it. -/
theorem instruction_id_owner
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (schema : Schema types)
    (id : ColumnId)
    (member :
      id ∈ (instructionColumns path schema).toSchemaBundles.ids) :
    id.owner = .typed (.instruction path) := by
  have canonicalMember :
      id ∈
        ColumnPlan.schemaColumnIds
          (instructionColumns path schema) := by
    rw [← schemaIds_eq]
    exact member
  exact
    (ColumnPlan.mem_allocateSchemaFrom
      (fun _ => PhysicalOwner.typed (.instruction path))
      0 schema id canonicalMember).1

/-- Instruction outputs are disjoint from any exact typed context that
excludes that instruction owner. -/
theorem ContextExcludesOwner.instructionOutputsDisjoint
    {types : TypeSystem.{u}}
    {context output : Schema types}
    (path : OwnerPath)
    (contextColumns : Columns context)
    (excludes :
      ContextExcludesOwner (.typed (.instruction path))
        contextColumns) :
    IdsDisjoint
      (instructionColumns path output).toSchemaBundles.ids
      contextColumns.toSchemaBundles.ids := by
  intro id outputMember contextMember
  exact
    (ContextExcludesOwner.id_excludes excludes id contextMember)
      (instruction_id_owner path output id outputMember)

/-- Canonical instruction outputs cannot alias the verifier's public
constant-one coordinate. -/
theorem instructionOutputsDisjointOne
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (output : Schema types) :
    IdsDisjoint
      (instructionColumns path output).toSchemaBundles.ids
      [oneColumn] := by
  rw [schemaIds_eq]
  intro id outputMember oneMember
  exact
    (ColumnPlan.prelude_typed_ids_disjoint
      (fun _ => .instruction path) 0 output)
      id oneMember outputMember

/-- Canonical instruction outputs cannot alias either activation coordinate
of any structural branch. -/
theorem instructionOutputsDisjointActivations
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (output : Schema types)
    (branchPath : OwnerPath) :
    IdsDisjoint
      (instructionColumns path output).toSchemaBundles.ids
      ((activationColumns branchPath).map fun column => column.id) := by
  rw [schemaIds_eq]
  exact ColumnPlan.typed_activation_ids_disjoint
    (fun _ => .instruction path) 0 output branchPath

/-- Canonical typed input coordinates cannot alias the verifier's public
constant-one coordinate. -/
theorem inputDisjointOne
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    IdsDisjoint (inputColumns schema).toSchemaBundles.ids [oneColumn] := by
  rw [schemaIds_eq]
  intro id inputMember oneMember
  exact
    (ColumnPlan.prelude_typed_ids_disjoint
      (fun slot => .input slot) 0 schema)
      id oneMember inputMember

/-- Canonical typed input coordinates cannot alias either activation
coordinate of any structural branch. -/
theorem inputDisjointActivations
    {types : TypeSystem.{u}}
    (schema : Schema types)
    (branchPath : OwnerPath) :
    IdsDisjoint
      (inputColumns schema).toSchemaBundles.ids
      ((activationColumns branchPath).map fun column => column.id) := by
  rw [schemaIds_eq]
  exact ColumnPlan.typed_activation_ids_disjoint
    (fun slot => .input slot) 0 schema branchPath

/-- Every canonical temporary coordinate has the exact instruction owner
used to allocate it. -/
theorem temporary_id_owner
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (outputSchema : Schema types)
    (layouts : List Layout)
    (id : ColumnId)
    (member :
      id ∈
        (temporaryColumns path outputSchema layouts).toLayoutBundles.ids) :
    id.owner = .typed (.instruction path) := by
  have canonicalMember :
      id ∈
        ColumnPlan.schemaColumnIds
          (temporaryColumns path outputSchema layouts) := by
    rw [← layoutIds_eq]
    exact member
  exact
    (ColumnPlan.mem_allocateSchemaFrom
      (fun _ => PhysicalOwner.typed (.instruction path))
      outputSchema.length
      (layouts.map fun layout =>
        { kind := (TypeSystem.Kind.field : types.Kind)
          layout := layout })
      id canonicalMember).1

private theorem allocations_nodup
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (outputSchema : Schema types)
    (layouts : List Layout) :
    ((instructionColumns path outputSchema).toSchemaBundles.ids ++
      (temporaryColumns path outputSchema layouts).toLayoutBundles.ids
    ).Nodup := by
  rw [schemaIds_eq, layoutIds_eq]
  exact
    ColumnPlan.instructionOutputs_append_temporaryColumns_ids_nodup
      path outputSchema layouts

private theorem reverse_disjoint_of_nodup_append
    {left right : List ColumnId}
    (nodup : (left ++ right).Nodup) :
    IdsDisjoint right left := by
  rw [List.nodup_append] at nodup
  intro id rightMember leftMember
  exact nodup.2.2 id leftMember id rightMember rfl

private theorem controls_and_context_exclude
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (columns : Columns schema)
    (one active : ColumnId)
    (owner : PhysicalOwner)
    (oneExcludes : one.owner ≠ owner)
    (activeExcludes : active.owner ≠ owner)
    (contextExcludes : ContextExcludesOwner owner columns) :
    ∀ id,
      id ∈ [one, active] ++ columns.toSchemaBundles.ids ->
        id.owner ≠ owner := by
  intro id member
  rcases List.mem_append.mp member with controlMember | contextMember
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at controlMember
    rcases controlMember with equal | equal
    · subst id
      exact oneExcludes
    · subst id
      exact activeExcludes
  · exact ContextExcludesOwner.id_excludes
      contextExcludes id contextMember

private theorem temporaries_disjoint_visible
    {types : TypeSystem.{u}}
    {context output : Schema types}
    (path : OwnerPath)
    (layouts : List Layout)
    (contextColumns : Columns context)
    (one active : ColumnId)
    (oneExcludes :
      one.owner ≠ .typed (.instruction path))
    (activeExcludes :
      active.owner ≠ .typed (.instruction path))
    (contextExcludes :
      ContextExcludesOwner (.typed (.instruction path))
        contextColumns) :
    IdsDisjoint
      (temporaryColumns path output layouts).toLayoutBundles.ids
      ([one, active] ++
        contextColumns.toSchemaBundles.ids ++
        (instructionColumns path output).toSchemaBundles.ids) := by
  intro id temporaryMember visibleMember
  rcases List.mem_append.mp visibleMember with
    controlOrContext | outputMember
  rcases List.mem_append.mp controlOrContext with
    controlMember | contextMember
  · have controlsExclude :=
      controls_and_context_exclude
        contextColumns one active (.typed (.instruction path))
        oneExcludes activeExcludes contextExcludes
    have ownerExact :=
      temporary_id_owner path output layouts id temporaryMember
    exact controlsExclude id
      (List.mem_append_left _ controlMember) ownerExact
  · have ownerExact :=
      temporary_id_owner path output layouts id temporaryMember
    exact
      (ContextExcludesOwner.id_excludes
        contextExcludes id contextMember) ownerExact
  · exact
      (reverse_disjoint_of_nodup_append
        (allocations_nodup path output layouts))
        id temporaryMember outputMember

private theorem outputs_disjoint_preexisting
    {types : TypeSystem.{u}}
    {context output : Schema types}
    (path : OwnerPath)
    (contextColumns : Columns context)
    (one active : ColumnId)
    (oneExcludes :
      one.owner ≠ .typed (.instruction path))
    (activeExcludes :
      active.owner ≠ .typed (.instruction path))
    (contextExcludes :
      ContextExcludesOwner (.typed (.instruction path))
        contextColumns) :
    IdsDisjoint
      (instructionColumns path output).toSchemaBundles.ids
      ([one, active] ++ contextColumns.toSchemaBundles.ids) := by
  intro id outputMember preexistingMember
  have outputOwner :=
    instruction_id_owner path output id outputMember
  exact
    (controls_and_context_exclude
      contextColumns one active (.typed (.instruction path))
      oneExcludes activeExcludes contextExcludes)
      id preexistingMember outputOwner

private theorem allocations_owned
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (output : Schema types)
    (layouts : List Layout) :
    ∀ column,
      column ∈
          (instructionColumns path output).toSchemaBundles.columns ++
            (temporaryColumns path output layouts).toLayoutBundles.columns ->
        column.id.owner = .typed (.instruction path) := by
  intro column member
  rcases List.mem_append.mp member with outputMember | temporaryMember
  · apply instruction_id_owner path output column.id
    exact List.mem_map.mpr
      ⟨column, outputMember, rfl⟩
  · apply temporary_id_owner path output layouts column.id
    exact List.mem_map.mpr
      ⟨column, temporaryMember, rfl⟩

private theorem instruction_head_ids_nodup
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (port : Port types) :
    (HVec.head (instructionColumns path [port])).toColumnBundle.ids.Nodup := by
  have all :=
    ColumnPlan.allocateSchemaFrom_ids_nodup
      (fun _ => .typed (.instruction path)) 0 [port]
  simpa [instructionColumns, allocateSchema, allocateSchemaFrom,
    ColumnPlan.schemaColumnIds, schemaOwnedColumns,
    ColumnBundle.ids, Bundle.toColumnBundle_columns] using all

/-- Canonical literal recipe: one instruction-owned output bundle and one pin
row per codec coordinate, starting at occurrence-local ordinal zero. -/
def literalRecipe
    {parameters : Parameters}
    (profile : Profile parameters)
    (port : Port (typeSystem parameters))
    (value : (typeSystem parameters).Value port.kind)
    (path : OwnerPath)
    (one : ColumnId)
    (widthAgrees :
      PortWidthAgrees (profile.family parameters) port) :
    LiteralPinRecipe
      ((profile.family parameters).codecFor port.kind)
      port.layout where
  owner := .typed (.instruction path)
  firstOrdinal := 0
  one := one
  output :=
    (HVec.head (instructionColumns path [port])).toColumnBundle
  value := value
  widthAgrees := widthAgrees

/-- Canonical local plan for one verifier-static literal.  Codec
admissibility is the only semantic premise; rows, allocations, owners, and
their identities are derived from the typed port and structural path. -/
def literal
    {parameters : Parameters}
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (port : Port (typeSystem parameters))
    (value : (typeSystem parameters).Value port.kind)
    (path : OwnerPath)
    (contextColumns : Columns context)
    (one active : ColumnId)
    (widthAgrees :
      PortWidthAgrees (profile.family parameters) port)
    (admissible :
      ((profile.family parameters).codecFor port.kind).Admissible value) :
    LiteralPlan parameters profile port value path
      contextColumns one active where
  recipe := literalRecipe profile port value path one widthAgrees
  ownerExact := rfl
  ordinalExact := rfl
  oneExact := rfl
  outputExact := rfl
  valueExact := rfl
  admissible := admissible
  rowsOwned := LiteralPinRecipe.rows_owned _
  rowsSupported := LiteralPinRecipe.rows_supported _
  columnIdsNodup := instruction_head_ids_nodup path port
  rowIdsNodup := LiteralPinRecipe.row_ids_nodup _

private theorem toSchemaBundles_get_eq_refBundle
    {types : TypeSystem.{u}}
    {schema : Schema types}
    {kind : types.Kind}
    (reference : Ref types schema kind)
    (columns : Columns schema) :
    columns.toSchemaBundles.get reference =
      (refBundle reference columns).toColumnBundle := by
  induction reference with
  | here =>
      cases columns
      rfl
  | there reference inductionHypothesis =>
      cases columns with
      | cons head tail =>
          exact inductionHypothesis tail

private theorem bit_ref_layout_width_one
    {parameters : Parameters}
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (condition : Ref (typeSystem parameters) context .bit)
    (contextWidths :
      SchemaWidthAgrees (profile.family parameters) context) :
    condition.port.layout.owners.length = 1 := by
  have width :=
    contextWidths condition.port (ref_port_mem condition)
  unfold PortWidthAgrees at width
  rw [condition.port_sort] at width
  calc
    condition.port.layout.owners.length =
        ((profile.family parameters).codecFor .bit).width :=
      width.symm
    _ = 1 := by
      rfl

/-- The sole coordinate of a typed Boolean reference.  Its existence follows
from the canonical Boolean codec width, not from a numeric artifact column. -/
def bitCoordinate
    {parameters : Parameters}
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (condition : Ref (typeSystem parameters) context .bit)
    (columns : Columns context)
    (contextWidths :
      SchemaWidthAgrees (profile.family parameters) context) :
    ColumnId :=
  (refBundle condition columns).column
    ⟨0, by
      rw [bit_ref_layout_width_one profile condition contextWidths]
      omega⟩

private theorem list_ofFn_eq_singleton_of_length_one
    {α : Type u}
    {length : Nat}
    (function : Fin length -> α)
    (lengthOne : length = 1) :
    List.ofFn function =
      [function ⟨0, by omega⟩] := by
  subst length
  rfl

/-- The selected Boolean reference has exactly the canonical one-coordinate
bundle used by assertion and branch-control semantics. -/
theorem bitReferenceIdsExact
    {parameters : Parameters}
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (condition : Ref (typeSystem parameters) context .bit)
    (columns : Columns context)
    (contextWidths :
      SchemaWidthAgrees (profile.family parameters) context) :
    (columns.toSchemaBundles.get condition).ids =
      [bitCoordinate profile condition columns contextWidths] := by
  rw [toSchemaBundles_get_eq_refBundle condition columns]
  unfold bitCoordinate
  rw [ColumnBundle.ids, Bundle.toColumnBundle_columns,
    bundleOwnedColumns, List.map_ofFn]
  simpa only [Function.comp_apply] using
    list_ofFn_eq_singleton_of_length_one
      (refBundle condition columns).column
      (bit_ref_layout_width_one profile condition contextWidths)

/-- The selected Boolean coordinate is one of the exact coordinates of its
typed context. -/
theorem bitCoordinate_mem
    {parameters : Parameters}
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (condition : Ref (typeSystem parameters) context .bit)
    (columns : Columns context)
    (contextWidths :
      SchemaWidthAgrees (profile.family parameters) context) :
    bitCoordinate profile condition columns contextWidths ∈
      columns.toSchemaBundles.ids := by
  apply SchemaBundles.get_ids_subset condition columns.toSchemaBundles
  rw [bitReferenceIdsExact profile condition columns contextWidths]
  simp

/-- Canonical one-row active Boolean assertion over the sole coordinate of
the exact typed condition reference. -/
def assertion
    {parameters : Parameters}
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (condition : Ref (typeSystem parameters) context .bit)
    (path : OwnerPath)
    (contextColumns : Columns context)
    (one active : ColumnId)
    (contextWidths :
      SchemaWidthAgrees (profile.family parameters) context) :
    AssertPlan parameters profile condition path
      contextColumns one active where
  recipe :=
    { owner := .typed (.instruction path)
      ordinal := 0
      one := one
      active := active
      condition :=
        bitCoordinate profile condition contextColumns contextWidths }
  ownerExact := rfl
  ordinalExact := rfl
  oneExact := rfl
  activeExact := rfl
  conditionIdsExact :=
    bitReferenceIdsExact profile condition contextColumns contextWidths

/-- Canonical call frame at one fixed-one instruction path. -/
def callFrame
    {parameters : Parameters}
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (call : (SelectedSignature parameters).Call)
    (operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call))
    (path : OwnerPath)
    (contextColumns : Columns context)
    (one active : ColumnId)
    (contextWidths :
      SchemaWidthAgrees (profile.family parameters) context)
    (oneExcludes :
      one.owner ≠ .typed (.instruction path))
    (activeExcludes :
      active.owner ≠ .typed (.instruction path))
    (contextExcludes :
      ContextExcludesOwner (.typed (.instruction path))
        contextColumns) :
    CallFrame (profile.family parameters) call operands :=
  callFrameOfColumns
    (.typed (.instruction path))
    one active contextColumns
    (instructionColumns path
      ((SelectedSignature parameters).callOutputs call))
    (temporaryColumns path
      ((SelectedSignature parameters).callOutputs call)
      ((SelectedSignature parameters).callFootprint call).temporaries)
    (refBundles_widthsAgree_of_schema contextWidths operands)
    (profile.callOutputs_widthsAgree parameters call)
    (allocations_nodup path
      ((SelectedSignature parameters).callOutputs call)
      ((SelectedSignature parameters).callFootprint call).temporaries)
    (temporaries_disjoint_visible path
      ((SelectedSignature parameters).callFootprint call).temporaries
      contextColumns one active oneExcludes activeExcludes contextExcludes)
    (outputs_disjoint_preexisting path contextColumns one active
      oneExcludes activeExcludes contextExcludes)
    (allocations_owned path
      ((SelectedSignature parameters).callOutputs call)
      ((SelectedSignature parameters).callFootprint call).temporaries)

/-- Canonical local plan for one invoked closed call.  The caller selects only
the certified recipe set, typed operands, structural path, and already
allocated context. -/
def invoke
    {parameters : Parameters}
    (profile : Profile parameters)
    (recipes :
      CallRecipes (SelectedSignature parameters)
        (profile.family parameters))
    {context : Schema (typeSystem parameters)}
    (call : (SelectedSignature parameters).Call)
    (operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call))
    (path : OwnerPath)
    (contextColumns : Columns context)
    (one active : ColumnId)
    (contextWidths :
      SchemaWidthAgrees (profile.family parameters) context)
    (oneExcludes :
      one.owner ≠ .typed (.instruction path))
    (activeExcludes :
      active.owner ≠ .typed (.instruction path))
    (contextExcludes :
      ContextExcludesOwner (.typed (.instruction path))
        contextColumns) :
    InvokePlan parameters profile call operands path
      contextColumns one active where
  recipe := recipes.recipe call
  frame :=
    callFrame profile call operands path contextColumns one active
      contextWidths oneExcludes activeExcludes contextExcludes
  ownerExact := rfl
  oneExact := rfl
  activeExact := rfl
  contextExact := rfl
  outputsExact := rfl
  temporariesExact := rfl

end CanonicalPrimitivePlan

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
