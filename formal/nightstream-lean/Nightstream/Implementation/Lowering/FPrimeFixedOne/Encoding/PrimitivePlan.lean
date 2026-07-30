import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Profile
import Nightstream.Implementation.Lowering.Goldilocks.BundleBridge
import Nightstream.Implementation.Lowering.Goldilocks.InstructionReceipts

/-!
Contract: one source-indexed physical occurrence for each primitive admitted
by the fixed-one verifier.

Owns:
- exact call, literal, and Boolean-assertion recipes at one structural path;
- deterministic produced/context columns and exact instruction receipts;
- typed active soundness, honest active completion, inactive satisfiability;
- receipt-local row support and physical identity uniqueness.

Does not own: linear/product lowering, block traversal, branch receipts,
generated artifacts, Rust behavior, or an arbitrary semantic proposition.
Linear and product primitives are deliberately uninhabited in this plan.

Emits constraints: exactly the single instruction receipt selected by the
indexed constructor.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

universe u

abbrev SelectedSignature (parameters : Parameters) :=
  Vocabulary.signature parameters

abbrev SelectedFamily
    (parameters : Parameters)
    (profile : Profile parameters) :=
  profile.family parameters

def Columns.Decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (assignment : ColumnId -> Field)
    (values : Schema.Values types schema) : Prop :=
  columns.toSchemaBundles.Decodes family assignment values

def Columns.Encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    {schema : Schema types}
    (columns : Columns schema)
    (assignment : ColumnId -> Field)
    (values : Schema.Values types schema) : Prop :=
  columns.toSchemaBundles.Encodes family assignment values

/-- Physical schema identities compose in the same exact order as typed
column contexts. -/
theorem Columns.append_ids
    {types : TypeSystem.{u}}
    {left right : Schema types}
    (leftColumns : Columns left)
    (rightColumns : Columns right) :
    (Columns.toSchemaBundles
      (HVec.append leftColumns rightColumns)).ids =
      leftColumns.toSchemaBundles.ids ++
        rightColumns.toSchemaBundles.ids := by
  unfold SchemaBundles.ids
  rw [Columns.toSchemaBundles_columns
      (HVec.append leftColumns rightColumns),
    Columns.toSchemaBundles_columns leftColumns,
    Columns.toSchemaBundles_columns rightColumns]
  induction leftColumns with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp [HVec.append, schemaOwnedColumns,
        List.map_append, inductionHypothesis, List.append_assoc]

/-- Context decoding projects to every exact typed reference bundle. -/
theorem SchemaBundles.get_decodes
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

/-- Context encoding projects to every exact typed reference bundle. -/
theorem SchemaBundles.get_encodes
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

private theorem RefBundles.fromSchema_decodes
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
  | nil => trivial
  | cons reference tail inductionHypothesis =>
      exact ⟨
        SchemaBundles.get_decodes
          family assignment reference bundles values decoded,
        inductionHypothesis⟩

private theorem RefBundles.fromSchema_encodes
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
  | nil => trivial
  | cons reference tail inductionHypothesis =>
      exact ⟨
        SchemaBundles.get_encodes
          family assignment reference bundles values encoded,
        inductionHypothesis⟩

/-- Decoding composes across the exact ordered schema append. -/
theorem Columns.append_decodes
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

/-- Honest encoding composes across the exact ordered schema append. -/
theorem Columns.append_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {left right : Schema types}
    (leftColumns : Columns left)
    (rightColumns : Columns right)
    (leftValues : Schema.Values types left)
    (rightValues : Schema.Values types right)
    (leftEncoded :
      Columns.Encodes family leftColumns assignment leftValues)
    (rightEncoded :
      Columns.Encodes family rightColumns assignment rightValues) :
    Columns.Encodes family (leftColumns.append rightColumns) assignment
      (leftValues.append rightValues) := by
  induction leftColumns with
  | nil =>
      cases leftValues
      exact rightEncoded
  | cons head tail inductionHypothesis =>
      cases leftValues with
      | cons value values =>
          exact ⟨leftEncoded.1,
            inductionHypothesis values leftEncoded.2⟩

private theorem Bundle.cast_values
    {types : TypeSystem.{u}}
    {source target : Port types}
    (equal : source = target)
    (bundle : Bundle source)
    (assignment : ColumnId -> Field) :
    (castBundle equal bundle).toColumnBundle.values assignment =
      bundle.toColumnBundle.values assignment := by
  cases equal
  rfl

/-- Schema-bundle conversion preserves exact typed reference resolution. -/
theorem Columns.toSchemaBundles_get
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
      cases columns
      exact inductionHypothesis _

/-- Exact typed exports preserve decoding and introduce no fallback
coordinates. -/
theorem Columns.export_decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context result : Schema types}
    (exports : Exports types context result)
    (compatible : ExportsCompatible exports)
    (columns : Columns context)
    (values : Schema.Values types context)
    (decoded : Columns.Decodes family columns assignment values) :
    Columns.Decodes family
      (exportColumns exports compatible columns) assignment
      (exports.get values) := by
  induction exports with
  | nil =>
      cases compatible
      trivial
  | @cons port tail reference exports inductionHypothesis =>
      cases compatible with
      | cons equal rest =>
          have sourceDecoded :=
            SchemaBundles.get_decodes family assignment reference
              columns.toSchemaBundles values decoded
          rw [Columns.toSchemaBundles_get] at sourceDecoded
          have castDecoded :
              (castBundle equal
                  (refBundle reference columns)).toColumnBundle.Decodes
                family port.kind assignment (reference.get values) := by
            unfold ColumnBundle.Decodes at sourceDecoded ⊢
            rw [Bundle.cast_values]
            exact sourceDecoded
          exact ⟨castDecoded,
            inductionHypothesis rest⟩

/-- Exact typed exports preserve honest encoding and introduce no fallback
coordinates. -/
theorem Columns.export_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context result : Schema types}
    (exports : Exports types context result)
    (compatible : ExportsCompatible exports)
    (columns : Columns context)
    (values : Schema.Values types context)
    (encoded : Columns.Encodes family columns assignment values) :
    Columns.Encodes family
      (exportColumns exports compatible columns) assignment
      (exports.get values) := by
  induction exports with
  | nil =>
      cases compatible
      trivial
  | @cons port tail reference exports inductionHypothesis =>
      cases compatible with
      | cons equal rest =>
          have sourceEncoded :=
            SchemaBundles.get_encodes family assignment reference
              columns.toSchemaBundles values encoded
          rw [Columns.toSchemaBundles_get] at sourceEncoded
          have castEncoded :
              (castBundle equal
                  (refBundle reference columns)).toColumnBundle.Encodes
                family port.kind assignment (reference.get values) := by
            unfold ColumnBundle.Encodes at sourceEncoded ⊢
            rw [Bundle.cast_values]
            exact sourceEncoded
          exact ⟨castEncoded,
            inductionHypothesis rest⟩

/-- An honest encoding of an appended context restricts to its exact left
prefix. -/
theorem Columns.left_encodes_of_append
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
          exact ⟨encoded.1, inductionHypothesis values encoded.2⟩

/-- An honest encoding of an appended context restricts to its exact right
suffix. -/
theorem Columns.right_encodes_of_append
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

/-- Bundle values are assignment lookup over the exact ordered identities. -/
theorem ColumnBundle.values_eq_ids_map
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field) :
    bundle.values assignment = bundle.ids.map assignment := by
  simp [ColumnBundle.values, ColumnBundle.ids, List.map_map]

private theorem instructionHead_owned
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (port : Port types) :
    ∀ column,
      column ∈
          (HVec.head (instructionColumns path [port])).toColumnBundle.columns ->
        column.id.owner = .typed (.instruction path) := by
  intro column member
  simp only [Bundle.toColumnBundle_columns, instructionColumns,
    allocateSchema, allocateSchemaFrom, bundleOwnedColumns,
    List.mem_ofFn] at member
  rcases member with ⟨coordinate, rfl⟩
  rfl

structure InvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (call : (SelectedSignature parameters).Call)
    (operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call))
    (path : OwnerPath)
    (inputColumns : Columns context)
    (one active : ColumnId) where
  recipe :
    CallRecipe (SelectedSignature parameters)
      (SelectedFamily parameters profile) call
  frame :
    CallFrame (SelectedFamily parameters profile) call operands
  ownerExact : frame.owner = .typed (.instruction path)
  oneExact : frame.one = one
  activeExact : frame.active = active
  contextExact : frame.contextBundles = inputColumns.toSchemaBundles
  outputsExact :
    frame.outputs =
      (instructionColumns path
        ((SelectedSignature parameters).callOutputs call)).toSchemaBundles
  temporariesExact :
    frame.temporaries =
      (temporaryColumns path
        ((SelectedSignature parameters).callOutputs call)
        ((SelectedSignature parameters).callFootprint call).temporaries
      ).toLayoutBundles

structure LiteralPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (port : Port (typeSystem parameters))
    (value : (typeSystem parameters).Value port.kind)
    (path : OwnerPath)
    (inputColumns : Columns context)
    (one active : ColumnId) where
  recipe :
    LiteralPinRecipe
      ((SelectedFamily parameters profile).codecFor port.kind)
      port.layout
  ownerExact : recipe.owner = .typed (.instruction path)
  ordinalExact : recipe.firstOrdinal = 0
  oneExact : recipe.one = one
  outputExact :
    recipe.output =
      (HVec.head (instructionColumns path [port])).toColumnBundle
  valueExact : recipe.value = value
  admissible :
    ((SelectedFamily parameters profile).codecFor port.kind).Admissible value
  rowsOwned :
    ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner
  rowsSupported :
    ∀ row, row ∈ recipe.rows ->
      ∀ column, column ∈ row.columnIds ->
        column ∈ [recipe.one] ++ recipe.output.ids
  columnIdsNodup : recipe.output.ids.Nodup
  rowIdsNodup : (recipe.rows.map fun row => row.id).Nodup

structure AssertPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    {context : Schema (typeSystem parameters)}
    (condition : Ref (typeSystem parameters) context .bit)
    (path : OwnerPath)
    (inputColumns : Columns context)
    (one active : ColumnId) where
  recipe : BoolAssertRecipe
  ownerExact : recipe.owner = .typed (.instruction path)
  ordinalExact : recipe.ordinal = 0
  oneExact : recipe.one = one
  activeExact : recipe.active = active
  conditionIdsExact :
    (inputColumns.toSchemaBundles.get condition).ids = [recipe.condition]

/-- Only the three primitive forms used by fixed-one inhabit this plan.
`linear` and `product` have no constructor. -/
inductive PrimitivePlan
    (parameters : Parameters)
    (profile : Profile parameters) :
    {input output : Schema (typeSystem parameters)} ->
    (primitive : Primitive (SelectedSignature parameters) input output) ->
    (path : OwnerPath) ->
    (inputColumns : Columns input) ->
    (one active : ColumnId) -> Type (u + 2) where
  | invoke
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
      PrimitivePlan parameters profile (.invoke call operands)
        path inputColumns one active
  | literal
      {context : Schema (typeSystem parameters)}
      {port : Port (typeSystem parameters)}
      {value : (typeSystem parameters).Value port.kind}
      {path : OwnerPath}
      {inputColumns : Columns context}
      {one active : ColumnId}
      (plan :
        LiteralPlan parameters profile port value path
          inputColumns one active) :
      PrimitivePlan parameters profile
        (@Primitive.literal (SelectedSignature parameters)
          context port value)
        path inputColumns one active
  | assertTrue
      {context : Schema (typeSystem parameters)}
      {condition : Ref (typeSystem parameters) context .bit}
      {path : OwnerPath}
      {inputColumns : Columns context}
      {one active : ColumnId}
      (plan :
        AssertPlan parameters profile condition path
          inputColumns one active) :
      PrimitivePlan parameters profile (.assertTrue condition)
        path inputColumns one active

namespace InvokePlan

def receipt
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
        inputColumns one active) : InstructionReceipt :=
  InstructionReceipt.ofCall plan.recipe plan.frame

/-! The allocation projection must not inspect the semantic recipe or any
proof stored in its frame.  Only the canonical output and temporary column
plans allocate columns. -/
@[simp] theorem receipt_allocations_exact
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
    plan.receipt.allocations =
      schemaOwnedColumns
          (instructionColumns path
            ((SelectedSignature parameters).callOutputs call)) ++
        schemaOwnedColumns
          (temporaryColumns path
            ((SelectedSignature parameters).callOutputs call)
            ((SelectedSignature parameters).callFootprint call).temporaries) := by
  unfold receipt InstructionReceipt.ofCall CallFrame.allocations
  change
    plan.frame.outputs.columns ++ plan.frame.temporaries.columns =
      _
  have outputsEqual :=
    congrArg SchemaBundles.columns plan.outputsExact
  have temporariesEqual :=
    congrArg LayoutBundles.columns plan.temporariesExact
  rw [outputsEqual, temporariesEqual]
  simp only [Columns.toSchemaBundles_columns,
    Columns.toLayoutBundles_columns]

end InvokePlan

namespace LiteralPlan

theorem allocationsOwned
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
    ∀ column, column ∈ plan.recipe.output.columns ->
      column.id.owner = plan.recipe.owner := by
  intro column member
  rw [plan.outputExact] at member
  rw [plan.ownerExact]
  exact instructionHead_owned path port column member

def receipt
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
        inputColumns one active) : InstructionReceipt :=
  InstructionReceipt.ofLiteral plan.recipe
    plan.allocationsOwned plan.rowsOwned

@[simp] theorem receipt_allocations_exact
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
    plan.receipt.allocations =
      bundleOwnedColumns port
        (HVec.head (instructionColumns path [port])) := by
  unfold receipt InstructionReceipt.ofLiteral
  change plan.recipe.output.columns = _
  exact congrArg ColumnBundle.columns plan.outputExact

end LiteralPlan

namespace AssertPlan

def receipt
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {condition : Ref (typeSystem parameters) context .bit}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      AssertPlan parameters profile condition path
        inputColumns one active) : InstructionReceipt :=
  InstructionReceipt.ofAssertion plan.recipe

@[simp] theorem receipt_allocations_exact
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {condition : Ref (typeSystem parameters) context .bit}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      AssertPlan parameters profile condition path
        inputColumns one active) :
    plan.receipt.allocations = [] :=
  rfl

end AssertPlan

namespace PrimitivePlan

/-! The allocation projection is a small proof-free description of the
columns created by one typed primitive.  It deliberately omits input and
context columns because those columns already have earlier receipt owners. -/
def expectedAllocations
    {parameters : Parameters}
    {input output : Schema (typeSystem parameters)}
    (path : OwnerPath) :
    Primitive (SelectedSignature parameters) input output ->
      List OwnedColumn
  | .literal port _ =>
      bundleOwnedColumns port
        (HVec.head (instructionColumns path [port]))
  | .linear _ _ _ => []
  | .product _ _ _ => []
  | .invoke call _ =>
      schemaOwnedColumns
          (instructionColumns path
            ((SelectedSignature parameters).callOutputs call)) ++
        schemaOwnedColumns
          (temporaryColumns path
            ((SelectedSignature parameters).callOutputs call)
            ((SelectedSignature parameters).callFootprint call).temporaries)
  | .assertTrue _ => []

/-- The one and only physical receipt selected by a fixed-one primitive
plan.  The indexed match rules out unsupported primitive forms rather than
providing an escape receipt for them. -/
def receipt
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
        inputColumns one active) : InstructionReceipt :=
  match plan with
  | .invoke invokePlan => invokePlan.receipt
  | .literal literalPlan => literalPlan.receipt
  | .assertTrue assertPlan => assertPlan.receipt

@[simp] theorem receipt_allocations_exact
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
    plan.receipt.allocations = expectedAllocations path primitive := by
  cases plan with
  | invoke plan =>
      simpa [PrimitivePlan.receipt, expectedAllocations] using
        InvokePlan.receipt_allocations_exact plan
  | literal plan =>
      change
        plan.receipt.allocations =
          bundleOwnedColumns _ (HVec.head (instructionColumns path [_]))
      exact LiteralPlan.receipt_allocations_exact plan
  | assertTrue plan =>
      simpa [PrimitivePlan.receipt, expectedAllocations] using
        AssertPlan.receipt_allocations_exact plan

/-- Every supported primitive receipt has exactly its source instruction
owner. -/
theorem receipt_owner
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
    plan.receipt.owner = .typed (.instruction path) := by
  cases plan with
  | invoke plan =>
      simpa [receipt, InvokePlan.receipt, InstructionReceipt.ofCall] using
        plan.ownerExact
  | literal plan =>
      simpa [receipt, LiteralPlan.receipt, InstructionReceipt.ofLiteral] using
        plan.ownerExact
  | @assertTrue condition path inputColumns one active plan =>
      simpa [receipt, AssertPlan.receipt, InstructionReceipt.ofAssertion] using
        plan.ownerExact

/-- Canonical instruction outputs and call temporaries are locally
collision-free.  This is derived from the selected local recipe/frame, not
supplied by the whole-program assembler. -/
theorem columnIdsNodup
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
    plan.receipt.columnIds.Nodup := by
  cases plan with
  | invoke plan =>
      simpa [receipt, InvokePlan.receipt,
        InstructionReceipt.columnIds, CallFrame.allocations,
        SchemaBundles.ids, LayoutBundles.ids, List.map_append] using
          plan.frame.allocationsNodup
  | literal plan =>
      simpa [receipt, LiteralPlan.receipt,
        InstructionReceipt.columnIds, ColumnBundle.ids] using
          plan.columnIdsNodup
  | assertTrue plan =>
      simp [receipt, AssertPlan.receipt, InstructionReceipt.columnIds]

/-- Every supported primitive recipe has locally unique row occurrences. -/
theorem rowIdsNodup
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
    plan.receipt.rowIds.Nodup := by
  cases plan with
  | invoke plan =>
      simpa [receipt, InvokePlan.receipt,
        InstructionReceipt.rowIds] using
          plan.recipe.rowIdsNodup plan.frame
  | literal plan =>
      simpa [receipt, LiteralPlan.receipt,
        InstructionReceipt.rowIds] using plan.rowIdsNodup
  | assertTrue plan =>
      simp [receipt, AssertPlan.receipt, InstructionReceipt.rowIds,
        BoolAssertRecipe.rows]

/-- The pre-existing coordinates required by one primitive: verifier one,
the enclosing activation, and every coordinate of its exact input context. -/
def InputsAvailable
    {parameters : Parameters}
    {input : Schema (typeSystem parameters)}
    (inputColumns : Columns input)
    (one active : ColumnId)
    (available : List ColumnId) : Prop :=
  ∀ column,
    column ∈ [one, active] ++ inputColumns.toSchemaBundles.ids ->
      column ∈ available

/-- A supported primitive references only pre-existing context coordinates
or coordinates allocated by its own receipt. -/
theorem wellScopedAfter
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
    (available : List ColumnId)
    (inputsAvailable :
      InputsAvailable inputColumns one active available) :
    plan.receipt.WellScopedAfter available := by
  intro column member
  cases plan with
  | invoke plan =>
      rcases List.mem_flatMap.mp member with
        ⟨row, rowMember, columnMember⟩
      have rowMember' :
          row ∈ plan.recipe.rows plan.frame := by
        simpa [receipt, InvokePlan.receipt] using rowMember
      have columnMember' : column ∈ row.columnIds := by
        simpa [InstructionReceipt.rowColumns, OwnedRow.columnIds,
          Row.columnIds] using columnMember
      have supported :=
        plan.recipe.rowsSupported plan.frame row rowMember'
          column columnMember'
      rcases List.mem_append.mp supported with
        visibleMember | temporaryMember
      · simp only [CallFrame.visibleIds] at visibleMember
        rcases List.mem_append.mp visibleMember with
          controlOrContext | outputMember
        · left
          apply inputsAvailable column
          simpa [plan.oneExact, plan.activeExact,
            plan.contextExact] using controlOrContext
        · right
          simpa [receipt, InvokePlan.receipt,
            InstructionReceipt.columnIds, CallFrame.allocations,
            SchemaBundles.ids, LayoutBundles.ids,
            List.map_append] using
              List.mem_append_left plan.frame.temporaries.ids outputMember
      · right
        simpa [receipt, InvokePlan.receipt,
          InstructionReceipt.columnIds, CallFrame.allocations,
          SchemaBundles.ids, LayoutBundles.ids,
          List.map_append] using
            List.mem_append_right plan.frame.outputs.ids temporaryMember
  | literal plan =>
      rcases List.mem_flatMap.mp member with
        ⟨row, rowMember, columnMember⟩
      have rowMember' : row ∈ plan.recipe.rows := by
        simpa [receipt, LiteralPlan.receipt] using rowMember
      have columnMember' : column ∈ row.columnIds := by
        simpa [InstructionReceipt.rowColumns, OwnedRow.columnIds,
          Row.columnIds] using columnMember
      have supported :=
        plan.rowsSupported row rowMember' column columnMember'
      rcases List.mem_append.mp supported with
        oneMember | outputMember
      · left
        apply inputsAvailable column
        simp only [List.mem_singleton] at oneMember
        subst column
        simp [plan.oneExact]
      · right
        simpa [receipt, LiteralPlan.receipt,
          InstructionReceipt.columnIds, ColumnBundle.ids] using outputMember
  | @assertTrue condition path inputColumns one active plan =>
      have conditionAvailable : plan.recipe.condition ∈ available := by
        apply inputsAvailable plan.recipe.condition
        have conditionContext :
            plan.recipe.condition ∈
              inputColumns.toSchemaBundles.ids := by
          have selected :
              plan.recipe.condition ∈
                (inputColumns.toSchemaBundles.get
                  condition).ids := by
            rw [plan.conditionIdsExact]
            simp
          exact SchemaBundles.get_ids_subset
            condition inputColumns.toSchemaBundles
            plan.recipe.condition selected
        simp [conditionContext]
      have referenceMember :
          column = plan.recipe.active ∨
            column = plan.recipe.one ∨
              column = plan.recipe.condition := by
        simpa [receipt, AssertPlan.receipt,
          InstructionReceipt.referencedColumns,
          InstructionReceipt.rowColumns, InstructionReceipt.ofAssertion,
          BoolAssertRecipe.rows, CanonicalRow.row, Goldilocks.singleton,
          oneMinus, Row.columnIds] using member
      left
      rcases referenceMember with activeMember | oneMember | conditionMember
      · subst column
        apply inputsAvailable plan.recipe.active
        simp [plan.activeExact]
      · subst column
        apply inputsAvailable plan.recipe.one
        simp [plan.oneExact]
      · subst column
        exact conditionAvailable

end PrimitivePlan
