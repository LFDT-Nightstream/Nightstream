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

private theorem SchemaBundles.get_decodes
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

private theorem SchemaBundles.get_encodes
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
        inductionHypothesis bundles values decoded⟩

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
        inductionHypothesis bundles values encoded⟩

private theorem Columns.append_decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {left right : Schema types}
    (leftColumns : Columns left)
    (rightColumns : Columns right)
    (leftValues : Schema.Values types left)
    (rightValues : Schema.Values types right)
    (leftDecoded :
      leftColumns.Decodes family assignment leftValues)
    (rightDecoded :
      rightColumns.Decodes family assignment rightValues) :
    (leftColumns.append rightColumns).Decodes family assignment
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

private theorem Columns.left_encodes_of_append
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {left right : Schema types}
    (leftColumns : Columns left)
    (rightColumns : Columns right)
    (leftValues : Schema.Values types left)
    (rightValues : Schema.Values types right)
    (encoded :
      (leftColumns.append rightColumns).Encodes family assignment
        (leftValues.append rightValues)) :
    leftColumns.Encodes family assignment leftValues := by
  induction leftColumns with
  | nil =>
      trivial
  | cons head tail inductionHypothesis =>
      cases leftValues with
      | cons value values =>
          exact ⟨encoded.1, inductionHypothesis values encoded.2⟩

private theorem ColumnBundle.values_eq_ids_map
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
      {context call operands path inputColumns one active}
      (plan :
        InvokePlan parameters profile call operands path
          inputColumns one active) :
      PrimitivePlan parameters profile (.invoke call operands)
        path inputColumns one active
  | literal
      {context port value path inputColumns one active}
      (plan :
        LiteralPlan parameters profile port value path
          inputColumns one active) :
      PrimitivePlan parameters profile (.literal port value)
        path inputColumns one active
  | assertTrue
      {context condition path inputColumns one active}
      (plan :
        AssertPlan parameters profile condition path
          inputColumns one active) :
      PrimitivePlan parameters profile (.assertTrue condition)
        path inputColumns one active

namespace InvokePlan

def receipt
    {parameters : Parameters}
    {profile : Profile parameters}
    {context call operands path inputColumns one active}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active) : InstructionReceipt :=
  InstructionReceipt.ofCall plan.recipe plan.frame

end InvokePlan

namespace LiteralPlan

theorem allocationsOwned
    {parameters : Parameters}
    {profile : Profile parameters}
    {context port value path inputColumns one active}
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
    {context port value path inputColumns one active}
    (plan :
      LiteralPlan parameters profile port value path
        inputColumns one active) : InstructionReceipt :=
  InstructionReceipt.ofLiteral plan.recipe
    plan.allocationsOwned plan.rowsOwned

end LiteralPlan

namespace AssertPlan

def receipt
    {parameters : Parameters}
    {profile : Profile parameters}
    {context condition path inputColumns one active}
    (plan :
      AssertPlan parameters profile condition path
        inputColumns one active) : InstructionReceipt :=
  InstructionReceipt.ofAssertion plan.recipe

end AssertPlan

