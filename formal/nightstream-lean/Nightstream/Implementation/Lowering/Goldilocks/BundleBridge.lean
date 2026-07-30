import Nightstream.Implementation.Lowering.Goldilocks.CallRecipe
import Nightstream.Implementation.Lowering.Goldilocks.ColumnPlan

/-!
Contract: exact conversion from the compiler's dependent column plan to the
ordered bundles consumed by certified call recipes.

Owns: one structural conversion for ports, schemas, references, temporary
layouts, and call frames, preserving coordinate order and ownership exactly.

Does not own: allocation policy, call rows, semantic codecs, Rust numeric
columns, or generated artifacts.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

/-- Materialize a dependent compiler bundle as the exact ordered owned-column
bundle required by a call recipe. -/
def Bundle.toColumnBundle
    {types : TypeSystem.{u}}
    {port : Port types}
    (bundle : Bundle port) : ColumnBundle port.layout where
  columns := bundleOwnedColumns port bundle
  ownerships_exact := by
    rw [bundleOwnedColumns, List.map_ofFn]
    change List.ofFn (fun coordinate : Fin port.layout.owners.length =>
      port.layout.owners.get coordinate) = port.layout.owners
    exact List.ofFn_getElem

@[simp] theorem Bundle.toColumnBundle_columns
    {types : TypeSystem.{u}}
    {port : Port types}
    (bundle : Bundle port) :
    bundle.toColumnBundle.columns = bundleOwnedColumns port bundle :=
  rfl

/-- Preserve the exact schema order while converting every dependent bundle. -/
def Columns.toSchemaBundles {types : TypeSystem.{u}} :
    {schema : Schema types} -> Columns schema -> SchemaBundles schema
  | [], HVec.nil => .nil
  | _ :: _, HVec.cons head tail =>
      .cons head.toColumnBundle (toSchemaBundles tail)

@[simp] theorem Columns.toSchemaBundles_columns
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (columns : Columns schema) :
    columns.toSchemaBundles.columns = schemaOwnedColumns columns := by
  induction columns with
  | nil => rfl
  | @cons port tail head rest inductionHypothesis =>
      have restExact :
          (toSchemaBundles rest).portColumns.flatten =
            schemaOwnedColumns rest := by
        simpa only [SchemaBundles.columns] using inductionHypothesis
      simp only [toSchemaBundles, SchemaBundles.columns,
        SchemaBundles.portColumns, List.flatten_cons,
        Bundle.toColumnBundle_columns, schemaOwnedColumns, restExact]

/-- Resolve call operands from the same schema bundle used by the compiler. -/
def Columns.toRefBundles
    {types : TypeSystem.{u}}
    {context : Schema types}
    {sorts : List types.Kind}
    (references : Refs types context sorts)
    (columns : Columns context) : RefBundles references :=
  RefBundles.fromSchema references columns.toSchemaBundles

theorem ref_port_mem
    {types : TypeSystem.{u}}
    {context : Schema types}
    {kind : types.Kind}
    (reference : Ref types context kind) :
    reference.port ∈ context := by
  induction reference with
  | here => exact List.mem_cons_self
  | there reference inductionHypothesis =>
      exact List.mem_cons_of_mem _ inductionHypothesis

/-- Schema-wide width agreement entails the ordered operand-width contract
for every exact reference list. -/
theorem refBundles_widthsAgree_of_schema
    {types : TypeSystem.{u}}
    {family : Family types}
    {context : Schema types}
    (contextWidths : SchemaWidthAgrees family context) :
    {sorts : List types.Kind} ->
    (references : Refs types context sorts) ->
    RefBundles.WidthsAgree family references
  | [], .nil => True.intro
  | _ :: _, .cons reference tail =>
      ⟨by
        have width :=
          contextWidths reference.port (ref_port_mem reference)
        unfold PortWidthAgrees at width
        rw [reference.port_sort] at width
        exact width,
        refBundles_widthsAgree_of_schema contextWidths tail⟩

/-- Convert the exact compiler temporary allocation, without reconstructing
it from numeric columns or optional metadata. -/
def Columns.toLayoutBundles
    {types : TypeSystem.{u}} :
    {layouts : List Layout} ->
    Columns (layouts.map fun layout =>
      { kind := (TypeSystem.Kind.field : types.Kind), layout := layout }) ->
    LayoutBundles layouts
  | [], HVec.nil => .nil
  | _ :: _, HVec.cons head tail =>
      .cons head.toColumnBundle (toLayoutBundles tail)

@[simp] theorem Columns.toLayoutBundles_columns
    {types : TypeSystem.{u}}
    {layouts : List Layout}
    (columns :
      Columns (layouts.map fun layout =>
        { kind := (TypeSystem.Kind.field : types.Kind), layout := layout })) :
    columns.toLayoutBundles.columns = schemaOwnedColumns columns := by
  induction layouts with
  | nil =>
      cases columns
      rfl
  | cons layout tail inductionHypothesis =>
      cases columns with
      | cons head rest =>
          have restExact :
              (toLayoutBundles rest).bundleColumns.flatten =
                schemaOwnedColumns rest := by
            simpa only [LayoutBundles.columns] using
              inductionHypothesis rest
          simp only [toLayoutBundles, LayoutBundles.columns,
            LayoutBundles.bundleColumns, List.flatten_cons,
            Bundle.toColumnBundle_columns, restExact]
          change bundleOwnedColumns
              { kind := TypeSystem.Kind.field, layout := layout } head ++
              schemaOwnedColumns rest =
            bundleOwnedColumns
              { kind := TypeSystem.Kind.field, layout := layout } head ++
              schemaOwnedColumns rest
          rfl

/-- Build a call frame directly from compiler allocations.  Every order and
width certificate is explicit; there is no receipt archaeology. -/
def callFrameOfColumns
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (contextColumns : Columns context)
    (outputColumns : Columns (signature.callOutputs call))
    (temporaryColumns :
      Columns ((signature.callFootprint call).temporaries.map fun layout =>
        { kind := (TypeSystem.Kind.field : signature.types.Kind)
          layout := layout }))
    (operandWidths : RefBundles.WidthsAgree family references)
    (outputWidths :
      SchemaWidthAgrees family (signature.callOutputs call))
    (allocationsNodup :
      (outputColumns.toSchemaBundles.ids ++
        temporaryColumns.toLayoutBundles.ids).Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint temporaryColumns.toLayoutBundles.ids
        ([one, active] ++
          contextColumns.toSchemaBundles.ids ++
          outputColumns.toSchemaBundles.ids))
    (outputsDisjointPreexisting :
      IdsDisjoint outputColumns.toSchemaBundles.ids
        ([one, active] ++
          contextColumns.toSchemaBundles.ids))
    (allocationsOwned :
      ∀ column,
        column ∈ outputColumns.toSchemaBundles.columns ++
            temporaryColumns.toLayoutBundles.columns ->
          column.id.owner = owner) :
    CallFrame family call references where
  owner := owner
  one := one
  active := active
  contextBundles := contextColumns.toSchemaBundles
  outputs := outputColumns.toSchemaBundles
  temporaries := temporaryColumns.toLayoutBundles
  operandWidthsAgree := operandWidths
  outputWidthsAgree := outputWidths
  allocationsNodup := allocationsNodup
  temporariesDisjointVisible := temporariesDisjointVisible
  outputsDisjointPreexisting := outputsDisjointPreexisting
  allocationsOwned := allocationsOwned

@[simp] theorem callFrameOfColumns_operands
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (contextColumns : Columns context)
    (outputColumns : Columns (signature.callOutputs call))
    (temporaryColumns :
      Columns ((signature.callFootprint call).temporaries.map fun layout =>
        { kind := (TypeSystem.Kind.field : signature.types.Kind)
          layout := layout }))
    (operandWidths : RefBundles.WidthsAgree family references)
    (outputWidths :
      SchemaWidthAgrees family (signature.callOutputs call))
    (allocationsNodup :
      (outputColumns.toSchemaBundles.ids ++
        temporaryColumns.toLayoutBundles.ids).Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint temporaryColumns.toLayoutBundles.ids
        ([one, active] ++
          contextColumns.toSchemaBundles.ids ++
          outputColumns.toSchemaBundles.ids))
    (outputsDisjointPreexisting :
      IdsDisjoint outputColumns.toSchemaBundles.ids
        ([one, active] ++
          contextColumns.toSchemaBundles.ids))
    (allocationsOwned :
      ∀ column,
        column ∈ outputColumns.toSchemaBundles.columns ++
            temporaryColumns.toLayoutBundles.columns ->
          column.id.owner = owner) :
    (callFrameOfColumns owner one active contextColumns outputColumns
      temporaryColumns operandWidths outputWidths allocationsNodup
      temporariesDisjointVisible outputsDisjointPreexisting
      allocationsOwned).operands =
        contextColumns.toRefBundles references :=
  rfl

@[simp] theorem callFrameOfColumns_allocations
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (contextColumns : Columns context)
    (outputColumns : Columns (signature.callOutputs call))
    (temporaryColumns :
      Columns ((signature.callFootprint call).temporaries.map fun layout =>
        { kind := (TypeSystem.Kind.field : signature.types.Kind)
          layout := layout }))
    (operandWidths : RefBundles.WidthsAgree family references)
    (outputWidths :
      SchemaWidthAgrees family (signature.callOutputs call))
    (allocationsNodup :
      (outputColumns.toSchemaBundles.ids ++
        temporaryColumns.toLayoutBundles.ids).Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint temporaryColumns.toLayoutBundles.ids
        ([one, active] ++
          contextColumns.toSchemaBundles.ids ++
          outputColumns.toSchemaBundles.ids))
    (outputsDisjointPreexisting :
      IdsDisjoint outputColumns.toSchemaBundles.ids
        ([one, active] ++
          contextColumns.toSchemaBundles.ids))
    (allocationsOwned :
      ∀ column,
        column ∈ outputColumns.toSchemaBundles.columns ++
            temporaryColumns.toLayoutBundles.columns ->
          column.id.owner = owner) :
    (callFrameOfColumns owner one active contextColumns outputColumns
      temporaryColumns operandWidths outputWidths allocationsNodup
      temporariesDisjointVisible outputsDisjointPreexisting
      allocationsOwned).allocations =
      schemaOwnedColumns outputColumns ++
        schemaOwnedColumns temporaryColumns := by
  simp [callFrameOfColumns, CallFrame.allocations]

end Nightstream.Implementation.Lowering.Goldilocks
