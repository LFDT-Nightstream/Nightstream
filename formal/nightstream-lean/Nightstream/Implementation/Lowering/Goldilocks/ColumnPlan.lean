import Nightstream.Implementation.Lowering.Goldilocks.Rows

/-!
Contract: finite, receipt-owned physical column plans for the canonical direct
Goldilocks encoding.

Owns:
- exact coordinate bundles for typed ports and schemas;
- reference/export projection without fallback columns;
- explicit input, instruction-output, call-temporary, branch-join, and
  activation allocations;
- the verifier-fixed constant-one prelude coordinate;
- four-way physical cost computed from actual owned occurrences.

Does not own: semantic codecs, call rows, Rust numeric column indices, or
generated artifacts.

Emits constraints: no.  The distinguished constant-one coordinate is fixed by
the physical verifier boundary (`assignment oneColumn = 1`), not by a circular
R1CS equation.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

/-- Exact physical coordinates allocated for one logical typed port. -/
structure Bundle {types : TypeSystem.{u}} (port : Port types) : Type u where
  column : Fin port.layout.owners.length -> ColumnId

namespace Bundle

def ids {types : TypeSystem.{u}} {port : Port types}
    (bundle : Bundle port) : List ColumnId :=
  List.ofFn bundle.column

@[simp] theorem ids_length {types : TypeSystem.{u}} {port : Port types}
    (bundle : Bundle port) :
    bundle.ids.length = port.layout.owners.length := by
  simp [ids]

end Bundle

/-- A column bundle for every port in an exact static schema. -/
abbrev Columns {types : TypeSystem.{u}} (schema : Schema types) :=
  HVec (fun port => Bundle port) schema

/-- Resolve an exact typed reference to its already allocated physical bundle.
There is no `getD` or constant-one fallback. -/
def refBundle {types : TypeSystem.{u}} {schema : Schema types}
    {kind : types.Kind} :
    (reference : Typed.Ref types schema kind) ->
    Columns schema -> Bundle reference.port
  | .here _, HVec.cons head _ => head
  | .there reference, HVec.cons _ tail => refBundle reference tail

def refColumnIds {types : TypeSystem.{u}} {schema : Schema types}
    {kind : types.Kind}
    (reference : Typed.Ref types schema kind)
    (columns : Columns schema) : List ColumnId :=
  (refBundle reference columns).ids

/-- Preserve call operand order while exposing each operand's exact physical
coordinate bundle. -/
def refsColumnIds {types : TypeSystem.{u}} {schema : Schema types} :
    {kinds : List types.Kind} ->
    Typed.Refs types schema kinds -> Columns schema -> List (List ColumnId)
  | [], .nil, _ => []
  | _ :: _, .cons reference tail, columns =>
      refColumnIds reference columns :: refsColumnIds tail columns

/-- Physical exports may reuse a source bundle only when the exported port,
including its ownership layout, is definitionally the same port.  The typed
IR guarantees the semantic sort; this separate certificate prevents a yield
from silently changing physical width or ownership. -/
inductive ExportsCompatible {types : TypeSystem.{u}}
    {context : Schema types} :
    {result : Schema types} -> Typed.Exports types context result -> Type (u + 1)
  | nil : ExportsCompatible .nil
  | cons {port : Port types} {tail : Schema types}
      {reference : Typed.Ref types context port.kind}
      {exports : Typed.Exports types context tail}
      (head : reference.port = port)
      (rest : ExportsCompatible exports) :
      ExportsCompatible (.cons reference exports)

/-- Transport a coordinate bundle only across equality of the complete ports,
including both the semantic kind and the physical ownership layout. -/
def castBundle {types : TypeSystem.{u}} {source target : Port types}
    (equal : source = target) (bundle : Bundle source) : Bundle target :=
  equal ▸ bundle

/-- Exact physical bundles selected by a compatible typed yield/export list. -/
def exportColumns {types : TypeSystem.{u}} {context : Schema types} :
    {result : Schema types} ->
    (exports : Typed.Exports types context result) ->
    ExportsCompatible exports -> Columns context -> Columns result
  | [], .nil, .nil, _ => HVec.nil
  | _ :: _, .cons reference exports, .cons equal rest, source =>
      HVec.cons
        (castBundle equal (refBundle reference source))
        (exportColumns exports rest source)

/-- Deterministically allocate one schema.  `ownerAt` chooses the structural
owner for each logical bundle; the coordinate index is intrinsically bounded
by that port's declared layout. -/
def allocateSchemaFrom {types : TypeSystem.{u}} :
    (ownerAt : Nat -> PhysicalOwner) ->
    (bundleIndex : Nat) ->
    (schema : Schema types) -> Columns schema
  | _, _, [] => HVec.nil
  | ownerAt, bundleIndex, port :: tail =>
      HVec.cons
        ⟨fun coordinate =>
          { owner := ownerAt bundleIndex
            bundleIndex := bundleIndex
            coordinateIndex := coordinate.val }
        ⟩
        (allocateSchemaFrom ownerAt (bundleIndex + 1) tail)

def allocateSchema {types : TypeSystem.{u}}
    (ownerAt : Nat -> PhysicalOwner)
    (schema : Schema types) : Columns schema :=
  allocateSchemaFrom ownerAt 0 schema

def bundleOwnedColumns {types : TypeSystem.{u}}
    (port : Port types) (bundle : Bundle port) : List OwnedColumn :=
  List.ofFn fun coordinate : Fin port.layout.owners.length =>
    { id := bundle.column coordinate
      ownership := port.layout.owners.get coordinate }

def schemaOwnedColumns {types : TypeSystem.{u}} :
    {schema : Schema types} -> Columns schema -> List OwnedColumn
  | [], HVec.nil => []
  | port :: _, HVec.cons head tail =>
      bundleOwnedColumns port head ++ schemaOwnedColumns tail

/-- Input ownership is role-local: input slot `i` owns only its own bundle. -/
def inputColumns {types : TypeSystem.{u}}
    (schema : Schema types) : Columns schema :=
  allocateSchema (fun slot => .typed (.input slot)) schema

def instructionColumns {types : TypeSystem.{u}}
    (path : OwnerPath) (schema : Schema types) : Columns schema :=
  allocateSchema (fun _ => .typed (.instruction path)) schema

def branchJoinColumns {types : TypeSystem.{u}}
    (path : OwnerPath) (schema : Schema types) : Columns schema :=
  allocateSchema (fun _ => .typed (.branch path)) schema

def temporaryColumns {types : TypeSystem.{u}}
    (path : OwnerPath)
    (outputSchema : Schema types)
    (layouts : List Layout) :
    Columns (layouts.map fun layout =>
      { kind := (TypeSystem.Kind.field : types.Kind)
        layout := layout }) :=
  allocateSchemaFrom (fun _ => .typed (.instruction path))
    outputSchema.length
    (layouts.map fun layout =>
      { kind := (TypeSystem.Kind.field : types.Kind)
        layout := layout })

/-- Canonical verifier-fixed constant-one coordinate. -/
def oneColumn : ColumnId :=
  { owner := .prelude, bundleIndex := 0, coordinateIndex := 0 }

def preludeColumns : List OwnedColumn :=
  [{ id := oneColumn, ownership := .publicColumn }]

def activationColumn (path : OwnerPath) (selected : Bool) : ColumnId :=
  { owner := .branchActivation path selected
    bundleIndex := 0
    coordinateIndex := 0 }

def activationColumns (path : OwnerPath) : List OwnedColumn :=
  [{ id := activationColumn path true, ownership := .auxiliaryColumn },
    { id := activationColumn path false, ownership := .auxiliaryColumn }]

/-- Column-side cost is a fold over the actual owned occurrences. -/
def columnCost (columns : List OwnedColumn) : Cost :=
  Cost.sum (columns.map fun column => Cost.oneColumn column.ownership)

def rowCost (rows : List OwnedRow) : Cost :=
  Cost.sum (rows.map fun _ => Cost.oneRow)

def physicalCost (columns : List OwnedColumn) (rows : List OwnedRow) : Cost :=
  columnCost columns + rowCost rows

private theorem cost_eq_of_components {left right : Cost}
    (rows : left.recurringRows = right.recurringRows)
    (committed : left.committedColumns = right.committedColumns)
    (publicColumns : left.publicColumns = right.publicColumns)
    (auxiliary : left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

theorem prelude_cost :
    columnCost preludeColumns = Cost.oneColumn .publicColumn := by
  rfl

theorem physicalCost_append
    (leftColumns rightColumns : List OwnedColumn)
    (leftRows rightRows : List OwnedRow) :
    physicalCost (leftColumns ++ rightColumns) (leftRows ++ rightRows) =
      physicalCost leftColumns leftRows +
        physicalCost rightColumns rightRows := by
  apply cost_eq_of_components <;>
    simp [physicalCost, columnCost, rowCost, List.map_append,
      Cost.sum_append, Cost.add, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

end Nightstream.Implementation.Lowering.Goldilocks
