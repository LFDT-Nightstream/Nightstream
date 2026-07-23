import Nightstream.Implementation.Lowering.Goldilocks.Codec
import Nightstream.Implementation.Lowering.Goldilocks.Rows
import Nightstream.Implementation.Lowering.Typed.Program

/-!
Contract: activation-aware physical recipes for the closed calls of a typed
lowering signature.

Owns:
- ordered physical bundles for typed schemas and addressed call operands;
- exact-width decoding and admissible honest encoding through a selected
  Goldilocks codec family;
- explicit output and temporary allocations for one call occurrence;
- certified active soundness, active honest completeness, and inactive
  satisfiability;
- a canonical receipt whose allocations and rows are definitionally the
  recipe's complete emission.

Does not own: a whole-verifier acceptance predicate, branch activation
construction, physical allocation of the enclosing SSA context, a compiler
from arbitrary programs, Rust layouts, or generated artifacts.

The honest encoding predicate includes `Codec.Admissible`.  In particular,
using `boundedNatCodec` never asserts that every natural number has a
fixed-width round trip.

Emits constraints: exactly `Signature.callFootprint.recurringRows` rows for
each call occurrence.  Output and temporary columns are accounted separately
and in their declared bundle order.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

/-! ## Exact ordered physical bundles -/

/-- The physical coordinates allocated for one logical ownership layout.

The equality retains both multiplicity and order.  Consequently its length is
the exact coordinate count of the layout, rather than an upper bound or a
post-hoc census. -/
structure ColumnBundle (layout : Layout) where
  columns : List OwnedColumn
  ownerships_exact :
    columns.map (fun column => column.ownership) = layout.owners

namespace ColumnBundle

/-- Exact coordinate count of one bundle. -/
theorem length_eq
    {layout : Layout}
    (bundle : ColumnBundle layout) :
    bundle.columns.length = layout.owners.length := by
  have equal := congrArg List.length bundle.ownerships_exact
  simpa only [List.length_map] using equal

/-- Ordered physical identities of a bundle. -/
def ids {layout : Layout} (bundle : ColumnBundle layout) : List ColumnId :=
  bundle.columns.map (fun column => column.id)

/-- Read the ordered field coordinates of a bundle from an assignment. -/
def values
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field) : List Field :=
  bundle.columns.map (fun column => assignment column.id)

@[simp] theorem values_length
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field) :
    (bundle.values assignment).length = layout.owners.length := by
  simp only [values, List.length_map, bundle.length_eq]

/-- One bundle decodes to one semantic value through its kind's selected
codec. -/
def Decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (kind : types.Kind)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field)
    (value : types.Value kind) : Prop :=
  (family.codecFor kind).decode (bundle.values assignment) = some value

/-- Honest coordinates use the selected canonical encoding and explicitly
remain inside its finite admissible domain. -/
def Encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (kind : types.Kind)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field)
    (value : types.Value kind) : Prop :=
  (family.codecFor kind).Admissible value ∧
    bundle.values assignment = (family.codecFor kind).encode value

/-- Honest encoding implies successful decoding.  The admissibility premise
inside `Encodes` is load-bearing for bounded semantic domains such as `Nat`. -/
theorem decodes_of_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (kind : types.Kind)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (assignment : ColumnId -> Field)
    (value : types.Value kind)
    (encoded : bundle.Encodes family kind assignment value) :
    bundle.Decodes family kind assignment value := by
  rcases encoded with ⟨admissible, coordinates⟩
  unfold Decodes
  rw [coordinates]
  exact (family.codecFor kind).decode_encode value admissible

end ColumnBundle

/-- Ordered bundles for an exact typed schema.  The constructor order is the
schema order; no map or column-number search is involved. -/
inductive SchemaBundles {types : TypeSystem.{u}} :
    Schema types -> Type (u + 1) where
  | nil : SchemaBundles []
  | cons {port : Port types} {tail : Schema types} :
      ColumnBundle port.layout ->
      SchemaBundles tail ->
      SchemaBundles (port :: tail)

namespace SchemaBundles

/-- Preserve logical port boundaries and their declared order. -/
def portColumns
    {types : TypeSystem.{u}} :
    {schema : Schema types} ->
    SchemaBundles schema -> List (List OwnedColumn)
  | [], .nil => []
  | _ :: _, .cons head tail =>
      head.columns :: portColumns tail

/-- Flatten an ordered schema bundle without changing coordinate order. -/
def columns
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (bundles : SchemaBundles schema) : List OwnedColumn :=
  bundles.portColumns.flatten

def ids
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (bundles : SchemaBundles schema) : List ColumnId :=
  bundles.columns.map (fun column => column.id)

@[simp] theorem ids_cons
    {types : TypeSystem.{u}}
    {port : Port types}
    {tail : Schema types}
    (head : ColumnBundle port.layout)
    (rest : SchemaBundles tail) :
    (SchemaBundles.cons head rest).ids = head.ids ++ rest.ids := by
  simp [ids, columns, portColumns, ColumnBundle.ids]

@[simp] theorem portColumns_length
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (bundles : SchemaBundles schema) :
    bundles.portColumns.length = schema.length := by
  induction bundles with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [portColumns, List.length_cons, inductionHypothesis]

/-- Exact total coordinate count of an ordered schema bundle. -/
theorem columns_length
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (bundles : SchemaBundles schema) :
    bundles.columns.length =
      (schema.map fun port => port.layout.owners.length).sum := by
  induction bundles with
  | nil => rfl
  | @cons port tail head rest inductionHypothesis =>
      have restLength : rest.portColumns.flatten.length =
          (tail.map fun item => item.layout.owners.length).sum := by
        simpa only [columns] using inductionHypothesis
      simp only [columns, portColumns, List.flatten_cons,
        List.length_append, List.map_cons, List.sum_cons,
        head.length_eq, restLength]

/-- Resolve a typed de Bruijn reference to the exact bundle of the addressed
schema occurrence. -/
def get
    {types : TypeSystem.{u}}
    {schema : Schema types}
    {kind : types.Kind} :
    (reference : Ref types schema kind) ->
    (bundles : SchemaBundles schema) ->
    ColumnBundle reference.port.layout
  | .here _, .cons bundle _ => bundle
  | .there reference, .cons _ tail => get reference tail

theorem get_ids_subset
    {types : TypeSystem.{u}}
    {schema : Schema types}
    {kind : types.Kind}
    (reference : Ref types schema kind)
    (bundles : SchemaBundles schema) :
    ∀ id, id ∈ (bundles.get reference).ids -> id ∈ bundles.ids := by
  induction reference with
  | here =>
      cases bundles with
      | cons head tail =>
          intro id member
          rw [ids_cons]
          exact List.mem_append_left _ member
  | there reference inductionHypothesis =>
      cases bundles with
      | cons head tail =>
          intro id member
          rw [ids_cons]
          exact List.mem_append_right _
            (inductionHypothesis tail id member)

/-- Decode every bundle in schema order. -/
def Decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field) :
    {schema : Schema types} ->
    (bundles : SchemaBundles schema) ->
    Schema.Values types schema -> Prop
  | [], .nil, .nil => True
  | _ :: _, .cons head tail, .cons value values =>
      head.Decodes family _ assignment value ∧
        Decodes family assignment tail values

/-- Honest encoding of every schema value.  Each head conjunct contains its
codec's admissibility condition. -/
def Encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field) :
    {schema : Schema types} ->
    (bundles : SchemaBundles schema) ->
    Schema.Values types schema -> Prop
  | [], .nil, .nil => True
  | _ :: _, .cons head tail, .cons value values =>
      head.Encodes family _ assignment value ∧
        Encodes family assignment tail values

theorem decodes_of_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {schema : Schema types}
    (bundles : SchemaBundles schema)
    (values : Schema.Values types schema)
    (encoded : bundles.Encodes family assignment values) :
    bundles.Decodes family assignment values := by
  induction bundles with
  | nil =>
      cases values
      trivial
  | @cons port tail head rest inductionHypothesis =>
      cases values with
      | cons value values =>
          exact ⟨head.decodes_of_encodes family port.kind assignment value
              encoded.1,
            inductionHypothesis values encoded.2⟩

end SchemaBundles

/-- Ordered bundles for exact addressed operands.  Repeated references remain
repeated entries, which is semantically different from allocating a repeated
column. -/
inductive RefBundles
    {types : TypeSystem.{u}}
    {context : Schema types} :
    {sorts : List types.Kind} ->
    (references : Refs types context sorts) ->
    Type (u + 1) where
  | nil : RefBundles Refs.nil
  | cons
      {kind : types.Kind}
      {sorts : List types.Kind}
      {reference : Ref types context kind}
      {references : Refs types context sorts} :
      ColumnBundle reference.port.layout ->
      RefBundles references ->
      RefBundles (Refs.cons reference references)

namespace RefBundles

/-- Resolve all addressed operands from a schema allocation, preserving the
call signature's operand order exactly. -/
def fromSchema
    {types : TypeSystem.{u}}
    {context : Schema types} :
    {sorts : List types.Kind} ->
    (references : Refs types context sorts) ->
    SchemaBundles context ->
    RefBundles references
  | [], .nil, _ => .nil
  | _ :: _, .cons reference tail, bundles =>
      .cons (bundles.get reference) (fromSchema tail bundles)

def portColumns
    {types : TypeSystem.{u}}
    {context : Schema types}
    {sorts : List types.Kind}
    {references : Refs types context sorts} :
    RefBundles references -> List (List OwnedColumn)
  | .nil => []
  | .cons head tail => head.columns :: portColumns tail

def columns
    {types : TypeSystem.{u}}
    {context : Schema types}
    {sorts : List types.Kind}
    {references : Refs types context sorts}
    (bundles : RefBundles references) : List OwnedColumn :=
  bundles.portColumns.flatten

def ids
    {types : TypeSystem.{u}}
    {context : Schema types}
    {sorts : List types.Kind}
    {references : Refs types context sorts}
    (bundles : RefBundles references) : List ColumnId :=
  bundles.columns.map (fun column => column.id)

@[simp] theorem ids_cons
    {types : TypeSystem.{u}}
    {context : Schema types}
    {kind : types.Kind}
    {sorts : List types.Kind}
    {reference : Ref types context kind}
    {references : Refs types context sorts}
    (head : ColumnBundle reference.port.layout)
    (tail : RefBundles references) :
    (RefBundles.cons head tail).ids = head.ids ++ tail.ids := by
  simp [ids, columns, portColumns, ColumnBundle.ids]

theorem fromSchema_ids_subset
    {types : TypeSystem.{u}}
    {context : Schema types}
    {sorts : List types.Kind}
    (references : Refs types context sorts)
    (bundles : SchemaBundles context) :
    ∀ id, id ∈ (RefBundles.fromSchema references bundles).ids ->
      id ∈ bundles.ids := by
  induction references with
  | nil =>
      intro id member
      simp [fromSchema, ids, columns, portColumns] at member
  | cons reference tail inductionHypothesis =>
      intro id member
      simp only [fromSchema, ids_cons] at member
      rcases List.mem_append.mp member with headMember | tailMember
      · exact SchemaBundles.get_ids_subset reference bundles id headMember
      · exact inductionHypothesis id tailMember

/-- One bundle occurs for every operand occurrence, including repeated
references. -/
theorem portColumns_length
    {types : TypeSystem.{u}}
    {context : Schema types}
    {sorts : List types.Kind}
    {references : Refs types context sorts}
    (bundles : RefBundles references) :
    bundles.portColumns.length = references.toList.length := by
  induction bundles with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [portColumns, Refs.toList, List.length_cons,
        inductionHypothesis]

/-- Per-operand codec/layout width agreement in declared call order. -/
def WidthsAgree
    {types : TypeSystem.{u}}
    (family : Family types)
    {context : Schema types} :
    {sorts : List types.Kind} ->
    (references : Refs types context sorts) -> Prop
  | [], .nil => True
  | kind :: _, .cons reference tail =>
      (family.codecFor kind).width =
          reference.port.layout.owners.length ∧
        WidthsAgree family tail

/-- Decode all operands in the signature's declared order. -/
def Decodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context : Schema types} :
    {sorts : List types.Kind} ->
    {references : Refs types context sorts} ->
    RefBundles references ->
    HVec types.Value sorts -> Prop
  | [], .nil, .nil, .nil => True
  | _ :: _, .cons _ _, .cons head tail, .cons value values =>
      head.Decodes family _ assignment value ∧
        Decodes family assignment tail values

/-- Honest, admissible encoding of all operands in declared order. -/
def Encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context : Schema types} :
    {sorts : List types.Kind} ->
    {references : Refs types context sorts} ->
    RefBundles references ->
    HVec types.Value sorts -> Prop
  | [], .nil, .nil, .nil => True
  | _ :: _, .cons _ _, .cons head tail, .cons value values =>
      head.Encodes family _ assignment value ∧
        Encodes family assignment tail values

theorem decodes_of_encodes
    {types : TypeSystem.{u}}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {context : Schema types}
    {sorts : List types.Kind}
    {references : Refs types context sorts}
    (bundles : RefBundles references)
    (values : HVec types.Value sorts)
    (encoded : bundles.Encodes family assignment values) :
    bundles.Decodes family assignment values := by
  induction bundles with
  | nil =>
      cases values
      trivial
  | @cons kind sorts reference references head tail inductionHypothesis =>
      cases values with
      | cons value values =>
          exact ⟨head.decodes_of_encodes family kind assignment value
              encoded.1,
            inductionHypothesis values encoded.2⟩

end RefBundles

/-- Ordered physical bundles for the exact temporary layouts declared by a
call footprint. -/
inductive LayoutBundles : List Layout -> Type where
  | nil : LayoutBundles []
  | cons {layout : Layout} {tail : List Layout} :
      ColumnBundle layout ->
      LayoutBundles tail ->
      LayoutBundles (layout :: tail)

namespace LayoutBundles

def bundleColumns :
    {layouts : List Layout} ->
    LayoutBundles layouts -> List (List OwnedColumn)
  | [], .nil => []
  | _ :: _, .cons head tail =>
      head.columns :: bundleColumns tail

def columns
    {layouts : List Layout}
    (bundles : LayoutBundles layouts) : List OwnedColumn :=
  bundles.bundleColumns.flatten

def ids
    {layouts : List Layout}
    (bundles : LayoutBundles layouts) : List ColumnId :=
  bundles.columns.map (fun column => column.id)

@[simp] theorem bundleColumns_length
    {layouts : List Layout}
    (bundles : LayoutBundles layouts) :
    bundles.bundleColumns.length = layouts.length := by
  induction bundles with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [bundleColumns, List.length_cons, inductionHypothesis]

/-- Exact total coordinate count of all temporary bundles. -/
theorem columns_length
    {layouts : List Layout}
    (bundles : LayoutBundles layouts) :
    bundles.columns.length =
      (layouts.map fun layout => layout.owners.length).sum := by
  induction bundles with
  | nil => rfl
  | @cons layout tail head rest inductionHypothesis =>
      have restLength : rest.bundleColumns.flatten.length =
          (tail.map fun item => item.owners.length).sum := by
        simpa only [columns] using inductionHypothesis
      simp only [columns, bundleColumns, List.flatten_cons,
        List.length_append, List.map_cons, List.sum_cons,
        head.length_eq, restLength]

/-- The type index is the exact temporary-layout order. -/
def declaredLayouts
    {layouts : List Layout}
    (_bundles : LayoutBundles layouts) : List Layout :=
  layouts

@[simp] theorem declaredLayouts_exact
    {layouts : List Layout}
    (bundles : LayoutBundles layouts) :
    bundles.declaredLayouts = layouts :=
  rfl

end LayoutBundles

/-! ## One explicit physical call occurrence -/

/-- No physical identity occurs in both ordered collections.  This explicit
predicate avoids treating list order or multiplicity as set structure. -/
def IdsDisjoint (left right : List ColumnId) : Prop :=
  ∀ id, id ∈ left -> id ∉ right

/-- Explicit physical inputs and fresh allocations for one invocation.

`one`, `active`, and the complete context are pre-existing columns.  Operand
bundles are resolved from that same context rather than supplied as a second,
potentially inconsistent view.  Only `outputs` and `temporaries` are fresh
allocations in this call receipt. -/
structure CallFrame
    {signature : Signature.{u}}
    (family : Family signature.types)
    (call : signature.Call)
    {context : Schema signature.types}
    (references :
      Refs signature.types context (signature.callInputs call)) where
  /-- Structural owner of every row emitted for this exact call occurrence. -/
  owner : PhysicalOwner
  one : ColumnId
  active : ColumnId
  contextBundles : SchemaBundles context
  outputs : SchemaBundles (signature.callOutputs call)
  temporaries :
    LayoutBundles (signature.callFootprint call).temporaries
  operandWidthsAgree : RefBundles.WidthsAgree family references
  outputWidthsAgree :
    SchemaWidthAgrees family (signature.callOutputs call)
  /-- Every freshly allocated output or temporary coordinate has a distinct
  physical identity. -/
  allocationsNodup :
    (outputs.ids ++ temporaries.ids).Nodup
  /-- A temporary witness can be completed without changing a visible input,
  activation, constant-one, unrelated context value, or output coordinate. -/
  temporariesDisjointVisible :
    IdsDisjoint temporaries.ids
      ([one, active] ++ contextBundles.ids ++ outputs.ids)
  /-- Call outputs are fresh relative to every pre-existing visible
  coordinate.  `one = active` remains allowed for the top-level call. -/
  outputsDisjointPreexisting :
    IdsDisjoint outputs.ids ([one, active] ++ contextBundles.ids)
  /-- Every fresh allocation belongs to this call occurrence's structural
  owner. -/
  allocationsOwned :
    ∀ column,
      column ∈ outputs.columns ++ temporaries.columns ->
        column.id.owner = owner

namespace CallFrame

/-- Resolve the call operands from the authoritative complete context stored
in the frame. -/
def operands
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) : RefBundles references :=
  RefBundles.fromSchema references frame.contextBundles

def visibleIds
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) : List ColumnId :=
  [frame.one, frame.active] ++
    frame.contextBundles.ids ++ frame.outputs.ids

/-- The complete fresh allocation list, in output order followed by
temporary-layout order. -/
def allocations
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) : List OwnedColumn :=
  frame.outputs.columns ++ frame.temporaries.columns

end CallFrame

/-- Pointwise preservation of already assigned physical coordinates. -/
def AgreesOn
    (ids : List ColumnId)
    (before after : ColumnId -> Field) : Prop :=
  ∀ id, id ∈ ids -> after id = before id

/-- A completion may change only the explicitly allocated coordinates.
This is the compositional half of a receipt: later calls cannot invalidate
rows emitted by earlier instructions. -/
def ChangesOnly
    (ids : List ColumnId)
    (before after : ColumnId -> Field) : Prop :=
  ∀ id, id ∉ ids -> after id = before id

/-- Complete physical receipt of one call occurrence.  There is deliberately
no generic `extraRows` or `glue` field. -/
structure CallReceipt where
  outputBundles : List (List OwnedColumn)
  temporaryBundles : List (List OwnedColumn)
  rows : List OwnedRow

namespace CallReceipt

def allocations (receipt : CallReceipt) : List OwnedColumn :=
  receipt.outputBundles.flatten ++ receipt.temporaryBundles.flatten

end CallReceipt

/-! ## Certified activation-aware recipes -/

/-- A certified physical recipe for one closed call.

The row function receives every physical dependency explicitly.  Soundness is
required only when the enclosing activation equals one.  Completeness
preserves the visible assignment and may fill only the recipe's temporary
coordinates.  When activation is zero, arbitrary visible operand and output
coordinates remain satisfiable, so an unselected branch cannot impose the
call relation.
-/
structure CallRecipe
    (signature : Signature.{u})
    (family : Family signature.types)
    (call : signature.Call) where
  rows :
    {context : Schema signature.types} ->
    {references :
      Refs signature.types context (signature.callInputs call)} ->
    CallFrame family call references ->
    List OwnedRow
  rowCount :
    ∀ {context}
      {references :
        Refs signature.types context (signature.callInputs call)}
      (frame : CallFrame family call references),
      (rows frame).length =
        (signature.callFootprint call).recurringRows
  rowsOwned :
    ∀ {context}
      {references :
        Refs signature.types context (signature.callInputs call)}
      (frame : CallFrame family call references)
      (row : OwnedRow),
      row ∈ rows frame ->
        row.id.owner = frame.owner
  rowIdsNodup :
    ∀ {context}
      {references :
        Refs signature.types context (signature.callInputs call)}
      (frame : CallFrame family call references),
      ((rows frame).map fun row => row.id).Nodup
  /-- Every physical dependency of every emitted row is either visible before
  completion or is an explicitly declared temporary of this call. -/
  rowsSupported :
    ∀ {context}
      {references :
        Refs signature.types context (signature.callInputs call)}
      (frame : CallFrame family call references)
      (row : OwnedRow),
      row ∈ rows frame ->
      ∀ column, column ∈ row.columnIds ->
        column ∈ frame.visibleIds ++ frame.temporaries.ids
  activeSoundness :
    ∀ {context}
      {references :
        Refs signature.types context (signature.callInputs call)}
      (frame : CallFrame family call references)
      (assignment : ColumnId -> Field)
      (inputs :
        HVec signature.types.Value (signature.callInputs call)),
      assignment frame.one = 1 ->
      assignment frame.active = 1 ->
      frame.operands.Decodes family assignment inputs ->
      Satisfies (rows frame) assignment ->
      ∃ outputs :
          Schema.Values signature.types (signature.callOutputs call),
        signature.callEval call inputs = some outputs ∧
          frame.outputs.Decodes family assignment outputs
  activeHonestCompleteness :
    ∀ {context}
      {references :
        Refs signature.types context (signature.callInputs call)}
      (frame : CallFrame family call references)
      (assignment : ColumnId -> Field)
      (inputs :
        HVec signature.types.Value (signature.callInputs call))
      (outputs :
        Schema.Values signature.types (signature.callOutputs call)),
      assignment frame.one = 1 ->
      assignment frame.active = 1 ->
      frame.operands.Encodes family assignment inputs ->
      frame.outputs.Encodes family assignment outputs ->
      signature.callEval call inputs = some outputs ->
      ∃ completed : ColumnId -> Field,
        AgreesOn frame.visibleIds assignment completed ∧
          ChangesOnly frame.temporaries.ids assignment completed ∧
          Satisfies (rows frame) completed
  inactiveSatisfiable :
    ∀ {context}
      {references :
        Refs signature.types context (signature.callInputs call)}
      (frame : CallFrame family call references)
      (assignment : ColumnId -> Field),
      assignment frame.one = 1 ->
      assignment frame.active = 0 ->
      ∃ completed : ColumnId -> Field,
        AgreesOn frame.visibleIds assignment completed ∧
          ChangesOnly frame.temporaries.ids assignment completed ∧
          Satisfies (rows frame) completed

namespace CallRecipe

/-- The only receipt exported by a recipe.  Its two allocation lists come
directly from the explicit frame, and its row list is exactly `rows frame`. -/
def receipt
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) : CallReceipt where
  outputBundles := frame.outputs.portColumns
  temporaryBundles := frame.temporaries.bundleColumns
  rows := recipe.rows frame

/-- Exact receipt equality.  This theorem is the no-hidden-emission boundary:
every output allocation, temporary allocation, and row is visible on its
right-hand side. -/
theorem receipt_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    recipe.receipt frame =
      { outputBundles := frame.outputs.portColumns
        temporaryBundles := frame.temporaries.bundleColumns
        rows := recipe.rows frame } :=
  rfl

@[simp] theorem receipt_allocations_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    (recipe.receipt frame).allocations = frame.allocations :=
  rfl

@[simp] theorem receipt_rows_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    (recipe.receipt frame).rows = recipe.rows frame :=
  rfl

theorem receipt_row_count
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    (recipe.receipt frame).rows.length =
      (signature.callFootprint call).recurringRows :=
  recipe.rowCount frame

@[simp] theorem receipt_output_order_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    (recipe.receipt frame).outputBundles = frame.outputs.portColumns :=
  rfl

@[simp] theorem receipt_temporary_order_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    (recipe.receipt frame).temporaryBundles =
      frame.temporaries.bundleColumns :=
  rfl

/-- Exact temporary layouts are carried by the frame's type index, not
recovered from emitted column metadata. -/
theorem temporary_layouts_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    frame.temporaries.declaredLayouts =
      (signature.callFootprint call).temporaries := by
  exact frame.temporaries.declaredLayouts_exact

end CallRecipe

/-- One certified recipe for every call in a closed typed signature. -/
structure CallRecipes
    (signature : Signature.{u})
    (family : Family signature.types) where
  recipe : (call : signature.Call) -> CallRecipe signature family call

end Nightstream.Implementation.Lowering.Goldilocks
