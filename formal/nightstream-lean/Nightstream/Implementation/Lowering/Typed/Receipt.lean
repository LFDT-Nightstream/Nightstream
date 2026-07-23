import Nightstream.Implementation.Lowering.Typed.Program

/-!
Contract: total abstract emission receipts for typed lowering programs.

Owns:
- canonical allocation receipts for every coordinate in the runtime schema;
- exact addressed operands, outputs, call temporaries, and row tokens for
  every semantic primitive;
- structural owner paths for sequential instructions and both sides of every
  branch;
- explicit branch selector, arm-gating, and joined-output events;
- definitional four-way cost folds and receipt conservation.

Does not own: sparse R1CS rows, physical column numbers, a concrete encoding,
generated artifacts, Rust emission, or obligation-10 minimality.

Every event is constructed by a receipt.  There is no optional metadata,
external call plan, or secondary emission list to which glue can be appended.
-/

namespace Nightstream.Implementation.Lowering.Typed

universe u

/-- One coordinate in an ordered collection of logical layouts. -/
structure Coordinate where
  portIndex : Nat
  coordinateIndex : Nat
  ownership : Ownership
deriving DecidableEq, Repr

/-- Enumerate one ownership layout with stable zero-based coordinates. -/
def layoutCoordinatesFrom (portIndex coordinateIndex : Nat) :
    List Ownership -> List Coordinate
  | [] => []
  | ownership :: tail =>
      ⟨portIndex, coordinateIndex, ownership⟩ ::
        layoutCoordinatesFrom portIndex (coordinateIndex + 1) tail

/-- Enumerate every coordinate of an ordered layout collection. -/
def layoutsCoordinatesFrom (portIndex : Nat) :
    List Layout -> List Coordinate
  | [] => []
  | layout :: tail =>
      layoutCoordinatesFrom portIndex 0 layout.owners ++
        layoutsCoordinatesFrom (portIndex + 1) tail

def layoutsCoordinates (layouts : List Layout) : List Coordinate :=
  layoutsCoordinatesFrom 0 layouts

def schemaCoordinates {types : TypeSystem.{u}}
    (schema : Schema types) : List Coordinate :=
  layoutsCoordinates (schema.map Port.layout)

/-- Collision-free structural address of an instruction inside a block. -/
inductive OwnerPath where
  | root
  | rest (parent : OwnerPath)
  | trueArm (parent : OwnerPath)
  | falseArm (parent : OwnerPath)
  | continuation (parent : OwnerPath)
deriving DecidableEq, Repr

/-- Every event belongs either to one runtime input slot, one semantic
instruction, or one branch-control/join node. -/
inductive Owner where
  | input (slot : Nat)
  | instruction (path : OwnerPath)
  | branch (path : OwnerPath)
deriving DecidableEq, Repr

/-- Stable identity of one event within its structural owner. -/
structure EventId where
  owner : Owner
  ordinal : Nat
deriving DecidableEq, Repr

/-- A call temporary is globally identified by its structural instruction
owner, temporary bundle, and coordinate. -/
structure TemporaryId where
  owner : Owner
  temporaryIndex : Nat
  coordinateIndex : Nat
deriving DecidableEq, Repr

/-- Abstract allocation roles.  Obligation 10 will choose their physical
column encoding. -/
inductive AllocationRole where
  | input
  | output
  | temporary
  | branchJoin
deriving DecidableEq, Repr

/-- Semantic row token.  Tokens retain exact coefficients and addressed
operands; opaque call rows also retain the call and their fixed row index. -/
inductive RowToken (signature : Signature.{u}) : Type (u + 2) where
  | literal (port : Port signature.types)
      (value : signature.types.Value port.kind)
  | linear (context : Schema signature.types)
      (constant : signature.types.Field)
      (terms :
        List (signature.types.Field ×
          Ref signature.types context .field))
  | product (context : Schema signature.types)
      (left right : Ref signature.types context .field)
  | invoke (context : Schema signature.types)
      (call : signature.Call)
      (operands :
        Refs signature.types context (signature.callInputs call))
      (row : Fin (signature.callFootprint call).recurringRows)
  | assertTrue (context : Schema signature.types)
      (condition : Ref signature.types context .bit)
  | branchSelector (context : Schema signature.types)
      (condition : Ref signature.types context .bit)
  | branchGate (selected : Bool)
  | branchJoin (selected : Bool) (coordinate : Coordinate)

/-- One unowned event local to a receipt. -/
inductive Event (signature : Signature.{u}) : Type (u + 2) where
  | allocate (role : AllocationRole) (coordinate : Coordinate)
  | row (token : RowToken signature)

/-- A globally owned event.  `ordinal` is assigned monotonically within the
owner's receipt; together with the structural owner it is its unique ID. -/
structure OwnedEvent (signature : Signature.{u}) : Type (u + 2) where
  owner : Owner
  ordinal : Nat
  event : Event signature

namespace OwnedEvent

def id {signature : Signature.{u}}
    (event : OwnedEvent signature) : EventId :=
  ⟨event.owner, event.ordinal⟩

def cost {signature : Signature.{u}}
    (event : OwnedEvent signature) : Cost :=
  match event.event with
  | .allocate _ coordinate => Cost.oneColumn coordinate.ownership
  | .row _ => Cost.oneRow

def temporaryId? {signature : Signature.{u}}
    (event : OwnedEvent signature) : Option TemporaryId :=
  match event.event with
  | .allocate .temporary coordinate =>
      some ⟨event.owner, coordinate.portIndex, coordinate.coordinateIndex⟩
  | _ => none

theorem id_ne_of_owner_ne {signature : Signature.{u}}
    {left right : OwnedEvent signature}
    (different : left.owner ≠ right.owner) :
    left.id ≠ right.id := by
  intro equal
  apply different
  exact congrArg EventId.owner equal

end OwnedEvent

/-- Assign collision-free local indices to one owner's raw events. -/
def ownEventsFrom {signature : Signature.{u}}
    (owner : Owner) : Nat -> List (Event signature) ->
      List (OwnedEvent signature)
  | _, [] => []
  | ordinal, event :: tail =>
      ⟨owner, ordinal, event⟩ :: ownEventsFrom owner (ordinal + 1) tail

def ownEvents {signature : Signature.{u}}
    (owner : Owner) (events : List (Event signature)) :
    List (OwnedEvent signature) :=
  ownEventsFrom owner 0 events

@[simp] theorem ownEventsFrom_length {signature : Signature.{u}}
    (owner : Owner) (ordinal : Nat) (events : List (Event signature)) :
    (ownEventsFrom owner ordinal events).length = events.length := by
  induction events generalizing ordinal with
  | nil => rfl
  | cons _ tail inductionHypothesis =>
      simp [ownEventsFrom, inductionHypothesis]

/-- Exact cost of an owned event stream. -/
def eventsCost {signature : Signature.{u}}
    (events : List (OwnedEvent signature)) : Cost :=
  Cost.sum (events.map OwnedEvent.cost)

theorem eventsCost_append {signature : Signature.{u}}
    (left right : List (OwnedEvent signature)) :
    eventsCost (left ++ right) = eventsCost left + eventsCost right := by
  unfold eventsCost
  rw [List.map_append, Cost.sum_append]

namespace Primitive

/-- Exact SSA operands of a semantic primitive. -/
def operands {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output) :
    List (SomeRef signature.types input) :=
  match primitive with
  | .literal _ _ => []
  | .linear _ _ terms =>
      terms.map fun term => ⟨.field, term.2⟩
  | .product _ left right =>
      [⟨.field, left⟩, ⟨.field, right⟩]
  | .invoke _ references => references.toList
  | .assertTrue condition => [⟨.bit, condition⟩]

/-- Fresh logical ports produced by a primitive. -/
def producedSchema {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output) :
    Schema signature.types :=
  match primitive with
  | .literal port _ => [port]
  | .linear layout _ _ => [{ kind := .field, layout := layout }]
  | .product layout _ _ => [{ kind := .field, layout := layout }]
  | .invoke call _ => signature.callOutputs call
  | .assertTrue _ => []

theorem output_eq_produced_append_input
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output) :
    output = primitive.producedSchema ++ input := by
  cases primitive <;> rfl

/-- Fixed call-temporary layouts; all other primitives have none. -/
def temporaryLayouts {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output) : List Layout :=
  match primitive with
  | .invoke call _ => (signature.callFootprint call).temporaries
  | _ => []

/-- Rows tied to the exact semantic primitive. -/
def rowTokens {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output) :
    List (RowToken signature) :=
  match primitive with
  | .literal port value => [.literal port value]
  | .linear _ constant terms => [.linear input constant terms]
  | .product _ left right => [.product input left right]
  | .invoke call references =>
      List.ofFn fun row : Fin (signature.callFootprint call).recurringRows =>
        RowToken.invoke input call references row
  | .assertTrue condition => [.assertTrue input condition]

def rawEvents {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output) :
    List (Event signature) :=
  (schemaCoordinates primitive.producedSchema).map
      (Event.allocate .output) ++
    (layoutsCoordinates primitive.temporaryLayouts).map
      (Event.allocate .temporary) ++
    primitive.rowTokens.map Event.row

end Primitive

/-- A primitive receipt cannot omit or substitute data: every projection is
computed from its indexed primitive and structural path. -/
structure PrimitiveReceipt
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output) where
  path : OwnerPath

namespace PrimitiveReceipt

def operands {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (_receipt : PrimitiveReceipt primitive) :
    List (SomeRef signature.types input) :=
  primitive.operands

def outputs {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (_receipt : PrimitiveReceipt primitive) : List Coordinate :=
  schemaCoordinates primitive.producedSchema

def temporaries {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (_receipt : PrimitiveReceipt primitive) : List Coordinate :=
  layoutsCoordinates primitive.temporaryLayouts

def rows {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (_receipt : PrimitiveReceipt primitive) : List (RowToken signature) :=
  primitive.rowTokens

def events {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (receipt : PrimitiveReceipt primitive) :
    List (OwnedEvent signature) :=
  ownEvents (.instruction receipt.path) primitive.rawEvents

def cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (receipt : PrimitiveReceipt primitive) : Cost :=
  eventsCost receipt.events

theorem operands_exact {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (receipt : PrimitiveReceipt primitive) :
    receipt.operands = primitive.operands :=
  rfl

theorem outputs_exact {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (receipt : PrimitiveReceipt primitive) :
    receipt.outputs = schemaCoordinates primitive.producedSchema :=
  rfl

theorem temporaries_exact {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (receipt : PrimitiveReceipt primitive) :
    receipt.temporaries = layoutsCoordinates primitive.temporaryLayouts :=
  rfl

theorem rows_exact {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (receipt : PrimitiveReceipt primitive) :
    receipt.rows = primitive.rowTokens :=
  rfl

theorem cost_eq_event_cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    {primitive : Primitive signature input output}
    (receipt : PrimitiveReceipt primitive) :
    receipt.cost = eventsCost receipt.events :=
  rfl

end PrimitiveReceipt

namespace Primitive

def receiptAt {signature : Signature.{u}}
    {input output : Schema signature.types}
    (path : OwnerPath)
    (primitive : Primitive signature input output) :
    PrimitiveReceipt primitive :=
  ⟨path⟩

def cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    (path : OwnerPath)
    (primitive : Primitive signature input output) : Cost :=
  (primitive.receiptAt path).cost

end Primitive

/-- Exact, nonoptional allocation receipt for the runtime input schema. -/
structure InputReceipt
    (signature : Signature.{u})
    (schema : Schema signature.types) where
  events : List (OwnedEvent signature)
  events_exact :
    events =
      (schemaCoordinates schema).map fun coordinate =>
        { owner := .input coordinate.portIndex
          ordinal := coordinate.coordinateIndex
          event := .allocate .input coordinate }

def programInputReceipt
    (signature : Signature.{u})
    (schema : Schema signature.types) :
    InputReceipt signature schema where
  events :=
    (schemaCoordinates schema).map fun coordinate =>
      { owner := .input coordinate.portIndex
        ordinal := coordinate.coordinateIndex
        event := .allocate .input coordinate }
  events_exact := rfl

theorem input_allocations_exact
    (signature : Signature.{u})
    (schema : Schema signature.types) :
    (programInputReceipt signature schema).events =
      (schemaCoordinates schema).map fun coordinate =>
        { owner := .input coordinate.portIndex
          ordinal := coordinate.coordinateIndex
          event := .allocate .input coordinate } :=
  rfl

/-- Receipt for a branch's selector, arm gates, joined allocations, and the
two selected-arm join equations for every joined coordinate. -/
structure BranchReceipt
    {signature : Signature.{u}}
    {input : Schema signature.types}
    (condition : Ref signature.types input .bit)
    (joined : Schema signature.types) where
  path : OwnerPath

namespace BranchReceipt

def rawEvents {signature : Signature.{u}}
    {input : Schema signature.types}
    {condition : Ref signature.types input .bit}
    {joined : Schema signature.types}
    (_receipt : BranchReceipt condition joined) :
    List (Event signature) :=
  let coordinates := schemaCoordinates joined
  coordinates.map (Event.allocate .branchJoin) ++
    [.row (.branchSelector input condition),
      .row (.branchGate true),
      .row (.branchGate false)] ++
    coordinates.map (fun coordinate =>
      .row (.branchJoin true coordinate)) ++
    coordinates.map (fun coordinate =>
      .row (.branchJoin false coordinate))

def events {signature : Signature.{u}}
    {input : Schema signature.types}
    {condition : Ref signature.types input .bit}
    {joined : Schema signature.types}
    (receipt : BranchReceipt condition joined) :
    List (OwnedEvent signature) :=
  ownEvents (.branch receipt.path) receipt.rawEvents

def cost {signature : Signature.{u}}
    {input : Schema signature.types}
    {condition : Ref signature.types input .bit}
    {joined : Schema signature.types}
    (receipt : BranchReceipt condition joined) : Cost :=
  eventsCost receipt.events

theorem cost_eq_event_cost {signature : Signature.{u}}
    {input : Schema signature.types}
    {condition : Ref signature.types input .bit}
    {joined : Schema signature.types}
    (receipt : BranchReceipt condition joined) :
    receipt.cost = eventsCost receipt.events :=
  rfl

end BranchReceipt

/-- Receipt structure mirrors the complete block, including both branch arms
and the post-join continuation. -/
inductive BlockReceipt {signature : Signature.{u}} :
    {input output : Schema signature.types} ->
    Block signature input output -> Type (u + 4) where
  | yield {input output : Schema signature.types}
      {exports : Exports signature.types input output} :
      BlockReceipt (.yield exports)
  | step {input middle output : Schema signature.types}
      {primitive : Primitive signature input middle}
      {rest : Block signature middle output} :
      PrimitiveReceipt primitive ->
      BlockReceipt rest ->
      BlockReceipt (.step primitive rest)
  | branch {input joined output : Schema signature.types}
      {condition : Ref signature.types input .bit}
      {onTrue onFalse : Block signature input joined}
      {continuation : Block signature (joined ++ input) output} :
      BranchReceipt condition joined ->
      BlockReceipt onTrue ->
      BlockReceipt onFalse ->
      BlockReceipt continuation ->
      BlockReceipt (.branch condition onTrue onFalse continuation)

namespace BlockReceipt

def events {signature : Signature.{u}} :
    {input output : Schema signature.types} ->
    {block : Block signature input output} ->
    BlockReceipt block -> List (OwnedEvent signature)
  | _, _, _, .yield => []
  | _, _, _, .step head tail => head.events ++ tail.events
  | _, _, _, .branch control onTrue onFalse continuation =>
      control.events ++ onTrue.events ++ onFalse.events ++
        continuation.events

def cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    {block : Block signature input output}
    (receipt : BlockReceipt block) : Cost :=
  eventsCost receipt.events

theorem cost_eq_event_cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    {block : Block signature input output}
    (receipt : BlockReceipt block) :
    receipt.cost = eventsCost receipt.events :=
  rfl

end BlockReceipt

namespace Block

/-- Total structural receipt construction. -/
def receiptAt {signature : Signature.{u}} :
    {input output : Schema signature.types} ->
    (path : OwnerPath) ->
    (block : Block signature input output) ->
    BlockReceipt block
  | _, _, _, .yield _ => .yield
  | _, _, path, .step primitive rest =>
      .step (primitive.receiptAt path)
        (rest.receiptAt (.rest path))
  | _, _, path, .branch _ onTrue onFalse continuation =>
      .branch ⟨path⟩
        (onTrue.receiptAt (.trueArm path))
        (onFalse.receiptAt (.falseArm path))
        (continuation.receiptAt (.continuation path))

def emissions {signature : Signature.{u}}
    {input output : Schema signature.types}
    (path : OwnerPath)
    (block : Block signature input output) :
    List (OwnedEvent signature) :=
  (block.receiptAt path).events

def cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    (path : OwnerPath)
    (block : Block signature input output) : Cost :=
  (block.receiptAt path).cost

theorem cost_eq_receipt_event_cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    (path : OwnerPath)
    (block : Block signature input output) :
    block.cost path = eventsCost (block.emissions path) :=
  rfl

end Block

/-- Complete receipt: the exact input allocation receipt followed by the
structural receipt of the program body. -/
structure ProgramReceipt {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output) where
  inputs : InputReceipt signature input
  body : BlockReceipt program.body

namespace ProgramReceipt

def events {signature : Signature.{u}}
    {input output : Schema signature.types}
    {program : Program signature input output}
    (receipt : ProgramReceipt program) :
    List (OwnedEvent signature) :=
  receipt.inputs.events ++ receipt.body.events

def cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    {program : Program signature input output}
    (receipt : ProgramReceipt program) : Cost :=
  eventsCost receipt.events

end ProgramReceipt

namespace Program

def receipt {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output) :
    ProgramReceipt program where
  inputs := programInputReceipt signature input
  body := program.body.receiptAt .root

/-- The only abstract emission stream is the flattened total receipt. -/
def emissions {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output) :
    List (OwnedEvent signature) :=
  program.receipt.events

/-- Program cost is computed from its receipt, never measured afterward. -/
def cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output) : Cost :=
  program.receipt.cost

/-- Conservation/no-orphan theorem: the complete event stream is exactly the
input allocation receipt followed by the structurally complete block receipt.
Both branch arms, branch control/join, and continuation occur inside `body`. -/
theorem flattened_conservation {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output) :
    program.emissions =
      (programInputReceipt signature input).events ++
        (program.body.receiptAt .root).events :=
  rfl

theorem cost_eq_receipt_event_cost {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output) :
    program.cost = eventsCost program.emissions :=
  rfl

end Program

end Nightstream.Implementation.Lowering.Typed
