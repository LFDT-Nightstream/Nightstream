import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: artifact-independent typed values, static runtime schemas, exact SSA
references, and closed deterministic call signatures for the lowering IR.

Owns:
- the semantic field, Boolean, and protocol-data sorts;
- heterogeneous runtime values and call operands;
- multi-coordinate ownership layouts for every logical port;
- exact addressed references into a static schema;
- N-ary deterministic partial calls with fixed output and temporary layouts.

Does not own: a physical R1CS encoding, Rust layout, generated artifact, or a
caller-supplied acceptance proposition.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Typed

universe u

/-- Artifact-independent semantic types available to a lowering vocabulary.
Protocol data are named by `Data`; the IR never reconstructs those types from
Rust columns. -/
structure TypeSystem where
  Field : Type u
  zero : Field
  add : Field -> Field -> Field
  mul : Field -> Field -> Field
  Bit : Type u
  bitValue : Bit -> Bool
  Data : Type u
  dataValue : Data -> Type u

namespace TypeSystem

/-- A reusable semantic sort.  Call outputs use the same kinds as runtime
inputs, so the result of one call can be an operand of a later call. -/
inductive Kind (types : TypeSystem.{u}) where
  | field
  | bit
  | data (tag : types.Data)

/-- The Lean value denoted by a semantic sort. -/
def Value (types : TypeSystem.{u}) : Kind types -> Type u
  | .field => types.Field
  | .bit => types.Bit
  | .data tag => types.dataValue tag

end TypeSystem

/-- A heterogeneous vector whose shape is a type-level list. -/
inductive HVec {α : Type u} (value : α -> Type u) :
    List α -> Type (u + 1) where
  | nil : HVec value []
  | cons {head : α} {tail : List α} :
      value head -> HVec value tail -> HVec value (head :: tail)

namespace HVec

def head {α : Type u} {value : α -> Type u} {kind : α} {tail : List α} :
    HVec value (kind :: tail) -> value kind :=
  fun values => match values with
    | .cons head _ => head

def tail {α : Type u} {value : α -> Type u} {kind : α} {tail : List α} :
    HVec value (kind :: tail) -> HVec value tail :=
  fun values => match values with
    | .cons _ tail => tail

/-- Concatenate heterogeneous values in the same order as `List.append`. -/
def append {α : Type u} {value : α -> Type u} :
    {left right : List α} ->
    HVec value left -> HVec value right -> HVec value (left ++ right)
  | [], _, _, rightValues => rightValues
  | _ :: _, _, .cons head tail, rightValues =>
      .cons head (append tail rightValues)

end HVec

/-- Physical ownership of the coordinates allocated for one logical value.
Mixed layouts are explicit: a structured value is not silently counted as one
column with one ownership class. -/
structure Layout where
  owners : List Ownership
deriving DecidableEq, Repr

namespace Layout

/-- Definitional column cost of a logical value layout. -/
def cost (layout : Layout) : Cost :=
  Cost.sum (layout.owners.map Cost.oneColumn)

@[simp] theorem cost_empty :
    (Layout.mk []).cost = Cost.zero :=
  rfl

end Layout

/-- A statically allocated logical port. -/
structure Port (types : TypeSystem.{u}) where
  kind : types.Kind
  layout : Layout

namespace Port

def cost {types : TypeSystem.{u}} (port : Port types) : Cost :=
  port.layout.cost

end Port

/-- Runtime schemas and SSA contexts are ordered lists of statically typed,
statically owned logical ports. -/
abbrev Schema (types : TypeSystem.{u}) := List (Port types)

namespace Schema

/-- Runtime values for an exact static schema. -/
abbrev Values (types : TypeSystem.{u}) (schema : Schema types) :=
  HVec (fun port => types.Value port.kind) schema

/-- Definitional allocation cost of a static schema. -/
def cost {types : TypeSystem.{u}} (schema : Schema types) : Cost :=
  Cost.sum (schema.map Port.cost)

end Schema

/-- A typed de Bruijn reference into an SSA context.  Unlike port metadata,
the constructor path identifies which occurrence is used when several ports
have the same sort and layout. -/
inductive Ref (types : TypeSystem.{u}) :
    Schema types -> types.Kind -> Type u where
  | here (port : Port types) {tail : Schema types} :
      Ref types (port :: tail) port.kind
  | there {port : Port types} {tail : Schema types}
      {kind : types.Kind} :
      Ref types tail kind -> Ref types (port :: tail) kind

namespace Ref

/-- Zero-based address of a reference in its current context. -/
def address {types : TypeSystem.{u}} {schema : Schema types}
    {kind : types.Kind} : Ref types schema kind -> Nat
  | .here _ => 0
  | .there reference => reference.address + 1

/-- Exact port selected by a reference, including its coordinate ownership
layout. -/
def port {types : TypeSystem.{u}} {schema : Schema types}
    {kind : types.Kind} : Ref types schema kind -> Port types
  | .here selected => selected
  | .there reference => reference.port

theorem port_sort {types : TypeSystem.{u}} {schema : Schema types}
    {kind : types.Kind} (reference : Ref types schema kind) :
    reference.port.kind = kind := by
  induction reference with
  | here => rfl
  | there _ inductionHypothesis => exact inductionHypothesis

theorem address_lt {types : TypeSystem.{u}} {schema : Schema types}
    {kind : types.Kind} (reference : Ref types schema kind) :
    reference.address < schema.length := by
  induction reference with
  | here => simp [address]
  | there _ inductionHypothesis =>
      simp only [address, List.length_cons]
      exact Nat.add_lt_add_right inductionHypothesis 1

/-- Read the value at the exact addressed reference. -/
def get {types : TypeSystem.{u}} {schema : Schema types}
    {kind : types.Kind} :
    (reference : Ref types schema kind) ->
    Schema.Values types schema -> types.Value kind
  | .here _, .cons value _ => value
  | .there reference, .cons _ tail => reference.get tail

/-- Public reduction rule for reading the head of a typed context.  Keeping
this equation explicit prevents callers from unfolding an entire dependent
program merely to reduce one addressed lookup. -/
@[simp] theorem get_here
    {types : TypeSystem.{u}}
    {port : Port types}
    {tail : Schema types}
    (value : types.Value port.kind)
    (values : Schema.Values types tail) :
    (Ref.here port : Ref types (port :: tail) port.kind).get
        (.cons value values) = value :=
  rfl

/-- Public reduction rule for reading through one context entry. -/
@[simp] theorem get_there
    {types : TypeSystem.{u}}
    {port : Port types}
    {tail : Schema types}
    {kind : types.Kind}
    (reference : Ref types tail kind)
    (value : types.Value port.kind)
    (values : Schema.Values types tail) :
    (Ref.there reference : Ref types (port :: tail) kind).get
        (.cons value values) = reference.get values :=
  rfl

end Ref

/-- Existential packaging used by receipts.  It erases only the result sort;
the complete addressed reference remains present. -/
structure SomeRef (types : TypeSystem.{u}) (schema : Schema types) where
  kind : types.Kind
  reference : Ref types schema kind

namespace SomeRef

def address {types : TypeSystem.{u}} {schema : Schema types}
    (reference : SomeRef types schema) : Nat :=
  reference.reference.address

def port {types : TypeSystem.{u}} {schema : Schema types}
    (reference : SomeRef types schema) : Port types :=
  reference.reference.port

end SomeRef

/-- Exact N-ary operands drawn from one SSA context. -/
inductive Refs (types : TypeSystem.{u}) (schema : Schema types) :
    List types.Kind -> Type (u + 1) where
  | nil : Refs types schema []
  | cons {kind : types.Kind} {tail : List types.Kind} :
      Ref types schema kind ->
      Refs types schema tail ->
      Refs types schema (kind :: tail)

namespace Refs

/-- Preserve the identity and order of every N-ary operand for receipts. -/
def toList {types : TypeSystem.{u}} {schema : Schema types} :
    {sorts : List types.Kind} ->
    Refs types schema sorts -> List (SomeRef types schema)
  | [], _ => []
  | kind :: _, .cons reference tail =>
      ⟨kind, reference⟩ :: toList tail

/-- Evaluate N-ary operands in their declared order. -/
def get {types : TypeSystem.{u}} {schema : Schema types} :
    {sorts : List types.Kind} ->
    Refs types schema sorts ->
    Schema.Values types schema ->
    HVec types.Value sorts
  | [], _, _ => .nil
  | _ :: _, .cons reference tail, values =>
      .cons (reference.get values) (get tail values)

end Refs

/-- Static resources intrinsic to one call, excluding its declared output
ports.  A parameterized call shape must be represented by a distinct `Call`
value, so its footprint cannot change at emission time. -/
structure CallFootprint where
  recurringRows : Nat
  temporaries : List Layout
deriving DecidableEq, Repr

namespace CallFootprint

def cost (footprint : CallFootprint) : Cost :=
  ⟨footprint.recurringRows, 0, 0, 0⟩ +
    Cost.sum (footprint.temporaries.map Layout.cost)

end CallFootprint

/-- Closed typed call vocabulary.  Inputs are separate authoritative SSA
operands, outputs are ordinary reusable ports, evaluation is deterministic
and partial, and the abstract resource footprint is fixed by the call itself.
No field accepts a semantic proposition from a caller. -/
structure Signature where
  types : TypeSystem
  Call : Type u
  callInputs : Call -> List types.Kind
  callOutputs : Call -> Schema types
  callEval : (call : Call) ->
    HVec types.Value (callInputs call) ->
    Option (Schema.Values types (callOutputs call))
  callFootprint : Call -> CallFootprint

namespace Signature

/-- Intrinsic call cost: fixed recurring/temporary resources plus the exact
multi-coordinate ownership cost of every result port. -/
def callCost (signature : Signature) (call : signature.Call) : Cost :=
  (signature.callFootprint call).cost +
    (signature.callOutputs call).cost

end Signature

end Nightstream.Implementation.Lowering.Typed
