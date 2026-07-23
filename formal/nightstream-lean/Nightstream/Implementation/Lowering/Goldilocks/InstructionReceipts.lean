import Nightstream.Implementation.Lowering.Goldilocks.Compiler
import Nightstream.Implementation.Lowering.Goldilocks.PrimitiveRecipes

/-!
Contract: exact instruction receipts for the selected primitive recipes.

Owns:
- the verifier-fixed constant-one prelude receipt;
- lossless conversion of call, literal, assertion, activation, and mux
  emissions into `InstructionReceipt`;
- explicit ownership premises wherever an allocated bundle is supplied by a
  surrounding column plan;
- exact equations showing that receipt rows and allocations are the recipe's
  complete physical emission.

Does not own: column allocation policy, program traversal, call semantics,
normal-form selection, Rust emission, or generated artifacts.

There is no generic extra-row or glue field in this bridge.  In particular,
the two branch-activation equations are two separately owned instruction
receipts because their physical owners are distinct.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

namespace InstructionReceipt

/-- The constant-one coordinate is a real public allocation.  Its value is
fixed by `Encoding.PhysicalSatisfies`, so this receipt emits no circular row
attempting to prove one from itself. -/
def prelude : InstructionReceipt where
  owner := .prelude
  kind := .prelude
  allocations := preludeColumns
  rows := []
  allocationsOwned := by
    intro column member
    simp only [preludeColumns, List.mem_singleton] at member
    subst column
    rfl
  rowsOwned := by
    simp

@[simp] theorem prelude_allocations : prelude.allocations = preludeColumns :=
  rfl

@[simp] theorem prelude_rows : prelude.rows = [] :=
  rfl

@[simp] theorem prelude_columnIds : prelude.columnIds = [oneColumn] :=
  rfl

/-- One logical input slot owns exactly its declared physical coordinate
bundle and emits no rows.  The surrounding input column plan supplies the
owner certificate because slot indices are determined by schema position. -/
def ofInputSlot
    {types : TypeSystem.{u}}
    {port : Port types}
    (slot : Nat)
    (bundle : Bundle port)
    (allocationsOwned :
      ∀ column, column ∈ bundleOwnedColumns port bundle ->
        column.id.owner = .typed (.input slot)) :
    InstructionReceipt where
  owner := .typed (.input slot)
  kind := .input
  allocations := bundleOwnedColumns port bundle
  rows := []
  allocationsOwned := allocationsOwned
  rowsOwned := by
    simp

@[simp] theorem ofInputSlot_allocations
    {types : TypeSystem.{u}}
    {port : Port types}
    (slot : Nat)
    (bundle : Bundle port)
    (allocationsOwned :
      ∀ column, column ∈ bundleOwnedColumns port bundle ->
        column.id.owner = .typed (.input slot)) :
    (ofInputSlot slot bundle allocationsOwned).allocations =
      bundleOwnedColumns port bundle :=
  rfl

@[simp] theorem ofInputSlot_rows
    {types : TypeSystem.{u}}
    {port : Port types}
    (slot : Nat)
    (bundle : Bundle port)
    (allocationsOwned :
      ∀ column, column ∈ bundleOwnedColumns port bundle ->
        column.id.owner = .typed (.input slot)) :
    (ofInputSlot slot bundle allocationsOwned).rows = [] :=
  rfl

/-- Convert one certified call occurrence without changing either its fresh
allocation list or its row list.  The surrounding column plan must prove that
both lists use the instruction's structural owner. -/
def ofCall
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    InstructionReceipt where
  owner := frame.owner
  kind := .call
  allocations := frame.allocations
  rows := recipe.rows frame
  allocationsOwned := frame.allocationsOwned
  rowsOwned := recipe.rowsOwned frame

@[simp] theorem ofCall_allocations
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    (ofCall recipe frame).allocations =
      frame.allocations :=
  rfl

@[simp] theorem ofCall_rows
    {signature : Signature.{u}}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    (ofCall recipe frame).rows =
      recipe.rows frame :=
  rfl

/-- Literal outputs are allocated and pinned by the same instruction. -/
def ofLiteral
    {alpha : Type u}
    {codec : Codec alpha}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout)
    (allocationsOwned :
      ∀ column, column ∈ recipe.output.columns ->
        column.id.owner = recipe.owner)
    (rowsOwned :
      ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner) :
    InstructionReceipt where
  owner := recipe.owner
  kind := .literal
  allocations := recipe.output.columns
  rows := recipe.rows
  allocationsOwned := allocationsOwned
  rowsOwned := rowsOwned

@[simp] theorem ofLiteral_allocations
    {alpha : Type u}
    {codec : Codec alpha}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout)
    (allocationsOwned :
      ∀ column, column ∈ recipe.output.columns ->
        column.id.owner = recipe.owner)
    (rowsOwned :
      ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner) :
    (ofLiteral recipe allocationsOwned rowsOwned).allocations =
      recipe.output.columns :=
  rfl

@[simp] theorem ofLiteral_rows
    {alpha : Type u}
    {codec : Codec alpha}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout)
    (allocationsOwned :
      ∀ column, column ∈ recipe.output.columns ->
        column.id.owner = recipe.owner)
    (rowsOwned :
      ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner) :
    (ofLiteral recipe allocationsOwned rowsOwned).rows = recipe.rows :=
  rfl

/-- Assertions allocate nothing and emit exactly their one gated row. -/
def ofAssertion (recipe : BoolAssertRecipe) : InstructionReceipt where
  owner := recipe.owner
  kind := .assertion
  allocations := []
  rows := recipe.rows
  allocationsOwned := by
    simp
  rowsOwned := by
    intro row member
    simp only [BoolAssertRecipe.rows, List.mem_singleton] at member
    subst row
    rfl

@[simp] theorem ofAssertion_allocations (recipe : BoolAssertRecipe) :
    (ofAssertion recipe).allocations = [] :=
  rfl

@[simp] theorem ofAssertion_rows (recipe : BoolAssertRecipe) :
    (ofAssertion recipe).rows = recipe.rows :=
  rfl

/-- The selected-true activation wire and its defining equation have one
physical owner. -/
def ofTrueActivation
    (recipe : BranchActivationRecipe)
    (columnOwner :
      recipe.onTrue.owner = .branchActivation recipe.path true) :
    InstructionReceipt where
  owner := .branchActivation recipe.path true
  kind := .branchControl
  allocations :=
    [{ id := recipe.onTrue, ownership := .auxiliaryColumn }]
  rows :=
    [{ id :=
         { owner := .branchActivation recipe.path true
           ordinal := 0 }
       row :=
         (CanonicalRow.activateTrue
            recipe.onTrue recipe.active recipe.selector).row }]
  allocationsOwned := by
    intro column member
    simp only [List.mem_singleton] at member
    subst column
    exact columnOwner
  rowsOwned := by
    intro row member
    simp only [List.mem_singleton] at member
    subst row
    rfl

/-- The selected-false activation wire and its defining equation have one
physical owner. -/
def ofFalseActivation
    (recipe : BranchActivationRecipe)
    (columnOwner :
      recipe.onFalse.owner = .branchActivation recipe.path false) :
    InstructionReceipt where
  owner := .branchActivation recipe.path false
  kind := .branchControl
  allocations :=
    [{ id := recipe.onFalse, ownership := .auxiliaryColumn }]
  rows :=
    [{ id :=
         { owner := .branchActivation recipe.path false
           ordinal := 0 }
       row :=
         (CanonicalRow.activateFalse
            recipe.one recipe.onFalse recipe.active recipe.selector).row }]
  allocationsOwned := by
    intro column member
    simp only [List.mem_singleton] at member
    subst column
    exact columnOwner
  rowsOwned := by
    intro row member
    simp only [List.mem_singleton] at member
    subst row
    rfl

/-- Splitting activation ownership does not alter the recipe's exact row
stream. -/
theorem activation_rows_conserved
    (recipe : BranchActivationRecipe)
    (trueOwner :
      recipe.onTrue.owner = .branchActivation recipe.path true)
    (falseOwner :
      recipe.onFalse.owner = .branchActivation recipe.path false) :
    (ofTrueActivation recipe trueOwner).rows ++
        (ofFalseActivation recipe falseOwner).rows =
      recipe.rows :=
  rfl

/-- A branch join allocates the joined coordinates and emits exactly one mux
row per joined coordinate. -/
def ofMux
    {layout : Layout}
    (recipe : MuxRecipe layout)
    (allocationsOwned :
      ∀ column, column ∈ recipe.joined.columns ->
        column.id.owner = recipe.owner)
    (rowsOwned :
      ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner) :
    InstructionReceipt where
  owner := recipe.owner
  kind := .branchJoin
  allocations := recipe.joined.columns
  rows := recipe.rows
  allocationsOwned := allocationsOwned
  rowsOwned := rowsOwned

@[simp] theorem ofMux_allocations
    {layout : Layout}
    (recipe : MuxRecipe layout)
    (allocationsOwned :
      ∀ column, column ∈ recipe.joined.columns ->
        column.id.owner = recipe.owner)
    (rowsOwned :
      ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner) :
    (ofMux recipe allocationsOwned rowsOwned).allocations =
      recipe.joined.columns :=
  rfl

@[simp] theorem ofMux_rows
    {layout : Layout}
    (recipe : MuxRecipe layout)
    (allocationsOwned :
      ∀ column, column ∈ recipe.joined.columns ->
        column.id.owner = recipe.owner)
    (rowsOwned :
      ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner) :
    (ofMux recipe allocationsOwned rowsOwned).rows = recipe.rows :=
  rfl

end InstructionReceipt

end Nightstream.Implementation.Lowering.Goldilocks
