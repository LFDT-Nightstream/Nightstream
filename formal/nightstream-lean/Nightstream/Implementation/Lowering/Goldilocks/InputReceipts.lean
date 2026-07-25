import Nightstream.Implementation.Lowering.Goldilocks.InstructionReceipts
import Nightstream.Implementation.Lowering.Goldilocks.BundleBridge

/-!
Contract: canonical physical receipts for the exact typed input schema.

Owns:
- one row-free input receipt per schema port, in schema order;
- zero-based input-slot owners and the matching `inputColumns` allocation;
- local/global identity uniqueness for those receipts;
- well-scoping of the prelude followed immediately by the input receipts.

Does not own: semantic input values, source-program traversal, call outputs,
arbitrary receipt lists, Rust emission, or generated artifacts.

Emits constraints: no rows.  Its allocated columns are definitionally the
columns selected by `inputColumns`.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

namespace InputReceipts

/-- The exact bundle allocated for one input slot. -/
private def inputBundle
    {types : TypeSystem.{u}}
    (slot : Nat)
    (port : Port types) : Bundle port where
  column coordinate :=
    { owner := .typed (.input slot)
      bundleIndex := slot
      coordinateIndex := coordinate.val }

private theorem inputBundle_owned
    {types : TypeSystem.{u}}
    (slot : Nat)
    (port : Port types) :
    ∀ column,
      column ∈ bundleOwnedColumns port (inputBundle slot port) ->
        column.id.owner = .typed (.input slot) := by
  intro column member
  rw [bundleOwnedColumns, List.mem_ofFn] at member
  rcases member with ⟨coordinate, rfl⟩
  rfl

/-- Structural recursion over the schema, carrying the next input-slot
index. -/
private def receiptsFrom
    {types : TypeSystem.{u}}
    (slot : Nat) :
    (schema : Schema types) -> List InstructionReceipt
  | [] => []
  | port :: tail =>
      InstructionReceipt.ofInputSlot slot (inputBundle slot port)
        (inputBundle_owned slot port) ::
      receiptsFrom (slot + 1) tail

/-- One canonical receipt for each port of the exact input schema. -/
def receipts
    {types : TypeSystem.{u}}
    (schema : Schema types) : List InstructionReceipt :=
  receiptsFrom 0 schema

private theorem receiptsFrom_length
    {types : TypeSystem.{u}}
    (slot : Nat)
    (schema : Schema types) :
    (receiptsFrom slot schema).length = schema.length := by
  induction schema generalizing slot with
  | nil => rfl
  | cons port tail inductionHypothesis =>
      simp only [receiptsFrom, List.length_cons, inductionHypothesis]

/-- There is exactly one receipt per input port. -/
@[simp] theorem receipts_length
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    (receipts schema).length = schema.length :=
  receiptsFrom_length 0 schema

private theorem receiptsFrom_owners
    {types : TypeSystem.{u}}
    (slot : Nat)
    (schema : Schema types) :
    (receiptsFrom slot schema).map (fun receipt => receipt.owner) =
      (List.range' slot schema.length).map
        (fun index => PhysicalOwner.typed (.input index)) := by
  induction schema generalizing slot with
  | nil => rfl
  | cons port tail inductionHypothesis =>
      simp only [receiptsFrom, List.map_cons,
        InstructionReceipt.ofInputSlot, List.length_cons,
        List.range'_succ, inductionHypothesis]

/-- Receipt owners are exactly input slots `0, ..., schema.length - 1` in
schema order. -/
theorem owners_exact
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    (receipts schema).map (fun receipt => receipt.owner) =
      (List.range schema.length).map
        (fun index => PhysicalOwner.typed (.input index)) := by
  simpa only [receipts, List.range_eq_range'] using
    receiptsFrom_owners 0 schema

/-- No two input receipts reuse a structural owner. -/
theorem ownersNodup
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    ((receipts schema).map fun receipt => receipt.owner).Nodup := by
  rw [owners_exact]
  exact (List.nodup_range (n := schema.length)).map
    (fun index => PhysicalOwner.typed (.input index)) (by
      intro first second different equal
      apply different
      exact Owner.input.inj (PhysicalOwner.typed.inj equal))

/-- The prelude owner is also disjoint from every canonical input owner. -/
theorem ownersNodupAfterPrelude
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    ((InstructionReceipt.prelude :: receipts schema).map
      fun receipt => receipt.owner).Nodup := by
  simp only [List.map_cons, List.nodup_cons,
    InstructionReceipt.prelude, ownersNodup]
  rw [owners_exact]
  simp

private theorem allocationsFrom_exact
    {types : TypeSystem.{u}}
    (slot : Nat)
    (schema : Schema types) :
    (receiptsFrom slot schema).flatMap
        (fun receipt => receipt.allocations) =
      schemaOwnedColumns
        (allocateSchemaFrom
          (fun index => PhysicalOwner.typed (.input index))
          slot schema) := by
  induction schema generalizing slot with
  | nil => rfl
  | cons port tail inductionHypothesis =>
      simp only [receiptsFrom, List.flatMap_cons,
        InstructionReceipt.ofInputSlot_allocations,
        allocateSchemaFrom, schemaOwnedColumns,
        inputBundle, inductionHypothesis]

/-- Flattening the input receipts yields exactly the canonical input column
plan, with no additional allocation. -/
theorem allocations_exact
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    (receipts schema).flatMap (fun receipt => receipt.allocations) =
      schemaOwnedColumns (inputColumns schema) := by
  simpa only [receipts, inputColumns, allocateSchema] using
    allocationsFrom_exact 0 schema

private theorem flatMap_columnIds_eq_allocation_ids
    (receipts : List InstructionReceipt) :
    receipts.flatMap InstructionReceipt.columnIds =
      (receipts.flatMap fun receipt => receipt.allocations).map
        (fun column => column.id) := by
  induction receipts with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [InstructionReceipt.columnIds, inductionHypothesis,
        List.map_append]

/-- Flattening input receipt identities yields exactly the canonical typed
input context identities in the same port/coordinate order. -/
theorem columnIds_exact
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    (receipts schema).flatMap InstructionReceipt.columnIds =
      (inputColumns schema).toSchemaBundles.ids := by
  rw [flatMap_columnIds_eq_allocation_ids, allocations_exact,
    ← Columns.toSchemaBundles_columns]
  rfl

private theorem rowsFrom_empty
    {types : TypeSystem.{u}}
    (slot : Nat)
    (schema : Schema types) :
    (receiptsFrom slot schema).flatMap
        (fun receipt => receipt.rows) = [] := by
  induction schema generalizing slot with
  | nil => rfl
  | cons port tail inductionHypothesis =>
      simp only [receiptsFrom, List.flatMap_cons,
        InstructionReceipt.ofInputSlot_rows,
        List.nil_append, inductionHypothesis]

/-- Input receipts emit no physical rows. -/
theorem rows_empty
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    (receipts schema).flatMap (fun receipt => receipt.rows) = [] :=
  rowsFrom_empty 0 schema

private theorem nodup_ofFn_of_injective
    {alpha : Type} :
    ∀ {n : Nat}
      (function : Fin n -> alpha),
      Function.Injective function ->
      (List.ofFn function).Nodup
  | 0, function, injective => by
      simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal =>
            Fin.succ_inj.mp (injective equal))

private theorem inputBundle_columnIds_nodup
    {types : TypeSystem.{u}}
    (slot : Nat)
    (port : Port types) :
    ((bundleOwnedColumns port (inputBundle slot port)).map
      fun column => column.id).Nodup := by
  rw [bundleOwnedColumns, List.map_ofFn]
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  exact congrArg (fun id : ColumnId => id.coordinateIndex) equal

private theorem localColumnIdsNodupFrom
    {types : TypeSystem.{u}}
    (slot : Nat)
    (schema : Schema types) :
    ∀ receipt, receipt ∈ receiptsFrom slot schema ->
      receipt.columnIds.Nodup := by
  induction schema generalizing slot with
  | nil =>
      intro receipt member
      simp [receiptsFrom] at member
  | cons port tail inductionHypothesis =>
      intro receipt member
      rcases List.mem_cons.mp member with equal | tailMember
      · subst receipt
        exact inputBundle_columnIds_nodup slot port
      · exact inductionHypothesis (slot + 1) receipt tailMember

/-- Every canonical input receipt has locally unique physical column
identities. -/
theorem localColumnIdsNodup
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    ∀ receipt, receipt ∈ receipts schema ->
      receipt.columnIds.Nodup :=
  localColumnIdsNodupFrom 0 schema

private theorem localRowIdsNodupFrom
    {types : TypeSystem.{u}}
    (slot : Nat)
    (schema : Schema types) :
    ∀ receipt, receipt ∈ receiptsFrom slot schema ->
      receipt.rowIds.Nodup := by
  induction schema generalizing slot with
  | nil =>
      intro receipt member
      simp [receiptsFrom] at member
  | cons port tail inductionHypothesis =>
      intro receipt member
      rcases List.mem_cons.mp member with equal | tailMember
      · subst receipt
        exact List.nodup_nil
      · exact inductionHypothesis (slot + 1) receipt tailMember

/-- Every canonical input receipt has the empty, hence unique, row-ID list. -/
theorem localRowIdsNodup
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    ∀ receipt, receipt ∈ receipts schema ->
      receipt.rowIds.Nodup :=
  localRowIdsNodupFrom 0 schema

private theorem receiptsFrom_wellScoped
    {types : TypeSystem.{u}}
    (slot : Nat)
    (schema : Schema types)
    (available : List ColumnId) :
    ReceiptsWellScoped available (receiptsFrom slot schema) := by
  induction schema generalizing slot available with
  | nil =>
      trivial
  | cons port tail inductionHypothesis =>
      constructor
      · intro column member
        simp [InstructionReceipt.referencedColumns,
          InstructionReceipt.ofInputSlot_rows] at member
      · exact inductionHypothesis (slot + 1) _

/-- The verifier prelude followed immediately by all input receipts is
well-scoped from the empty allocation context. -/
theorem wellScopedAfterPrelude
    {types : TypeSystem.{u}}
    (schema : Schema types) :
    ReceiptsWellScoped []
      (InstructionReceipt.prelude :: receipts schema) := by
  constructor
  · intro column member
    simp [InstructionReceipt.referencedColumns,
      InstructionReceipt.prelude_rows] at member
  · exact receiptsFrom_wellScoped 0 schema _

end InputReceipts

end Nightstream.Implementation.Lowering.Goldilocks
