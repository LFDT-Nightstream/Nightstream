import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPrimitivePlan

/-!
Contract: canonical branch-control and fixed-one join receipts.

Owns:
- the two canonical activation columns and their separately owned receipts;
- the one-port coordinate-wise mux join used by the fixed-one Step program;
- the mandatory empty join receipt used by the fixed-one Terminal program;
- local identity uniqueness and receipt scoping from explicit earlier inputs.

Does not own: branch-arm traversal, whole-program receipt order, source-owner
alignment, semantic branch selection, Rust artifacts, or arbitrary multi-port
joins.

Every physical identity is derived from the branch path.  The only supplied
column identities are already allocated parent controls or exact arm bundles.

Emits constraints: two activation rows, one mux row per joined coordinate,
and no rows for the mandatory empty Terminal join.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks

universe u

namespace CanonicalBranchPlan

/-- Canonical branch activation recipe at one structural branch path. -/
def activationRecipe
    (path : OwnerPath)
    (one active selector : ColumnId) :
    BranchActivationRecipe where
  path := path
  one := one
  active := active
  selector := selector
  onTrue := activationColumn path true
  onFalse := activationColumn path false

def trueActivationReceipt
    (path : OwnerPath)
    (one active selector : ColumnId) :
    InstructionReceipt :=
  InstructionReceipt.ofTrueActivation
    (activationRecipe path one active selector) rfl

def falseActivationReceipt
    (path : OwnerPath)
    (one active selector : ColumnId) :
    InstructionReceipt :=
  InstructionReceipt.ofFalseActivation
    (activationRecipe path one active selector) rfl

@[simp] theorem trueActivationReceipt_owner
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (trueActivationReceipt path one active selector).owner =
      .branchActivation path true :=
  rfl

@[simp] theorem falseActivationReceipt_owner
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (falseActivationReceipt path one active selector).owner =
      .branchActivation path false :=
  rfl

theorem activation_rows_conserved
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (trueActivationReceipt path one active selector).rows ++
        (falseActivationReceipt path one active selector).rows =
      (activationRecipe path one active selector).rows :=
  rfl

theorem trueActivationReceipt_columnIdsNodup
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (trueActivationReceipt path one active selector).columnIds.Nodup := by
  simp [trueActivationReceipt, InstructionReceipt.columnIds,
    InstructionReceipt.ofTrueActivation]

theorem falseActivationReceipt_columnIdsNodup
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (falseActivationReceipt path one active selector).columnIds.Nodup := by
  simp [falseActivationReceipt, InstructionReceipt.columnIds,
    InstructionReceipt.ofFalseActivation]

theorem trueActivationReceipt_rowIdsNodup
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (trueActivationReceipt path one active selector).rowIds.Nodup := by
  simp [trueActivationReceipt, InstructionReceipt.rowIds,
    InstructionReceipt.ofTrueActivation]

theorem falseActivationReceipt_rowIdsNodup
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (falseActivationReceipt path one active selector).rowIds.Nodup := by
  simp [falseActivationReceipt, InstructionReceipt.rowIds,
    InstructionReceipt.ofFalseActivation]

/-- The three pre-existing branch-control coordinates. -/
def ControlsAvailable
    (one active selector : ColumnId)
    (available : List ColumnId) : Prop :=
  one ∈ available ∧ active ∈ available ∧ selector ∈ available

theorem trueActivationReceipt_wellScoped
    (path : OwnerPath)
    (one active selector : ColumnId)
    (available : List ColumnId)
    (controls : ControlsAvailable one active selector available) :
    (trueActivationReceipt path one active selector).WellScopedAfter
      available := by
  intro column member
  simp [trueActivationReceipt, InstructionReceipt.referencedColumns,
    InstructionReceipt.rowColumns, InstructionReceipt.ofTrueActivation,
    CanonicalRow.row, Goldilocks.singleton] at member
  rcases member with equal | equal | equal
  · exact Or.inl (equal ▸ controls.2.1)
  · exact Or.inl (equal ▸ controls.2.2)
  · right
    unfold InstructionReceipt.columnIds
    apply List.mem_map.mpr
    exact ⟨
      { id := (activationRecipe path one active selector).onTrue
        ownership := .auxiliaryColumn },
      List.mem_singleton.mpr rfl,
      equal.symm⟩

theorem falseActivationReceipt_wellScoped
    (path : OwnerPath)
    (one active selector : ColumnId)
    (available : List ColumnId)
    (controls : ControlsAvailable one active selector available) :
    (falseActivationReceipt path one active selector).WellScopedAfter
      available := by
  intro column member
  simp [falseActivationReceipt, InstructionReceipt.referencedColumns,
    InstructionReceipt.rowColumns, InstructionReceipt.ofFalseActivation,
    CanonicalRow.row, Goldilocks.singleton, oneMinus] at member
  rcases member with equal | equal | equal | equal
  · exact Or.inl (equal ▸ controls.2.1)
  · exact Or.inl (equal ▸ controls.1)
  · exact Or.inl (equal ▸ controls.2.2)
  · right
    unfold InstructionReceipt.columnIds
    apply List.mem_map.mpr
    exact ⟨
      { id := (activationRecipe path one active selector).onFalse
        ownership := .auxiliaryColumn },
      List.mem_singleton.mpr rfl,
      equal.symm⟩

private theorem branch_head_owned
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (port : Port types) :
    ∀ column,
      column ∈
          (HVec.head
            (branchJoinColumns path [port])).toColumnBundle.columns ->
        column.id.owner = .typed (.branch path) := by
  intro column member
  simp only [Bundle.toColumnBundle_columns, branchJoinColumns,
    allocateSchema, allocateSchemaFrom, bundleOwnedColumns,
    List.mem_ofFn] at member
  rcases member with ⟨coordinate, rfl⟩
  rfl

private theorem branch_head_ids_nodup
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (port : Port types) :
    (HVec.head
      (branchJoinColumns path [port])).toColumnBundle.ids.Nodup := by
  have all :=
    ColumnPlan.allocateSchemaFrom_ids_nodup
      (fun _ => .typed (.branch path)) 0 [port]
  simpa [branchJoinColumns, allocateSchema, allocateSchemaFrom,
    ColumnPlan.schemaColumnIds, schemaOwnedColumns,
    ColumnBundle.ids, Bundle.toColumnBundle_columns] using all

/-- Canonical one-port mux recipe.  This is the only nonempty join shape in
the fixed-one verifier. -/
def onePortJoinRecipe
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (selector : ColumnId)
    (port : Port types)
    (onTrue onFalse : Bundle port) :
    MuxRecipe port.layout where
  owner := .typed (.branch path)
  firstOrdinal := 0
  selector := selector
  joined :=
    (HVec.head (branchJoinColumns path [port])).toColumnBundle
  onTrue := onTrue.toColumnBundle
  onFalse := onFalse.toColumnBundle

/-- Exact receipt for the selected one-port join. -/
def onePortJoinReceipt
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (selector : ColumnId)
    (port : Port types)
    (onTrue onFalse : Bundle port) :
    InstructionReceipt :=
  InstructionReceipt.ofMux
    (onePortJoinRecipe path selector port onTrue onFalse)
    (branch_head_owned path port)
    (MuxRecipe.rows_owned _)

@[simp] theorem onePortJoinReceipt_owner
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (selector : ColumnId)
    (port : Port types)
    (onTrue onFalse : Bundle port) :
    (onePortJoinReceipt path selector port onTrue onFalse).owner =
      .typed (.branch path) :=
  rfl

theorem onePortJoinReceipt_columnIdsNodup
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (selector : ColumnId)
    (port : Port types)
    (onTrue onFalse : Bundle port) :
    (onePortJoinReceipt path selector port onTrue onFalse).columnIds.Nodup := by
  simpa [onePortJoinReceipt, InstructionReceipt.columnIds,
    InstructionReceipt.ofMux, onePortJoinRecipe] using
      branch_head_ids_nodup path port

theorem onePortJoinReceipt_rowIdsNodup
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (selector : ColumnId)
    (port : Port types)
    (onTrue onFalse : Bundle port) :
    (onePortJoinReceipt path selector port onTrue onFalse).rowIds.Nodup := by
  simpa [onePortJoinReceipt, InstructionReceipt.rowIds,
    InstructionReceipt.ofMux] using
      MuxRecipe.row_ids_nodup
        (onePortJoinRecipe path selector port onTrue onFalse)

def JoinInputsAvailable
    {types : TypeSystem.{u}}
    {port : Port types}
    (selector : ColumnId)
    (onTrue onFalse : Bundle port)
    (available : List ColumnId) : Prop :=
  ∀ column,
    column ∈
        [selector] ++ onTrue.toColumnBundle.ids ++
          onFalse.toColumnBundle.ids ->
      column ∈ available

theorem onePortJoinReceipt_wellScoped
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (selector : ColumnId)
    (port : Port types)
    (onTrue onFalse : Bundle port)
    (available : List ColumnId)
    (inputs :
      JoinInputsAvailable selector onTrue onFalse available) :
    (onePortJoinReceipt path selector port onTrue onFalse).WellScopedAfter
      available := by
  intro column member
  rcases List.mem_flatMap.mp member with
    ⟨row, rowMember, columnMember⟩
  have rowMember' :
      row ∈ (onePortJoinRecipe path selector port onTrue onFalse).rows := by
    simpa [onePortJoinReceipt] using rowMember
  have columnMember' : column ∈ row.columnIds := by
    simpa [InstructionReceipt.rowColumns, OwnedRow.columnIds,
      Row.columnIds] using columnMember
  have supported :=
    MuxRecipe.rows_supported
      (onePortJoinRecipe path selector port onTrue onFalse)
      row rowMember' column columnMember'
  simp only [onePortJoinRecipe, List.mem_append,
    List.mem_singleton] at supported
  rcases supported with
    ((selectorMember | joinedMember) | trueMember) | falseMember
  · left
    apply inputs column
    simp [selectorMember]
  · right
    simpa [onePortJoinReceipt, InstructionReceipt.columnIds,
      InstructionReceipt.ofMux, onePortJoinRecipe,
      ColumnBundle.ids] using joinedMember
  · left
    apply inputs column
    simp [trueMember]
  · left
    apply inputs column
    simp [falseMember]

/-- The empty terminal branch join is still a mandatory source-aligned
receipt, even though it emits no physical occurrence. -/
def emptyJoinReceipt (path : OwnerPath) : InstructionReceipt where
  owner := .typed (.branch path)
  kind := .branchJoin
  allocations := []
  rows := []
  allocationsOwned := by simp
  rowsOwned := by simp

@[simp] theorem emptyJoinReceipt_owner (path : OwnerPath) :
    (emptyJoinReceipt path).owner = .typed (.branch path) :=
  rfl

theorem emptyJoinReceipt_columnIdsNodup (path : OwnerPath) :
    (emptyJoinReceipt path).columnIds.Nodup := by
  simp [emptyJoinReceipt, InstructionReceipt.columnIds]

theorem emptyJoinReceipt_rowIdsNodup (path : OwnerPath) :
    (emptyJoinReceipt path).rowIds.Nodup := by
  simp [emptyJoinReceipt, InstructionReceipt.rowIds]

theorem emptyJoinReceipt_wellScoped
    (path : OwnerPath)
    (available : List ColumnId) :
    (emptyJoinReceipt path).WellScopedAfter available := by
  intro column member
  simp [emptyJoinReceipt, InstructionReceipt.referencedColumns] at member

end CanonicalBranchPlan

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
