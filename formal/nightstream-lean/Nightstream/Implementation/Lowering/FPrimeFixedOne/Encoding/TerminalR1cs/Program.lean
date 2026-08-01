import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Layout

/-!
Contract: exact cost and semantic round trip for the physical terminal
program.

Assurance tier: model-level.

Owns: full terminal input-plus-row cost, equality between the receipt stream
and the terminal row program, proof-free normalization, and preservation of
R1CS satisfaction.

Does not own: concrete statement values, private witness values, physical
assignment construction, Spartan, WHIR, Rust, or Ajtai binding security.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Program

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev RelationShape
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :=
  NativeCcsPhi81.shape program domain publicRingColumns publicFits

private theorem cost_eq_of_components {left right : Cost}
    (rows : left.recurringRows = right.recurringRows)
    (committed : left.committedColumns = right.committedColumns)
    (publicColumns : left.publicColumns = right.publicColumns)
    (auxiliary : left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Closed cost of a homogeneous column block. -/
def blockCost (count : Nat) : Ownership → Cost
  | .committedColumn => ⟨0, count, 0, 0⟩
  | .publicColumn => ⟨0, 0, count, 0⟩
  | .auxiliaryColumn => ⟨0, 0, 0, count⟩

private theorem columnCost_ofFn
    {count : Nat}
    (columns : Fin count → ColumnId)
    (ownership : Ownership) :
    columnCost
        (List.ofFn fun coordinate =>
          ({ id := columns coordinate
             ownership := ownership } : OwnedColumn)) =
      blockCost count ownership := by
  induction count with
  | zero =>
      cases ownership <;> rfl
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      simp only [columnCost, List.map_cons, Cost.sum]
      change
        Cost.oneColumn ownership +
            columnCost
              (List.ofFn fun coordinate =>
                ({ id := columns coordinate.succ
                   ownership := ownership } : OwnedColumn)) =
          blockCost (count + 1) ownership
      rw [inductionHypothesis (fun coordinate => columns coordinate.succ)]
      cases ownership <;>
        apply cost_eq_of_components <;>
        simp [blockCost, Cost.oneColumn] <;>
        omega

@[simp] theorem columnBlock_cost
    (owner : PhysicalOwner)
    (start count : Nat)
    (ownership : Ownership) :
    columnCost (Layout.columnBlock owner start count ownership) =
      blockCost count ownership := by
  exact columnCost_ofFn
    (fun coordinate => Layout.localColumn owner (start + coordinate.val))
    ownership

private theorem columnCost_append
    (left right : List OwnedColumn) :
    columnCost (left ++ right) = columnCost left + columnCost right := by
  simp [columnCost, List.map_append, Cost.sum_append]

private theorem rowCost_exact (rows : List OwnedRow) :
    rowCost rows = ⟨rows.length, 0, 0, 0⟩ := by
  induction rows with
  | nil =>
      rfl
  | cons row rest inductionHypothesis =>
      simp only [rowCost, List.map_cons, Cost.sum]
      change
        Cost.oneRow + rowCost rest =
          { recurringRows := (row :: rest).length
            committedColumns := 0
            publicColumns := 0
            auxiliaryColumns := 0 }
      rw [inductionHypothesis]
      apply cost_eq_of_components <;>
        simp [Cost.oneRow] <;>
        omega

/-- Full physical cost for one running receipt. -/
def runningCost
    (shape : Phi81Relation.Shape) (verifierRows : Nat) : Cost :=
  ⟨verifierRows * ringDegree + shape.publicWidth +
      2 * shape.carrierWidth +
      2 * (shape.matrixCount * ringDegree),
    shape.carrierWidth,
    Layout.runningStatementWidth shape verifierRows,
    shape.carrierWidth⟩

/-- Full physical cost for the fresh receipt. -/
def freshCost
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape) (verifierRows : Nat) : Cost :=
  ⟨verifierRows * ringDegree + shape.publicWidth +
      2 * shape.carrierWidth + 2 * program.rows.length,
    shape.carrierWidth,
    Layout.freshStatementWidth shape verifierRows,
    shape.carrierWidth + program.rows.length⟩

private def repeatedCost (count : Nat) (item : Cost) : Cost :=
  ⟨count * item.recurringRows,
    count * item.committedColumns,
    count * item.publicColumns,
    count * item.auxiliaryColumns⟩

private theorem sum_map_constant
    {alpha : Type}
    (items : List alpha)
    (itemCost : alpha → Cost)
    (constant : Cost)
    (same : ∀ item ∈ items, itemCost item = constant) :
    Cost.sum (items.map itemCost) =
      repeatedCost items.length constant := by
  induction items with
  | nil =>
      simp [Cost.sum, repeatedCost, Cost.zero]
  | cons head rest inductionHypothesis =>
      simp only [List.map_cons, Cost.sum, List.length_cons]
      rw [same head (by simp)]
      rw [inductionHypothesis (fun item member =>
        same item (by simp [member]))]
      apply cost_eq_of_components <;>
        simp [repeatedCost, Nat.succ_mul, Nat.add_comm]

@[simp] theorem runningReceipt_cost
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (child : Fin productionGlobalParams.k) :
    (Layout.runningReceipt key statements child).cost =
      runningCost
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows := by
  change
    physicalCost
        (Layout.runningAllocations
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows child)
        (Running.rows (Layout.runningFrame key child) (statements child)) =
      runningCost
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows
  unfold physicalCost
  rw [rowCost_exact]
  simp only [Layout.runningAllocations, columnCost_append, columnBlock_cost]
  apply cost_eq_of_components <;>
    simp [blockCost, runningCost, Layout.runningStatementWidth,
      Running.rows_length] <;>
    omega

@[simp] theorem freshReceipt_cost
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows) :
    (Layout.freshReceipt valid key).cost =
      freshCost program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows := by
  change
    physicalCost
        (Layout.freshAllocations program
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows)
        (Fresh.rows valid (Layout.freshFrame key)) =
      freshCost program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows
  unfold physicalCost
  rw [rowCost_exact]
  simp only [Layout.freshAllocations, columnCost_append, columnBlock_cost]
  apply cost_eq_of_components <;>
    simp [blockCost, freshCost, Layout.freshStatementWidth,
      Fresh.rows_length] <;>
    omega

private theorem runningReceipts_cost
    {source : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain source}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth source.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements source domain publicRingColumns publicFits
        verifierRows) :
    Cost.sum
        ((List.finRange productionGlobalParams.k).map fun child =>
          (Layout.runningReceipt key statements child).cost) =
      repeatedCost productionGlobalParams.k
        (runningCost
          (RelationShape source domain publicRingColumns publicFits)
          verifierRows) := by
  apply sum_map_constant
  intro child _
  exact runningReceipt_cost key statements child

private theorem preludeReceipt_cost :
    InstructionReceipt.prelude.cost = ⟨0, 0, 1, 0⟩ :=
  rfl

/-- Full terminal cost, including statement and witness input columns. -/
def cost
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) : Cost :=
  ⟨productionGlobalParams.k *
        (verifierRows * ringDegree + shape.publicWidth +
          2 * shape.carrierWidth +
          2 * (shape.matrixCount * ringDegree)) +
      (verifierRows * ringDegree + shape.publicWidth +
        2 * shape.carrierWidth + 2 * program.rows.length),
    (productionGlobalParams.k + 1) * shape.carrierWidth,
    1 +
      productionGlobalParams.k *
        Layout.runningStatementWidth shape verifierRows +
      Layout.freshStatementWidth shape verifierRows,
    productionGlobalParams.k * shape.carrierWidth +
      shape.carrierWidth + program.rows.length⟩

@[simp] theorem manifest_cost_exact
    {source : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain source}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth source.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid source)
    (key :
      Commitment.Key
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements source domain publicRingColumns publicFits
        verifierRows) :
    (Layout.program valid key statements).cost =
      cost source
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows := by
  change
    Cost.sum
        (((Layout.receipts valid key statements).map
          CanonicalManifest.ManifestReceipt.ofReceipt).map
            CanonicalManifest.ManifestReceipt.cost) =
      cost source
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows
  rw [List.map_map]
  have costFunction :
      CanonicalManifest.ManifestReceipt.cost ∘
          CanonicalManifest.ManifestReceipt.ofReceipt =
        InstructionReceipt.cost := by
    funext receipt
    exact CanonicalManifest.ManifestReceipt.cost_ofReceipt receipt
  rw [costFunction]
  unfold Layout.receipts
  simp only [List.map_cons, List.map_append, Cost.sum, Cost.sum_append]
  rw [List.map_map]
  change
    InstructionReceipt.prelude.cost +
        (Cost.sum
            ((List.finRange productionGlobalParams.k).map fun child =>
              (Layout.runningReceipt key statements child).cost) +
          ((Layout.freshReceipt valid key).cost + Cost.zero)) =
      cost source
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows
  rw [runningReceipts_cost key statements, freshReceipt_cost valid key,
    preludeReceipt_cost]
  apply cost_eq_of_components <;>
    simp [Cost.zero, repeatedCost, cost, runningCost, freshCost,
      Layout.runningStatementWidth, Layout.freshStatementWidth] <;>
    omega

private theorem proofRows_exact
    {source : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain source}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth source.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid source)
    (key :
      Commitment.Key
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements source domain publicRingColumns publicFits
        verifierRows) :
    (Layout.receipts valid key statements).flatMap
        InstructionReceipt.rows =
      Terminal.rows valid (Layout.frame key) statements := by
  simp [Layout.receipts, Layout.runningReceipt, Layout.freshReceipt,
    Terminal.rows, Terminal.runningRows, List.flatMap_map, Layout.frame]

private theorem decode_receipt_rows
    (receipts : List InstructionReceipt) :
    ((receipts.map
        CanonicalManifest.ManifestReceipt.ofReceipt).map
          CanonicalManifest.ManifestReceipt.decode).flatMap
            CanonicalManifest.ReceiptImage.rows =
      (receipts.flatMap InstructionReceipt.rows).map
        CanonicalManifest.normalizeOwnedRow := by
  induction receipts with
  | nil =>
      rfl
  | cons receipt rest inductionHypothesis =>
      simp only [List.map_cons, List.flatMap_cons, List.map_append]
      change
        (CanonicalManifest.ManifestReceipt.ofReceipt receipt).decode.rows ++
            ((rest.map
                CanonicalManifest.ManifestReceipt.ofReceipt).map
              CanonicalManifest.ManifestReceipt.decode).flatMap
                CanonicalManifest.ReceiptImage.rows =
          receipt.rows.map CanonicalManifest.normalizeOwnedRow ++
            (rest.flatMap InstructionReceipt.rows).map
              CanonicalManifest.normalizeOwnedRow
      rw [CanonicalManifest.ManifestReceipt.decode_ofReceipt,
        inductionHypothesis]
      rfl

theorem decoded_rows_exact
    {source : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain source}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth source.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid source)
    (key :
      Commitment.Key
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements source domain publicRingColumns publicFits
        verifierRows) :
    (Layout.program valid key statements).decode.rows =
      (Terminal.rows valid (Layout.frame key) statements).map
        CanonicalManifest.normalizeOwnedRow := by
  unfold Layout.program CanonicalManifest.Program.decode
    CanonicalManifest.ProgramImage.rows
  rw [decode_receipt_rows, proofRows_exact]

/-- The proof-free manifest accepts exactly the terminal R1CS program. -/
theorem decoded_satisfies_iff
    {source : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain source}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth source.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid source)
    (key :
      Commitment.Key
        (RelationShape source domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements source domain publicRingColumns publicFits
        verifierRows)
    (assignment : ColumnId → F) :
    Satisfies (Layout.program valid key statements).decode.rows assignment ↔
      Satisfies (Terminal.rows valid (Layout.frame key) statements)
        assignment := by
  rw [decoded_rows_exact]
  exact
    CanonicalManifest.satisfies_map_normalizeOwnedRow
      (Terminal.rows valid (Layout.frame key) statements) assignment

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Program
