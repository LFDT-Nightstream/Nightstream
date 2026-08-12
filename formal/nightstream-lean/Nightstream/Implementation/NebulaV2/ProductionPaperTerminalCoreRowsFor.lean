import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalInvocationRowsSoundFor
import Nightstream.Implementation.NebulaV2.TerminalBundleOpeningRows
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.Phi81SharedEvaluationRows

/-!
Contract: row-derived terminal CE core for one exponent-indexed PiDEC child.

The complete assignment is the same witness used by the four-component
terminal opening. Public-projection rows bind its aligned public prefix to
the verifier-input columns. Shared-tensor Phi81 rows bind every claimed
matrix lane to the independent SuperNeo evaluator at the verifier-input
point.

`VerifierInputPlacement` is an explicit ABI boundary. It identifies typed
columns with the final NIFS running value. It does not state a public
projection equation, an evaluation equation, CE membership, or acceptance.
A complete terminal manifest must derive this placement from the final-fold
public carrier.

This module does not own commitment opening, norm rows, final-fold carrier
placement, a generated terminal manifest, Rust refinement, a compact backend,
or cryptographic soundness.

Assurance tier: model-level terminal-core row soundness.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCoreRowsFor

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows
open Nightstream.Implementation.R1CS.Phi81SharedTensorRows
open Nightstream.Protocol.NebulaV2.Terminal
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductionPaperTerminalInvocationRowsSoundFor.FullShape rowVariables
    logicalWidth publicFits

abbrev Running
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductNifsCodec.RunningFor rowVariables
    (FullShape rowVariables logicalWidth publicFits)

/-- Read one dimension-checked point coordinate without a default branch. -/
def pointCoordinate {rowVariables : Nat}
    (point : CubePoint K rowVariables) (index : Fin rowVariables) : K :=
  point.coordinates.get
    ⟨index.val, by rw [point.dimension]; exact index.isLt⟩

private theorem point_coordinates_ofFn
    {rowVariables : Nat} (point : CubePoint K rowVariables) :
    List.ofFn (pointCoordinate point) = point.coordinates := by
  apply List.ext_get
  · simp [point.dimension]
  · intro index leftBound rightBound
    rw [List.get_ofFn]
    rfl

private theorem cubePoint_eq_of_coordinates
    {rowVariables : Nat} {left right : CubePoint K rowVariables}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- Physical terminal-core columns. The evaluator frame is constructed below
so its witness is definitionally the terminal-opening witness. -/
structure Layout
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape) where
  publicInput : Fin (FullShape rowVariables logicalWidth publicFits).publicWidth ->
    ColumnId
  pointLow : Fin rowVariables -> ColumnId
  pointHigh : Fin rowVariables -> ColumnId
  evaluationLow : Fin 14 -> Fin ringDegree -> ColumnId
  evaluationHigh : Fin 14 -> Fin ringDegree -> ColumnId
  publicOwner : PhysicalOwner
  publicFirstOrdinal : Nat
  tensorOwner : PhysicalOwner
  tensorFirstOrdinal : Nat -> Nat -> Nat
  tensorFrame : Nat -> Nat -> Extension.Frame
  productOwner : PhysicalOwner
  productFirstOrdinal : Fin 14 -> Fin ringDegree ->
    BooleanVertex rowVariables -> Nat
  productFrame : Fin 14 -> Fin ringDegree ->
    BooleanVertex rowVariables -> ProductFrame
  outputOwner : PhysicalOwner
  outputFirstOrdinal : Fin 14 -> Fin ringDegree -> Nat

def Layout.tensor
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening) :
    Nightstream.Implementation.R1CS.Phi81SharedTensorRows.Frame
      rowVariables where
  one := opening.one
  pointLow := layout.pointLow
  pointHigh := layout.pointHigh
  owner := layout.tensorOwner
  tensorFirstOrdinal := layout.tensorFirstOrdinal
  tensorFrame := layout.tensorFrame

/-- The exact sparse evaluator. Its assignment columns are the same complete
witness columns used by `opening`. -/
def Layout.evaluator
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening) :
    Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.Frame
      (FullShape rowVariables logicalWidth publicFits) where
  tensor := layout.tensor
  witness := opening.fullWitness
  claimLow := layout.evaluationLow
  claimHigh := layout.evaluationHigh
  productOwner := layout.productOwner
  productFirstOrdinal := layout.productFirstOrdinal
  productFrame := layout.productFrame
  outputOwner := layout.outputOwner
  outputFirstOrdinal := layout.outputFirstOrdinal

def publicLeft
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (column : Fin (FullShape rowVariables logicalWidth publicFits).publicWidth) :
    LinearCombination :=
  [⟨opening.fullWitness
      ((FullShape rowVariables logicalWidth publicFits).publicColumn column), 1⟩]

def publicRight
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening)
    (column : Fin (FullShape rowVariables logicalWidth publicFits).publicWidth) :
    LinearCombination :=
  [⟨layout.publicInput column, 1⟩]

/-- One actual equality row for every aligned public coordinate. -/
def publicRows
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening) : List OwnedRow :=
  (canonicalFinIndices
      (FullShape rowVariables logicalWidth publicFits).publicWidth).map
    fun column => Atoms.linearCheckOwnedRow layout.publicOwner
      (layout.publicFirstOrdinal + column.val) opening.one
      (publicLeft (opening := opening) column) (publicRight layout column)

@[simp] theorem publicRows_length
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening) :
    (publicRows layout).length =
      (FullShape rowVariables logicalWidth publicFits).publicWidth := by
  simp [publicRows, canonicalFinIndices_length]

/-- Exact interpretation of the final-running public columns. This is an ABI
placement contract, not a terminal relation result. -/
structure VerifierInputPlacement
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening) (assignment : ColumnId -> F)
    (running : Running rowVariables logicalWidth publicFits)
    (child : FoldedChild) : Prop where
  publicInput : forall column,
    assignment (layout.publicInput column) = running.publicInputs child column
  pointLow : forall coordinate,
    assignment (layout.pointLow coordinate) =
      (pointCoordinate running.point coordinate).c0
  pointHigh : forall coordinate,
    assignment (layout.pointHigh coordinate) =
      (pointCoordinate running.point coordinate).c1
  evaluationLow : forall matrix lane,
    assignment (layout.evaluationLow matrix lane) =
      (running.evaluations child matrix lane).c0
  evaluationHigh : forall matrix lane,
    assignment (layout.evaluationHigh matrix lane) =
      (running.evaluations child matrix lane).c1

namespace VerifierInputPlacement

theorem point_exact
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    {layout : Layout opening} {assignment : ColumnId -> F}
    {running : Running rowVariables logicalWidth publicFits}
    {child : FoldedChild}
    (placement : VerifierInputPlacement layout assignment running child) :
    Nightstream.Implementation.R1CS.Phi81SharedTensorRows.decodedPoint
      layout.tensor assignment =
      running.point := by
  apply cubePoint_eq_of_coordinates
  change
    List.ofFn (fun coordinate : Fin rowVariables =>
      ⟨assignment (layout.pointLow coordinate),
       assignment (layout.pointHigh coordinate)⟩) = running.point.coordinates
  rw [← point_coordinates_ofFn running.point]
  apply congrArg List.ofFn
  funext coordinate
  exact congrArg₂ K.mk (placement.pointLow coordinate)
    (placement.pointHigh coordinate)

theorem evaluations_exact
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    {layout : Layout opening} {assignment : ColumnId -> F}
    {running : Running rowVariables logicalWidth publicFits}
    {child : FoldedChild}
    (placement : VerifierInputPlacement layout assignment running child) :
    Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.decodedEvaluations
      layout.evaluator assignment =
      Array.ofFn (running.evaluations child) := by
  unfold Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.decodedEvaluations
  congr 1
  funext matrix lane
  exact congrArg₂ K.mk (placement.evaluationLow matrix lane)
    (placement.evaluationHigh matrix lane)

end VerifierInputPlacement

/-- Actual public and Phi81 row evidence for one child. -/
structure RowsEvidence
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening) (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : ColumnId -> F) : Prop where
  publicRows : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
    (ProductionPaperTerminalCoreRowsFor.publicRows layout) assignment
  evaluationRows :
    Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.RowsSatisfied
      layout.evaluator system assignment

/-- Actual public and Phi81 row evidence plus the separately derived canonical
verifier-input placement for one child. -/
structure Evidence
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening) (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : ColumnId -> F)
    (running : Running rowVariables logicalWidth publicFits)
    (child : FoldedChild) : Prop where
  placement : VerifierInputPlacement layout assignment running child
  rows : RowsEvidence layout system assignment

namespace Evidence

def ofRows
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    {layout : Layout opening} {system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)}
    {assignment : ColumnId -> F}
    {running : Running rowVariables logicalWidth publicFits}
    {child : FoldedChild}
    (placement : VerifierInputPlacement layout assignment running child)
    (rows : RowsEvidence layout system assignment) :
    Evidence layout system assignment running child :=
  { placement := placement, rows := rows }

end Evidence

private theorem satisfies_member
    {rows : List OwnedRow} {assignment : ColumnId -> F}
    (satisfied : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      rows assignment) {row : OwnedRow} (member : row ∈ rows) :
    row.row.Holds assignment := by
  induction rows with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      rcases satisfied with ⟨headHolds, tailHolds⟩
      rcases List.mem_cons.mp member with equal | tailMember
      · simpa [equal] using headHolds
      · exact inductionHypothesis tailHolds tailMember

theorem public_row_exact
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (layout : Layout opening) (assignment : ColumnId -> F)
    (constantOne : assignment opening.one = 1)
    (satisfied : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (publicRows layout) assignment)
    (column : Fin (FullShape rowVariables logicalWidth publicFits).publicWidth) :
    assignment (opening.fullWitness
        ((FullShape rowVariables logicalWidth publicFits).publicColumn column)) =
      assignment (layout.publicInput column) := by
  have rowMember : Atoms.linearCheckOwnedRow layout.publicOwner
      (layout.publicFirstOrdinal + column.val) opening.one
      (publicLeft (opening := opening) column) (publicRight layout column) ∈
      publicRows layout := by
    apply List.mem_map.mpr
    exact ⟨column, List.mem_ofFn.mpr ⟨column, rfl⟩, rfl⟩
  have holds := satisfies_member satisfied rowMember
  have equality :=
    (Atoms.linearCheckRow_iff assignment opening.one _ _ constantOne).1 holds
  change
    1 * assignment (opening.fullWitness
        ((FullShape rowVariables logicalWidth publicFits).publicColumn column)) +
        0 =
      1 * assignment (layout.publicInput column) + 0 at equality
  simpa only [Fin.one_mul, Fin.add_zero] using equality

theorem public_exact
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    {layout : Layout opening}
    {system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)}
    {assignment : ColumnId -> F}
    {running : Running rowVariables logicalWidth publicFits}
    {child : FoldedChild}
    (constantOne : assignment opening.one = 1)
    (evidence : Evidence layout system assignment running child) :
    projectPublicInput (opening.fullAssignment assignment) =
      running.publicInputs child := by
  funext column
  calc
    projectPublicInput (opening.fullAssignment assignment) column =
        assignment (opening.fullWitness
          ((FullShape rowVariables logicalWidth publicFits).publicColumn column)) :=
      rfl
    _ = assignment (layout.publicInput column) :=
      public_row_exact layout assignment constantOne evidence.rows.publicRows
        column
    _ = running.publicInputs child column :=
      evidence.placement.publicInput column

theorem evaluations_exact
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    {layout : Layout opening}
    {system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)}
    {assignment : ColumnId -> F}
    {running : Running rowVariables logicalWidth publicFits}
    {child : FoldedChild}
    (constantOne : assignment opening.one = 1)
    (evidence : Evidence layout system assignment running child) :
    Phi81Relation.evaluations system (opening.fullAssignment assignment)
        running.point = Array.ofFn (running.evaluations child) := by
  have rowsExact :=
    Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.rows_sound
      layout.evaluator system assignment constantOne
        evidence.rows.evaluationRows
  have pointExact :
      Nightstream.Implementation.R1CS.Phi81SharedTensorRows.decodedPoint
          layout.evaluator.tensor assignment = running.point := by
    simpa [Layout.evaluator] using evidence.placement.point_exact
  have claimsExact :
      Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.decodedEvaluations
          layout.evaluator assignment =
        Array.ofFn (running.evaluations child) :=
    evidence.placement.evaluations_exact
  calc
    Phi81Relation.evaluations system (opening.fullAssignment assignment)
        running.point =
      Phi81Relation.evaluations system
        (Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.decodedAssignment
          layout.evaluator assignment)
        (Nightstream.Implementation.R1CS.Phi81SharedTensorRows.decodedPoint
          layout.evaluator.tensor assignment) := by
      rw [pointExact]
      rfl
    _ = Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.decodedEvaluations
          layout.evaluator assignment := rowsExact.symm
    _ = Array.ofFn (running.evaluations child) := claimsExact

end Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCoreRowsFor
