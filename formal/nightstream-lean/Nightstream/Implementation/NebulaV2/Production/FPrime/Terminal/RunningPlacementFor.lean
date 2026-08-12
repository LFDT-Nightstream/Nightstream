import Nightstream.Implementation.NebulaV2.NIFS.Running.RunningCoordinatesFor
import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.CoreRowsFor

/-!
Contract: canonical final-running carrier placement for the exponent-indexed
terminal core.

One physical carrier stores the complete canonical `runningCodecFor` field
vector. Exact column aliases select the common point and, for each PiDEC
child, its public input and complete Phi81 evaluation family. The main theorem
derives `VerifierInputPlacement` from that one complete carrier placement.

`Placed` is an explicit producer-to-consumer ABI boundary. It states only the
field values in the complete running carrier. A complete terminal manifest
must derive it from the final paper-NIFS output rows. It does not state CE
membership, commitment opening, evaluation correctness, or acceptance.

This module does not own final-fold output rows, the terminal manifest, Rust
refinement, a compact backend, or cryptographic soundness.

Assurance tier: model-level canonical-carrier placement refinement.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperTerminalRunningPlacementFor

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor
open Nightstream.Protocol.NebulaV2.Terminal
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductionPaperTerminalCoreRowsFor.FullShape rowVariables logicalWidth
    publicFits

abbrev Running
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductNifsCodec.RunningFor rowVariables
    (FullShape rowVariables logicalWidth publicFits)

/-- One complete field-native final-running carrier. -/
structure Carrier (rowVariables : Nat) where
  column : Fin (ProductNifsCodec.runningFieldCountFor rowVariables) -> ColumnId

/-- Exact placement of the canonical running codec. This is a narrow physical
ABI fact. It contains no verifier result or terminal relation conclusion. -/
structure Placed
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (carrier : Carrier rowVariables) (assignment : ColumnId -> F)
    (running : Running rowVariables logicalWidth publicFits) : Prop where
  coordinate : forall index,
    assignment (carrier.column index) =
      ((ProductNifsCodec.runningCodecFor rowVariables
        (FullShape rowVariables logicalWidth publicFits)).encode running).getD
          index.val 0

def pointIndex
    {rowVariables : Nat} (coordinate : Fin rowVariables) (limb : Fin 2) :
    Fin (ProductNifsCodec.runningFieldCountFor rowVariables) :=
  ⟨pointCoordinateIndex coordinate limb,
    point_coordinate_bound coordinate limb⟩

def publicInputIndex
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (child : FoldedChild)
    (column : Fin (FullShape rowVariables logicalWidth publicFits).publicWidth) :
    Fin (ProductNifsCodec.runningFieldCountFor rowVariables) :=
  ⟨publicInputCoordinateIndex (rowVariables := rowVariables) child column,
    public_input_coordinate_bound
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) child column⟩

def evaluationIndex
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (child : FoldedChild) (matrix : Fin 14) (lane : Fin ringDegree)
    (limb : Fin 2) :
    Fin (ProductNifsCodec.runningFieldCountFor rowVariables) :=
  ⟨evaluationCoordinateIndex (rowVariables := rowVariables)
      (fullShape := FullShape rowVariables logicalWidth publicFits)
      child matrix lane limb,
    evaluation_coordinate_bound
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) child matrix lane limb⟩

/-- Exact zero-copy aliases from one full running carrier into one terminal
child checker. -/
structure Aliases
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    (carrier : Carrier rowVariables)
    (layout : ProductionPaperTerminalCoreRowsFor.Layout opening)
    (child : FoldedChild) : Prop where
  pointLow : forall coordinate,
    layout.pointLow coordinate = carrier.column (pointIndex coordinate 0)
  pointHigh : forall coordinate,
    layout.pointHigh coordinate = carrier.column (pointIndex coordinate 1)
  publicInput : forall column,
    layout.publicInput column = carrier.column (publicInputIndex child column)
  evaluationLow : forall matrix lane,
    layout.evaluationLow matrix lane =
      carrier.column (evaluationIndex (logicalWidth := logicalWidth)
        (publicFits := publicFits) child matrix lane 0)
  evaluationHigh : forall matrix lane,
    layout.evaluationHigh matrix lane =
      carrier.column (evaluationIndex (logicalWidth := logicalWidth)
        (publicFits := publicFits) child matrix lane 1)

private theorem pointCoordinate_eq_getD
    {rowVariables : Nat} (point : CubePoint K rowVariables)
    (coordinate : Fin rowVariables) :
    ProductionPaperTerminalCoreRowsFor.pointCoordinate point coordinate =
      point.coordinates.getD coordinate.val K.zero := by
  unfold ProductionPaperTerminalCoreRowsFor.pointCoordinate
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by rw [point.dimension]; exact coordinate.isLt)]
  rfl

/-- One canonical full-carrier placement and exact aliases derive every ABI
fact consumed by the terminal CE rows. -/
theorem toVerifierInputPlacement
    {manifest : SeedSchedule.Manifest}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {opening : TerminalBundleOpeningRows.Layout manifest
      (FullShape rowVariables logicalWidth publicFits)
      operationsShape snapshotShape}
    {carrier : Carrier rowVariables}
    {layout : ProductionPaperTerminalCoreRowsFor.Layout opening}
    {assignment : ColumnId -> F}
    {running : Running rowVariables logicalWidth publicFits}
    {child : FoldedChild}
    (placed : Placed carrier assignment running)
    (aliases : Aliases carrier layout child) :
    ProductionPaperTerminalCoreRowsFor.VerifierInputPlacement layout assignment
      running child := by
  constructor
  · intro column
    rw [aliases.publicInput column, placed.coordinate]
    exact ProductNifsRunningCoordinatesFor.runningCodecFor_publicInput_getD
      running child column
  · intro coordinate
    rw [aliases.pointLow coordinate, placed.coordinate]
    simpa [pointIndex, pointCoordinate_eq_getD] using
      ProductNifsRunningCoordinatesFor.runningCodecFor_point_getD running
        coordinate (0 : Fin 2)
  · intro coordinate
    rw [aliases.pointHigh coordinate, placed.coordinate]
    simpa [pointIndex, pointCoordinate_eq_getD] using
      ProductNifsRunningCoordinatesFor.runningCodecFor_point_getD running
        coordinate (1 : Fin 2)
  · intro matrix lane
    rw [aliases.evaluationLow matrix lane, placed.coordinate]
    simpa [evaluationIndex] using
      ProductNifsRunningCoordinatesFor.runningCodecFor_evaluation_getD running
        child matrix lane (0 : Fin 2)
  · intro matrix lane
    rw [aliases.evaluationHigh matrix lane, placed.coordinate]
    simpa [evaluationIndex] using
      ProductNifsRunningCoordinatesFor.runningCodecFor_evaluation_getD running
        child matrix lane (1 : Fin 2)

end Nightstream.Implementation.NebulaV2.ProductionPaperTerminalRunningPlacementFor
