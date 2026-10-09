import NightstreamFPrime.Layout.Stage1.PiDECInputs
import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDEC
import NightstreamFPrime.Lifecycle.Stage1.RunningTransition

/-! Owns the fixed zero-copy data map for the Stage 1 running transition. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

/-- The completed PiDEC source-column endpoint. -/
def phaseOffset : Nat := PiDECStarts.outputFreshStart

/-- PiDEC inputs precede its logical and fresh allocations. -/
theorem piDecPhaseOffset_le : PiDECInputs.phaseOffset ≤ phaseOffset := by
  dsimp only [phaseOffset, PiDECStarts.outputFreshStart,
    PiDECStarts.evalAFreshStart, PiDECStarts.evalKFreshStart,
    PiDECStarts.commitmentFreshStart, PiDECStarts.publicInputFreshStart,
    PiDECStarts.inputFreshStart, PiDECStarts.phaseFreshStart,
    PiDECStarts.phaseLogicalStart]
  exact Nat.le_trans (Nat.le_add_right _ _) (Nat.le_add_right _ _)

theorem phaseOffset_matches_piDec
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseOffset = PilotPiCCSPiRLCPiDEC.physicalColumnCount relation := by
  rw [PilotPiCCSPiRLCPiDEC.physicalColumnCount_eq]
  rfl

/-- Tail positions in the state block: `vk` at 27,806, then `i`, `z0`, `zi`. -/
def iterationWordIndex : Nat := 27810

def initialStateWordStart : Nat := 27811
def currentStateWordStart : Nat := 27815

def iterationExpr : Expr :=
  Expr.var (PilotProduction.priorPreimageStart + iterationWordIndex)

def initialStateExpr (index : RunningTransition.StateIndex) : Expr :=
  Expr.var (PilotProduction.priorPreimageStart + initialStateWordStart + index.val)

def currentStateExpr (index : RunningTransition.StateIndex) : Expr :=
  Expr.var (PilotProduction.priorPreimageStart + currentStateWordStart + index.val)

def outputBase : Nat := PilotProduction.outputPreimageStart

/-- Running word `index` of the output state block, in place. -/
def outputWord (index : RunningTransition.WordIndex) : Expr :=
  Expr.var (outputBase + PiCCSInputs.priorRunningStart + index.val)

theorem runningCount_eq_childCount :
    productionShape.runningCount = productionGlobalParams.k := by
  decide

def childOfRunning
    (source : Fin productionShape.runningCount) : Radix.ChildIndex :=
  Fin.cast runningCount_eq_childCount source

@[simp] theorem childOfRunning_val
    (source : Fin productionShape.runningCount) :
    (childOfRunning source).val = source.val := by
  rfl

theorem publicWidth_eq_coordinateCount
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (FullShape logicalWidth publicFits).publicWidth =
      NightstreamFPrime.Lifecycle.PiDEC.v1_2.PublicInputSplit.coordinateCount
        logicalWidth publicFits := by
  rw [NightstreamFPrime.Lifecycle.PiDEC.v1_2.PublicInputSplit.coordinateCount_eq]
  rfl

def digitCoordinate
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (coordinate : Fin (FullShape logicalWidth publicFits).publicWidth) :
    Fin (NightstreamFPrime.Lifecycle.PiDEC.v1_2.PublicInputSplit.coordinateCount
      logicalWidth publicFits) :=
  Fin.cast (publicWidth_eq_coordinateCount logicalWidth publicFits) coordinate

@[simp] theorem digitCoordinate_val
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (coordinate : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (digitCoordinate coordinate).val = coordinate.val := by
  rfl

def piDecInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  PiDECInputs.interface logicalWidth publicFits

def recursiveRunningExpr
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    StatementAbsorption.RunningExpr logicalWidth publicFits :=
  let piDec := piDecInterface logicalWidth publicFits
  { point := piDec.point PiDECInputs.phaseOffset
    commitment := fun source =>
      (piDec.message PiDECInputs.phaseOffset
        (childOfRunning source)).commitment
    publicInput := fun source coordinate =>
      piDec.digit PiDECInputs.phaseOffset (childOfRunning source)
        (digitCoordinate coordinate)
    evaluation := fun source =>
      (piDec.message PiDECInputs.phaseOffset
        (childOfRunning source)).evaluation }

/-- The sole logical transition interface in cumulative Stage 1 source order. -/
def interface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    RunningTransition.Interface logicalWidth publicFits where
  iteration := fun _ => iterationExpr
  initialState := fun _ => initialStateExpr
  currentState := fun _ => currentStateExpr
  recursive := fun _ => recursiveRunningExpr logicalWidth publicFits
  output := fun _ => outputWord

end NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
