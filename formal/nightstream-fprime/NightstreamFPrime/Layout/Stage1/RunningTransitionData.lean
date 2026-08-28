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
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

/-- The completed PiDEC source-column endpoint. -/
def phaseOffset : Nat := 27374284

theorem phaseOffset_matches_piDec
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseOffset = PilotPiCCSPiRLCPiDEC.physicalColumnCount relation := by
  rw [PilotPiCCSPiRLCPiDEC.physicalColumnCount_eq]
  rfl

def iterationWordIndex : Nat := 28

def iterationExpr : Expr :=
  Expr.var (PilotProduction.priorPreimageStart + iterationWordIndex)

def outputBase : Nat := PilotProduction.outputPreimageStart

def outputPairAt (relative : Nat) : KExpr :=
  ⟨Expr.var (outputBase + relative), Expr.var (outputBase + relative + 1)⟩

def outputPoint
    (coordinate : Fin productionShape.cubeVariables) : KExpr :=
  outputPairAt (PiCCSInputs.runningPointStart + coordinate.val * 2)

def outputCommitment
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) : Expr :=
  Expr.var (outputBase + PiCCSInputs.runningCommitmentStart source.val +
    row.val * ringDegree + coefficient.val)

def outputPublicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) : Expr :=
  Expr.var (outputBase + PiCCSInputs.runningPublicStart source.val + column.val)

def outputEval_K
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  outputPairAt
    (PiCCSInputs.runningEvaluationStart source.val + coefficient.val * 2)

def outputEval_A
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  outputPairAt (PiCCSInputs.runningEvaluationStart source.val + 108 +
    matrix.val * 108 + coefficient.val * 2)

def outputRunningExpr
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    StatementAbsorption.RunningExpr logicalWidth publicFits where
  point := outputPoint
  commitment := outputCommitment
  publicInput := outputPublicInput
  evaluation := fun source => {
    eval_K := outputEval_K source
    eval_A := outputEval_A source }

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
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount
        logicalWidth publicFits := by
  rw [NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount_eq]
  rfl

def digitCoordinate
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (coordinate : Fin (FullShape logicalWidth publicFits).publicWidth) :
    Fin (NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount
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
  recursive := fun _ => recursiveRunningExpr logicalWidth publicFits
  output := fun _ => outputRunningExpr logicalWidth publicFits

end NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
