import Mathlib.Data.List.GetD
import NightstreamFPrime.Layout.PilotProduction
import NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs

/-!
Paper authority: SuperNeo v1.2, section 7.3, PiCCS input and output messages.
Obligation: Own the concrete parent columns read by the production PiCCS
circuit.

The running commitments, `Eval_K`, `Eval_A`, and point are zero-copy reads of
the prior state block. The block stores only the packed parent public input,
so the sixteen child public inputs (4,320 digits) occupy one prover-supplied
region after the expected context. The statement-binding leaf computes one
sign per lane as a hinted column, and its rows check the digits against the
packed prior words. The fresh public input
reuses the pilot public-input columns. The fresh commitment, 28 degree-eight
SumCheck messages, and separate output `Eval_K`/`Eval_A` families follow.

No equality row is added at this boundary. The following PiCCS allocation
starts after this complete input interval.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-! ## Prior state block positions -/

/-- Start of the running fields: after the 12-word domain chunk. -/
def priorRunningStart : Nat := 12
def runningCommitmentWords : Nat := 1188
def runningEvalKWords : Nat := 108
def runningEvalAWords : Nat := 432
def runningPublicWords : Nat := 270

def runningCommitmentStart (source : Nat) : Nat :=
  priorRunningStart + source * runningCommitmentWords

def runningEvalKStart (source : Nat) : Nat :=
  priorRunningStart + 19008 + source * runningEvalKWords

def runningEvalAStart (source : Nat) : Nat :=
  priorRunningStart + 20736 + source * runningEvalAWords

def runningPointStart : Nat := priorRunningStart + 27648

/-! ## Expected context, prior children, and proof inputs -/

/-- End of the completed pilot source-column interval and start of the
verifier-owned expected-context words. -/
def expectedContextStart : Nat := 5141760

def expectedContextWords : Nat := 4

/-- The prior child region: the 4,320 child digits, child-major. The
statement-binding leaf owns the sign of each parent coordinate. -/
def priorChildrenStart : Nat := expectedContextStart + expectedContextWords

def priorChildrenWords : Nat := 4320

def runningPublicStart (source : Nat) : Nat :=
  priorChildrenStart + source * runningPublicWords

def proofInputStart : Nat := priorChildrenStart + priorChildrenWords

theorem expectedContextStart_matches_pilot :
    expectedContextStart =
      Pilot.physicalColumnCount PilotProduction.interface
        PilotProduction.witnessOffset := by
  rw [PilotProduction.physicalColumnCount_eq]
  rfl

theorem expectedContextStart_eq : expectedContextStart = 5141760 := by
  rfl

theorem expectedContextWords_eq : expectedContextWords = 4 := by
  rfl

theorem priorChildrenStart_eq : priorChildrenStart = 5141764 := by
  rfl

theorem proofInputStart_eq : proofInputStart = 5146084 := by
  rfl

/-- New proof-input intervals. -/
def freshCommitmentStart : Nat := proofInputStart
def freshCommitmentWords : Nat := 1188
def roundMessageStart : Nat := freshCommitmentStart + freshCommitmentWords
def roundMessageWords : Nat := 504
def outputEvaluationStart : Nat := roundMessageStart + roundMessageWords
def outputEvaluationWords : Nat := 9180
/-- One source's output `Eval_K` then `Eval_A` words. -/
def outputEvaluationSourceWords : Nat := 540
def proofInputColumnCount : Nat :=
  freshCommitmentWords + roundMessageWords + outputEvaluationWords
def phaseOffset : Nat := proofInputStart + proofInputColumnCount

theorem freshCommitmentWords_eq :
    freshCommitmentWords = productionProfile.commitmentWidth * ringDegree := by
  norm_num [freshCommitmentWords, productionProfile, ringDegree]

theorem roundMessageWords_eq :
    roundMessageWords =
      productionShape.cubeVariables * (8 + 1) * 2 := by
  norm_num [roundMessageWords, productionShape, cubeVariables,
    Phi81MatrixSource.phi81Shape]

theorem outputEvaluationWords_eq :
    outputEvaluationWords =
      productionShape.sourceCount * (productionShape.matrixCount + 1) *
        productionShape.coefficientCount * 2 := by
  norm_num [outputEvaluationWords, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.sourceCount, ringDegree]

theorem proofInputColumnCount_eq : proofInputColumnCount = 10872 := by
  rfl

theorem phaseOffset_eq : phaseOffset = 5156956 := by
  rfl

/-! ## Symbolic inputs -/

def pairAt (start : Nat) : KExpr :=
  ⟨Expr.var start, Expr.var (start + 1)⟩

theorem pairAt_linear (start : Nat) : KExprLinear (pairAt start) := by
  refine ⟨rfl, rfl, ?_, ?_⟩ <;>
    simp [pairAt, Nonconstant]

def runningPoint
    (coordinate : Fin productionShape.cubeVariables) : KExpr :=
  pairAt (runningPointStart + coordinate.val * 2)

def runningCommitment
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) : Expr :=
  Expr.var (runningCommitmentStart source.val + row.val * ringDegree +
    coefficient.val)

def runningPublicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) : Expr :=
  Expr.var (runningPublicStart source.val + column.val)

def runningEval_K
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  pairAt (runningEvalKStart source.val + coefficient.val * 2)

def runningEval_A
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  pairAt (runningEvalAStart source.val + matrix.val * 108 + coefficient.val * 2)

def runningExpr
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    StatementAbsorption.RunningExpr logicalWidth publicFits where
  point := runningPoint
  commitment := runningCommitment
  publicInput := runningPublicInput
  evaluation := fun source => {
    eval_K := runningEval_K source
    eval_A := runningEval_A source
  }

def freshCommitment
    (source : Fin productionShape.freshCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) : Expr :=
  Expr.var (freshCommitmentStart + source.val * freshCommitmentWords +
    row.val * ringDegree + coefficient.val)

def freshPublicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (_source : Fin productionShape.freshCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) : Expr :=
  Expr.var (PilotProduction.priorPublicInputStart + column.val)

def roundCoefficient
    (roundIndex : Fin productionShape.cubeVariables)
    (coefficient : Fin (8 + 1)) : KExpr :=
  pairAt (roundMessageStart + roundIndex.val * 18 + coefficient.val * 2)

def outputEval_K
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  pairAt (outputEvaluationStart + source.val * 540 + coefficient.val * 2)

def outputEval_A
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  pairAt (outputEvaluationStart + source.val * 540 + 108 +
    matrix.val * 108 + coefficient.val * 2)

def freshExpr
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    StatementAbsorption.FreshExpr logicalWidth publicFits where
  commitment := freshCommitment
  publicInput := freshPublicInput

def roundMessage (roundIndex : Fin productionShape.cubeVariables) :
    RoundTranscript.Message 8 where
  coefficient := roundCoefficient roundIndex

def outputExpr : OutputBinding.OutputExpr where
  padCoordinate := outputEval_K
  matrixCoordinate := outputEval_A

def priorStateWord (index : Nat) : Expr :=
  Expr.var (PilotProduction.priorPreimageStart + index)

def outputStateWord (index : Nat) : Expr :=
  Expr.var (PilotProduction.outputPreimageStart + index)

def expectedContext (lane : Fin 4) : Expr :=
  Expr.var (expectedContextStart + lane.val)

/-- The one concrete symbolic PiCCS interface for this production prefix. -/
def interface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Formal.Interface logicalWidth 8 publicFits where
  baseOffset := phaseOffset
  priorState := fun _ => priorStateWord
  outputState := fun _ => outputStateWord
  expectedContext := fun _ => expectedContext
  running := fun _ => runningExpr logicalWidth publicFits
  fresh := fun _ => freshExpr logicalWidth publicFits
  round := fun _ => roundMessage
  output := fun _ => outputExpr

private theorem publicColumn_lt_270
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    column.val < 270 := by
  have bound := column.isLt
  norm_num [FullShape, fullShape, Phi81Relation.Shape.publicWidth,
    publicRingColumns, ringDegree] at bound
  exact bound

private theorem runningSource_lt_16
    (source : Fin productionShape.runningCount) : source.val < 16 := by
  have bound := source.isLt
  norm_num [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape] at bound
  exact bound

private theorem allSource_lt_17
    (source : Fin productionShape.sourceCount) : source.val < 17 := by
  have bound := source.isLt
  norm_num [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.sourceCount] at bound
  exact bound

private theorem round_lt_28
    (roundIndex : Fin productionShape.cubeVariables) :
    roundIndex.val < 28 := by
  have bound := roundIndex.isLt
  norm_num [productionShape, cubeVariables,
    Phi81MatrixSource.phi81Shape] at bound
  exact bound

private theorem matrix_lt_4
    (matrix : Fin productionShape.matrixCount) : matrix.val < 4 := by
  have bound := matrix.isLt
  norm_num [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape] at bound
  exact bound

private theorem coefficient_lt_54
    (coefficient : Fin productionShape.coefficientCount) :
    coefficient.val < 54 := by
  have bound := coefficient.isLt
  norm_num [productionShape, Phi81MatrixSource.phi81Shape,
    ringDegree] at bound
  exact bound

private theorem commitmentRow_lt_22
    (row : Fin productionProfile.commitmentWidth) : row.val < 22 := by
  have bound := row.isLt
  norm_num [productionProfile] at bound
  exact bound

private theorem ringCoefficient_lt_54
    (coefficient : Fin ringDegree) : coefficient.val < 54 := by
  have bound := coefficient.isLt
  norm_num [ringDegree] at bound
  exact bound

private theorem freshSource_lt_1
    (source : Fin productionShape.freshCount) : source.val < 1 := by
  have bound := source.isLt
  change source.val < 1 at bound
  exact bound

/-- Every external PiCCS expression is owned before the phase allocation. -/
theorem externalInputsBelow
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Formal.ExternalInputsBelow (interface logicalWidth publicFits)
      phaseOffset := by
  constructor
  · intro word member
    simp only [interface, priorStateWord, Expr.VarsBelow]
    have bound := StateBinding.fixedWord_index_lt word member
    rw [phaseOffset_eq]
    norm_num [PilotProduction.priorPreimageStart]
    omega
  · intro word member
    simp only [interface, outputStateWord, Expr.VarsBelow]
    have bound := StateBinding.fixedWord_index_lt word member
    rw [phaseOffset_eq]
    norm_num [PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq, PriorStateHash.publicWidth_eq]
    omega
  · intro lane
    simp only [interface, priorStateWord, Expr.VarsBelow]
    have bound := lane.isLt
    rw [phaseOffset_eq]
    norm_num [StateBinding.contextWordStart,
      PilotProduction.priorPreimageStart] at bound ⊢
    omega
  · intro lane
    simp only [interface, outputStateWord, Expr.VarsBelow]
    have bound := lane.isLt
    rw [phaseOffset_eq]
    norm_num [StateBinding.contextWordStart,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq, PriorStateHash.publicWidth_eq]
    omega
  · intro lane
    simp only [interface, expectedContext, Expr.VarsBelow]
    have bound := lane.isLt
    rw [phaseOffset_eq, expectedContextStart_eq]
    omega
  · intro word
    simp only [interface, priorStateWord, Expr.VarsBelow]
    have bound := word.isLt
    rw [phaseOffset_eq]
    norm_num [StateBinding.packedWordStart, PilotProduction.priorPreimageStart,
      packedParentWords] at bound ⊢
    omega
  · intro coordinate
    change (pairAt (runningPointStart + coordinate.val * 2)).VarsBelow
      phaseOffset
    simp only [pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have coordinateBound := round_lt_28 coordinate
    rw [phaseOffset_eq]
    unfold runningPointStart priorRunningStart
    omega
  · intro source row coefficient
    simp only [interface, runningExpr, runningCommitment, Expr.VarsBelow]
    have sourceBound := runningSource_lt_16 source
    have rowBound := commitmentRow_lt_22 row
    have coefficientBound := ringCoefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [runningCommitmentStart, priorRunningStart, runningCommitmentWords,
      ringDegree]
    omega
  · intro source column
    simp only [interface, runningExpr, runningPublicInput, Expr.VarsBelow]
    have sourceBound := runningSource_lt_16 source
    have columnBound := publicColumn_lt_270 column
    rw [phaseOffset_eq]
    norm_num [runningPublicStart, priorChildrenStart, expectedContextStart,
      expectedContextWords, runningPublicWords]
    omega
  · intro source coefficient
    change (runningEval_K source coefficient).VarsBelow phaseOffset
    simp only [runningEval_K, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have sourceBound := runningSource_lt_16 source
    have coefficientBound := coefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [runningEvalKStart, priorRunningStart, runningEvalKWords]
    omega
  · intro source matrix coefficient
    change (runningEval_A source matrix coefficient).VarsBelow phaseOffset
    simp only [runningEval_A, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have sourceBound := runningSource_lt_16 source
    have matrixBound := matrix_lt_4 matrix
    have coefficientBound := coefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [runningEvalAStart, priorRunningStart, runningEvalAWords]
    omega
  · intro source row coefficient
    simp only [interface, freshExpr, freshCommitment, Expr.VarsBelow]
    have sourceBound := freshSource_lt_1 source
    have rowBound := commitmentRow_lt_22 row
    have coefficientBound := ringCoefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [freshCommitmentStart, proofInputStart, priorChildrenStart,
      priorChildrenWords, expectedContextStart, expectedContextWords,
      freshCommitmentWords, ringDegree]
    omega
  · intro source column
    simp only [interface, freshExpr, freshPublicInput, Expr.VarsBelow]
    have columnBound := publicColumn_lt_270 column
    rw [phaseOffset_eq]
    norm_num [PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq]
    omega
  · intro roundIndex coefficient
    change (roundCoefficient roundIndex coefficient).VarsBelow phaseOffset
    simp only [roundCoefficient, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have roundBound := round_lt_28 roundIndex
    have coefficientBound := coefficient.isLt
    rw [phaseOffset_eq]
    norm_num [roundMessageStart, freshCommitmentStart,
      proofInputStart, priorChildrenStart, priorChildrenWords,
      expectedContextStart, expectedContextWords,
      freshCommitmentWords, productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape]
    omega
  · intro source coefficient
    change (outputEval_K source coefficient).VarsBelow phaseOffset
    simp only [outputEval_K, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have sourceBound := allSource_lt_17 source
    have coefficientBound := coefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [outputEvaluationStart, roundMessageStart,
      freshCommitmentStart, proofInputStart, priorChildrenStart,
      priorChildrenWords, freshCommitmentWords,
      expectedContextStart, expectedContextWords, roundMessageWords,
      ringDegree]
    omega
  · intro source matrix coefficient
    change (outputEval_A source matrix coefficient).VarsBelow phaseOffset
    simp only [outputEval_A, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have sourceBound := allSource_lt_17 source
    have matrixBound := matrix_lt_4 matrix
    have coefficientBound := coefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [outputEvaluationStart, roundMessageStart,
      freshCommitmentStart, proofInputStart, priorChildrenStart,
      priorChildrenWords, freshCommitmentWords,
      expectedContextStart, expectedContextWords, roundMessageWords,
      ringDegree]
    omega

/-- Every concrete external value is affine; every extension pair is a
nonconstant direct variable pair. -/
def externalInputsLinear
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.ExternalInputsLinear
      (interface logicalWidth publicFits) phaseOffset where
  below := externalInputsBelow logicalWidth publicFits
  priorState := fun _ => R1CS.isAffine_var _
  outputState := fun _ => R1CS.isAffine_var _
  expectedContext := fun _ => R1CS.isAffine_var _
  runningPoint := fun _ => pairAt_linear _
  runningCommitment := fun _ _ _ => R1CS.isAffine_var _
  runningPublicInput := fun _ _ => ⟨R1CS.isAffine_var _, fun _ equal => by cases equal⟩
  runningEval_K := fun _ _ => pairAt_linear _
  runningEval_A := fun _ _ _ => pairAt_linear _
  freshCommitment := fun _ _ _ => R1CS.isAffine_var _
  freshPublicInput := fun _ _ => R1CS.isAffine_var _
  roundCoefficient := fun _ _ => pairAt_linear _
  outputEval_K := fun _ _ => pairAt_linear _
  outputEval_A := fun _ _ _ => pairAt_linear _

end NightstreamFPrime.Layout.Stage1.PiCCSInputs
