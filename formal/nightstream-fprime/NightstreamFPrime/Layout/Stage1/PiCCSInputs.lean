import Mathlib.Data.List.GetD
import NightstreamFPrime.Layout.PilotProduction
import NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS input and output messages.
Obligation: Own the concrete parent columns read by the production PiCCS
circuit.

The running instance reuses its exact serialization in the pilot prior-state
preimage. The fresh public input reuses the pilot public-input columns. Only
the fresh commitment, 25 degree-nine SumCheck messages, and separate output
`Eval_K`/`Eval_A` families allocate new proof-input columns.

No equality row is added at this boundary. The following PiCCS allocation
starts after this complete input interval.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Fixed start of `serializeRunning` inside the prior state preimage. -/
def priorRunningStart : Nat := 39

private theorem serializeKList_getD_c0
    (values : List K) (index : Nat) (bound : index < values.length) :
    (values.flatMap NightstreamFPrime.Lifecycle.serializeK).getD
        (index * 2) 0 =
      (values.getD index K.zero).c0 := by
  induction values generalizing index with
  | nil => simp at bound
  | cons value values inductionHypothesis =>
      cases index with
      | zero => rfl
      | succ index =>
          have tailBound : index < values.length := by
            simpa using bound
          rw [List.flatMap_cons]
          rw [List.getD_append_right]
          · change
              (values.flatMap NightstreamFPrime.Lifecycle.serializeK).getD
                  ((index + 1) * 2 - 2) 0 =
                (values.getD index K.zero).c0
            have shifted : (index + 1) * 2 - 2 = index * 2 := by
              omega
            rw [shifted]
            exact inductionHypothesis index tailBound
          · simp [NightstreamFPrime.Lifecycle.serializeK]

private theorem serializeKList_getD_c1
    (values : List K) (index : Nat) (bound : index < values.length) :
    (values.flatMap NightstreamFPrime.Lifecycle.serializeK).getD
        (index * 2 + 1) 0 =
      (values.getD index K.zero).c1 := by
  induction values generalizing index with
  | nil => simp at bound
  | cons value values inductionHypothesis =>
      cases index with
      | zero => rfl
      | succ index =>
          have tailBound : index < values.length := by
            simpa using bound
          rw [List.flatMap_cons]
          rw [List.getD_append_right]
          · change
              (values.flatMap NightstreamFPrime.Lifecycle.serializeK).getD
                  ((index + 1) * 2 + 1 - 2) 0 =
                (values.getD index K.zero).c1
            have shifted :
                (index + 1) * 2 + 1 - 2 = index * 2 + 1 := by
              omega
            rw [shifted]
            exact inductionHypothesis index tailBound
          · simp [NightstreamFPrime.Lifecycle.serializeK]
            omega

private def preRunningPrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  stateDomainTag ++ block (prior.verifierKeys functionIndex) ++
    [natWord prior.iteration] ++ block prior.z0 ++ block prior.current

private theorem serializePreimage_eq_runningPrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    serializePreimage (publicFits := publicFits) prior =
      preRunningPrefix prior ++ (
        serializeRunning (publicFits := publicFits)
          (prior.running functionIndex) ++ [natWord prior.pc]) := by
  simp [serializePreimage, preRunningPrefix, List.append_assoc]

private theorem preRunningPrefix_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage prior) :
    (preRunningPrefix prior).length = priorRunningStart := by
  rcases fixed with ⟨keyLength, z0Length, currentLength⟩
  norm_num [preRunningPrefix, stateDomainTag_length, keyLength, z0Length,
    currentLength, priorRunningStart, PilotProduction.digestWords,
    PilotValues.digestWords]

private theorem fixedList_apply_eq_getD
    {count : Nat} (values : List F) (lengthEquals : values.length = count)
    (index : Fin count) :
    PilotProduction.fixedList values lengthEquals index =
      values.getD index.val 0 := by
  unfold PilotProduction.fixedList
  symm
  exact List.getD_eq_get values 0 (Fin.cast lengthEquals.symm index)

private theorem serializePreimage_running_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage prior)
    (index : Nat)
    (bound : index <
      (serializeRunning (publicFits := publicFits)
        (prior.running functionIndex)).length) :
    (serializePreimage (publicFits := publicFits) prior).getD
        (priorRunningStart + index) 0 =
      (serializeRunning (publicFits := publicFits)
        (prior.running functionIndex)).getD index 0 := by
  rw [serializePreimage_eq_runningPrefix]
  rw [List.getD_append_right]
  · rw [preRunningPrefix_length prior fixed]
    simp [priorRunningStart]
    exact List.getD_append _ _ _ _ bound
  · rw [preRunningPrefix_length prior fixed]
    omega

private theorem serializeRunning_point_getD_c0
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (coordinate : Fin productionShape.cubeVariables) :
    (serializeRunning (publicFits := publicFits) running).getD
        (1 + coordinate.val * 2) 0 =
      (running.point.coordinates.getD coordinate.val K.zero).c0 := by
  unfold serializeRunning
  rw [List.getD_append]
  · change
      ([natWord (serializePoint running.point).length] ++
        serializePoint running.point).getD
          (1 + coordinate.val * 2) 0 =
        (running.point.coordinates.getD coordinate.val K.zero).c0
    rw [List.getD_append_right]
    simp only [List.length_singleton]
    have shifted :
        1 + coordinate.val * 2 - 1 = coordinate.val * 2 := by
      omega
    rw [shifted]
    change
      (running.point.coordinates.flatMap serializeK).getD
          (coordinate.val * 2) 0 =
        (running.point.coordinates.getD coordinate.val K.zero).c0
    apply serializeKList_getD_c0
    rw [running.point.dimension]
    exact coordinate.isLt
    simp
  · rw [block_length, serializePoint_length]
    have coordinateBound := coordinate.isLt
    norm_num [productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at coordinateBound ⊢
    omega

private theorem serializeRunning_point_getD_c1
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (coordinate : Fin productionShape.cubeVariables) :
    (serializeRunning (publicFits := publicFits) running).getD
        (1 + coordinate.val * 2 + 1) 0 =
      (running.point.coordinates.getD coordinate.val K.zero).c1 := by
  unfold serializeRunning
  rw [List.getD_append]
  · change
      ([natWord (serializePoint running.point).length] ++
        serializePoint running.point).getD
          (1 + coordinate.val * 2 + 1) 0 =
        (running.point.coordinates.getD coordinate.val K.zero).c1
    rw [List.getD_append_right]
    simp only [List.length_singleton]
    have shifted :
        1 + coordinate.val * 2 + 1 - 1 =
          coordinate.val * 2 + 1 := by
      omega
    rw [shifted]
    change
      (running.point.coordinates.flatMap serializeK).getD
          (coordinate.val * 2 + 1) 0 =
        (running.point.coordinates.getD coordinate.val K.zero).c1
    apply serializeKList_getD_c1
    rw [running.point.dimension]
    exact coordinate.isLt
    simp
  · rw [block_length, serializePoint_length]
    have coordinateBound := coordinate.isLt
    norm_num [productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at coordinateBound ⊢
    omega

/-- End of the completed pilot source-column interval and start of the
verifier-owned expected-context words. -/
def expectedContextStart : Nat := 12659088

def expectedContextWords : Nat := 4

def proofInputStart : Nat := expectedContextStart + expectedContextWords

theorem expectedContextStart_matches_pilot :
    expectedContextStart =
      Pilot.physicalColumnCount PilotProduction.interface
        PilotProduction.witnessOffset := by
  rw [PilotProduction.physicalColumnCount_eq]
  rfl

theorem expectedContextStart_eq : expectedContextStart = 12659088 := by
  rfl

theorem expectedContextWords_eq : expectedContextWords = 4 := by
  rfl

theorem proofInputStart_eq : proofInputStart = 12659092 := by
  rfl

/-- Fixed serialized running-state positions inside the prior preimage. -/
def runningPointStart : Nat := priorRunningStart + 1
def runningGroupsStart : Nat := priorRunningStart + 51
def runningGroupWords : Nat := 2649
def runningCommitmentWords : Nat := 972
def runningPublicWords : Nat := 54
def runningEvaluationWords : Nat := 1620

/-- A word position in the fixed serialized running-instance payload. -/
def priorRunningIndex (index : Fin 42435) :
    Fin PilotProduction.stateHashWords :=
  ⟨priorRunningStart + index.val, by
    have indexBound := index.isLt
    norm_num [priorRunningStart, PilotProduction.stateHashWords_eq] at *
    omega⟩

@[simp] theorem priorRunningIndex_val (index : Fin 42435) :
    (priorRunningIndex index).val = priorRunningStart + index.val := by
  rfl

def runningGroupStart (source : Nat) : Nat :=
  runningGroupsStart + source * runningGroupWords

def runningCommitmentStart (source : Nat) : Nat :=
  runningGroupStart source + 1

def runningPublicStart (source : Nat) : Nat :=
  runningGroupStart source + 974

def runningEvaluationStart (source : Nat) : Nat :=
  runningGroupStart source + 1029

/-- New proof-input intervals. -/
def freshCommitmentStart : Nat := proofInputStart
def freshCommitmentWords : Nat := 972
def roundMessageStart : Nat := freshCommitmentStart + freshCommitmentWords
def roundMessageWords : Nat := 500
def outputEvaluationStart : Nat := roundMessageStart + roundMessageWords
def outputEvaluationWords : Nat := 27540
def proofInputColumnCount : Nat :=
  freshCommitmentWords + roundMessageWords + outputEvaluationWords
def phaseOffset : Nat := proofInputStart + proofInputColumnCount

theorem freshCommitmentWords_eq :
    freshCommitmentWords = productionProfile.commitmentWidth * ringDegree := by
  norm_num [freshCommitmentWords, productionProfile, ringDegree]

theorem roundMessageWords_eq :
    roundMessageWords =
      productionShape.cubeVariables * (9 + 1) * 2 := by
  norm_num [roundMessageWords, productionShape, cubeVariables,
    Phi81MatrixSource.phi81Shape]

theorem outputEvaluationWords_eq :
    outputEvaluationWords =
      productionShape.sourceCount * (productionShape.matrixCount + 1) *
        productionShape.coefficientCount * 2 := by
  norm_num [outputEvaluationWords, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.sourceCount, ringDegree]

theorem proofInputColumnCount_eq : proofInputColumnCount = 29012 := by
  rfl

theorem phaseOffset_eq : phaseOffset = 12688104 := by
  rfl

def pairAt (start : Nat) : KExpr :=
  ⟨Expr.var start, Expr.var (start + 1)⟩

theorem pairAt_linear (start : Nat) : KExprLinear (pairAt start) := by
  refine ⟨rfl, rfl, ?_, ?_⟩ <;>
    simp [pairAt, Nonconstant]

/-- The pilot prior-preimage interval is an authoritative zero-copy word
source for the PiCCS running instance. -/
theorem eval_priorWord (values : PilotProduction.ExternalValues)
    (index : Fin PilotProduction.stateHashWords) :
    PilotProduction.loadExternal values index.val =
      values.priorPreimage index := by
  have inPrior := index.isLt
  change index.val < PilotProduction.priorPublicInputStart at inPrior
  simp [PilotProduction.loadExternal, inPrior]

def runningPointC0Index
    (coordinate : Fin productionShape.cubeVariables) :
    Fin PilotProduction.stateHashWords :=
  ⟨runningPointStart + coordinate.val * 2, by
    have coordinateBound := coordinate.isLt
    norm_num [runningPointStart, priorRunningStart,
      PilotProduction.stateHashWords_eq, productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at *
    omega⟩

def runningPointC1Index
    (coordinate : Fin productionShape.cubeVariables) :
    Fin PilotProduction.stateHashWords :=
  ⟨runningPointStart + coordinate.val * 2 + 1, by
    have coordinateBound := coordinate.isLt
    norm_num [runningPointStart, priorRunningStart,
      PilotProduction.stateHashWords_eq, productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at *
    omega⟩

def runningPoint
    (coordinate : Fin productionShape.cubeVariables) : KExpr :=
  pairAt (runningPointStart + coordinate.val * 2)

/-- Exact zero-copy evaluation of one running-point coordinate. -/
theorem runningPoint_eval_loadExternal
    (values : PilotProduction.ExternalValues)
    (coordinate : Fin productionShape.cubeVariables) :
    (runningPoint coordinate).eval (PilotProduction.loadExternal values) =
      ⟨values.priorPreimage (runningPointC0Index coordinate),
        values.priorPreimage (runningPointC1Index coordinate)⟩ := by
  apply congrArg₂ K.mk
  · exact eval_priorWord values (runningPointC0Index coordinate)
  · exact eval_priorWord values (runningPointC1Index coordinate)

def runningCommitment
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) : Expr :=
  Expr.var (runningCommitmentStart source.val + row.val * ringDegree +
    coefficient.val)

def runningCommitmentIndex
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    Fin PilotProduction.stateHashWords :=
  ⟨runningCommitmentStart source.val + row.val * ringDegree +
      coefficient.val, by
    have sourceBound := source.isLt
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    norm_num [runningCommitmentStart, runningGroupStart,
      runningGroupsStart, priorRunningStart, runningGroupWords,
      PilotProduction.stateHashWords_eq, productionShape,
      productionProfile, Phi81MatrixSource.phi81Shape, ringDegree] at *
    omega⟩

theorem runningCommitment_eval_loadExternal
    (values : PilotProduction.ExternalValues)
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    (runningCommitment source row coefficient).eval
        (PilotProduction.loadExternal values) =
      values.priorPreimage
        (runningCommitmentIndex source row coefficient) := by
  exact eval_priorWord values (runningCommitmentIndex source row coefficient)

def runningPublicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) : Expr :=
  Expr.var (runningPublicStart source.val + column.val)

def runningPublicInputIndex
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    Fin PilotProduction.stateHashWords :=
  ⟨runningPublicStart source.val + column.val, by
    have sourceBound := source.isLt
    have columnBound := column.isLt
    norm_num [runningPublicStart, runningGroupStart,
      runningGroupsStart, priorRunningStart, runningGroupWords,
      PilotProduction.stateHashWords_eq, productionShape,
      productionProfile, Phi81MatrixSource.phi81Shape, FullShape, fullShape,
      Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree] at *
    omega⟩

theorem runningPublicInput_eval_loadExternal
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (values : PilotProduction.ExternalValues)
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (runningPublicInput source column).eval
        (PilotProduction.loadExternal values) =
      values.priorPreimage (runningPublicInputIndex source column) := by
  exact eval_priorWord values (runningPublicInputIndex source column)

def runningEval_K
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  pairAt (runningEvaluationStart source.val + coefficient.val * 2)

def runningEval_KIndex
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) : Fin PilotProduction.stateHashWords :=
  ⟨runningEvaluationStart source.val + coefficient.val * 2 + component.val,
    by
      have sourceBound := source.isLt
      have coefficientBound := coefficient.isLt
      have componentBound := component.isLt
      norm_num [runningEvaluationStart, runningGroupStart,
        runningGroupsStart, priorRunningStart, runningGroupWords,
        PilotProduction.stateHashWords_eq, productionShape,
        productionProfile, Phi81MatrixSource.phi81Shape, ringDegree] at *
      omega⟩

theorem runningEval_K_eval_loadExternal
    (values : PilotProduction.ExternalValues)
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (runningEval_K source coefficient).eval
        (PilotProduction.loadExternal values) =
      ⟨values.priorPreimage (runningEval_KIndex source coefficient 0),
        values.priorPreimage (runningEval_KIndex source coefficient 1)⟩ := by
  apply congrArg₂ K.mk
  · exact eval_priorWord values (runningEval_KIndex source coefficient 0)
  · exact eval_priorWord values (runningEval_KIndex source coefficient 1)

def runningEval_A
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  pairAt (runningEvaluationStart source.val + 108 + matrix.val * 108 +
    coefficient.val * 2)

def runningEval_AIndex
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) : Fin PilotProduction.stateHashWords :=
  ⟨runningEvaluationStart source.val + 108 + matrix.val * 108 +
      coefficient.val * 2 + component.val, by
    have sourceBound := source.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    have componentBound := component.isLt
    norm_num [runningEvaluationStart, runningGroupStart,
      runningGroupsStart, priorRunningStart, runningGroupWords,
      PilotProduction.stateHashWords_eq, productionShape,
      productionProfile, Phi81MatrixSource.phi81Shape, ringDegree] at *
    omega⟩

theorem runningEval_A_eval_loadExternal
    (values : PilotProduction.ExternalValues)
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (runningEval_A source matrix coefficient).eval
        (PilotProduction.loadExternal values) =
      ⟨values.priorPreimage
          (runningEval_AIndex source matrix coefficient 0),
        values.priorPreimage
          (runningEval_AIndex source matrix coefficient 1)⟩ := by
  apply congrArg₂ K.mk
  · exact eval_priorWord values
      (runningEval_AIndex source matrix coefficient 0)
  · exact eval_priorWord values
      (runningEval_AIndex source matrix coefficient 1)

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

/-- Typed running instance decoded from the authoritative prior-preimage
word interval. -/
def decodedRunning
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (values : PilotProduction.ExternalValues) :
    NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Running K
      PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape where
  point := {
    coordinates := List.ofFn fun coordinate =>
      ⟨values.priorPreimage (runningPointC0Index coordinate),
        values.priorPreimage (runningPointC1Index coordinate)⟩
    dimension := by simp
  }
  commitments := fun source row coefficient =>
    values.priorPreimage (runningCommitmentIndex source row coefficient)
  publicInputs := fun source column =>
    values.priorPreimage (runningPublicInputIndex source column)
  evaluations := fun source => {
    pad := fun coefficient =>
      ⟨values.priorPreimage (runningEval_KIndex source coefficient 0),
        values.priorPreimage (runningEval_KIndex source coefficient 1)⟩
    matrix := fun matrix coefficient =>
      ⟨values.priorPreimage
          (runningEval_AIndex source matrix coefficient 0),
        values.priorPreimage
          (runningEval_AIndex source matrix coefficient 1)⟩
  }

private theorem protocolValues_priorPreimage_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage output)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (index : Fin PilotProduction.stateHashWords) :
    (PilotProduction.protocolValues prior priorPublic output digest
        priorFixed outputFixed digestFixed).priorPreimage index =
      (serializePreimage (publicFits := publicFits) prior).getD
        index.val 0 := by
  unfold PilotProduction.protocolValues
  exact fixedList_apply_eq_getD
    (serializePreimage (publicFits := publicFits) prior)
    (PilotProduction.serializePreimage_length_fixed prior priorFixed) index

/-- `protocolValues` exposes the exact serialized running payload without a
copy constraint or a second witness value. -/
theorem protocolValues_runningWord
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage output)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (index : Fin 42435) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    values.priorPreimage (priorRunningIndex index) =
      (serializeRunning (publicFits := publicFits)
        (prior.running functionIndex)).getD index.val 0 := by
  dsimp only
  rw [protocolValues_priorPreimage_getD]
  apply serializePreimage_running_getD prior priorFixed
  rw [serializeRunning_length]
  exact index.isLt

/-- One decoded point coordinate is exactly the matching coordinate in the
authoritative prior running instance. -/
theorem decodedRunning_protocolValues_pointCoordinate
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage output)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (coordinate : Fin productionShape.cubeVariables) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    (decodedRunning logicalWidth publicFits values).point.coordinates.getD
        coordinate.val K.zero =
      (prior.running functionIndex).point.coordinates.getD
        coordinate.val K.zero := by
  dsimp only
  change
    (List.ofFn (fun coordinate : Fin productionShape.cubeVariables =>
      ⟨(PilotProduction.protocolValues prior priorPublic output digest
          priorFixed outputFixed digestFixed).priorPreimage
            (runningPointC0Index coordinate),
        (PilotProduction.protocolValues prior priorPublic output digest
          priorFixed outputFixed digestFixed).priorPreimage
            (runningPointC1Index coordinate)⟩)).getD
        coordinate.val K.zero =
      (prior.running functionIndex).point.coordinates.getD
        coordinate.val K.zero
  rw [PriorStateHash.ofFn_getD]
  apply congrArg₂ K.mk
  · calc
      _ = (serializePreimage (publicFits := publicFits) prior).getD
          (runningPointC0Index coordinate).val 0 :=
        protocolValues_priorPreimage_getD prior priorPublic output digest
          priorFixed outputFixed digestFixed (runningPointC0Index coordinate)
      _ = (serializeRunning (publicFits := publicFits)
          (prior.running functionIndex)).getD
            (1 + coordinate.val * 2) 0 := by
        rw [show (runningPointC0Index coordinate).val =
          priorRunningStart + (1 + coordinate.val * 2) by
            simp [runningPointC0Index, runningPointStart]
            omega]
        apply serializePreimage_running_getD prior priorFixed
        rw [serializeRunning_length]
        have coordinateBound := coordinate.isLt
        norm_num [productionShape, cubeVariables,
          Phi81MatrixSource.phi81Shape] at coordinateBound ⊢
        omega
      _ = _ := serializeRunning_point_getD_c0
        (prior.running functionIndex) coordinate
  · calc
      _ = (serializePreimage (publicFits := publicFits) prior).getD
          (runningPointC1Index coordinate).val 0 :=
        protocolValues_priorPreimage_getD prior priorPublic output digest
          priorFixed outputFixed digestFixed (runningPointC1Index coordinate)
      _ = (serializeRunning (publicFits := publicFits)
          (prior.running functionIndex)).getD
            (1 + coordinate.val * 2 + 1) 0 := by
        rw [show (runningPointC1Index coordinate).val =
          priorRunningStart + (1 + coordinate.val * 2 + 1) by
            simp [runningPointC1Index, runningPointStart]
            omega]
        apply serializePreimage_running_getD prior priorFixed
        rw [serializeRunning_length]
        have coordinateBound := coordinate.isLt
        norm_num [productionShape, cubeVariables,
          Phi81MatrixSource.phi81Shape] at coordinateBound ⊢
        omega
      _ = _ := serializeRunning_point_getD_c1
        (prior.running functionIndex) coordinate

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The decoded point is the complete point in the authoritative prior
running instance. -/
theorem decodedRunning_protocolValues_point
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage output)
    (digestFixed : digest.length = PilotProduction.digestWords) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    (decodedRunning logicalWidth publicFits values).point =
      (prior.running functionIndex).point := by
  dsimp only
  apply cubePoint_eq_of_coordinates
  apply List.ext_get
  · simp [decodedRunning, (prior.running functionIndex).point.dimension]
  · intro index leftLt rightLt
    have indexBound : index < productionShape.cubeVariables := by
      simpa [decodedRunning] using leftLt
    let coordinate : Fin productionShape.cubeVariables :=
      ⟨index, indexBound⟩
    have equality := decodedRunning_protocolValues_pointCoordinate
      prior priorPublic output digest priorFixed outputFixed digestFixed
        coordinate
    dsimp only at equality
    rw [List.getD_eq_get _ _ ⟨index, leftLt⟩,
      List.getD_eq_get _ _ ⟨index, rightLt⟩] at equality
    simpa [coordinate, List.get_eq_getElem] using equality

private theorem evaluationFamily_eq
    (left right : StrongReduction.EvaluationFamily K productionShape)
    (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem running_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right :
      NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Running K
        PaperAlgebra.Commitment
        (PaperAlgebra.PublicInput
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        productionShape)
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp only [NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Running.mk.injEq]
  exact ⟨point, commitments, publicInputs, evaluations⟩

/-- The complete PiCCS running statement is exactly the typed zero-copy view
of the pilot prior-preimage interval. -/
theorem evalRunning_eq_decodedRunning
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (values : PilotProduction.ExternalValues) :
    StatementAbsorption.evalRunning
        (runningExpr logicalWidth publicFits)
        (PilotProduction.loadExternal values) =
      decodedRunning logicalWidth publicFits values := by
  let evaluated := StatementAbsorption.evalRunning
    (runningExpr logicalWidth publicFits)
    (PilotProduction.loadExternal values)
  let decoded := decodedRunning logicalWidth publicFits values
  have pointEq : evaluated.point = decoded.point := by
    apply cubePoint_eq_of_coordinates
    change List.ofFn (fun coordinate =>
      (runningPoint coordinate).eval (PilotProduction.loadExternal values)) =
      List.ofFn (fun coordinate =>
        ⟨values.priorPreimage (runningPointC0Index coordinate),
          values.priorPreimage (runningPointC1Index coordinate)⟩)
    apply congrArg List.ofFn
    funext coordinate
    exact runningPoint_eval_loadExternal values coordinate
  have commitmentEq : evaluated.commitments = decoded.commitments := by
    funext source row coefficient
    exact runningCommitment_eval_loadExternal values source row coefficient
  have publicEq : evaluated.publicInputs = decoded.publicInputs := by
    funext source column
    exact runningPublicInput_eval_loadExternal values source column
  have evaluationEq : evaluated.evaluations = decoded.evaluations := by
    funext source
    apply evaluationFamily_eq
    · funext coefficient
      exact runningEval_K_eval_loadExternal values source coefficient
    · funext matrix coefficient
      exact runningEval_A_eval_loadExternal values source matrix coefficient
  exact running_eq evaluated decoded pointEq commitmentEq publicEq evaluationEq

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
    (coefficient : Fin (9 + 1)) : KExpr :=
  pairAt (roundMessageStart + roundIndex.val * 20 + coefficient.val * 2)

def outputEval_K
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  pairAt (outputEvaluationStart + source.val * 1620 + coefficient.val * 2)

def outputEval_A
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  pairAt (outputEvaluationStart + source.val * 1620 + 108 +
    matrix.val * 108 + coefficient.val * 2)

def freshExpr
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    StatementAbsorption.FreshExpr logicalWidth publicFits where
  commitment := freshCommitment
  publicInput := freshPublicInput

def roundMessage (roundIndex : Fin productionShape.cubeVariables) :
    RoundTranscript.Message 9 where
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
    Formal.Interface logicalWidth 9 publicFits where
  baseOffset := phaseOffset
  priorState := fun _ => priorStateWord
  outputState := fun _ => outputStateWord
  expectedContext := fun _ => expectedContext
  running := fun _ => runningExpr logicalWidth publicFits
  fresh := fun _ => freshExpr logicalWidth publicFits
  round := fun _ => roundMessage
  output := fun _ => outputExpr

private theorem publicColumn_lt_54
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    column.val < 54 := by
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

private theorem round_lt_25
    (roundIndex : Fin productionShape.cubeVariables) :
    roundIndex.val < 25 := by
  have bound := roundIndex.isLt
  norm_num [productionShape, cubeVariables,
    Phi81MatrixSource.phi81Shape] at bound
  exact bound

private theorem matrix_lt_14
    (matrix : Fin productionShape.matrixCount) : matrix.val < 14 := by
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

private theorem commitmentRow_lt_18
    (row : Fin productionProfile.commitmentWidth) : row.val < 18 := by
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
  · intro coordinate
    change (pairAt (runningPointStart + coordinate.val * 2)).VarsBelow
      phaseOffset
    simp only [pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have coordinateBound := round_lt_25 coordinate
    rw [phaseOffset_eq]
    norm_num [runningPointStart, priorRunningStart]
    omega
  · intro source row coefficient
    simp only [interface, runningExpr, runningCommitment, Expr.VarsBelow]
    have sourceBound := runningSource_lt_16 source
    have rowBound := commitmentRow_lt_18 row
    have coefficientBound := ringCoefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [runningCommitmentStart, runningGroupStart,
      runningGroupsStart, priorRunningStart, runningGroupWords,
      ringDegree]
    omega
  · intro source column
    simp only [interface, runningExpr, runningPublicInput, Expr.VarsBelow]
    have sourceBound := runningSource_lt_16 source
    have columnBound := publicColumn_lt_54 column
    rw [phaseOffset_eq]
    norm_num [runningPublicStart, runningGroupStart,
      runningGroupsStart, priorRunningStart, runningGroupWords,
      ringDegree]
    omega
  · intro source coefficient
    change (runningEval_K source coefficient).VarsBelow phaseOffset
    simp only [runningEval_K, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have sourceBound := runningSource_lt_16 source
    have coefficientBound := coefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [runningEvaluationStart, runningGroupStart,
      runningGroupsStart, priorRunningStart, runningGroupWords,
      ringDegree]
    omega
  · intro source matrix coefficient
    change (runningEval_A source matrix coefficient).VarsBelow phaseOffset
    simp only [runningEval_A, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have sourceBound := runningSource_lt_16 source
    have matrixBound := matrix_lt_14 matrix
    have coefficientBound := coefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [runningEvaluationStart, runningGroupStart,
      runningGroupsStart, priorRunningStart, runningGroupWords,
      ringDegree]
    omega
  · intro source row coefficient
    simp only [interface, freshExpr, freshCommitment, Expr.VarsBelow]
    have sourceBound := freshSource_lt_1 source
    have rowBound := commitmentRow_lt_18 row
    have coefficientBound := ringCoefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [freshCommitmentStart, proofInputStart,
      expectedContextStart, expectedContextWords, freshCommitmentWords,
      ringDegree]
    omega
  · intro source column
    simp only [interface, freshExpr, freshPublicInput, Expr.VarsBelow]
    have columnBound := publicColumn_lt_54 column
    rw [phaseOffset_eq]
    change 42475 + column.val < 12688104
    omega
  · intro roundIndex coefficient
    change (roundCoefficient roundIndex coefficient).VarsBelow phaseOffset
    simp only [roundCoefficient, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have roundBound := round_lt_25 roundIndex
    have coefficientBound := coefficient.isLt
    rw [phaseOffset_eq]
    norm_num [roundMessageStart, freshCommitmentStart,
      proofInputStart, expectedContextStart, expectedContextWords,
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
      freshCommitmentStart, proofInputStart, freshCommitmentWords,
      expectedContextStart, expectedContextWords, roundMessageWords,
      ringDegree]
    omega
  · intro source matrix coefficient
    change (outputEval_A source matrix coefficient).VarsBelow phaseOffset
    simp only [outputEval_A, pairAt, KExpr.VarsBelow, Expr.VarsBelow]
    have sourceBound := allSource_lt_17 source
    have matrixBound := matrix_lt_14 matrix
    have coefficientBound := coefficient_lt_54 coefficient
    rw [phaseOffset_eq]
    norm_num [outputEvaluationStart, roundMessageStart,
      freshCommitmentStart, proofInputStart, freshCommitmentWords,
      expectedContextStart, expectedContextWords, roundMessageWords,
      ringDegree]
    omega

/-- Every concrete external value is affine; every extension pair is a
nonconstant direct variable pair. -/
def externalInputsLinear
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.ExternalInputsLinear
      (interface logicalWidth publicFits) phaseOffset where
  below := externalInputsBelow logicalWidth publicFits
  priorState := fun _ => R1CS.isAffine_var _
  outputState := fun _ => R1CS.isAffine_var _
  expectedContext := fun _ => R1CS.isAffine_var _
  runningPoint := fun _ => pairAt_linear _
  runningCommitment := fun _ _ _ => R1CS.isAffine_var _
  runningPublicInput := fun _ _ => R1CS.isAffine_var _
  runningEval_K := fun _ _ => pairAt_linear _
  runningEval_A := fun _ _ _ => pairAt_linear _
  freshCommitment := fun _ _ _ => R1CS.isAffine_var _
  freshPublicInput := fun _ _ => R1CS.isAffine_var _
  roundCoefficient := fun _ _ => pairAt_linear _
  outputEval_K := fun _ _ => pairAt_linear _
  outputEval_A := fun _ _ _ => pairAt_linear _

end NightstreamFPrime.Layout.Stage1.PiCCSInputs
