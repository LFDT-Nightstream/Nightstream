import NightstreamFPrime.Layout.Stage1.PiCCSRepresentation

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS prover messages.
Obligation: Give the concrete caller-owned PiCCS columns one typed value
source and one canonical encoding.

Inputs:
- one fresh Ajtai commitment;
- 25 degree-nine SumCheck coefficient vectors;
- separate output `Eval_K` and `Eval_A` families.

Outputs:
- a 29,012-word canonical proof-input encoding;
- one environment that preserves the pilot prefix and loads that encoding.

Parent coverage:
- `Lifecycle.PiCCS.v1_1.Formal.evalFresh`;
- `Lifecycle.PiCCS.v1_1.Formal.evalProof`.

This module owns external values only. It adds no constraint row and does not
derive a verifier challenge from witness data.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSProofInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout.Stage1.PiCCSInputs
open NightstreamFPrime.Layout.Stage1.PiCCSRepresentation

/-- Exactly the prover-owned PiCCS values on the production interface. -/
structure ProofValues where
  freshCommitment : PaperAlgebra.Commitment
  roundCoefficient : Fin productionShape.cubeVariables → Fin (9 + 1) → K
  outputEval_K : Fin productionShape.sourceCount →
    Fin productionShape.coefficientCount → K
  outputEval_A : Fin productionShape.sourceCount →
    Fin productionShape.matrixCount →
    Fin productionShape.coefficientCount → K

/-- The semantic fixed polynomial carried by one round message. -/
def roundPolynomial (values : ProofValues)
    (roundIndex : Fin productionShape.cubeVariables) :
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial K 9 where
  coefficients := List.ofFn (values.roundCoefficient roundIndex)
  coefficients_length := by simp

/-- The semantic v1_1 output with Pad and CCS matrix families kept separate. -/
def output (values : ProofValues) :
    FullOutputCoordinates.FullOutput K productionShape where
  padCoordinate := values.outputEval_K
  matrixCoordinate := values.outputEval_A

def outputEvaluation (values : ProofValues)
    (source : Fin productionShape.sourceCount) :
    StrongReduction.EvaluationFamily K productionShape where
  pad := values.outputEval_K source
  matrix := values.outputEval_A source

def serializeRounds (values : ProofValues) : List F :=
  (List.finRange productionShape.cubeVariables).flatMap fun roundIndex =>
    (List.finRange (9 + 1)).flatMap fun coefficient =>
      serializeK (values.roundCoefficient roundIndex coefficient)

def serializeOutput (values : ProofValues) : List F :=
  (List.finRange productionShape.sourceCount).flatMap fun source =>
    serializeEvaluations (outputEvaluation values source)

/-- Canonical external proof-input order. It is the exact order used by
`PiCCSInputs`: commitment, rounds, then separate output evaluation families. -/
def serializeProofInputs (values : ProofValues) : List F :=
  serializeCommitment values.freshCommitment ++
    serializeRounds values ++ serializeOutput values

theorem serializeRounds_length (values : ProofValues) :
    (serializeRounds values).length = roundMessageWords := by
  simp [serializeRounds, roundMessageWords, productionShape, cubeVariables,
    Phi81MatrixSource.phi81Shape]

theorem serializeOutput_length (values : ProofValues) :
    (serializeOutput values).length = outputEvaluationWords := by
  simp [serializeOutput, outputEvaluation, outputEvaluationWords,
    productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
    Shape.sourceCount, ringDegree]

theorem serializeProofInputs_length (values : ProofValues) :
    (serializeProofInputs values).length = proofInputColumnCount := by
  simp [serializeProofInputs, serializeRounds_length, serializeOutput_length,
    proofInputColumnCount, freshCommitmentWords, roundMessageWords,
    outputEvaluationWords, productionProfile, ringDegree]

/-- One combined source for the pilot columns and the new PiCCS proof-input
columns. -/
structure ExternalValues where
  pilot : PilotProduction.ExternalValues
  proof : ProofValues

/-- Load the pilot prefix unchanged, then the canonical PiCCS proof words. -/
def loadExternal (values : ExternalValues) : Env := fun index =>
  if index < proofInputStart then
    PilotProduction.loadExternal values.pilot index
  else
    (serializeProofInputs values.proof).getD (index - proofInputStart) 0

theorem eval_pilotPrefix (values : ExternalValues) (index : Nat)
    (bound : index < proofInputStart) :
    loadExternal values index =
      PilotProduction.loadExternal values.pilot index := by
  simp [loadExternal, bound]

theorem eval_proofWord (values : ExternalValues)
    (index : Fin proofInputColumnCount) :
    loadExternal values (proofInputStart + index.val) =
      (serializeProofInputs values.proof).getD index.val 0 := by
  unfold loadExternal
  split
  · omega
  · have shifted :
        proofInputStart + index.val - proofInputStart = index.val := by
      omega
    rw [shifted]

private def freshCommitmentWordIndex
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) : Fin proofInputColumnCount :=
  ⟨row.val * ringDegree + coefficient.val, by
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionProfile] at rowBound
    norm_num [ringDegree] at coefficientBound
    norm_num [proofInputColumnCount, freshCommitmentWords,
      roundMessageWords, outputEvaluationWords, productionProfile, ringDegree]
    omega⟩

private theorem serializeProofInputs_freshCommitment_getD
    (values : ProofValues)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    (serializeProofInputs values).getD
        (row.val * ringDegree + coefficient.val) 0 =
      values.freshCommitment row coefficient := by
  unfold serializeProofInputs
  rw [List.getD_append]
  · rw [List.getD_append]
    · exact serializeCommitment_getD
        values.freshCommitment row coefficient
    · rw [serializeCommitment_length]
      have rowBound := row.isLt
      have coefficientBound := coefficient.isLt
      norm_num [productionProfile] at rowBound
      norm_num [ringDegree] at coefficientBound
      norm_num [productionProfile, ringDegree]
      omega
  · simp only [List.length_append]
    rw [serializeCommitment_length, serializeRounds_length]
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionProfile] at rowBound
    norm_num [ringDegree] at coefficientBound
    norm_num [productionProfile, ringDegree, roundMessageWords]
    omega

/-- One fresh commitment coefficient evaluates from its canonical external
proof word. -/
theorem eval_freshCommitment
    (values : ExternalValues)
    (source : Fin productionShape.freshCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    (freshCommitment source row coefficient).eval (loadExternal values) =
      values.proof.freshCommitment row coefficient := by
  have sourceZero : source.val = 0 := by
    have sourceBound := source.isLt
    change source.val < 1 at sourceBound
    omega
  change loadExternal values
      (freshCommitmentStart + source.val * freshCommitmentWords +
        row.val * ringDegree + coefficient.val) =
    values.proof.freshCommitment row coefficient
  rw [sourceZero]
  simp only [Nat.zero_mul, Nat.add_zero]
  rw [show freshCommitmentStart = proofInputStart by rfl]
  rw [Nat.add_assoc]
  rw [show row.val * ringDegree + coefficient.val =
    (freshCommitmentWordIndex row coefficient).val by rfl]
  rw [eval_proofWord]
  exact serializeProofInputs_freshCommitment_getD
    values.proof row coefficient

/-- One fresh public-input coordinate reuses the authoritative pilot public
column. -/
theorem eval_freshPublicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (values : ExternalValues)
    (source : Fin productionShape.freshCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (freshPublicInput source column).eval (loadExternal values) =
      values.pilot.priorPublicInput column := by
  change loadExternal values
      (PilotProduction.priorPublicInputStart + column.val) =
    values.pilot.priorPublicInput column
  rw [eval_pilotPrefix]
  · have loaded := congrFun
      (PilotProduction.eval_priorPublicInput values.pilot) column
    simpa [PilotProduction.priorInterface_publicInput_apply,
      PilotProduction.priorPublicInput] using loaded
  · have columnBound := column.isLt
    norm_num [proofInputStart, expectedContextStart, expectedContextWords,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords,
      PilotProduction.digestWords,
      FullShape, fullShape, Phi81Relation.Shape.publicWidth,
      publicRingColumns, ringDegree] at columnBound ⊢
    omega

/-- The typed fresh PiCCS statement represented by the combined external
values. -/
def fresh
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (values : ExternalValues) :
    NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Fresh
      PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape where
  commitments := fun _ => values.proof.freshCommitment
  publicInputs := fun _ => values.pilot.priorPublicInput

private theorem fresh_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right :
      NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Fresh
        PaperAlgebra.Commitment
        (PaperAlgebra.PublicInput
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        productionShape)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs) : left = right := by
  cases left
  cases right
  simp_all

/-- The concrete symbolic fresh interface evaluates to the typed proof and
pilot values. -/
theorem evalFresh_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (values : ExternalValues) :
    Formal.evalFresh (interface logicalWidth publicFits) phaseOffset
        (loadExternal values) =
      fresh logicalWidth publicFits values := by
  apply fresh_ext
  · funext source row coefficient
    exact eval_freshCommitment values source row coefficient
  · funext source column
    exact eval_freshPublicInput values source column

private theorem serializeRound_length (values : ProofValues)
    (roundIndex : Fin productionShape.cubeVariables) :
    ((List.finRange (9 + 1)).flatMap fun coefficient =>
      serializeK (values.roundCoefficient roundIndex coefficient)).length =
        20 := by
  simp

private theorem serializeRounds_getD
    (values : ProofValues)
    (roundIndex : Fin productionShape.cubeVariables)
    (coefficient : Fin (9 + 1))
    (component : Fin 2) :
    (serializeRounds values).getD
        (roundIndex.val * 20 + coefficient.val * 2 + component.val) 0 =
      (serializeK
        (values.roundCoefficient roundIndex coefficient)).getD
          component.val 0 := by
  unfold serializeRounds
  have coefficientBound := coefficient.isLt
  have componentBound := component.isLt
  have innerBound : coefficient.val * 2 + component.val < 20 := by
    norm_num at coefficientBound componentBound
    omega
  calc
    _ = ((List.finRange (9 + 1)).flatMap fun index =>
        serializeK (values.roundCoefficient roundIndex index)).getD
          (coefficient.val * 2 + component.val) 0 := by
      simpa [Nat.add_assoc] using
        (finRange_flatMap_getD
          (fun index =>
            (List.finRange (9 + 1)).flatMap fun coefficient =>
              serializeK (values.roundCoefficient index coefficient))
          (serializeRound_length values) roundIndex
          (coefficient.val * 2 + component.val) innerBound)
    _ = _ := by
      exact finRange_flatMap_getD
        (fun index =>
          serializeK (values.roundCoefficient roundIndex index))
        (fun index => serializeK_length
          (values.roundCoefficient roundIndex index))
        coefficient component.val component.isLt

private theorem serializeProofInputs_round_getD
    (values : ProofValues)
    (roundIndex : Fin productionShape.cubeVariables)
    (coefficient : Fin (9 + 1))
    (component : Fin 2) :
    (serializeProofInputs values).getD
        (freshCommitmentWords + roundIndex.val * 20 +
          coefficient.val * 2 + component.val) 0 =
      (serializeK
        (values.roundCoefficient roundIndex coefficient)).getD
          component.val 0 := by
  unfold serializeProofInputs
  rw [List.getD_append]
  · rw [List.getD_append_right]
    · rw [serializeCommitment_length]
      have shifted :
          freshCommitmentWords + roundIndex.val * 20 +
                coefficient.val * 2 + component.val -
              productionProfile.commitmentWidth * ringDegree =
            roundIndex.val * 20 + coefficient.val * 2 + component.val := by
        norm_num [freshCommitmentWords, productionProfile, ringDegree]
        omega
      rw [shifted]
      exact serializeRounds_getD
        values roundIndex coefficient component
    · rw [serializeCommitment_length]
      norm_num [freshCommitmentWords, productionProfile, ringDegree]
      omega
  · simp only [List.length_append]
    rw [serializeCommitment_length, serializeRounds_length]
    have roundBound := roundIndex.isLt
    have coefficientBound := coefficient.isLt
    have componentBound := component.isLt
    norm_num [productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at roundBound
    norm_num at coefficientBound componentBound
    norm_num [freshCommitmentWords, productionProfile, ringDegree,
      roundMessageWords]
    omega

private def roundProofWordIndex
    (roundIndex : Fin productionShape.cubeVariables)
    (coefficient : Fin (9 + 1))
    (component : Fin 2) : Fin proofInputColumnCount :=
  ⟨freshCommitmentWords + roundIndex.val * 20 +
      coefficient.val * 2 + component.val, by
    have roundBound := roundIndex.isLt
    have coefficientBound := coefficient.isLt
    have componentBound := component.isLt
    norm_num [productionShape, cubeVariables,
      Phi81MatrixSource.phi81Shape] at roundBound
    norm_num at coefficientBound componentBound
    norm_num [proofInputColumnCount, freshCommitmentWords,
      roundMessageWords, outputEvaluationWords]
    omega⟩

private theorem eval_roundComponent
    (values : ExternalValues)
    (roundIndex : Fin productionShape.cubeVariables)
    (coefficient : Fin (9 + 1))
    (component : Fin 2) :
    loadExternal values
        (roundMessageStart + roundIndex.val * 20 +
          coefficient.val * 2 + component.val) =
      (serializeK
        (values.proof.roundCoefficient roundIndex coefficient)).getD
          component.val 0 := by
  rw [show roundMessageStart = proofInputStart + freshCommitmentWords by rfl]
  rw [show proofInputStart + freshCommitmentWords +
      roundIndex.val * 20 + coefficient.val * 2 + component.val =
    proofInputStart +
      (freshCommitmentWords + roundIndex.val * 20 +
        coefficient.val * 2 + component.val) by omega]
  rw [show freshCommitmentWords + roundIndex.val * 20 +
      coefficient.val * 2 + component.val =
    (roundProofWordIndex roundIndex coefficient component).val by rfl]
  rw [eval_proofWord]
  exact serializeProofInputs_round_getD
    values.proof roundIndex coefficient component

/-- One degree-nine round coefficient evaluates from the canonical round
segment. -/
theorem eval_roundCoefficient
    (values : ExternalValues)
    (roundIndex : Fin productionShape.cubeVariables)
    (coefficient : Fin (9 + 1)) :
    (roundCoefficient roundIndex coefficient).eval (loadExternal values) =
      values.proof.roundCoefficient roundIndex coefficient := by
  apply congrArg₂ K.mk
  · have componentEquality :=
      eval_roundComponent values roundIndex coefficient 0
    simpa [roundCoefficient, pairAt, serializeK] using componentEquality
  · have componentEquality :=
      eval_roundComponent values roundIndex coefficient 1
    simpa [roundCoefficient, pairAt, serializeK] using componentEquality

private theorem fixedPolynomial_ext {degree : Nat}
    (left right :
      NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial K degree)
    (coefficients : left.coefficients = right.coefficients) : left = right := by
  cases left
  cases right
  simp_all

/-- One symbolic round message evaluates to its constant-first semantic
degree-nine polynomial. -/
theorem eval_roundMessage
    (values : ExternalValues)
    (roundIndex : Fin productionShape.cubeVariables) :
    (roundMessage roundIndex).semanticPolynomial (loadExternal values) =
      roundPolynomial values.proof roundIndex := by
  apply fixedPolynomial_ext
  change
    (List.ofFn fun coefficient : Fin (9 + 1) =>
      (roundCoefficient roundIndex coefficient).eval
        (loadExternal values)) =
      List.ofFn (values.proof.roundCoefficient roundIndex)
  apply congrArg List.ofFn
  funext coefficient
  exact eval_roundCoefficient values roundIndex coefficient

/-- All 25 symbolic round messages evaluate to the typed proof values. -/
theorem eval_rounds (values : ExternalValues) :
    (fun roundIndex =>
      (roundMessage roundIndex).semanticPolynomial (loadExternal values)) =
      fun roundIndex => roundPolynomial values.proof roundIndex := by
  funext roundIndex
  exact eval_roundMessage values roundIndex

private theorem serializeOutputSource_length
    (values : ProofValues)
    (source : Fin productionShape.sourceCount) :
    (serializeEvaluations (outputEvaluation values source)).length =
      runningEvaluationWords := by
  rw [serializeEvaluations_length]
  norm_num [runningEvaluationWords, productionShape, productionProfile,
    ringDegree, Phi81MatrixSource.phi81Shape]

private theorem serializeOutput_evalK_getD
    (values : ProofValues)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeOutput values).getD
        (source.val * runningEvaluationWords +
          coefficient.val * 2 + component.val) 0 =
      (serializeK (values.outputEval_K source coefficient)).getD
        component.val 0 := by
  unfold serializeOutput
  have coefficientBound := coefficient.isLt
  have componentBound := component.isLt
  have innerBound : coefficient.val * 2 + component.val <
      runningEvaluationWords := by
    norm_num [productionShape, ringDegree,
      Phi81MatrixSource.phi81Shape] at coefficientBound
    norm_num at componentBound
    norm_num [runningEvaluationWords]
    omega
  calc
    _ = (serializeEvaluations (outputEvaluation values source)).getD
        (coefficient.val * 2 + component.val) 0 := by
      simpa [Nat.add_assoc] using
        (finRange_flatMap_getD
          (fun index => serializeEvaluations
            (outputEvaluation values index))
          (serializeOutputSource_length values) source
          (coefficient.val * 2 + component.val) innerBound)
    _ = _ := by
      exact serializeEvaluations_evalK_getD
        (outputEvaluation values source) coefficient component

private theorem serializeOutput_evalA_getD
    (values : ProofValues)
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeOutput values).getD
        (source.val * runningEvaluationWords + 108 + matrix.val * 108 +
          coefficient.val * 2 + component.val) 0 =
      (serializeK
        (values.outputEval_A source matrix coefficient)).getD
          component.val 0 := by
  unfold serializeOutput
  have matrixBound := matrix.isLt
  have coefficientBound := coefficient.isLt
  have componentBound := component.isLt
  have innerBound :
      108 + matrix.val * 108 + coefficient.val * 2 + component.val <
        runningEvaluationWords := by
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at matrixBound
    norm_num [productionShape, ringDegree,
      Phi81MatrixSource.phi81Shape] at coefficientBound
    norm_num at componentBound
    norm_num [runningEvaluationWords]
    omega
  calc
    _ = (serializeEvaluations (outputEvaluation values source)).getD
        (108 + matrix.val * 108 + coefficient.val * 2 + component.val) 0 := by
      simpa [Nat.add_assoc] using
        (finRange_flatMap_getD
          (fun index => serializeEvaluations
            (outputEvaluation values index))
          (serializeOutputSource_length values) source
          (108 + matrix.val * 108 + coefficient.val * 2 + component.val)
          innerBound)
    _ = _ := by
      exact serializeEvaluations_evalA_getD
        (outputEvaluation values source) matrix coefficient component

private theorem serializeProofInputs_output_getD
    (values : ProofValues)
    (index : Nat) :
    (serializeProofInputs values).getD
        (freshCommitmentWords + roundMessageWords + index) 0 =
      (serializeOutput values).getD index 0 := by
  unfold serializeProofInputs
  rw [List.getD_append_right]
  · simp only [List.length_append]
    rw [serializeCommitment_length, serializeRounds_length]
    have shifted :
        freshCommitmentWords + roundMessageWords + index -
              (productionProfile.commitmentWidth * ringDegree +
                roundMessageWords) =
            index := by
      norm_num [freshCommitmentWords, productionProfile, ringDegree]
    rw [shifted]
  · simp only [List.length_append]
    rw [serializeCommitment_length, serializeRounds_length]
    norm_num [freshCommitmentWords, productionProfile, ringDegree]

private theorem serializeProofInputs_outputEvalK_getD
    (values : ProofValues)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeProofInputs values).getD
        (freshCommitmentWords + roundMessageWords +
          source.val * runningEvaluationWords +
          coefficient.val * 2 + component.val) 0 =
      (serializeK (values.outputEval_K source coefficient)).getD
        component.val 0 := by
  rw [show freshCommitmentWords + roundMessageWords +
      source.val * runningEvaluationWords + coefficient.val * 2 +
        component.val =
    freshCommitmentWords + roundMessageWords +
      (source.val * runningEvaluationWords + coefficient.val * 2 +
        component.val) by omega]
  rw [serializeProofInputs_output_getD]
  exact serializeOutput_evalK_getD
    values source coefficient component

private theorem serializeProofInputs_outputEvalA_getD
    (values : ProofValues)
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeProofInputs values).getD
        (freshCommitmentWords + roundMessageWords +
          source.val * runningEvaluationWords + 108 + matrix.val * 108 +
          coefficient.val * 2 + component.val) 0 =
      (serializeK
        (values.outputEval_A source matrix coefficient)).getD
          component.val 0 := by
  rw [show freshCommitmentWords + roundMessageWords +
      source.val * runningEvaluationWords + 108 + matrix.val * 108 +
        coefficient.val * 2 + component.val =
    freshCommitmentWords + roundMessageWords +
      (source.val * runningEvaluationWords + 108 + matrix.val * 108 +
        coefficient.val * 2 + component.val) by omega]
  rw [serializeProofInputs_output_getD]
  exact serializeOutput_evalA_getD
    values source matrix coefficient component

private def outputEvalKProofWordIndex
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) : Fin proofInputColumnCount :=
  ⟨freshCommitmentWords + roundMessageWords +
      source.val * runningEvaluationWords +
      coefficient.val * 2 + component.val, by
    have sourceBound := source.isLt
    have coefficientBound := coefficient.isLt
    have componentBound := component.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape, Shape.sourceCount] at sourceBound
    norm_num [productionShape, ringDegree,
      Phi81MatrixSource.phi81Shape] at coefficientBound
    norm_num at componentBound
    norm_num [proofInputColumnCount, freshCommitmentWords,
      roundMessageWords, outputEvaluationWords, runningEvaluationWords]
    omega⟩

private def outputEvalAProofWordIndex
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) : Fin proofInputColumnCount :=
  ⟨freshCommitmentWords + roundMessageWords +
      source.val * runningEvaluationWords + 108 + matrix.val * 108 +
      coefficient.val * 2 + component.val, by
    have sourceBound := source.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    have componentBound := component.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape, Shape.sourceCount]
        at sourceBound matrixBound
    norm_num [productionShape, ringDegree,
      Phi81MatrixSource.phi81Shape] at coefficientBound
    norm_num at componentBound
    norm_num [proofInputColumnCount, freshCommitmentWords,
      roundMessageWords, outputEvaluationWords, runningEvaluationWords]
    omega⟩

private theorem eval_outputEvalKComponent
    (values : ExternalValues)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    loadExternal values
        (outputEvaluationStart + source.val * runningEvaluationWords +
          coefficient.val * 2 + component.val) =
      (serializeK (values.proof.outputEval_K source coefficient)).getD
        component.val 0 := by
  rw [show outputEvaluationStart =
    proofInputStart + freshCommitmentWords + roundMessageWords by rfl]
  rw [show proofInputStart + freshCommitmentWords + roundMessageWords +
      source.val * runningEvaluationWords + coefficient.val * 2 +
        component.val =
    proofInputStart +
      (freshCommitmentWords + roundMessageWords +
        source.val * runningEvaluationWords + coefficient.val * 2 +
          component.val) by omega]
  rw [show freshCommitmentWords + roundMessageWords +
      source.val * runningEvaluationWords + coefficient.val * 2 +
        component.val =
    (outputEvalKProofWordIndex source coefficient component).val by rfl]
  rw [eval_proofWord]
  exact serializeProofInputs_outputEvalK_getD
    values.proof source coefficient component

private theorem eval_outputEvalAComponent
    (values : ExternalValues)
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    loadExternal values
        (outputEvaluationStart + source.val * runningEvaluationWords + 108 +
          matrix.val * 108 + coefficient.val * 2 + component.val) =
      (serializeK
        (values.proof.outputEval_A source matrix coefficient)).getD
          component.val 0 := by
  rw [show outputEvaluationStart =
    proofInputStart + freshCommitmentWords + roundMessageWords by rfl]
  rw [show proofInputStart + freshCommitmentWords + roundMessageWords +
      source.val * runningEvaluationWords + 108 + matrix.val * 108 +
        coefficient.val * 2 + component.val =
    proofInputStart +
      (freshCommitmentWords + roundMessageWords +
        source.val * runningEvaluationWords + 108 + matrix.val * 108 +
          coefficient.val * 2 + component.val) by omega]
  rw [show freshCommitmentWords + roundMessageWords +
      source.val * runningEvaluationWords + 108 + matrix.val * 108 +
        coefficient.val * 2 + component.val =
    (outputEvalAProofWordIndex source matrix coefficient component).val by rfl]
  rw [eval_proofWord]
  exact serializeProofInputs_outputEvalA_getD
    values.proof source matrix coefficient component

/-- One output `Eval_K` coordinate evaluates from the separate Pad segment. -/
theorem eval_outputEval_K
    (values : ExternalValues)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (outputEval_K source coefficient).eval (loadExternal values) =
      values.proof.outputEval_K source coefficient := by
  apply congrArg₂ K.mk
  · have componentEquality :=
      eval_outputEvalKComponent values source coefficient 0
    simpa [outputEval_K, pairAt, serializeK] using componentEquality
  · have componentEquality :=
      eval_outputEvalKComponent values source coefficient 1
    simpa [outputEval_K, pairAt, serializeK] using componentEquality

/-- One output `Eval_A` coordinate evaluates from the separate CCS-matrix
segment. -/
theorem eval_outputEval_A
    (values : ExternalValues)
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (outputEval_A source matrix coefficient).eval (loadExternal values) =
      values.proof.outputEval_A source matrix coefficient := by
  apply congrArg₂ K.mk
  · have componentEquality :=
      eval_outputEvalAComponent values source matrix coefficient 0
    simpa [outputEval_A, pairAt, serializeK] using componentEquality
  · have componentEquality :=
      eval_outputEvalAComponent values source matrix coefficient 1
    simpa [outputEval_A, pairAt, serializeK] using componentEquality

private theorem fullOutput_ext
    (left right : FullOutputCoordinates.FullOutput K productionShape)
    (eval_K : left.padCoordinate = right.padCoordinate)
    (eval_A : left.matrixCoordinate = right.matrixCoordinate) : left = right := by
  cases left
  cases right
  simp_all

/-- The concrete symbolic output evaluates to the exact separate v1_1 output
families. -/
theorem evalOutput_eq (values : ExternalValues) :
    ∀ {logicalWidth : Nat}
      {publicFits : ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth logicalWidth},
    Formal.evalOutput (interface logicalWidth publicFits) phaseOffset
        (loadExternal values) =
      output values.proof := by
  intro logicalWidth publicFits
  apply fullOutput_ext
  · funext source coefficient
    exact eval_outputEval_K values source coefficient
  · funext source matrix coefficient
    exact eval_outputEval_A values source matrix coefficient

/-- Replace only the PiCCS-owned fields of the one semantic proof. -/
def proof (values : ProofValues) (template : Proof 9) : Proof 9 where
  piCcsRounds := roundPolynomial values
  piCcsOutput := output values
  piDecCommitments := template.piDecCommitments
  piDecEvaluations := template.piDecEvaluations

private def relationCoefficientIndex
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin (ProductionKey.degreeBound relation + 1)) : Fin (9 + 1) :=
  ⟨index.val, by
    exact index.isLt⟩

private def relationMessage
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (value : RoundTranscript.Message 9) :
    RoundTranscript.Message (ProductionKey.degreeBound relation) where
  coefficient := fun index =>
    value.coefficient (relationCoefficientIndex relation index)

private def relationRoundPolynomial
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (values : ProofValues)
    (roundIndex : Fin productionShape.cubeVariables) :
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial K
      (ProductionKey.degreeBound relation) where
  coefficients := List.ofFn fun coefficient =>
    values.roundCoefficient roundIndex
      (relationCoefficientIndex relation coefficient)
  coefficients_length := by simp

/-- Normalize only the degree-indexed round field. All other interface fields
stay definitionally equal to the concrete production interface. -/
def relationInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Formal.Interface logicalWidth (ProductionKey.degreeBound relation)
      publicFits where
  baseOffset := phaseOffset
  priorState := (interface logicalWidth publicFits).priorState
  outputState := (interface logicalWidth publicFits).outputState
  expectedContext := (interface logicalWidth publicFits).expectedContext
  running := (interface logicalWidth publicFits).running
  fresh := (interface logicalWidth publicFits).fresh
  round := fun offset roundIndex => relationMessage relation
    ((interface logicalWidth publicFits).round offset roundIndex)
  output := (interface logicalWidth publicFits).output

/-- Build the relation-typed proof directly from the fixed production proof
values. Only the PiDEC fields come from the template. -/
def relationProof
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (values : ProofValues) (template : Proof 9) :
    Proof (ProductionKey.degreeBound relation) where
  piCcsRounds := relationRoundPolynomial relation values
  piCcsOutput := output values
  piDecCommitments := template.piDecCommitments
  piDecEvaluations := template.piDecEvaluations

private theorem proof_ext
    {degree : Nat}
    (left right : Proof degree)
    (rounds : left.piCcsRounds = right.piCcsRounds)
    (piCcsOutput : left.piCcsOutput = right.piCcsOutput)
    (piDecCommitments : left.piDecCommitments = right.piDecCommitments)
    (piDecEvaluations : left.piDecEvaluations = right.piDecEvaluations) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem eval_relationMessage
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (values : ExternalValues)
    (roundIndex : Fin productionShape.cubeVariables) :
    (relationMessage relation (roundMessage roundIndex)).semanticPolynomial
        (loadExternal values) =
      relationRoundPolynomial relation values.proof roundIndex := by
  apply fixedPolynomial_ext
  change
    List.ofFn (fun coefficient =>
      (roundCoefficient roundIndex
        (relationCoefficientIndex relation coefficient)).eval
          (loadExternal values)) =
    List.ofFn (fun coefficient =>
      values.proof.roundCoefficient roundIndex
        (relationCoefficientIndex relation coefficient))
  apply congrArg List.ofFn
  funext coefficient
  exact eval_roundCoefficient values roundIndex
    (relationCoefficientIndex relation coefficient)

private theorem evalProof_round_apply
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (values : ExternalValues) (template : Proof 9)
    (roundIndex : Fin productionShape.cubeVariables) :
    (Formal.evalProof relation (relationInterface relation) phaseOffset
      (loadExternal values)
      (relationProof relation values.proof template)).piCcsRounds
        roundIndex =
      relationRoundPolynomial relation values.proof roundIndex :=
  eval_relationMessage relation values roundIndex

private theorem evalProof_output
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (values : ExternalValues) (template : Proof 9) :
    (Formal.evalProof relation (relationInterface relation) phaseOffset
      (loadExternal values)
      (relationProof relation values.proof template)).piCcsOutput =
      Formal.evalOutput (interface logicalWidth publicFits) phaseOffset
        (loadExternal values) := by
  rfl

private theorem evalRunning_relationInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    Formal.evalRunning (relationInterface relation) offset env =
      Formal.evalRunning (interface logicalWidth publicFits) offset env := by
  rfl

private theorem evalFresh_relationInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    Formal.evalFresh (relationInterface relation) offset env =
      Formal.evalFresh (interface logicalWidth publicFits) offset env := by
  rfl

/-- `Formal.evalProof` reads exactly the typed PiCCS rounds and separate
output families, while it leaves both later PiDEC fields unchanged. -/
theorem evalProof_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (values : ExternalValues)
    (template : Proof 9) :
    Formal.evalProof relation (relationInterface relation) phaseOffset
        (loadExternal values)
        (relationProof relation values.proof template) =
      relationProof relation values.proof template := by
  apply proof_ext
  · funext roundIndex
    exact evalProof_round_apply relation values template roundIndex
  · exact (evalProof_output relation values template).trans
      (evalOutput_eq values)
  · rfl
  · rfl

private theorem pilotIndex_beforeProof
    (index : Fin PilotProduction.stateHashWords) :
    index.val < proofInputStart := by
  have indexBound := index.isLt
  norm_num [proofInputStart, expectedContextStart, expectedContextWords,
    PilotProduction.stateHashWords_eq] at *
  omega

private theorem eval_runningPoint_eq_pilot
    (values : ExternalValues)
    (coordinate : Fin productionShape.cubeVariables) :
    (runningPoint coordinate).eval (loadExternal values) =
      (runningPoint coordinate).eval
        (PilotProduction.loadExternal values.pilot) := by
  apply congrArg₂ K.mk
  · exact eval_pilotPrefix values _
      (pilotIndex_beforeProof (runningPointC0Index coordinate))
  · exact eval_pilotPrefix values _
      (pilotIndex_beforeProof (runningPointC1Index coordinate))

private theorem eval_runningCommitment_eq_pilot
    (values : ExternalValues)
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    (runningCommitment source row coefficient).eval (loadExternal values) =
      (runningCommitment source row coefficient).eval
        (PilotProduction.loadExternal values.pilot) := by
  exact eval_pilotPrefix values _
    (pilotIndex_beforeProof
      (runningCommitmentIndex source row coefficient))

private theorem eval_runningPublicInput_eq_pilot
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (values : ExternalValues)
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (runningPublicInput source column).eval (loadExternal values) =
      (runningPublicInput source column).eval
        (PilotProduction.loadExternal values.pilot) := by
  exact eval_pilotPrefix values _
    (pilotIndex_beforeProof (runningPublicInputIndex source column))

private theorem eval_runningEvalK_eq_pilot
    (values : ExternalValues)
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (runningEval_K source coefficient).eval (loadExternal values) =
      (runningEval_K source coefficient).eval
        (PilotProduction.loadExternal values.pilot) := by
  apply congrArg₂ K.mk
  · exact eval_pilotPrefix values _
      (pilotIndex_beforeProof (runningEval_KIndex source coefficient 0))
  · exact eval_pilotPrefix values _
      (pilotIndex_beforeProof (runningEval_KIndex source coefficient 1))

private theorem eval_runningEvalA_eq_pilot
    (values : ExternalValues)
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (runningEval_A source matrix coefficient).eval (loadExternal values) =
      (runningEval_A source matrix coefficient).eval
        (PilotProduction.loadExternal values.pilot) := by
  apply congrArg₂ K.mk
  · exact eval_pilotPrefix values _
      (pilotIndex_beforeProof
        (runningEval_AIndex source matrix coefficient 0))
  · exact eval_pilotPrefix values _
      (pilotIndex_beforeProof
        (runningEval_AIndex source matrix coefficient 1))

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem evaluationFamily_ext
    (left right : StrongReduction.EvaluationFamily K productionShape)
    (eval_K : left.pad = right.pad)
    (eval_A : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem running_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp only [NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Running.mk.injEq]
  exact ⟨point, commitments, publicInputs, evaluations⟩

/-- Extending the pilot environment with proof-input words cannot change the
PiCCS running statement. -/
theorem evalRunning_eq_pilot
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (values : ExternalValues) :
    StatementAbsorption.evalRunning (runningExpr logicalWidth publicFits)
        (loadExternal values) =
      StatementAbsorption.evalRunning (runningExpr logicalWidth publicFits)
        (PilotProduction.loadExternal values.pilot) := by
  apply running_ext
  · apply cubePoint_ext
    change
      List.ofFn (fun coordinate =>
        (runningPoint coordinate).eval (loadExternal values)) =
      List.ofFn (fun coordinate =>
        (runningPoint coordinate).eval
          (PilotProduction.loadExternal values.pilot))
    apply congrArg List.ofFn
    funext coordinate
    exact eval_runningPoint_eq_pilot values coordinate
  · funext source row coefficient
    exact eval_runningCommitment_eq_pilot values source row coefficient
  · funext source column
    exact eval_runningPublicInput_eq_pilot values source column
  · funext source
    apply evaluationFamily_ext
    · funext coefficient
      exact eval_runningEvalK_eq_pilot values source coefficient
    · funext matrix coefficient
      exact eval_runningEvalA_eq_pilot values source matrix coefficient

/-- Concrete protocol values for the pilot plus one typed PiCCS proof input. -/
def protocolValues
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues) : ExternalValues where
  pilot := PilotProduction.protocolValues prior priorPublic outputPreimage
    digest priorFixed outputFixed digestFixed
  proof := proofValues

def protocolEnv
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues) : Env :=
  loadExternal (protocolValues prior priorPublic outputPreimage digest
    priorFixed outputFixed digestFixed proofValues)

/-- The combined protocol environment presents the exact authoritative prior
running instance to PiCCS. -/
theorem evalRunning_protocolEnv_eq_priorRunning
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues) :
    StatementAbsorption.evalRunning (runningExpr logicalWidth publicFits)
        (protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues) =
      prior.running functionIndex := by
  calc
    _ = StatementAbsorption.evalRunning
        (runningExpr logicalWidth publicFits)
        (PilotProduction.protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed) := by
      exact evalRunning_eq_pilot
        (protocolValues prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues)
    _ = _ := PiCCSRepresentation.evalRunning_protocolEnv_eq_priorRunning
      prior priorPublic outputPreimage digest priorFixed outputFixed digestFixed

/-- Exact semantic fresh instance: the proof commitment and the lifecycle
public input. -/
def protocolFresh
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proofValues : ProofValues) :
    NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Fresh
      PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape where
  commitments := fun _ => proofValues.freshCommitment
  publicInputs := fun _ => priorPublic

/-- Relation-typed parent view of the authoritative running instance. -/
theorem formalEvalRunning_protocolEnv_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues) :
    Formal.evalRunning (relationInterface relation) phaseOffset
        (protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues) =
      prior.running functionIndex := by
  rw [evalRunning_relationInterface]
  exact evalRunning_protocolEnv_eq_priorRunning prior priorPublic outputPreimage
    digest priorFixed outputFixed digestFixed proofValues

/-- Relation-typed parent view of the exact fresh instance. -/
theorem formalEvalFresh_protocolEnv_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues) :
    Formal.evalFresh (relationInterface relation) phaseOffset
        (protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues) =
      protocolFresh logicalWidth publicFits priorPublic proofValues := by
  rw [evalFresh_relationInterface]
  have evaluated := evalFresh_eq
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    (protocolValues prior priorPublic outputPreimage digest
      priorFixed outputFixed digestFixed proofValues)
  simpa [protocolEnv, protocolValues, fresh, protocolFresh] using evaluated

/-- Relation-typed parent view of the proof with only PiCCS fields replaced. -/
theorem formalEvalProof_protocolEnv_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues)
    (template : Proof 9) :
    Formal.evalProof relation (relationInterface relation) phaseOffset
        (protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues)
        (relationProof relation proofValues template) =
      relationProof relation proofValues template := by
  exact evalProof_eq relation
    (protocolValues prior priorPublic outputPreimage digest
      priorFixed outputFixed digestFixed proofValues) template

/-- Complete semantic coverage of every caller-owned value read by the
production PiCCS parent. -/
theorem protocolInputs_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues)
    (template : Proof 9) :
    Formal.evalRunning (relationInterface relation) phaseOffset
        (protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues) =
        prior.running functionIndex ∧
      Formal.evalFresh (relationInterface relation) phaseOffset
          (protocolEnv prior priorPublic outputPreimage digest
            priorFixed outputFixed digestFixed proofValues) =
        protocolFresh logicalWidth publicFits priorPublic proofValues ∧
      Formal.evalProof relation (relationInterface relation) phaseOffset
          (protocolEnv prior priorPublic outputPreimage digest
            priorFixed outputFixed digestFixed proofValues)
          (relationProof relation proofValues template) =
        relationProof relation proofValues template := by
  exact ⟨formalEvalRunning_protocolEnv_eq relation prior priorPublic
      outputPreimage digest priorFixed outputFixed digestFixed proofValues,
    formalEvalFresh_protocolEnv_eq relation prior priorPublic outputPreimage
      digest priorFixed outputFixed digestFixed proofValues,
    formalEvalProof_protocolEnv_eq relation prior priorPublic outputPreimage
      digest priorFixed outputFixed digestFixed proofValues template⟩

end NightstreamFPrime.Layout.Stage1.PiCCSProofInputs
