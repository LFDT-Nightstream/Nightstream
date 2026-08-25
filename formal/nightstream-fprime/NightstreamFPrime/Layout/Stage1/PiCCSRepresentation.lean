import NightstreamFPrime.Layout.Stage1.PiCCSInputs

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS input evaluation claims.
Obligation: Prove that the zero-copy columns selected by `PiCCSInputs`
represent the typed prior running instance and fresh PiCCS statement.

Inputs:
- the authoritative pilot `protocolValues`;
- the fixed serialization in `Lifecycle.XOut`;
- the concrete symbolic PiCCS interface.

Outputs:
- field-by-field representation theorems;
- one theorem for the complete PiCCS semantic input.

Parent coverage:
- `Lifecycle.PiCCS.v1_1.StatementBinding.SpecHolds`;
- `Lifecycle.Stage1` pilot-to-PiCCS wiring.

This module proves value identity only. It adds no circuit row, column, or
witness value.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSRepresentation

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout.Stage1.PiCCSInputs

theorem finRange_flatMap_getD
    {count width : Nat}
    (encode : Fin count → List F)
    (encodedLength : ∀ index, (encode index).length = width)
    (position : Fin count)
    (inner : Nat)
    (innerBound : inner < width) :
    ((List.finRange count).flatMap encode).getD
        (position.val * width + inner) 0 =
      (encode position).getD inner 0 := by
  induction count with
  | zero => exact Fin.elim0 position
  | succ count inductionHypothesis =>
      rw [List.finRange_succ, List.flatMap_cons]
      refine Fin.cases ?_ (fun tail => ?_) position
      · simp only [Fin.val_zero, Nat.zero_mul, Nat.zero_add]
        rw [List.getD_append]
        rw [encodedLength]
        exact innerBound
      · simp only [Fin.val_succ]
        have offset :
            (tail.val + 1) * width + inner =
              width + (tail.val * width + inner) := by
          simp only [Nat.add_mul, Nat.one_mul]
          omega
        rw [List.getD_append_right]
        · rw [encodedLength]
          rw [offset]
          simp only [Nat.add_sub_cancel_left]
          rw [List.flatMap_map]
          exact inductionHypothesis (fun index => encode index.succ)
            (fun index => encodedLength index.succ) tail
        · rw [encodedLength]
          rw [offset]
          omega

private theorem serializeRingF_getD
    (value : RingF) (coefficient : Fin ringDegree) :
    (serializeRingF value).getD coefficient.val 0 = value coefficient := by
  unfold serializeRingF
  rw [List.getD_eq_get _ _
    ⟨coefficient.val, by simp⟩]
  simp only [List.get_eq_getElem, List.getElem_map,
    List.getElem_finRange, Fin.eta]
  apply congrArg value
  exact Fin.ext rfl

theorem serializeCommitment_getD
    (commitment : PaperAlgebra.Commitment)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    (serializeCommitment commitment).getD
        (row.val * ringDegree + coefficient.val) 0 =
      commitment row coefficient := by
  unfold serializeCommitment
  calc
    _ = (serializeRingF (commitment row)).getD coefficient.val 0 := by
      exact finRange_flatMap_getD
        (fun index => serializeRingF (commitment index))
        (fun index => serializeRingF_length (commitment index))
        row coefficient.val coefficient.isLt
    _ = _ := serializeRingF_getD (commitment row) coefficient

private def serializeRunningGroup
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount) : List F :=
  block (serializeCommitment (running.commitments source)) ++
    block (serializePublicInput (publicFits := publicFits)
      (running.publicInputs source)) ++
    block (serializeEvaluations (running.evaluations source))

private theorem serializeRunningGroup_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount) :
    (serializeRunningGroup running source).length = runningGroupWords := by
  simp [serializeRunningGroup, runningGroupWords, productionShape,
    productionProfile, FullShape, fullShape,
    Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree,
    Phi81MatrixSource.phi81Shape]

private theorem serializeRunningGroup_commitment_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    (serializeRunningGroup running source).getD
        (1 + row.val * ringDegree + coefficient.val) 0 =
      running.commitments source row coefficient := by
  unfold serializeRunningGroup
  rw [List.getD_append]
  · rw [List.getD_append]
    · change
        ([natWord (serializeCommitment
            (running.commitments source)).length] ++
          serializeCommitment (running.commitments source)).getD
            (1 + row.val * ringDegree + coefficient.val) 0 =
          running.commitments source row coefficient
      rw [List.getD_append_right]
      · simp only [List.length_singleton]
        have shifted :
            1 + row.val * ringDegree + coefficient.val - 1 =
              row.val * ringDegree + coefficient.val := by
          omega
        rw [shifted]
        exact serializeCommitment_getD
          (running.commitments source) row coefficient
      · simp only [List.length_singleton]
        omega
    · rw [block_length, serializeCommitment_length]
      have rowBound := row.isLt
      have coefficientBound := coefficient.isLt
      norm_num [productionProfile, ringDegree] at rowBound coefficientBound
      norm_num [productionProfile, ringDegree]
      omega
  · simp only [List.length_append, block_length,
      serializeCommitment_length, serializePublicInput_length]
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionProfile, ringDegree] at rowBound coefficientBound
    norm_num [productionProfile, FullShape, fullShape,
      Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree]
    omega

private theorem serializeRunning_commitment_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    (serializeRunning (publicFits := publicFits) running).getD
        (51 + source.val * runningGroupWords +
          (1 + row.val * ringDegree + coefficient.val)) 0 =
      running.commitments source row coefficient := by
  change
    (block (serializePoint running.point) ++
      (List.finRange productionShape.runningCount).flatMap
        (serializeRunningGroup running)).getD
          (51 + source.val * runningGroupWords +
            (1 + row.val * ringDegree + coefficient.val)) 0 =
      running.commitments source row coefficient
  rw [List.getD_append_right]
  · rw [block_length, serializePoint_length]
    have pointBlockWords : cubeVariables * 2 + 1 = 51 := by
      norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]
    rw [pointBlockWords]
    have shifted :
        51 + source.val * runningGroupWords +
              (1 + row.val * ringDegree + coefficient.val) - 51 =
          source.val * runningGroupWords +
            (1 + row.val * ringDegree + coefficient.val) := by
      omega
    rw [shifted]
    calc
      _ = (serializeRunningGroup running source).getD
          (1 + row.val * ringDegree + coefficient.val) 0 := by
        apply finRange_flatMap_getD
        · exact serializeRunningGroup_length running
        · have rowBound := row.isLt
          have coefficientBound := coefficient.isLt
          norm_num [runningGroupWords, productionProfile, ringDegree]
            at rowBound coefficientBound ⊢
          omega
      _ = _ := serializeRunningGroup_commitment_getD
        running source row coefficient
  · rw [block_length, serializePoint_length]
    have pointBlockWords : cubeVariables * 2 + 1 = 51 := by
      norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]
    rw [pointBlockWords]
    omega

private def runningCommitmentPayloadIndex
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) : Fin 42435 :=
  ⟨51 + source.val * runningGroupWords +
      (1 + row.val * ringDegree + coefficient.val), by
    have sourceBound := source.isLt
    have rowBound := row.isLt
    have coefficientBound := coefficient.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound rowBound
    norm_num [ringDegree] at coefficientBound
    norm_num [runningGroupWords, ringDegree]
    omega⟩

private theorem runningCommitmentIndex_eq_priorRunningIndex
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    runningCommitmentIndex source row coefficient =
      priorRunningIndex
        (runningCommitmentPayloadIndex source row coefficient) := by
  apply Fin.ext
  simp [runningCommitmentIndex, priorRunningIndex,
    runningCommitmentPayloadIndex, runningCommitmentStart,
    runningGroupStart, runningGroupsStart]
  omega

/-- One decoded commitment coefficient is exactly the matching coefficient
in the authoritative prior running instance. -/
theorem decodedRunning_protocolValues_commitmentCoordinate
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
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    (decodedRunning logicalWidth publicFits values).commitments
        source row coefficient =
      (prior.running functionIndex).commitments source row coefficient := by
  dsimp only
  change
    (PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed).priorPreimage
        (runningCommitmentIndex source row coefficient) =
      (prior.running functionIndex).commitments source row coefficient
  rw [runningCommitmentIndex_eq_priorRunningIndex]
  calc
    _ = (serializeRunning (publicFits := publicFits)
        (prior.running functionIndex)).getD
          (runningCommitmentPayloadIndex source row coefficient).val 0 :=
      protocolValues_runningWord prior priorPublic output digest
        priorFixed outputFixed digestFixed
          (runningCommitmentPayloadIndex source row coefficient)
    _ = _ := by
      exact serializeRunning_commitment_getD
        (prior.running functionIndex) source row coefficient

/-- All 16 decoded commitments are the commitments in the authoritative
prior running instance. -/
theorem decodedRunning_protocolValues_commitments
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
    (decodedRunning logicalWidth publicFits values).commitments =
      (prior.running functionIndex).commitments := by
  dsimp only
  funext source row coefficient
  exact decodedRunning_protocolValues_commitmentCoordinate
    prior priorPublic output digest priorFixed outputFixed digestFixed
      source row coefficient

private theorem publicColumn_lt_54
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    column.val < 54 := by
  have columnBound := column.isLt
  norm_num [FullShape, fullShape, Phi81Relation.Shape.publicWidth,
    publicRingColumns, ringDegree] at columnBound
  exact columnBound

private theorem serializePublicInput_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (publicInput : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (serializePublicInput (publicFits := publicFits) publicInput).getD
        column.val 0 =
      publicInput column := by
  unfold serializePublicInput
  rw [List.getD_eq_get _ _ ⟨column.val, by simp⟩]
  simp only [List.get_eq_getElem, List.getElem_map,
    List.getElem_finRange, Fin.eta]
  apply congrArg publicInput
  exact Fin.ext rfl

private theorem serializeRunningGroup_publicInput_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (serializeRunningGroup running source).getD
        (974 + column.val) 0 =
      running.publicInputs source column := by
  unfold serializeRunningGroup
  rw [List.getD_append]
  · rw [List.getD_append_right]
    · rw [block_length, serializeCommitment_length]
      have commitmentBlockWords :
          productionProfile.commitmentWidth * ringDegree + 1 = 973 := by
        norm_num [productionProfile, ringDegree]
      rw [commitmentBlockWords]
      have shifted : 974 + column.val - 973 = 1 + column.val := by
        omega
      rw [shifted]
      change
        ([natWord (serializePublicInput (publicFits := publicFits)
            (running.publicInputs source)).length] ++
          serializePublicInput (publicFits := publicFits)
            (running.publicInputs source)).getD
              (1 + column.val) 0 =
          running.publicInputs source column
      rw [List.getD_append_right]
      · simp only [List.length_singleton]
        have headerShifted : 1 + column.val - 1 = column.val := by
          omega
        rw [headerShifted]
        exact serializePublicInput_getD
          (running.publicInputs source) column
      · simp only [List.length_singleton]
        omega
    · rw [block_length, serializeCommitment_length]
      have commitmentBlockWords :
          productionProfile.commitmentWidth * ringDegree + 1 = 973 := by
        norm_num [productionProfile, ringDegree]
      rw [commitmentBlockWords]
      omega
  · simp only [List.length_append, block_length,
      serializeCommitment_length, serializePublicInput_length]
    have columnBound := publicColumn_lt_54 column
    norm_num [productionProfile, FullShape, fullShape,
      Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree]
    omega

private theorem serializeRunning_publicInput_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (serializeRunning (publicFits := publicFits) running).getD
        (51 + source.val * runningGroupWords + (974 + column.val)) 0 =
      running.publicInputs source column := by
  change
    (block (serializePoint running.point) ++
      (List.finRange productionShape.runningCount).flatMap
        (serializeRunningGroup running)).getD
          (51 + source.val * runningGroupWords + (974 + column.val)) 0 =
      running.publicInputs source column
  rw [List.getD_append_right]
  · rw [block_length, serializePoint_length]
    have pointBlockWords : cubeVariables * 2 + 1 = 51 := by
      norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]
    rw [pointBlockWords]
    have shifted :
        51 + source.val * runningGroupWords + (974 + column.val) - 51 =
          source.val * runningGroupWords + (974 + column.val) := by
      omega
    rw [shifted]
    calc
      _ = (serializeRunningGroup running source).getD
          (974 + column.val) 0 := by
        apply finRange_flatMap_getD
        · exact serializeRunningGroup_length running
        · have columnBound := publicColumn_lt_54 column
          norm_num [runningGroupWords]
          omega
      _ = _ := serializeRunningGroup_publicInput_getD
        running source column
  · rw [block_length, serializePoint_length]
    have pointBlockWords : cubeVariables * 2 + 1 = 51 := by
      norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]
    rw [pointBlockWords]
    omega

private def runningPublicInputPayloadIndex
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    Fin 42435 :=
  ⟨51 + source.val * runningGroupWords + (974 + column.val), by
    have sourceBound := source.isLt
    have columnBound := publicColumn_lt_54 column
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound
    norm_num [runningGroupWords]
    omega⟩

private theorem runningPublicInputIndex_eq_priorRunningIndex
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    runningPublicInputIndex source column =
      priorRunningIndex (runningPublicInputPayloadIndex source column) := by
  apply Fin.ext
  simp [runningPublicInputIndex, priorRunningIndex,
    runningPublicInputPayloadIndex, runningPublicStart,
    runningGroupStart, runningGroupsStart]
  omega

/-- One decoded public-input coordinate is exactly the matching coordinate
in the authoritative prior running instance. -/
theorem decodedRunning_protocolValues_publicInputCoordinate
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
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    (decodedRunning logicalWidth publicFits values).publicInputs
        source column =
      (prior.running functionIndex).publicInputs source column := by
  dsimp only
  change
    (PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed).priorPreimage
        (runningPublicInputIndex source column) =
      (prior.running functionIndex).publicInputs source column
  rw [runningPublicInputIndex_eq_priorRunningIndex]
  calc
    _ = (serializeRunning (publicFits := publicFits)
        (prior.running functionIndex)).getD
          (runningPublicInputPayloadIndex source column).val 0 :=
      protocolValues_runningWord prior priorPublic output digest
        priorFixed outputFixed digestFixed
          (runningPublicInputPayloadIndex source column)
    _ = _ := by
      exact serializeRunning_publicInput_getD
        (prior.running functionIndex) source column

/-- All 16 decoded public-input vectors are the public inputs in the
authoritative prior running instance. -/
theorem decodedRunning_protocolValues_publicInputs
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
    (decodedRunning logicalWidth publicFits values).publicInputs =
      (prior.running functionIndex).publicInputs := by
  dsimp only
  funext source column
  exact decodedRunning_protocolValues_publicInputCoordinate
    prior priorPublic output digest priorFixed outputFixed digestFixed
      source column

private theorem serializePadEvaluations_length
    (evaluations : StrongReduction.EvaluationFamily K productionShape) :
    ((List.finRange productionShape.coefficientCount).flatMap
      (fun coefficient => serializeK (evaluations.pad coefficient))).length =
        108 := by
  simp [productionShape, ringDegree, Phi81MatrixSource.phi81Shape]

theorem serializeEvaluations_evalK_getD
    (evaluations : StrongReduction.EvaluationFamily K productionShape)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeEvaluations evaluations).getD
        (coefficient.val * 2 + component.val) 0 =
      (serializeK (evaluations.pad coefficient)).getD component.val 0 := by
  unfold serializeEvaluations
  rw [List.getD_append]
  · exact finRange_flatMap_getD
      (fun index => serializeK (evaluations.pad index))
      (fun index => serializeK_length (evaluations.pad index))
      coefficient component.val component.isLt
  · rw [serializePadEvaluations_length]
    have coefficientBound := coefficient.isLt
    have componentBound := component.isLt
    norm_num [productionShape, ringDegree,
      Phi81MatrixSource.phi81Shape] at coefficientBound componentBound ⊢
    omega

private theorem serializeRunningGroup_evaluationPrefix_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount) :
    (block (serializeCommitment (running.commitments source)) ++
      block (serializePublicInput (publicFits := publicFits)
        (running.publicInputs source))).length = 1028 := by
  simp [productionProfile, FullShape, fullShape,
    Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree]

private theorem serializeRunningGroup_evalK_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeRunningGroup running source).getD
        (1029 + coefficient.val * 2 + component.val) 0 =
      (serializeK ((running.evaluations source).pad coefficient)).getD
        component.val 0 := by
  unfold serializeRunningGroup
  rw [List.getD_append_right]
  · rw [serializeRunningGroup_evaluationPrefix_length]
    have shifted :
        1029 + coefficient.val * 2 + component.val - 1028 =
          1 + coefficient.val * 2 + component.val := by
      omega
    rw [shifted]
    change
      ([natWord (serializeEvaluations
          (running.evaluations source)).length] ++
        serializeEvaluations (running.evaluations source)).getD
          (1 + coefficient.val * 2 + component.val) 0 =
        (serializeK ((running.evaluations source).pad coefficient)).getD
          component.val 0
    rw [List.getD_append_right]
    · simp only [List.length_singleton]
      have headerShifted :
          1 + coefficient.val * 2 + component.val - 1 =
            coefficient.val * 2 + component.val := by
        omega
      rw [headerShifted]
      exact serializeEvaluations_evalK_getD
        (running.evaluations source) coefficient component
    · simp only [List.length_singleton]
      omega
  · rw [serializeRunningGroup_evaluationPrefix_length]
    omega

private theorem serializeRunning_evalK_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeRunning (publicFits := publicFits) running).getD
        (51 + source.val * runningGroupWords +
          (1029 + coefficient.val * 2 + component.val)) 0 =
      (serializeK ((running.evaluations source).pad coefficient)).getD
        component.val 0 := by
  change
    (block (serializePoint running.point) ++
      (List.finRange productionShape.runningCount).flatMap
        (serializeRunningGroup running)).getD
          (51 + source.val * runningGroupWords +
            (1029 + coefficient.val * 2 + component.val)) 0 =
      (serializeK ((running.evaluations source).pad coefficient)).getD
        component.val 0
  rw [List.getD_append_right]
  · rw [block_length, serializePoint_length]
    have pointBlockWords : cubeVariables * 2 + 1 = 51 := by
      norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]
    rw [pointBlockWords]
    have shifted :
        51 + source.val * runningGroupWords +
              (1029 + coefficient.val * 2 + component.val) - 51 =
          source.val * runningGroupWords +
            (1029 + coefficient.val * 2 + component.val) := by
      omega
    rw [shifted]
    calc
      _ = (serializeRunningGroup running source).getD
          (1029 + coefficient.val * 2 + component.val) 0 := by
        apply finRange_flatMap_getD
        · exact serializeRunningGroup_length running
        · have coefficientBound := coefficient.isLt
          have componentBound := component.isLt
          norm_num [productionShape, ringDegree,
            Phi81MatrixSource.phi81Shape] at coefficientBound
          norm_num at componentBound
          norm_num [runningGroupWords]
          omega
      _ = _ := serializeRunningGroup_evalK_getD
        running source coefficient component
  · rw [block_length, serializePoint_length]
    have pointBlockWords : cubeVariables * 2 + 1 = 51 := by
      norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]
    rw [pointBlockWords]
    omega

private def runningEvalKPayloadIndex
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) : Fin 42435 :=
  ⟨51 + source.val * runningGroupWords +
      (1029 + coefficient.val * 2 + component.val), by
    have sourceBound := source.isLt
    have coefficientBound := coefficient.isLt
    have componentBound := component.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound
    norm_num [productionShape, ringDegree,
      Phi81MatrixSource.phi81Shape] at coefficientBound
    norm_num at componentBound
    norm_num [runningGroupWords]
    omega⟩

private theorem runningEvalKIndex_eq_priorRunningIndex
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    runningEval_KIndex source coefficient component =
      priorRunningIndex
        (runningEvalKPayloadIndex source coefficient component) := by
  apply Fin.ext
  simp [runningEval_KIndex, priorRunningIndex, runningEvalKPayloadIndex,
    runningEvaluationStart, runningGroupStart, runningGroupsStart]
  omega

private theorem decodedRunning_protocolValues_evalKComponent
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
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    values.priorPreimage (runningEval_KIndex source coefficient component) =
      (serializeK
        (((prior.running functionIndex).evaluations source).pad
          coefficient)).getD component.val 0 := by
  dsimp only
  rw [runningEvalKIndex_eq_priorRunningIndex]
  calc
    _ = (serializeRunning (publicFits := publicFits)
        (prior.running functionIndex)).getD
          (runningEvalKPayloadIndex source coefficient component).val 0 :=
      protocolValues_runningWord prior priorPublic output digest
        priorFixed outputFixed digestFixed
          (runningEvalKPayloadIndex source coefficient component)
    _ = _ := by
      exact serializeRunning_evalK_getD
        (prior.running functionIndex) source coefficient component

/-- One separate `Eval_K` coordinate is exactly the matching Pad evaluation
in the authoritative prior running instance. -/
theorem decodedRunning_protocolValues_evalKCoordinate
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
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    ((decodedRunning logicalWidth publicFits values).evaluations source).pad
        coefficient =
      ((prior.running functionIndex).evaluations source).pad coefficient := by
  dsimp only
  apply congrArg₂ K.mk
  · have componentEquality :=
      decodedRunning_protocolValues_evalKComponent prior priorPublic output
        digest priorFixed outputFixed digestFixed source coefficient 0
    simpa [serializeK] using componentEquality
  · have componentEquality :=
      decodedRunning_protocolValues_evalKComponent prior priorPublic output
        digest priorFixed outputFixed digestFixed source coefficient 1
    simpa [serializeK] using componentEquality

/-- The complete separate `Eval_K` family is the Pad evaluation family in
the authoritative prior running instance. -/
theorem decodedRunning_protocolValues_eval_K
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
    (fun source =>
      ((decodedRunning logicalWidth publicFits values).evaluations source).pad) =
      (fun source =>
        ((prior.running functionIndex).evaluations source).pad) := by
  dsimp only
  funext source coefficient
  exact decodedRunning_protocolValues_evalKCoordinate
    prior priorPublic output digest priorFixed outputFixed digestFixed
      source coefficient

private theorem serializeMatrixEvaluation_length
    (evaluations : StrongReduction.EvaluationFamily K productionShape)
    (matrix : Fin productionShape.matrixCount) :
    ((List.finRange productionShape.coefficientCount).flatMap
      (fun coefficient =>
        serializeK (evaluations.matrix matrix coefficient))).length = 108 := by
  simp [productionShape, ringDegree, Phi81MatrixSource.phi81Shape]

theorem serializeEvaluations_evalA_getD
    (evaluations : StrongReduction.EvaluationFamily K productionShape)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeEvaluations evaluations).getD
        (108 + matrix.val * 108 + coefficient.val * 2 + component.val) 0 =
      (serializeK (evaluations.matrix matrix coefficient)).getD
        component.val 0 := by
  unfold serializeEvaluations
  rw [List.getD_append_right]
  · rw [serializePadEvaluations_length]
    have shifted :
        108 + matrix.val * 108 + coefficient.val * 2 +
              component.val - 108 =
          matrix.val * 108 + coefficient.val * 2 + component.val := by
      omega
    rw [shifted]
    calc
      _ = ((List.finRange productionShape.coefficientCount).flatMap
          (fun index =>
            serializeK (evaluations.matrix matrix index))).getD
              (coefficient.val * 2 + component.val) 0 := by
        have coefficientBound := coefficient.isLt
        have componentBound := component.isLt
        have innerBound :
            coefficient.val * 2 + component.val < 108 := by
          norm_num [productionShape, ringDegree,
            Phi81MatrixSource.phi81Shape] at coefficientBound
          norm_num at componentBound
          omega
        simpa [Nat.add_assoc] using
          (finRange_flatMap_getD
            (fun index =>
              (List.finRange productionShape.coefficientCount).flatMap
                (fun coefficient =>
                  serializeK (evaluations.matrix index coefficient)))
            (serializeMatrixEvaluation_length evaluations) matrix
            (coefficient.val * 2 + component.val) innerBound)
      _ = _ := by
        exact finRange_flatMap_getD
          (fun index => serializeK (evaluations.matrix matrix index))
          (fun index => serializeK_length
            (evaluations.matrix matrix index))
          coefficient component.val component.isLt
  · rw [serializePadEvaluations_length]
    omega

private theorem serializeRunningGroup_evalA_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeRunningGroup running source).getD
        (1029 + 108 + matrix.val * 108 + coefficient.val * 2 +
          component.val) 0 =
      (serializeK
        ((running.evaluations source).matrix matrix coefficient)).getD
          component.val 0 := by
  unfold serializeRunningGroup
  rw [List.getD_append_right]
  · rw [serializeRunningGroup_evaluationPrefix_length]
    have shifted :
        1029 + 108 + matrix.val * 108 + coefficient.val * 2 +
              component.val - 1028 =
          1 + (108 + matrix.val * 108 + coefficient.val * 2 +
            component.val) := by
      omega
    rw [shifted]
    change
      ([natWord (serializeEvaluations
          (running.evaluations source)).length] ++
        serializeEvaluations (running.evaluations source)).getD
          (1 + (108 + matrix.val * 108 + coefficient.val * 2 +
            component.val)) 0 =
        (serializeK
          ((running.evaluations source).matrix matrix coefficient)).getD
            component.val 0
    rw [List.getD_append_right]
    · simp only [List.length_singleton]
      have headerShifted :
          1 + (108 + matrix.val * 108 + coefficient.val * 2 +
                component.val) - 1 =
            108 + matrix.val * 108 + coefficient.val * 2 +
              component.val := by
        omega
      rw [headerShifted]
      exact serializeEvaluations_evalA_getD
        (running.evaluations source) matrix coefficient component
    · simp only [List.length_singleton]
      omega
  · rw [serializeRunningGroup_evaluationPrefix_length]
    omega

private theorem serializeRunning_evalA_getD
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    (serializeRunning (publicFits := publicFits) running).getD
        (51 + source.val * runningGroupWords +
          (1029 + 108 + matrix.val * 108 + coefficient.val * 2 +
            component.val)) 0 =
      (serializeK
        ((running.evaluations source).matrix matrix coefficient)).getD
          component.val 0 := by
  change
    (block (serializePoint running.point) ++
      (List.finRange productionShape.runningCount).flatMap
        (serializeRunningGroup running)).getD
          (51 + source.val * runningGroupWords +
            (1029 + 108 + matrix.val * 108 + coefficient.val * 2 +
              component.val)) 0 =
      (serializeK
        ((running.evaluations source).matrix matrix coefficient)).getD
          component.val 0
  rw [List.getD_append_right]
  · rw [block_length, serializePoint_length]
    have pointBlockWords : cubeVariables * 2 + 1 = 51 := by
      norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]
    rw [pointBlockWords]
    have shifted :
        51 + source.val * runningGroupWords +
              (1029 + 108 + matrix.val * 108 + coefficient.val * 2 +
                component.val) - 51 =
          source.val * runningGroupWords +
            (1029 + 108 + matrix.val * 108 + coefficient.val * 2 +
              component.val) := by
      omega
    rw [shifted]
    calc
      _ = (serializeRunningGroup running source).getD
          (1029 + 108 + matrix.val * 108 + coefficient.val * 2 +
            component.val) 0 := by
        apply finRange_flatMap_getD
        · exact serializeRunningGroup_length running
        · have matrixBound := matrix.isLt
          have coefficientBound := coefficient.isLt
          have componentBound := component.isLt
          norm_num [productionShape, productionProfile,
            Phi81MatrixSource.phi81Shape] at matrixBound
          norm_num [productionShape, ringDegree,
            Phi81MatrixSource.phi81Shape] at coefficientBound
          norm_num at componentBound
          norm_num [runningGroupWords]
          omega
      _ = _ := serializeRunningGroup_evalA_getD
        running source matrix coefficient component
  · rw [block_length, serializePoint_length]
    have pointBlockWords : cubeVariables * 2 + 1 = 51 := by
      norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]
    rw [pointBlockWords]
    omega

private def runningEvalAPayloadIndex
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) : Fin 42435 :=
  ⟨51 + source.val * runningGroupWords +
      (1029 + 108 + matrix.val * 108 + coefficient.val * 2 +
        component.val), by
    have sourceBound := source.isLt
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    have componentBound := component.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound matrixBound
    norm_num [productionShape, ringDegree,
      Phi81MatrixSource.phi81Shape] at coefficientBound
    norm_num at componentBound
    norm_num [runningGroupWords]
    omega⟩

private theorem runningEvalAIndex_eq_priorRunningIndex
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    runningEval_AIndex source matrix coefficient component =
      priorRunningIndex
        (runningEvalAPayloadIndex source matrix coefficient component) := by
  apply Fin.ext
  simp [runningEval_AIndex, priorRunningIndex, runningEvalAPayloadIndex,
    runningEvaluationStart, runningGroupStart, runningGroupsStart]
  omega

private theorem decodedRunning_protocolValues_evalAComponent
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
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (component : Fin 2) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    values.priorPreimage
        (runningEval_AIndex source matrix coefficient component) =
      (serializeK
        (((prior.running functionIndex).evaluations source).matrix
          matrix coefficient)).getD component.val 0 := by
  dsimp only
  rw [runningEvalAIndex_eq_priorRunningIndex]
  calc
    _ = (serializeRunning (publicFits := publicFits)
        (prior.running functionIndex)).getD
          (runningEvalAPayloadIndex source matrix coefficient component).val
            0 :=
      protocolValues_runningWord prior priorPublic output digest
        priorFixed outputFixed digestFixed
          (runningEvalAPayloadIndex source matrix coefficient component)
    _ = _ := by
      exact serializeRunning_evalA_getD
        (prior.running functionIndex) source matrix coefficient component

/-- One separate `Eval_A` coordinate is exactly the matching CCS-matrix
evaluation in the authoritative prior running instance. -/
theorem decodedRunning_protocolValues_evalACoordinate
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
    (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    let values := PilotProduction.protocolValues prior priorPublic output digest
      priorFixed outputFixed digestFixed
    ((decodedRunning logicalWidth publicFits values).evaluations source).matrix
        matrix coefficient =
      ((prior.running functionIndex).evaluations source).matrix
        matrix coefficient := by
  dsimp only
  apply congrArg₂ K.mk
  · have componentEquality :=
      decodedRunning_protocolValues_evalAComponent prior priorPublic output
        digest priorFixed outputFixed digestFixed source matrix coefficient 0
    simpa [serializeK] using componentEquality
  · have componentEquality :=
      decodedRunning_protocolValues_evalAComponent prior priorPublic output
        digest priorFixed outputFixed digestFixed source matrix coefficient 1
    simpa [serializeK] using componentEquality

/-- The complete separate `Eval_A` family is the CCS-matrix evaluation
family in the authoritative prior running instance. -/
theorem decodedRunning_protocolValues_eval_A
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
    (fun source =>
      ((decodedRunning logicalWidth publicFits values).evaluations source).matrix) =
      (fun source =>
        ((prior.running functionIndex).evaluations source).matrix) := by
  dsimp only
  funext source matrix coefficient
  exact decodedRunning_protocolValues_evalACoordinate
    prior priorPublic output digest priorFixed outputFixed digestFixed
      source matrix coefficient

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

/-- The typed zero-copy decoder reconstructs the complete authoritative prior
running instance, with separate `Eval_K` and `Eval_A` families. -/
theorem decodedRunning_protocolValues
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
    decodedRunning logicalWidth publicFits values =
      prior.running functionIndex := by
  dsimp only
  apply running_ext
  · exact decodedRunning_protocolValues_point
      prior priorPublic output digest priorFixed outputFixed digestFixed
  · exact decodedRunning_protocolValues_commitments
      prior priorPublic output digest priorFixed outputFixed digestFixed
  · exact decodedRunning_protocolValues_publicInputs
      prior priorPublic output digest priorFixed outputFixed digestFixed
  · funext source
    apply evaluationFamily_ext
    · funext coefficient
      exact decodedRunning_protocolValues_evalKCoordinate
        prior priorPublic output digest priorFixed outputFixed digestFixed
          source coefficient
    · funext matrix coefficient
      exact decodedRunning_protocolValues_evalACoordinate
        prior priorPublic output digest priorFixed outputFixed digestFixed
          source matrix coefficient

/-- The symbolic running input used by the production PiCCS circuit evaluates
to the exact running instance carried by the pilot prior-state preimage. -/
theorem evalRunning_protocolEnv_eq_priorRunning
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
    StatementAbsorption.evalRunning (runningExpr logicalWidth publicFits)
        (PilotProduction.protocolEnv prior priorPublic output digest
          priorFixed outputFixed digestFixed) =
      prior.running functionIndex := by
  unfold PilotProduction.protocolEnv
  rw [evalRunning_eq_decodedRunning]
  exact decodedRunning_protocolValues prior priorPublic output digest
    priorFixed outputFixed digestFixed

private def encodingPreimage
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits) where
  verifierKeys := fun _ => [0, 0, 0, 0]
  iteration := 0
  z0 := [0, 0, 0, 0]
  current := [0, 0, 0, 0]
  running := fun _ => running
  pc := 1

private theorem encodingPreimage_fixed
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    PilotProduction.FixedPreimage (encodingPreimage running) := by
  exact ⟨rfl, rfl, rfl⟩

private def zeroPublicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun _ => 0

private def zeroDigest : Digest := [0, 0, 0, 0]

/-- The canonical running serializer is injective. The proof reuses the
production zero-copy decoder, which already proves every separate `Eval_K`
and `Eval_A` coordinate. -/
theorem serializeRunning_injective
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    Function.Injective
      (serializeRunning (logicalWidth := logicalWidth)
        (publicFits := publicFits)) := by
  intro left right encodedEqual
  let leftPreimage := encodingPreimage left
  let rightPreimage := encodingPreimage right
  let leftFixed := encodingPreimage_fixed left
  let rightFixed := encodingPreimage_fixed right
  let leftValues := PilotProduction.protocolValues leftPreimage
    zeroPublicInput leftPreimage zeroDigest leftFixed leftFixed rfl
  let rightValues := PilotProduction.protocolValues rightPreimage
    zeroPublicInput rightPreimage zeroDigest rightFixed rightFixed rfl
  have preimageEqual :
      serializePreimage (publicFits := publicFits) leftPreimage =
        serializePreimage (publicFits := publicFits) rightPreimage := by
    simp [leftPreimage, rightPreimage, encodingPreimage, serializePreimage,
      encodedEqual]
  have leftWords :
      List.ofFn leftValues.priorPreimage =
        serializePreimage (publicFits := publicFits) leftPreimage := by
    simpa [leftValues, PilotProduction.protocolValues] using
      PilotProduction.ofFn_fixedList
        (serializePreimage (publicFits := publicFits) leftPreimage)
        (PilotProduction.serializePreimage_length_fixed leftPreimage leftFixed)
  have rightWords :
      List.ofFn rightValues.priorPreimage =
        serializePreimage (publicFits := publicFits) rightPreimage := by
    simpa [rightValues, PilotProduction.protocolValues] using
      PilotProduction.ofFn_fixedList
        (serializePreimage (publicFits := publicFits) rightPreimage)
        (PilotProduction.serializePreimage_length_fixed rightPreimage rightFixed)
  have wordsEqual :
      List.ofFn leftValues.priorPreimage =
        List.ofFn rightValues.priorPreimage := by
    rw [leftWords, rightWords, preimageEqual]
  have priorValuesEqual :
      leftValues.priorPreimage = rightValues.priorPreimage := by
    funext index
    have selected := congrArg (fun words => words.getD index.val 0) wordsEqual
    simpa using selected
  have decodedEqual :
      decodedRunning logicalWidth publicFits leftValues =
        decodedRunning logicalWidth publicFits rightValues := by
    unfold decodedRunning
    rw [priorValuesEqual]
  calc
    left = decodedRunning logicalWidth publicFits leftValues := by
      symm
      simpa [leftValues, leftPreimage] using
        decodedRunning_protocolValues leftPreimage zeroPublicInput
          leftPreimage zeroDigest leftFixed leftFixed rfl
    _ = decodedRunning logicalWidth publicFits rightValues := decodedEqual
    _ = right := by
      simpa [rightValues, rightPreimage] using
        decodedRunning_protocolValues rightPreimage zeroPublicInput
          rightPreimage zeroDigest rightFixed rightFixed rfl

end NightstreamFPrime.Layout.Stage1.PiCCSRepresentation
