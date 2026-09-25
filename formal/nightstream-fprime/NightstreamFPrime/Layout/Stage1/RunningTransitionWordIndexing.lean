import NightstreamFPrime.Layout.Stage1.RunningTransitionData

/-! Structural indexing of the existing running-state expression serializer.
Each proof selects one framed block; no complete running list is expanded. -/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionWordIndexing

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StatementAbsorption

def component (value : KExpr) (part : Fin 2) : Expr :=
  if part.val = 0 then value.c0 else value.c1

@[simp] theorem component_zero (value : KExpr) : component value 0 = value.c0 := rfl
@[simp] theorem component_one (value : KExpr) : component value 1 = value.c1 := rfl

def pointIndex (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    RunningTransition.WordIndex :=
  ⟨PiCCSInputs.runningPointStart - PiCCSInputs.priorRunningStart +
    coordinate.val * 2 + part.val, by
    have coordinateBound : coordinate.val < 28 := coordinate.isLt
    have partBound := part.isLt
    change 1 + coordinate.val * 2 + part.val < 49353
    omega⟩

def groupIndex (source : Fin productionShape.runningCount)
    (position : Fin PiCCSInputs.runningGroupWords) : RunningTransition.WordIndex :=
  ⟨PiCCSInputs.runningGroupStart source.val - PiCCSInputs.priorRunningStart +
    position.val, by
    have sourceBound : source.val < 16 := source.isLt
    have positionBound : position.val < 3081 := position.isLt
    change 96 + source.val * 3081 - 39 + position.val < 49353
    omega⟩

@[simp] theorem pointIndex_val (coordinate : Fin productionShape.cubeVariables)
    (part : Fin 2) : (pointIndex coordinate part).val =
      1 + coordinate.val * 2 + part.val := by rfl

@[simp] theorem groupIndex_val (source : Fin productionShape.runningCount)
    (position : Fin PiCCSInputs.runningGroupWords) :
    (groupIndex source position).val = 57 + source.val * 3081 + position.val := by
  change 96 + source.val * 3081 - 39 + position.val = _
  omega

def commitmentPosition (row : Fin productionProfile.commitmentWidth)
    (coefficient : Fin ringDegree) : Fin PiCCSInputs.runningGroupWords :=
  ⟨PiCCSInputs.runningCommitmentStart 0 - PiCCSInputs.runningGroupStart 0 +
    row.val * ringDegree + coefficient.val, by
    have rowBound : row.val < 22 := row.isLt
    have coefficientBound : coefficient.val < 54 := coefficient.isLt
    change 1 + row.val * 54 + coefficient.val < 3081
    omega⟩

def publicPosition {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    Fin PiCCSInputs.runningGroupWords :=
  ⟨PiCCSInputs.runningPublicStart 0 - PiCCSInputs.runningGroupStart 0 + column.val, by
    have bound : column.val < 270 := column.isLt
    change 1190 + column.val < 3081
    omega⟩

def evalKPosition (coefficient : Fin productionShape.coefficientCount)
    (part : Fin 2) : Fin PiCCSInputs.runningGroupWords :=
  ⟨PiCCSInputs.runningEvaluationStart 0 - PiCCSInputs.runningGroupStart 0 +
    coefficient.val * 2 + part.val, by
    have bound : coefficient.val < 54 := coefficient.isLt
    have partBound := part.isLt
    change 1461 + coefficient.val * 2 + part.val < 3081
    omega⟩

def evalAPosition (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) (part : Fin 2) :
    Fin PiCCSInputs.runningGroupWords :=
  ⟨PiCCSInputs.runningEvaluationStart 0 - PiCCSInputs.runningGroupStart 0 +
    PiDECInputs.evalKWordsPerChild + matrix.val * PiDECInputs.evalKWordsPerChild +
      coefficient.val * 2 + part.val, by
    have matrixBound : matrix.val < 14 := matrix.isLt
    have coefficientBound : coefficient.val < 54 := coefficient.isLt
    have partBound := part.isLt
    change 1461 + 108 + matrix.val * 108 + coefficient.val * 2 + part.val < 3081
    omega⟩

private theorem finRange_flatMap_getD {Alpha : Type} {count width : Nat}
    (fallback : Alpha) (encode : Fin count → List Alpha)
    (lengths : ∀ index, (encode index).length = width)
    (outer : Fin count) (inner : Nat) (bound : inner < width) :
    ((List.finRange count).flatMap encode).getD (outer.val * width + inner) fallback =
      (encode outer).getD inner fallback := by
  induction count with
  | zero => exact Fin.elim0 outer
  | succ count ih =>
      rw [List.finRange_succ, List.flatMap_cons]
      refine Fin.cases ?_ (fun tail => ?_) outer
      · simp only [Fin.val_zero, Nat.zero_mul, Nat.zero_add]
        exact List.getD_append _ _ _ _ (by rw [lengths]; exact bound)
      · simp only [Fin.val_succ]
        have offset : (tail.val + 1) * width + inner =
            width + (tail.val * width + inner) := by
          simp only [Nat.add_mul, Nat.one_mul]
          omega
        rw [List.getD_append_right]
        · rw [lengths, offset, Nat.add_sub_cancel_left, List.flatMap_map]
          exact ih (fun index => encode index.succ)
            (fun index => lengths index.succ) tail
        · rw [lengths, offset]
          omega

private theorem finRange_map_getD {count : Nat} (value : Fin count → Expr)
    (index : Fin count) : ((List.finRange count).map value).getD index.val 0 =
      value index := by
  rw [List.getD_eq_get _ _ ⟨index.val, by simp⟩]
  simp only [List.get_eq_getElem, List.getElem_map, List.getElem_finRange]
  apply congrArg value
  exact Fin.ext rfl

private theorem k_getD (value : KExpr) (part : Fin 2) :
    (serializeKExpr value).getD part.val 0 = component value part := by
  fin_cases part <;> rfl

private theorem point_length (point : Fin productionShape.cubeVariables → KExpr) :
    (serializePointExpr point).length = 56 := by
  simp [serializePointExpr, serializeKExpr, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem commitment_length (value : Fin productionProfile.commitmentWidth →
    Fin ringDegree → Expr) : (serializeCommitmentExpr value).length = 1188 := by
  simp [serializeCommitmentExpr, productionProfile, ringDegree]

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem public_length (value : Fin (FullShape logicalWidth publicFits).publicWidth → Expr) :
    (serializePublicInputExpr value).length = 270 := by
  simp [serializePublicInputExpr, fullShape, Phi81Relation.Shape.publicWidth,
    publicRingColumns, ringDegree]

private theorem evaluation_length (value : EvaluationExpr) :
    (serializeEvaluationExpr value).length = 1620 := by
  simp [serializeEvaluationExpr, serializeKExpr, productionShape,
    productionProfile, Phi81MatrixSource.phi81Shape, ringDegree]

private def group (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount) : List Expr :=
  blockExpr (serializeCommitmentExpr (running.commitment source)) ++
    blockExpr (serializePublicInputExpr (running.publicInput source)) ++
    blockExpr (serializeEvaluationExpr (running.evaluation source))

private theorem group_length (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount) : (group running source).length = 3081 := by
  simp [group, blockExpr, commitment_length, public_length, evaluation_length]

private theorem group_word (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount) (position : Fin PiCCSInputs.runningGroupWords) :
    RunningTransition.runningWord running (groupIndex source position) =
      (group running source).getD position.val 0 := by
  unfold RunningTransition.runningWord
  rw [groupIndex_val, serializeRunningExpr, List.getD_append_right]
  · have pointLength : (blockExpr (serializePointExpr running.point)).length = 57 := by
      simp [blockExpr, point_length]
    rw [pointLength]
    have shift : 57 + source.val * 3081 + position.val - 57 =
        source.val * 3081 + position.val := by omega
    rw [shift]
    exact finRange_flatMap_getD 0 (group running) (group_length running)
      source position.val position.isLt
  · simp only [blockExpr, List.length_cons, point_length]
    omega

private theorem framed_word (before payload after : List Expr) (index : Nat)
    (bound : index < payload.length) :
    (before ++ blockExpr payload ++ after).getD (before.length + 1 + index) 0 =
      payload.getD index 0 := by
  rw [List.append_assoc, List.getD_append_right]
  · have shift : before.length + 1 + index - before.length = index + 1 := by omega
    rw [shift, blockExpr, List.cons_append, List.getD_cons_succ]
    exact List.getD_append _ _ _ _ bound
  · omega

private theorem framed_header (before payload after : List Expr) :
    (before ++ blockExpr payload ++ after).getD before.length 0 =
      Expr.const (natWord payload.length) := by
  rw [List.append_assoc, List.getD_append_right _ _ _ _ le_rfl, Nat.sub_self]
  rfl

theorem runningWord_pointHeader (running : RunningExpr logicalWidth publicFits) :
    RunningTransition.runningWord running ⟨0, by decide⟩ = Expr.const (natWord 56) := by
  unfold RunningTransition.runningWord serializeRunningExpr blockExpr
  simp only [List.cons_append, List.getD_cons_zero, point_length]

theorem runningWord_point (running : RunningExpr logicalWidth publicFits)
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    RunningTransition.runningWord running (pointIndex coordinate part) =
      component (running.point coordinate) part := by
  have coordinateBound : coordinate.val < 28 := coordinate.isLt
  have partBound := part.isLt
  unfold RunningTransition.runningWord
  rw [pointIndex_val, serializeRunningExpr, List.getD_append]
  · have shift : 1 + coordinate.val * 2 + part.val =
        (coordinate.val * 2 + part.val) + 1 := by omega
    rw [shift, blockExpr, List.getD_cons_succ, serializePointExpr]
    rw [finRange_flatMap_getD (width := 2) 0 _ (fun _ => rfl) coordinate part.val part.isLt]
    exact k_getD _ part
  · simp only [blockExpr, List.length_cons, point_length]
    omega

theorem runningWord_commitmentHeader (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount) :
    RunningTransition.runningWord running (groupIndex source ⟨0, by decide⟩) =
      Expr.const (natWord PiCCSInputs.runningCommitmentWords) := by
  rw [group_word]
  simp [group, blockExpr, commitment_length, PiCCSInputs.runningCommitmentWords]

theorem runningWord_publicHeader (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount) :
    RunningTransition.runningWord running (groupIndex source ⟨1189, by decide⟩) =
      Expr.const (natWord PiCCSInputs.runningPublicWords) := by
  rw [group_word]
  have selected := framed_header
    (blockExpr (serializeCommitmentExpr (running.commitment source)))
    (serializePublicInputExpr (running.publicInput source))
    (blockExpr (serializeEvaluationExpr (running.evaluation source)))
  simpa only [group, blockExpr, List.length_cons, commitment_length,
    public_length, PiCCSInputs.runningPublicWords] using selected

theorem runningWord_evaluationHeader (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount) :
    RunningTransition.runningWord running (groupIndex source ⟨1460, by decide⟩) =
      Expr.const (natWord PiCCSInputs.runningEvaluationWords) := by
  rw [group_word]
  have selected := framed_header
    (blockExpr (serializeCommitmentExpr (running.commitment source)) ++
      blockExpr (serializePublicInputExpr (running.publicInput source)))
    (serializeEvaluationExpr (running.evaluation source)) []
  simpa only [group, List.append_nil, List.length_append, blockExpr,
    List.length_cons, commitment_length, public_length, evaluation_length,
    PiCCSInputs.runningEvaluationWords] using selected

theorem runningWord_commitment (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth) (coefficient : Fin ringDegree) :
    RunningTransition.runningWord running
        (groupIndex source (commitmentPosition row coefficient)) =
      running.commitment source row coefficient := by
  rw [group_word]
  have rowBound : row.val < 22 := row.isLt
  have coefficientBound : coefficient.val < 54 := coefficient.isLt
  have selected := framed_word []
    (serializeCommitmentExpr (running.commitment source))
    (blockExpr (serializePublicInputExpr (running.publicInput source)) ++
      blockExpr (serializeEvaluationExpr (running.evaluation source)))
    (row.val * ringDegree + coefficient.val) (by
      rw [commitment_length]
      change row.val * 54 + coefficient.val < 1188
      omega)
  have leaf : (serializeCommitmentExpr (running.commitment source)).getD
      (row.val * ringDegree + coefficient.val) 0 =
        running.commitment source row coefficient := by
    unfold serializeCommitmentExpr
    rw [finRange_flatMap_getD 0 _ (fun _ => by simp) row
      coefficient.val coefficient.isLt]
    exact finRange_map_getD _ coefficient
  simpa only [group, commitmentPosition, PiCCSInputs.runningCommitmentStart,
    Nat.add_sub_cancel_left, List.nil_append, List.length_nil, Nat.zero_add,
    List.append_assoc, Nat.add_assoc] using selected.trans leaf

theorem runningWord_publicInput (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    RunningTransition.runningWord running (groupIndex source (publicPosition column)) =
      running.publicInput source column := by
  rw [group_word]
  have selected := framed_word
    (blockExpr (serializeCommitmentExpr (running.commitment source)))
    (serializePublicInputExpr (running.publicInput source))
    (blockExpr (serializeEvaluationExpr (running.evaluation source)))
    column.val (by simpa only [public_length] using! column.isLt)
  have leaf : (serializePublicInputExpr (running.publicInput source)).getD column.val 0 =
      running.publicInput source column := finRange_map_getD _ column
  simpa only [group, publicPosition, PiCCSInputs.runningPublicStart,
    Nat.add_sub_cancel_left, blockExpr, List.length_cons, commitment_length]
    using selected.trans leaf

private theorem evaluation_k (value : EvaluationExpr)
    (coefficient : Fin productionShape.coefficientCount) (part : Fin 2) :
    (serializeEvaluationExpr value).getD (coefficient.val * 2 + part.val) 0 =
      component (value.eval_K coefficient) part := by
  have coefficientBound : coefficient.val < 54 := coefficient.isLt
  have partBound := part.isLt
  rw [serializeEvaluationExpr, List.getD_append]
  · rw [finRange_flatMap_getD (width := 2) 0 _ (fun _ => rfl)
      coefficient part.val part.isLt]
    exact k_getD _ part
  · have length : ((List.finRange productionShape.coefficientCount).flatMap
        fun coefficient => serializeKExpr (value.eval_K coefficient)).length = 108 := by
      simp [serializeKExpr, productionShape, productionProfile, Phi81MatrixSource.phi81Shape, ringDegree]
    rw [length]
    omega

private theorem evaluation_a (value : EvaluationExpr)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) (part : Fin 2) :
    (serializeEvaluationExpr value).getD
      (108 + matrix.val * 108 + coefficient.val * 2 + part.val) 0 =
        component (value.eval_A matrix coefficient) part := by
  have coefficientBound : coefficient.val < 54 := coefficient.isLt
  have partBound := part.isLt
  have length (entries : Fin productionShape.coefficientCount → KExpr) :
      ((List.finRange productionShape.coefficientCount).flatMap
        fun coefficient => serializeKExpr (entries coefficient)).length = 108 := by
    simp [serializeKExpr, productionShape, productionProfile, Phi81MatrixSource.phi81Shape, ringDegree]
  rw [serializeEvaluationExpr, List.getD_append_right]
  · rw [length]
    have shift : 108 + matrix.val * 108 + coefficient.val * 2 + part.val - 108 =
        matrix.val * 108 + (coefficient.val * 2 + part.val) := by omega
    rw [shift, finRange_flatMap_getD 0 _ (fun matrix => length (value.eval_A matrix))
      matrix (coefficient.val * 2 + part.val) (by omega)]
    rw [finRange_flatMap_getD (width := 2) 0 _ (fun _ => rfl) coefficient part.val part.isLt]
    exact k_getD _ part
  · rw [length]
    omega

private theorem evaluation_word (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount) (index : Nat) (bound : index < 1620) :
    (group running source).getD (1461 + index) 0 =
      (serializeEvaluationExpr (running.evaluation source)).getD index 0 := by
  have selected := framed_word
    (blockExpr (serializeCommitmentExpr (running.commitment source)) ++
      blockExpr (serializePublicInputExpr (running.publicInput source)))
    (serializeEvaluationExpr (running.evaluation source)) [] index
    (by rwa [evaluation_length])
  simpa only [group, List.append_nil, List.length_append, blockExpr,
    List.length_cons, commitment_length, public_length] using selected

theorem runningWord_evalK (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) (part : Fin 2) :
    RunningTransition.runningWord running
        (groupIndex source (evalKPosition coefficient part)) =
      component ((running.evaluation source).eval_K coefficient) part := by
  rw [group_word]
  have coefficientBound : coefficient.val < 54 := coefficient.isLt
  have partBound := part.isLt
  change (group running source).getD (1461 + coefficient.val * 2 + part.val) 0 = _
  rw [Nat.add_assoc, evaluation_word running source _ (by omega)]
  exact evaluation_k _ coefficient part

theorem runningWord_evalA (running : RunningExpr logicalWidth publicFits)
    (source : Fin productionShape.runningCount) (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) (part : Fin 2) :
    RunningTransition.runningWord running
        (groupIndex source (evalAPosition matrix coefficient part)) =
      component ((running.evaluation source).eval_A matrix coefficient) part := by
  rw [group_word]
  have matrixBound : matrix.val < 14 := matrix.isLt
  have coefficientBound : coefficient.val < 54 := coefficient.isLt
  have partBound := part.isLt
  change (group running source).getD
    (1461 + 108 + matrix.val * 108 + coefficient.val * 2 + part.val) 0 = _
  have associate : 1461 + 108 + matrix.val * 108 + coefficient.val * 2 + part.val =
      1461 + (108 + matrix.val * 108 + coefficient.val * 2 + part.val) := by omega
  rw [associate, evaluation_word running source _ (by omega)]
  exact evaluation_a _ matrix coefficient part

private def zeroExpr : RunningExpr logicalWidth publicFits where
  point := fun _ => KExpr.zero
  commitment := fun _ _ _ => 0
  publicInput := fun _ _ => 0
  evaluation := fun _ => ⟨fun _ => KExpr.zero, fun _ _ => KExpr.zero⟩

private theorem zeroExpr_eval (env : Env) :
    evalRunning (zeroExpr (logicalWidth := logicalWidth) (publicFits := publicFits)) env =
      defaultRunning := by
  simp [evalRunning, zeroExpr, defaultRunning, evalPoint, zeroPoint, KExpr.eval,
    KExpr.zero, Expr.eval, evalEvaluation, List.ofFn_const, K.zero]
  constructor <;> rfl

private theorem defaultWord_eq (index : RunningTransition.WordIndex) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits) index =
      (RunningTransition.runningWord (zeroExpr (logicalWidth := logicalWidth)
        (publicFits := publicFits)) index).eval (fun _ => 0) := by
  rw [RunningTransition.runningWord_eval, zeroExpr_eval]
  rfl

theorem defaultWord_pointHeader :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits) ⟨0, by decide⟩ =
      natWord 56 := by
  rw [defaultWord_eq, runningWord_pointHeader]
  rfl

theorem defaultWord_point (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits)
      (pointIndex coordinate part) = 0 := by
  rw [defaultWord_eq, runningWord_point]
  simp [component, zeroExpr, KExpr.zero, Expr.eval]

theorem defaultWord_commitmentHeader (source : Fin productionShape.runningCount) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits)
      (groupIndex source ⟨0, by decide⟩) = natWord PiCCSInputs.runningCommitmentWords := by
  rw [defaultWord_eq, runningWord_commitmentHeader]
  rfl

theorem defaultWord_publicHeader (source : Fin productionShape.runningCount) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits)
      (groupIndex source ⟨1189, by decide⟩) = natWord PiCCSInputs.runningPublicWords := by
  rw [defaultWord_eq, runningWord_publicHeader]
  rfl

theorem defaultWord_evaluationHeader (source : Fin productionShape.runningCount) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits)
      (groupIndex source ⟨1460, by decide⟩) = natWord PiCCSInputs.runningEvaluationWords := by
  rw [defaultWord_eq, runningWord_evaluationHeader]
  rfl

theorem defaultWord_commitment (source : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth) (coefficient : Fin ringDegree) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits)
      (groupIndex source (commitmentPosition row coefficient)) = 0 := by
  rw [defaultWord_eq, runningWord_commitment]
  rfl

theorem defaultWord_publicInput (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits)
      (groupIndex source (publicPosition column)) = 0 := by
  rw [defaultWord_eq, runningWord_publicInput]
  rfl

theorem defaultWord_evalK (source : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) (part : Fin 2) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits)
      (groupIndex source (evalKPosition coefficient part)) = 0 := by
  rw [defaultWord_eq, runningWord_evalK]
  simp [component, zeroExpr, KExpr.zero, Expr.eval]

theorem defaultWord_evalA (source : Fin productionShape.runningCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) (part : Fin 2) :
    RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits)
      (groupIndex source (evalAPosition matrix coefficient part)) = 0 := by
  rw [defaultWord_eq, runningWord_evalA]
  simp [component, zeroExpr, KExpr.zero, Expr.eval]

end NightstreamFPrime.Layout.Stage1.RunningTransitionWordIndexing
