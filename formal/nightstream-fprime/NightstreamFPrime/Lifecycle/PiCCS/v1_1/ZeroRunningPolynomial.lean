import NightstreamFPrime.Lifecycle.Types
import NightstreamFPrime.Spec.ProductionRelation
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FullOutputCoordinates
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanReproduction

/-!
Owns the SuperNeo v1.1 Section 7.3 polynomial reduction for the production
shape with literal zero running assignments. All tables come from the same
connected matrix source. Matrix 13 stays a separate zero CCS matrix; Pad
retains its complete coefficient family. No CCS validity premise is used.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.ZeroRunningPolynomial

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open MatrixCoefficientSource PaperLinearAlgebra UnifiedSources
open FullOutputCoordinates

private theorem evaluate_zero_table {arity : Nat}
    (values : BooleanVertex arity → F)
    (zero : ∀ vertex, values vertex = 0)
    (point : CubePoint K arity) :
    (BooleanTable.tabulate fun vertex => K.embed (values vertex)).evaluate
      extensionOps point = K.zero := by
  have valuesEqual : (fun vertex => K.embed (values vertex)) =
      (fun _ => K.zero) := by
    funext vertex
    rw [zero vertex]
    exact embed_zero
  rw [valuesEqual]
  exact BooleanReproduction.evaluate_tabulate_constant
    extensionOps extensionLaws K.zero point

private theorem evaluate_zero_assignment {arity columns : Nat}
    (matrix : BooleanMatrix F arity columns)
    (assignment : Assignment F columns)
    (zero : ∀ column, assignment column = 0)
    (point : CubePoint K arity) :
    (BooleanTable.tabulate fun vertex =>
      K.embed (matrixVectorAt baseOps matrix assignment vertex)).evaluate
        extensionOps point = K.zero := by
  apply evaluate_zero_table
  intro vertex
  have assignmentEqual : assignment = fun _ => baseOps.zero := funext zero
  rw [assignmentEqual]
  exact matrixVectorAt_zero baseOps baseLaws matrix vertex

private theorem evaluate_zero_matrix {arity columns : Nat}
    (matrix : BooleanMatrix F arity columns)
    (zero : ∀ vertex column, matrix vertex column = 0)
    (assignment : Assignment F columns)
    (point : CubePoint K arity) :
    (BooleanTable.tabulate fun vertex =>
      K.embed (matrixVectorAt baseOps matrix assignment vertex)).evaluate
        extensionOps point = K.zero := by
  apply evaluate_zero_table
  intro vertex
  exact matrixVectorAt_zeroRow baseOps baseLaws matrix assignment vertex
    (zero vertex)

private theorem sumMap_zero {Index : Type}
    (indices : List Index) (term : Index → K)
    (zero : ∀ index, term index = K.zero) :
    SignedJointIdentity.sumMap extensionOps indices term = K.zero := by
  calc
    _ = FiniteSumAlgebra.sumMap extensionOps indices (fun _ => K.zero) :=
      FiniteSumAlgebra.sumMap_congr extensionOps indices term
        (fun _ => K.zero) (fun index _ => zero index)
    _ = _ := FiniteSumAlgebra.sumMap_zero extensionOps extensionLaws indices

private theorem sumMap_first {count : Nat}
    (term : Fin (count + 1) → K)
    (tailZero : ∀ index : Fin count, term index.succ = K.zero) :
    SignedJointIdentity.sumMap extensionOps
      (canonicalFinIndices (count + 1)) term = term 0 := by
  have indices : canonicalFinIndices (count + 1) =
      (0 : Fin (count + 1)) :: (canonicalFinIndices count).map Fin.succ := by
    simp only [canonicalFinIndices, List.ofFn_succ, List.map_ofFn,
      Function.comp_def, id_eq]
  rw [indices]
  change extensionOps.add (term 0)
    (SignedJointIdentity.sumMap extensionOps
      ((canonicalFinIndices count).map Fin.succ) term) = term 0
  have tail : SignedJointIdentity.sumMap extensionOps
      ((canonicalFinIndices count).map Fin.succ) term = K.zero := by
    simpa only [SignedJointIdentity.sumMap, List.map_map] using
      sumMap_zero (canonicalFinIndices count) (fun index => term index.succ) tailZero
  rw [tail]
  exact extensionLaws.add_zero _

private theorem sumMap_single (term : Fin 1 → K) :
    SignedJointIdentity.sumMap extensionOps (canonicalFinIndices 1) term =
      term 0 := by
  apply sumMap_first
  intro index
  exact Fin.elim0 index

private theorem gammaTerm_zero (gamma : K) (exponent : Nat) :
    SignedJointIdentity.gammaTerm extensionOps gamma exponent K.zero =
      K.zero := extensionLaws.mul_zero _

private theorem gammaTerm_zero_exponent (gamma value : K) :
    SignedJointIdentity.gammaTerm extensionOps gamma 0 value = value :=
  extensionLaws.one_mul value

private theorem gammaTerm_freshCount (gamma value : K) :
    SignedJointIdentity.gammaTerm extensionOps gamma productionShape.freshCount value =
      extensionOps.mul gamma value := by
  change extensionOps.mul (extensionOps.mul gamma extensionOps.one) value = _
  rw [extensionLaws.mul_one]

private theorem zero_add (value : K) :
    extensionOps.add K.zero value = value := extensionLaws.zero_add value

private theorem strictNorm_zero :
    ProtocolPolynomial.strictNormResidual extensionOps K.zero = K.zero := by
  change extensionOps.mul
    (extensionOps.mul (extensionOps.add extensionOps.zero extensionOps.one)
      extensionOps.zero)
    (extensionOps.sub extensionOps.zero extensionOps.one) = extensionOps.zero
  rw [extensionLaws.mul_zero, extensionLaws.mul_comm,
    extensionLaws.mul_zero]

private abbrev freshIndex : Fin productionShape.freshCount := ⟨0, by decide⟩

private abbrev sourceIndex : Fin productionShape.sourceCount :=
  freshSourceIndex freshIndex

variable {columns blockCount : Nat}
  (data : ConnectedInputs K productionShape columns blockCount)

/-- Every coefficient of both running output families is zero because each
family reads the literal zero source assignment. -/
theorem runningOutput_zero
    (runningZero : ∀ index column,
      data.assignments (runningSourceIndex index) column = 0)
    (point : CubePoint K cubeVariables)
    (index : Fin productionShape.runningCount) :
    (∀ coefficient,
      (FullOutput.honestAt baseOps extensionOps K.embed data point).padCoordinate
        (runningSourceIndex index) coefficient = K.zero) ∧
    (∀ matrix coefficient,
      (FullOutput.honestAt baseOps extensionOps K.embed data point).matrixCoordinate
        (runningSourceIndex index) matrix coefficient = K.zero) := by
  constructor
  · intro coefficient
    exact evaluate_zero_assignment _ _ (runningZero index) point
  · intro matrix coefficient
    exact evaluate_zero_assignment _ _ (runningZero index) point

private theorem zeroPort_coefficientMatrix
    (matrix13Zero : ∀ vertex column,
      data.matrixSource.matrices Spec.ProductionRelation.zeroPort vertex column = 0)
    (coefficient : Fin productionShape.coefficientCount)
    (vertex : BooleanVertex cubeVariables) (column : Fin columns) :
    data.matrixSource.coefficientMatrix baseOps Spec.ProductionRelation.zeroPort
      coefficient vertex column = 0 := by
  unfold MatrixSource.coefficientMatrix MatrixSource.coefficientMatrixOf
  apply sumRange_eq_zero baseOps baseLaws
  intro rowIndex rowLt
  rw [dif_pos rowLt]
  dsimp only
  have entryZero : data.matrixSource.paddedEntry baseOps
      (data.matrixSource.matrices Spec.ProductionRelation.zeroPort) vertex
      (data.matrixSource.columnLayout.decode column).1 ⟨rowIndex, rowLt⟩ = 0 := by
    unfold MatrixSource.paddedEntry
    cases data.matrixSource.columnLayout.encode?
        (data.matrixSource.columnLayout.decode column).1 ⟨rowIndex, rowLt⟩ with
    | none => rfl
    | some selected => exact matrix13Zero vertex selected
  rw [entryZero]
  change baseOps.mul baseOps.zero _ = baseOps.zero
  rw [baseLaws.mul_comm, baseLaws.mul_zero]

/-- The complete matrix-13 output is zero for every source and every ring
coefficient. This follows from its literal matrix entries and the existing
coefficient expansion, without a constant-term-only replacement. -/
theorem matrix13Output_zero
    (matrix13Zero : ∀ vertex column,
      data.matrixSource.matrices Spec.ProductionRelation.zeroPort vertex column = 0)
    (point : CubePoint K cubeVariables)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (FullOutput.honestAt baseOps extensionOps K.embed data point).matrixCoordinate
      source Spec.ProductionRelation.zeroPort coefficient = K.zero := by
  exact evaluate_zero_matrix _
    (zeroPort_coefficientMatrix data matrix13Zero coefficient) _ point

/-- The norm input of each running source is the MLE of its zero-padded
assignment, so it is zero at every field point. -/
theorem runningAssignment_zero
    (runningZero : ∀ index column,
      data.assignments (runningSourceIndex index) column = 0)
    (point : CubePoint K cubeVariables)
    (index : Fin productionShape.runningCount) :
    (ProtocolPolynomial.messageAt extensionOps
      (ProtocolDataRefinement.toProtocolData baseOps K.embed
        (data.toUnifiedInputs baseOps)) point).sourceAssignment
      (runningSourceIndex index) = K.zero := by
  change (BooleanTable.tabulate fun vertex => K.embed
    (data.cubeLayout.paddedValue 0
      (data.assignments (runningSourceIndex index)) vertex)).evaluate
        extensionOps point = K.zero
  apply evaluate_zero_table
  intro vertex
  unfold ColumnLayout.paddedValue
  cases data.cubeLayout.toColumn? vertex with
  | none => rfl
  | some column => exact runningZero index column

/-- The fresh scalar matrix-13 image is zero before the nonlinear CCS
polynomial is applied. -/
theorem freshMatrix13_zero
    (matrix13Zero : ∀ vertex column,
      data.matrixSource.matrices Spec.ProductionRelation.zeroPort vertex column = 0)
    (point : CubePoint K cubeVariables) :
    (ProtocolPolynomial.messageAt extensionOps
      (ProtocolDataRefinement.toProtocolData baseOps K.embed
        (data.toUnifiedInputs baseOps)) point).freshMatrixImage
      freshIndex Spec.ProductionRelation.zeroPort = K.zero := by
  exact evaluate_zero_matrix _ matrix13Zero _ point

private theorem padAtMessage_zero
    (runningZero : ∀ index column,
      data.assignments (runningSourceIndex index) column = 0)
    (point : CubePoint K cubeVariables) (gamma : K) :
    let protocol := ProtocolDataRefinement.toProtocolData baseOps K.embed
      (data.toUnifiedInputs baseOps)
    ProtocolPolynomial.padAtMessage extensionOps protocol.toVerifierInput gamma
      point (ProtocolPolynomial.messageAt extensionOps protocol point) = K.zero := by
  dsimp only
  unfold ProtocolPolynomial.padAtMessage
  have zero : ∀ coordinate : PadCoordinate productionShape,
      SignedJointIdentity.gammaTerm extensionOps gamma coordinate.localGammaExponent
        ((ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps K.embed
            (data.toUnifiedInputs baseOps)) point).padImage coordinate) = K.zero := by
    intro coordinate
    have valueZero := (runningOutput_zero data runningZero point coordinate.running).1
      coordinate.coefficient
    change SignedJointIdentity.gammaTerm extensionOps gamma coordinate.localGammaExponent
      ((FullOutput.honestAt baseOps extensionOps K.embed data point).padCoordinate
        (runningSourceIndex coordinate.running) coordinate.coefficient) = K.zero
    rw [valueZero]
    exact gammaTerm_zero gamma _
  rw [sumMap_zero _ _ zero]
  exact extensionLaws.mul_zero _

private theorem matrixAtMessage_zero
    (runningZero : ∀ index column,
      data.assignments (runningSourceIndex index) column = 0)
    (point : CubePoint K cubeVariables) (gamma : K) :
    let protocol := ProtocolDataRefinement.toProtocolData baseOps K.embed
      (data.toUnifiedInputs baseOps)
    ProtocolPolynomial.matrixAtMessage extensionOps protocol.toVerifierInput gamma
      point (ProtocolPolynomial.messageAt extensionOps protocol point) = K.zero := by
  dsimp only
  unfold ProtocolPolynomial.matrixAtMessage
  have zero : ∀ coordinate : MatrixCoordinate productionShape,
      SignedJointIdentity.gammaTerm extensionOps gamma coordinate.localGammaExponent
        ((ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps K.embed
            (data.toUnifiedInputs baseOps)) point).matrixImage coordinate) = K.zero := by
    intro coordinate
    have valueZero := (runningOutput_zero data runningZero point coordinate.running).2
      coordinate.matrix coordinate.coefficient
    change SignedJointIdentity.gammaTerm extensionOps gamma coordinate.localGammaExponent
      ((FullOutput.honestAt baseOps extensionOps K.embed data point).matrixCoordinate
        (runningSourceIndex coordinate.running) coordinate.matrix coordinate.coefficient) = K.zero
    rw [valueZero]
    exact gammaTerm_zero gamma _
  rw [sumMap_zero _ _ zero]
  exact extensionLaws.mul_zero _

private theorem normAtMessage_eq_fresh
    (runningZero : ∀ index column,
      data.assignments (runningSourceIndex index) column = 0)
    (point : CubePoint K cubeVariables) (gamma : K) :
    let message := ProtocolPolynomial.messageAt extensionOps
      (ProtocolDataRefinement.toProtocolData baseOps K.embed
        (data.toUnifiedInputs baseOps)) point
    ProtocolPolynomial.normAtMessage extensionOps gamma message =
      ProtocolPolynomial.strictNormResidual extensionOps
        (message.sourceAssignment sourceIndex) := by
  dsimp only
  let message := ProtocolPolynomial.messageAt extensionOps
    (ProtocolDataRefinement.toProtocolData baseOps K.embed
      (data.toUnifiedInputs baseOps)) point
  have tailZero : ∀ index : Fin 16,
      SignedJointIdentity.gammaTerm extensionOps gamma index.succ.val
        (ProtocolPolynomial.strictNormResidual extensionOps
          (message.sourceAssignment index.succ)) = K.zero := by
    intro index
    have indexEqual : (index.succ : Fin productionShape.sourceCount) =
        runningSourceIndex (shape := productionShape) index := by
      apply Fin.ext
      change index.val + 1 = 1 + index.val
      omega
    have valueZero : message.sourceAssignment (runningSourceIndex index) = K.zero :=
      runningAssignment_zero data runningZero point index
    rw [indexEqual, valueZero, strictNorm_zero]
    exact gammaTerm_zero gamma _
  exact (sumMap_first (count := 16)
    (fun source => SignedJointIdentity.gammaTerm extensionOps gamma source.val
      (ProtocolPolynomial.strictNormResidual extensionOps
        (message.sourceAssignment source))) tailZero).trans
    (gammaTerm_zero_exponent gamma _)

private theorem ccsAtMessage_eq_fresh
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (message : ProtocolPolynomial.OutputMessage K productionShape)
    (gamma : K) :
    ProtocolPolynomial.ccsAtMessage extensionOps input gamma message =
      CCSResidualTable.evaluatePolynomial extensionOps input.constraintPolynomial
        (message.freshMatrixImage freshIndex) := by
  calc
    _ = SignedJointIdentity.gammaTerm extensionOps gamma 0
        (CCSResidualTable.evaluatePolynomial extensionOps input.constraintPolynomial
          (message.freshMatrixImage freshIndex)) := sumMap_single _
    _ = _ := gammaTerm_zero_exponent gamma _

/-- The exact source-derived polynomial with zero running openings reduces
to one fresh CCS residual and one fresh strict-norm residual. The fixed
`gamma^12960` offset and zero matrix-13 slot are preserved. -/
theorem qAtPoint_eq_fresh
    (runningZero : ∀ index column,
      data.assignments (runningSourceIndex index) column = 0)
    (matrix13Zero : ∀ vertex column,
      data.matrixSource.matrices Spec.ProductionRelation.zeroPort vertex column = 0)
    (alpha point : CubePoint K cubeVariables) (gamma : K) :
    let protocol := ProtocolDataRefinement.toProtocolData baseOps K.embed
      (data.toUnifiedInputs baseOps)
    let message := ProtocolPolynomial.messageAt extensionOps protocol point
    ProtocolPolynomial.qAtPoint extensionOps protocol alpha gamma point =
      SignedJointIdentity.gammaTerm extensionOps gamma 12960
        (extensionOps.mul (SumCheckTruthPath.pointEquality extensionOps point alpha)
          (extensionOps.add
            (CCSResidualTable.evaluatePolynomial extensionOps protocol.constraintPolynomial
              (fun matrix => if matrix = Spec.ProductionRelation.zeroPort then K.zero
                else message.freshMatrixImage freshIndex matrix))
            (extensionOps.mul gamma
              (ProtocolPolynomial.strictNormResidual extensionOps
                (message.sourceAssignment sourceIndex))))) := by
  dsimp only
  unfold ProtocolPolynomial.qAtPoint ProtocolPolynomial.terminalFromMessage
  rw [padAtMessage_zero data runningZero point gamma,
    matrixAtMessage_zero data runningZero point gamma,
    gammaTerm_zero, zero_add, zero_add,
    ccsAtMessage_eq_fresh, normAtMessage_eq_fresh data runningZero point gamma,
    gammaTerm_freshCount]
  have images :
      (fun matrix => if matrix = Spec.ProductionRelation.zeroPort then K.zero
        else (ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps K.embed
            (data.toUnifiedInputs baseOps)) point).freshMatrixImage freshIndex matrix) =
      (ProtocolPolynomial.messageAt extensionOps
        (ProtocolDataRefinement.toProtocolData baseOps K.embed
          (data.toUnifiedInputs baseOps)) point).freshMatrixImage freshIndex := by
    funext matrix
    by_cases equal : matrix = Spec.ProductionRelation.zeroPort
    · rw [if_pos equal, equal, freshMatrix13_zero data matrix13Zero point]
    · rw [if_neg equal]
  exact congrArg
    (fun values : Fin productionShape.matrixCount → K =>
      SignedJointIdentity.gammaTerm extensionOps gamma 12960
        (extensionOps.mul (SumCheckTruthPath.pointEquality extensionOps point alpha)
          (extensionOps.add
            (CCSResidualTable.evaluatePolynomial extensionOps
              (ProtocolDataRefinement.toProtocolData baseOps K.embed
                (data.toUnifiedInputs baseOps)).constraintPolynomial values)
            (extensionOps.mul gamma
              (ProtocolPolynomial.strictNormResidual extensionOps
                ((ProtocolPolynomial.messageAt extensionOps
                  (ProtocolDataRefinement.toProtocolData baseOps K.embed
                    (data.toUnifiedInputs baseOps)) point).sourceAssignment
                  sourceIndex)))))) images.symm

private theorem foldr_map_zero {Index : Type}
    (indices : List Index) (term : Index → K)
    (zero : ∀ index, term index = K.zero) :
    (indices.map term).foldr extensionOps.add K.zero = K.zero := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      rw [List.map_cons, List.foldr_cons, zero index, inductionHypothesis]
      exact extensionLaws.zero_add K.zero

/-- The verifier's initial claim is zero from the literal public coefficient
fields alone. This theorem does not use an assignment or its validity. -/
theorem initial_zero
    (padZero : ∀ coordinate, data.claimedPadCoefficient coordinate = K.zero)
    (matrixZero : ∀ coordinate, data.claimedMatrixCoefficient coordinate = K.zero)
    (gamma : K) :
    (ProtocolDataRefinement.toProtocolData baseOps K.embed
      (data.toUnifiedInputs baseOps)).toVerifierInput.initial extensionOps gamma =
      K.zero := by
  let coefficients : TargetPolynomial.TargetCoefficients K productionShape :=
    { pad := data.claimedPadCoefficient, matrix := data.claimedMatrixCoefficient }
  have pad : TargetPolynomial.evaluatePad extensionOps.toOps coefficients gamma =
      K.zero := by
    rw [TargetPolynomial.evaluatePad_eq_foldr]
    apply foldr_map_zero
    intro coordinate
    change extensionOps.mul _ (data.claimedPadCoefficient coordinate) = K.zero
    rw [padZero coordinate]
    exact extensionLaws.mul_zero _
  have matrix : TargetPolynomial.evaluateMatrix extensionOps.toOps coefficients gamma =
      K.zero := by
    change ((canonicalMatrixCoordinates productionShape).map
      (TargetPolynomial.matrixTerm extensionOps.toOps coefficients gamma)).foldr
        extensionOps.add K.zero = K.zero
    apply foldr_map_zero
    intro coordinate
    change extensionOps.mul _ (data.claimedMatrixCoefficient coordinate) = K.zero
    rw [matrixZero coordinate]
    exact extensionLaws.mul_zero _
  change extensionOps.add
    (TargetPolynomial.evaluatePad extensionOps.toOps coefficients gamma)
    (TargetPolynomial.evaluateMatrix extensionOps.toOps coefficients gamma) = K.zero
  rw [pad, matrix]
  exact extensionLaws.zero_add K.zero

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.ZeroRunningPolynomial
