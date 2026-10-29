import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.Types
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SumCheckInitial

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.SumCheck

universe uField

def strictNormResidual
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (value : Field) : Field :=
  ops.mul (ops.mul (ops.add value ops.one) value)
    (ops.sub value ops.one)

structure Data (Field : Type uField) (shape : Shape) where
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial Field shape.matrixCount
  freshMatrixImages : Fin shape.freshCount -> Fin shape.matrixCount ->
    BooleanTable Field shape.cubeVariables
  sourceAssignments : Fin shape.sourceCount ->
    BooleanTable Field shape.cubeVariables
  priorPoint : CubePoint Field shape.cubeVariables
  padImages : PadCoordinate shape ->
    BooleanTable Field shape.cubeVariables
  matrixImages : MatrixCoordinate shape ->
    BooleanTable Field shape.cubeVariables
  claimedPadCoefficient : PadCoordinate shape -> Field
  claimedMatrixCoefficient : MatrixCoordinate shape -> Field

namespace Data

def toJointData
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape) :
    SignedJointIdentity.JointData Field shape where
  ccs := fun source => BooleanTable.tabulate fun vertex =>
    CCSResidualTable.evaluatePolynomial ops data.constraintPolynomial
      (fun matrix => (data.freshMatrixImages source matrix).valueAt vertex)
  norm := fun source => BooleanTable.tabulate fun vertex =>
    strictNormResidual ops ((data.sourceAssignments source).valueAt vertex)
  priorPoint := data.priorPoint
  padImage := data.padImages
  matrixImage := data.matrixImages
  claimedPadCoefficient := data.claimedPadCoefficient
  claimedMatrixCoefficient := data.claimedMatrixCoefficient

def toVerifierInput
    {Field : Type uField}
    {shape : Shape}
    (data : Data Field shape) : VerifierInput Field shape where
  constraintPolynomial := data.constraintPolynomial
  priorPoint := data.priorPoint
  claimedPadCoefficient := data.claimedPadCoefficient
  claimedMatrixCoefficient := data.claimedMatrixCoefficient

theorem toVerifierInput_eq
    {Field : Type uField}
    {shape : Shape}
    (left right : Data Field shape)
    (constraintPolynomial :
      left.constraintPolynomial = right.constraintPolynomial)
    (priorPoint : left.priorPoint = right.priorPoint)
    (claimedPadCoefficient :
      left.claimedPadCoefficient = right.claimedPadCoefficient)
    (claimedMatrixCoefficient :
      left.claimedMatrixCoefficient = right.claimedMatrixCoefficient) :
    left.toVerifierInput = right.toVerifierInput := by
  apply VerifierInput.ext
  · exact constraintPolynomial
  · exact priorPoint
  · exact claimedPadCoefficient
  · exact claimedMatrixCoefficient

end Data

def messageAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (point : CubePoint Field shape.cubeVariables) :
    OutputMessage Field shape where
  freshMatrixImage := fun source matrix =>
    (data.freshMatrixImages source matrix).evaluate ops point
  sourceAssignment := fun source =>
    (data.sourceAssignments source).evaluate ops point
  padImage := fun coordinate =>
    (data.padImages coordinate).evaluate ops point
  matrixImage := fun coordinate =>
    (data.matrixImages coordinate).evaluate ops point

def vertexMessage
    {Field : Type uField}
    {shape : Shape}
    (data : Data Field shape)
    (vertex : BooleanVertex shape.cubeVariables) :
    OutputMessage Field shape where
  freshMatrixImage := fun source matrix =>
    (data.freshMatrixImages source matrix).valueAt vertex
  sourceAssignment := fun source =>
    (data.sourceAssignments source).valueAt vertex
  padImage := fun coordinate =>
    (data.padImages coordinate).valueAt vertex
  matrixImage := fun coordinate =>
    (data.matrixImages coordinate).valueAt vertex

namespace VerifierInput

def targetCoefficients
    {Field : Type uField}
    {shape : Shape}
    (input : VerifierInput Field shape) :
    TargetPolynomial.TargetCoefficients Field shape where
  pad := input.claimedPadCoefficient
  matrix := input.claimedMatrixCoefficient

def initial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (gamma : Field) : Field :=
  TargetPolynomial.evaluate ops.toOps input.targetCoefficients gamma

end VerifierInput

theorem verifierInput_initial_eq_joint_initial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (gamma : Field) :
    data.toVerifierInput.initial ops gamma =
      SumCheckInitial.verifierInitial ops (data.toJointData ops) gamma := by
  rfl

def ccsAtMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (gamma : Field)
    (message : OutputMessage Field shape) : Field :=
  SignedJointIdentity.sumMap ops
    (canonicalFinIndices shape.freshCount) fun source =>
      SignedJointIdentity.gammaTerm ops gamma source.val <|
        CCSResidualTable.evaluatePolynomial ops input.constraintPolynomial
          (message.freshMatrixImage source)

def normAtMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (gamma : Field)
    (message : OutputMessage Field shape) : Field :=
  SignedJointIdentity.sumMap ops
    (canonicalFinIndices shape.sourceCount) fun source =>
      SignedJointIdentity.gammaTerm ops gamma source.val <|
        strictNormResidual ops (message.sourceAssignment source)

def padAtMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape) : Field :=
  ops.mul (SumCheckTruthPath.pointEquality ops point input.priorPoint) <|
    SignedJointIdentity.sumMap ops
      (canonicalPadCoordinates shape) fun coordinate =>
        SignedJointIdentity.gammaTerm ops gamma
          coordinate.localGammaExponent (message.padImage coordinate)

def matrixAtMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape) : Field :=
  ops.mul (SumCheckTruthPath.pointEquality ops point input.priorPoint) <|
    SignedJointIdentity.sumMap ops
      (canonicalMatrixCoordinates shape) fun coordinate =>
        SignedJointIdentity.gammaTerm ops gamma
          coordinate.localGammaExponent (message.matrixImage coordinate)

/-- SuperNeo v1.2 Section 7.3, Step 4. -/
def terminalFromMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape) : Field :=
  ops.add (padAtMessage ops input gamma point message) <|
    ops.add
      (SignedJointIdentity.gammaTerm ops gamma shape.matrixEvaluationOffset
        (matrixAtMessage ops input gamma point message))
      (SignedJointIdentity.gammaTerm ops gamma shape.constraintOffset <|
        ops.mul (SumCheckTruthPath.pointEquality ops point alpha)
          (ops.add
            (ccsAtMessage ops input gamma message)
            (SignedJointIdentity.gammaTerm ops gamma shape.freshCount
              (normAtMessage ops gamma message))))

def qAtPoint
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables) : Field :=
  terminalFromMessage ops data.toVerifierInput alpha gamma point
    (messageAt ops data point)

def polynomial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (coordinates : List Field) : Field :=
  if dimension : coordinates.length = shape.cubeVariables then
    qAtPoint ops data alpha gamma ⟨coordinates, dimension⟩
  else
    ops.zero

theorem messageAt_toCubePoint_eq_vertexMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape)
    (vertex : BooleanVertex shape.cubeVariables) :
    messageAt ops data (BooleanVertex.toCubePoint ops vertex) =
      vertexMessage data vertex := by
  apply OutputMessage.ext
  · funext source matrix
    exact SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
      ops laws (data.freshMatrixImages source matrix) vertex
  · funext source
    exact SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
      ops laws (data.sourceAssignments source) vertex
  · funext coordinate
    exact SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
      ops laws (data.padImages coordinate) vertex
  · funext coordinate
    exact SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
      ops laws (data.matrixImages coordinate) vertex

theorem terminalFromVertexMessage_eq_tableQ
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    terminalFromMessage ops data.toVerifierInput alpha gamma
        (BooleanVertex.toCubePoint ops vertex)
        (vertexMessage data vertex) =
      SignedJointIdentity.qAt ops (data.toJointData ops) alpha gamma vertex := by
  simp only [terminalFromMessage, ccsAtMessage, normAtMessage,
    padAtMessage, matrixAtMessage, vertexMessage, Data.toVerifierInput,
    Data.toJointData,
    SignedJointIdentity.qAt, SignedJointIdentity.ccsAt,
    SignedJointIdentity.normAt, SignedJointIdentity.padAt,
    SignedJointIdentity.matrixAt, SignedJointIdentity.constraintAt,
    BooleanTable.valueAt_tabulate,
    SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight ops laws]

theorem qAtPoint_toCubePoint_eq_tableQ
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    qAtPoint ops data alpha gamma
        (BooleanVertex.toCubePoint ops vertex) =
      SignedJointIdentity.qAt ops (data.toJointData ops) alpha gamma vertex := by
  unfold qAtPoint
  rw [messageAt_toCubePoint_eq_vertexMessage ops laws]
  exact terminalFromVertexMessage_eq_tableQ ops laws data alpha gamma vertex

theorem polynomial_fieldCoordinates_eq_tableQ
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    polynomial ops data alpha gamma
        (BooleanVertex.fieldCoordinates ops vertex) =
      SignedJointIdentity.qAt ops (data.toJointData ops) alpha gamma vertex := by
  unfold polynomial
  rw [dite_eq_left
    (BooleanVertex.fieldCoordinates_length ops vertex)]
  exact qAtPoint_toCubePoint_eq_tableQ ops laws data alpha gamma vertex

theorem sumCompletions_polynomial_eq_summedQ
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
        (polynomial ops data alpha gamma) [] shape.cubeVariables =
      SignedJointIdentity.summedQ ops (data.toJointData ops) alpha gamma := by
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  unfold SignedJointIdentity.summedQ SignedJointIdentity.sumMap
  simp only [List.nil_append]
  congr 1
  apply List.map_congr_left
  intro vertex _
  exact polynomial_fieldCoordinates_eq_tableQ
    ops laws data alpha gamma vertex

def canonicalExpected
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (challenges : List Field) : List (Field -> Field) :=
  SumCheck.Finite.HypercubeTruth.expectedPolynomials ops.toOps
    (polynomial ops data alpha gamma) challenges

def canonicalGhosts
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (challenges : List Field) : SumCheck.Finite.SemanticGhosts Field where
  trueInitial := SignedJointIdentity.summedQ
    ops (data.toJointData ops) alpha gamma
  expected := canonicalExpected ops data alpha gamma challenges

theorem canonicalGhosts_honest
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (initial : Field)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (certificate : SumCheck.Finite.Certificate Field)
    (sameLength : certificate.rounds.length = roundPoint.coordinates.length) :
    (canonicalGhosts ops data alpha gamma roundPoint.coordinates).Honest
      ops.toOps maxDegree challengeSetSize initial roundPoint.coordinates
      (qAtPoint ops data alpha gamma roundPoint) certificate := by
  have generic := SumCheck.Finite.HypercubeTruth.semanticGhosts_honest
    ops.toOps (polynomial ops data alpha gamma) maxDegree challengeSetSize
    initial roundPoint.coordinates certificate sameLength
  simpa only [canonicalGhosts, canonicalExpected,
    SumCheck.Finite.HypercubeTruth.semanticGhosts,
    sumCompletions_polynomial_eq_summedQ ops laws data alpha gamma,
    polynomial, roundPoint.dimension, dite_eq_left] using generic

def OutputMismatch
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape) : Prop :=
  terminalFromMessage ops data.toVerifierInput alpha gamma point message ≠
    qAtPoint ops data alpha gamma point

def check
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape)
    (certificate : SumCheck.Finite.Certificate Field) : Bool :=
  SumCheck.Finite.check ops.toOps input.sumcheckDegreeBound
    (input.initial ops gamma)
    roundPoint.coordinates
    (terminalFromMessage ops input alpha gamma roundPoint message)
    certificate

theorem check_eq_true_iff_accepted
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape)
    (certificate : SumCheck.Finite.Certificate Field) :
    check ops input alpha gamma roundPoint message certificate = true <->
      SumCheck.Finite.Accepted ops.toOps input.sumcheckDegreeBound
        (input.initial ops gamma)
        roundPoint.coordinates
        (terminalFromMessage ops input alpha gamma roundPoint message)
        certificate := by
  exact SumCheck.Finite.check_eq_true_iff_accepted ops.toOps
    input.sumcheckDegreeBound
    (input.initial ops gamma)
    roundPoint.coordinates
    (terminalFromMessage ops input alpha gamma roundPoint message)
    certificate

theorem check_implies_tableTruth_or_badEvent
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (challengeSetSize : Nat)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape)
    (certificate : SumCheck.Finite.Certificate Field)
    (checked : check ops data.toVerifierInput alpha gamma roundPoint message
      certificate = true) :
    (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops
          (data.toJointData ops))).AllHold \/
      SignedCoefficientObject.MixingRoot ops (data.toJointData ops)
        alpha gamma \/
      (exists round,
        SumCheck.BadChallenge
          (SumCheckInitial.symbolicInstance ops (data.toJointData ops)
            alpha gamma data.toVerifierInput.sumcheckDegreeBound
            challengeSetSize roundPoint.coordinates
            (terminalFromMessage ops data.toVerifierInput alpha gamma roundPoint message)
            certificate
            (canonicalExpected ops data alpha gamma roundPoint.coordinates))
          round) \/
      OutputMismatch ops data alpha gamma roundPoint message := by
  by_cases outputMatches :
      terminalFromMessage ops data.toVerifierInput alpha gamma roundPoint message =
        qAtPoint ops data alpha gamma roundPoint
  · have accepted :=
      (check_eq_true_iff_accepted ops data.toVerifierInput alpha gamma roundPoint
        message certificate).1 checked
    have sameLength :
        certificate.rounds.length = roundPoint.coordinates.length :=
      SumCheck.Finite.Chain.messages_length_eq_challenges_length ops.toOps
        data.toVerifierInput.sumcheckDegreeBound
        (data.toVerifierInput.initial ops gamma)
        (terminalFromMessage ops data.toVerifierInput alpha gamma roundPoint message)
        certificate.rounds roundPoint.coordinates accepted
    have honestAtPolynomial := canonicalGhosts_honest ops laws data alpha gamma
      data.toVerifierInput.sumcheckDegreeBound challengeSetSize
      (SumCheckInitial.verifierInitial ops (data.toJointData ops) gamma)
      roundPoint certificate sameLength
    have honestAtMessage :
        (SumCheckInitial.semanticGhosts ops (data.toJointData ops) alpha gamma
          (canonicalExpected ops data alpha gamma
            roundPoint.coordinates)).Honest
          ops.toOps data.toVerifierInput.sumcheckDegreeBound challengeSetSize
          (SumCheckInitial.verifierInitial ops (data.toJointData ops) gamma)
          roundPoint.coordinates
          (terminalFromMessage ops data.toVerifierInput alpha gamma roundPoint message)
          certificate := by
      simpa only [SumCheckInitial.semanticGhosts, canonicalGhosts,
        outputMatches] using! honestAtPolynomial
    have reduced :=
      SumCheckInitial.checked_implies_tableObligations_or_mixingRoot_or_badChallenge
        ops laws zeroLaws (data.toJointData ops) alpha gamma
        data.toVerifierInput.sumcheckDegreeBound
        challengeSetSize roundPoint.coordinates
        (terminalFromMessage ops data.toVerifierInput alpha gamma roundPoint message)
        certificate
        (canonicalExpected ops data alpha gamma roundPoint.coordinates)
        checked honestAtMessage
    rcases reduced with truth | mixing | round
    · exact Or.inl truth
    · exact Or.inr (Or.inl mixing)
    · exact Or.inr (Or.inr (Or.inl round))
  · exact Or.inr (Or.inr (Or.inr outputMatches))

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial
