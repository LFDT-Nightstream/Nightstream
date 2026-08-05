import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.Types
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial

/-!
The actual off-cube polynomial and terminal authority for paper joint `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: one-joint SumCheck truth path and post-SumCheck output evaluation check.
Constraint family: semantic polynomial ownership only; this file emits no rows.

Owns: explicit matrix-image, assignment, and carried-image multilinear tables;
the nonlinear paper expressions `F`, `NC`, `Eval`, and `Q` evaluated after
those multilinear images; projection to the minimal verifier-visible input;
the verifier terminal derived from that input and the typed output message;
Boolean-cube agreement with the independent signed residual object; and a
deterministic checker theorem that exposes output-message mismatch instead of
assuming it away.

Does not own: construction of these image tables from concrete CCS matrices
and witnesses, ring coefficient recomposition, output CE serialization,
Fiat--Shamir, semantic degree bounds, root probabilities, Rust,
R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: residual-table interpolation is not used as the off-cube
paper polynomial. `qAtPoint` first evaluates each underlying multilinear image
and then applies the nonlinear CCS/norm formulas, exactly in paper order.
Executable `check` receives only `VerifierInput`, never the hidden tables in
`Data`. If the raw output message does not match the semantic polynomial at the
derived point, the main theorem returns `OutputMismatch` as a named event.

| Protocol | Phase | Family | Mathematical owner |
|---|---|---|---|
| `Pi_CCS` | source images | matrix / assignment / carried tables | `Data` |
| `Pi_CCS` | verifier input | polynomial / prior point / public claims | `Data.toVerifierInput` |
| `Pi_CCS` | nonlinear point evaluation | `F`, `NC`, `Eval`, `Q` | `ccsAtMessage`, `normAtMessage`, `carriedAtMessage`, `terminalFromMessage` |
| `Pi_CCS` | prover output | values at `r'` only | `OutputMessage` |
| `Pi_CCS` | honest output | evaluate every source table at `r'` | `messageAt` |
| `Pi_CCS` | Boolean restriction | actual `Q` equals signed residual `Q` on the cube | `qAtPoint_toCubePoint_eq_tableQ` |
| `Pi_CCS` | SumCheck truth path | completion sums of actual `Q` | `canonicalGhosts_honest` |
| `Pi_CCS` | executable check | finite chain with message-derived terminal | `check` |
| assurance | deterministic soundness | table truth or mixing root or round collision or output mismatch | `check_implies_tableTruth_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck

universe uField

/-- The paper's strict-`b = 2` norm polynomial over the active carrier.
The raw source value is interpolated first; the cubic is applied afterwards. -/
def strictNormResidual
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (value : Field) : Field :=
  ops.mul (ops.mul (ops.add value ops.one) value)
    (ops.sub value ops.one)

/-- Explicit data from which the paper polynomial is constructed.

Unlike `SignedJointIdentity.JointData`, the CCS and norm fields are the
underlying image tables rather than already-composed residual truth tables.
That distinction preserves nonlinear off-cube evaluation. -/
structure Data (Field : Type uField) (shape : Shape) where
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial Field shape.matrixCount
  freshMatrixImages : Fin shape.freshCount -> Fin shape.matrixCount ->
    BooleanTable Field shape.cubeVariables
  sourceAssignments : Fin shape.sourceCount ->
    BooleanTable Field shape.cubeVariables
  priorPoint : CubePoint Field shape.cubeVariables
  carriedImages : CarriedCoordinate shape ->
    BooleanTable Field shape.cubeVariables
  claimedCoefficient : CarriedCoordinate shape -> Field

namespace Data

/-- Restrict the actual protocol formulas to Boolean vertices. This produces
the independent signed residual object used for alpha/gamma coefficient
reasoning; it does not define off-cube evaluation. -/
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
  carriedImage := data.carriedImages
  claimedCoefficient := data.claimedCoefficient

/-- Erase every hidden semantic table from the executable verifier surface.
All retained fields are verifier-owned structure or public claim data. -/
def toVerifierInput
    {Field : Type uField}
    {shape : Shape}
    (data : Data Field shape) : VerifierInput Field shape where
  constraintPolynomial := data.constraintPolynomial
  priorPoint := data.priorPoint
  claimedCoefficient := data.claimedCoefficient

/-- Rich semantic sources with the same three authoritative fields project to
the same executable input, regardless of every hidden assignment/image table. -/
theorem toVerifierInput_eq
    {Field : Type uField}
    {shape : Shape}
    (left right : Data Field shape)
    (constraintPolynomial :
      left.constraintPolynomial = right.constraintPolynomial)
    (priorPoint : left.priorPoint = right.priorPoint)
    (claimedCoefficient :
      left.claimedCoefficient = right.claimedCoefficient) :
    left.toVerifierInput = right.toVerifierInput := by
  apply VerifierInput.ext
  · exact constraintPolynomial
  · exact priorPoint
  · exact claimedCoefficient

end Data

/-- Canonical output values at one typed point, derived from the polynomial
source tables rather than supplied by a semantic callback. -/
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
  carriedImage := fun coordinate =>
    (data.carriedImages coordinate).evaluate ops point

/-- The exact output-message values at a Boolean vertex. -/
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
  carriedImage := fun coordinate =>
    (data.carriedImages coordinate).valueAt vertex

namespace VerifierInput

/-- The carried target polynomial is constructed solely from public claimed
coefficients. -/
def targetCoefficients
    {Field : Type uField}
    {shape : Shape}
    (input : VerifierInput Field shape) :
    TargetPolynomial.CarriedTargetCoefficients Field shape where
  coefficient := input.claimedCoefficient

/-- Verifier-owned initial claim under the corrected absolute exponent
convention. Hidden semantic tables cannot affect this value. -/
def initial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (gamma : Field) : Field :=
  TargetPolynomial.evaluateShifted ops.toOps input.targetCoefficients gamma

end VerifierInput

/-- Projecting rich semantic data does not change the paper-joint initial
claim used by the existing truth-path theorem. -/
theorem verifierInput_initial_eq_joint_initial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (gamma : Field) :
    data.toVerifierInput.initial ops gamma =
      SumCheckInitial.verifierInitial ops (data.toJointData ops) gamma := by
  rfl

/-- Paper `F(r', gamma)` derived from the fresh output evaluations. -/
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

/-- Paper `NC(r', gamma)` derived from all output assignment evaluations. -/
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

/-- Paper `Eval(r', gamma)` derived from the running output evaluations. -/
def carriedAtMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape) : Field :=
  ops.mul (SumCheckTruthPath.pointEquality ops point input.priorPoint) <|
    SignedJointIdentity.sumMap ops
      (canonicalCarriedCoordinates shape) fun coordinate =>
        SignedJointIdentity.gammaTerm ops gamma
          coordinate.localGammaExponent (message.carriedImage coordinate)

/-- Step 4's paper terminal formula, using only verifier context, derived
coins, the derived point, and the prover's output-evaluation message. -/
def terminalFromMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (input : VerifierInput Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape) : Field :=
  ops.add
    (ops.mul (SumCheckTruthPath.pointEquality ops point alpha)
      (ops.add
        (ccsAtMessage ops input gamma message)
        (SignedJointIdentity.gammaTerm ops gamma shape.freshCount
          (normAtMessage ops gamma message))))
    (SignedJointIdentity.gammaTerm ops gamma shape.carriedEvaluationOffset
      (carriedAtMessage ops input gamma point message))

/-- The actual paper polynomial at an arbitrary field point. Underlying image
tables are evaluated first and nonlinear formulas are applied second. -/
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

/-- Total coordinate-list form of the actual paper polynomial. Wrong arity
fails closed rather than selecting an arbitrary point representation. -/
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

/-- The output message derived at a Boolean point contains exactly the
underlying table leaves. -/
theorem messageAt_toCubePoint_eq_vertexMessage
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape)
    (vertex : BooleanVertex shape.cubeVariables) :
    messageAt ops data (SumCheckTruthPath.VertexEncoding.toCubePoint ops vertex) =
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
      ops laws (data.carriedImages coordinate) vertex

/-- At a Boolean vertex, the message-derived paper formula is exactly the
independently constructed signed residual `Q`. -/
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
        (SumCheckTruthPath.VertexEncoding.toCubePoint ops vertex)
        (vertexMessage data vertex) =
      SignedJointIdentity.qAt ops (data.toJointData ops) alpha gamma vertex := by
  simp only [terminalFromMessage, ccsAtMessage, normAtMessage,
    carriedAtMessage, vertexMessage, Data.toVerifierInput, Data.toJointData,
    SignedJointIdentity.qAt, SignedJointIdentity.ccsAt,
    SignedJointIdentity.normAt, SignedJointIdentity.carriedAt,
    BooleanTable.valueAt_tabulate,
    SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight ops laws]

/-- The actual off-cube polynomial and residual-table object agree exactly on
the Boolean cube, which is the only equality needed for the initial sum. -/
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
        (SumCheckTruthPath.VertexEncoding.toCubePoint ops vertex) =
      SignedJointIdentity.qAt ops (data.toJointData ops) alpha gamma vertex := by
  unfold qAtPoint
  rw [messageAt_toCubePoint_eq_vertexMessage ops laws]
  exact terminalFromVertexMessage_eq_tableQ ops laws data alpha gamma vertex

/-- Coordinate-list evaluation at a canonical Boolean vertex is the signed
residual `Q` leaf. -/
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
        (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex) =
      SignedJointIdentity.qAt ops (data.toJointData ops) alpha gamma vertex := by
  unfold polynomial
  rw [dif_pos
    (SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex)]
  exact qAtPoint_toCubePoint_eq_tableQ ops laws data alpha gamma vertex

/-- The actual paper polynomial has the same Boolean initial sum as the
independent signed residual object. No off-cube equality is asserted. -/
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

/-- The semantic expected-round list is derived from the actual protocol
polynomial, never from a prover callback or the residual-table MLE. -/
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

/-- Semantic ghosts for the actual protocol polynomial. -/
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

/-- Structurally derived truth path for the actual paper polynomial. The
semantic degree bound remains intentionally separate. -/
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
    polynomial, roundPoint.dimension, dif_pos] using generic

/-- The output message fails to represent the actual polynomial terminal at
the verifier-derived point. This is a named semantic event, not acceptance. -/
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

/-- Executable finite verifier with the initial target derived from public
carried claims and the terminal derived from the prover's output message. -/
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

/-- Exact executable/logical correspondence for the message-terminal checker. -/
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

/-- Deterministic soundness boundary for the actual protocol polynomial.

Acceptance yields residual-table truth, an alpha/gamma mixing root, a concrete
SumCheck collision under the actual polynomial's expected rounds, or an
explicit mismatch between the output message and the semantic terminal.
No probability or extraction claim is hidden in this theorem. -/
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
        outputMatches] using honestAtPolynomial
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

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial
