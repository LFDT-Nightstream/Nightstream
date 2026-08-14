import Nightstream.Implementation.Nebula.NIFS.Running.Codec
import Nightstream.Implementation.Transcript.Construction3Poseidon2

/-!
Contract: exact Poseidon2 transcript for the V2 product-commitment NIFS.

Assurance tier: executable transcript model.

Owns the fixed V2 profile frame, the 25-round Construction 3 event schedule,
the complete public running and fresh claim serialization, the complete
PiCCS output absorption, the four-component PiDEC message, and the selected
bounded PiRLC sampler.

Does not own generated transcript rows, Poseidon2 random-oracle security,
Ajtai binding, the concrete application matrix family, Rust, or the deployed
verifier.

Every authority-bearing commitment is serialized in component order
`full, operations, initial snapshot, final snapshot`. No digest or optional
sidecar replaces a component.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductPoseidon2

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Transcript
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.SumCheck.Finite

abbrev State := Construction3Poseidon2.State
abbrev StatementId := Construction3Poseidon2.StatementId
abbrev PaperShape :=
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape
abbrev SelectedCommitment := ProductCommitmentAlgebra.BundleValue
abbrev SelectedEvaluation := EvaluationFamily K ProductNifsCodec.shape

def constants := Construction3Poseidon2.constants
def initialState : State := Construction3Poseidon2.initialState
def word (value : Nat) : Nat := Construction3Poseidon2.word value

def construction3DomainFields : List Nat :=
  Construction3Poseidon2.construction3DomainFields
def statementIdLabelFields : List Nat :=
  Construction3Poseidon2.statementIdLabelFields
def proofLabelFields : List Nat := Construction3Poseidon2.proofLabelFields
def proverMessageLabelFields : List Nat :=
  Construction3Poseidon2.proverMessageLabelFields
def verifierChallengeLabelFields : List Nat :=
  Construction3Poseidon2.verifierChallengeLabelFields

def statementIdentifierTag : Nat :=
  Construction3Poseidon2.statementIdentifierTag
def piRlcCandidateTag : Nat := Construction3Poseidon2.piRlcCandidateTag
def piDecOutputTag : Nat := Construction3Poseidon2.piDecOutputTag

abbrev Event := Construction3Poseidon2.Event

def eventSchedule : List Event :=
  [.verifierCoins 1 1 42 48,
    .verifierCoins 2 2 43 2] ++
  ((List.range ProductNifsCodec.shape.cubeVariables).flatMap fun round =>
    [.proverMessage (3 + 2 * round) (1 + round) 45 20,
      .verifierCoins (4 + 2 * round) (3 + round) 46 2]) ++
  [.proverMessage 53 26 47 22680,
    .verifierCoins 54 28 piRlcCandidateTag 810,
    .proverMessage 55 27 piDecOutputTag 75600]

@[simp] theorem eventSchedule_length : eventSchedule.length = 55 := by
  decide

def eventScheduleFields : List Nat :=
  proverMessageLabelFields ++ verifierChallengeLabelFields ++
    [word 33, word eventSchedule.length] ++
      eventSchedule.flatMap Construction3Poseidon2.Event.fields

@[simp] theorem eventScheduleFields_length :
    eventScheduleFields.length = 313 := by
  decide

def statementIdentifierPrefixFields : List Nat :=
  construction3DomainFields ++ statementIdLabelFields ++ eventScheduleFields

@[simp] theorem statementIdentifierPrefixFields_length :
    statementIdentifierPrefixFields.length = 363 := by
  decide

def statementIdFields (statementId : StatementId) : List Nat :=
  [word statementIdentifierTag, word 4] ++
    (canonicalFinIndices 4).map fun lane => (statementId lane).val

@[simp] theorem statementIdFields_length (statementId : StatementId) :
    (statementIdFields statementId).length = 6 := by
  simp [statementIdFields, canonicalFinIndices_length]

def proofPrefixFields (statementId : StatementId) : List Nat :=
  construction3DomainFields ++ proofLabelFields ++ eventScheduleFields ++
    [word 36, word 2, word 1, word 1] ++ statementIdFields statementId

@[simp] theorem proofPrefixFields_length (statementId : StatementId) :
    (proofPrefixFields statementId).length = 366 := by
  simp [proofPrefixFields, construction3DomainFields, proofLabelFields,
    Construction3Poseidon2.proofLabelFields,
    Construction3Poseidon2.stringFields,
    Construction3Poseidon2.proofLabelBytes,
    Construction3Poseidon2.construction3DomainFields,
    Construction3Poseidon2.construction3DomainBytes,
    canonicalFinIndices_length]

def statementIdentifierFields (statementId : StatementId) : List Nat :=
  proofPrefixFields statementId

def initialStateForStatement (statementId : StatementId) : State :=
  Poseidon2Duplex.absorbList constants
    (statementIdentifierFields statementId) initialState

def fFields (value : F) : List Nat := [value.val]
def kFields (value : K) : List Nat := [value.c0.val, value.c1.val]

def finFields
    {count : Nat} {Value : Type}
    (encode : Value -> List Nat) (values : Fin count -> Value) : List Nat :=
  (canonicalFinIndices count).flatMap fun index => encode (values index)

def ringFFields (value : RingF) : List Nat := finFields fFields value

def shapeFields (value : PaperShape) : List Nat :=
  [word value.cubeVariables, word value.freshCount,
    word value.runningCount, word value.matrixCount,
    word value.coefficientCount]

def monomialFields
    {valueShape : PaperShape}
    (monomial : CCSResidualTable.Monomial K valueShape.matrixCount) : List Nat :=
  kFields monomial.coefficient ++
    (canonicalFinIndices valueShape.matrixCount).map fun index =>
      word (monomial.exponents index)

def polynomialFields
    {valueShape : PaperShape}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial K valueShape.matrixCount) : List Nat :=
  word polynomial.degreeBound :: word polynomial.terms.length ::
    polynomial.terms.flatMap monomialFields

def pointFields
    {variableCount : Nat} (point : CubePoint K variableCount) : List Nat :=
  point.coordinates.flatMap kFields

def componentCommitmentFields
    (commitment : ProductNifsCodec.ComponentCommitment) : List Nat :=
  finFields ringFFields commitment

def bundleFields (bundle : SelectedCommitment) : List Nat :=
  componentCommitmentFields (bundle .full) ++
    componentCommitmentFields (bundle .operations) ++
    componentCommitmentFields (bundle .initialSnapshot) ++
    componentCommitmentFields (bundle .finalSnapshot)

def publicInputFields
    {fullShape : Phi81Relation.Shape}
    (input : PublicInput fullShape) : List Nat :=
  finFields fFields input

def evaluationFields (evaluation : SelectedEvaluation) : List Nat :=
  finFields (fun coefficients => finFields kFields coefficients) evaluation

/-- ASCII `NSN2`, encoded as one Goldilocks field. -/
def publicInputTag : Nat := 1314082354
def protocolVersion : Nat := 2
def profileName : Nat := 2
def checkedStepFactor : Nat := 1
def commitmentEncodingTag : Nat := 1

def profileFields
    (fullShape : Phi81Relation.Shape) (degreeBound : Nat) : List Nat :=
  [word profileName, word protocolVersion, word checkedStepFactor,
    word commitmentEncodingTag] ++
  shapeFields ProductNifsCodec.shape ++
    [word fullShape.logicalWidth, word fullShape.carrierWidth,
      word (Phi81ColumnLayout.blockCount fullShape.carrierWidth),
      word ProductCommitmentAlgebra.Rank, word fullShape.publicWidth,
      word degreeBound]

@[simp] theorem profileFields_length
    (fullShape : Phi81Relation.Shape) (degreeBound : Nat) :
    (profileFields fullShape degreeBound).length = 15 := by
  rfl

def runningFields
    {fullShape : Phi81Relation.Shape}
    (running : ProductNifsCodec.Running fullShape) : List Nat :=
  pointFields running.point ++
    finFields bundleFields running.commitments ++
    finFields publicInputFields running.publicInputs ++
    finFields evaluationFields running.evaluations

def freshFields
    {fullShape : Phi81Relation.Shape}
    (fresh : ProductNifsCodec.Fresh fullShape) : List Nat :=
  finFields bundleFields fresh.commitments ++
    finFields publicInputFields fresh.publicInputs

def publicNifsFields
    {fullShape : Phi81Relation.Shape}
    (degreeBound : Nat)
    (running : ProductNifsCodec.Running fullShape)
    (fresh : ProductNifsCodec.Fresh fullShape) : List Nat :=
  [word publicInputTag, word protocolVersion] ++
    profileFields fullShape degreeBound ++ runningFields running ++
      freshFields fresh

def absorbPublicInput
    {fullShape : Phi81Relation.Shape}
    (degreeBound : Nat) (state : State)
    (running : ProductNifsCodec.Running fullShape)
    (fresh : ProductNifsCodec.Fresh fullShape) : State :=
  Poseidon2Duplex.absorbList constants
    (publicNifsFields degreeBound running fresh) state

def statementFields
    (statement : ProtocolVerifier.Statement K State ProductNifsCodec.shape) :
    List Nat :=
  [word 41] ++ shapeFields ProductNifsCodec.shape ++
    polynomialFields statement.input.constraintPolynomial ++
    [word ProductNifsCodec.shape.cubeVariables] ++
    pointFields statement.input.priorPoint ++
    [word ProductNifsCodec.shape.carriedEvaluationCount] ++
    (canonicalCarriedCoordinates ProductNifsCodec.shape).flatMap fun coordinate =>
      kFields (statement.input.claimedCoefficient coordinate)

def proverMessageFields
    (eventIndex messageIndex messageType : Nat)
    (payload : List Nat) : List Nat :=
  proverMessageLabelFields ++
    [word eventIndex, word messageIndex, word messageType,
      word payload.length] ++ payload

def roundFields
    (round : Fin ProductNifsCodec.shape.cubeVariables)
    (message : Message K) : List Nat :=
  proverMessageFields (3 + 2 * round.val) (1 + round.val) 45
    (message.coefficients.flatMap kFields)

def projectedOutputFields
    (message : ProtocolPolynomial.OutputMessage K ProductNifsCodec.shape) :
    List Nat :=
  proverMessageFields 53 26 47
    (finFields (fun matrices => finFields kFields matrices)
        message.freshMatrixImage ++
      finFields kFields message.sourceAssignment ++
      (canonicalCarriedCoordinates ProductNifsCodec.shape).flatMap fun coordinate =>
        kFields (message.carriedImage coordinate))

def outputFields
    (message : FullOutputCoordinates.FullOutput K ProductNifsCodec.shape) :
    List Nat :=
  proverMessageFields 53 26 47
    (finFields
      (fun matrices => finFields
        (fun coefficients => finFields kFields coefficients) matrices)
      message.coordinate)

def piDecOutputFields
    (proof : Proof K SelectedCommitment ProductNifsCodec.shape 9) : List Nat :=
  proverMessageFields 55 27 piDecOutputTag
    (finFields bundleFields proof.piDecCommitments ++
      finFields evaluationFields proof.piDecEvaluations)

def finalTranscriptState
    (state : State)
    (proof : Proof K SelectedCommitment ProductNifsCodec.shape 9) : State :=
  Poseidon2Duplex.absorbList constants (piDecOutputFields proof) state

def absorbFullOutput
    (state : State)
    (message : FullOutputCoordinates.FullOutput K ProductNifsCodec.shape) :
    State :=
  Poseidon2Duplex.absorbList constants (outputFields message) state

/-! ## Verifier-key-selected augmented-relation exponent -/

def statementFieldsFor
    (rowVariables : Nat)
    (statement : ProtocolVerifier.Statement K State
      (ProductNifsCodec.shapeFor rowVariables)) : List Nat :=
  let shape := ProductNifsCodec.shapeFor rowVariables
  [word 41] ++ shapeFields shape ++
    polynomialFields statement.input.constraintPolynomial ++
    [word shape.cubeVariables] ++
    pointFields statement.input.priorPoint ++
    [word shape.carriedEvaluationCount] ++
    (canonicalCarriedCoordinates shape).flatMap fun coordinate =>
      kFields (statement.input.claimedCoefficient coordinate)

def roundFieldsFor
    {rowVariables : Nat}
    (round : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (message : Message K) : List Nat :=
  proverMessageFields (3 + 2 * round.val) (1 + round.val) 45
    (message.coefficients.flatMap kFields)

def projectedOutputFieldsFor
    (rowVariables : Nat)
    (message : ProtocolPolynomial.OutputMessage K
      (ProductNifsCodec.shapeFor rowVariables)) : List Nat :=
  let shape := ProductNifsCodec.shapeFor rowVariables
  proverMessageFields (3 + 2 * rowVariables) (rowVariables + 1) 47
    (finFields (fun matrices => finFields kFields matrices)
        message.freshMatrixImage ++
      finFields kFields message.sourceAssignment ++
      (canonicalCarriedCoordinates shape).flatMap fun coordinate =>
        kFields (message.carriedImage coordinate))

def outputFieldsFor
    (rowVariables : Nat)
    (message : FullOutputCoordinates.FullOutput K
      (ProductNifsCodec.shapeFor rowVariables)) : List Nat :=
  proverMessageFields (3 + 2 * rowVariables) (rowVariables + 1) 47
    (finFields
      (fun matrices => finFields
        (fun coefficients => finFields kFields coefficients) matrices)
      message.coordinate)

def piDecOutputFieldsFor
    (rowVariables : Nat)
    (proof : Proof K SelectedCommitment
      (ProductNifsCodec.shapeFor rowVariables) 9) : List Nat :=
  proverMessageFields (5 + 2 * rowVariables) (rowVariables + 2)
    piDecOutputTag
    (finFields bundleFields proof.piDecCommitments ++
      finFields evaluationFields proof.piDecEvaluations)

def finalTranscriptStateFor
    (rowVariables : Nat) (state : State)
    (proof : Proof K SelectedCommitment
      (ProductNifsCodec.shapeFor rowVariables) 9) : State :=
  Poseidon2Duplex.absorbList constants
    (piDecOutputFieldsFor rowVariables proof) state

def absorbFullOutputFor
    (rowVariables : Nat) (state : State)
    (message : FullOutputCoordinates.FullOutput K
      (ProductNifsCodec.shapeFor rowVariables)) : State :=
  Poseidon2Duplex.absorbList constants
    (outputFieldsFor rowVariables message) state

private theorem finFields_length
    {count width : Nat} {Value : Type}
    (encode : Value -> List Nat) (values : Fin count -> Value)
    (fixed : forall value, (encode value).length = width) :
    (finFields encode values).length = count * width := by
  unfold finFields
  calc
    _ = (canonicalFinIndices count).length * width := by
      apply Poseidon2Program.length_flatMap_uniform
      intro index
      exact fixed (values index)
    _ = count * width := by rw [canonicalFinIndices_length]

private theorem ringFFields_length (value : RingF) :
    (ringFFields value).length = ringDegree := by
  simpa [ringFFields] using
    finFields_length fFields value (fun _ => rfl)

private theorem componentCommitmentFields_length
    (value : ProductNifsCodec.ComponentCommitment) :
    (componentCommitmentFields value).length = 972 := by
  rw [componentCommitmentFields,
    finFields_length ringFFields value ringFFields_length]
  rfl

@[simp] theorem bundleFields_length (value : SelectedCommitment) :
    (bundleFields value).length = 3888 := by
  simp [bundleFields, componentCommitmentFields_length]

private theorem publicInputFields_length
    {fullShape : Phi81Relation.Shape} (value : PublicInput fullShape) :
    (publicInputFields value).length = fullShape.publicWidth := by
  simpa [publicInputFields] using
    finFields_length fFields value (fun _ => rfl)

private theorem kFields_length (value : K) :
    (kFields value).length = 2 := rfl

private theorem coefficientFields_length
    (values : Fin ProductNifsCodec.shape.coefficientCount -> K) :
    (finFields kFields values).length = 108 := by
  rw [finFields_length (width := 2) kFields values kFields_length]
  rfl

private theorem evaluationFields_length (value : SelectedEvaluation) :
    (evaluationFields value).length = 1512 := by
  unfold evaluationFields
  rw [finFields_length (width := 108) _ value coefficientFields_length]
  rfl

private theorem pointFields_length
    {variableCount : Nat} (point : CubePoint K variableCount) :
    (pointFields point).length = variableCount * 2 := by
  unfold pointFields
  calc
    _ = point.coordinates.length * 2 := by
      apply Poseidon2Program.length_flatMap_uniform
      intro value
      exact kFields_length value
    _ = variableCount * 2 := by rw [point.dimension]

@[simp] theorem runningFields_length
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (running : ProductNifsCodec.Running fullShape) :
    (runningFields running).length = 83210 := by
  unfold runningFields
  rw [List.length_append, List.length_append, List.length_append,
    pointFields_length,
    finFields_length bundleFields running.commitments bundleFields_length,
    finFields_length publicInputFields running.publicInputs
      publicInputFields_length,
    finFields_length evaluationFields running.evaluations
      evaluationFields_length,
    contract.publicWidth]
  rfl

@[simp] theorem freshFields_length
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (fresh : ProductNifsCodec.Fresh fullShape) :
    (freshFields fresh).length = 4428 := by
  unfold freshFields
  rw [List.length_append,
    finFields_length bundleFields fresh.commitments bundleFields_length,
    finFields_length publicInputFields fresh.publicInputs
      publicInputFields_length,
    contract.publicWidth]
  rfl

@[simp] theorem publicNifsFields_length
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (degreeBound : Nat)
    (running : ProductNifsCodec.Running fullShape)
    (fresh : ProductNifsCodec.Fresh fullShape) :
    (publicNifsFields degreeBound running fresh).length = 87655 := by
  simp [publicNifsFields, profileFields_length, runningFields_length contract,
    freshFields_length contract]

private theorem matrixFields_length
    (values : Fin ProductNifsCodec.shape.matrixCount ->
      Fin ProductNifsCodec.shape.coefficientCount -> K) :
    (finFields (fun coefficients => finFields kFields coefficients)
      values).length = 1512 := by
  rw [finFields_length _ values coefficientFields_length]
  rfl

private theorem piDecPayloadFields_length
    (proof : Proof K SelectedCommitment ProductNifsCodec.shape 9) :
    (finFields bundleFields proof.piDecCommitments ++
      finFields evaluationFields proof.piDecEvaluations).length = 75600 := by
  rw [List.length_append,
    finFields_length bundleFields proof.piDecCommitments bundleFields_length,
    finFields_length evaluationFields proof.piDecEvaluations
      evaluationFields_length]
  rfl

private theorem outputCoordinateFields_length
    (values : Fin ProductNifsCodec.shape.sourceCount ->
      Fin ProductNifsCodec.shape.matrixCount ->
      Fin ProductNifsCodec.shape.coefficientCount -> K) :
    (finFields
      (fun matrices => finFields
        (fun coefficients => finFields kFields coefficients) matrices)
      values).length = 22680 := by
  rw [finFields_length _ values matrixFields_length]
  rfl

@[simp] theorem outputFields_length
    (message : FullOutputCoordinates.FullOutput K ProductNifsCodec.shape) :
    (outputFields message).length = 22700 := by
  unfold outputFields proverMessageFields
  rw [List.length_append, outputCoordinateFields_length]
  decide

@[simp] theorem piDecOutputFields_length
    (proof : Proof K SelectedCommitment ProductNifsCodec.shape 9) :
    (piDecOutputFields proof).length = 75620 := by
  unfold piDecOutputFields proverMessageFields
  rw [List.length_append, piDecPayloadFields_length]
  decide

def challengeValue (state : State) : K :=
  Construction3Poseidon2.challengeValue state
def squeezeK (state : State) : K × State :=
  Construction3Poseidon2.squeezeK state

def verifierChallengeFields
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) : List Nat :=
  construction3DomainFields ++ verifierChallengeLabelFields ++
    [word eventIndex, word challengeIndex, word challengeType,
      word coordinates.length] ++ coordinates.map word

def squeezeVerifierChallenge
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (state : State) : K × State :=
  squeezeK (Poseidon2Duplex.absorbList constants
    (verifierChallengeFields eventIndex challengeIndex challengeType coordinates)
    state)

def transcript :
    FiatShamir.Oracle
      (ProtocolVerifier.Statement K State ProductNifsCodec.shape)
      K State ProductNifsCodec.shape where
  initialState statement :=
    Poseidon2Duplex.absorbList constants (statementFields statement)
      statement.priorState
  absorbRound state round message :=
    Poseidon2Duplex.absorbList constants (roundFields round message) state
  squeeze state label :=
    match label with
    | .alpha coordinate =>
        squeezeVerifierChallenge 1 1 42 [coordinate.val] state
    | .gamma => squeezeVerifierChallenge 2 2 43 [] state
    | .sumcheck round =>
        squeezeVerifierChallenge (4 + 2 * round.val) (3 + round.val) 46 [] state

def oracle : ProtocolVerifier.Oracle K State ProductNifsCodec.shape where
  transcript := transcript
  absorbOutput state message :=
    Poseidon2Duplex.absorbList constants (projectedOutputFields message) state

def transcriptFor (rowVariables : Nat) :
    FiatShamir.Oracle
      (ProtocolVerifier.Statement K State
        (ProductNifsCodec.shapeFor rowVariables))
      K State (ProductNifsCodec.shapeFor rowVariables) where
  initialState statement :=
    Poseidon2Duplex.absorbList constants
      (statementFieldsFor rowVariables statement) statement.priorState
  absorbRound state round message :=
    Poseidon2Duplex.absorbList constants (roundFieldsFor round message) state
  squeeze state label :=
    match label with
    | .alpha coordinate =>
        squeezeVerifierChallenge 1 1 42 [coordinate.val] state
    | .gamma => squeezeVerifierChallenge 2 2 43 [] state
    | .sumcheck round =>
        squeezeVerifierChallenge (4 + 2 * round.val) (3 + round.val) 46 [] state

def oracleFor (rowVariables : Nat) :
    ProtocolVerifier.Oracle K State
      (ProductNifsCodec.shapeFor rowVariables) where
  transcript := transcriptFor rowVariables
  absorbOutput state message :=
    Poseidon2Duplex.absorbList constants
      (projectedOutputFieldsFor rowVariables message) state

abbrev Coefficient := Construction3Poseidon2.Coefficient
abbrev Scalar := Construction3Poseidon2.Scalar

def samplerAttemptCount : Nat := Construction3Poseidon2.samplerAttemptCount
def firstAttempt : Fin samplerAttemptCount :=
  Construction3Poseidon2.firstAttempt
def secondAttempt : Fin samplerAttemptCount :=
  Construction3Poseidon2.secondAttempt
def thirdAttempt : Fin samplerAttemptCount :=
  Construction3Poseidon2.thirdAttempt
def samplerCoefficientCount : Nat :=
  Construction3Poseidon2.samplerCoefficientCount
def candidateFields := Construction3Poseidon2.candidateFields
def candidateValue := Construction3Poseidon2.candidateValue
def candidateAccepted := Construction3Poseidon2.candidateAccepted
def candidateDigit := Construction3Poseidon2.candidateDigit
def sampleCoefficient := Construction3Poseidon2.sampleCoefficient

@[simp] theorem sampleCoefficient_of_first
    (state : State)
    (source : Fin Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (accepted : candidateAccepted
      (candidateValue state source coefficient firstAttempt) = true) :
    sampleCoefficient state source coefficient =
      some (candidateDigit
        (candidateValue state source coefficient firstAttempt)) := by
  change Construction3Poseidon2.candidateAccepted
    (Construction3Poseidon2.candidateValue state source coefficient
      Construction3Poseidon2.firstAttempt) = true at accepted
  change Construction3Poseidon2.sampleCoefficient state source coefficient =
    some (Construction3Poseidon2.candidateDigit
      (Construction3Poseidon2.candidateValue state source coefficient
        Construction3Poseidon2.firstAttempt))
  simp only [Construction3Poseidon2.sampleCoefficient, accepted, if_true]

@[simp] theorem sampleCoefficient_of_second
    (state : State)
    (source : Fin Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (firstRejected : candidateAccepted
      (candidateValue state source coefficient firstAttempt) = false)
    (secondAccepted : candidateAccepted
      (candidateValue state source coefficient secondAttempt) = true) :
    sampleCoefficient state source coefficient =
      some (candidateDigit
        (candidateValue state source coefficient secondAttempt)) := by
  change Construction3Poseidon2.candidateAccepted
    (Construction3Poseidon2.candidateValue state source coefficient
      Construction3Poseidon2.firstAttempt) = false at firstRejected
  change Construction3Poseidon2.candidateAccepted
    (Construction3Poseidon2.candidateValue state source coefficient
      Construction3Poseidon2.secondAttempt) = true at secondAccepted
  change Construction3Poseidon2.sampleCoefficient state source coefficient =
    some (Construction3Poseidon2.candidateDigit
      (Construction3Poseidon2.candidateValue state source coefficient
        Construction3Poseidon2.secondAttempt))
  simp only [Construction3Poseidon2.sampleCoefficient, firstRejected,
    secondAccepted, Bool.false_eq_true, if_false, if_true]

@[simp] theorem sampleCoefficient_of_third
    (state : State)
    (source : Fin Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (firstRejected : candidateAccepted
      (candidateValue state source coefficient firstAttempt) = false)
    (secondRejected : candidateAccepted
      (candidateValue state source coefficient secondAttempt) = false)
    (thirdAccepted : candidateAccepted
      (candidateValue state source coefficient thirdAttempt) = true) :
    sampleCoefficient state source coefficient =
      some (candidateDigit
        (candidateValue state source coefficient thirdAttempt)) := by
  change Construction3Poseidon2.candidateAccepted
    (Construction3Poseidon2.candidateValue state source coefficient
      Construction3Poseidon2.firstAttempt) = false at firstRejected
  change Construction3Poseidon2.candidateAccepted
    (Construction3Poseidon2.candidateValue state source coefficient
      Construction3Poseidon2.secondAttempt) = false at secondRejected
  change Construction3Poseidon2.candidateAccepted
    (Construction3Poseidon2.candidateValue state source coefficient
      Construction3Poseidon2.thirdAttempt) = true at thirdAccepted
  change Construction3Poseidon2.sampleCoefficient state source coefficient =
    some (Construction3Poseidon2.candidateDigit
      (Construction3Poseidon2.candidateValue state source coefficient
        Construction3Poseidon2.thirdAttempt))
  simp only [Construction3Poseidon2.sampleCoefficient, firstRejected,
    secondRejected, thirdAccepted, Bool.false_eq_true, if_false, if_true]

def SamplerShortfall := Construction3Poseidon2.SamplerShortfall
def SamplerAvailable := Construction3Poseidon2.SamplerAvailable
noncomputable def samplerSucceeded :=
  Construction3Poseidon2.samplerSucceeded
def scalarResponse := Construction3Poseidon2.scalarResponse
def piRlcResponse := Construction3Poseidon2.piRlcResponse

@[simp] theorem samplerSucceeded_eq_true_iff (state : State) :
    samplerSucceeded state = true ↔ SamplerAvailable state := by
  simpa only [samplerSucceeded, SamplerAvailable] using
    Construction3Poseidon2.samplerSucceeded_eq_true_iff state

def ResponseRefinesAt := Construction3Poseidon2.ResponseRefinesAt

theorem scalarResponse_refines_of_available
    {state : State} (available : SamplerAvailable state) :
    ResponseRefinesAt scalarResponse state := by
  exact Construction3Poseidon2.piRlcResponse_refines_of_available available

theorem samplerAvailable_of_all
    {state : State}
    (allSucceeded : forall source coefficient,
      sampleCoefficient state source coefficient ≠ none) :
    SamplerAvailable state := by
  intro shortfall
  rcases shortfall with ⟨source, coefficient, failed⟩
  exact allSucceeded source coefficient failed

theorem scalarResponse_eq_of_sampled
    (state : State)
    (source : Fin Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total)
    (coefficient : Fin
      Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount)
    (selected : Coefficient)
    (sampled : sampleCoefficient state source
      (Fin.cast (by rfl) coefficient) = some selected) :
    scalarResponse state source coefficient = selected := by
  change Construction3Poseidon2.sampleCoefficient state source
    (Fin.cast (by rfl) coefficient) = some selected at sampled
  unfold scalarResponse Construction3Poseidon2.scalarResponse
  rw [sampled]
  rfl

theorem piRlcResponse_valid (state : State)
    (index : Fin Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.challengeValid
      (piRlcResponse state index) :=
  Construction3Poseidon2.piRlcResponse_valid state index

end Nightstream.Implementation.Nebula.ProductPoseidon2
