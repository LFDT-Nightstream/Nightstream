import NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
import NightstreamFPrime.Spec.Folding.PiRLC.PaperCompleteness
import NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier
import NightstreamFPrime.Spec.SumCheck.FixedPhase

/-!
Typed public data and deterministic dataflow for the paper SuperNeo NIFS.

Source: SuperNeo Sections 7.3--7.5 and HyperNova Definition 12.

Owns: the separated fresh/running public carriers; one prover message with a
joint-`Pi_CCS` certificate, coefficient-complete output, and `Pi_DEC` child
messages; the exact paper statement; verifier-owned absorption of the complete
public NIFS input before any challenge; verifier-derived `Pi_CCS` and
`Pi_RLC` challenges; the computed combined parent; and the computed running
output.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uState

structure Fresh
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (shape : Shape) where
  commitments : Fin shape.freshCount -> Commitment
  publicInputs : Fin shape.freshCount -> PublicInput

structure Running
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (shape : Shape) where
  point : CubePoint Extension shape.cubeVariables
  commitments : Fin shape.runningCount -> Commitment
  publicInputs : Fin shape.runningCount -> PublicInput
  evaluations : Fin shape.runningCount -> EvaluationFamily Extension shape

structure Proof
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (shape : Shape)
    (degreeBound : Nat) where
  piCcsRounds : Fin shape.cubeVariables ->
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial Extension degreeBound
  piCcsOutput : FullOutputCoordinates.FullOutput Extension shape
  piDecCommitments : Fin shape.runningCount -> Commitment
  piDecEvaluations : Fin shape.runningCount -> EvaluationFamily Extension shape

structure Key
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (Scalar : Type uScalar)
    (State : Type uState)
    (shape : Shape)
    (columns blockCount degreeBound : Nat) where
  baseOps : InterpolationOps F
  baseLaws : InterpolationEvaluationLaws baseOps
  baseZero : NormResidualTable.BaseZeroAgreement baseOps
  noZeroDivisors : NormRange.BaseFieldNoZeroDivisors
  extensionOps : InterpolationOps Extension
  extensionLaws : InterpolationEvaluationLaws extensionOps
  extensionZeroLaws : InterpolationZeroLaws extensionOps
  lift : F -> Extension
  liftLaws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift
  openingMaps : OpeningMaps Commitment PublicInput columns
  params : GlobalParams
  freshBound : params.b = 2
  arity : BatchArity params
  freshCount_eq : arity.freshCount = shape.freshCount
  runningCount_eq : arity.mode.count params = shape.runningCount
  outputCount_eq : params.k = shape.runningCount
  kPositive : 0 < params.k
  cubeLayout : UnifiedSources.ColumnLayout shape.cubeVariables columns
  matrixSource : MatrixSource F shape columns blockCount
  degreeBoundExact :
    Nat.max
      (ConstraintPolynomialLift.liftConstraintPolynomial lift
        matrixSource.constraintPolynomial).canonicalEqualityGatedDegreeBound 4 =
      degreeBound
  constantLaw : ConstantTermLaw baseOps matrixSource.kernel
  challengeSetSize : Nat
  piRlcSemantics : RelationSemantics
    (RelationSource shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape)
    Commitment
  openingAgreement : forall
      (normBound : Nat)
      (commitment : Commitment)
      (publicInput : PublicInput)
      (assignment : Assignment F columns),
    (Opening.Holds
        (paperRelationSemantics (shape := shape) (blockCount := blockCount)
          baseOps extensionOps lift openingMaps)
        normBound commitment publicInput assignment <->
      Opening.Holds piRlcSemantics normBound commitment publicInput assignment)
  ambientAgreement : forall
      (statement : CE.Instance
        (RelationSource shape columns blockCount)
        PublicInput
        (CubePoint Extension shape.cubeVariables)
        (EvaluationFamily Extension shape)
        Commitment)
      (assignment : Assignment F columns),
    statement.constraintSystem =
        ({ cubeLayout := cubeLayout
           matrixSource := matrixSource } :
          RelationSource shape columns blockCount) ->
      (PiRLC.PaperCorrections.CorrectedAmbientHolds
          (paperRelationSemantics baseOps extensionOps lift openingMaps)
          params statement assignment <->
        PiRLC.PaperCorrections.CorrectedAmbientHolds
          piRlcSemantics params statement assignment)
  evaluationAgreement : forall
      (assignment : Assignment F columns)
      (point : CubePoint Extension shape.cubeVariables),
    piRlcSemantics.evaluationPointValid
        ({ cubeLayout := cubeLayout
           matrixSource := matrixSource } :
          RelationSource shape columns blockCount) point /\
      piRlcSemantics.evaluations
          ({ cubeLayout := cubeLayout
             matrixSource := matrixSource } :
            RelationSource shape columns blockCount)
          assignment point =
        (paperRelationSemantics baseOps extensionOps lift openingMaps).evaluations
          ({ cubeLayout := cubeLayout
             matrixSource := matrixSource } :
            RelationSource shape columns blockCount)
          assignment point
  piRlcEvaluationsSize : forall system assignment point,
    (piRlcSemantics.evaluations system assignment point).size = 1
  piRlcAlgebra : PiRLC.Algebra
    (RelationSource shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape)
    Commitment Scalar
    piRlcSemantics
    params
  piDecAlgebra : PiDEC.Algebra
    (RelationSource shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape)
    Commitment
    piRlcSemantics
    params
  piDecPublicInputSplit : PiDEC.PaperVerifier.PublicInputSplit piDecAlgebra
  piDecEvaluationArity : PiDEC.PaperVerifier.EvaluationArity
    piRlcSemantics
  piDecEvaluationCount :
    piDecEvaluationArity.count
      ({ cubeLayout := cubeLayout
         matrixSource := matrixSource } :
        RelationSource shape columns blockCount) = 1
  piDecDecision : forall attempt,
    Decidable (PiDEC.PaperVerifier.Accepted piDecAlgebra
      piDecPublicInputSplit piDecEvaluationArity attempt)
  oracle : NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Oracle
    Extension State shape
  initialTranscriptState : State
  absorbPublicInput : State ->
    Running Extension Commitment PublicInput shape ->
    Fresh Commitment PublicInput shape ->
    State
  absorbPiCcsOutput : State ->
    FullOutputCoordinates.FullOutput Extension shape -> State
  piRlcResponse : State -> Option (Fin arity.total -> Scalar)
  piRlcResponseValid : forall state response,
    piRlcResponse state = some response -> forall index,
      piRlcAlgebra.challengeValid (response index)

namespace Key

def relationSource
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    RelationSource shape columns blockCount where
  cubeLayout := key.cubeLayout
  matrixSource := key.matrixSource

theorem total_eq_sourceCount
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    key.arity.total = shape.sourceCount := by
  simp only [BatchArity.total, Shape.sourceCount]
  rw [key.freshCount_eq, key.runningCount_eq]

theorem runningCount_eq_outputCount
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    shape.runningCount = key.params.k :=
  key.outputCount_eq.symm

def semantics
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :=
  key.piRlcSemantics

def statement
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    StrongReduction.Statement Extension Commitment PublicInput shape
      columns blockCount key.baseOps where
  cubeLayout := key.cubeLayout
  matrixSource := key.matrixSource
  commitments := Fin.addCases fresh.commitments running.commitments
  publicInputs := Fin.addCases fresh.publicInputs running.publicInputs
  priorPoint := running.point
  claimedPadCoefficient := fun coordinate =>
    (running.evaluations coordinate.running).pad coordinate.coefficient
  claimedMatrixCoefficient := fun coordinate =>
    (running.evaluations coordinate.running).matrix coordinate.matrix
      coordinate.coefficient

theorem statement_sumcheckDegreeBound_eq
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    ((key.statement running fresh).verifierInput key.lift).sumcheckDegreeBound =
      degreeBound := by
  exact key.degreeBoundExact

theorem statement_sumcheckDegreeBound_le
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    ((key.statement running fresh).verifierInput key.lift).sumcheckDegreeBound <=
      degreeBound := by
  exact Nat.le_of_eq (key.statement_sumcheckDegreeBound_eq running fresh)

def publicInputState
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) : State :=
  key.absorbPublicInput key.initialTranscriptState running fresh

def piCcsCertificate
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Certificate
      Extension shape where
  rounds := fun round => (proof.piCcsRounds round).toMessage
  output := (key.statement running fresh).projectOutput proof.piCcsOutput

@[simp] theorem piCcsCertificate_round
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (round : Fin shape.cubeVariables) :
    (key.piCcsCertificate running fresh proof).rounds round =
      (proof.piCcsRounds round).toMessage := by
  rfl

@[simp] theorem piCcsCertificate_toTranscript_round
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (round : Fin shape.cubeVariables) :
    ((key.piCcsCertificate running fresh proof).toTranscript).rounds round =
      (proof.piCcsRounds round).toMessage := by
  rfl

def piCcsFixedCertificate
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (_key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (_running : Running Extension Commitment PublicInput shape)
    (_fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    SumCheck.Finite.FixedPhase.Certificate Extension degreeBound where
  rounds := List.ofFn proof.piCcsRounds

structure PiCcsExecution
    (Extension : Type uExtension)
    (State : Type uState)
    (shape : Shape) where
  coins : FiatShamir.DerivedCoins Extension State shape
  outgoingState : State

def piCcsExecution
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    PiCcsExecution Extension State shape :=
  let replay :=
    NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.derive key.oracle
      (key.publicInputState running fresh)
      (StrongReduction.Statement.verifierInput key.lift
        (key.statement running fresh))
      (key.piCcsCertificate running fresh proof)
  { coins := replay.coins
    outgoingState :=
      key.absorbPiCcsOutput replay.coins.finalState proof.piCcsOutput }

theorem piCcsExecution_coins_eq_derive
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    (key.piCcsExecution running fresh proof).coins =
      FiatShamir.derive key.oracle.transcript
        ({ priorState := key.publicInputState running fresh
           input := (key.statement running fresh).verifierInput key.lift } :
          NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Statement
            Extension State shape)
        ({ rounds := fun round => (proof.piCcsRounds round).toMessage } :
          FiatShamir.Certificate Extension shape) := by
  rfl

theorem piCcsExecution_outgoingState_eq_absorbPiCcsOutput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    (key.piCcsExecution running fresh proof).outgoingState =
      key.absorbPiCcsOutput
        (key.piCcsExecution running fresh proof).coins.finalState
        proof.piCcsOutput := by
  rfl

def piCcsProbe
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    StrongReduction.Probe Extension shape where
  coins := {
    alpha := (key.piCcsExecution running fresh proof).coins.alpha
    gamma := (key.piCcsExecution running fresh proof).coins.gamma
    roundPoint := (key.piCcsExecution running fresh proof).coins.roundPoint
  }
  response := {
    rounds := (key.piCcsCertificate running fresh proof).toFinite
    fullOutput := proof.piCcsOutput
  }

def piCcsOutputs
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    Fin key.arity.total -> CE.Instance
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment :=
  fun source =>
    (key.statement running fresh).publicOutput
      (key.piCcsProbe running fresh proof)
      (Fin.cast key.total_eq_sourceCount source)

def piRlcChallenges
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    Option (Fin key.arity.total -> Scalar) :=
  key.piRlcResponse
    (key.piCcsExecution running fresh proof).outgoingState

def parentForChallenges
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (challenges : Fin key.arity.total -> Scalar) :=
  PiRLC.combinedOutput key.piRlcAlgebra key.relationSource
    (key.piCcsExecution running fresh proof).coins.roundPoint
    (key.piCcsOutputs running fresh proof)
    challenges

def parent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :=
  (key.piRlcChallenges running fresh proof).map
    (key.parentForChallenges running fresh proof)

@[simp] theorem parentForChallenges_point
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (challenges : Fin key.arity.total -> Scalar) :
    (key.parentForChallenges running fresh proof challenges).point =
      (key.piCcsExecution running fresh proof).coins.roundPoint := by
  rfl

def piDecAttemptForParent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (proof : Proof Extension Commitment shape degreeBound)
    (parent : CE.Instance
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment) :
    PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params where
  parent := parent
  messages := fun child => {
    commitment := proof.piDecCommitments (Fin.cast key.outputCount_eq child)
    evaluations := #[proof.piDecEvaluations (Fin.cast key.outputCount_eq child)]
  }

def piDecAttempt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :=
  (key.parent running fresh proof).map
    (key.piDecAttemptForParent proof)

def outputForAttempt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (proof : Proof Extension Commitment shape degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params)
    (publicInputs : Fin key.params.k → PublicInput) :
    Running Extension Commitment PublicInput shape :=
  {
    point := attempt.parent.point
    commitments := fun runningIndex =>
      (attempt.messages
        (Fin.cast key.runningCount_eq_outputCount runningIndex)).commitment
    publicInputs := fun runningIndex =>
      publicInputs (Fin.cast key.runningCount_eq_outputCount runningIndex)
    evaluations := proof.piDecEvaluations
  }

def output
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    Option (Running Extension Commitment PublicInput shape) :=
  (key.piDecAttempt running fresh proof).bind fun attempt =>
    (key.piDecPublicInputSplit.checked attempt.parent.publicInput).map
      (key.outputForAttempt proof attempt)

@[simp] theorem outputForAttempt_point
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (proof : Proof Extension Commitment shape degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params)
    (publicInputs : Fin key.params.k → PublicInput) :
    (key.outputForAttempt proof attempt publicInputs).point =
      attempt.parent.point := by
  rfl

@[simp] theorem outputForAttempt_commitment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (proof : Proof Extension Commitment shape degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params)
    (publicInputs : Fin key.params.k → PublicInput)
    (runningIndex : Fin shape.runningCount) :
    (key.outputForAttempt proof attempt publicInputs).commitments runningIndex =
      (attempt.messages
        (Fin.cast key.runningCount_eq_outputCount runningIndex)).commitment := by
  rfl

@[simp] theorem outputForAttempt_publicInput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (proof : Proof Extension Commitment shape degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params)
    (publicInputs : Fin key.params.k → PublicInput)
    (runningIndex : Fin shape.runningCount) :
    (key.outputForAttempt proof attempt publicInputs).publicInputs runningIndex =
      publicInputs (Fin.cast key.runningCount_eq_outputCount runningIndex) := by
  rfl

@[simp] theorem outputForAttempt_evaluation
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (proof : Proof Extension Commitment shape degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params)
    (publicInputs : Fin key.params.k → PublicInput)
    (runningIndex : Fin shape.runningCount) :
    (key.outputForAttempt proof attempt publicInputs).evaluations runningIndex =
      proof.piDecEvaluations runningIndex := by
  rfl

theorem output_eq_none_of_parentUnbounded
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params)
    (attemptEq : key.piDecAttempt running fresh proof = some attempt)
    (unbounded :
      ¬ key.piDecPublicInputSplit.parentBounded attempt.parent.publicInput) :
    key.output running fresh proof = none := by
  simp [output, attemptEq,
    key.piDecPublicInputSplit.checked_eq_none attempt.parent.publicInput
      unbounded]

theorem output_eq_some_of_parentBounded
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params)
    (attemptEq : key.piDecAttempt running fresh proof = some attempt)
    (bounded :
      key.piDecPublicInputSplit.parentBounded attempt.parent.publicInput) :
    key.output running fresh proof = some
      (key.outputForAttempt proof attempt
        (key.piDecPublicInputSplit.split attempt.parent.publicInput)) := by
  simp [output, attemptEq,
    key.piDecPublicInputSplit.checked_eq_some attempt.parent.publicInput bounded]

end Key

end NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
