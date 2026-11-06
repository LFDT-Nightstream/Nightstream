import NightstreamFPrime.Spec.Folding.Light.PiCCSCompleteness
import NightstreamFPrime.Spec.Folding.Light.Rlc
import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Types

/-! The light fold's verifier under Fiat-Shamir: the light Π_CCS on one fresh and one running
source, then the light Π_RLC. Cyclo (ePrint 2026/359) contribution 2 on SuperNeo v1.2
Sections 7.3 and 7.4. There is no Π_DEC, and the output is one accumulator claim. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Fresh Running)

universe uCommitment uPublicInput uScalar uState

structure Proof (shape : Shape) (width : Nat) where
  piCcsRounds : Fin shape.cubeVariables -> SumCheck.Finite.FixedPolynomial K width
  piCcsOutput : FullOutputCoordinates.FullOutput K shape

structure Key
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (Scalar : Type uScalar)
    (State : Type uState)
    (shape : Shape)
    (columns blockCount width : Nat) where
  baseOps : InterpolationOps F
  baseLaws : InterpolationEvaluationLaws baseOps
  baseZero : NormResidualTable.BaseZeroAgreement baseOps
  noZeroDivisors : NormRange.BaseFieldNoZeroDivisors
  extensionOps : InterpolationOps K
  extensionLaws : InterpolationEvaluationLaws extensionOps
  extensionZeroLaws : InterpolationZeroLaws extensionOps
  lift : F -> K
  liftLaws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift
  openingMaps : OpeningMaps Commitment PublicInput columns
  params : GlobalParams
  freshCount_eq : shape.freshCount = 1
  runningCount_eq : shape.runningCount = 1
  cubeLayout : UnifiedSources.ColumnLayout shape.cubeVariables columns
  matrixSource : MatrixSource F shape columns blockCount
  degreeBoundExact :
    Nat.max
      (ConstraintPolynomialLift.liftConstraintPolynomial lift
        matrixSource.constraintPolynomial).canonicalEqualityGatedDegreeBound 4 = width
  constantLaw : ConstantTermLaw baseOps matrixSource.kernel
  piRlcSemantics : RelationSemantics
    (RelationSource shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint K shape.cubeVariables)
    (EvaluationFamily K shape)
    Commitment
  openingAgreement : ∀ (normBound : Nat) (commitment : Commitment) (publicInput : PublicInput)
      (assignment : Assignment F columns),
    Opening.Holds
        (paperRelationSemantics (shape := shape) (blockCount := blockCount)
          baseOps extensionOps lift openingMaps)
        normBound commitment publicInput assignment ↔
      Opening.Holds piRlcSemantics normBound commitment publicInput assignment
  evaluationAgreement : ∀ (assignment : Assignment F columns)
      (point : CubePoint K shape.cubeVariables),
    piRlcSemantics.evaluationPointValid
        ({ cubeLayout := cubeLayout, matrixSource := matrixSource } :
          RelationSource shape columns blockCount) point ∧
      piRlcSemantics.evaluations
          ({ cubeLayout := cubeLayout, matrixSource := matrixSource } :
            RelationSource shape columns blockCount) assignment point =
        (paperRelationSemantics baseOps extensionOps lift openingMaps).evaluations
          ({ cubeLayout := cubeLayout, matrixSource := matrixSource } :
            RelationSource shape columns blockCount) assignment point
  piRlcAlgebra : PiRLC.Algebra
    (RelationSource shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint K shape.cubeVariables)
    (EvaluationFamily K shape)
    Commitment Scalar
    piRlcSemantics
    params
  one : Scalar
  normLaw : Rlc.NormLaw piRlcAlgebra one
  oracle : PiCCS.TranscriptReplay.Oracle K State shape
  initialTranscriptState : State
  absorbPublicInput : State -> Running K Commitment PublicInput shape ->
    Fresh Commitment PublicInput shape -> State
  absorbPiCcsOutput : State -> FullOutputCoordinates.FullOutput K shape -> State
  piRlcResponse : State -> Option Scalar
  piRlcResponseValid : ∀ state challenge,
    piRlcResponse state = some challenge -> piRlcAlgebra.challengeValid challenge

namespace Key

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount width : Nat}
  (key : Key Commitment PublicInput Scalar State shape columns blockCount width)

def relationSource : RelationSource shape columns blockCount where
  cubeLayout := key.cubeLayout
  matrixSource := key.matrixSource

def freshSource : Fin shape.sourceCount :=
  ⟨0, by simp only [Shape.sourceCount, key.freshCount_eq, key.runningCount_eq]; omega⟩

def runningSource : Fin shape.sourceCount :=
  ⟨1, by simp only [Shape.sourceCount, key.freshCount_eq, key.runningCount_eq]; omega⟩

def statement (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    Statement K Commitment PublicInput shape columns blockCount key.baseOps where
  cubeLayout := key.cubeLayout
  matrixSource := key.matrixSource
  commitments := Fin.addCases fresh.commitments running.commitments
  publicInputs := Fin.addCases fresh.publicInputs running.publicInputs
  priorPoint := running.point
  claimedPadCoefficient := fun coordinate =>
    (running.evaluations coordinate.running).pad coordinate.coefficient
  claimedMatrixCoefficient := fun coordinate =>
    (running.evaluations coordinate.running).matrix coordinate.matrix coordinate.coefficient

theorem statement_sumcheckDegreeBound_eq (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    ((key.statement running fresh).verifierInput key.lift).sumcheckDegreeBound = width :=
  key.degreeBoundExact

def transcriptStatement (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    PiCCS.TranscriptReplay.Statement K State shape where
  priorState := key.absorbPublicInput key.initialTranscriptState running fresh
  input := (key.statement running fresh).verifierInput key.lift

def coins (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (rounds : Fin shape.cubeVariables -> SumCheck.Finite.FixedPolynomial K width) :
    FiatShamir.DerivedCoins K State shape :=
  FiatShamir.derive key.oracle.transcript (key.transcriptStatement running fresh)
    { rounds := fun round => (rounds round).toMessage }

def probe (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) (proof : Proof shape width) :
    Probe K shape where
  coins := {
    alpha := (key.coins running fresh proof.piCcsRounds).alpha
    gamma := (key.coins running fresh proof.piCcsRounds).gamma
    roundPoint := (key.coins running fresh proof.piCcsRounds).roundPoint }
  response := {
    rounds := SumCheck.Finite.FixedPhase.RawCertificate.encode
      { rounds := List.ofFn proof.piCcsRounds }
    fullOutput := proof.piCcsOutput }

def challenge (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) (proof : Proof shape width) : Option Scalar :=
  key.piRlcResponse
    (key.absorbPiCcsOutput (key.coins running fresh proof.piCcsRounds).finalState
      proof.piCcsOutput)

def outputFor (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) (proof : Proof shape width)
    (challenge : Scalar) :
    CE.Instance (RelationSource shape columns blockCount) PublicInput
      (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment :=
  let outputs := (key.statement running fresh).publicOutput (key.probe running fresh proof)
  Rlc.output key.piRlcAlgebra key.one challenge key.relationSource
    (key.probe running fresh proof).coins.roundPoint
    (outputs key.freshSource) (outputs key.runningSource)

end Key

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount width : Nat}

/-- The light fold's verifier: Cyclo contribution 2 on SuperNeo v1.2 Sections 7.3 and 7.4. -/
def verify
    (key : Key Commitment PublicInput Scalar State shape columns blockCount width)
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof shape width) :
    Option (CE.Instance (RelationSource shape columns blockCount) PublicInput
      (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment) :=
  if PiCCS.check key.extensionOps key.lift (key.statement running fresh) width
      (key.probe running fresh proof) then
    (key.challenge running fresh proof).map (key.outputFor running fresh proof)
  else
    none

end NightstreamFPrime.Spec.Folding.Light
