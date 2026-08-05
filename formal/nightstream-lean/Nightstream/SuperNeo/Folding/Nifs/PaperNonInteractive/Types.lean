import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.HonestCompleteness
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
import Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness
import Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier
import Nightstream.SuperNeo.SumCheck.FixedPhase

/-!
Typed public data and deterministic dataflow for the paper SuperNeo NIFS.

Source: SuperNeo Sections 7.3--7.5 and HyperNova Definition 12.

Owns: the separated fresh/running public carriers; one prover message with a
joint-`Pi_CCS` certificate, coefficient-complete output, and `Pi_DEC` child
messages; the exact paper statement; verifier-owned absorption of the complete
public NIFS input before any challenge; verifier-derived `Pi_CCS` and
`Pi_RLC` challenges; the computed combined parent; and the computed running
output.

Does not own: soundness, completeness, extraction, concrete Poseidon2/Ajtai,
Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

No prover field carries alpha, gamma, a SumCheck point, a `Pi_RLC` challenge,
the combined parent, a terminal value, or an acceptance bit.

| Protocol phase | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| NIFS input | separate the fresh CCS batch from the running CE batch | direct dataflow | `Fresh`, `Running` |
| prover message | carry only joint-SumCheck messages, the complete `Pi_CCS` output, and `Pi_DEC` children | direct dataflow | `Proof` |
| statement | reconstruct the paper source statement from verifier-owned key and public claims | computed | `Key.statement` |
| non-interactive input | absorb the complete running/fresh public pair before any challenge | computed | `Key.publicInputState` |
| `Pi_CCS` | replay all coins and the complete output absorption in protocol order | computed | `Key.piCcsExecution`, `Key.piCcsProbe` |
| `Pi_RLC` | sample after `Pi_CCS` and compute the combined parent | computed | `Key.piRlcChallenges`, `Key.parent` |
| `Pi_DEC` | form the operational parent/children attempt | computed/direct dataflow | `Key.piDecAttempt` |
| NIFS output | derive the running result from the checked child messages | computed | `Key.output` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uState

/-- Public fresh CCS claims.  The common relation structure and fresh norm
stage are fixed by the verifier key and therefore are not repeated here. -/
structure Fresh
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (shape : Shape) where
  commitments : Fin shape.freshCount -> Commitment
  publicInputs : Fin shape.freshCount -> PublicInput

/-- Public running CE claims in the exact paper coefficient layout.  One
shared point and one complete matrix/coefficient family per running claim
replace an untyped array plus duplicated structure fields. -/
structure Running
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (shape : Shape) where
  point : CubePoint Extension shape.cubeVariables
  commitments : Fin shape.runningCount -> Commitment
  publicInputs : Fin shape.runningCount -> PublicInput
  evaluations : Fin shape.runningCount -> EvaluationFamily Extension shape

/-- The sole prover message.  All challenge and parent data are absent. -/
structure Proof
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (shape : Shape)
    (degreeBound : Nat) where
  piCcsRounds : Fin shape.cubeVariables ->
    Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial Extension degreeBound
  piCcsOutput : FullOutputCoordinates.FullOutput Extension shape
  piDecCommitments : Fin shape.runningCount -> Commitment
  piDecEvaluations : Fin shape.runningCount -> EvaluationFamily Extension shape

/-- Static paper verifier key and permitted primitive contracts.

The transcript oracle is abstract but its typed input/order is fixed by
`ProtocolVerifier`.  Before that replay starts, `absorbPublicInput` receives
the complete running/fresh public pair and the key-owned initial state.
`piRlcResponse` starts only from the state obtained after the complete
`Pi_CCS` output has been absorbed.  Returning a valid strong-set element is
the paper's sampling-set contract; a bounded concrete sampler may later add
its explicit shortfall event.  These abstract functions may still ignore or
collide on their inputs; the random-oracle boundary names those events
separately. -/
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
  matrixCountPositive : 0 < shape.matrixCount
  identityFirstEntry : forall
      (vertex : BooleanVertex shape.cubeVariables)
      (column : Fin columns),
    matrixSource.matrices ⟨0, matrixCountPositive⟩ vertex column =
      cubeLayout.paddedIdentityEntry baseOps.zero baseOps.one vertex column
  constantLaw : ConstantTermLaw baseOps matrixSource.kernel
  challengeSetSize : Nat
  /-- Canonical semantic view used by `Pi_RLC`, `Pi_DEC`, and the public NIFS
  relation. It may normalize malformed layout fields, but it must agree with
  the paper relation at `matrixSource`. -/
  piRlcSemantics : RelationSemantics
    (MatrixSource F shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape)
    Commitment
  /-- The authority-bearing opening fields of the semantic adapter must agree
  with the paper relation at every norm bound. A concrete adapter may change
  evaluation representation, but it cannot change commitment, public input,
  or norm membership. -/
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
        (MatrixSource F shape columns blockCount)
        PublicInput
        (CubePoint Extension shape.cubeVariables)
        (EvaluationFamily Extension shape)
        Commitment)
      (assignment : Assignment F columns),
    statement.constraintSystem = matrixSource ->
      (PiRLC.PaperCorrections.CorrectedAmbientHolds
          (paperRelationSemantics baseOps extensionOps lift openingMaps)
          params statement assignment <->
        PiRLC.PaperCorrections.CorrectedAmbientHolds
          piRlcSemantics params statement assignment)
  /-- The concrete evaluator must agree with the paper evaluator at the
  verifier-owned matrix source. This field also prevents an adapter from
  changing which evaluation points are valid. -/
  evaluationAgreement : forall
      (assignment : Assignment F columns)
      (point : CubePoint Extension shape.cubeVariables),
    piRlcSemantics.evaluationPointValid matrixSource point /\
      piRlcSemantics.evaluations matrixSource assignment point =
        (paperRelationSemantics baseOps extensionOps lift openingMaps).evaluations
          matrixSource assignment point
  piRlcEvaluationsSize : forall system assignment point,
    (piRlcSemantics.evaluations system assignment point).size = 1
  piRlcAlgebra : PiRLC.Algebra
    (MatrixSource F shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape)
    Commitment Scalar
    piRlcSemantics
    params
  piDecAlgebra : PiDEC.Algebra
    (MatrixSource F shape columns blockCount)
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
    piDecEvaluationArity.count matrixSource = 1
  piDecDecision : forall attempt,
    Decidable (PiDEC.PaperVerifier.Accepted piDecAlgebra
      piDecEvaluationArity attempt)
  oracle : ProtocolVerifier.Oracle Extension State shape
  initialTranscriptState : State
  absorbPublicInput : State ->
    Running Extension Commitment PublicInput shape ->
    Fresh Commitment PublicInput shape ->
    State
  /-- Authority-bearing post-SumCheck absorption. It receives the complete
  paper `y'` family, not the scalar projection used by the terminal check. -/
  absorbPiCcsOutput : State ->
    FullOutputCoordinates.FullOutput Extension shape -> State
  piRlcResponse : State -> Fin arity.total -> Scalar
  piRlcResponseValid : forall state index,
    piRlcAlgebra.challengeValid (piRlcResponse state index)

namespace Key

/-- Exact equality between the joint paper source count and `K+k`. -/
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

/-- Exact equality between the public running carrier and the `Pi_DEC`
output arity. -/
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

/-- Canonical relation semantics selected by the key. -/
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

/-- Public paper statement reconstructed without a witness. -/
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
  claimedCoefficient := fun coordinate =>
    running.evaluations coordinate.running coordinate.matrix
      coordinate.coefficient
  matrixCountPositive := key.matrixCountPositive
  identityFirstEntry := key.identityFirstEntry

/-- The common fixed SumCheck width is exactly the syntactic degree selected
by this key's paper constraint polynomial. -/
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

/-- Exact key selection supplies the older representability inequality as a
derived fact. -/
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

/-- Transcript state after the verifier has absorbed the complete public NIFS
input.  This operation precedes every `Pi_CCS` challenge. -/
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

/-- Minimal certificate checked by the joint-polynomial verifier. -/
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
    ProtocolVerifier.Certificate Extension shape where
  rounds := fun round => (proof.piCcsRounds round).toMessage
  output := (key.statement running fresh).projectOutput proof.piCcsOutput

/-- The same prover data as a ghost-free fixed-width SumCheck certificate. -/
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

/-- Exact verifier-derived `Pi_CCS` replay. -/
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
    (proof : Proof Extension Commitment shape degreeBound) :=
  let execution :=
    ProtocolVerifier.derive key.oracle (key.publicInputState running fresh)
      (StrongReduction.Statement.verifierInput key.lift
        (key.statement running fresh))
      (key.piCcsCertificate running fresh proof)
  { execution with
    outgoingState :=
      key.absorbPiCcsOutput execution.coins.finalState proof.piCcsOutput }

/-- The coefficient-complete public-coin probe represented by the one NIFS
message.  Its coins and finite certificate are verifier-derived projections;
only `piCcsOutput` is supplied by the prover. -/
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

/-- Coefficient-complete public output of `Pi_CCS`, indexed in the exact
`K+k` order expected by `Pi_RLC`. -/
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
      (MatrixSource F shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment :=
  fun source =>
    (key.statement running fresh).publicOutput
      (key.piCcsProbe running fresh proof)
      (Fin.cast key.total_eq_sourceCount source)

/-- `Pi_RLC` challenges begin at the transcript state reached only after the
complete joint output has been absorbed. -/
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
    Fin key.arity.total -> Scalar :=
  key.piRlcResponse
    (key.piCcsExecution running fresh proof).outgoingState

/-- Verifier-computed `CE(B)` parent. -/
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
  PiRLC.combinedOutput key.piRlcAlgebra key.matrixSource
    (key.piCcsExecution running fresh proof).coins.roundPoint
    (key.piCcsOutputs running fresh proof)
    (key.piRlcChallenges running fresh proof)

/-- Operational `Pi_DEC` message boundary over the verifier-computed parent. -/
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
    (proof : Proof Extension Commitment shape degreeBound) :
    PiDEC.PaperVerifier.Attempt
      (MatrixSource F shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params where
  parent := key.parent running fresh proof
  messages := fun child => {
    commitment := proof.piDecCommitments (Fin.cast key.outputCount_eq child)
    evaluations := #[proof.piDecEvaluations (Fin.cast key.outputCount_eq child)]
  }

/-- Public running product computed from the parent and typed child messages. -/
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
    Running Extension Commitment PublicInput shape :=
  let attempt := key.piDecAttempt running fresh proof
  let children := PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt
  {
    point := attempt.parent.point
    commitments := fun runningIndex =>
      (children (Fin.cast key.runningCount_eq_outputCount runningIndex)).commitment
    publicInputs := fun runningIndex =>
      (children (Fin.cast key.runningCount_eq_outputCount runningIndex)).publicInput
    evaluations := proof.piDecEvaluations
  }

end Key

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
