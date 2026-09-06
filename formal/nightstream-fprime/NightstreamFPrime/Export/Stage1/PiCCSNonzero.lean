import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Export.Stage1.PerApplicationStreamingIdentity
import NightstreamFPrime.Layout.Stage1.PiCCSProofInputs
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

/-!
Owns a deterministic nonzero SuperNeo v1.1 verifier-result fixture.
Its source openings and matrix evaluations are synthetic; it does not
establish valid-input phase conformance. Lean computes every verifier coin
from the public statement and the causal round messages.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSNonzero

open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

abbrev fixtureLogicalWidth : Nat :=
  PerApplicationFixedPoint.logicalWidth
    Poseidon2HashChainV1Package.application

abbrev fixturePublicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth fixtureLogicalWidth :=
  PerApplicationFixedPoint.publicFits
    Poseidon2HashChainV1Package.application

abbrev FixturePublicInput :=
  PaperAlgebra.PublicInput
    (logicalWidth := fixtureLogicalWidth)
    (publicFits := fixturePublicFits)

abbrev FixtureFresh :=
  Fresh PaperAlgebra.Commitment FixturePublicInput productionShape

def field (value : Nat) : F := Poseidon2.ofNat (value + 1)

def extension (low high : Nat) : K :=
  ⟨field low, field high⟩

def running : Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput
      (logicalWidth := fixtureLogicalWidth)
      (publicFits := fixturePublicFits))
    productionShape where
  point := {
    coordinates := List.ofFn fun coordinate :
        Fin productionShape.cubeVariables =>
      extension (100 + coordinate.val) (200 + coordinate.val)
    dimension := by simp
  }
  commitments := fun source row coefficient =>
    field (1_000 + source.val * 2_000 + row.val * ringDegree + coefficient.val)
  publicInputs := fun source column => field (source.val + column.val)
  evaluations := fun source => {
    pad := fun coefficient =>
      extension
        (2_000_000 + source.val * 10_000 + coefficient.val)
        (3_000_000 + source.val * 10_000 + coefficient.val)
    matrix := fun matrix coefficient =>
      extension
        (4_000_000 + source.val * 1_000_000 +
          matrix.val * 10_000 + coefficient.val)
        (5_000_000 + source.val * 1_000_000 +
          matrix.val * 10_000 + coefficient.val)
  }

def stateVerifierKey (_delay : Unit := ()) : KeyDigest :=
  let structural :=
    PerApplicationCanonicalPackage.structuralPackageIdentityFast
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits
  (PerApplicationCanonicalPackage.verifierContextDescriptorFromStructural
    Poseidon2HashChainV1Package.fits
    Poseidon2HashChainV1Setup.productionSetup
    structural).digest4.toList

theorem stateVerifierKey_eq_productionContext :
    stateVerifierKey () =
      PerApplicationCanonicalPackage.verifierContextDigest
        Poseidon2HashChainV1Package.fits
        Poseidon2HashChainV1Setup.productionSetup := by
  unfold stateVerifierKey PerApplicationCanonicalPackage.verifierContextDigest
  simp only [PerApplicationCanonicalPackage.structuralPackageIdentityFast_eq,
    PerApplicationCanonicalPackage.verifierContextDescriptorFromStructural_canonical]

theorem stateVerifierKey_length : (stateVerifierKey ()).length = 4 := by
  exact Lifecycle.VerifierContext.Digest4.toList_length _

def stateZ0 : AppState :=
  [field 201, field 202, field 203, field 204]

def stateCurrent : AppState :=
  [field 301, field 302, field 303, field 304]

def statePreimage (vk : KeyDigest := stateVerifierKey ()) : HashPreimage
    (logicalWidth := fixtureLogicalWidth)
    (publicFits := fixturePublicFits) where
  verifierKeys := fun _ => vk
  iteration := 7
  z0 := stateZ0
  current := stateCurrent
  running := fun _ => running
  pc := 1

def statePreimageWords (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : List F :=
  serializePreimage (publicFits := fixturePublicFits)
    (statePreimage vk)

def stateDigest (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : Digest :=
  stateHash (publicFits := fixturePublicFits) (statePreimage vk)

def lifecyclePublicInputFromDigest (digest : Digest) : FixturePublicInput :=
  fun column => encHash (publicFits := fixturePublicFits)
    digest column

def lifecyclePublicInput (vk : KeyDigest := stateVerifierKey ()) :
    FixturePublicInput :=
  lifecyclePublicInputFromDigest (stateDigest () vk)

def statePublicInputWords (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : List F :=
  List.ofFn (lifecyclePublicInput vk)

def freshCommitment : PaperAlgebra.Commitment :=
  fun row coefficient =>
    field (30_000_000 + row.val * ringDegree + coefficient.val)

def freshFromPublicInput (publicInput : FixturePublicInput) : FixtureFresh where
  commitments := fun _ => freshCommitment
  publicInputs := fun _ => publicInput

def fresh (vk : KeyDigest := stateVerifierKey ()) : FixtureFresh :=
  freshFromPublicInput (lifecyclePublicInput vk)

def verifierInput : ProtocolPolynomial.VerifierInput K productionShape where
  constraintPolynomial :=
    ConstraintPolynomialLift.liftConstraintPolynomial K.embed
      ProductionRelation.polynomial
  priorPoint := running.point
  claimedPadCoefficient := fun coordinate =>
    (running.evaluations coordinate.running).pad coordinate.coefficient
  claimedMatrixCoefficient := fun coordinate =>
    (running.evaluations coordinate.running).matrix coordinate.matrix
      coordinate.coefficient

/-- Linear Horner evaluation of the exact verifier-owned initial-claim
coefficient list. -/
def initialClaimFast (gamma : K) : K :=
  NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
    extensionOps.toOps gamma
    (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.targetCoefficientList
      verifierInput)

theorem initialClaimFast_eq_initial (gamma : K) :
    initialClaimFast gamma = verifierInput.initial extensionOps gamma := by
  exact
    NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.evaluateTargetCoefficients_eq_initial
      extensionOps ConcreteCarrier.extensionLaws verifierInput gamma

private def powerLoop (value : K) : Nat → K → K
  | 0, accumulated => accumulated
  | exponent + 1, accumulated =>
      powerLoop value exponent (extensionOps.toOps.mul value accumulated)

/-- Tail-recursive execution of the exact right-nested paper power. -/
def powerFast (value : K) (exponent : Nat) : K :=
  powerLoop value exponent extensionOps.toOps.one

private theorem powerLoop_power (value : K) : ∀ count exponent,
    powerLoop value count
        (TargetPolynomial.power extensionOps.toOps value exponent) =
      TargetPolynomial.power extensionOps.toOps value (count + exponent) := by
  intro count
  induction count with
  | zero =>
      intro exponent
      simp [powerLoop]
  | succ count inductionHypothesis =>
      intro exponent
      simp only [powerLoop]
      change powerLoop value count
        (TargetPolynomial.power extensionOps.toOps value (exponent + 1)) = _
      rw [inductionHypothesis]
      apply congrArg (TargetPolynomial.power extensionOps.toOps value)
      omega

theorem powerFast_eq_power (value : K) (exponent : Nat) :
    powerFast value exponent =
      TargetPolynomial.power extensionOps.toOps value exponent := by
  unfold powerFast
  change powerLoop value exponent
    (TargetPolynomial.power extensionOps.toOps value 0) = _
  rw [powerLoop_power]
  simp

/-- Linear evaluation of the exact paper `Eval_K` terminal. -/
def padTerminalFast (gamma : K)
    (point : CubePoint K productionShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K productionShape) : K :=
  extensionOps.mul
    (SumCheckTruthPath.pointEquality extensionOps point verifierInput.priorPoint)
    (NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
      extensionOps.toOps gamma
      (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.outputPadCoefficientList
        message))

theorem padTerminalFast_eq_paper (gamma : K)
    (point : CubePoint K productionShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K productionShape) :
    padTerminalFast gamma point message =
      ProtocolPolynomial.padAtMessage extensionOps verifierInput gamma point
        message := by
  exact
    (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.padAtMessage_eq_pointEquality_mul_horner
      extensionOps ConcreteCarrier.extensionLaws verifierInput gamma point
        message).symm

/-- Linear evaluation of the exact paper-local `Eval_A` terminal. -/
def matrixTerminalFast (gamma : K)
    (point : CubePoint K productionShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K productionShape) : K :=
  extensionOps.mul
    (SumCheckTruthPath.pointEquality extensionOps point verifierInput.priorPoint)
    (NightstreamFPrime.Spec.SumCheck.Finite.Message.evaluateCoefficients
      extensionOps.toOps gamma
      (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.outputMatrixCoefficientList
        message))

theorem matrixTerminalFast_eq_paper (gamma : K)
    (point : CubePoint K productionShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K productionShape) :
    matrixTerminalFast gamma point message =
      ProtocolPolynomial.matrixAtMessage extensionOps verifierInput gamma point
        message := by
  exact
    (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.matrixAtMessage_eq_pointEquality_mul_horner
      extensionOps ConcreteCarrier.extensionLaws verifierInput gamma point
        message).symm

/-- Assemble the exact paper terminal from independently computed terminal
parts. Only the fixed gamma shifts use the tail-recursive power adapter. -/
def assembleTerminalFast
    (alpha : CubePoint K productionShape.cubeVariables)
    (gamma : K)
    (point : CubePoint K productionShape.cubeVariables)
    (pad matrix ccs norm : K) : K :=
  extensionOps.add pad <| extensionOps.add
    (extensionOps.mul
      (powerFast gamma productionShape.matrixEvaluationOffset) matrix)
    (extensionOps.mul (powerFast gamma productionShape.constraintOffset) <|
      extensionOps.mul
        (SumCheckTruthPath.pointEquality extensionOps point alpha) <|
        extensionOps.add ccs
          (extensionOps.mul
            (powerFast gamma productionShape.freshCount) norm))

theorem assembleTerminalFast_eq_paper
    (alpha : CubePoint K productionShape.cubeVariables)
    (gamma : K)
    (point : CubePoint K productionShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K productionShape) :
    assembleTerminalFast alpha gamma point
        (padTerminalFast gamma point message)
        (matrixTerminalFast gamma point message)
        (ProtocolPolynomial.ccsAtMessage extensionOps verifierInput gamma message)
        (ProtocolPolynomial.normAtMessage extensionOps gamma message) =
      ProtocolPolynomial.terminalFromMessage extensionOps verifierInput alpha
        gamma point message := by
  rw [padTerminalFast_eq_paper, matrixTerminalFast_eq_paper]
  unfold assembleTerminalFast ProtocolPolynomial.terminalFromMessage
    SignedJointIdentity.gammaTerm
  simp only [powerFast_eq_power]

def publicStateFromFresh (freshValue : FixtureFresh) : Transcript.State :=
  ProductionKey.absorbPublicInput
    (Transcript.absorb Transcript.initialState Transcript.piCcsDigestDomainTag)
    running freshValue

def publicState (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : Transcript.State :=
  publicStateFromFresh (fresh vk)

def statementState (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : Transcript.State :=
  publicState () vk

def addCopies : Nat → K → K
  | 0, _ => K.zero
  | count + 1, value => K.add value (addCopies count value)

def roundPolynomial (current : K) :
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial K 9 :=
  let unit := extension 7 11
  let rest := addCopies 8 unit
  let solve := fun first =>
    K.sub current (K.add (K.add first first) rest)
  let firstCandidate := extension 13 17
  let firstAlternative := extension 14 17
  let first :=
    if solve firstCandidate = K.zero then firstAlternative else firstCandidate
  {
    coefficients :=
      [first, solve first, unit, unit, unit, unit, unit, unit, unit, unit]
    coefficients_length := by simp
  }

structure RoundTrace where
  state : Transcript.State
  claim : K
  rounds : List
    (NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial K 9)
  challenges : List K
  states : List Transcript.State
  claims : List K

def advanceRound (trace : RoundTrace)
    (round : Fin productionShape.cubeVariables) : RoundTrace :=
  let polynomial := roundPolynomial trace.claim
  let absorbed := Transcript.piCcsOracle.transcript.absorbRound
    trace.state round polynomial.toMessage
  let sample := Transcript.piCcsOracle.transcript.squeeze absorbed
    (.sumcheck round)
  let nextClaim := polynomial.evaluate extensionOps.toOps sample.1
  {
    state := sample.2
    claim := nextClaim
    rounds := trace.rounds ++ [polynomial]
    challenges := trace.challenges ++ [sample.1]
    states := trace.states ++ [sample.2]
    claims := trace.claims ++ [nextClaim]
  }

def buildRoundTrace
    (preSumcheck : FiatShamir.PreSumcheck K Transcript.State productionShape)
    (initialClaim : K) : RoundTrace :=
  (canonicalFinIndices productionShape.cubeVariables).foldl advanceRound {
    state := preSumcheck.state
    claim := initialClaim
    rounds := []
    challenges := []
    states := []
    claims := []
  }

def zeroRound : NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial K 9 where
  coefficients := List.replicate 10 K.zero
  coefficients_length := by simp

def RoundTrace.round (trace : RoundTrace)
    (index : Fin productionShape.cubeVariables) :=
  trace.rounds.getD index.val zeroRound

def RoundTrace.roundCoefficient (trace : RoundTrace)
    (roundIndex : Fin productionShape.cubeVariables)
    (coefficient : Fin (9 + 1)) : K :=
  (trace.round roundIndex).coefficients.getD coefficient.val K.zero

def basePad (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) : K :=
  extension
    (40_000_000 + source.val * 10_000 + coefficient.val)
    (50_000_000 + source.val * 10_000 + coefficient.val)

def baseMatrix (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : K :=
  extension
    (60_000_000 + source.val * 1_000_000 +
      matrix.val * 10_000 + coefficient.val)
    (80_000_000 + source.val * 1_000_000 +
      matrix.val * 10_000 + coefficient.val)

def outputWithTarget (target : K) :
    FullOutputCoordinates.FullOutput K productionShape where
  padCoordinate := fun source coefficient =>
    if source.val = 1 ∧ coefficient.val = 1 then
      target
    else
      basePad source coefficient
  matrixCoordinate := baseMatrix

def outputMessage
    (output : FullOutputCoordinates.FullOutput K productionShape) :
    ProtocolPolynomial.OutputMessage K productionShape where
  freshMatrixImage := fun source matrix =>
    output.matrixCoordinate (freshSourceIndex source) matrix
      Phi81CoefficientKernel.constant
  sourceAssignment := fun source =>
    output.padCoordinate source Phi81CoefficientKernel.constant
  padImage := fun coordinate =>
    output.padCoordinate (runningSourceIndex coordinate.running)
      coordinate.coefficient
  matrixImage := fun coordinate =>
    output.matrixCoordinate (runningSourceIndex coordinate.running)
      coordinate.matrix coordinate.coefficient

structure TranscriptCore where
  preSumcheck : FiatShamir.PreSumcheck K Transcript.State productionShape
  initialClaim : K
  roundTrace : RoundTrace
  verifierRoundResult : List K × Transcript.State
  verifierRoundPoint : CubePoint K productionShape.cubeVariables

def transcriptCoreFromState (state : Transcript.State) : TranscriptCore :=
  let preSumcheck :=
    NightstreamFPrime.Spec.Folding.PiCCS.Transcript.deriveFromState
      Transcript.piCcsOracle.transcript state
  let initialClaim := initialClaimFast preSumcheck.gamma
  let roundTrace := buildRoundTrace preSumcheck initialClaim
  let verifierRoundResult :=
    FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript
      (fun index => (roundTrace.round index).toMessage) preSumcheck.state
        (canonicalFinIndices productionShape.cubeVariables)
  let verifierRoundPoint : CubePoint K productionShape.cubeVariables := {
    coordinates := verifierRoundResult.1
    dimension := by
      dsimp only [verifierRoundResult]
      rw [FiatShamir.deriveRoundsFrom_values_length,
        canonicalFinIndices_length]
  }
  { preSumcheck
    initialClaim
    roundTrace
    verifierRoundResult
    verifierRoundPoint }

def transcriptCore (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : TranscriptCore :=
  transcriptCoreFromState (statementState () vk)

def terminalForTarget (core : TranscriptCore) (target : K) : K :=
  let message := outputMessage (outputWithTarget target)
  let padTerminal :=
    padTerminalFast core.preSumcheck.gamma core.verifierRoundPoint message
  let matrixTerminal :=
    matrixTerminalFast core.preSumcheck.gamma core.verifierRoundPoint message
  let ccsTerminal :=
    ProtocolPolynomial.ccsAtMessage extensionOps verifierInput
      core.preSumcheck.gamma message
  let normTerminal :=
    ProtocolPolynomial.normAtMessage extensionOps core.preSumcheck.gamma message
  assembleTerminalFast core.preSumcheck.alpha core.preSumcheck.gamma
    core.verifierRoundPoint padTerminal matrixTerminal ccsTerminal normTerminal

/-- Executable inverse in `K = F[X]/(X² - 7)`. The base inverse is the
existing Goldilocks witness computation. Fixture acceptance independently
checks the result. -/
def inverseExtension (value : K) : K :=
  let norm := value.c0 * value.c0 - 7 * value.c1 * value.c1
  let inverseNorm := NightstreamFPrime.Circuit.Hint.inverse norm
  ⟨value.c0 * inverseNorm, (0 - value.c1) * inverseNorm⟩

/-- The chosen output coordinate is affine in the final SumCheck terminal.
Solve it from terminal evaluations at zero and one. -/
def solveTarget (core : TranscriptCore) : K :=
  let terminalZero := terminalForTarget core K.zero
  let terminalOne := terminalForTarget core K.one
  K.mul (K.sub core.roundTrace.claim terminalZero)
    (inverseExtension (K.sub terminalOne terminalZero))

structure FixtureStatement where
  stateKey : KeyDigest
  preimageWords : List F
  digest : Digest
  publicInput : FixturePublicInput
  freshValue : FixtureFresh
  publicState : Transcript.State
  preimageWords_eq : preimageWords = statePreimageWords () stateKey
  digest_eq : digest = stateDigest () stateKey
  publicInput_eq : publicInput = lifecyclePublicInput stateKey
  freshValue_eq : freshValue = fresh stateKey
  publicState_eq : publicState = PiCCSNonzero.publicState () stateKey

/-- Compute and retain the one state digest used by every fixture view. The
proof fields bind each retained value to the unchanged semantic definition. -/
def fixtureStatement (vk : KeyDigest) : FixtureStatement :=
  let preimageWords := statePreimageWords () vk
  let digest := Poseidon2.hash preimageWords
  let publicInput := lifecyclePublicInputFromDigest digest
  let freshValue := freshFromPublicInput publicInput
  let publicState := publicStateFromFresh freshValue
  {
    stateKey := vk
    preimageWords := preimageWords
    digest := digest
    publicInput := publicInput
    freshValue := freshValue
    publicState := publicState
    preimageWords_eq := by rfl
    digest_eq := by rfl
    publicInput_eq := by rfl
    freshValue_eq := by rfl
    publicState_eq := by rfl
  }

structure FixtureBase where
  statement : FixtureStatement
  core : TranscriptCore
  core_eq : core = transcriptCore () statement.stateKey
  output : FullOutputCoordinates.FullOutput K productionShape
  message : ProtocolPolynomial.OutputMessage K productionShape

/-- Build the one key-dependent transcript and solved output shared by the
pure and parallel fixture evaluators. -/
def fixtureBase (vk : KeyDigest) : FixtureBase :=
  let statement := fixtureStatement vk
  let core := transcriptCoreFromState statement.publicState
  let fixtureOutput := outputWithTarget (solveTarget core)
  {
    statement := statement
    core := core
    core_eq := by
      calc
        core = transcriptCoreFromState statement.publicState := rfl
        _ = transcriptCoreFromState
            (PiCCSNonzero.publicState () statement.stateKey) :=
          congrArg transcriptCoreFromState statement.publicState_eq
        _ = transcriptCore () statement.stateKey := rfl
    output := fixtureOutput
    message := outputMessage fixtureOutput
  }

/-- Lazy compatibility projection for dependent fixture modules. -/
def output (vk : KeyDigest := stateVerifierKey ()) :
    FullOutputCoordinates.FullOutput K productionShape :=
  (fixtureBase vk).output

def proofMessagesNonzero (trace : RoundTrace) : Bool :=
  trace.rounds.all fun polynomial =>
    polynomial.coefficients.all fun value => decide (value ≠ K.zero)

def outputEval_KNonzero
    (fixtureOutput : FullOutputCoordinates.FullOutput K productionShape) : Bool :=
  (List.finRange productionShape.sourceCount).all fun source =>
    (List.finRange productionShape.coefficientCount).all fun coefficient =>
      decide (fixtureOutput.padCoordinate source coefficient ≠ K.zero)

def outputEval_ANonzero
    (fixtureOutput : FullOutputCoordinates.FullOutput K productionShape) : Bool :=
  (List.finRange productionShape.sourceCount).all fun source =>
    (List.finRange productionShape.matrixCount).all fun matrix =>
      (List.finRange productionShape.coefficientCount).all fun coefficient =>
        decide (fixtureOutput.matrixCoordinate source matrix coefficient ≠ K.zero)

def freshCommitmentNonzero : Bool :=
  (List.finRange productionProfile.commitmentWidth).all fun row =>
    (List.finRange ringDegree).all fun coefficient =>
      decide (freshCommitment row coefficient ≠ 0)

structure Computed where
  statement : FixtureStatement
  preSumcheck : FiatShamir.PreSumcheck K Transcript.State productionShape
  roundTrace : RoundTrace
  verifierRoundResult : List K × Transcript.State
  verifierRoundPoint : CubePoint K productionShape.cubeVariables
  output : FullOutputCoordinates.FullOutput K productionShape
  initialClaim : K
  padTerminal : K
  matrixTerminal : K
  ccsTerminal : K
  normTerminal : K
  verifierTerminal : K
  outgoingState : Transcript.State
  accepted : Bool
  proofMessagesNonzero : Bool

private def finishComputed (base : FixtureBase)
    (padTerminal matrixTerminal ccsTerminal normTerminal : K)
    (outgoingState : Transcript.State)
    (proofMessagesAreNonzero : Bool) : Computed :=
  let core := base.core
  let verifierTerminal :=
    assembleTerminalFast core.preSumcheck.alpha core.preSumcheck.gamma
      core.verifierRoundPoint padTerminal matrixTerminal ccsTerminal normTerminal
  let accepted :=
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.checkChain
      extensionOps.toOps core.initialClaim (List.ofFn core.roundTrace.round)
        core.verifierRoundResult.1 verifierTerminal
  {
    statement := base.statement
    preSumcheck := core.preSumcheck
    roundTrace := core.roundTrace
    verifierRoundResult := core.verifierRoundResult
    verifierRoundPoint := core.verifierRoundPoint
    output := base.output
    initialClaim := core.initialClaim
    padTerminal := padTerminal
    matrixTerminal := matrixTerminal
    ccsTerminal := ccsTerminal
    normTerminal := normTerminal
    verifierTerminal := verifierTerminal
    outgoingState := outgoingState
    accepted := accepted
    proofMessagesNonzero := proofMessagesAreNonzero
  }

/-- Evaluate the complete fixture with one shared transcript and round trace. -/
def compute (_ : Unit)
    (vk : KeyDigest := stateVerifierKey ()) : Computed :=
  let base := fixtureBase vk
  let core := base.core
  let padTerminal :=
    padTerminalFast core.preSumcheck.gamma core.verifierRoundPoint base.message
  let matrixTerminal :=
    matrixTerminalFast core.preSumcheck.gamma core.verifierRoundPoint base.message
  let ccsTerminal :=
    ProtocolPolynomial.ccsAtMessage extensionOps verifierInput
      core.preSumcheck.gamma base.message
  let normTerminal :=
    ProtocolPolynomial.normAtMessage extensionOps core.preSumcheck.gamma
      base.message
  let outgoingState :=
    ProductionKey.absorbFullOutput core.verifierRoundResult.2 base.output
  finishComputed base padTerminal matrixTerminal ccsTerminal normTerminal
    outgoingState (proofMessagesNonzero core.roundTrace)

private abbrev ComputedTask (Alpha : Type) :=
  Task (Except IO.Error Alpha)

private def prepareComputedValue {Alpha : Type}
    (build : Unit → Alpha) : IO Alpha := do
  pure (build ())

private def preparedComputedValue {Alpha : Type}
    (task : ComputedTask Alpha) : IO Alpha :=
  match task.get with
  | .ok value => pure value
  | .error error => throw error

/-- Evaluate the same fixture record as `compute`, but schedule the independent
terminal formulas and outgoing-state absorption on separate native tasks. -/
def computeIO (vk : KeyDigest := stateVerifierKey ()) : IO Computed := do
  let base := fixtureBase vk
  let core := base.core
  let preSumcheck := core.preSumcheck
  let roundTrace := core.roundTrace
  let verifierRoundResult := core.verifierRoundResult
  let verifierRoundPoint := core.verifierRoundPoint
  let padTerminalTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepareComputedValue fun _ =>
      padTerminalFast preSumcheck.gamma verifierRoundPoint base.message)
  let matrixTerminalTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepareComputedValue fun _ =>
      matrixTerminalFast preSumcheck.gamma verifierRoundPoint base.message)
  let ccsTerminalTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepareComputedValue fun _ =>
      ProtocolPolynomial.ccsAtMessage extensionOps verifierInput
        preSumcheck.gamma base.message)
  let normTerminalTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepareComputedValue fun _ =>
      ProtocolPolynomial.normAtMessage extensionOps preSumcheck.gamma
        base.message)
  let outgoingStateTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepareComputedValue fun _ =>
      ProductionKey.absorbFullOutput verifierRoundResult.2 base.output)
  let proofMessagesNonzeroTask ←
    IO.asTask (prio := Task.Priority.dedicated)
      (prepareComputedValue fun _ => proofMessagesNonzero roundTrace)
  let padTerminal ← preparedComputedValue padTerminalTask
  let matrixTerminal ← preparedComputedValue matrixTerminalTask
  let ccsTerminal ← preparedComputedValue ccsTerminalTask
  let normTerminal ← preparedComputedValue normTerminalTask
  let outgoingState ← preparedComputedValue outgoingStateTask
  let proofMessagesNonzero ←
    preparedComputedValue proofMessagesNonzeroTask
  pure (finishComputed base padTerminal matrixTerminal ccsTerminal normTerminal
    outgoingState proofMessagesNonzero)

def Computed.proofValues (computed : Computed) :
    PiCCSProofInputs.ProofValues where
  freshCommitment := freshCommitment
  roundCoefficient := computed.roundTrace.roundCoefficient
  outputEval_K := computed.output.padCoordinate
  outputEval_A := computed.output.matrixCoordinate

end NightstreamFPrime.Export.Stage1.PiCCSNonzero
