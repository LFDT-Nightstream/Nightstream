import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Layout.Stage1.PiCCSProofInputs
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns one deterministic, valid, nonzero SuperNeo v1.1 PiCCS conformance
fixture. Lean computes every verifier coin from the complete public statement
and the causal round messages. This file does not define a second verifier.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSNonzero

open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

def field (value : Nat) : F := Poseidon2.ofNat (value + 1)

def extension (low high : Nat) : K :=
  ⟨field low, field high⟩

def running : Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    productionShape where
  point := {
    coordinates := List.ofFn fun coordinate :
        Fin productionShape.cubeVariables =>
      extension (100 + coordinate.val) (200 + coordinate.val)
    dimension := by simp
  }
  commitments := fun source row coefficient =>
    field (1_000 + source.val * 2_000 + row.val * ringDegree + coefficient.val)
  publicInputs := fun source column =>
    field (100_000 + source.val * 100_000 + column.val)
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

/-- Exact static words for this conformance fixture. Production setup must
replace these fixture words with its authoritative relation, application,
NIFS-key, and commitment-key serializations before it computes a context. -/
def fixtureContextAuthority : VerifierContext.Authority where
  relationWords := VerifierContext.profileWords ++
    VerifierContext.scheduleWords
  applicationWords := stateDomainTag
  nifsKeyWords := Transcript.piCcsDigestDomainTag ++
    VerifierContext.scheduleWords
  commitmentKeyWords :=
    [Poseidon2.ofNat productionProfile.commitmentWidth,
      Poseidon2.ofNat ringDegree, Poseidon2.ofNat 81, Poseidon2.ofNat 1]

def stateVerifierKey : KeyDigest :=
  VerifierContext.digest fixtureContextAuthority

theorem stateVerifierKey_length : stateVerifierKey.length = 4 := by
  exact VerifierContext.digest_length fixtureContextAuthority

def stateZ0 : AppState :=
  [field 201, field 202, field 203, field 204]

def stateCurrent : AppState :=
  [field 301, field 302, field 303, field 304]

def statePreimage : HashPreimage
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) where
  verifierKeys := fun _ => stateVerifierKey
  iteration := 7
  z0 := stateZ0
  current := stateCurrent
  running := fun _ => running
  pc := 1

def statePreimageWords : List F :=
  serializePreimage (publicFits := Data.publicFits) statePreimage

def stateDigest : Digest :=
  stateHash (publicFits := Data.publicFits) statePreimage

def lifecyclePublicInput : PaperAlgebra.PublicInput
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) :=
  fun column => encHash (publicFits := Data.publicFits) stateDigest column

def statePublicInputWords : List F :=
  List.ofFn lifecyclePublicInput

def freshCommitment : PaperAlgebra.Commitment :=
  fun row coefficient =>
    field (30_000_000 + row.val * ringDegree + coefficient.val)

def fresh : Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    productionShape where
  commitments := fun _ => freshCommitment
  publicInputs := fun _ => lifecyclePublicInput

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

def publicState : Transcript.State :=
  ProductionKey.absorbPublicInput
    (Transcript.absorb Transcript.initialState Transcript.piCcsDigestDomainTag)
    running fresh

def statementState : Transcript.State :=
  publicState

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

/-- The exact coordinate solved once from the verifier terminal equation.
Its acceptance is checked again by executable Lean and every emitted row. -/
def solvedTarget : K :=
  ⟨14649974621493256179, 14821119126112530179⟩

def output : FullOutputCoordinates.FullOutput K productionShape where
  padCoordinate := fun source coefficient =>
    if source.val = 1 ∧ coefficient.val = 1 then
      solvedTarget
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

def proofMessagesNonzero (trace : RoundTrace) : Bool :=
  trace.rounds.all fun polynomial =>
    polynomial.coefficients.all fun value => decide (value ≠ K.zero)

def outputEval_KNonzero : Bool :=
  (List.finRange productionShape.sourceCount).all fun source =>
    (List.finRange productionShape.coefficientCount).all fun coefficient =>
      decide (output.padCoordinate source coefficient ≠ K.zero)

def outputEval_ANonzero : Bool :=
  (List.finRange productionShape.sourceCount).all fun source =>
    (List.finRange productionShape.matrixCount).all fun matrix =>
      (List.finRange productionShape.coefficientCount).all fun coefficient =>
        decide (output.matrixCoordinate source matrix coefficient ≠ K.zero)

def freshCommitmentNonzero : Bool :=
  (List.finRange productionProfile.commitmentWidth).all fun row =>
    (List.finRange ringDegree).all fun coefficient =>
      decide (freshCommitment row coefficient ≠ 0)

structure Computed where
  preSumcheck : FiatShamir.PreSumcheck K Transcript.State productionShape
  roundTrace : RoundTrace
  verifierRoundResult : List K × Transcript.State
  verifierRoundPoint : CubePoint K productionShape.cubeVariables
  initialClaim : K
  padTerminal : K
  matrixTerminal : K
  ccsTerminal : K
  normTerminal : K
  verifierTerminal : K
  outgoingState : Transcript.State
  accepted : Bool
  proofMessagesNonzero : Bool

/-- Evaluate the complete fixture with one shared transcript and round trace. -/
def compute : Computed :=
  let preSumcheck :=
    NightstreamFPrime.Spec.Folding.PiCCS.Transcript.deriveFromState
      Transcript.piCcsOracle.transcript statementState
  let initialClaim :=
    verifierInput.initial extensionOps preSumcheck.gamma
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
  let message := outputMessage output
  let padTerminal :=
    ProtocolPolynomial.padAtMessage extensionOps verifierInput
      preSumcheck.gamma verifierRoundPoint message
  let matrixTerminal :=
    ProtocolPolynomial.matrixAtMessage extensionOps verifierInput
      preSumcheck.gamma verifierRoundPoint message
  let ccsTerminal :=
    ProtocolPolynomial.ccsAtMessage extensionOps verifierInput
      preSumcheck.gamma message
  let normTerminal :=
    ProtocolPolynomial.normAtMessage extensionOps preSumcheck.gamma message
  let verifierTerminal :=
    ProtocolPolynomial.terminalFromMessage extensionOps verifierInput
      preSumcheck.alpha preSumcheck.gamma verifierRoundPoint message
  let outgoingState :=
    ProductionKey.absorbFullOutput verifierRoundResult.2 output
  let accepted :=
    NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.checkChain
      extensionOps.toOps initialClaim (List.ofFn roundTrace.round)
        verifierRoundResult.1 verifierTerminal
  {
    preSumcheck := preSumcheck
    roundTrace := roundTrace
    verifierRoundResult := verifierRoundResult
    verifierRoundPoint := verifierRoundPoint
    initialClaim := initialClaim
    padTerminal := padTerminal
    matrixTerminal := matrixTerminal
    ccsTerminal := ccsTerminal
    normTerminal := normTerminal
    verifierTerminal := verifierTerminal
    outgoingState := outgoingState
    accepted := accepted
    proofMessagesNonzero := proofMessagesNonzero roundTrace
  }

def Computed.proofValues (computed : Computed) :
    PiCCSProofInputs.ProofValues where
  freshCommitment := freshCommitment
  roundCoefficient := computed.roundTrace.roundCoefficient
  outputEval_K := output.padCoordinate
  outputEval_A := output.matrixCoordinate

end NightstreamFPrime.Export.Stage1.PiCCSNonzero
