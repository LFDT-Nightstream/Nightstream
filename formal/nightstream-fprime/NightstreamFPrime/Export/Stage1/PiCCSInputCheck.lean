import Lean.Data.Json
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Layout.Stage1.PiCCSProofInputs
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

/-!
Owns an executable PiCCS check on caller-supplied messages and running claims.
It does not construct a proof or establish valid openings.
Input schema: [2, commitment[1188], public[270], rounds[28][10][2],
Eval_K[17][54][2], Eval_A[17][14][54][2], running]. The running statement is
[point[28][2], commitments[16][1188], public[16][270],
Eval_K[16][54][2], Eval_A[16][14][54][2]]. Words are canonical Goldilocks
integers. The file is canonical numeric JSON with an optional final newline.
No input data is embedded in a Lean declaration.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSInputCheck

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

private abbrev logicalWidth :=
  PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application

private abbrev publicFits :=
  PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application

abbrev PublicInput := PaperAlgebra.PublicInput
  (logicalWidth := logicalWidth) (publicFits := publicFits)

structure RunningInput where
  point : Vector K 28
  commitments : Vector (Vector F 1188) 16
  publicInputs : Vector (Vector F 270) 16
  evalK : Vector (Vector K 54) 16
  evalA : Vector (Vector (Vector K 54) 14) 16

structure Input where
  commitment : Vector F 1188
  publicInput : Vector F 270
  rounds : Vector (Vector K 10) 28
  evalK : Vector (Vector K 54) 17
  evalA : Vector (Vector (Vector K 54) 14) 17
  running : RunningInput

private def decodeVector {Alpha : Type} (length : Nat)
    (decodeItem : Lean.Json → Except String Alpha) (value : Lean.Json) :
    Except String (Vector Alpha length) := do
  let values ← value.getArr?
  if exactSize : values.size = length then
    (Vector.mk values exactSize).mapM decodeItem
  else
    throw s!"expected array length {length}, got {values.size}"

private def decodeField (value : Lean.Json) : Except String F := do
  let word ← value.getNat?
  if canonical : word < goldilocksModulus then
    pure ⟨word, canonical⟩
  else
    throw "noncanonical Goldilocks word"

private def decodeExtension (value : Lean.Json) : Except String K := do
  let words ← decodeVector 2 decodeField value
  pure ⟨words.get 0, words.get 1⟩

private def decodeRunning (value : Lean.Json) : Except String RunningInput := do
  let values ← value.getArr?
  match values.toList with
  | [point, commitments, publicInputs, evalK, evalA] =>
      pure {
        point := ← decodeVector 28 decodeExtension point
        commitments := ← decodeVector 16 (decodeVector 1188 decodeField)
          commitments
        publicInputs := ← decodeVector 16 (decodeVector 270 decodeField)
          publicInputs
        evalK := ← decodeVector 16 (decodeVector 54 decodeExtension) evalK
        evalA := ← decodeVector 16
          (decodeVector 14 (decodeVector 54 decodeExtension)) evalA }
  | _ => throw "expected five running statement fields"

def decode (value : Lean.Json) : Except String Input := do
  let values ← value.getArr?
  match values.toList with
  | [schema, commitment, publicInput, rounds, evalK, evalA, running] =>
      if (← schema.getNat?) != 2 then
        throw "expected PiCCS input schema 2"
      pure {
        commitment := ← decodeVector 1188 decodeField commitment
        publicInput := ← decodeVector 270 decodeField publicInput
        rounds := ← decodeVector 28 (decodeVector 10 decodeExtension) rounds
        evalK := ← decodeVector 17 (decodeVector 54 decodeExtension) evalK
        evalA := ← decodeVector 17
          (decodeVector 14 (decodeVector 54 decodeExtension)) evalA
        running := ← decodeRunning running }
  | _ => throw "expected seven PiCCS input fields"

private def fieldValue (value : F) : Value := .atom value.val

private def extensionValue (value : K) : Value :=
  .array [fieldValue value.c0, fieldValue value.c1]

private def wordsValue (words : List F) : Value :=
  .array (words.map fieldValue)

private def extensionsValue (values : List K) : Value :=
  .array (values.map extensionValue)

private def vectorValue {Alpha : Type} {length : Nat}
    (encode : Alpha → Value) (values : Vector Alpha length) : Value :=
  .array (values.toList.map encode)

private def runningInputValue (input : RunningInput) : Value :=
  .array [vectorValue extensionValue input.point,
    vectorValue (vectorValue fieldValue) input.commitments,
    vectorValue (vectorValue fieldValue) input.publicInputs,
    vectorValue (vectorValue extensionValue) input.evalK,
    vectorValue (vectorValue (vectorValue extensionValue)) input.evalA]

def inputValue (input : Input) : Value :=
  .array [.atom 2, vectorValue fieldValue input.commitment,
    vectorValue fieldValue input.publicInput,
    vectorValue (vectorValue extensionValue) input.rounds,
    vectorValue (vectorValue extensionValue) input.evalK,
    vectorValue (vectorValue (vectorValue extensionValue)) input.evalA,
    runningInputValue input.running]

/-- Reject alternate number spellings, extra fields, and noncanonical bytes
after the typed decoder has checked every dimension and field word. -/
def parse (text : String) : Except String Input := do
  let input ← decode (← Lean.Json.parse text)
  let canonical := (inputValue input).render
  if text != canonical && text != canonical ++ "\n" then
    throw "expected canonical numeric JSON with an optional final newline"
  pure input

/-- Read the same canonical running-claim encoding at a fixture handoff. -/
def parseRunning (text : String) : Except String RunningInput := do
  let input ← decodeRunning (← Lean.Json.parse text)
  let canonical := (runningInputValue input).render
  if text != canonical && text != canonical ++ "\n" then
    throw "expected canonical running-claim JSON with an optional final newline"
  pure input

def proofValues (input : Input) : PiCCSProofInputs.ProofValues where
  freshCommitment := fun row coefficient => input.commitment.get
    ⟨row.val * ringDegree + coefficient.val, by
      have rowBound : row.val < 22 := row.isLt
      have coefficientBound : coefficient.val < 54 := coefficient.isLt
      change row.val * 54 + coefficient.val < 1188
      omega⟩
  roundCoefficient := fun round coefficient =>
    (input.rounds.get round).get coefficient
  outputEval_K := fun source coefficient =>
    (input.evalK.get source).get coefficient
  outputEval_A := fun source matrix coefficient =>
    ((input.evalA.get source).get matrix).get coefficient

def running (input : Input) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) where
  point := {
    coordinates := input.running.point.toList
    dimension := by
      change input.running.point.toList.length = 28
      simp }
  commitments := fun source row coefficient =>
    (input.running.commitments.get source).get
      ⟨row.val * ringDegree + coefficient.val, by
        have rowBound : row.val < 22 := row.isLt
        have coefficientBound : coefficient.val < 54 := coefficient.isLt
        change row.val * 54 + coefficient.val < 1188
        omega⟩
  publicInputs := fun source => (input.running.publicInputs.get source).get
  evaluations := fun source => {
    pad := (input.running.evalK.get source).get
    matrix := fun matrix => ((input.running.evalA.get source).get matrix).get }

def fresh (input : Input) :
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) where
  commitments := fun _ => (proofValues input).freshCommitment
  publicInputs := fun _ => input.publicInput.get

def verifierInput (input : Input) :
    ProtocolPolynomial.VerifierInput K productionShape where
  constraintPolynomial :=
    ConstraintPolynomialLift.liftConstraintPolynomial K.embed
      Spec.ProductionRelation.polynomial
  priorPoint := (running input).point
  claimedPadCoefficient := fun coordinate =>
    ((running input).evaluations coordinate.running).pad coordinate.coefficient
  claimedMatrixCoefficient := fun coordinate =>
    ((running input).evaluations coordinate.running).matrix coordinate.matrix
      coordinate.coefficient

def outputMessage (input : Input) :
    ProtocolPolynomial.OutputMessage K productionShape where
  freshMatrixImage := fun source matrix =>
    (proofValues input).outputEval_A (freshSourceIndex source) matrix
      Phi81CoefficientKernel.constant
  sourceAssignment := fun source =>
    (proofValues input).outputEval_K source Phi81CoefficientKernel.constant
  padImage := fun coordinate =>
    (proofValues input).outputEval_K (runningSourceIndex coordinate.running)
      coordinate.coefficient
  matrixImage := fun coordinate =>
    (proofValues input).outputEval_A (runningSourceIndex coordinate.running)
      coordinate.matrix coordinate.coefficient

/-- The executable input uses the same statement projection as the one
production key. The key is used only in this proof, not by the checker. -/
theorem verifierInput_eq_production (input : Input)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    verifierInput input =
      ((ProductionKey.key relation ajtai).statement (running input) (fresh input)
        ).verifierInput K.embed := by
  rfl

theorem outputMessage_eq_production (input : Input)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    outputMessage input =
      ((ProductionKey.key relation ajtai).statement (running input) (fresh input)
        ).projectOutput (PiCCSProofInputs.output (proofValues input)) := by
  rfl

def initialClaimFast (input : Input) (gamma : K) : K :=
  SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps gamma
    (Folding.PiCCS.FinalIdentity.targetCoefficientList (verifierInput input))

theorem initialClaimFast_eq_initial (input : Input) (gamma : K) :
    initialClaimFast input gamma = (verifierInput input).initial extensionOps gamma :=
  Folding.PiCCS.FinalIdentity.evaluateTargetCoefficients_eq_initial
    extensionOps extensionLaws (verifierInput input) gamma

def padTerminalFast (input : Input) (gamma : K) (point : PaperAlgebra.Point)
    (message : ProtocolPolynomial.OutputMessage K productionShape) : K :=
  extensionOps.mul
    (SumCheckTruthPath.pointEquality extensionOps point (verifierInput input).priorPoint)
    (SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps gamma
      (Folding.PiCCS.FinalIdentity.outputPadCoefficientList message))

theorem padTerminalFast_eq_paper (input : Input) (gamma : K) (point : PaperAlgebra.Point)
    (message : ProtocolPolynomial.OutputMessage K productionShape) :
    padTerminalFast input gamma point message =
      ProtocolPolynomial.padAtMessage extensionOps (verifierInput input) gamma point
        message :=
  (Folding.PiCCS.FinalIdentity.padAtMessage_eq_pointEquality_mul_horner
    extensionOps extensionLaws (verifierInput input) gamma point message).symm

def matrixTerminalFast (input : Input) (gamma : K) (point : PaperAlgebra.Point)
    (message : ProtocolPolynomial.OutputMessage K productionShape) : K :=
  extensionOps.mul
    (SumCheckTruthPath.pointEquality extensionOps point (verifierInput input).priorPoint)
    (SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps gamma
      (Folding.PiCCS.FinalIdentity.outputMatrixCoefficientList message))

theorem matrixTerminalFast_eq_paper (input : Input) (gamma : K) (point : PaperAlgebra.Point)
    (message : ProtocolPolynomial.OutputMessage K productionShape) :
    matrixTerminalFast input gamma point message =
      ProtocolPolynomial.matrixAtMessage extensionOps (verifierInput input) gamma point
        message :=
  (Folding.PiCCS.FinalIdentity.matrixAtMessage_eq_pointEquality_mul_horner
    extensionOps extensionLaws (verifierInput input) gamma point message).symm

private def powerLoop (value : K) : Nat → K → K
  | 0, accumulated => accumulated
  | exponent + 1, accumulated =>
      powerLoop value exponent (extensionOps.mul value accumulated)

def powerFast (value : K) (exponent : Nat) : K :=
  powerLoop value exponent extensionOps.one

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

def terminalFast (alpha : PaperAlgebra.Point) (gamma : K)
    (point : PaperAlgebra.Point) (pad matrix ccs norm : K) : K :=
  extensionOps.add pad <| extensionOps.add
    (extensionOps.mul
      (powerFast gamma productionShape.matrixEvaluationOffset) matrix)
    (extensionOps.mul (powerFast gamma productionShape.constraintOffset) <|
      extensionOps.mul
        (SumCheckTruthPath.pointEquality extensionOps point alpha) <|
        extensionOps.add ccs
          (extensionOps.mul (powerFast gamma productionShape.freshCount) norm))

theorem terminalFast_eq_paper (input : Input) (alpha : PaperAlgebra.Point) (gamma : K)
    (point : PaperAlgebra.Point)
    (message : ProtocolPolynomial.OutputMessage K productionShape) :
    terminalFast alpha gamma point
        (padTerminalFast input gamma point message)
        (matrixTerminalFast input gamma point message)
        (ProtocolPolynomial.ccsAtMessage extensionOps (verifierInput input) gamma message)
        (ProtocolPolynomial.normAtMessage extensionOps gamma message) =
      ProtocolPolynomial.terminalFromMessage extensionOps (verifierInput input) alpha
        gamma point message := by
  rw [padTerminalFast_eq_paper, matrixTerminalFast_eq_paper]
  unfold terminalFast ProtocolPolynomial.terminalFromMessage
    SignedJointIdentity.gammaTerm
  simp only [powerFast_eq_power]

structure RoundTrace where
  challenges : List K
  states : List Transcript.State
  state : Transcript.State

/-- Absorb each supplied polynomial before deriving its challenge. -/
def traceFrom (input : Input) :
    Transcript.State → List (Fin productionShape.cubeVariables) → RoundTrace
  | state, [] => ⟨[], [], state⟩
  | state, round :: remaining =>
      let polynomial := PiCCSProofInputs.roundPolynomial (proofValues input) round
      let absorbed := Transcript.piCcsOracle.transcript.absorbRound
        state round polynomial.toMessage
      let sample := Transcript.piCcsOracle.transcript.squeeze absorbed
        (.sumcheck round)
      let tail := traceFrom input sample.2 remaining
      ⟨sample.1 :: tail.challenges, sample.2 :: tail.states, tail.state⟩

theorem traceFrom_eq_derive (input : Input) (state : Transcript.State)
    (indices : List (Fin productionShape.cubeVariables)) :
    ((traceFrom input state indices).challenges,
      (traceFrom input state indices).state) =
      FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript
        (fun round =>
          (PiCCSProofInputs.roundPolynomial (proofValues input) round).toMessage)
        state indices := by
  induction indices generalizing state with
  | nil => rfl
  | cons round remaining inductionHypothesis =>
      let polynomial := PiCCSProofInputs.roundPolynomial (proofValues input) round
      let absorbed := Transcript.piCcsOracle.transcript.absorbRound
        state round polynomial.toMessage
      let sample := Transcript.piCcsOracle.transcript.squeeze absorbed
        (.sumcheck round)
      simpa only [traceFrom, FiatShamir.deriveRoundsFrom, polynomial,
        absorbed, sample] using
        congrArg (fun result : List K × Transcript.State =>
          (sample.1 :: result.1, result.2)) (inductionHypothesis sample.2)

private def postRoundClaims :
    List (SumCheck.Finite.FixedPolynomial K 9) → List K → List K
  | [], _ => []
  | _, [] => []
  | polynomial :: rounds, challenge :: challenges =>
      let claim := polynomial.evaluate extensionOps.toOps challenge
      claim :: postRoundClaims rounds challenges

private def outputCommitments (input : Input) :
    Fin productionShape.sourceCount → PaperAlgebra.Commitment :=
  Fin.addCases (fresh input).commitments (running input).commitments

private def outputPublicInputs (input : Input) :
    Fin productionShape.sourceCount → PublicInput :=
  Fin.addCases (fresh input).publicInputs (running input).publicInputs

private def runningValue (input : Input) : Value :=
  let running := running input
  .array [extensionsValue running.point.coordinates,
    .array ((List.finRange productionShape.runningCount).map fun source =>
      wordsValue (serializeCommitment (running.commitments source))),
    .array ((List.finRange productionShape.runningCount).map fun source =>
      wordsValue (serializePublicInput (running.publicInputs source))),
    .array ((List.finRange productionShape.runningCount).map fun source =>
      extensionsValue (List.ofFn (running.evaluations source).pad)),
    .array ((List.finRange productionShape.runningCount).map fun source =>
      .array ((List.finRange productionShape.matrixCount).map fun matrix =>
        extensionsValue (List.ofFn ((running.evaluations source).matrix matrix))))]

/-- Typed handoff and complete numeric result from one PiCCS execution.
Opening validity remains a separate obligation. -/
structure Execution where
  accepted : Bool
  point : PaperAlgebra.Point
  outgoing : Transcript.State
  encoded : Value

def execute (input : Input) : Execution :=
  let running := running input
  let verifierInput := verifierInput input
  let statementState := ProductionKey.absorbPublicInput
    (Transcript.absorb Transcript.initialState Transcript.piCcsDigestDomainTag)
    running (fresh input)
  let pre := Folding.PiCCS.Transcript.deriveFromState
    Transcript.piCcsOracle.transcript statementState
  let indices := canonicalFinIndices productionShape.cubeVariables
  let trace := traceFrom input pre.state indices
  let point : PaperAlgebra.Point := {
    coordinates := trace.challenges
    dimension := by
      have same := congrArg Prod.fst (traceFrom_eq_derive input pre.state indices)
      change trace.challenges = _ at same
      rw [same, FiatShamir.deriveRoundsFrom_values_length,
        canonicalFinIndices_length] }
  let rounds := List.ofFn (PiCCSProofInputs.roundPolynomial (proofValues input))
  let initial := initialClaimFast input pre.gamma
  let claims := postRoundClaims rounds trace.challenges
  let message := outputMessage input
  let pad := padTerminalFast input pre.gamma point message
  let matrix := matrixTerminalFast input pre.gamma point message
  let ccs := ProtocolPolynomial.ccsAtMessage extensionOps verifierInput
    pre.gamma message
  let norm := ProtocolPolynomial.normAtMessage extensionOps pre.gamma message
  let terminal := terminalFast pre.alpha pre.gamma point pad matrix ccs norm
  let accepted := SumCheck.Finite.FixedPhase.checkChain extensionOps.toOps
    initial rounds trace.challenges terminal
  let outgoing := ProductionKey.absorbFullOutput trace.state
    (PiCCSProofInputs.output (proofValues input))
  { accepted, point, outgoing
    encoded := .array [.atom (if accepted then 1 else 0),
      extensionsValue pre.alpha.coordinates, extensionValue pre.gamma,
      wordsValue pre.state, extensionsValue trace.challenges,
      .array (trace.states.map wordsValue), extensionsValue point.coordinates,
      extensionValue initial, extensionsValue claims,
      extensionsValue [pad, matrix, ccs, norm, terminal, claims.getLastD initial],
      .array ((List.finRange productionShape.sourceCount).map fun source =>
        wordsValue (serializeCommitment (outputCommitments input source))),
      .array ((List.finRange productionShape.sourceCount).map fun source =>
        wordsValue (serializePublicInput (outputPublicInputs input source))),
      vectorValue (vectorValue extensionValue) input.evalK,
      vectorValue (vectorValue (vectorValue extensionValue)) input.evalA,
      wordsValue outgoing] }

/-- Result schema: [1, inputEcho, runningEcho, publicBlocks, verifierBlocks,
[accepted, alpha, gamma, preState, roundChallenges, roundStates, rPrime,
initialClaim, postRoundClaims, terminal[6], commitments[17], public[17],
Eval_K, Eval_A, outgoingState]]. Acceptance checks the transcript-bound
fixed-width chain only; input opening validity remains a separate obligation. -/
def checkValue (input : Input) : Value :=
  let result := execute input
  let publicBlocks := ProductionKey.publicInputBlocks (running input) (fresh input)
  .array [.atom 1, inputValue input, runningValue input,
    .array (publicBlocks.map wordsValue),
    .array ((Transcript.verifierInputBlocks (verifierInput input)).map wordsValue),
    result.encoded]

end NightstreamFPrime.Export.Stage1.PiCCSInputCheck
