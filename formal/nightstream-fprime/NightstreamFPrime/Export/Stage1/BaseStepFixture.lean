import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Layout.Stage1.PiCCSProofInputs
import NightstreamFPrime.Lifecycle.PilotZeroRunning

/-!
Owns compact caller inputs for one base step of the canonical application.
The inner zero PiCCS transcript is a base-only placeholder, not valid PiCCS
conformance. The resulting base witness must pass every physical and logical
row before its bounded carrier can be used as a fresh CCS opening.

Schema: [1, context, privateInputs, publicInputs, [applicationOutput,
outputDigest, nextFreshPublicInput, roundPoint, piCcsState, piRlcState,
piRlcParentPublicInput]]. All fields are canonical Goldilocks words; each
round-point element is a two-word extension value. No rows or full assignment
are emitted. The Rust consumer checks the context against its loaded package.
-/

namespace NightstreamFPrime.Export.Stage1.BaseStepFixture

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private abbrev logicalWidth :=
  PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application

private abbrev publicFits :=
  PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application

private abbrev PublicInput := PaperAlgebra.PublicInput
  (logicalWidth := logicalWidth) (publicFits := publicFits)

private def initialState : AppState :=
  [202, 203, 204, 205].map Poseidon2.ofNat

def applicationMessage : AppWitness :=
  [7, 11, 13, 17].map Poseidon2.ofNat

private def priorPreimage (context : VerifierContext.Digest4) :
    HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits) where
  verifierKeys := fun _ => context.toList
  iteration := 0
  z0 := initialState
  current := initialState
  running := fun _ => defaultRunning
  pc := 1

/-- The actual output preimage of this base fixture, used as the next prior. -/
def outputPreimage (context : VerifierContext.Digest4) :
    HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  { priorPreimage context with
    iteration := 1
    current := Lifecycle.Stage1.Poseidon2HashChainV1.step
      initialState applicationMessage }

private def zeroProof : PiCCSProofInputs.ProofValues where
  freshCommitment := fun _ _ => 0
  roundCoefficient := fun _ _ => K.zero
  outputEval_K := fun _ _ => K.zero
  outputEval_A := fun _ _ _ => K.zero

private def wordsValue (words : List F) : Value :=
  .array (words.map fun word => .atom word.val)

private def pointValue (point : List K) : Value :=
  .array (point.map fun value => wordsValue [value.c0, value.c1])

/-- Construct one base-step caller packet from the explicit fixture context.
Sampler shortfall and an out-of-range parent fail before any file is emitted.
All 17 challenges use the fixed five-symbol alphabet and existing `T = 216`
profile; this constructor does not select a sampler or decomposition policy. -/
def valueIO (context : VerifierContext.Digest4) : IO Value := do
  let prior := priorPreimage context
  let priorWords := serializePreimage (publicFits := publicFits) prior
  let priorDigest := Poseidon2.hash priorWords
  let priorPublic : PublicInput := encHash (publicFits := publicFits) priorDigest
  let fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) := {
    commitments := fun _ => zeroProof.freshCommitment
    publicInputs := fun _ => priorPublic }
  let statementState := ProductionKey.absorbPublicInput
    (Transcript.absorb Transcript.initialState Transcript.piCcsDigestDomainTag)
    (defaultRunning (logicalWidth := logicalWidth) (publicFits := publicFits))
    fresh
  let preSumcheck := Folding.PiCCS.Transcript.deriveFromState
    Transcript.piCcsOracle.transcript statementState
  let rounds := FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript
    (fun index => (PiCCSProofInputs.roundPolynomial zeroProof index).toMessage)
    preSumcheck.state (canonicalFinIndices productionShape.cubeVariables)
  let piCcsState := ProductionKey.absorbFullOutput rounds.2
    (PiCCSProofInputs.output zeroProof)
  let some batch := Transcript.PiRlcSampler.piRlcChallengesWithState
      piCcsState productionShape.sourceCount
    | throw (IO.userError "base-step fixture: PiRLC sampler shortfall")
  let sourcePublic : Fin productionShape.sourceCount → PublicInput :=
    Fin.addCases (fun _ : Fin productionShape.freshCount => priorPublic)
      (fun _ : Fin productionShape.runningCount => fun _ => 0)
  let parentValues := Array.ofFn
    (Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
      batch.challenges sourcePublic)
  let parentPublic : PublicInput := fun column =>
    parentValues[column.val]'(by
      simp only [parentValues, Array.size_ofFn]
      exact column.isLt)
  let some childPublic := Folding.PiDEC.PaperVerifier.PublicInputSplit.checked
      (PaperAlgebra.publicInputSplit
        Poseidon2HashChainV1Setup.productionAjtaiKey) parentPublic
    | throw (IO.userError "base-step fixture: PiDEC parent is out of range")
  let childPublicWords := (List.finRange productionGlobalParams.k).flatMap
    fun child => List.ofFn (childPublic child)
  let next := outputPreimage context
  let applicationOutput := next.current
  let outputWords := serializePreimage (publicFits := publicFits) next
  let outputDigest := Poseidon2.hash outputWords
  let nextFreshPublic : PublicInput :=
    encHash (publicFits := publicFits) outputDigest
  let childCommitmentWords := List.replicate
    (productionGlobalParams.k * productionProfile.commitmentWidth * ringDegree)
    (0 : F)
  let childEvalKWords := List.replicate
    (productionGlobalParams.k * productionShape.coefficientCount * 2) (0 : F)
  let childEvalAWords := List.replicate
    (productionGlobalParams.k * productionShape.matrixCount *
      productionShape.coefficientCount * 2) (0 : F)
  let privateInputs := priorWords ++ outputWords ++
    PiCCSProofInputs.serializeProofInputs zeroProof ++
    childCommitmentWords ++ childEvalKWords ++ childEvalAWords ++
    childPublicWords ++ applicationMessage
  let publicInputs := List.ofFn priorPublic ++ outputDigest ++ context.toList
  pure <| .array [.atom 1, wordsValue context.toList,
    wordsValue privateInputs, wordsValue publicInputs,
    .array [wordsValue applicationOutput, wordsValue outputDigest,
      wordsValue (List.ofFn nextFreshPublic), pointValue rounds.1,
      wordsValue piCcsState, wordsValue batch.finalState,
      wordsValue parentValues.toList]]

end NightstreamFPrime.Export.Stage1.BaseStepFixture
