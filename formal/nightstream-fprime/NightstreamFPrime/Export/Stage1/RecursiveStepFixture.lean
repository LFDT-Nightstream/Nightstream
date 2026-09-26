import NightstreamFPrime.Export.Stage1.BaseStepFixture
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck

/-!
Owns the next caller packet from an explicit prior state, application message,
actual PiCCS proof and sixteen child claims. It uses the existing typed state,
transcript and caller-packet schema. Opening validity and the complete exported
witness assignment must be checked separately.
-/

namespace NightstreamFPrime.Export.Stage1.RecursiveStepFixture

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

private def wordsValue (words : List F) : Value :=
  .array (words.map fun word => .atom word.val)

private def extensionWords (value : K) : List F := [value.c0, value.c1]

private def pointValue (point : List K) : Value :=
  .array (point.map fun value => wordsValue (extensionWords value))

private def valueFromPriorIO
    (sampler : Transcript.State → Option (Transcript.PiRlcSampler.Batch productionShape.sourceCount))
    (context : VerifierContext.Digest4)
    (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (message : AppWitness)
    (input : PiCCSInputCheck.Input) (children : PiCCSInputCheck.RunningInput) :
    IO Value := do
  unless decide (0 < prior.iteration ∧ prior.iteration + 1 < goldilocksModulus) do
    throw (IO.userError "recursive fixture: prior counter must be positive with a canonical successor")
  unless decide (prior.z0.length = 4 ∧ prior.current.length = 4 ∧ message.length = 4) do
    throw (IO.userError "recursive fixture: states and message must contain four words")
  let priorWords := serializePreimage (publicFits := publicFits) prior
  let priorDigest := Poseidon2.hash priorWords
  let priorPublic : PublicInput := encHash (publicFits := publicFits) priorDigest
  unless decide (input.publicInput.toList = List.ofFn priorPublic) do
    throw (IO.userError "recursive fixture: fresh public input differs from the actual prior hash")
  unless decide (serializeRunning (PiCCSInputCheck.running input) =
      serializeRunning (prior.running functionIndex)) do
    throw (IO.userError "recursive fixture: PiCCS running input differs from the actual prior")
  let phase := PiCCSInputCheck.execute input
  unless phase.accepted do
    throw (IO.userError "recursive fixture: PiCCS proof rejected")
  unless decide (children.point.toList = phase.point.coordinates) do
    throw (IO.userError "recursive fixture: child point differs from the PiCCS output")
  let some batch := sampler phase.outgoing
    | throw (IO.userError "recursive fixture: PiRLC sampler shortfall")
  let sourcePublic : Fin productionShape.sourceCount → PublicInput :=
    Fin.addCases (PiCCSInputCheck.fresh input).publicInputs
      (PiCCSInputCheck.running input).publicInputs
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
    | throw (IO.userError "recursive fixture: PiDEC parent is out of range")
  let childPublicWords := (List.finRange productionGlobalParams.k).flatMap
    fun child => List.ofFn (childPublic child)
  unless decide (childPublicWords = children.publicInputs.toList.flatMap
      (fun values => values.toList)) do
    throw (IO.userError "recursive fixture: child public inputs differ from the checked split")
  let childRunning := PiCCSInputCheck.running { input with running := children }
  let applicationOutput := Lifecycle.Stage1.Poseidon2HashChainV1.step
    prior.current message
  let next := { prior with
    iteration := prior.iteration + 1
    current := applicationOutput
    running := fun _ => childRunning }
  let outputWords := serializePreimage (publicFits := publicFits) next
  let outputDigest := Poseidon2.hash outputWords
  let nextFreshPublic : PublicInput := encHash (publicFits := publicFits) outputDigest
  let childCommitmentWords := (List.finRange productionGlobalParams.k).flatMap
    fun child => serializeCommitment (childRunning.commitments child)
  let childEvalKWords := (List.finRange productionGlobalParams.k).flatMap
    fun child => (List.ofFn (childRunning.evaluations child).pad).flatMap extensionWords
  let childEvalAWords := (List.finRange productionGlobalParams.k).flatMap
    fun child => (List.finRange productionShape.matrixCount).flatMap
      fun matrix => (List.ofFn ((childRunning.evaluations child).matrix matrix)
        ).flatMap extensionWords
  let privateInputs := priorWords ++ outputWords ++
    PiCCSProofInputs.serializeProofInputs (PiCCSInputCheck.proofValues input) ++
    childCommitmentWords ++ childEvalKWords ++ childEvalAWords ++
    childPublicWords ++ message
  let publicInputs := List.ofFn priorPublic ++ outputDigest ++ context.toList
  pure <| .array [.atom 1, wordsValue context.toList,
    wordsValue privateInputs, wordsValue publicInputs,
    .array [wordsValue applicationOutput, wordsValue outputDigest,
      wordsValue (List.ofFn nextFreshPublic), pointValue phase.point.coordinates,
      wordsValue phase.outgoing, wordsValue batch.finalState,
      wordsValue parentValues.toList]]

/-- Construct a later caller packet from the actual prior state and C input.
The caller supplies the verifier context; running claims come from the same
typed C input whose fresh public hash is checked by the shared constructor. -/
def valueFromStateIOWith
    (sampler : Transcript.State → Option (Transcript.PiRlcSampler.Batch productionShape.sourceCount))
    (context : VerifierContext.Digest4)
    (iteration : Nat) (z0 current : AppState) (message : AppWitness)
    (input : PiCCSInputCheck.Input) (children : PiCCSInputCheck.RunningInput) :
    IO Value :=
  valueFromPriorIO sampler context {
    verifierKeys := fun _ => context.toList
    iteration := iteration
    z0 := z0
    current := current
    running := fun _ => PiCCSInputCheck.running input
    pc := 1 } message input children

/-- Preserve the checked base-output fixture as the default caller. -/
def valueIOWith
    (sampler : Transcript.State → Option (Transcript.PiRlcSampler.Batch productionShape.sourceCount))
    (context : VerifierContext.Digest4)
    (input : PiCCSInputCheck.Input) (children : PiCCSInputCheck.RunningInput) :
    IO Value :=
  valueFromPriorIO sampler context (BaseStepFixture.outputPreimage context)
    BaseStepFixture.applicationMessage input children

def valueFromStateIO (context : VerifierContext.Digest4)
    (iteration : Nat) (z0 current : AppState) (message : AppWitness)
    (input : PiCCSInputCheck.Input) (children : PiCCSInputCheck.RunningInput) : IO Value :=
  valueFromStateIOWith (fun state => Transcript.PiRlcSampler.piRlcChallengesWithState state productionShape.sourceCount)
    context iteration z0 current message input children

def valueIO (context : VerifierContext.Digest4)
    (input : PiCCSInputCheck.Input) (children : PiCCSInputCheck.RunningInput) : IO Value :=
  valueIOWith (fun state => Transcript.PiRlcSampler.piRlcChallengesWithState state productionShape.sourceCount)
    context input children

end NightstreamFPrime.Export.Stage1.RecursiveStepFixture
