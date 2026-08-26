import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Lifecycle.Transcript

/-!
Owns compact executable vectors for the production PiRLC sampler. The vectors
cover the exact 16-bit decoder, the fixed 54-of-64 bound, explicit shortfall,
all eight Poseidon2 digest windows per scalar, and transcript-state chaining.
They emit no constraints and do not claim PiRLC phase closure.
-/

namespace NightstreamFPrime.Export.Stage1.PiRlcSamplerParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Lifecycle.Transcript
open NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Sampling
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet
open ProductionSchedule

def boolValue (value : Bool) : Value :=
  .atom (if value then 1 else 0)

def fieldValue (value : F) : Value := .atom value.val

def fieldWordsValue (values : List F) : Value :=
  .array (values.map fieldValue)

def chunkValue (value : Chunk) : Value := .atom value.val

def chunkWordsValue (values : List Chunk) : Value :=
  .array (values.map chunkValue)

def coefficientValue (value : Coefficient) : Value := .atom value.val

def coefficientWordsValue (values : List Coefficient) : Value :=
  .array (values.map coefficientValue)

def optionCoefficientsValue : Option (List Coefficient) → Value
  | none => .array [.atom 0]
  | some values => .array [.atom 1, coefficientWordsValue values]

def optionScalarValue : Option ProductionStrongSet.Scalar → Value
  | none => .array [.atom 0]
  | some scalar =>
      .array [.atom 1,
        .array ((List.finRange coefficientCount).map fun position =>
          coefficientValue (scalar position))]

def optionRingValue : Option RingF → Value
  | none => .array [.atom 0]
  | some challenge =>
      .array [.atom 1,
        fieldWordsValue ((List.finRange ringDegree).map challenge)]

def chunkOfNat (value : Nat) : Chunk :=
  ⟨value % chunkModulus, Nat.mod_lt _ (by decide)⟩

def rejectionChunk : Chunk := chunkOfNat rejectionBucket

def acceptedChunks (count : Nat) : List Chunk :=
  (List.range count).map chunkOfNat

/-- Exactly ten rejections followed by 54 accepted chunks. The least
successful cursor is the final candidate. -/
def successCandidates : List Chunk :=
  List.replicate 10 rejectionChunk ++ acceptedChunks 54

/-- Exactly eleven rejections and 53 accepted chunks. This must fail. -/
def shortfallCandidates : List Chunk :=
  List.replicate 11 rejectionChunk ++ acceptedChunks 53

def candidateAt (candidates : List Chunk) (index : Nat) : Chunk :=
  candidates.getD index rejectionChunk

def packPair (low high : Chunk) : F :=
  Poseidon2.ofNat (low.val + 2 ^ 16 * high.val)

/-- Pack the 64 direct candidates into the same eight-by-four field-lane
shape used by the concrete transcript sampler. -/
def packedDigestWords (candidates : List Chunk) : List (List F) :=
  (List.range digestRounds).map fun round =>
    (List.range (chunksPerDigest / 2)).map fun lane =>
      let start := round * chunksPerDigest + lane * 2
      packPair (candidateAt candidates start)
        (candidateAt candidates (start + 1))

def directSample (candidates : List Chunk) : Option (List Coefficient) :=
  FirstAccepted.boundedSample verifier coefficientCount candidates

def injectedCaseValue (candidates : List Chunk) : Value :=
  .array [chunkWordsValue candidates,
    .array ((packedDigestWords candidates).map fieldWordsValue),
    optionCoefficientsValue (directSample candidates)]

def decodedCoefficient (candidate : Chunk) : Option Coefficient :=
  if verifier.accepts candidate then some (verifier.symbol candidate) else none

def decoderCaseValue (candidate : Chunk) : Value :=
  .array [chunkValue candidate,
    boolValue (verifier.accepts candidate),
    match decodedCoefficient candidate with
    | none => .array [.atom 0]
    | some coefficient =>
        .array [.atom 1, coefficientValue coefficient,
          fieldValue (Phi81StrongSet.embedCoefficient coefficient)]]

def initialState : State := Poseidon2.zeroState

structure CoordinateTrace where
  coordinate : Nat
  beforeState : State
  enteredState : State
  blockStates : List State
  candidates : List Chunk
  coefficients : Option (List Coefficient)
  nextState : State

/-- Evaluate each complete digest block once. The counter follows the abstract
schedule even though the concrete `digestBlock` does not absorb it. -/
def collectBlockStates (coordinate : Nat) :
    (round remaining : Nat) → State → List State × State
  | _, 0, state => ([], state)
  | round, remaining + 1, state =>
      let nextState := (machine.digestBlock state (coordinate + round)).1
      let tail := collectBlockStates coordinate (round + 1) remaining nextState
      (state :: tail.1, tail.2)

def computeCoordinate (coordinate : Nat) (beforeState : State) :
    CoordinateTrace :=
  let enteredState := enterScalar beforeState coordinate
  let collected := collectBlockStates coordinate 0 digestRounds enteredState
  let candidates := collected.1.flatMap fun state => List.ofFn (digestChunks state)
  { coordinate
    beforeState
    enteredState
    blockStates := collected.1
    candidates
    coefficients := directSample candidates
    nextState := collected.2 }

def computeTranscript : Nat → Nat → State → List CoordinateTrace × State
  | 0, _, state => ([], state)
  | remaining + 1, coordinate, state =>
      let entry := computeCoordinate coordinate state
      let rest := computeTranscript remaining (coordinate + 1) entry.nextState
      (entry :: rest.1, rest.2)

def blockValue (state : State) : Value :=
  .array [fieldWordsValue state,
    fieldWordsValue (state.take (chunksPerDigest / 2)),
    chunkWordsValue (List.ofFn (digestChunks state))]

def transcriptEntryValue (entry : CoordinateTrace) : Value :=
  let scalar := entry.coefficients.map scalarOfList
  .array [.atom entry.coordinate,
    fieldWordsValue entry.beforeState,
    fieldWordsValue entry.enteredState,
    .array (entry.blockStates.map blockValue),
    chunkWordsValue entry.candidates,
    optionScalarValue scalar,
    optionRingValue (scalar.map Phi81StrongSet.embedScalar),
    fieldWordsValue entry.nextState]

def transcriptCount : Nat := 2

def transcriptValue : Value :=
  let trace := computeTranscript transcriptCount 0 initialState
  .array [fieldWordsValue initialState,
    .atom transcriptCount,
    .array (trace.1.map transcriptEntryValue),
    fieldWordsValue trace.2,
    boolValue (trace.1.all fun entry => entry.coefficients.isSome)]

def parameterValue : Value :=
  .array [.atom chunkModulus,
    .atom rejectionBucket,
    .atom alphabetSize,
    .atom coefficientCount,
    .atom chunksPerDigest,
    .atom digestRounds,
    .atom candidateBound]

def decoderCasesValue : Value :=
  .array ([0, 1, 2, 3, 4, 5, 65534, 65535].map fun value =>
    decoderCaseValue (chunkOfNat value))

/-- Schema 1 fixes the sampler constants, direct boundary cases, and one
two-coordinate concrete transcript execution. -/
def parityValue : Value :=
  .array [.atom 1,
    parameterValue,
    decoderCasesValue,
    injectedCaseValue successCandidates,
    injectedCaseValue shortfallCandidates,
    transcriptValue]

def render : String := parityValue.render

end NightstreamFPrime.Export.Stage1.PiRlcSamplerParity
