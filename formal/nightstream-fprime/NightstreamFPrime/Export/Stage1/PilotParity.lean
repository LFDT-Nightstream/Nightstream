import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.PilotData
import NightstreamFPrime.Export.Stage1.PiCCSNonzero

/-!
Owns one standalone nonzero pilot result for Lean--Rust conformance. It reuses
the PiCCS nonzero running-state fixture, but gives the output hash a distinct
next-state preimage. The result contains both hashes and the complete
58-field public vector in production order.
-/

namespace NightstreamFPrime.Export.Stage1.PilotParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1.PiCCSNonzero
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

def outputCurrent : AppState :=
  [field 401, field 402, field 403, field 404]

def outputPreimage : HashPreimage
    (logicalWidth := fixtureLogicalWidth)
    (publicFits := fixturePublicFits) :=
  { statePreimage (stateVerifierKey ()) with
    iteration := 8
    current := outputCurrent }

def outputPreimageWords : List F :=
  serializePreimage (publicFits := fixturePublicFits)
    outputPreimage

def priorDigest : Digest := stateDigest ()

def outputDigest : Digest :=
  stateHash (publicFits := fixturePublicFits) outputPreimage

def publicValues : List F := statePublicInputWords () ++ outputDigest

def boolValue (value : Bool) : Value :=
  .atom (if value then 1 else 0)

def fieldValue (value : F) : Value := .atom value.val

def fieldWordsValue (values : List F) : Value :=
  .array (values.map fieldValue)

def segmentValue (role start length : Nat) : Value :=
  .array [.atom role, .atom start, .atom length]

/-- Encode the public instance from one already-computed state digest. -/
def publicInputWordsForDigest (digest : Digest) : List F :=
  List.ofFn fun column =>
    encHash (publicFits := VerifierContext.candidatePublicFits) digest column

/-- Caller-owned pilot inputs: prior preimage, prior public instance, output
preimage, and claimed output digest. -/
def inputValueFrom (priorPreimage priorPublicInput outputPreimage : List F)
    (claimedOutputDigest : Digest) : Value :=
  .array [fieldWordsValue priorPreimage,
    fieldWordsValue priorPublicInput,
    fieldWordsValue outputPreimage,
    fieldWordsValue claimedOutputDigest]

/-- Lean-computed pilot result: both digests, the complete public vector,
relative public-segment map, and fixture assurance flags. -/
def resultValueFrom (computedPriorDigest computedOutputDigest : Digest)
    (priorPublicInput : List F) : Value :=
  let computedPublicValues := priorPublicInput ++ computedOutputDigest
  .array [fieldWordsValue computedPriorDigest,
    fieldWordsValue computedOutputDigest,
    fieldWordsValue computedPublicValues,
    .array [segmentValue PilotData.Role.priorPublicInput 0
        PilotValues.priorPublicInputWords,
      segmentValue PilotData.Role.outputDigest
        PilotValues.priorPublicInputWords PilotValues.digestWords],
    .array [boolValue (decide
        (computedPriorDigest.length = PilotValues.digestWords)),
      boolValue (decide
        (computedOutputDigest.length = PilotValues.digestWords)),
      boolValue (decide (computedPriorDigest ≠ computedOutputDigest)),
      boolValue (decide
        (computedPublicValues.length = PilotValues.publicColumnCount))]]

def inputValue : Value :=
  inputValueFrom (statePreimageWords ()) (statePublicInputWords ())
    outputPreimageWords outputDigest

def resultValue : Value :=
  resultValueFrom priorDigest outputDigest (statePublicInputWords ())

def parityValueFrom (priorPreimage priorPublicInput outputPreimage : List F)
    (computedPriorDigest computedOutputDigest : Digest) : Value :=
  .array [.atom 1,
    inputValueFrom priorPreimage priorPublicInput outputPreimage
      computedOutputDigest,
    resultValueFrom computedPriorDigest computedOutputDigest priorPublicInput]

/-- Schema 1 is the first standalone complete nonzero pilot parity object. -/
def parityValue : Value :=
  parityValueFrom (statePreimageWords ()) (statePublicInputWords ())
    outputPreimageWords priorDigest outputDigest

private abbrev PreparedTask (Alpha : Type) := Task (Except IO.Error Alpha)

private def prepare {Alpha : Type} (build : Unit → Alpha) : IO Alpha := do
  pure (build ())

private def prepared {Alpha : Type} (task : PreparedTask Alpha) : IO Alpha :=
  match task.get with
  | .ok value => pure value
  | .error error => throw error

/-- Compute independent preimages and hashes on native tasks, then serialize
the canonical schema once from the shared digest values. -/
def parityValueIO : IO Value := do
  let priorPreimageTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepare fun _ => statePreimageWords ())
  let outputPreimageTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepare fun _ => outputPreimageWords)
  let priorDigestTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepare fun _ => priorDigest)
  let outputDigestTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepare fun _ => outputDigest)
  let priorPreimage ← prepared priorPreimageTask
  let outputPreimage ← prepared outputPreimageTask
  let computedPriorDigest ← prepared priorDigestTask
  let computedOutputDigest ← prepared outputDigestTask
  let priorPublicInput := publicInputWordsForDigest computedPriorDigest
  pure <| parityValueFrom priorPreimage priorPublicInput outputPreimage
    computedPriorDigest computedOutputDigest

def render : String := parityValue.render

end NightstreamFPrime.Export.Stage1.PilotParity
