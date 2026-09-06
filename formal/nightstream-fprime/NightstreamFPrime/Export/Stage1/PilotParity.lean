import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.PilotData
import NightstreamFPrime.Export.Stage1.PiCCSNonzero
import NightstreamFPrime.Lifecycle.PilotZeroRunning

/-!
Owns one standalone nonzero pilot result for Lean--Rust conformance. Its
running claims have zero openings under the canonical key. The output hash
has a distinct next-state preimage. The result contains both hashes and the
complete 274-field public vector in production order.
-/

namespace NightstreamFPrime.Export.Stage1.PilotParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1.PiCCSNonzero
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

def outputCurrent : AppState :=
  [field 401, field 402, field 403, field 404]

def priorPreimage (vk : KeyDigest) : HashPreimage
    (logicalWidth := fixtureLogicalWidth)
    (publicFits := fixturePublicFits) :=
  { statePreimage vk with
    running := fun _ => defaultRunning }

def priorPreimageWords (vk : KeyDigest) : List F :=
  serializePreimage (publicFits := fixturePublicFits) (priorPreimage vk)

def outputPreimage (vk : KeyDigest) : HashPreimage
    (logicalWidth := fixtureLogicalWidth)
    (publicFits := fixturePublicFits) :=
  { priorPreimage vk with
    iteration := 8
    current := outputCurrent }

def outputPreimageWords (vk : KeyDigest) : List F :=
  serializePreimage (publicFits := fixturePublicFits)
    (outputPreimage vk)

def priorDigest (vk : KeyDigest) : Digest :=
  stateHash (publicFits := fixturePublicFits) (priorPreimage vk)

def outputDigest (vk : KeyDigest) : Digest :=
  stateHash (publicFits := fixturePublicFits) (outputPreimage vk)

/-- Encode the public instance from one already-computed state digest. -/
def publicInputWordsForDigest (digest : Digest) : List F :=
  List.ofFn fun column =>
    encHash (publicFits := fixturePublicFits) digest column

def priorPublicInputWords (vk : KeyDigest) : List F :=
  publicInputWordsForDigest (priorDigest vk)

def publicValues (vk : KeyDigest) : List F :=
  priorPublicInputWords vk ++ outputDigest vk

def boolValue (value : Bool) : Value :=
  .atom (if value then 1 else 0)

def fieldValue (value : F) : Value := .atom value.val

def fieldWordsValue (values : List F) : Value :=
  .array (values.map fieldValue)

def segmentValue (role start length : Nat) : Value :=
  .array [.atom role, .atom start, .atom length]

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

def inputValue (vk : KeyDigest) : Value :=
  inputValueFrom (priorPreimageWords vk) (priorPublicInputWords vk)
    (outputPreimageWords vk) (outputDigest vk)

def resultValue (vk : KeyDigest) : Value :=
  resultValueFrom (priorDigest vk) (outputDigest vk) (priorPublicInputWords vk)

def parityValueFrom (priorPreimage priorPublicInput outputPreimage : List F)
    (computedPriorDigest computedOutputDigest : Digest) : Value :=
  .array [.atom 1,
    inputValueFrom priorPreimage priorPublicInput outputPreimage
      computedOutputDigest,
    resultValueFrom computedPriorDigest computedOutputDigest priorPublicInput]

/-- Schema 1 is the first standalone complete nonzero pilot parity object. -/
def parityValue (vk : KeyDigest) : Value :=
  parityValueFrom (priorPreimageWords vk) (priorPublicInputWords vk)
    (outputPreimageWords vk) (priorDigest vk) (outputDigest vk)

private abbrev PreparedTask (Alpha : Type) := Task (Except IO.Error Alpha)

private def prepare {Alpha : Type} (build : Unit → Alpha) : IO Alpha := do
  pure (build ())

private def prepared {Alpha : Type} (task : PreparedTask Alpha) : IO Alpha :=
  match task.get with
  | .ok value => pure value
  | .error error => throw error

/-- Compute independent preimages and hashes on native tasks, then serialize
the canonical schema once from the shared digest values. The explicit fixture
context is checked against the canonical package by the Rust parity gate. -/
def parityValueIO (vk : KeyDigest) : IO Value := do
  let priorPreimageTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepare fun _ => priorPreimageWords vk)
  let outputPreimageTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepare fun _ => outputPreimageWords vk)
  let priorPreimage ← prepared priorPreimageTask
  let outputPreimage ← prepared outputPreimageTask
  let priorDigestTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepare fun _ => Poseidon2.hash priorPreimage)
  let outputDigestTask ← IO.asTask (prio := Task.Priority.dedicated)
    (prepare fun _ => Poseidon2.hash outputPreimage)
  let computedPriorDigest ← prepared priorDigestTask
  let computedOutputDigest ← prepared outputDigestTask
  let priorPublicInput := publicInputWordsForDigest computedPriorDigest
  pure <| parityValueFrom priorPreimage priorPublicInput outputPreimage
    computedPriorDigest computedOutputDigest

def render (vk : KeyDigest) : String := (parityValue vk).render

end NightstreamFPrime.Export.Stage1.PilotParity
