import NightstreamFPrime.Export.Codec
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
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) where
  verifierKeys := fun _ => stateVerifierKey
  iteration := 8
  z0 := stateZ0
  current := outputCurrent
  running := fun _ => running
  pc := 1

def outputPreimageWords : List F :=
  serializePreimage (publicFits := Data.publicFits) outputPreimage

def priorDigest : Digest := stateDigest

def outputDigest : Digest :=
  stateHash (publicFits := Data.publicFits) outputPreimage

def publicValues : List F := statePublicInputWords ++ outputDigest

def boolValue (value : Bool) : Value :=
  .atom (if value then 1 else 0)

def fieldValue (value : F) : Value := .atom value.val

def fieldWordsValue (values : List F) : Value :=
  .array (values.map fieldValue)

def segmentValue (role start length : Nat) : Value :=
  .array [.atom role, .atom start, .atom length]

/-- Caller-owned pilot inputs: prior preimage, prior public instance, output
preimage, and claimed output digest. -/
def inputValue : Value :=
  .array [fieldWordsValue statePreimageWords,
    fieldWordsValue statePublicInputWords,
    fieldWordsValue outputPreimageWords,
    fieldWordsValue outputDigest]

/-- Lean-computed pilot result: both digests, the complete public vector,
relative public-segment map, and fixture assurance flags. -/
def resultValue : Value :=
  .array [fieldWordsValue priorDigest,
    fieldWordsValue outputDigest,
    fieldWordsValue publicValues,
    .array [segmentValue PilotData.Role.priorPublicInput 0
        PilotValues.priorPublicInputWords,
      segmentValue PilotData.Role.outputDigest
        PilotValues.priorPublicInputWords PilotValues.digestWords],
    .array [boolValue (decide (priorDigest.length = PilotValues.digestWords)),
      boolValue (decide (outputDigest.length = PilotValues.digestWords)),
      boolValue (decide (priorDigest ≠ outputDigest)),
      boolValue (decide (publicValues.length = PilotValues.publicColumnCount))]]

/-- Schema 1 is the first standalone complete nonzero pilot parity object. -/
def parityValue : Value :=
  .array [.atom 1, inputValue, resultValue]

def render : String := parityValue.render

end NightstreamFPrime.Export.Stage1.PilotParity
