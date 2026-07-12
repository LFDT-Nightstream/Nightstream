import Nightstream.Checks.Common
import Nightstream.Implementation.FPrime.Envelope

namespace Nightstream.Checks.Envelope

open Nightstream.HyperNova.Construction2
open Nightstream.Implementation.FPrime.Envelope

abbrev ProbeState := State Nat Unit Unit

def probeInitial : ProbeState where
  chunkCount := 0
  stepCount := 0
  z0 := 7
  zi := 7
  initialSemanticState := 11
  semanticState := 11
  pc := 1
  accumulatorDigest := 0
  publicTrace := 0
  proof := .initial

def probeNext : ProbeState where
  chunkCount := 1
  stepCount := 2
  z0 := 7
  zi := 13
  initialSemanticState := 11
  semanticState := 17
  pc := 1
  accumulatorDigest := 19
  publicTrace := 13
  proof := .active () [(), ()]

/-- The empty-step forgery Rust rejects with `Error::EmptyStep`. -/
def probeEmptyStep : ProbeState :=
  { probeNext with stepCount := 0, proof := .active () [] }

/-- Envelope probes: expected boolean per case, evaluated sequentially. -/
def probes : List Nightstream.Checks.Probe :=
  [ ⟨"envelope_honest_transition", fun _ => check 2 probeInitial probeNext, true⟩
  , ⟨"envelope_rejects_empty_step", fun _ => check 0 probeInitial probeEmptyStep, false⟩
  , ⟨"envelope_rejects_count_forgery", fun _ => check 1 probeInitial probeNext, false⟩
  ]

def run : IO Bool :=
  Nightstream.Checks.runProbes probes

end Nightstream.Checks.Envelope
