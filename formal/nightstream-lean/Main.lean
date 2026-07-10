import Nightstream

/-!
`lake exe check` — the executable assurance gate.

Every line of output is a computed result: the envelope probes actually run
`Envelope.check`, and the drift gate actually reads the mapped Rust sources
and verifies the symbol anchors the model claims parity with. Any failure
exits nonzero. Run from `formal/nightstream-lean` (spec §13).
-/

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

/-- Envelope probes: expected boolean per case, all computed here. -/
def envelopeProbes : List (String × Bool × Bool) :=
  [ ("envelope_honest_transition", check 2 probeInitial probeNext, true)
  , ("envelope_rejects_empty_step", check 0 probeInitial probeEmptyStep, false)
  , ("envelope_rejects_count_forgery", check 1 probeInitial probeNext, false)
  ]

/--
Symbol anchors in the mapped Rust sources. If an anchor disappears, the
mapped surface has drifted and every parity claim resting on it is stale —
the gate fails instead of printing success. Content hashes over generated
artifacts land with the tracer bullet; these anchors guard the symbols the
current model claims parity with.
-/
def rustAnchors : List (String × String) :=
  [ ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "Err(Error::EmptyStep)")
  , ("../../crates/neo-fold-clean/src/paper/construction2/transition.rs",
     "fn state_base_case_check")
  , ("../../crates/neo-fold-clean/src/paper/construction2/transition.rs",
     "fn advance_state")
  , ("../../crates/neo-fold-clean/src/paper/construction2/state.rs",
     "pub struct State")
  ]

def containsSubstr (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

def main : IO UInt32 := do
  let mut ok := true

  for (name, got, expected) in envelopeProbes do
    let pass := got == expected
    IO.println s!"{name}={pass}"
    unless pass do ok := false

  for (path, needle) in rustAnchors do
    let pass ← do
      try
        let content ← IO.FS.readFile ⟨path⟩
        pure (containsSubstr content needle)
      catch _ =>
        pure false
    IO.println s!"rust_anchor {path} :: {needle} => {pass}"
    unless pass do ok := false

  IO.println "rust_conformance=pending (tracer-bullet artifact gate not yet built)"
  if ok then
    IO.println "check=pass"
    return 0
  else
    IO.println "check=FAIL"
    return 1
