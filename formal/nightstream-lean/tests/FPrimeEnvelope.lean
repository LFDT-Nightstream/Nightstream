import Nightstream.Implementation.FPrime.Envelope

namespace NightstreamTests.FPrimeEnvelope

open Nightstream.HyperNova.Construction2
open Nightstream.Implementation.FPrime.Envelope

abbrev TestState := State Nat Unit Unit

def validInitial : TestState where
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

def validNext : TestState where
  chunkCount := 1
  stepCount := 2
  z0 := 7
  zi := 13
  initialSemanticState := 11
  semanticState := 17
  pc := 1
  accumulatorDigest := 19
  publicTrace := 13
  proof := .active () ()

def forgedInitialCounter : TestState :=
  { validInitial with chunkCount := 1 }

def forgedActiveZeroSteps : TestState :=
  { validInitial with proof := .active () () }

example : check 2 validInitial validNext = true := by decide

example : check 2 forgedInitialCounter validNext = false := by decide

example : check 2 forgedActiveZeroSteps validNext = false := by decide

end NightstreamTests.FPrimeEnvelope
