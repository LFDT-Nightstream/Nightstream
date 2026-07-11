import Nightstream.Implementation.FPrime.CounterRefinement

/-!
Counter-refinement witnesses: a two-invocation schedule with batch sizes 2 and
3, plus forged terminal counters and a zero-sized schedule entry.
-/

namespace NightstreamTests.CounterRefinement

open Nightstream.HyperNova.Construction2
open Nightstream.Implementation.FPrime.Envelope
open Nightstream.Implementation.FPrime.CounterRefinement

abbrev TestState := State Nat Unit Unit

def initial : TestState where
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

def afterTwo : TestState where
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

def afterFive : TestState where
  chunkCount := 2
  stepCount := 5
  z0 := 7
  zi := 23
  initialSemanticState := 11
  semanticState := 29
  pc := 1
  accumulatorDigest := 31
  publicTrace := 23
  proof := .active () [(), (), ()]

def honestTrace : TraceCoherent initial [2, 3] afterFive :=
  .snoc
    (prior := afterTwo)
    (.snoc (next := afterTwo) .nil (by simp [AdvanceCoherent, initial, afterTwo]))
    (by simp [AdvanceCoherent, afterTwo, afterFive])

example : CounterRefines [2, 3] afterFive :=
  counter_refinement (by simp [InputCoherent, initial]) rfl honestTrace

-- A self-consistent trace cannot claim an extra F' invocation.
example : ¬ CounterRefines [2, 3] { afterFive with chunkCount := 3 } := by
  simp [CounterRefines, afterFive]

-- The instance count is the sum of batch cardinalities, not the chunk count.
example : ¬ CounterRefines [2, 3] { afterFive with stepCount := 4 } := by
  simp [CounterRefines, afterFive]

-- Zero-sized entries are excluded even when the arithmetic total matches.
example : ¬ CounterRefines [2, 0, 3] { afterFive with chunkCount := 3 } := by
  simp [CounterRefines, ValidSchedule, afterFive]

end NightstreamTests.CounterRefinement
