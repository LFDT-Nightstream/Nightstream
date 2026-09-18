import Nightstream.Implementation.FPrime.Envelope

/-!
Envelope witnesses: one honest transition plus a negative sweep. Every
rejected case mutates exactly one authority-bearing coordinate of the honest
pair, so each `false` pins one specific equation of `Holds`.

`emptyFreshBatch` is the permanent regression for the model/Rust mismatch
found in review: Rust rejects an empty fresh batch (`Error::EmptyStep` in
`native.rs`), and the model must too.
-/

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
  proof := .active () [(), ()]

-- Honest transition: two fresh instances installed, counters advance.
example : check 2 validInitial validNext = true := by decide

-- ── Forged input states ─────────────────────────────────────────────────

def forgedInitialCounter : TestState :=
  { validInitial with chunkCount := 1 }

def forgedActiveZeroSteps : TestState :=
  { validInitial with proof := .active () [()] }

example : check 2 forgedInitialCounter validNext = false := by decide

example : check 2 forgedActiveZeroSteps validNext = false := by decide

-- ── Empty-step regression (review Finding 6) ────────────────────────────
-- Rust: `if next_latest_claims.is_empty() { return Err(Error::EmptyStep) }`.
-- The declared count and the installed batch are both empty and the step
-- counter does not move — the model must reject, as Rust does.

def emptyFreshBatch : TestState :=
  { validNext with stepCount := 0, proof := .active () [] }

example : check 0 validInitial emptyFreshBatch = false := by decide

-- A nonzero declared count with an empty installed batch must also fail:
-- the count is derived from the batch in Rust, so the divergence is a forgery.
example : check 2 validInitial { validNext with proof := .active () [] } = false := by
  decide

-- Declared count differing from the installed batch cardinality.
example : check 1 validInitial validNext = false := by decide

-- ── Forged successor states ─────────────────────────────────────────────

-- Chunk counter must advance by exactly one.
example : check 2 validInitial { validNext with chunkCount := 2 } = false := by decide

-- Step counter must advance by exactly the fresh count.
example : check 2 validInitial { validNext with stepCount := 3 } = false := by decide

-- The initial boundary digest is immutable.
example : check 2 validInitial { validNext with z0 := 8 } = false := by decide

-- The program counter is fixed.
example : check 2 validInitial { validNext with pc := 2 } = false := by decide

-- The initial semantic state is immutable.
example : check 2 validInitial { validNext with initialSemanticState := 12 } = false := by
  decide

-- The public trace must copy the new boundary digest.
example : check 2 validInitial { validNext with publicTrace := 14 } = false := by decide

-- A successor may never present the initial fold tag.
example : check 2 validInitial { validNext with proof := .initial } = false := by decide

end NightstreamTests.FPrimeEnvelope
