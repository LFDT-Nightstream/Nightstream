/-!
HyperNova Construction-2 carrier state.

This file owns the state coordinates needed to state the base/recursive split.
It mirrors the authority-bearing core of Rust `construction2::State`, including
the optional Nebula memory lane now absorbed by `state_x_out`. It does not claim
Rust representation or transition conformance.
-/

namespace Nightstream.HyperNova.Construction2

universe uDigest uRunning uFresh

/-- The fold pair is absent only before the first Construction-2 step.
`latest` is the fresh batch as an actual list so its cardinality is
expressible — Rust's `advance_state` derives `fresh_count` from
`next_latest_claims.len()` and rejects the empty batch (`Error::EmptyStep`). -/
inductive ProofState (Running : Type uRunning) (Fresh : Type uFresh) where
  | initial
  | active (running : Running) (latest : List Fresh)
deriving Repr, DecidableEq

/-- The current direct-F' state coordinates used by the active Rust path. -/
structure State
    (Digest : Type uDigest)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Nebula : Type := Unit) where
  chunkCount : Nat
  stepCount : Nat
  z0 : Digest
  zi : Digest
  initialSemanticState : Digest
  semanticState : Digest
  pc : Nat
  accumulatorDigest : Digest
  publicTrace : Digest
  proof : ProofState Running Fresh
  nebula : Option Nebula := none
deriving Repr, DecidableEq

end Nightstream.HyperNova.Construction2
