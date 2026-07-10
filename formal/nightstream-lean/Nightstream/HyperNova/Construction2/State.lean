/-!
HyperNova Construction-2 carrier state.

This file owns the state coordinates needed to state the base/recursive split.
It mirrors the authority-bearing core of Rust `construction2::State`. Nebula is
deliberately excluded because the first theorem does not inspect that lane.
It does not claim Rust representation or transition conformance.
-/

namespace Nightstream.HyperNova.Construction2

universe uDigest uRunning uFresh

/-- The fold pair is absent only before the first Construction-2 step. -/
inductive ProofState (Running : Type uRunning) (Fresh : Type uFresh) where
  | initial
  | active (running : Running) (latest : Fresh)
deriving Repr

/-- The current direct-F' state coordinates used by the active Rust path. -/
structure State
    (Digest : Type uDigest)
    (Running : Type uRunning)
    (Fresh : Type uFresh) where
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
deriving Repr

end Nightstream.HyperNova.Construction2
