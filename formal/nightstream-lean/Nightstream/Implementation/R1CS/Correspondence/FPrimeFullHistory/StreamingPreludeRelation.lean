import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Contract: typed meaning of the first phased F-prime work item.

Owns the canonical ten-field prior-state replay start, the zero incoming local
digest, and the exact domain-separated digest required as the Prelude output.
It does not own generated rows, a concrete Poseidon2 trace, replay chunks, or
terminal acceptance.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeRelation

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram

abbrev Field := Fin Nightstream.Implementation.R1CS.goldilocksP
abbrev Digest := Fin 4 -> Field

/-- Verifier-fixed local state before the first work item. -/
def zeroDigest : Digest := fun _ => 0

/-- Exact typed fields opened by the first prior-state replay phase: eight
Poseidon2 lanes, the absorb cursor, and the frame cursor. -/
def initialReplayFields : List Nat :=
  List.replicate 8 0 ++ [0, 0]

theorem initialReplayFields_exact :
    initialReplayFields = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] := by
  rfl

theorem initialReplayFields_length : initialReplayFields.length = 10 := by
  simp [initialReplayFields]

/-- Concrete profiles instantiate this function with the protocol-bound
Poseidon2 local-state digest. The preimage remains explicit and authoritative. -/
structure Semantics where
  stateDigest : List Nat → Digest

/-- Exact local transition of the Prelude arm. -/
def Holds (semantics : Semantics) (before after : Digest) : Prop :=
  before = zeroDigest ∧ after = semantics.stateDigest initialReplayFields

theorem holds_iff
    (semantics : Semantics) (before after : Digest) :
    Holds semantics before after ↔
      before = zeroDigest ∧
        after = semantics.stateDigest initialReplayFields := by
  rfl

theorem before_exact
    {semantics : Semantics} {before after : Digest}
    (holds : Holds semantics before after) :
    before = zeroDigest :=
  holds.1

theorem after_exact
    {semantics : Semantics} {before after : Digest}
    (holds : Holds semantics before after) :
    after = semantics.stateDigest initialReplayFields :=
  holds.2

/-- Schedule-level semantic adapter. It accepts only the exact first work
item and then applies `Holds` to the shared local digests. -/
def AtWorkItem
    (semantics : Semantics) (item : WorkItem)
    (before after : Digest) : Prop :=
  item = ProductionStreamingFPrimeProgram.singleton Phase.prelude ∧
    Holds semantics before after

theorem workItem_exact
    {semantics : Semantics} {item : WorkItem} {before after : Digest}
    (accepted : AtWorkItem semantics item before after) :
    item.phase = .prelude ∧ item.index = 0 := by
  rw [accepted.1]
  exact ⟨rfl, rfl⟩

theorem atWorkItem_implies_holds
    {semantics : Semantics} {item : WorkItem} {before after : Digest}
    (accepted : AtWorkItem semantics item before after) :
    Holds semantics before after :=
  accepted.2

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeRelation
