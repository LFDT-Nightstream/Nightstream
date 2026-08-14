import Nightstream.Protocol.Nebula.Segment

/-!
Contract: deterministic composition of closed V2 memory segments.

Assurance tier: model-level.

Owns exact boundary-snapshot and global-timestamp threading. It proves that a
chain of valid closed segments is one operational memory execution.

Does not own the cryptographic reduction from equal roots and commitments to
equal snapshots.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula

open Memory

namespace Memory.Executes

theorem append
    {initial middle final : Multiset MemTuple}
    {timestampIn timestampMiddle timestampOut : Nat}
    {first second : List Access}
    (head : Executes initial timestampIn first middle timestampMiddle)
    (tail : Executes middle timestampMiddle second final timestampOut) :
    Executes initial timestampIn (first ++ second) final timestampOut := by
  induction head with
  | nil =>
      simpa using tail
  | cons applies rest inductionHypothesis =>
      exact .cons applies (inductionHypothesis tail)

end Memory.Executes

/-- A valid chain shares the exact boundary snapshot and timestamp between
adjacent segments. Each list element is the access trace of one segment. -/
inductive ValidChain :
    Snapshot → Nat → List (List Access) → Snapshot → Nat → Prop
  | nil (snapshot : Snapshot) (timestamp : Nat) :
      ValidChain snapshot timestamp [] snapshot timestamp
  | cons
      {initial middle final : Snapshot}
      {timestampIn timestampMiddle timestampOut : Nat}
      {accesses : List Access}
      {rest : List (List Access)}
      (head : ValidSegment initial middle timestampIn accesses timestampMiddle)
      (tail : ValidChain middle timestampMiddle rest final timestampOut) :
      ValidChain initial timestampIn (accesses :: rest) final timestampOut

namespace ValidChain

theorem executes
    {initial final : Snapshot}
    {timestampIn timestampOut : Nat}
    {segments : List (List Access)}
    (chain : ValidChain initial timestampIn segments final timestampOut) :
    Memory.Executes initial.tuples timestampIn segments.flatten
      final.tuples timestampOut := by
  induction chain with
  | nil snapshot timestamp =>
      exact .nil snapshot.tuples timestamp
  | cons head _ inductionHypothesis =>
      exact head.executes.append inductionHypothesis

theorem timestampOut_eq
    {initial final : Snapshot}
    {timestampIn timestampOut : Nat}
    {segments : List (List Access)}
    (chain : ValidChain initial timestampIn segments final timestampOut) :
    timestampOut = timestampIn + segments.flatten.length := by
  induction chain with
  | nil => simp
  | @cons _ _ _ timestampIn timestampMiddle timestampOut accesses rest
      head _ inductionHypothesis =>
      rw [head.timestampOut_eq] at inductionHypothesis
      rw [inductionHypothesis]
      simp [Nat.add_assoc]

end ValidChain

end Nightstream.Protocol.Nebula
