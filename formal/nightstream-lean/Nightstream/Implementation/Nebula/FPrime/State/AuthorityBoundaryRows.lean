import Nightstream.Implementation.Nebula.FPrime.State.AuthoritativeOutputBinding
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact cross-invocation authority link for Nebula V2 state outputs.

Assurance tier: implementation model and cryptographic boundary.

Owns a normalized typed state that is independent of generated column
numbers, four direct digest-equality rows at one delayed F-prime boundary,
recovery of the complete typed state or one named Poseidon2 collision, and
inductive composition over a lifetime of invocation boundaries.

Does not own placement of either local state in a recursive artifact, the
generated wrapper that selects the two boundary column sets, Poseidon2
collision resistance, NIFS extraction, or Rust conformance.

The equality rows are authority. A caller cannot replace them with an assumed
digest equality or with a self-consistent digest chain.

Emits constraints: yes, through `EqualityPins.rows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.StateAuthorityBoundaryRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.EqualityPins

abbrev Digest := Fin 4 → Nat

/-- One normalized authority-bearing V2 state. The canonicality proof is only
the domain condition needed to construct exact Poseidon2 collision events. It
does not assume digest binding or equality with another state. -/
structure Authority where
  payload : StateOutputAuthorityRows.Payload
  carryBlock : MemoryCarryParser.Block
  frameCanonical :
    ∀ value ∈ AuthoritativeStateOutputBinding.typedFrame payload carryBlock,
      value < goldilocksP

def Authority.digest (authority : Authority) : Digest :=
  AuthoritativeStateOutputBinding.typedDigest authority.payload
    authority.carryBlock

/-- Exact semantic equality recovered at a sound boundary. Proof fields in
`Authority` are intentionally not part of this predicate. -/
def Same (left right : Authority) : Prop :=
  left.payload = right.payload ∧ left.carryBlock = right.carryBlock

theorem Same.refl (authority : Authority) : Same authority authority :=
  ⟨rfl, rfl⟩

theorem Same.trans {left middle right : Authority}
    (leftMiddle : Same left middle) (middleRight : Same middle right) :
    Same left right := by
  exact ⟨leftMiddle.1.trans middleRight.1,
    leftMiddle.2.trans middleRight.2⟩

theorem digest_eq_of_same {left right : Authority}
    (same : Same left right) : left.digest = right.digest := by
  rcases same with ⟨payloadEqual, blockEqual⟩
  simp only [Authority.digest]
  rw [payloadEqual, blockEqual]

/-- The only cryptographic failures exposed by the two-stage V2 state hash. -/
inductive Failure : Prop where
  | outer (collision : StateOutputPoseidonBinding.OuterCollision)
  | inner (collision : MemoryCarryPoseidonBinding.PoseidonCollision)

/-- Exact columns selected by one generated delayed-boundary wrapper. -/
structure Layout where
  outgoingColumn : Fin 4 → Nat
  incomingColumn : Fin 4 → Nat
deriving DecidableEq, Repr

def lanes : List (Fin 4) := List.ofFn id

theorem lanes_length : lanes.length = 4 := by
  simp [lanes]

theorem lane_mem (lane : Fin 4) : lane ∈ lanes := by
  fin_cases lane <;> simp [lanes]

def Layout.pairAt (layout : Layout) (lane : Fin 4) : Nat × Nat :=
  (layout.outgoingColumn lane, layout.incomingColumn lane)

def Layout.pairs (layout : Layout) : List (Nat × Nat) :=
  lanes.map layout.pairAt

def rows (layout : Layout) : List Row :=
  EqualityPins.rows layout.pairs

theorem Layout.pairs_length (layout : Layout) : layout.pairs.length = 4 := by
  simp [Layout.pairs, lanes_length]

theorem rows_length_exact (layout : Layout) : (rows layout).length = 4 := by
  simp [rows, EqualityPins.rows, layout.pairs_length]

private theorem pairAt_mem (layout : Layout) (lane : Fin 4) :
    layout.pairAt lane ∈ layout.pairs := by
  exact List.mem_map.mpr ⟨lane, lane_mem lane, rfl⟩

/-- Both sides of the boundary rows are tied to independently normalized
typed state digests. -/
def Placed (layout : Layout) (assignment : Nat → Nat)
    (outgoing incoming : Authority) : Prop :=
  ∀ lane,
    assignment (layout.outgoingColumn lane) = outgoing.digest lane ∧
      assignment (layout.incomingColumn lane) = incoming.digest lane

/-- Four satisfying equality rows derive equality of the complete digest.
There is no digest-equality premise. -/
theorem digest_eq_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    {outgoing incoming : Authority}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment outgoing incoming)
    (holds : Satisfies (rows layout) assignment) :
    outgoing.digest = incoming.digest := by
  funext lane
  have equalColumns := EqualityPins.rows_sound canonical one holds
    (layout.pairAt lane) (pairAt_mem layout lane)
  simpa [Layout.pairAt, placed lane] using equalColumns

/-- One selected boundary row gives equality of its two concrete columns.
This theorem does not require either column to be interpreted as a typed
state. It is the narrow row fact used when another authority path first
recovers exact typed-state equality. -/
theorem columns_eq_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (lane : Fin 4) :
    assignment (layout.outgoingColumn lane) =
      assignment (layout.incomingColumn lane) := by
  exact EqualityPins.rows_sound canonical one holds
    (layout.pairAt lane) (pairAt_mem layout lane)

/-- Honest equal digests satisfy all four boundary rows when both normalized
digests are placed at the selected columns. -/
theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    {outgoing incoming : Authority}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment outgoing incoming)
    (equal : outgoing.digest = incoming.digest) :
    Satisfies (rows layout) assignment := by
  apply EqualityPins.rows_complete canonical one
  intro pair member
  rcases List.mem_map.mp member with ⟨lane, _laneMember, pairEqual⟩
  subst pair
  simp only [Layout.pairAt]
  rw [(placed lane).1, (placed lane).2, congrFun equal lane]

/-- One row-derived delayed boundary. Every equality premise is represented
by a concrete placement or by satisfaction of the four emitted rows. -/
structure Boundary (outgoing incoming : Authority) where
  layout : Layout
  assignment : Nat → Nat
  canonicalAssignment : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  placed : Placed layout assignment outgoing incoming
  satisfies : Satisfies (rows layout) assignment

namespace Boundary

/-- A concrete boundary first proves equality of the four authority-bearing
digests. This result does not use collision resistance. -/
theorem digest_eq
    {outgoing incoming : Authority}
    (boundary : Boundary outgoing incoming) :
    outgoing.digest = incoming.digest :=
  digest_eq_of_rows boundary.canonicalAssignment boundary.one
    boundary.placed boundary.satisfies

/-- A satisfying delayed boundary recovers the complete typed state and carry,
or exposes one of the two exact Poseidon2 collision events. -/
theorem sound
    {outgoing incoming : Authority}
    (boundary : Boundary outgoing incoming) :
    Same outgoing incoming ∨ Failure := by
  classical
  have digestEqual := boundary.digest_eq
  rcases
      AuthoritativeStateOutputBinding.typed_authority_eq_or_two_stage_collision
        (StateOutputAuthorityRows.fullFrame_length _ _)
        (StateOutputAuthorityRows.fullFrame_length _ _)
        outgoing.frameCanonical incoming.frameCanonical digestEqual with
    same | outerOrInner
  · exact Or.inl same
  · rcases outerOrInner with outer | inner
    · exact Or.inr (.outer outer)
    · exact Or.inr (.inner inner)

end Boundary

/-- The two authority views of one recursive invocation. The local transition
is allowed to change the state. Only consecutive output/input boundaries must
be equal. -/
structure Invocation where
  incoming : Authority
  outgoing : Authority

/-- Row-derived candidate lifetime. Each constructor owns the exact boundary
between one invocation output and the next invocation input. -/
inductive CandidateChain : Invocation → List Invocation → Prop where
  | nil (last : Invocation) : CandidateChain last []
  | cons {current next : Invocation} {rest : List Invocation}
      (boundary : Boundary current.outgoing next.incoming)
      (tail : CandidateChain next rest) :
      CandidateChain current (next :: rest)

/-- Collision-free semantic result for the same lifetime shape. -/
inductive ExactChain : Invocation → List Invocation → Prop where
  | nil (last : Invocation) : ExactChain last []
  | cons {current next : Invocation} {rest : List Invocation}
      (same : Same current.outgoing next.incoming)
      (tail : ExactChain next rest) :
      ExactChain current (next :: rest)

/-- Global delayed-state theorem. It does not assume that accepted boundaries
are exact. It derives every exact boundary from its four rows and returns the
first named two-stage collision if this derivation cannot recover the state. -/
theorem candidate_sound_or_collision
    {first : Invocation} {rest : List Invocation}
    (chain : CandidateChain first rest) :
    ExactChain first rest ∨ Failure := by
  induction chain with
  | nil last => exact Or.inl (.nil last)
  | cons boundary _ inductionHypothesis =>
      rcases boundary.sound with exactBoundary | failure
      · rcases inductionHypothesis with exactTail | failure
        · exact Or.inl (.cons exactBoundary exactTail)
        · exact Or.inr failure
      · exact Or.inr failure

end Nightstream.Implementation.Nebula.StateAuthorityBoundaryRows
