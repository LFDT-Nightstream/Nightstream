import Nightstream.Implementation.Nebula.FPrime.State.AuthorityBoundaryRows

/-! Focused regressions for the exact cross-invocation V2 state boundary. -/

namespace NightstreamTests.NebulaStateAuthorityBoundaryRows

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.StateAuthorityBoundaryRows

example (layout : Layout) : (rows layout).length = 4 :=
  rows_length_exact layout

example {outgoing incoming : Authority}
    (boundary : Boundary outgoing incoming) :
    Same outgoing incoming ∨ Failure :=
  boundary.sound

example {outgoing incoming : Authority}
    (boundary : Boundary outgoing incoming) :
    outgoing.digest = incoming.digest :=
  boundary.digest_eq

example {first : Invocation} {rest : List Invocation}
    (chain : CandidateChain first rest) :
    ExactChain first rest ∨ Failure :=
  candidate_sound_or_collision chain

namespace ConstantDigestCountermodel

/-- Digest equality alone cannot be an authority theorem. A constant digest
accepts two different states. The production theorem needs both the exact
four-lane boundary rows and the named collision branch. -/
inductive ToyState where
  | left
  | right
deriving DecidableEq

def digest (_state : ToyState) : Nat := 0

theorem distinct_states_have_equal_digest :
    ToyState.left ≠ ToyState.right ∧
      digest ToyState.left = digest ToyState.right := by
  decide

end ConstantDigestCountermodel

namespace MissingSemanticLinkCountermodel

/-- Minimal selected-phase envelope. The Boolean fields stand for exact typed
phase fields, not for a digest supplied by the prover. -/
structure PhaseEnvelope where
  localState : Bool
  delayedPayload : Bool
deriving DecidableEq

def semanticDigest (phase : PhaseEnvelope) : Bool :=
  xor phase.localState phase.delayedPayload

structure Candidate where
  phase : PhaseEnvelope
  outerSemantic : Bool
deriving DecidableEq

def Linked (candidate : Candidate) : Prop :=
  candidate.outerSemantic = semanticDigest candidate.phase

def honest : Candidate where
  phase := { localState := false, delayedPayload := false }
  outerSemantic := false

def omittedLink : Candidate where
  phase := honest.phase
  outerSemantic := true

/-- If the semantic-link family is absent, all selected-phase fields can stay
fixed while the outer semantic lane changes. The exact link rejects this
candidate without a cryptographic assumption. -/
theorem retained_phase_allows_wrong_outer_semantic :
    omittedLink.phase = honest.phase ∧
      ¬ Linked omittedLink := by
  simp [omittedLink, honest, Linked, semanticDigest]

end MissingSemanticLinkCountermodel

end NightstreamTests.NebulaStateAuthorityBoundaryRows
