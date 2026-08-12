import Nightstream.Protocol.NebulaV2.GlobalFPrime

/-!
Contract: exact base, recursive-continuation, and terminal invocation data
flow for factor-one Nebula V2.

Assurance tier: independent protocol model.

Owns the closed intermediate carry at a segment boundary, immediate reopening
inside the same nonterminal recursive invocation, one accepted full claim per
post-base invocation, and the final no-reopen terminal rule.

Does not own a concrete challenge oracle, NIFS extraction, generated branch
selectors, recursive-size closure, or terminal cryptography.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.AugmentedLifecycle

open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.FullClaim
open Nightstream.Protocol.NebulaV2.GlobalFPrime
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.ProductState

/-- A nonterminal recursive invocation always leaves an active carry. An
interior step copies its active intermediate carry. A closing step must reopen
the next segment from its exact closed intermediate carry. -/
inductive Continues
    {ChallengeField : Type} [One ChallengeField]
    {Digest : Type}
    (derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField)
    (headers : ChainHeaders Digest) :
    Carry Digest (Challenges ChallengeField) (State ChallengeField) →
      Carry Digest (Challenges ChallengeField) (State ChallengeField) → Prop
  | interior
      (active : ActiveCarry Digest (Challenges ChallengeField)
        (State ChallengeField)) :
      Continues derive headers (.active active) (.active active)
  | boundary
      (closed : ClosedCarry Digest)
      (precommit : Roots Digest)
      (activeAccessCount : Nat)
      (canOpen : closed.CanOpen)
      (activeCountInRange : activeAccessCount < operationCountLimit)
      (endTimestampInRange :
        closed.globalTimestamp + activeAccessCount < timestampLimit) :
      Continues derive headers (.closed closed)
        (openSegment derive headers precommit activeAccessCount closed canOpen
          activeCountInRange endTimestampInRange)

namespace Continues

/-- Change the lifetime challenge derivation only when the old and new
derivations agree at the closed boundary that this continuation actually
opens. Interior continuations do not derive a challenge. -/
theorem changeDeriveAtBoundary
    {ChallengeField : Type} [One ChallengeField]
    {Digest : Type}
    {deriveLeft deriveRight : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {intermediate outgoing :
      Carry Digest (Challenges ChallengeField) (State ChallengeField)}
    (continuation : Continues deriveLeft headers intermediate outgoing)
    (sameAtClosed : forall closed,
      intermediate = .closed closed ->
      forall precommit activeAccessCount,
        deriveLeft closed precommit activeAccessCount =
          deriveRight closed precommit activeAccessCount) :
    Continues deriveRight headers intermediate outgoing := by
  cases continuation with
  | interior active =>
      exact Continues.interior active
  | boundary closed precommit activeAccessCount canOpen activeRange endRange =>
      have challengeExact :=
        sameAtClosed closed rfl precommit activeAccessCount
      simpa [openSegment, challengeExact] using
        (Continues.boundary
          (derive := deriveRight) (headers := headers) closed precommit
          activeAccessCount canOpen activeRange endRange)

/-- A nonterminal continuation cannot expose a closed outgoing carry. -/
theorem outgoing_active
    {ChallengeField : Type} [One ChallengeField]
    {Digest : Type}
    {derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {intermediate outgoing :
      Carry Digest (Challenges ChallengeField) (State ChallengeField)}
    (continuation : Continues derive headers intermediate outgoing) :
    ∃ active, outgoing = .active active := by
  cases continuation with
  | interior active => exact ⟨active, rfl⟩
  | boundary closed precommit activeAccessCount canOpen activeRange endRange =>
      refine ⟨
        { segmentIndex := closed.segmentIndex
          stepIndex := ⟨0, by decide⟩
          globalTimestamp := closed.globalTimestamp
          segmentStartTimestamp := closed.globalTimestamp
          segmentActiveAccessCount := activeAccessCount
          segmentEndTimestamp := closed.globalTimestamp + activeAccessCount
          challenge := derive closed precommit activeAccessCount
          products := ProductState.one
          dPre := precommit
          dSeen := headers.roots
          memoryRoot := closed.memoryRoot }, ?_⟩
      rfl

/-- A closed intermediate carry can continue only through the boundary-open
constructor. It cannot be copied as a closed outgoing carry. -/
theorem closed_intermediate_reopens
    {ChallengeField : Type} [One ChallengeField]
    {Digest : Type}
    {derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {closed : ClosedCarry Digest}
    {outgoing :
      Carry Digest (Challenges ChallengeField) (State ChallengeField)}
    (continuation : Continues derive headers (.closed closed) outgoing) :
    ∃ precommit activeAccessCount canOpen activeCountInRange
        endTimestampInRange,
      outgoing = openSegment derive headers precommit activeAccessCount closed
        canOpen activeCountInRange endTimestampInRange := by
  cases continuation with
  | boundary _ precommit activeAccessCount canOpen activeRange endRange =>
      exact ⟨precommit, activeAccessCount, canOpen, activeRange, endRange, rfl⟩

end Continues

/-- Exactly one verified complete claim is consumed in each invocation after
base. Every nonfinal invocation then applies `Continues`. The final invocation
must consume its claim directly into a closed carry and performs no reopen. -/
inductive DelayedRun
    {ChallengeField : Type} [Field ChallengeField]
    {schema : FullClaim.Schema} {Digest : Type}
    (verify : V2Verifier schema Digest ChallengeField)
    (derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField)
    (headers : ChainHeaders Digest) :
    Carry Digest (Challenges ChallengeField) (State ChallengeField) →
      List (Receipt schema Digest ChallengeField verify) →
      ClosedCarry Digest → Prop
  | terminal
      {before : Carry Digest (Challenges ChallengeField)
        (State ChallengeField)}
      (receipt : Receipt schema Digest ChallengeField verify)
      (final : ClosedCarry Digest)
      (consumes : Consumes ProductState.Balanced before
        receipt.claim.memory (.closed final)) :
      DelayedRun verify derive headers before [receipt] final
  | recursive
      {before intermediate outgoing :
        Carry Digest (Challenges ChallengeField) (State ChallengeField)}
      {tail : List (Receipt schema Digest ChallengeField verify)}
      {final : ClosedCarry Digest}
      (receipt : Receipt schema Digest ChallengeField verify)
      (consumes : Consumes ProductState.Balanced before
        receipt.claim.memory intermediate)
      (continues : Continues derive headers intermediate outgoing)
      (rest : DelayedRun verify derive headers outgoing tail final) :
      DelayedRun verify derive headers before (receipt :: tail) final

namespace DelayedRun

theorem claims_nonempty
    {ChallengeField : Type} [Field ChallengeField]
    {schema : FullClaim.Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : Carry Digest (Challenges ChallengeField) (State ChallengeField)}
    {claims : List (Receipt schema Digest ChallengeField verify)}
    {final : ClosedCarry Digest}
    (run : DelayedRun verify derive headers before claims final) :
    claims ≠ [] := by
  cases run <;> simp

/-- Every carry passed to a later claim is active, including a carry produced
by an immediate segment-boundary reopen. -/
theorem recursive_outgoing_active
    {ChallengeField : Type} [Field ChallengeField]
    {schema : FullClaim.Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before intermediate outgoing :
      Carry Digest (Challenges ChallengeField) (State ChallengeField)}
    {tail : List (Receipt schema Digest ChallengeField verify)}
    {final : ClosedCarry Digest}
    (receipt : Receipt schema Digest ChallengeField verify)
    (consumes : Consumes ProductState.Balanced before
      receipt.claim.memory intermediate)
    (continues : Continues derive headers intermediate outgoing)
    (rest : DelayedRun verify derive headers outgoing tail final) :
    ∃ active, outgoing = .active active :=
  continues.outgoing_active

end DelayedRun

/-- One complete lifetime starts with the base segment opening, then consumes
one verified full claim per invocation, and ends at the terminal closed carry.
The fixed `derive` function is shared by every segment opening. -/
structure CompleteRun
    {ChallengeField : Type} [Field ChallengeField]
    {schema : FullClaim.Schema} {Digest : Type}
    (verify : V2Verifier schema Digest ChallengeField)
    (derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField)
    (headers : ChainHeaders Digest)
    (initial final : ClosedCarry Digest) where
  basePrecommit : Roots Digest
  baseActiveAccessCount : Nat
  baseCanOpen : initial.CanOpen
  baseActiveCountInRange : baseActiveAccessCount < operationCountLimit
  baseEndTimestampInRange :
    initial.globalTimestamp + baseActiveAccessCount < timestampLimit
  baseActive : ActiveCarry Digest (Challenges ChallengeField)
    (State ChallengeField)
  baseOpened :
    openSegment derive headers basePrecommit baseActiveAccessCount initial
      baseCanOpen baseActiveCountInRange baseEndTimestampInRange =
        .active baseActive
  claims : List (Receipt schema Digest ChallengeField verify)
  delayed : DelayedRun verify derive headers (.active baseActive) claims final

namespace CompleteRun

theorem claims_nonempty
    {ChallengeField : Type} [Field ChallengeField]
    {schema : FullClaim.Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    (run : CompleteRun verify derive headers initial final) :
    run.claims ≠ [] :=
  run.delayed.claims_nonempty

/-- Base contributes one invocation. Each verified claim contributes exactly
one later invocation, including the terminal claim. There is no extra
claim-free segment-boundary invocation. -/
theorem augmented_invocation_count
    {ChallengeField : Type} [Field ChallengeField]
    {schema : FullClaim.Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {derive : ClosedCarry Digest → Roots Digest → Nat →
      Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    (run : CompleteRun verify derive headers initial final) :
    1 + run.claims.length = run.claims.length + 1 := by
  omega

end CompleteRun

end Nightstream.Protocol.NebulaV2.AugmentedLifecycle
