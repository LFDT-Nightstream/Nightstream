import Nightstream.Protocol.Nebula.CommitmentBundle
import Nightstream.Protocol.Nebula.ProductState

/-!
Contract: independent delayed-consumption state machine for V2 memory carry.

Assurance tier: model-level.

Owns canonical closed-versus-active state separation, exact prior-claim
matching, interior step advance, and all deterministic segment-close checks.

Does not own NIFS cryptography, transcript or hash binding, circuit rows,
recursive-size closure, or reverse fold extraction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.FPrime

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Lifecycle

/-- The operations lane has its own header. Initial and final snapshots use
one shared memory-lane header, as required for cross-segment equality. -/
structure ChainHeaders (Digest : Type) where
  operations : Digest
  memory : Digest
deriving DecidableEq, Repr

structure Roots (Digest : Type) where
  operations : Digest
  initialSnapshot : Digest
  finalSnapshot : Digest
deriving DecidableEq, Repr

def ChainHeaders.roots
    {Digest : Type} (headers : ChainHeaders Digest) : Roots Digest :=
  { operations := headers.operations
    initialSnapshot := headers.memory
    finalSnapshot := headers.memory }

@[ext]
theorem Roots.ext
    {Digest : Type} {left right : Roots Digest}
    (operations : left.operations = right.operations)
    (initialSnapshot :
      left.initialSnapshot = right.initialSnapshot)
    (finalSnapshot : left.finalSnapshot = right.finalSnapshot) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Closed carries contain no inactive challenge, product, precommit, or step
fields. Their canonical serialized zero values are an encoding obligation. -/
structure ClosedCarry (Digest : Type) where
  segmentIndex : Nat
  globalTimestamp : Nat
  memoryRoot : Digest
deriving DecidableEq, Repr

@[ext]
theorem ClosedCarry.ext
    {Digest : Type} {left right : ClosedCarry Digest}
    (segmentIndex : left.segmentIndex = right.segmentIndex)
    (globalTimestamp : left.globalTimestamp = right.globalTimestamp)
    (memoryRoot : left.memoryRoot = right.memoryRoot) :
    left = right := by
  cases left
  cases right
  simp_all

structure ActiveCarry
    (Digest Challenge Products : Type) where
  segmentIndex : Nat
  stepIndex : Fin claimsPerSegment
  globalTimestamp : Nat
  segmentStartTimestamp : Nat
  segmentActiveAccessCount : Nat
  segmentEndTimestamp : Nat
  challenge : Challenge
  products : Products
  dPre : Roots Digest
  dSeen : Roots Digest
  memoryRoot : Digest
deriving Repr

@[ext]
theorem ActiveCarry.ext
    {Digest Challenge Products : Type}
    {left right : ActiveCarry Digest Challenge Products}
    (segmentIndex : left.segmentIndex = right.segmentIndex)
    (stepIndex : left.stepIndex = right.stepIndex)
    (globalTimestamp : left.globalTimestamp = right.globalTimestamp)
    (segmentStartTimestamp :
      left.segmentStartTimestamp = right.segmentStartTimestamp)
    (segmentActiveAccessCount :
      left.segmentActiveAccessCount = right.segmentActiveAccessCount)
    (segmentEndTimestamp :
      left.segmentEndTimestamp = right.segmentEndTimestamp)
    (challenge : left.challenge = right.challenge)
    (products : left.products = right.products)
    (dPre : left.dPre = right.dPre)
    (dSeen : left.dSeen = right.dSeen)
    (memoryRoot : left.memoryRoot = right.memoryRoot) :
    left = right := by
  cases left
  cases right
  simp_all

def ClosedCarry.CanOpen
    {Digest : Type} (closed : ClosedCarry Digest) : Prop :=
  closed.segmentIndex < maximumSegments ∧
    closed.globalTimestamp < timestampLimit

def ActiveCarry.WellFormed
    {Digest Challenge Products : Type}
    (active : ActiveCarry Digest Challenge Products) : Prop :=
  active.segmentIndex < maximumSegments ∧
    active.segmentActiveAccessCount < operationCountLimit ∧
    active.segmentEndTimestamp =
      active.segmentStartTimestamp + active.segmentActiveAccessCount ∧
    active.segmentEndTimestamp < timestampLimit ∧
    active.segmentStartTimestamp ≤ active.globalTimestamp ∧
    active.globalTimestamp ≤ active.segmentEndTimestamp

inductive Carry (Digest Challenge Products : Type) where
  | closed (state : ClosedCarry Digest)
  | active (state : ActiveCarry Digest Challenge Products)
deriving Repr

/-- The one global integer timestamp carried by either canonical phase. -/
def carryTimestamp
    {Digest Challenge Products : Type} :
    Carry Digest Challenge Products → Nat
  | .closed state => state.globalTimestamp
  | .active state => state.globalTimestamp

/-- Public suffix of the exact prior fresh claim that can update memory carry. -/
structure ClaimSuffix (Digest Challenge Products : Type) where
  segmentIndex : Nat
  stepIndex : Fin claimsPerSegment
  timestampIn : Nat
  timestampOut : Nat
  segmentStartTimestamp : Nat
  segmentEndTimestamp : Nat
  activeAccessCount : Nat
  challenge : Challenge
  dPre : Roots Digest
  dSeenBefore : Roots Digest
  dSeenAfter : Roots Digest
  productsBefore : Products
  productsAfter : Products
deriving Repr

@[ext]
theorem ClaimSuffix.ext
    {Digest Challenge Products : Type}
    {left right : ClaimSuffix Digest Challenge Products}
    (segmentIndex : left.segmentIndex = right.segmentIndex)
    (stepIndex : left.stepIndex = right.stepIndex)
    (timestampIn : left.timestampIn = right.timestampIn)
    (timestampOut : left.timestampOut = right.timestampOut)
    (segmentStartTimestamp :
      left.segmentStartTimestamp = right.segmentStartTimestamp)
    (segmentEndTimestamp :
      left.segmentEndTimestamp = right.segmentEndTimestamp)
    (activeAccessCount : left.activeAccessCount = right.activeAccessCount)
    (challenge : left.challenge = right.challenge)
    (dPre : left.dPre = right.dPre)
    (dSeenBefore : left.dSeenBefore = right.dSeenBefore)
    (dSeenAfter : left.dSeenAfter = right.dSeenAfter)
    (productsBefore : left.productsBefore = right.productsBefore)
    (productsAfter : left.productsAfter = right.productsAfter) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Every consumed suffix must match the active carry that existed before the
claim was verified. -/
structure MatchesActive
    {Digest Challenge Products : Type}
    (active : ActiveCarry Digest Challenge Products)
    (claim : ClaimSuffix Digest Challenge Products) : Prop where
  activeWellFormed : active.WellFormed
  segmentIndex : claim.segmentIndex = active.segmentIndex
  stepIndex : claim.stepIndex = active.stepIndex
  timestampIn : claim.timestampIn = active.globalTimestamp
  segmentStartTimestamp :
    claim.segmentStartTimestamp = active.segmentStartTimestamp
  segmentEndTimestamp :
    claim.segmentEndTimestamp = active.segmentEndTimestamp
  challenge : claim.challenge = active.challenge
  dPre : claim.dPre = active.dPre
  dSeen : claim.dSeenBefore = active.dSeen
  products : claim.productsBefore = active.products
  timestampAdvance :
    claim.timestampOut = claim.timestampIn + claim.activeAccessCount
  activeCountBound : claim.activeAccessCount ≤ 63
  timestampWithinDeclaredEnd :
    claim.timestampOut ≤ active.segmentEndTimestamp
  timestampInRange : claim.timestampIn < timestampLimit
  timestampOutRange : claim.timestampOut < timestampLimit

namespace MatchesActive

/-- A matching claim carries the exact challenge and segment timestamp bounds
from the active carry. These fields cannot be replaced independently while
keeping the roots and products unchanged. -/
theorem binds_challenge_and_segment_bounds
    {Digest Challenge Products : Type}
    {active : ActiveCarry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    (agreement : MatchesActive active claim) :
    claim.challenge = active.challenge ∧
      claim.segmentStartTimestamp = active.segmentStartTimestamp ∧
      claim.segmentEndTimestamp = active.segmentEndTimestamp :=
  ⟨agreement.challenge, agreement.segmentStartTimestamp,
    agreement.segmentEndTimestamp⟩

end MatchesActive

structure CloseChecks
    {Digest Challenge Products : Type}
    (balanced : Products → Prop)
    (active : ActiveCarry Digest Challenge Products)
    (claim : ClaimSuffix Digest Challenge Products) : Prop where
  seenEqualsPrecommit : claim.dSeenAfter = active.dPre
  initialEqualsMemory :
    claim.dSeenAfter.initialSnapshot = active.memoryRoot
  productsBalanced : balanced claim.productsAfter
  timestampEqualsDeclaredEnd : claim.timestampOut = active.segmentEndTimestamp

def interiorCarry
    {Digest Challenge Products : Type}
    (active : ActiveCarry Digest Challenge Products)
    (claim : ClaimSuffix Digest Challenge Products)
    (notLast : active.stepIndex.val + 1 < claimsPerSegment) :
    ActiveCarry Digest Challenge Products :=
  { active with
    stepIndex := ⟨active.stepIndex.val + 1, notLast⟩
    globalTimestamp := claim.timestampOut
    products := claim.productsAfter
    dSeen := claim.dSeenAfter }

def closedCarryAfter
    {Digest Challenge Products : Type}
    (active : ActiveCarry Digest Challenge Products)
    (claim : ClaimSuffix Digest Challenge Products) : ClosedCarry Digest :=
  { segmentIndex := active.segmentIndex + 1
    globalTimestamp := claim.timestampOut
    memoryRoot := claim.dSeenAfter.finalSnapshot }

/-- Deterministic memory transition after the exact prior claim is accepted. -/
inductive Consumes
    {Digest Challenge Products : Type}
    (balanced : Products → Prop) :
    Carry Digest Challenge Products →
      ClaimSuffix Digest Challenge Products →
      Carry Digest Challenge Products → Prop
  | interior
      {active : ActiveCarry Digest Challenge Products}
      {claim : ClaimSuffix Digest Challenge Products}
      (agreement : MatchesActive active claim)
      (notLast : active.stepIndex.val + 1 < claimsPerSegment) :
      Consumes balanced (.active active) claim
        (.active (interiorCarry active claim notLast))
  | close
      {active : ActiveCarry Digest Challenge Products}
      {claim : ClaimSuffix Digest Challenge Products}
      (agreement : MatchesActive active claim)
      (last : active.stepIndex.val + 1 = claimsPerSegment)
      (checks : CloseChecks balanced active claim) :
      Consumes balanced (.active active) claim
        (.closed (closedCarryAfter active claim))

namespace Consumes

/-- Every valid consumption advances the global integer timestamp by exactly
the claim's derived active-access count. -/
theorem timestampAdvance
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    (consumption : Consumes balanced before claim after) :
    claim.timestampOut = claim.timestampIn + claim.activeAccessCount := by
  cases consumption with
  | interior agreement _ => exact agreement.timestampAdvance
  | close agreement _ _ => exact agreement.timestampAdvance

/-- Consumption starts at the exact timestamp in the prior authoritative
carry. -/
theorem timestampIn_eq_before
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    (consumption : Consumes balanced before claim after) :
    claim.timestampIn = carryTimestamp before := by
  cases consumption with
  | interior agreement _ => exact agreement.timestampIn
  | close agreement _ _ => exact agreement.timestampIn

/-- Consumption writes the exact claim endpoint into the next carry. -/
theorem timestampOut_eq_after
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    (consumption : Consumes balanced before claim after) :
    claim.timestampOut = carryTimestamp after := by
  cases consumption <;> rfl

/-- A transition from an active carry derives the complete active-carry
well-formedness predicate from its exact matching claim. -/
theorem activeWellFormed
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {after : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    (consumption : Consumes balanced (.active active) claim after) :
    active.WellFormed := by
  cases consumption with
  | interior agreement _ => exact agreement.activeWellFormed
  | close agreement _ _ => exact agreement.activeWellFormed

/-- Strengthening the accepted close predicate preserves an already checked
transition when every old close proof implies the new predicate. -/
theorem mono
    {Digest Challenge Products : Type}
    {before after : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    {weaker stronger : Products → Prop}
    (implies : ∀ products, weaker products → stronger products)
    (transition : Consumes weaker before claim after) :
    Consumes stronger before claim after := by
  cases transition with
  | interior agreement notLast =>
      exact .interior agreement notLast
  | close agreement last checks =>
      exact .close agreement last
        { seenEqualsPrecommit := checks.seenEqualsPrecommit
          initialEqualsMemory := checks.initialEqualsMemory
          productsBalanced := implies _ checks.productsBalanced
          timestampEqualsDeclaredEnd := checks.timestampEqualsDeclaredEnd }

/-- For fixed input carry and suffix, the memory transition has one output
carry. Proof witnesses and the close predicate cannot select another state. -/
theorem after_unique
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {before leftAfter rightAfter : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    (left : Consumes balanced before claim leftAfter)
    (right : Consumes balanced before claim rightAfter) :
    leftAfter = rightAfter := by
  cases left with
  | interior leftAgreement leftNotLast =>
      cases right with
      | interior rightAgreement rightNotLast => rfl
      | close rightAgreement rightLast rightChecks =>
          omega
  | close leftAgreement leftLast leftChecks =>
      cases right with
      | interior rightAgreement rightNotLast =>
          omega
      | close rightAgreement rightLast rightChecks => rfl

end Consumes

/-- The verifier predicate and consumed suffix share the same `claim` field.
This rules out verifying one claim and advancing from another by construction. -/
structure VerifiedTransition
    {Digest Challenge Products : Type}
    (verify : ClaimSuffix Digest Challenge Products → Prop)
    (balanced : Products → Prop)
    (before : Carry Digest Challenge Products)
    (claim : ClaimSuffix Digest Challenge Products)
    (after : Carry Digest Challenge Products) : Prop where
  verified : verify claim
  consumes : Consumes balanced before claim after

namespace VerifiedTransition

theorem mono
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    {weaker stronger : Products → Prop}
    (implies : ∀ products, weaker products → stronger products)
    (transition : VerifiedTransition verify weaker before claim after) :
    VerifiedTransition verify stronger before claim after where
  verified := transition.verified
  consumes := transition.consumes.mono implies

end VerifiedTransition

/-- Local suffix verifier fact. This theorem does not claim that the suffix
identifies a complete fresh claim. Production code must use
`FullClaim.accepted_claim_is_consumed`. -/
theorem verified_suffix
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    (transition : VerifiedTransition verify balanced before claim after) :
    verify claim :=
  transition.verified

theorem cannot_consume_from_closed
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {closed : ClosedCarry Digest}
    {claim : ClaimSuffix Digest Challenge Products}
    {after : Carry Digest Challenge Products} :
    ¬ Consumes balanced (.closed closed) claim after := by
  intro transition
  cases transition

def remainingSteps
    {Digest Challenge Products : Type} :
    Carry Digest Challenge Products → Nat
  | .closed _ => 0
  | .active active => claimsPerSegment - active.stepIndex.val

theorem consumes_decreases_remaining_by_one
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    (transition : Consumes balanced before claim after) :
    remainingSteps before = remainingSteps after + 1 := by
  cases transition with
  | @interior active claim _ notLast =>
      simp only [remainingSteps, interiorCarry]
      omega
  | @close active claim _ last _ =>
      simp only [remainingSteps]
      have stepBound := active.stepIndex.isLt
      omega

/-- A verified run consumes a list in order. Each transition verifies and
consumes the same claim value. -/
inductive VerifiedRun
    {Digest Challenge Products : Type}
    (verify : ClaimSuffix Digest Challenge Products → Prop)
    (balanced : Products → Prop) :
    Carry Digest Challenge Products →
      List (ClaimSuffix Digest Challenge Products) →
      Carry Digest Challenge Products → Prop
  | nil (state : Carry Digest Challenge Products) :
      VerifiedRun verify balanced state [] state
  | cons
      {before middle after : Carry Digest Challenge Products}
      {claim : ClaimSuffix Digest Challenge Products}
      {claims : List (ClaimSuffix Digest Challenge Products)}
      (head : VerifiedTransition verify balanced before claim middle)
      (tail : VerifiedRun verify balanced middle claims after) :
      VerifiedRun verify balanced before (claim :: claims) after

namespace VerifiedRun

theorem mono
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    {weaker stronger : Products → Prop}
    (implies : ∀ products, weaker products → stronger products)
    (run : VerifiedRun verify weaker before claims after) :
    VerifiedRun verify stronger before claims after := by
  induction run with
  | nil => exact .nil _
  | cons head _ inductionHypothesis =>
      exact .cons (head.mono implies) inductionHypothesis

theorem from_closed_is_empty
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {closed : ClosedCarry Digest}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    {after : Carry Digest Challenge Products}
    (run : VerifiedRun verify balanced (.closed closed) claims after) :
    claims = [] ∧ after = .closed closed := by
  cases run with
  | nil => exact ⟨rfl, rfl⟩
  | cons head _ =>
      exact False.elim (cannot_consume_from_closed head.consumes)

private theorem from_active_to_closed_has_balanced_products
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    (run : VerifiedRun verify balanced before claims after)
    (beforeActive : ∃ active, before = .active active)
    (afterClosed : ∃ closed, after = .closed closed) :
    ∃ finalClaim ∈ claims, balanced finalClaim.productsAfter := by
  induction run with
  | nil =>
      rcases beforeActive with ⟨active, beforeActive⟩
      rcases afterClosed with ⟨closed, afterClosed⟩
      rw [beforeActive] at afterClosed
      cases afterClosed
  | @cons _ _ _ claim _ head _ inductionHypothesis =>
      cases head.consumes with
      | interior _ _ =>
          rcases inductionHypothesis ⟨_, rfl⟩ afterClosed with
            ⟨finalClaim, member, productsBalanced⟩
          exact ⟨finalClaim, by simp [member], productsBalanced⟩
      | close _ _ checks =>
          exact ⟨claim, by simp, checks.productsBalanced⟩

/-- A run that reaches a closed carry contains a closing transition, and that
transition exposes the exact product predicate checked at close. This theorem
lets the ideal verifier obtain fingerprint acceptance from F-prime closure
instead of accepting it as a separate assumption. -/
theorem to_closed_has_balanced_products
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    (run : VerifiedRun verify balanced (.active active) claims
      (.closed closed)) :
    ∃ finalClaim ∈ claims, balanced finalClaim.productsAfter :=
  from_active_to_closed_has_balanced_products run
    ⟨active, rfl⟩ ⟨closed, rfl⟩

theorem remaining_eq_length_add
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    (run : VerifiedRun verify balanced before claims after) :
    remainingSteps before = claims.length + remainingSteps after := by
  induction run with
  | nil => simp
  | cons head _ inductionHypothesis =>
      have decrease := consumes_decreases_remaining_by_one head.consumes
      simp only [List.length_cons]
      omega

theorem every_claim_verified
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    (run : VerifiedRun verify balanced before claims after) :
    ∀ claim ∈ claims, verify claim := by
  induction run with
  | nil => simp
  | cons head _ inductionHypothesis =>
      intro claim member
      simp only [List.mem_cons] at member
      rcases member with equal | member
      · subst claim
        exact head.verified
      · exact inductionHypothesis claim member

theorem to_closed_has_exact_remaining_length
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    (run : VerifiedRun verify balanced (.active active) claims
      (.closed closed)) :
    claims.length = claimsPerSegment - active.stepIndex.val := by
  have accounting := remaining_eq_length_add run
  simp only [remainingSteps, Nat.add_zero] at accounting
  exact accounting.symm

theorem full_segment_has_exact_claim_count
    {Digest Challenge Products : Type}
    {verify : ClaimSuffix Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    (startsAtZero : active.stepIndex.val = 0)
    (run : VerifiedRun verify balanced (.active active) claims
      (.closed closed)) :
    claims.length = claimsPerSegment := by
  rw [to_closed_has_exact_remaining_length run, startsAtZero]
  omega

end VerifiedRun

theorem active_to_closed_requires_all_close_checks
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    {closed : ClosedCarry Digest}
    (transition :
      Consumes balanced (.active active) claim (.closed closed)) :
    claim.stepIndex.val + 1 = claimsPerSegment ∧
      CloseChecks balanced active claim ∧
      closed = closedCarryAfter active claim := by
  cases transition with
  | close agreement last checks =>
      exact ⟨by simpa [agreement.stepIndex] using last, checks, rfl⟩

theorem close_preserves_global_timestamp_and_final_root
    {Digest Challenge Products : Type}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {claim : ClaimSuffix Digest Challenge Products}
    {closed : ClosedCarry Digest}
    (transition :
      Consumes balanced (.active active) claim (.closed closed)) :
    closed.globalTimestamp = claim.timestampOut ∧
      closed.memoryRoot = claim.dSeenAfter.finalSnapshot ∧
      closed.segmentIndex = active.segmentIndex + 1 := by
  cases transition
  exact ⟨rfl, rfl, rfl⟩

/-- Open a segment only from a closed state. `derive` receives the complete
prechallenge roots before it returns the active challenge. -/
def openSegment
    {Digest Challenge ChallengeField : Type}
    [One ChallengeField]
    (derive : ClosedCarry Digest → Roots Digest → Nat → Challenge)
    (headers : ChainHeaders Digest)
    (precommit : Roots Digest)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest)
    (_canOpen : closed.CanOpen)
    (_activeCountInRange : activeAccessCount < operationCountLimit)
    (_endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    Carry Digest Challenge (ProductState.State ChallengeField) :=
  .active
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
      memoryRoot := closed.memoryRoot }

/-- Structural data-flow fact only. It does not claim that `derive` is a
secure or nonconstant Fiat-Shamir function. The transcript refinement owns
that separate obligation. -/
theorem openSegment_uses_derive_output
    {Digest Challenge ChallengeField : Type}
    [One ChallengeField]
    (derive : ClosedCarry Digest → Roots Digest → Nat → Challenge)
    (headers : ChainHeaders Digest)
    (precommit : Roots Digest)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    match openSegment (ChallengeField := ChallengeField) derive headers precommit
        activeAccessCount closed canOpen activeCountInRange
        endTimestampInRange with
    | .active active =>
        active.challenge = derive closed precommit activeAccessCount
    | .closed _ => False := by
  rfl

theorem openSegment_products_are_one
    {Digest Challenge ChallengeField : Type}
    [One ChallengeField]
    (derive : ClosedCarry Digest → Roots Digest → Nat → Challenge)
    (headers : ChainHeaders Digest)
    (precommit : Roots Digest)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    match openSegment (ChallengeField := ChallengeField) derive headers precommit
        activeAccessCount closed canOpen activeCountInRange
        endTimestampInRange with
    | .active active => active.products = ProductState.one
    | .closed _ => False := by
  rfl

theorem openSegment_uses_shared_memory_headers
    {Digest Challenge ChallengeField : Type}
    [One ChallengeField]
    (derive : ClosedCarry Digest → Roots Digest → Nat → Challenge)
    (headers : ChainHeaders Digest)
    (precommit : Roots Digest)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    match openSegment (ChallengeField := ChallengeField) derive headers precommit
        activeAccessCount closed canOpen activeCountInRange
        endTimestampInRange with
    | .active active =>
        active.dSeen.initialSnapshot = active.dSeen.finalSnapshot
    | .closed _ => False := by
  rfl

theorem openSegment_wellFormed
    {Digest Challenge ChallengeField : Type}
    [One ChallengeField]
    (derive : ClosedCarry Digest → Roots Digest → Nat → Challenge)
    (headers : ChainHeaders Digest)
    (precommit : Roots Digest)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    match openSegment (ChallengeField := ChallengeField) derive headers precommit
        activeAccessCount closed canOpen activeCountInRange
        endTimestampInRange with
    | .active active => active.WellFormed
    | .closed _ => False := by
  exact ⟨canOpen.1, activeCountInRange, rfl, endTimestampInRange,
    Nat.le_refl _, Nat.le_add_right _ _⟩

end Nightstream.Protocol.Nebula.FPrime
