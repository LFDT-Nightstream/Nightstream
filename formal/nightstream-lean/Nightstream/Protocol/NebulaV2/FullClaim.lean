import Nightstream.Protocol.NebulaV2.FPrime
import Nightstream.Protocol.NebulaV2.Profile

/-!
Contract: exact full-claim ownership for delayed Nebula V2 consumption.

Assurance tier: model-level.

Owns one mandatory full fresh claim, its NIFS verification receipt, and a run
that consumes the memory suffix of the same verified full claim. The claim has
named CCS, application, bundle, recursive-state, and memory components.

Does not own the concrete CCS types, NIFS verifier, bundle codec, generated
rows, fold extraction, or terminal backend.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.FullClaim

open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Lifecycle

/-- Types fixed by one concrete verifier key. `CommitmentBundle` is mandatory;
there is no optional memory sidecar in this schema. -/
structure Schema where
  CcsPublic : Type
  ApplicationPublic : Type
  CommitmentBundle : Type
  RecursiveState : Type
  NifsProof : Type

structure Claim
    (schema : Schema) (Digest Challenge Products : Type) where
  profile : Profile.Identity
  ccsPublic : schema.CcsPublic
  applicationPublic : schema.ApplicationPublic
  commitmentBundle : schema.CommitmentBundle
  recursiveState : schema.RecursiveState
  memory : ClaimSuffix Digest Challenge Products

/-- A verifier-owned NIFS predicate accepts this exact full claim and proof. -/
structure Verified
    (schema : Schema) (Digest Challenge Products : Type)
    (verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop) where
  claim : Claim schema Digest Challenge Products
  proof : schema.NifsProof
  profileExact : claim.profile = Profile.v2
  accepted : verify proof claim

/-- Delayed consumption uses the memory field of the exact verified object. -/
structure Transition
    {schema : Schema} {Digest Challenge Products : Type}
    (verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop)
    (balanced : Products → Prop)
    (before : Carry Digest Challenge Products)
    (verified : Verified schema Digest Challenge Products verify)
    (after : Carry Digest Challenge Products) : Prop where
  consumes : Consumes balanced before verified.claim.memory after

theorem accepted_claim_is_consumed
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {verified : Verified schema Digest Challenge Products verify}
    (transition : Transition verify balanced before verified after) :
    verify verified.proof verified.claim ∧
      Consumes balanced before verified.claim.memory after :=
  ⟨verified.accepted, transition.consumes⟩

namespace Transition

theorem mono
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {weaker stronger : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {verified : Verified schema Digest Challenge Products verify}
    (implies : ∀ products, weaker products → stronger products)
    (transition : Transition verify weaker before verified after) :
    Transition verify stronger before verified after where
  consumes := transition.consumes.mono implies

end Transition

inductive VerifiedRun
    {schema : Schema} {Digest Challenge Products : Type}
    (verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop)
    (balanced : Products → Prop) :
    Carry Digest Challenge Products →
      List (Verified schema Digest Challenge Products verify) →
      Carry Digest Challenge Products → Prop
  | nil (state : Carry Digest Challenge Products) :
      VerifiedRun verify balanced state [] state
  | cons
      {before middle after : Carry Digest Challenge Products}
      {head : Verified schema Digest Challenge Products verify}
      {tail : List (Verified schema Digest Challenge Products verify)}
      (step : Transition verify balanced before head middle)
      (rest : VerifiedRun verify balanced middle tail after) :
      VerifiedRun verify balanced before (head :: tail) after

namespace VerifiedRun

/-- A verified run from an active carry to a closed carry must start with a
real consumption step. That first step supplies the complete active-carry
well-formedness predicate. This is useful in the honest-completeness direction:
the caller does not need to repeat the range facts already present in the
verified run. -/
theorem initialActiveWellFormed
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {claims : List (Verified schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced (.active active) claims
      (.closed closed)) :
    active.WellFormed := by
  cases run with
  | cons step _ => exact step.consumes.activeWellFormed

theorem from_closed_is_empty
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {closed : ClosedCarry Digest}
    {claims : List (Verified schema Digest Challenge Products verify)}
    {after : Carry Digest Challenge Products}
    (run : VerifiedRun verify balanced (.closed closed) claims after) :
    claims = [] ∧ after = .closed closed := by
  cases run with
  | nil => exact ⟨rfl, rfl⟩
  | cons step _ =>
      exact False.elim (cannot_consume_from_closed step.consumes)

theorem mono
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (Verified schema Digest Challenge Products verify)}
    {weaker stronger : Products → Prop}
    (implies : ∀ products, weaker products → stronger products)
    (run : VerifiedRun verify weaker before claims after) :
    VerifiedRun verify stronger before claims after := by
  induction run with
  | nil => exact .nil _
  | cons step _ inductionHypothesis =>
      exact .cons (step.mono implies) inductionHypothesis

theorem remaining_eq_length_add
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (Verified schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced before claims after) :
    remainingSteps before = claims.length + remainingSteps after := by
  induction run with
  | nil => simp
  | cons step _ inductionHypothesis =>
      have decrease := consumes_decreases_remaining_by_one step.consumes
      simp only [List.length_cons]
      omega

theorem every_claim_accepted
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (Verified schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced before claims after) :
    ∀ receipt ∈ claims, verify receipt.proof receipt.claim := by
  induction run with
  | nil => simp
  | cons step _ inductionHypothesis =>
      intro receipt member
      simp only [List.mem_cons] at member
      rcases member with equal | member
      · subst receipt
        exact (accepted_claim_is_consumed step).1
      · exact inductionHypothesis receipt member

private theorem from_active_to_closed_has_balanced_products
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (Verified schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced before claims after)
    (beforeActive : ∃ active, before = .active active)
    (afterClosed : ∃ closed, after = .closed closed) :
    ∃ receipt ∈ claims, balanced receipt.claim.memory.productsAfter := by
  induction run with
  | nil =>
      rcases beforeActive with ⟨active, beforeActive⟩
      rcases afterClosed with ⟨closed, afterClosed⟩
      rw [beforeActive] at afterClosed
      cases afterClosed
  | cons step _ inductionHypothesis =>
      cases step.consumes with
      | interior _ _ =>
          rcases inductionHypothesis ⟨_, rfl⟩ afterClosed with
            ⟨receipt, member, productsBalanced⟩
          exact ⟨receipt, by simp [member], productsBalanced⟩
      | close _ _ checks =>
          exact ⟨_, by simp, checks.productsBalanced⟩

theorem to_closed_has_balanced_products
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {claims : List (Verified schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced (.active active) claims
      (.closed closed)) :
    ∃ receipt ∈ claims, balanced receipt.claim.memory.productsAfter :=
  from_active_to_closed_has_balanced_products run
    ⟨active, rfl⟩ ⟨closed, rfl⟩

theorem full_segment_has_exact_claim_count
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {claims : List (Verified schema Digest Challenge Products verify)}
    (startsAtZero : active.stepIndex.val = 0)
    (run : VerifiedRun verify balanced (.active active) claims
      (.closed closed)) :
    claims.length = claimsPerSegment := by
  have accounting := remaining_eq_length_add run
  simp only [remainingSteps, Nat.add_zero] at accounting
  rw [startsAtZero] at accounting
  omega

private theorem active_to_closed_segment_index
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (Verified schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced before claims after)
    (beforeActive : ∃ active, before = .active active)
    (afterClosed : ∃ closed, after = .closed closed) :
    ∀ active closed,
      before = .active active →
      after = .closed closed →
      closed.segmentIndex = active.segmentIndex + 1 := by
  induction run with
  | nil =>
      intro active closed beforeEqual afterEqual
      rw [beforeEqual] at afterEqual
      cases afterEqual
  | cons step rest inductionHypothesis =>
      intro active closed beforeEqual afterEqual
      cases step.consumes with
      | interior _ _ =>
          cases beforeEqual
          have restBoundary := inductionHypothesis
            ⟨_, rfl⟩ afterClosed _ closed rfl afterEqual
          simpa [interiorCarry] using restBoundary
      | close agreement last checks =>
          cases beforeEqual
          have tailExact := rest.from_closed_is_empty
          cases tailExact.2
          cases afterEqual
          rfl

theorem to_closed_segment_index
    {schema : Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof → Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {claims : List (Verified schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced (.active active) claims
      (.closed closed)) :
    closed.segmentIndex = active.segmentIndex + 1 :=
  active_to_closed_segment_index run ⟨active, rfl⟩ ⟨closed, rfl⟩
    active closed rfl rfl

end VerifiedRun

end Nightstream.Protocol.NebulaV2.FullClaim
