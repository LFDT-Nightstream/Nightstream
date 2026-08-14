import Nightstream.Protocol.Nebula.FPrime
import Nightstream.Protocol.Nebula.ProductionProfileCandidates

/-!
Contract: delayed F-prime consumption for one field-native checked-step batch.

One verified fresh claim owns exactly `E` consecutive memory suffixes. The
outer delayed transition consumes those suffixes in order only after the
complete claim is verified. The model proves exact per-segment claim counts
and does not reuse the factor-one `FullClaim` type.

This is sequential checked-step batching inside one relation. It is not a
batch of independently verified fresh claims.

Does not own generated rows, the concrete NIFS verifier, codecs, transcript
security, a selected production candidate, or terminal verification.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ProductionBatchedFPrime

open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates

/-- Ordered deterministic consumption without a separate verifier predicate
on each suffix. Authority comes from the one verified enclosing batch claim. -/
inductive ConsumesList
    {Digest Challenge Products : Type}
    (balanced : Products -> Prop) :
    Carry Digest Challenge Products ->
      List (ClaimSuffix Digest Challenge Products) ->
      Carry Digest Challenge Products -> Prop
  | nil (state : Carry Digest Challenge Products) :
      ConsumesList balanced state [] state
  | cons
      {before middle after : Carry Digest Challenge Products}
      {head : ClaimSuffix Digest Challenge Products}
      {tail : List (ClaimSuffix Digest Challenge Products)}
      (step : Consumes balanced before head middle)
      (rest : ConsumesList balanced middle tail after) :
      ConsumesList balanced before (head :: tail) after

namespace ConsumesList

theorem from_closed_is_empty
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {closed : ClosedCarry Digest}
    {suffixes : List (ClaimSuffix Digest Challenge Products)}
    {after : Carry Digest Challenge Products}
    (consumes : ConsumesList balanced (.closed closed) suffixes after) :
    suffixes = [] /\ after = .closed closed := by
  cases consumes with
  | nil => exact ⟨rfl, rfl⟩
  | cons step _ => exact False.elim (cannot_consume_from_closed step)

theorem remaining_eq_length_add
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {suffixes : List (ClaimSuffix Digest Challenge Products)}
    (consumes : ConsumesList balanced before suffixes after) :
    remainingSteps before = suffixes.length + remainingSteps after := by
  induction consumes with
  | nil => simp
  | cons step _ inductionHypothesis =>
      have decrease := consumes_decreases_remaining_by_one step
      simp only [List.length_cons]
      omega

theorem mono
    {Digest Challenge Products : Type}
    {weaker stronger : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {suffixes : List (ClaimSuffix Digest Challenge Products)}
    (implies : forall products, weaker products -> stronger products)
    (consumes : ConsumesList weaker before suffixes after) :
    ConsumesList stronger before suffixes after := by
  induction consumes with
  | nil => exact .nil _
  | cons step _ inductionHypothesis =>
      exact .cons (step.mono implies) inductionHypothesis

theorem append
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {before middle after : Carry Digest Challenge Products}
    {left right : List (ClaimSuffix Digest Challenge Products)}
    (firstPart : ConsumesList balanced before left middle)
    (secondPart : ConsumesList balanced middle right after) :
    ConsumesList balanced before (left ++ right) after := by
  induction firstPart with
  | nil => exact secondPart
  | cons step _ inductionHypothesis =>
      exact .cons step (inductionHypothesis secondPart)

/-- Fixed input carry and an exact ordered suffix list determine one output
carry. -/
theorem after_unique
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {before leftAfter rightAfter : Carry Digest Challenge Products}
    {suffixes : List (ClaimSuffix Digest Challenge Products)}
    (left : ConsumesList balanced before suffixes leftAfter)
    (right : ConsumesList balanced before suffixes rightAfter) :
    leftAfter = rightAfter := by
  induction left with
  | nil =>
      cases right
      rfl
  | @cons before middle leftAfter head tail first rest inductionHypothesis =>
      cases right with
      | cons second rightRest =>
          have middleExact := FPrime.Consumes.after_unique first second
          subst middle
          exact inductionHypothesis rightRest

private theorem from_active_to_closed_has_balanced_products
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {suffixes : List (ClaimSuffix Digest Challenge Products)}
    (consumes : ConsumesList balanced before suffixes after)
    (beforeActive : ∃ active, before = .active active)
    (afterClosed : ∃ closed, after = .closed closed) :
    ∃ finalSuffix ∈ suffixes, balanced finalSuffix.productsAfter := by
  induction consumes with
  | nil =>
      rcases beforeActive with ⟨active, beforeEqual⟩
      rcases afterClosed with ⟨closed, afterEqual⟩
      rw [beforeEqual] at afterEqual
      cases afterEqual
  | cons step _ inductionHypothesis =>
      cases step with
      | interior _ _ =>
          rcases inductionHypothesis ⟨_, rfl⟩ afterClosed with
            ⟨finalSuffix, member, productsBalanced⟩
          exact ⟨finalSuffix, by simp [member], productsBalanced⟩
      | close _ _ checks =>
          exact ⟨_, by simp, checks.productsBalanced⟩

theorem to_closed_has_balanced_products
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {suffixes : List (ClaimSuffix Digest Challenge Products)}
    (consumes : ConsumesList balanced (.active active) suffixes
      (.closed closed)) :
    ∃ finalSuffix ∈ suffixes, balanced finalSuffix.productsAfter :=
  from_active_to_closed_has_balanced_products consumes
    ⟨active, rfl⟩ ⟨closed, rfl⟩

private theorem from_active_to_closed_segment_index
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {suffixes : List (ClaimSuffix Digest Challenge Products)}
    (consumes : ConsumesList balanced before suffixes after)
    (beforeActive : ∃ active, before = .active active)
    (afterClosed : ∃ closed, after = .closed closed) :
    ∀ active closed,
      before = .active active ->
      after = .closed closed ->
      closed.segmentIndex = active.segmentIndex + 1 := by
  induction consumes with
  | nil =>
      intro active closed beforeEqual afterEqual
      rw [beforeEqual] at afterEqual
      cases afterEqual
  | cons step rest inductionHypothesis =>
      intro active closed beforeEqual afterEqual
      cases step with
      | interior _ _ =>
          cases beforeEqual
          have restBoundary := inductionHypothesis
            ⟨_, rfl⟩ afterClosed _ closed rfl afterEqual
          simpa [interiorCarry] using restBoundary
      | close _ _ _ =>
          cases beforeEqual
          have tailExact := rest.from_closed_is_empty
          cases tailExact.2
          cases afterEqual
          rfl

theorem to_closed_segment_index
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {suffixes : List (ClaimSuffix Digest Challenge Products)}
    (consumes : ConsumesList balanced (.active active) suffixes
      (.closed closed)) :
    closed.segmentIndex = active.segmentIndex + 1 :=
  from_active_to_closed_segment_index consumes
    ⟨active, rfl⟩ ⟨closed, rfl⟩ active closed rfl rfl

end ConsumesList

/-- Exact ordered memory suffix block carried by one production fresh claim. -/
structure SuffixBatch
    (candidate : Id) (Digest Challenge Products : Type) where
  suffixes : List (ClaimSuffix Digest Challenge Products)
  length_exact :
    suffixes.length = checkedStepsPerFreshClaim candidate
deriving Repr

@[ext]
theorem SuffixBatch.ext
    {candidate : Id} {Digest Challenge Products : Type}
    {left right : SuffixBatch candidate Digest Challenge Products}
    (suffixes : left.suffixes = right.suffixes) : left = right := by
  cases left
  cases right
  cases suffixes
  rfl

/-- Types fixed by one candidate-specific verifier key. The commitment bundle
is mandatory and the memory block is part of the claim itself.

The candidate index fixes the profile. The external application statement is
not repeated in a fresh claim: HyperNova authenticates the actual SuperNeo
instance, not an unauthenticated sidecar. -/
structure Schema where
  CcsPublic : Type
  CommitmentBundle : Type
  RecursiveState : Type
  NifsProof : Type

/-- A production batch claim is a new protocol object. It cannot decode as a
factor-one V2 claim because its profile and memory field are different. -/
structure Claim
    (candidate : Id) (schema : Schema)
    (Digest Challenge Products : Type) where
  ccsPublic : schema.CcsPublic
  commitmentBundle : schema.CommitmentBundle
  recursiveState : schema.RecursiveState
  memory : SuffixBatch candidate Digest Challenge Products

abbrev Verifier
    (candidate : Id) (schema : Schema)
    (Digest Challenge Products : Type) :=
  schema.NifsProof ->
    Claim candidate schema Digest Challenge Products -> Prop

/-- One receipt for the exact complete batch claim. -/
structure Verified
    (candidate : Id) (schema : Schema)
    (Digest Challenge Products : Type)
  (verify : Verifier candidate schema Digest Challenge Products) where
  claim : Claim candidate schema Digest Challenge Products
  proof : schema.NifsProof
  accepted : verify proof claim

/-- Delayed consumption uses the suffix list inside the exact verified claim. -/
structure Transition
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    (verify : Verifier candidate schema Digest Challenge Products)
    (balanced : Products -> Prop)
    (before : Carry Digest Challenge Products)
    (verified : Verified candidate schema Digest Challenge Products verify)
    (after : Carry Digest Challenge Products) : Prop where
  consumes :
    ConsumesList balanced before verified.claim.memory.suffixes after

namespace Transition

/-- A production batch has at least one suffix. Therefore a verified batch
transition cannot start from a closed carry. -/
theorem before_active
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {verified : Verified candidate schema Digest Challenge Products verify}
    (transition : Transition verify balanced before verified after) :
    exists active, before = .active active := by
  cases before with
  | active active => exact ⟨active, rfl⟩
  | closed closed =>
      have empty := transition.consumes.from_closed_is_empty.1
      have lengthExact := verified.claim.memory.length_exact
      rw [empty] at lengthExact
      cases candidate <;>
        simp [checkedStepsPerFreshClaim] at lengthExact

theorem accepted_claim_is_consumed
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {verified : Verified candidate schema Digest Challenge Products verify}
    (transition : Transition verify balanced before verified after) :
    verify verified.proof verified.claim /\
      ConsumesList balanced before verified.claim.memory.suffixes after :=
  ⟨verified.accepted, transition.consumes⟩

theorem decreases_by_exact_factor
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {verified : Verified candidate schema Digest Challenge Products verify}
    (transition : Transition verify balanced before verified after) :
    remainingSteps before =
      checkedStepsPerFreshClaim candidate + remainingSteps after := by
  have accounting := transition.consumes.remaining_eq_length_add
  rw [verified.claim.memory.length_exact] at accounting
  exact accounting

theorem mono
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {weaker stronger : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {verified : Verified candidate schema Digest Challenge Products verify}
    (implies : forall products, weaker products -> stronger products)
    (transition : Transition verify weaker before verified after) :
    Transition verify stronger before verified after where
  consumes := transition.consumes.mono implies

end Transition

/-- A run verifies and then consumes one complete batch claim at a time. -/
inductive VerifiedRun
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    (verify : Verifier candidate schema Digest Challenge Products)
    (balanced : Products -> Prop) :
    Carry Digest Challenge Products ->
      List (Verified candidate schema Digest Challenge Products verify) ->
      Carry Digest Challenge Products -> Prop
  | nil (state : Carry Digest Challenge Products) :
      VerifiedRun verify balanced state [] state
  | cons
      {before middle after : Carry Digest Challenge Products}
      {head : Verified candidate schema Digest Challenge Products verify}
      {tail : List
        (Verified candidate schema Digest Challenge Products verify)}
      (step : Transition verify balanced before head middle)
      (rest : VerifiedRun verify balanced middle tail after) :
      VerifiedRun verify balanced before (head :: tail) after

namespace VerifiedRun

def totalSuffixCount
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    (claims : List
      (Verified candidate schema Digest Challenge Products verify)) : Nat :=
  (claims.map fun claim => claim.claim.memory.suffixes.length).sum

theorem totalSuffixCount_exact
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    (claims : List
      (Verified candidate schema Digest Challenge Products verify)) :
    totalSuffixCount claims =
      claims.length * checkedStepsPerFreshClaim candidate := by
  induction claims with
  | nil => simp [totalSuffixCount]
  | cons head tail inductionHypothesis =>
      have tailExact :
          (tail.map fun claim => claim.claim.memory.suffixes.length).sum =
            tail.length * checkedStepsPerFreshClaim candidate := by
        simpa [totalSuffixCount] using inductionHypothesis
      simp only [totalSuffixCount, List.map_cons, List.sum_cons,
        List.length_cons]
      rw [head.claim.memory.length_exact, tailExact, Nat.add_mul]
      simp [Nat.add_comm]

theorem remaining_eq_totalSuffixCount_add
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List
      (Verified candidate schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced before claims after) :
    remainingSteps before = totalSuffixCount claims + remainingSteps after := by
  induction run with
  | nil => simp [totalSuffixCount]
  | cons step _ inductionHypothesis =>
      have headAccounting := step.consumes.remaining_eq_length_add
      simp only [totalSuffixCount, List.map_cons, List.sum_cons] at inductionHypothesis ⊢
      omega

theorem remaining_eq_batch_count_add
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List
      (Verified candidate schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced before claims after) :
    remainingSteps before =
      claims.length * checkedStepsPerFreshClaim candidate +
        remainingSteps after := by
  rw [run.remaining_eq_totalSuffixCount_add, totalSuffixCount_exact]

theorem every_claim_accepted
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List
      (Verified candidate schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced before claims after) :
    forall claim, claim ∈ claims -> verify claim.proof claim.claim := by
  induction run with
  | nil => simp
  | cons step _ inductionHypothesis =>
      intro claim member
      simp only [List.mem_cons] at member
      rcases member with equal | tailMember
      · subst claim
        exact step.accepted_claim_is_consumed.1
      · exact inductionHypothesis claim tailMember

/-- Forget the outer claim grouping only after each whole claim has been
verified. The flattened deterministic run keeps every suffix in exact order. -/
theorem flattenConsumes
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List
      (Verified candidate schema Digest Challenge Products verify)}
    (run : VerifiedRun verify balanced before claims after) :
    ConsumesList balanced before
      (claims.flatMap fun claim => claim.claim.memory.suffixes) after := by
  induction run with
  | nil => exact .nil _
  | cons step _ inductionHypothesis =>
      simpa using step.consumes.append inductionHypothesis

/-- A complete segment starting at step zero uses exactly `1088 / E` verified
batch claims. The trailing batch must be consumed before the state is closed. -/
theorem full_segment_has_exact_batch_count
    {candidate : Id} {schema : Schema}
    {Digest Challenge Products : Type}
    {verify : Verifier candidate schema Digest Challenge Products}
    {balanced : Products -> Prop}
    {active : ActiveCarry Digest Challenge Products}
    {closed : ClosedCarry Digest}
    {claims : List
      (Verified candidate schema Digest Challenge Products verify)}
    (startsAtZero : active.stepIndex.val = 0)
    (run : VerifiedRun verify balanced (.active active) claims
      (.closed closed)) :
    claims.length = claimsPerSegment candidate := by
  have accounting := run.remaining_eq_batch_count_add
  simp only [remainingSteps, Nat.add_zero] at accounting
  rw [startsAtZero, Nat.sub_zero] at accounting
  change 1088 =
    claims.length * checkedStepsPerFreshClaim candidate at accounting
  cases candidate <;>
    simp [ProductionProfileCandidates.claimsPerSegment,
      checkedStepsPerFreshClaim,
      ProductionProfileCandidates.stepsPerSegment] at accounting ⊢ <;>
    omega

end VerifiedRun

end Nightstream.Protocol.Nebula.ProductionBatchedFPrime
