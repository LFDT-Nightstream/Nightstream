import Nightstream.Protocol.Nebula.CompactCommit
import Nightstream.Protocol.Nebula.SequenceBinding

/-!
Contract: exact typed V2 lane chain over compact commitment tokens.

Assurance tier: model-level and cryptographic-reduction boundary.

Owns the header, leaf, and indexed-link input types; the exact fixed-length
chain function; and the deterministic reduction from a chain-root collision
to either a typed hash collision or a compact-token collision.

Does not own Poseidon2 collision resistance, Ajtai/Module-SIS hardness,
concrete field serialization, generated rows, Rust, or probability bounds.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.CompactChain

open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.SequenceBinding

def roleOfDomain : LaneDomain → Role
  | .operations => .operations
  | .memory => .memory

theorem roleOfDomain_injective : Function.Injective roleOfDomain := by
  intro left right equal
  cases left <;> cases right <;> simp_all [roleOfDomain]

/-- Typed inputs prevent an encoding-free proof from treating a digest as
authority. The implementation must prove that its canonical Poseidon2 codec
is injective into this type or expose a decode/framing failure. -/
inductive HashInput (Plan Digest : Type) where
  | header
      (role : Role) (profile : Profile.Identity) (plan : Plan)
  | leaf
      (role : Role) (profile : Profile.Identity) (plan : Plan) (token : Token)
  | link (role : Role) (index : Fin claimsPerSegment) (prior leaf : Digest)

def next
    {Plan Seed Digest : Type}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (role : Role) (profile : Profile.Identity) (plan : Plan)
    (index : Fin claimsPerSegment) (prior : Digest)
    (commitment : CommitmentEncoding) : Digest :=
  hash (.link role index prior
    (hash (.leaf role profile plan (key.token role commitment))))

def run
    {Plan Seed Digest : Type}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (role : Role) (profile : Profile.Identity) (plan : Plan) :
    Digest → List (Fin claimsPerSegment × CommitmentEncoding) → Digest
  | prior, [] => prior
  | prior, indexedCommitment :: rest =>
      run hash key role profile plan
        (next hash key role profile plan indexedCommitment.1 prior
          indexedCommitment.2) rest

def chainRoot
    {Plan Seed Digest : Type}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed) :
    FramedSequence Profile.Identity Plan CommitmentEncoding → Digest :=
  fun sequence =>
    let role := roleOfDomain sequence.domain
    let header :=
      hash (.header role sequence.profile sequence.plan)
    let indexedCommitments := List.ofFn fun index =>
      (index, sequence.commitments index)
    run hash key role sequence.profile sequence.plan header indexedCommitments

def HashCollision
    {Plan Digest : Type} (hash : HashInput Plan Digest → Digest) : Prop :=
  ∃ left right, left ≠ right ∧ hash left = hash right

def TokenCollision
    {Plan Seed : Type} (key : Key Plan Seed) : Prop :=
  ∃ role left right,
    left ≠ right ∧ key.token role left = key.token role right

def AnyPrimaryBindingFailure
    {Plan Seed : Type} (key : Key Plan Seed) : Prop :=
  ∃ role, PrimaryBindingFailure key role

def AnyShortBindingFailure
    {Plan Seed : Type} (key : Key Plan Seed) : Prop :=
  ∃ role, ShortBindingFailure key role

theorem hash_injective_or_collision
    {Plan Digest : Type} (hash : HashInput Plan Digest → Digest) :
    Function.Injective hash ∨ HashCollision hash := by
  classical
  by_cases injective : Function.Injective hash
  · exact Or.inl injective
  · rcases Function.not_injective_iff.mp injective with
      ⟨left, right, equalHash, different⟩
    exact Or.inr ⟨left, right, different, equalHash⟩

theorem tokens_injective_or_collision
    {Plan Seed : Type} (key : Key Plan Seed) :
    (∀ role, Function.Injective (key.token role)) ∨ TokenCollision key := by
  classical
  by_cases injective : ∀ role, Function.Injective (key.token role)
  · exact Or.inl injective
  · push Not at injective
    rcases injective with ⟨role, notInjective⟩
    rcases Function.not_injective_iff.mp notInjective with
      ⟨left, right, equalToken, different⟩
    exact Or.inr ⟨role, left, right, different, equalToken⟩

private theorem run_injective_of_length_eq
    {Plan Seed Digest : Type}
    {hash : HashInput Plan Digest → Digest}
    {key : Key Plan Seed}
    (hashInjective : Function.Injective hash)
    (tokenInjective : ∀ role, Function.Injective (key.token role)) :
    ∀ {leftRole rightRole : Role}
      {leftProfile rightProfile : Profile.Identity}
      {leftPlan rightPlan : Plan}
      {leftPrior rightPrior : Digest}
      {left right : List (Fin claimsPerSegment × CommitmentEncoding)},
      left.length = right.length →
      run hash key leftRole leftProfile leftPlan leftPrior left =
        run hash key rightRole rightProfile rightPlan rightPrior right →
      leftPrior = rightPrior ∧ left = right ∧
        (left ≠ [] →
          leftRole = rightRole ∧ leftProfile = rightProfile ∧
            leftPlan = rightPlan) := by
  intro leftRole rightRole leftProfile rightProfile leftPlan rightPlan
    leftPrior rightPrior left right
  induction left generalizing right leftRole rightRole leftProfile rightProfile
      leftPlan rightPlan leftPrior rightPrior with
  | nil =>
      intro equalLength equalRun
      cases right with
      | nil => exact ⟨equalRun, rfl, by simp⟩
      | cons _ _ => simp at equalLength
  | cons leftHead leftTail inductionHypothesis =>
      intro equalLength equalRun
      cases right with
      | nil => simp at equalLength
      | cons rightHead rightTail =>
          simp only [List.length_cons, Nat.succ.injEq] at equalLength
          simp only [run] at equalRun
          have recursive :=
            inductionHypothesis
              (leftRole := leftRole) (rightRole := rightRole)
              (leftProfile := leftProfile) (rightProfile := rightProfile)
              (leftPlan := leftPlan) (rightPlan := rightPlan)
              (leftPrior :=
                next hash key leftRole leftProfile leftPlan leftHead.1
                  leftPrior leftHead.2)
              (rightPrior :=
                next hash key rightRole rightProfile rightPlan rightHead.1
                  rightPrior rightHead.2)
              equalLength equalRun
          rcases recursive with
            ⟨nextEqual, tailEqual, _tailContext⟩
          have linkEqual := hashInjective nextEqual
          injection linkEqual with
            roleEqual indexEqual priorEqual leafHashEqual
          have leafInputEqual := hashInjective leafHashEqual
          injection leafInputEqual with
            _leafRoleEqual profileEqual planEqual tokenEqual
          cases roleEqual
          cases profileEqual
          cases planEqual
          have commitmentEqual := tokenInjective leftRole tokenEqual
          have headEqual : leftHead = rightHead := by
            apply Prod.ext
            · exact indexEqual
            · exact commitmentEqual
          exact
            ⟨priorEqual, by rw [headEqual, tailEqual],
              fun _ => ⟨rfl, rfl, rfl⟩⟩

private theorem indexed_claims_nonempty
    (commitments : Fin claimsPerSegment → CommitmentEncoding) :
    (List.ofFn fun index => (index, commitments index)) ≠ [] := by
  intro empty
  have lengthEqual := congrArg List.length empty
  have zero : claimsPerSegment = 0 := by
    simpa only [List.length_ofFn, List.length_nil] using lengthEqual
  unfold claimsPerSegment at zero
  omega

theorem chainRoot_injective
    {Plan Seed Digest : Type}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (hashInjective : Function.Injective hash)
    (tokenInjective : ∀ role, Function.Injective (key.token role)) :
    Function.Injective (chainRoot hash key) := by
  intro left right equalRoot
  have equalLengths :
      (List.ofFn fun index => (index, left.commitments index)).length =
        (List.ofFn fun index => (index, right.commitments index)).length := by
    simp
  change
    run hash key (roleOfDomain left.domain) left.profile left.plan
        (hash (.header (roleOfDomain left.domain) left.profile left.plan))
        (List.ofFn fun index => (index, left.commitments index)) =
      run hash key (roleOfDomain right.domain) right.profile right.plan
        (hash (.header (roleOfDomain right.domain) right.profile right.plan))
        (List.ofFn fun index => (index, right.commitments index)) at equalRoot
  have recovered :=
    run_injective_of_length_eq hashInjective tokenInjective
      (leftRole := roleOfDomain left.domain)
      (rightRole := roleOfDomain right.domain)
      (leftProfile := left.profile) (rightProfile := right.profile)
      (leftPlan := left.plan) (rightPlan := right.plan)
      (leftPrior :=
        hash (.header (roleOfDomain left.domain) left.profile left.plan))
      (rightPrior :=
        hash (.header (roleOfDomain right.domain) right.profile right.plan))
      (left := List.ofFn fun index => (index, left.commitments index))
      (right := List.ofFn fun index => (index, right.commitments index))
      equalLengths equalRoot
  rcases recovered with
    ⟨headerEqual, commitmentsEqual, contextEqual⟩
  have contexts := contextEqual (indexed_claims_nonempty left.commitments)
  rcases contexts with ⟨roleEqual, profileEqual, planEqual⟩
  have domainEqual : left.domain = right.domain :=
    roleOfDomain_injective roleEqual
  have valuesEqual := congrArg (List.map Prod.snd) commitmentsEqual
  have commitmentListsEqual :
      List.ofFn left.commitments = List.ofFn right.commitments := by
    simpa using valuesEqual
  have commitmentFunctionsEqual : left.commitments = right.commitments :=
    List.ofFn_injective commitmentListsEqual
  apply FramedSequence.ext
  · exact profileEqual
  · exact planEqual
  · exact domainEqual
  · exact commitmentFunctionsEqual

/-- A collision in the complete fixed-length framed sequence has only the two
named cryptographic causes. -/
theorem root_collision_implies_hash_or_token_collision
    {Plan Seed Digest : Type}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (collision : RootCollision (chainRoot hash key)) :
    HashCollision hash ∨ TokenCollision key := by
  rcases hash_injective_or_collision hash with hashInjective | hashCollision
  · rcases tokens_injective_or_collision key with
      tokenInjective | tokenCollision
    · rcases collision with ⟨left, right, different, equalRoot⟩
      exact False.elim
        (different (chainRoot_injective hash key hashInjective tokenInjective
          equalRoot))
    · exact Or.inr tokenCollision
  · exact Or.inl hashCollision

/-- Release-form reduction: a distinct framed commitment sequence with the
same root gives a typed Poseidon2 collision or one of the two exact Ajtai
binding failures. -/
theorem root_collision_implies_hash_or_ajtai_failure
    {Plan Seed Digest : Type}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (collision : RootCollision (chainRoot hash key)) :
    HashCollision hash ∨
      AnyPrimaryBindingFailure key ∨ AnyShortBindingFailure key := by
  rcases root_collision_implies_hash_or_token_collision hash key collision with
    hashCollision | tokenCollision
  · exact Or.inl hashCollision
  · rcases tokenCollision with
      ⟨role, left, right, different, equalToken⟩
    rcases token_collision_implies_primary_or_short_failure
        key role different equalToken with primaryFailure | shortFailure
    · exact Or.inr (Or.inl ⟨role, primaryFailure⟩)
    · exact Or.inr (Or.inr ⟨role, shortFailure⟩)

end Nightstream.Protocol.Nebula.CompactChain
