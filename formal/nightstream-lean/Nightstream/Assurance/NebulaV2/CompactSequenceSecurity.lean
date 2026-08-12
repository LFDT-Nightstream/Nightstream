import Nightstream.Protocol.NebulaV2.CompactChain
import Nightstream.Protocol.NebulaV2.IdealAcceptance

/-!
Contract: specialize the ideal V2 sequence-failure branch to the exact
fixed-length compact commitment chain.

Assurance tier: deterministic cryptographic-reduction boundary.

Owns the reduction from an ideal sequence-root collision to one typed
Poseidon2 collision or one exact primary/short Ajtai binding failure.

Does not assume collision resistance, Module-SIS hardness, generated-row
refinement, Rust conformance, Fiat--Shamir security, or probability bounds.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.NebulaV2.CompactSequenceSecurity

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CompactChain
open Nightstream.Protocol.NebulaV2.CompactCommit
open Nightstream.Protocol.NebulaV2.IdealAcceptance
open Nightstream.Protocol.NebulaV2.Soundness

/-- The generic ideal failure is retained only when it is a fingerprint or
snapshot-root failure. The generic sequence branch is excluded here because
this layer replaces it with exact compact-chain failures. -/
inductive NonSequenceFailure
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest) :
    Failure config → Prop where
  | fingerprint
      {schema : FullClaim.Schema}
      {bundleComponent :
        schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
      {verify : FullVerifier schema Digest ChallengeField}
      {segmentIndex timestampIn timestampOut : Nat}
      {initial final : Snapshot} {accesses : List Access}
      (segment : SegmentCheck config schema bundleComponent verify segmentIndex
        initial timestampIn accesses final timestampOut)
      (failure : IdealFingerprint.EvaluationFailure segment.fingerprint)
      (sequencesExact : segment.sequences.Exact) :
      NonSequenceFailure config
        (.fingerprint segment failure sequencesExact)
  | snapshotRoot (collision : SnapshotRootCollision config.snapshotRoot) :
      NonSequenceFailure config (.snapshotRoot collision)

/-- Release-level deterministic failures after the exact compact chain is
selected. Each constructor names one independent condition. -/
inductive ReleaseFailure
    {ChallengeField Plan Seed Digest : Type}
    [Field ChallengeField]
    (config :
      Config ChallengeField Profile.Identity Plan CommitmentEncoding Digest)
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed) : Prop where
  | nonSequence
      {failure : Failure config}
      (kind : NonSequenceFailure config failure) :
      ReleaseFailure config hash key
  | poseidon (collision : HashCollision hash) :
      ReleaseFailure config hash key
  | primary (failure : AnyPrimaryBindingFailure key) :
      ReleaseFailure config hash key
  | short (failure : AnyShortBindingFailure key) :
      ReleaseFailure config hash key

/-- A generic ideal failure is classified without a cryptographic
assumption. The only required equality says that the selected configuration
uses the exact compact-chain function. -/
theorem classify_failure
    {ChallengeField Plan Seed Digest : Type}
    [Field ChallengeField]
    {config :
      Config ChallengeField Profile.Identity Plan CommitmentEncoding Digest}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (chainRootExact : config.chainRoot = chainRoot hash key)
    (failure : Failure config) :
    ReleaseFailure config hash key := by
  cases failure with
  | fingerprint segment evaluationFailure sequencesExact =>
      exact .nonSequence
        (failure := .fingerprint segment evaluationFailure sequencesExact)
        (.fingerprint segment evaluationFailure sequencesExact)
  | snapshotRoot collision =>
      exact .nonSequence (failure := .snapshotRoot collision)
        (.snapshotRoot collision)
  | sequence collision =>
      rw [chainRootExact] at collision
      rcases root_collision_implies_hash_or_ajtai_failure hash key collision with
        hashCollision | primaryFailure | shortFailure
      · exact .poseidon hashCollision
      · exact .primary primaryFailure
      · exact .short shortFailure

/-- Ideal acceptance with the exact compact chain gives a certified execution
or a release-level named failure. This theorem does not take a manufactured
execution witness or an assumed final soundness conclusion. -/
theorem compact_acceptance_implies_execution_or_release_failure
    {ChallengeField Plan Seed Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config :
      Config ChallengeField Profile.Identity Plan CommitmentEncoding Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component →
        CommitmentEncoding}
    {Program ApplicationState : Type}
    {applicationSemantics :
      ApplicationTrace.Semantics Program ApplicationState}
    {statement : PublicStatement Program ApplicationState Digest}
    {verify : FullVerifier schema Digest ChallengeField}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (chainRootExact : config.chainRoot = chainRoot hash key)
    (acceptance :
      IdealAcceptV2 config schema bundleComponent verify applicationSemantics
        statement) :
    ReleaseFailure config hash key ∨ CertifiedExecution acceptance := by
  rcases ideal_acceptance_implies_execution_or_failure acceptance with
    failure | execution
  · exact Or.inl (classify_failure hash key chainRootExact failure)
  · exact Or.inr execution

end Nightstream.Assurance.NebulaV2.CompactSequenceSecurity
