import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.NebulaV2.Core.ConcreteField
import Nightstream.Implementation.NebulaV2.FPrime.Claim.NifsReceipt
import Nightstream.Implementation.NebulaV2.Memory.Transition.OpenSegment
import Nightstream.Protocol.NebulaV2.AugmentedLifecycle

/-!
Contract: fixed-profile V2 reference specialization of the exact augmented
lifecycle.

Assurance tier: implementation model.

Owns one fixed authority schedule and specializes every base and boundary
opening to the exact V2 Poseidon2 challenge function. It also uses the
complete claim selected by the exact V2 NIFS receipt interface.

This module is not part of the field-native candidate lifetime. Its selected
verifier has a proof field `profileExact : profile = Profile.v2`. Candidate
versions 3 through 6 use the profile-indexed
`ProductionPaperExactLifetime` path instead.

Does not prove that the authority schedule is connected to generated source
columns, that NIFS is extractable, or that the generated recursive relation
implements this lifecycle.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionAugmentedLifecycle

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

noncomputable local instance concreteKField : Field K :=
  ConcreteField.superNeoEquiv.field

/-- The enclosing generated relation must compute this function from its
verifier key, application statement, exact intermediate state, and post-fold
accumulator. It is fixed for one verifier and is not carried in a proof. -/
abbrev AuthoritySchedule :=
  ClosedCarry Digest.Value → Roots Digest.Value → Nat →
    MemoryOpenSegment.Authority

/-- The only fixed-V2 challenge derivation admitted by this reference
lifecycle. -/
def derive (authority : AuthoritySchedule) :
    ClosedCarry Digest.Value → Roots Digest.Value → Nat → Challenges K :=
  fun closed precommit activeAccessCount =>
    MemoryOpenSegment.derive
      (authority closed precommit activeAccessCount)
      closed precommit activeAccessCount

theorem derive_exact
    (authority : AuthoritySchedule)
    (closed : ClosedCarry Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat) :
    derive authority closed precommit activeAccessCount =
      MemoryTranscriptPoseidonRows.pureChallenges
        (MemoryOpenSegment.transcriptInput
          (authority closed precommit activeAccessCount)
          closed precommit activeAccessCount) :=
  rfl

/-- One fixed-V2 full-claim run. Arbitrary or constant challenge functions
cannot inhabit this type unless they are equal to the exact Poseidon2 result
for every opening input. -/
structure Run
    {widths : CompilerWidths}
    (selected : SelectedVerifier widths)
    (authority : AuthoritySchedule)
    (headers : ChainHeaders Digest.Value)
    (initial final : ClosedCarry Digest.Value) where
  model : AugmentedLifecycle.CompleteRun
    (schema := protocolSchema widths (PackedProof selected))
    (VerifyClaim selected) (derive authority) headers initial final

namespace Run

theorem claims_nonempty
    {widths : CompilerWidths}
    {selected : SelectedVerifier widths}
    {authority : AuthoritySchedule}
    {headers : ChainHeaders Digest.Value}
    {initial final : ClosedCarry Digest.Value}
    (run : Run selected authority headers initial final) :
    run.model.claims ≠ [] :=
  run.model.claims_nonempty

theorem augmented_invocations_exact
    {widths : CompilerWidths}
    {selected : SelectedVerifier widths}
    {authority : AuthoritySchedule}
    {headers : ChainHeaders Digest.Value}
    {initial final : ClosedCarry Digest.Value}
    (run : Run selected authority headers initial final) :
    1 + run.model.claims.length = run.model.claims.length + 1 :=
  run.model.augmented_invocation_count

/-- The base challenge is the exact transcript output for its exact authority,
closed carry, precommit roots, and declared count. -/
theorem base_challenge_exact
    {widths : CompilerWidths}
    {selected : SelectedVerifier widths}
    {authority : AuthoritySchedule}
    {headers : ChainHeaders Digest.Value}
    {initial final : ClosedCarry Digest.Value}
    (run : Run selected authority headers initial final) :
    run.model.baseActive.challenge =
      MemoryTranscriptPoseidonRows.pureChallenges
        (MemoryOpenSegment.transcriptInput
          (authority initial run.model.basePrecommit
            run.model.baseActiveAccessCount)
          initial run.model.basePrecommit run.model.baseActiveAccessCount) := by
  have opened := Carry.active.inj run.model.baseOpened
  have challenge := congrArg
    (fun active => active.challenge) opened
  simpa [derive, openSegment] using challenge.symm

end Run

end Nightstream.Implementation.NebulaV2.ProductionAugmentedLifecycle
