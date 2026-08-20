import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayFinalArtifact

/-!
Contract: authoritative lifecycle binding for the final prior-state replay
target.

Owns the typed equality between the four final replay target columns and the
prior-state digest recomputed from the exact active running instance. It then
combines that equality with the exact retained final-arm rows.

The `TargetLink` premise is the precise remaining selective-artifact
obligation. This module does not assume that current source rows emit it and
does not treat a digest as payload authority.

Assurance tier: model-level plus an explicit artifact-link boundary.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayLifecycleBridge

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayFinalArtifact

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

/-- The replay target is authoritative only when it is the canonical field
view of the digest carried by the common lifecycle envelope. -/
def TargetAuthorized
    (envelope :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.PublicEnvelope)
    (target :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayRelation.Digest) :
    Prop :=
  exists digest :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Digest,
    envelope.beforePriorStateDigest = some digest /\
      target = digestValues digest

/-- Exact four-lane equality that the final selective relation must emit.
The left side is the reviewed Rust replay source slice. The right side is
recomputed from the complete active running instance. -/
def TargetLink
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (recursive : Recursive configuration) (assignment : Nat -> Nat) : Prop :=
  forall lane : Fin 4,
    assignment (155 + lane.val) =
      (configuration.runningPriorStateDigest recursive.running lane).val

theorem target_authorized_of_link
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (recursive : Recursive configuration) (assignment : Nat -> Nat)
    (link : TargetLink recursive assignment) :
    TargetAuthorized recursive.commonPublic (targetDigestAt assignment) := by
  refine ⟨configuration.runningPriorStateDigest recursive.running,
    Recursive.before_prior_state_digest_exact recursive, ?_⟩
  funext lane
  exact link lane

/-- Exact retained final-arm rows plus the four missing selective link rows
give both the replay check and its verifier-owned lifecycle authority. -/
theorem final_rows_imply_authoritative_finalChecks
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (recursive : Recursive configuration)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment)
    (link : TargetLink recursive assignment) :
    FinalChecks .final (replayStateAt assignment 11)
        (finalChunk assignment) (targetDigestAt assignment) /\
      TargetAuthorized recursive.commonPublic (targetDigestAt assignment) := by
  exact ⟨final_rows_imply_finalChecks assignment canonical one satisfied,
    target_authorized_of_link recursive assignment link⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayLifecycleBridge
