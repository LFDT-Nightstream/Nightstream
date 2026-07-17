import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.SemanticHandoff
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.Batch

/-!
Terminal sampler refinement rooted at the exact typed post-`Pi_CCS` handoff.

Assurance tier: conditional implementation/R1CS refinement. This module
composes the independent output-message handoff with the exact bounded
15-challenge batch already reconstructed from accepted terminal rows.

Owns: substitution of the artifact sampler-entry state by the pure
post-NC/message-derived handoff state; and one complete `ConcretePhi81`
sampler bound over the decoded production challenge vector.

Does not own: the upstream output-message and minimal post-NC boundary
premises; fixed-profile
instantiation for the production F-prime shape; finite SIS sampler acceptance;
Rust ChaCha/Poseidon2 conformance; challenge algebra after sampling; rows,
costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: the theorem never accepts the artifact initial state as
semantic authority. Accepted rows first refine `SemanticHandoff.run`; only
that computed state is then used as the root of the exact bounded sampler.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest.handoff` | typed post-NC state and complete output message determine the sampler root | conditional refinement | `PiCcsOutputDigest.SemanticHandoff.accepted_refines_run` |
| `nifs.pi_rlc.challenge.batch` | all 15 decoded challenges come from one exact bounded batch | artifact-checked refinement | `Batch.accepted_refines_initialStateBound` |
| `nifs.pi_rlc.challenge.semantic_handoff` | the bounded batch starts at the typed computed handoff | derived composition | `accepted_refines_semanticHandoffBound` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Accepted terminal rows provide the exact concrete Phi81 sampler bound at
the independently computed post-`Pi_CCS` handoff state. -/
theorem accepted_refines_semanticHandoffBound
    (prime : EuclidPrime goldilocksP)
    {shape : SemanticShape}
    (profile : PiCcsOutputDigest.Projection.SplitNc.Profile shape)
    (postNc : State)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputAccepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment)
    (catchupAccepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment)
    (rlcAccepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (messageBound :
      PiCcsOutputDigest.SemanticHandoff.MessageBound
        profile message assignment canonical)
    (postNcBoundary :
      PiCcsOutputDigest.SemanticHandoff.CatchupInputBound
        postNc assignment canonical) :
    Nonempty
      (ConcretePhi81.Sampler.Bound machine
        (PiCcsOutputDigest.SemanticHandoff.run profile postNc message)
        (fun rho => RingAssembly.decodedChallenge assignment rho)) := by
  have stateEq :=
    PiCcsOutputDigest.SemanticHandoff.accepted_refines_run
      prime profile postNc message canonical one outputAccepted catchupAccepted
      rlcAccepted messageBound postNcBoundary
  have bounded :=
    Batch.accepted_refines_initialStateBound prime canonical one rlcAccepted
  rw [stateEq]
  exact bounded

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff
