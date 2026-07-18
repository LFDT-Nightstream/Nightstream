import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.TerminalSamplerArtifact
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff

/-!
Terminal R1CS sampler refinement into the exact concrete Phi81 NIFS
certificate predicate.

Assurance tier: artifact-checked diagnostic profile, conditional on the
explicitly listed upstream message and post-NC bindings. This module binds the
reviewed three-matrix terminal artifact and decoded challenge columns to the
independent `ConcretePhi81.Sampler.CertificateAccepted` boundary.

Owns: the exact reviewed terminal-artifact arity; context equality for the
semantic output handoff and transcript machine; canonical construction of the
certificate challenge field from decoded transcript output; and construction
of the context/certificate-specific sampler acceptance proof.

Does not own: Split-NC output-column authority; post-NC catch-up-input
authority; fixed-profile proof for a concrete F-prime shape; finite SIS
sampler acceptance; Rust ChaCha/Poseidon2 conformance; the PiRLC equations,
PiDEC, F-prime, rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `ContextBinding` identifies verifier-owned functions, not
prover data. `accepted_refines_certificateAccepted` binds the certificate's
challenge vector to the exact decoded sampler output before constructing
semantic acceptance.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest.profile.sources` | terminal context alignment derives exactly 15 sources | derived | `terminalProfile` |
| `nifs.pi_ccs.output_digest.profile.matrices` | actual F-prime relation has exactly three matrices | explicit remaining premise | `terminalProfile` |
| `nifs.pi_rlc.challenge.context.handoff` | schedule output absorption is the pure typed post-`Pi_CCS` handoff | checked refinement | `ContextBinding.outputHandoff` |
| `nifs.pi_rlc.challenge.context.post_nc` | derive the exact state after NC and before output absorption | computed | `postNcState` |
| `nifs.pi_rlc.challenge.context.machine` | context uses the exact production transcript machine | checked refinement | `ContextBinding.samplerMachine` |
| `nifs.pi_rlc.challenge.certificate` | construct the challenge field from the decoded terminal sampler vector | computed | `withDecodedChallenges` |
| `nifs.pi_rlc.challenge.columns` | constructed challenge field equals the RingF interpretation of the equation columns | derived | `withDecodedChallenges_challenge_eq_columns` |
| `nifs.pi_rlc.challenge.acceptance` | accepted rows construct `ConcretePhi81.Sampler.CertificateAccepted` | derived composition | `accepted_refines_withDecodedChallenges` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Terminal source count is derived from the context's partition alignment.
Only the reviewed diagnostic artifact's three-matrix fact remains an explicit
profile premise. The active selective relation has thirteen matrices; no
`13 -> 3` compression theorem is claimed here. -/
def terminalProfile
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows terminalArity)
    (matrixCount_eq :
      shape.matrixCount = PiCcsOutputDigest.Semantics.yRingRows) :
    PiCcsOutputDigest.Projection.SplitNc.Profile shape :=
  PiCcsOutputDigest.Projection.SplitNc.Profile.ofAlignment
    context.alignment (by rfl) matrixCount_eq

/-- Exact verifier-owned function bindings required by the terminal concrete
NIFS context. -/
structure ContextBinding
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : PiCcsOutputDigest.Projection.SplitNc.Profile shape)
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows terminalArity) : Prop where
  outputHandoff :
    ∀ postNc message,
      context.piCcsSchedule.absorbOutput postNc message =
        PiCcsOutputDigest.SemanticHandoff.run profile postNc message
  samplerMachine :
    context.piRlcMachine = machine

/-- Exact transcript state after canonical block×lane NC and before the output
message is absorbed. -/
def postNcState
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows terminalArity)
    (certificate :
      ConcretePhi81.Certificate (arity := terminalArity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : State :=
  let feExecution :=
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive
      context.feMachine context.initialState certificate.piCcs.fe
  (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.derive
    context.ncMachine feExecution.finalState certificate.piCcs.nc).finalState

/-- Terminal challenge vector in the exact NIFS source index, reusing the
canonical arity-to-sampler transport owned by `TerminalSamplerArtifact`. -/
def decodedChallenges
    (assignment : Nat -> Nat) :
    Fin terminalArity.total -> RingF :=
  fun index =>
    RingAssembly.decodedChallenge assignment
      (TerminalSamplerArtifact.scalarIndex index)

/-- Replace the prover-shaped challenge field by the exact vector decoded
from the transcript rows. The other raw phase messages are preserved. -/
def withDecodedChallenges
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows terminalArity}
    (assignment : Nat -> Nat)
    (certificate :
      ConcretePhi81.Certificate (arity := terminalArity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    ConcretePhi81.Certificate (arity := terminalArity)
      publicRingColumns publicFits verifierRows context.piCcsInput :=
  { certificate with piRlcChallenges := decodedChallenges assignment }

@[simp] theorem withDecodedChallenges_piRlcChallenges
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows terminalArity}
    (assignment : Nat -> Nat)
    (certificate :
      ConcretePhi81.Certificate (arity := terminalArity)
        publicRingColumns publicFits verifierRows context.piCcsInput)
    (index : Fin terminalArity.total) :
    (withDecodedChallenges assignment certificate).piRlcChallenges index =
      decodedChallenges assignment index := by
  rfl

/-- The constructed semantic challenge is exactly the RingF interpretation
of the challenge columns consumed by every public projection equation. -/
theorem withDecodedChallenges_challenge_eq_columns
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows terminalArity}
    (assignment : Nat -> Nat)
    (certificate :
      ConcretePhi81.Certificate (arity := terminalArity)
        publicRingColumns publicFits verifierRows context.piCcsInput)
    (index : Fin terminalArity.total) :
    (withDecodedChallenges assignment certificate).piRlcChallenges index =
      ringOfList (values assignment
        (TerminalSamplerArtifact.challengeColumns index)) := by
  change
    RingAssembly.decodedChallenge assignment
        (TerminalSamplerArtifact.scalarIndex index) =
      ringOfList (values assignment
        (TerminalSamplerArtifact.challengeColumns index))
  rw [TerminalSamplerArtifact.values_challengeColumns_eq_decoded]
  exact (ProductionRingAlgebra.ringOfList_canonicalRing _).symm

/-- Accepted terminal rows instantiate the exact context-bound sampler
predicate consumed by the independent concrete NIFS verifier. -/
theorem accepted_refines_certificateAccepted
    (prime : EuclidPrime goldilocksP)
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows terminalArity}
    {certificate :
      ConcretePhi81.Certificate (arity := terminalArity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (profile : PiCcsOutputDigest.Projection.SplitNc.Profile shape)
    (contextBinding : ContextBinding profile context)
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
        profile certificate.piCcs.output assignment canonical)
    (postNcBoundary :
      PiCcsOutputDigest.SemanticHandoff.CatchupInputBound
        (postNcState context certificate)
        assignment canonical)
    (challengesBound :
      certificate.piRlcChallenges = decodedChallenges assignment) :
    ConcretePhi81.Sampler.CertificateAccepted context certificate := by
  have bounded :=
    SemanticHandoff.accepted_refines_semanticHandoffBound
      prime profile
      (postNcState context certificate)
      certificate.piCcs.output canonical one outputAccepted catchupAccepted
      rlcAccepted messageBound postNcBoundary
  rcases bounded with ⟨bound⟩
  refine ⟨?_⟩
  change
    ConcretePhi81.Sampler.Bound context.piRlcMachine
      (ConcretePhi81.derive context certificate).piRlcInitialState
      certificate.piRlcChallenges
  rw [ConcretePhi81.derive_piRlcInitialState]
  change
    ConcretePhi81.Sampler.Bound context.piRlcMachine
      (context.piCcsSchedule.absorbOutput
        (postNcState context certificate) certificate.piCcs.output)
      certificate.piRlcChallenges
  rw [contextBinding.outputHandoff, contextBinding.samplerMachine,
    challengesBound]
  simpa [decodedChallenges, TerminalSamplerArtifact.scalarIndex,
    TerminalSamplerArtifact.terminalTotal_eq_scalarCount] using bound

/-- Accepted rows refine the sampler predicate for the certificate whose
challenge field is verifier-constructed from those rows. No independent
certificate-to-challenge equality premise remains. -/
theorem accepted_refines_withDecodedChallenges
    (prime : EuclidPrime goldilocksP)
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows terminalArity}
    (certificate :
      ConcretePhi81.Certificate (arity := terminalArity)
        publicRingColumns publicFits verifierRows context.piCcsInput)
    (profile : PiCcsOutputDigest.Projection.SplitNc.Profile shape)
    (contextBinding : ContextBinding profile context)
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
        profile certificate.piCcs.output assignment canonical)
    (postNcBoundary :
      PiCcsOutputDigest.SemanticHandoff.CatchupInputBound
        (postNcState context certificate)
        assignment canonical) :
    ConcretePhi81.Sampler.CertificateAccepted context
      (withDecodedChallenges assignment certificate) := by
  exact accepted_refines_certificateAccepted
    prime profile contextBinding canonical one outputAccepted catchupAccepted
    rlcAccepted (by simpa [withDecodedChallenges] using messageBound)
    (by simpa [withDecodedChallenges] using postNcBoundary) rfl

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate
