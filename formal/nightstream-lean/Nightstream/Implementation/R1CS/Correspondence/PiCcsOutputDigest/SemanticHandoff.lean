import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Handoff
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Projection.SplitNc
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Primitives
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.PostNcBoundary

/-!
Exact semantic handoff from the historical three-matrix Split-NC `Pi_CCS`
output message to its diagnostic `Pi_RLC` sampler state. This module fixes
6,683 serialized fields and cannot discharge the active 23,033-field handoff.

Assurance tier: conditional implementation/R1CS refinement. The pure handoff
is independently executable from a post-NC transcript state and the complete
typed Split-NC output message. The row-refinement theorem keeps the two
remaining upstream authority equalities explicit.

Owns: diagnostic-profile message serialization; the two exact artifact SIS map
applications; the isolated Poseidon2 digest; verifier catch-up; output-digest
absorption; and composition with accepted terminal owner rows.

Does not own: proof that the active F-prime shape satisfies this diagnostic
profile (it does not); equality of accepted output columns with the semantic Split-NC
message; equality of the catch-up input columns with the semantic post-NC
state; finite SIS sampler acceptance; Rust ChaCha/Poseidon2 conformance;
collision resistance; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: `run` recomputes the digest from the complete typed
message. `accepted_refines_run` does not promote decoded columns or a carried
state to authority: it requires exact `messageBound` and the minimal
post-NC observable boundary, then proves that accepted rows reach the same
sampler state.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest.serialize` | serialize all 15 projected messages in exact field order | computed | `serializedValues` |
| `nifs.pi_ccs.output_digest.sis` | apply the exact rank-2 then rank-1 production maps | computed/profile-specific | `digestValue` |
| `nifs.pi_ccs.output_digest.poseidon2` | recompute four digest fields from the 64-field envelope | computed/profile-specific | `digest` |
| `nifs.pi_ccs.nc.post_state.cursor` | carry a separately derived cursor-zero certificate | computed boundary | `CatchupInputBound` |
| `nifs.pi_ccs.nc.post_state.retained_lanes` | bind only lanes one through seven | checked boundary | `CatchupInputBound` |
| `nifs.pi_ccs.catchup` | derive the post-catch-up state from the post-NC state | computed | `run` |
| `nifs.pi_rlc.output_digest_bind` | absorb the exact label, count, and four recomputed fields | computed | `run` |
| `nifs.pi_rlc.output_digest_bind.r1cs` | accepted owners equal the same pure handoff | conditional refinement | `accepted_refines_run` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.SemanticHandoff

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Canonical natural representatives of the complete typed terminal output
serialization. -/
def serializedValues
    {shape : SemanticShape}
    (profile : Projection.SplitNc.Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape) :
    List Nat :=
  (Semantics.serializeTerminalOutputs
    (Projection.SplitNc.projectOutputs profile message)).map Fin.val

/-- Pure natural-valued digest lane after the exact two-map SIS composition
and isolated Poseidon2 sponge. The map coefficients remain the explicit
production profile until the separate sampler/conformance obligations close. -/
def digestValue
    {shape : SemanticShape}
    (profile : Projection.SplitNc.Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape)
    (lane : Nat) : Nat :=
  Poseidon2Sponge.runValueRounds Poseidon.Schedule.trace.rounds
    (Poseidon.EnvelopeSemantics.diagnosticEnvelope
      (Sis.Semantics.apply
        (Sis.Refinement.mapOfBlock Sis.ProductionBinding.compressionBlock)
        (Sis.Semantics.apply
          (Sis.Refinement.mapOfBlock Sis.ProductionBinding.primaryBlock)
          (serializedValues profile message))))
    (fun _ => 0) lane

/-- Four canonical digest fields recomputed from the typed Split-NC message. -/
def digest
    {shape : SemanticShape}
    (profile : Projection.SplitNc.Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape) :
    Fin 4 -> Field :=
  fun lane => fieldValue (digestValue profile message lane.val)

/-- Complete post-NC to pre-`Pi_RLC` state transition: verifier catch-up,
message digest recomputation, and exact digest-label absorption. -/
def run
    {shape : SemanticShape}
    (profile : Projection.SplitNc.Profile shape)
    (postNc : State)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape) :
    State :=
  PiRlcChallenge.Transcript.OutputDigestSemantics.appendInputClaimsDigest
    (PiCcsTranscript.Primitives.catchup postNc).1
    (digest profile message)

/-- Exact dynamic-message premise still required at the production boundary:
the decoded digest columns are precisely the projection of the accepted
Split-NC output message. -/
def MessageBound
    {shape : SemanticShape}
    (profile : Projection.SplitNc.Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop :=
  Projection.SplitNc.projectOutputs profile message =
    SourceLayout.decodedOutputs assignment canonical

/-- Minimal grouped transcript premise still required at the production
boundary. Its cursor child can be constructed from exact positive NC replay;
its artifact child contains only lanes one through seven. Lane zero is
excluded because the verifier overwrites it with the catch-up squeeze marker. -/
abbrev CatchupInputBound
    (postNc : State)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop :=
  PiCcsTranscript.Refinement.Terminal.PostNcBoundary.Bound
    postNc assignment canonical

private theorem serializedValues_eq_production
    {shape : SemanticShape}
    (profile : Projection.SplitNc.Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (messageBound : MessageBound profile message assignment canonical) :
    serializedValues profile message =
      Sis.ProductionBinding.serializedValues assignment canonical := by
  unfold serializedValues Sis.ProductionBinding.serializedValues
  unfold MessageBound at messageBound
  rw [messageBound]

private theorem digestValue_eq_recomputed
    {shape : SemanticShape}
    (profile : Projection.SplitNc.Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (messageBound : MessageBound profile message assignment canonical)
    (lane : Nat) :
    digestValue profile message lane =
      Handoff.recomputedDigestValue assignment canonical lane := by
  rw [digestValue, Handoff.recomputedDigestValue,
    serializedValues_eq_production profile message canonical messageBound]

/-- Accepted terminal owner rows refine the exact semantic handoff, provided
the preceding semantic-to-column and post-NC-state authority bridges hold.
These two premises are deliberately stronger than self-consistent digest or
state equalities. -/
theorem accepted_refines_run
    (prime : EuclidPrime goldilocksP)
    {shape : SemanticShape}
    (profile : Projection.SplitNc.Profile shape)
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
    (messageBound : MessageBound profile message assignment canonical)
    (postNcBoundary :
      CatchupInputBound postNc assignment canonical) :
    run profile postNc message =
      PiRlcChallenge.Transcript.Terminal.ScheduleRefinement.initialState
        assignment canonical := by
  have handoff := Handoff.accepted_conditionalDigestHandoff prime canonical one
    outputAccepted catchupAccepted rlcAccepted
  have catchupInputBound :=
    PiCcsTranscript.Refinement.Terminal.PostNcBoundary.refines_catchupInput
      canonical one catchupAccepted postNcBoundary
  have catchupState :
      (PiCcsTranscript.Primitives.catchup postNc).1 =
        PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.postCatchupState
          assignment canonical := by
    change permute (absorbElem postNc (wordField 1)) =
      PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.postCatchupState
        assignment canonical
    rw [catchupInputBound]
    exact handoff.2.1
  have digestEq :
      digest profile message =
        PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.outputDigest
          assignment canonical := by
    funext lane
    apply Fin.ext
    have valueEq :
        digestValue profile message lane.val =
          (PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.outputDigest
            assignment canonical lane).val := by
      rw [digestValue_eq_recomputed profile message canonical messageBound]
      exact (handoff.1 lane).symm
    simp [digest, fieldValue, valueEq,
      Nat.mod_eq_of_lt
        (PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.outputDigest
          assignment canonical lane).isLt]
  unfold run
  rw [catchupState, digestEq]
  exact handoff.2.2

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.SemanticHandoff
