import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Transcript.Handoff

/-!
Public theorem-shape regressions for the active PiRLC transcript handoff.

Assurance tier: model-level conditional correspondence. Exact transcript and
sampler row embeddings determine the independent first-accepted challenge
vector. Incoming PiCCS state/digest authority and whole-Rust-row identity
remain explicit external obligations.
-/

namespace NightstreamTests.FPrimeRecursivePiRlcChallengeTranscriptHandoff

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript

#check Replay.InputsBound
#check Replay.inputsBound_of_boundary_equalities
#check Replay.accepted_refines
#check Handoff.transcriptCandidates
#check Handoff.fieldCandidate_eq_transcriptCandidate
#check Handoff.semanticOutput_eq_transcriptOutput
#check Handoff.fieldDerivedChallenges_eq_transcriptDerivedChallenges
#check Handoff.embeddedRows_bind_transcriptDerivedChallenges

example
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (transcriptRows : Transcript.Schedule.RowsEmbedded fullRows)
    (samplerRows : Sampler.Rows.EmbeddedRowsSatisfied fullRows assignment) :
    ProjectionConsumer.SamplerChallengesBound assignment
      (Handoff.transcriptDerivedChallenges assignment canonical) :=
  Handoff.embeddedRows_bind_transcriptDerivedChallenges
    prime canonical one transcriptRows samplerRows

end NightstreamTests.FPrimeRecursivePiRlcChallengeTranscriptHandoff
