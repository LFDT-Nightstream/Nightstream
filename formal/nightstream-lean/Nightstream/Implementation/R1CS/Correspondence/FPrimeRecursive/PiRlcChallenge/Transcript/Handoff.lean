import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.ProjectionConsumer
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Transcript.Replay

/-!
Active PiRLC transcript-to-sampler handoff.

Owns: equality between each of the 15-by-64 sampler field candidates and the
same candidate computed by the handwritten Poseidon2 transcript machine;
the resulting transcript-owned first-accepted coefficient vectors; and their
handoff to the active projection challenge vector.

Does not own: authority for the incoming PiCCS state or four output-digest
fields, identity of the explicit row embeddings with the complete Rust
relation, ring projection arithmetic, cryptographic bad-event bounds, costs,
or row removal.

Emits constraints: no.

Authority boundary: `Replay.Refines` binds the active field columns to the
independent transcript execution. Sampler rows then prove first-accepted
selection from those candidates. The final row theorem remains conditional on
explicit transcript and sampler embeddings plus whole-program satisfaction.

| Stage path | Mathematical obligation | Assurance |
|---|---|---|
| `nifs.pi_rlc.challenge.transcript_to_sampler.candidate` | each field chunk equals the same transcript-source chunk | model-level conditional refinement |
| `nifs.pi_rlc.challenge.transcript_to_sampler.first_accepted` | all 15 vectors are the first 54 accepted transcript chunks | model-level conditional refinement |
| `nifs.pi_rlc.challenge.transcript_to_sampler.projection` | decoded projection inputs equal transcript-derived rings | model-level conditional refinement |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Handoff

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Sampling

namespace ActiveSampler

abbrev scalarCount := FPrimeRecursivePiRlcChallenge.SamplerLayout.scalarCount
abbrev layout := FPrimeRecursivePiRlcChallenge.Sampler.TailSources.layout
abbrev semanticOutput :=
  FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted.semanticOutput

end ActiveSampler

/-- One candidate computed from the verifier-owned transcript source. -/
def transcriptCandidate
    (assignment : Nat → Nat) (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ActiveSampler.scalarCount)
    (candidate : Fin PiRlcChallenge.Sampler.SelectionRows.candidateCount) :
    ProductionAlphabet.Chunk :=
  (sourceAt specification (Replay.postBindState assignment canonical) rho.val).stream
    candidate.val

/-- The production stream's quotient/remainder index is the same lane-major
position named by the sampler's independent candidate address. -/
private theorem candidateWithinBlock_eq_lanePart
    (candidate : Fin PiRlcChallenge.Sampler.SelectionRows.candidateCount) :
    (⟨candidate.val % ProductionAlphabet.chunksPerDigest,
      Nat.mod_lt _ (by decide)⟩ : Fin ProductionAlphabet.chunksPerDigest) =
      ⟨(CandidateOrder.address candidate).lane.val * 4 +
          (CandidateOrder.address candidate).part.val, by
        have laneLt := (CandidateOrder.address candidate).lane.isLt
        have partLt := (CandidateOrder.address candidate).part.isLt
        change (CandidateOrder.address candidate).lane.val * 4 +
          (CandidateOrder.address candidate).part.val <
            ProductionAlphabet.chunksPerDigest
        simp only [ProductionAlphabet.chunksPerDigest]
        omega⟩ := by
  apply Fin.ext
  have candidateLt := candidate.isLt
  have split16 := Nat.div_add_mod candidate.val 16
  have split4 := Nat.div_add_mod candidate.val 4
  have nestedSplit4 := Nat.div_add_mod (candidate.val % 16) 4
  simp only [ProductionAlphabet.chunksPerDigest, CandidateOrder.address]
  simp only [PiRlcChallenge.Sampler.SelectionRows.candidateCount] at candidateLt
  omega

/-- The active field candidate is exactly the independently scheduled
transcript candidate at the same block/lane/part address. -/
theorem fieldCandidate_eq_transcriptCandidate
    {assignment : Nat → Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    (replay : Replay.Refines assignment canonical)
    (rho : Fin ActiveSampler.scalarCount)
    (candidate : Fin PiRlcChallenge.Sampler.SelectionRows.candidateCount) :
    PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.fieldCandidate
        (ActiveSampler.layout rho) assignment canonical candidate =
      transcriptCandidate assignment canonical rho candidate := by
  let address := CandidateOrder.address candidate
  unfold PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.fieldCandidate
  rw [PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex_block,
    PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex_lane]
  change laneChunk
      (PiRlcChallenge.Transcript.CallRefinement.fieldAt assignment canonical
        (FPrimeRecursivePiRlcChallenge.SamplerLayout.fieldColumn
          rho address.block address.lane)) address.part = _
  rw [replay.fieldDigest rho address.block address.lane]
  let entered := enterScalar
    (stateAt specification (Replay.postBindState assignment canonical) rho.val)
    rho.val
  change laneChunk
      (PiRlcChallenge.Transcript.Operations.blockDigest entered rho.val
        address.block.val address.lane) address.part =
    digestChunks
      (PiRlcChallenge.Transcript.Operations.blockDigest entered rho.val
        address.block.val)
      ⟨candidate.val % ProductionAlphabet.chunksPerDigest,
        Nat.mod_lt _ (by decide)⟩
  rw [candidateWithinBlock_eq_lanePart candidate,
    digestChunks_lane_part _ address.lane address.part]

/-- Exact 64-candidate transcript prefix for one active scalar. -/
def transcriptCandidates
    (assignment : Nat → Nat) (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ActiveSampler.scalarCount) : List ProductionAlphabet.Chunk :=
  List.ofFn (transcriptCandidate assignment canonical rho)

/-- The sampler's field-column candidate list becomes the verifier-owned
transcript prefix once physical replay is established. -/
theorem fieldCandidates_eq_transcriptCandidates
    {assignment : Nat → Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    (replay : Replay.Refines assignment canonical)
    (rho : Fin ActiveSampler.scalarCount) :
    PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.candidates
        (ActiveSampler.layout rho) assignment canonical =
      transcriptCandidates assignment canonical rho := by
  unfold PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.candidates
    transcriptCandidates
  exact congrArg
    (fun candidateAt :
        Fin PiRlcChallenge.Sampler.SelectionRows.candidateCount →
          ProductionAlphabet.Chunk => List.ofFn candidateAt)
    (funext fun candidate =>
      fieldCandidate_eq_transcriptCandidate replay rho candidate)

/-- Independent first 54 accepted coefficients of one transcript source. -/
def transcriptOutput
    (assignment : Nat → Nat) (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ActiveSampler.scalarCount) : List ProductionAlphabet.Coefficient :=
  FirstAccepted.firstAccepted ProductionAlphabet.verifier
    ProductionAlphabet.coefficientCount
    (transcriptCandidates assignment canonical rho)

/-- The existing field-derived sampler output is exactly the first-accepted
output of the independently replayed transcript source. -/
theorem semanticOutput_eq_transcriptOutput
    {assignment : Nat → Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    (replay : Replay.Refines assignment canonical)
    (rho : Fin ActiveSampler.scalarCount) :
    ActiveSampler.semanticOutput assignment canonical rho =
      transcriptOutput assignment canonical rho := by
  unfold ActiveSampler.semanticOutput
    FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted.semanticOutput
    PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.semanticOutput
    transcriptOutput
  rw [fieldCandidates_eq_transcriptCandidates replay rho]

/-- Canonical Goldilocks coefficient vector assembled from the transcript's
first-accepted output. -/
def transcriptFieldOutput
    (assignment : Nat → Nat) (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ActiveSampler.scalarCount) : List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    CandidateOrder.centeredField
      ((transcriptOutput assignment canonical rho).getD position.val
        PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.defaultCoefficient)

/-- Exact centered field output equality for all 54 coefficients. -/
theorem semanticFieldOutput_eq_transcriptFieldOutput
    {assignment : Nat → Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    (replay : Replay.Refines assignment canonical)
    (rho : Fin ActiveSampler.scalarCount) :
    FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted.semanticFieldOutput
        assignment canonical rho =
      transcriptFieldOutput assignment canonical rho := by
  unfold FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted.semanticFieldOutput
    PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.semanticFieldOutput
    transcriptFieldOutput
  exact congrArg
    (fun output : List ProductionAlphabet.Coefficient =>
      List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
        CandidateOrder.centeredField
          (output.getD position.val
            PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.defaultCoefficient))
    (semanticOutput_eq_transcriptOutput replay rho)

/-- Fifteen Phi81 challenges computed only from the transcript machine's
first-accepted coefficient vectors. -/
def transcriptDerivedChallenges
    (assignment : Nat → Nat) (canonical : ChunkOrder.CanonicalAssignment assignment) :
    Fin FPrimeRecursivePiRlcProjection.YZcolNormalForm.sourceCount →
      Nightstream.SuperNeo.Concrete.RingF :=
  fun rho =>
    FPrimeFullHistoryNifsPaper.PiRlc.ringOfList
      ((transcriptFieldOutput assignment canonical rho).map
        FPrimeFullHistoryNifsPaper.residue)

/-- The prior field-derived challenge vector equals the transcript-owned
vector after replay; no digest or equality constraint is added. -/
theorem fieldDerivedChallenges_eq_transcriptDerivedChallenges
    {assignment : Nat → Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    (replay : Replay.Refines assignment canonical) :
    FPrimeRecursivePiRlcChallenge.ProjectionConsumer.fieldDerivedChallenges
        assignment canonical =
      transcriptDerivedChallenges assignment canonical := by
  funext rho
  unfold FPrimeRecursivePiRlcChallenge.ProjectionConsumer.fieldDerivedChallenges
    transcriptDerivedChallenges
  rw [semanticFieldOutput_eq_transcriptFieldOutput replay rho]

/-- Sampler rows plus transcript replay bind the projection consumer to the
independently derived challenge vector. -/
theorem samplerChallengesBound_transcriptDerived
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (samplerRows :
      FPrimeRecursivePiRlcChallenge.Sampler.Rows.EmbeddedRowsSatisfied
        fullRows assignment)
    (replay : Replay.Refines assignment canonical) :
    FPrimeRecursivePiRlcChallenge.ProjectionConsumer.SamplerChallengesBound
      assignment (transcriptDerivedChallenges assignment canonical) := by
  rw [← fieldDerivedChallenges_eq_transcriptDerivedChallenges replay]
  exact FPrimeRecursivePiRlcChallenge.ProjectionConsumer.samplerChallengesBound_fieldDerived
    prime canonical one samplerRows

/-- Complete conditional active row composition. The premise names both
physical ownership trees explicitly; no historical full-circuit oracle or
row count is trusted. -/
theorem embeddedRows_bind_transcriptDerivedChallenges
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (transcriptRows : Schedule.RowsEmbedded fullRows)
    (samplerRows :
      FPrimeRecursivePiRlcChallenge.Sampler.Rows.EmbeddedRowsSatisfied
        fullRows assignment) :
    FPrimeRecursivePiRlcChallenge.ProjectionConsumer.SamplerChallengesBound
      assignment (transcriptDerivedChallenges assignment canonical) := by
  have traceAccepted := Schedule.accepted_of_embedded canonical one
    transcriptRows samplerRows.fullSatisfies
  have replay := Replay.accepted_refines canonical one traceAccepted
  exact samplerChallengesBound_transcriptDerived
    prime canonical one samplerRows replay

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Handoff
