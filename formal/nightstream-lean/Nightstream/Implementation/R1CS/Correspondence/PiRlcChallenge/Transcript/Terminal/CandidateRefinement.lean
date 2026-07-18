import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted

/-!
Poseidon2 provenance for every terminal `Pi_RLC` bounded-sampler candidate.

Assurance tier: implementation/R1CS correspondence. This module connects the
canonical field columns consumed by the rejection sampler to the candidate
chunks jointly returned by the independently specified transcript machine.

Owns: the `rho -> digest block -> lane -> part` candidate-provenance tree;
artifact and machine views of each digest block; and equality of every one of
the 960 terminal field-derived chunks with its verifier-owned machine chunk.

Does not own: the state entering `Pi_RLC` from `Pi_CCS`, rejection/selection
correctness, coefficient assembly, Rust trace conformance, row removal, or
cost totals.

Emits constraints: no.

Authority boundary: a field column becomes a transcript candidate only here,
after the corresponding accepted Poseidon2 call has been replayed and its
input has been connected to the preceding machine state. Neither generated
row order nor a prover-carried digest supplies candidate authority.

| Protocol | Phase | Constraint family | Indexed leaf | Proven obligation |
|---|---|---|---|---|
| `Pi_RLC` | terminal challenge | scalar batch | `rho : Fin 15` | each scalar starts from the connected transcript schedule |
| `Pi_RLC` | digest generation | block state | `block : Fin 4` | machine output state equals the exact artifact call output |
| `Pi_RLC` | digest extraction | canonical lane | `lane : Fin 4` | sampler field column is the matching Poseidon2 output lane |
| `Pi_RLC` | bounded sampler | 16-bit candidate | `part : Fin 4` | field-derived chunk equals the jointly returned machine chunk |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.CandidateRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Exact artifact state after one of a scalar's four digest blocks. -/
def artifactBlockState
    (assignment : Nat -> Nat)
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 4) : State :=
  ChunkOrder.select4 block
    (ScheduleRefinement.block0State assignment canonical rho)
    (ScheduleRefinement.block1State assignment canonical rho)
    (ScheduleRefinement.block2State assignment canonical rho)
    (ScheduleRefinement.block3State assignment canonical rho)

/-- Exact connected state from which the machine executes one digest block. -/
def machineBlockInput
    (assignment : Nat -> Nat)
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 4) : State :=
  ChunkOrder.select4 block
    (ScheduleRefinement.afterEnterState assignment canonical rho)
    (ScheduleRefinement.block0State assignment canonical rho)
    (ScheduleRefinement.block1State assignment canonical rho)
    (ScheduleRefinement.block2State assignment canonical rho)

/-- Verifier-owned counter used by one scalar/block address. -/
def blockCounter (rho : Fin ScalarRows.scalarCount) (block : Fin 4) : Nat :=
  rho.val + block.val

private theorem fin4_value_cases (index : Fin 4) :
    index.val = 0 \/ index.val = 1 \/
      index.val = 2 \/ index.val = 3 := by
  have indexLt := index.isLt
  omega

/-- The first four lanes of each artifact block state are exactly the field
columns consumed by the terminal sampler at the same hierarchical address. -/
theorem artifactBlockState_lane
    {assignment : Nat -> Nat}
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) (block lane : Fin 4) :
    ChunkOrder.digestLanes
        (artifactBlockState assignment canonical rho block) lane =
      fieldAt assignment canonical
        (ScalarRows.fieldColumn rho block lane) := by
  rcases fin4_value_cases block with hb | hb | hb | hb <;>
    rcases fin4_value_cases lane with hl | hl | hl | hl <;>
    simp [artifactBlockState, ChunkOrder.digestLanes, ChunkOrder.select4,
      ScheduleRefinement.block0State, ScheduleRefinement.block1State,
      ScheduleRefinement.block2State, ScheduleRefinement.block3State,
      callOutputState, ScalarRows.fieldColumn, Schedule.block0DigestCall,
      Schedule.block1DigestCall, Schedule.block2DigestCall,
      Schedule.block3DigestCall, Schedule.laterDigestCall,
      Schedule.laterBlockPinBase, Schedule.scalarColumnStride,
      Poseidon2Call.Call.columnMap, hb, hl]
  all_goals congr 1 <;> omega

/-- Each connected machine digest execution reaches the exact artifact state
that also owns its candidate lanes. -/
theorem digestBlock_refines
    {assignment : Nat -> Nat}
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 4) :
    (digestBlock (machineBlockInput assignment canonical rho block)
        (blockCounter rho block)).1 =
      artifactBlockState assignment canonical rho block := by
  rcases fin4_value_cases block with hb | hb | hb | hb
  · simpa [machineBlockInput, artifactBlockState, blockCounter,
      ChunkOrder.select4, hb] using
      ScheduleRefinement.digestBlock0_refines canonical one accepted rho
  · simpa [machineBlockInput, artifactBlockState, blockCounter,
      ChunkOrder.select4, hb] using
      ScheduleRefinement.digestBlock1_refines canonical one accepted rho
  · simpa [machineBlockInput, artifactBlockState, blockCounter,
      ChunkOrder.select4, hb] using
      ScheduleRefinement.digestBlock2_refines canonical one accepted rho
  · simpa [machineBlockInput, artifactBlockState, blockCounter,
      ChunkOrder.select4, hb] using
      ScheduleRefinement.digestBlock3_refines canonical one accepted rho

/-- The four selected block inputs are not merely adjacent artifact states:
they are exactly the states obtained by recursively executing all preceding
blocks of the independent production schedule. -/
theorem stateBeforeBlock_refines
    {assignment : Nat -> Nat}
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 4) :
    ProductionSchedule.stateBeforeBlock TranscriptMachine.machine
        (ScheduleRefinement.afterEnterState assignment canonical rho)
        rho.val block.val =
      machineBlockInput assignment canonical rho block := by
  rcases fin4_value_cases block with hb | hb | hb | hb
  · simp [machineBlockInput, ChunkOrder.select4, hb]
  · simpa [ProductionSchedule.stateBeforeBlock, machineBlockInput,
      ChunkOrder.select4, hb] using
      ScheduleRefinement.digestBlock0_refines canonical one accepted rho
  · simp [ProductionSchedule.stateBeforeBlock, machineBlockInput,
      ChunkOrder.select4, hb,
      ScheduleRefinement.digestBlock0_refines canonical one accepted rho,
      ScheduleRefinement.digestBlock1_refines canonical one accepted rho]
  · simp [ProductionSchedule.stateBeforeBlock, machineBlockInput,
      ChunkOrder.select4, hb,
      ScheduleRefinement.digestBlock0_refines canonical one accepted rho,
      ScheduleRefinement.digestBlock1_refines canonical one accepted rho,
      ScheduleRefinement.digestBlock2_refines canonical one accepted rho]

/-- The schedule's flattened within-block index is the independently named
lane-major `(lane, part)` position. -/
theorem candidateWithinBlock_eq_chunkPosition
    (candidate : Fin SelectionRows.candidateCount) :
    (⟨candidate.val % ProductionAlphabet.chunksPerDigest,
      Nat.mod_lt _ (by decide)⟩ :
        Fin ProductionAlphabet.chunksPerDigest) =
      ChunkOrder.chunkPosition
        (CandidateOrder.address candidate).lane
        (CandidateOrder.address candidate).part := by
  apply Fin.ext
  have candidateLt := candidate.isLt
  have split16 := Nat.div_add_mod candidate.val 16
  have split4 := Nat.div_add_mod candidate.val 4
  have nestedSplit4 := Nat.div_add_mod (candidate.val % 16) 4
  simp only [ProductionAlphabet.chunksPerDigest, ChunkOrder.chunkPosition,
    CandidateOrder.address]
  simp only [SelectionRows.candidateCount] at candidateLt
  omega

/-- One field-derived candidate is the matching chunk of the exact artifact
digest state, before using the machine-execution theorem. -/
theorem fieldCandidate_eq_artifactStateChunk
    {assignment : Nat -> Nat}
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount)
    (candidate : Fin SelectionRows.candidateCount) :
    TailCandidateSemantics.fieldCandidate (TailSources.layout rho)
        assignment canonical candidate =
      ChunkOrder.stateChunks
        (artifactBlockState assignment canonical rho
          (CandidateOrder.address candidate).block)
        (ChunkOrder.chunkPosition
          (CandidateOrder.address candidate).lane
          (CandidateOrder.address candidate).part) := by
  rw [ChunkOrder.stateChunks_lane_part]
  rw [artifactBlockState_lane canonical rho
    (CandidateOrder.address candidate).block
    (CandidateOrder.address candidate).lane]
  unfold TailCandidateSemantics.fieldCandidate
  rw [TailCandidateSemantics.laneIndex_block,
    TailCandidateSemantics.laneIndex_lane]
  rfl

/-- Every terminal field-derived candidate is exactly the chunk jointly
returned by the connected verifier-owned digest execution at the same
`rho/block/lane/part` address. -/
theorem accepted_refines_machineCandidate
    {assignment : Nat -> Nat}
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount)
    (candidate : Fin SelectionRows.candidateCount) :
    TailCandidateSemantics.fieldCandidate (TailSources.layout rho)
        assignment canonical candidate =
      (digestBlock
        (machineBlockInput assignment canonical rho
          (CandidateOrder.address candidate).block)
        (blockCounter rho (CandidateOrder.address candidate).block)).2
          (ChunkOrder.chunkPosition
            (CandidateOrder.address candidate).lane
            (CandidateOrder.address candidate).part) := by
  let block := (CandidateOrder.address candidate).block
  let position := ChunkOrder.chunkPosition
    (CandidateOrder.address candidate).lane
    (CandidateOrder.address candidate).part
  have stateEq := digestBlock_refines canonical one accepted rho block
  have candidatesEq := congrFun
    (ChunkOrder.digestBlock_candidates_eq_stateChunks
      (machineBlockInput assignment canonical rho block)
      (blockCounter rho block)) position
  calc
    TailCandidateSemantics.fieldCandidate (TailSources.layout rho)
        assignment canonical candidate =
      ChunkOrder.stateChunks
          (artifactBlockState assignment canonical rho block) position :=
        fieldCandidate_eq_artifactStateChunk canonical rho candidate
    _ = ChunkOrder.stateChunks
          (digestBlock (machineBlockInput assignment canonical rho block)
            (blockCounter rho block)).1 position := by
        rw [stateEq]
    _ = (digestBlock (machineBlockInput assignment canonical rho block)
          (blockCounter rho block)).2 position := candidatesEq.symm

/-- Candidate provenance stated directly against the canonical flattened
stream of the independently specified production schedule. -/
theorem accepted_refines_candidateStream
    {assignment : Nat -> Nat}
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount)
    (candidate : Fin SelectionRows.candidateCount) :
    TailCandidateSemantics.fieldCandidate (TailSources.layout rho)
        assignment canonical candidate =
      ProductionSchedule.candidateStream TranscriptMachine.machine
        (ScheduleRefinement.afterEnterState assignment canonical rho)
        rho.val candidate.val := by
  rw [accepted_refines_machineCandidate canonical one accepted rho candidate]
  unfold ProductionSchedule.candidateStream ProductionSchedule.chunksAt
  have blockIndex :
      candidate.val / ProductionAlphabet.chunksPerDigest =
        (CandidateOrder.address candidate).block.val := by
    rfl
  rw [blockIndex]
  rw [stateBeforeBlock_refines canonical one accepted rho
    (CandidateOrder.address candidate).block]
  rw [candidateWithinBlock_eq_chunkPosition candidate]
  rfl

/-- Aggregate terminal closure at the transcript-to-candidate boundary. The
package includes state connectivity so candidate equality cannot be read as
an equality against four unrelated digest calls. -/
structure RefinesMachineCandidates
    (assignment : Nat -> Nat)
    (canonical : ScheduleRefinement.CanonicalAssignment assignment) : Prop where
  stateSchedule : ScheduleRefinement.StateScheduleRefined assignment canonical
  candidate : forall (rho : Fin ScalarRows.scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
    TailCandidateSemantics.fieldCandidate (TailSources.layout rho)
        assignment canonical candidate =
      ProductionSchedule.candidateStream TranscriptMachine.machine
        (ScheduleRefinement.afterEnterState assignment canonical rho)
        rho.val candidate.val

/-- Accepted terminal rows refine the complete connected state-and-candidate
machine schedule, starting from the still-explicit post-`Pi_CCS` state. -/
theorem accepted_refines_machineCandidates
    {assignment : Nat -> Nat}
    (canonical : ScheduleRefinement.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    RefinesMachineCandidates assignment canonical :=
  { stateSchedule :=
      ScheduleRefinement.stateScheduleRefined canonical one accepted
    candidate :=
      accepted_refines_candidateStream canonical one accepted }

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.CandidateRefinement
