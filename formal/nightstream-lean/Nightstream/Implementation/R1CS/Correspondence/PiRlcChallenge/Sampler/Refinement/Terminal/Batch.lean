import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler

/-!
Terminal `Pi_RLC` rows refined to the independent concrete Phi81 batch
sampler.

Assurance tier: implementation/R1CS correspondence. This module packages the
already-proved state schedule, candidate provenance, first-accepted selection,
and ring decoding as the exact `BatchExecution` consumed by the independent
NIFS transition semantics.

Owns: equality between the abstract batch state at every scalar coordinate
and the connected terminal artifact state; equality between each abstract
64-candidate source prefix and the machine-derived terminal prefix; one exact
bounded execution per scalar; and a concrete Phi81 sampler `Bound` for all
fifteen decoded production challenges.

Does not own: why the initial terminal state is the outgoing state of the
preceding `Pi_CCS` verifier, construction of a complete NIFS certificate,
native Rust conformance, row necessity, row removal, or cost totals.

Emits constraints: no.

Authority boundary: this file starts from the explicit terminal
`initialState`. It proves all sampler behavior after that point, but does not
silently identify that state with `Pi_CCS.finalState`. The later handoff
theorem must establish that equality before this result can discharge
`ConcretePhi81.Sampler.CertificateAccepted`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.challenge.batch.state` | abstract `stateAt` equals the connected artifact state before every scalar | verifier-owned computation | `stateAt_refines` |
| `nifs.pi_rlc.challenge.batch.prefix` | abstract source prefix equals the exact 64 machine candidates | derived | `sourcePrefix_eq_machineCandidates` |
| `nifs.pi_rlc.challenge.batch.execution` | every scalar has one successful least-cursor execution | checked rows refined to typed witness | `execution_exists` |
| `nifs.pi_rlc.challenge.batch.output` | assembled batch scalar equals the machine-derived Phi81 challenge | derived | `RefinedBatch.challenge_eq_machineChallenge` |
| `nifs.pi_rlc.challenge.batch.binding` | all decoded challenge columns form one semantic sampler `Bound` | checked/derived boundary | `accepted_refines_initialStateBound` |
| `nifs.pi_rlc.challenge.batch.handoff` | terminal initial state equals the preceding `Pi_CCS` outgoing state | open; not owned here | later transcript handoff theorem |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule

/-- The independent concrete-NIFS sampler instantiated by the exact
implementation transcript machine and direct Phi81 RingF assembly. -/
abbrev ringSpecification :=
  Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Specification
    TranscriptMachine.machine

/-- Four connected digest blocks reach the exact block-three artifact state.
This is the scalar-local state closure used by the batch induction. -/
theorem stateAfterFourBlocks_eq_block3
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    stateBeforeBlock TranscriptMachine.machine
        (Transcript.Terminal.ScheduleRefinement.afterEnterState
          assignment canonical rho)
        rho.val digestRounds =
      Transcript.Terminal.ScheduleRefinement.block3State
        assignment canonical rho := by
  simp [stateBeforeBlock, digestRounds,
    Transcript.Terminal.ScheduleRefinement.digestBlock0_refines
      canonical one accepted rho,
    Transcript.Terminal.ScheduleRefinement.digestBlock1_refines
      canonical one accepted rho,
    Transcript.Terminal.ScheduleRefinement.digestBlock2_refines
      canonical one accepted rho,
    Transcript.Terminal.ScheduleRefinement.digestBlock3_refines
      canonical one accepted rho]

/-- The abstract transcript-chained sampler state equals the connected
terminal artifact state before every one of the fifteen scalar coordinates. -/
theorem stateAt_refines
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    forall (index : Nat) (within : index < ScalarRows.scalarCount),
      stateAt ringSpecification
          (Transcript.Terminal.ScheduleRefinement.initialState
            assignment canonical)
          index =
        Transcript.Terminal.ScheduleRefinement.stateBeforeScalar
          assignment canonical ⟨index, within⟩ := by
  intro index within
  induction index with
  | zero =>
      simp [stateAt,
        Transcript.Terminal.ScheduleRefinement.stateBeforeScalar]
  | succ index inductionHypothesis =>
      have previousWithin : index < ScalarRows.scalarCount := by omega
      let previous : Fin ScalarRows.scalarCount := ⟨index, previousWithin⟩
      let current : Fin ScalarRows.scalarCount := ⟨index + 1, within⟩
      have priorState := inductionHypothesis previousWithin
      have entered :=
        Transcript.Terminal.ScheduleRefinement.enterScalar_refines
          canonical one accepted previous
      have completed :=
        stateAfterFourBlocks_eq_block3 canonical one accepted previous
      have currentNonzero : current.val ≠ 0 := by
        simp [current]
      have currentState :=
        Transcript.Terminal.ScheduleRefinement.stateBeforeScalar_nonzero
          canonical current currentNonzero
      have predecessor :
          Transcript.Terminal.ScheduleRefinement.previousScalar
              current currentNonzero =
            previous := by
        apply Fin.ext
        simp [Transcript.Terminal.ScheduleRefinement.previousScalar,
          current, previous]
      rw [stateAt_succ]
      change
        stateBeforeBlock TranscriptMachine.machine
            (enterScalar
              (stateAt ringSpecification
                (Transcript.Terminal.ScheduleRefinement.initialState
                  assignment canonical)
                index)
              index)
            index digestRounds =
          Transcript.Terminal.ScheduleRefinement.stateBeforeScalar
            assignment canonical current
      rw [priorState]
      change
        stateBeforeBlock TranscriptMachine.machine
            (enterScalar
              (Transcript.Terminal.ScheduleRefinement.stateBeforeScalar
                assignment canonical previous)
              previous.val)
            previous.val digestRounds =
          Transcript.Terminal.ScheduleRefinement.stateBeforeScalar
            assignment canonical current
      rw [entered, completed, currentState, predecessor]

/-- Enumerating the first 64 entries of the connected machine stream gives
the exact finite candidate list already used by the terminal row refinement. -/
theorem candidateStreamPrefix_eq_machineCandidates
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) :
    Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix
        (candidateStream TranscriptMachine.machine
          (Transcript.Terminal.ScheduleRefinement.afterEnterState
            assignment canonical rho)
          rho.val)
        candidateBound =
      MachineOutput.machineCandidates assignment canonical rho := by
  apply List.ext_get
  · simp [Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix,
      MachineOutput.machineCandidates, candidateBound,
      SelectionRows.candidateCount]
  · intro index leftLt rightLt
    simp only [Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix,
      MachineOutput.machineCandidates, List.get_eq_getElem,
      List.getElem_map, List.getElem_range, List.getElem_ofFn]

/-- The source prefix seen by the independent batch semantics is exactly the
machine-derived prefix decoded from the accepted terminal rows. -/
theorem sourcePrefix_eq_machineCandidates
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix
        (sourceAt ringSpecification
          (Transcript.Terminal.ScheduleRefinement.initialState
            assignment canonical)
          rho.val).stream
        candidateBound =
      MachineOutput.machineCandidates assignment canonical rho := by
  have priorState :=
    stateAt_refines canonical one accepted rho.val rho.isLt
  have entered :=
    Transcript.Terminal.ScheduleRefinement.enterScalar_refines
      canonical one accepted rho
  change
    Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix
        (candidateStream TranscriptMachine.machine
          (enterScalar
            (stateAt ringSpecification
              (Transcript.Terminal.ScheduleRefinement.initialState
                assignment canonical)
              rho.val)
            rho.val)
          rho.val)
        candidateBound =
      MachineOutput.machineCandidates assignment canonical rho
  rw [priorState, entered]
  exact candidateStreamPrefix_eq_machineCandidates assignment canonical rho

/-- Each accepted terminal scalar constructs one exact bounded execution over
the transcript-chained source used by the independent NIFS semantics. -/
theorem execution_exists
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    exists execution :
        CoefficientExecution ringSpecification candidateBound
          (Transcript.Terminal.ScheduleRefinement.initialState
            assignment canonical)
          rho.val,
      execution.output =
        MachineOutput.semanticOutput assignment canonical rho := by
  have prefixEq :=
    sourcePrefix_eq_machineCandidates canonical one accepted rho
  have enough :=
    MachineOutput.enoughAccepted prime canonical one accepted rho
  have success :
      Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample
          ProductionAlphabet.verifier coefficientCount
          (Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix
            (sourceAt ringSpecification
              (Transcript.Terminal.ScheduleRefinement.initialState
                assignment canonical)
              rho.val).stream
            candidateBound) =
        some (MachineOutput.semanticOutput assignment canonical rho) := by
    apply
      Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_eq_some_iff.mpr
    constructor
    · rw [prefixEq]
      exact enough
    · unfold MachineOutput.semanticOutput
      rw [prefixEq]
  exact Nightstream.SuperNeo.Sampling.FirstAccepted.BoundedExecution.exists_of_bounded_success
    success

/-- One typed implementation refinement of the complete fifteen-scalar batch.
The equality field retains the exact selected coefficient list for every
coordinate, so later ring assembly does not depend on a choice of witness. -/
structure RefinedBatch
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) where
  batch :
    BatchExecution ringSpecification ScalarRows.scalarCount candidateBound
      (Transcript.Terminal.ScheduleRefinement.initialState assignment canonical)
  output_eq : forall rho,
    (batch.execution rho).output =
      MachineOutput.semanticOutput assignment canonical rho

/-- Accepted rows construct the complete transcript-chained batch. Classical
choice selects the already-proved unique bounded execution at each finite
coordinate; it does not add a protocol assumption. -/
theorem accepted_refines_batch
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    Nonempty (RefinedBatch assignment canonical) := by
  classical
  let execution := fun rho : Fin ScalarRows.scalarCount =>
    Classical.choose (execution_exists prime canonical one accepted rho)
  have outputEq : forall rho : Fin ScalarRows.scalarCount,
      (execution rho).output =
        MachineOutput.semanticOutput assignment canonical rho := by
    intro rho
    exact Classical.choose_spec
      (execution_exists prime canonical one accepted rho)
  exact ⟨{
    batch := { execution := execution }
    output_eq := outputEq
  }⟩

/-- Every scalar assembled by a refined batch is exactly the independently
defined machine-derived Phi81 challenge. -/
theorem RefinedBatch.challenge_eq_machineChallenge
    {assignment : Nat -> Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    (refinement : RefinedBatch assignment canonical)
    (rho : Fin ScalarRows.scalarCount) :
    challenge refinement.batch rho =
      RingAssembly.machineChallenge assignment canonical rho := by
  rw [RingAssembly.machineChallenge_eq_embedScalar]
  apply congrArg Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedScalar
  funext position
  unfold coefficient MachineOutput.scalar
  calc
    (refinement.batch.execution rho).output.get ⟨position.val, by
        rw [(refinement.batch.execution rho).output_length]
        exact position.isLt⟩ =
        (refinement.batch.execution rho).output.getD position.val
          TailFirstAccepted.defaultCoefficient :=
      by
        rw [List.get_eq_getElem]
        exact List.getElem_eq_getD TailFirstAccepted.defaultCoefficient
    _ = (MachineOutput.semanticOutput assignment canonical rho).getD
          position.val TailFirstAccepted.defaultCoefficient :=
      congrArg
        (fun output =>
          output.getD position.val TailFirstAccepted.defaultCoefficient)
        (refinement.output_eq rho)

/-- Complete implementation-to-semantics sampler bridge at the explicit
terminal initial-state boundary. The separate `Pi_CCS` handoff remains visible
in the theorem type rather than being assumed. -/
theorem accepted_refines_initialStateBound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    Nonempty
      (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound
        TranscriptMachine.machine
        (Transcript.Terminal.ScheduleRefinement.initialState
          assignment canonical)
        (fun rho : Fin ScalarRows.scalarCount =>
          RingAssembly.decodedChallenge assignment rho)) := by
  rcases accepted_refines_batch prime canonical one accepted with
    ⟨refinement⟩
  exact ⟨{
    batch := refinement.batch
    challenges_eq := fun rho => by
      calc
        RingAssembly.decodedChallenge assignment rho =
            RingAssembly.machineChallenge assignment canonical rho :=
          RingAssembly.decodedChallenge_eq_machineChallenge
            prime canonical one accepted rho
        _ = challenge refinement.batch rho :=
          (refinement.challenge_eq_machineChallenge rho).symm
  }⟩

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch
