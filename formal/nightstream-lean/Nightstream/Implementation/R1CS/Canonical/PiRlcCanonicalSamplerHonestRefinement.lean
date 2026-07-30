import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonest
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram

/-!
Contract: identify the constructive fixed-active sampler witness with the
candidate stream consumed by the selected checker refinement.

The equality is proved from the same transcript/u64 assignment.  It neither
assumes sampler success nor imports a candidate list from a caller.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 800000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonestRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Sampling

private theorem foldl_congr_eq
    {α β : Type} (left right : α → β → α)
    (equal :
      ∀ accumulator value,
        left accumulator value = right accumulator value) :
    ∀ values initial,
      List.foldl left initial values = List.foldl right initial values := by
  intro values
  induction values with
  | nil =>
      intro initial
      rfl
  | cons head tail hypothesis =>
      intro initial
      simp only [List.foldl_cons]
      rw [equal]
      exact hypothesis _

/-- The constructive candidate list and the checker-refinement candidate list
are pointwise identical on the final u64 witness. -/
theorem honestCandidates_eq_semanticCandidates
    (prime : EuclidPrime goldilocksP)
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (u64Below :
      PiRlcCanonicalU64Honest.InputsBelow
        duplexBase u64Base count initialBuilder)
    (candidateBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (coordinate : Fin count)
    (canonical :
      ∀ column,
        PiRlcCanonicalSamplerHonest.u64Witness field duplexBase u64Base
            count initialBuilder initial column <
          goldilocksP)
    (constantWire :
      PiRlcCanonicalSamplerHonest.u64Witness field duplexBase u64Base count
          initialBuilder initial 0 = 1)
    (u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase u64Base count initialBuilder)
        (PiRlcCanonicalSamplerHonest.u64Witness field duplexBase u64Base count
          initialBuilder initial)) :
    PiRlcCanonicalSamplerHonest.honestCandidates field duplexBase u64Base
        candidateBase count initialBuilder initial u64Below candidateBelow
        coordinate =
      PiRlcCanonicalSamplerSound.semanticCandidates prime duplexBase u64Base
        candidateBase count initialBuilder canonical constantWire u64Satisfied
        coordinate := by
  apply List.ext_getElem
  · simp [PiRlcCanonicalSamplerHonest.honestCandidates,
      PiRlcCanonicalCandidatesBatchHonest.coordinateCandidates]
  · intro index leftBound _rightBound
    have indexLt :
        index < PiRlcCanonicalCandidates.candidatesPerScalar := by
      simpa [PiRlcCanonicalSamplerHonest.honestCandidates,
        PiRlcCanonicalCandidatesBatchHonest.coordinateCandidates] using
        leftBound
    let candidate :
        Fin PiRlcCanonicalCandidates.candidatesPerScalar :=
      ⟨index, indexLt⟩
    unfold PiRlcCanonicalSamplerHonest.honestCandidates
      PiRlcCanonicalCandidatesBatchHonest.coordinateCandidates
      PiRlcCanonicalScalarComplete.honestCandidates
      PiRlcCanonicalSamplerSound.semanticCandidates
    rw [List.getElem_ofFn, List.getElem_ofFn]
    change
      PiRlcCanonicalScalarComplete.honestCandidate duplexBase u64Base
          candidateBase initialBuilder coordinate
          (PiRlcCanonicalCandidatesBatchHonest.coordinatePrior field
            duplexBase u64Base candidateBase count initialBuilder
            (PiRlcCanonicalSamplerHonest.u64Witness field duplexBase u64Base
              count initialBuilder initial)
            coordinate)
          _ candidate =
        PiRlcCanonicalSamplerSound.semanticCandidate prime duplexBase u64Base
          candidateBase count initialBuilder canonical constantWire
          u64Satisfied coordinate candidate
    unfold PiRlcCanonicalScalarComplete.honestCandidate
      PiRlcCanonicalSamplerSound.semanticCandidate
    apply Fin.ext
    unfold PiRlcCanonicalCandidateSound.candidate
    simp only
    unfold PiRlcCanonicalCandidateSound.chunkValue
    apply foldl_congr_eq
    intro accumulator term
    congr 2
    apply
      PiRlcCanonicalCandidatesBatchHonest.batchPrefixWitness_before_candidateBase
    exact (candidateBelow coordinate).source candidate term

/-- A genuine selected sampler batch supplies the `Enough` evidence required
by the constructive physical sampler witness.  The bridge identifies the
constructive candidates with the exact executable-checker prefix before
transporting bounded success; no candidate list is accepted as a premise. -/
theorem honestEnough_of_bound
    (prime : EuclidPrime goldilocksP)
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (lanes : Poseidon2Core.State)
    (initial : Nat → Nat)
    (positive : 0 < duplexBase)
    (lanesInPrefix :
      ∀ lane : Fin Poseidon2Core.width,
        SymbolicDuplexPlacement.ValueInPrefix duplexBase (lanes lane))
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    {challenges :
      Fin PiRlcCanonicalSamplerProgram.coordinateCount →
        Nightstream.SuperNeo.Concrete.RingF}
    (bound :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound
        (PiRlcCanonicalMachine.machine constants)
        (SymbolicDuplexSemantics.decodedBuilder initial
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
        challenges) :
    ∀ coordinate : Fin PiRlcCanonicalSamplerProgram.coordinateCount,
      FirstAccepted.Enough ProductionAlphabet.verifier
        PiRlcCanonicalSelector.outputCount
        (PiRlcCanonicalSamplerHonest.honestCandidates field
          duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
          (PiRlcCanonicalSymbolicMachineHonest.fixedWitness
            duplexBase constants lanes initial)
          (PiRlcCanonicalSamplerProgram.inputsBelow duplexBase lanes)
          (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
            duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
            (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
            PiRlcCanonicalSamplerProgram.coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
            (PiRlcCanonicalSamplerProgram.u64_separated duplexBase))
          coordinate) := by
  let initialBuilder :=
    PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes
  let transcriptAssignment :=
    PiRlcCanonicalSymbolicMachineHonest.fixedWitness
      duplexBase constants lanes initial
  let u64Assignment :=
    PiRlcCanonicalSamplerHonest.u64Witness field duplexBase
      (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
      PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
      transcriptAssignment
  let u64Below :=
    PiRlcCanonicalSamplerProgram.inputsBelow duplexBase lanes
  let candidateBelow :=
    PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
      duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
      (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
      PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
      (PiRlcCanonicalSamplerProgram.u64_separated duplexBase)
  have lanesBefore :
      ∀ lane : Fin Poseidon2Core.width, ∀ column,
        LinCombNormal.Mentions (lanes lane) column →
          column < SymbolicDuplexHonest.outputBase duplexBase 0 := by
    intro lane column mentioned
    simpa [SymbolicDuplexHonest.outputBase,
      SymbolicDuplexHonest.callBase] using
      lanesInPrefix lane column mentioned
  have transcriptSatisfied :
      Satisfies
        (PiRlcCanonicalSamplerProgram.transcriptRows
          duplexBase constants lanes)
        transcriptAssignment := by
    exact PiRlcCanonicalSymbolicMachineHonest.fixedRows_honest
      duplexBase constants lanes initial lanesBefore positive
      initialCanonical constantWire
  have transcriptCanonical :
      ∀ column, transcriptAssignment column < goldilocksP :=
    PiRlcCanonicalSymbolicMachineHonest.fixedWitness_residues
      duplexBase constants lanes initial initialCanonical
  have transcriptWire : transcriptAssignment 0 = 1 := by
    exact (PiRlcCanonicalSymbolicMachineHonest.fixedWitness_constantWire
      duplexBase constants lanes initial positive).trans constantWire
  have transcriptBeforeBase :
      ∀ {column}, column < duplexBase →
        transcriptAssignment column = initial column := by
    intro column before
    unfold transcriptAssignment
      PiRlcCanonicalSymbolicMachineHonest.fixedWitness
    apply SymbolicDuplexHonest.witnesses_preserve_before
      (boundary := duplexBase) (column := column)
    · intro entry _member
      simp only [SymbolicDuplexHonest.outputBase,
        SymbolicDuplexHonest.callBase, SymbolicDuplex.stride]
      omega
    · exact before
  have u64Positive :
      0 < PiRlcCanonicalSamplerProgram.u64Base duplexBase := by
    unfold PiRlcCanonicalSamplerProgram.u64Base
    omega
  have u64Canonical : ∀ column, u64Assignment column < goldilocksP := by
    exact PiRlcCanonicalU64Honest.batchWitness_canonical field
      duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
      PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
      transcriptAssignment transcriptCanonical
  have u64Wire : u64Assignment 0 = 1 := by
    exact
      (PiRlcCanonicalU64Honest.batchWitness_before_u64Base field
        duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
        PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
        transcriptAssignment (by omega)).trans transcriptWire
  have u64Satisfied :
      Satisfies
        (PiRlcCanonicalU64.rows duplexBase
          (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder)
        u64Assignment := by
    exact PiRlcCanonicalU64Honest.rows_complete field duplexBase
      (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
      PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
      transcriptAssignment u64Positive transcriptWire u64Below
  have transcriptAtU64 :
      Satisfies
        (PiRlcCanonicalSamplerProgram.transcriptRows
          duplexBase constants lanes)
        u64Assignment := by
    apply KHornerSupport.satisfies_extend _
      transcriptAssignment u64Assignment
    · intro row member column mentioned
      have classified :=
        PiRlcCanonicalSymbolicMachineHonest.fixedRows_conservation
          duplexBase constants lanes positive lanesInPrefix
          row member column mentioned
      have beforeU64 :
          column < PiRlcCanonicalSamplerProgram.u64Base duplexBase := by
        rcases classified with beforeBase | allocated
        · exact Nat.lt_trans beforeBase (by
            simp [PiRlcCanonicalSamplerProgram.u64Base,
              PiRlcCanonicalSamplerProgram.transcriptCalls,
              SymbolicDuplex.stride])
        · have below :=
            SymbolicDuplexPhysical.temporaryColumns_lt_end
              duplexBase PiRlcCanonicalSamplerProgram.transcriptCalls
              column
              (by simpa
                  [PiRlcCanonicalSamplerProgram.transcriptRows,
                    PiRlcCanonicalSamplerProgram.transcriptAllocation,
                    PiRlcCanonicalSymbolicMachineHonest.fixedAllocation,
                    PiRlcCanonicalSamplerProgram.transcriptCalls]
                  using allocated)
          simpa [PiRlcCanonicalSamplerProgram.u64Base] using below
      change transcriptAssignment column =
        PiRlcCanonicalU64Honest.batchWitness field
          duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
          transcriptAssignment column
      exact
        (PiRlcCanonicalU64Honest.batchWitness_before_u64Base field
          duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
          transcriptAssignment beforeU64).symm
    · exact transcriptSatisfied
  have validFixed :
      SymbolicDuplexSemantics.Valid duplexBase constants u64Assignment
        (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder
          duplexBase lanes) := by
    exact SymbolicDuplexSemantics.valid_of_satisfied
      duplexBase constants
      (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder duplexBase lanes)
      u64Assignment u64Canonical u64Wire transcriptAtU64
  have validBatch :
      SymbolicDuplexSemantics.Valid duplexBase constants u64Assignment
        (PiRlcCanonicalSymbolicMachine.stateAt duplexBase initialBuilder
          PiRlcCanonicalSamplerProgram.coordinateCount) := by
    simpa [initialBuilder,
      PiRlcCanonicalSymbolicMachineHonest.fixedBuilder,
      PiRlcCanonicalSamplerProgram.coordinateCount] using validFixed
  intro coordinate
  have candidatesEqual :
      PiRlcCanonicalSamplerHonest.honestCandidates field
          duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
          transcriptAssignment u64Below candidateBelow coordinate =
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.candidatePrefix
          (PiRlcCanonicalMachine.machine constants)
          (SymbolicDuplexSemantics.decodedBuilder u64Assignment
            initialBuilder)
          coordinate.val := by
    calc
      PiRlcCanonicalSamplerHonest.honestCandidates field
            duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
            (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
            PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
            transcriptAssignment u64Below candidateBelow coordinate =
          PiRlcCanonicalSamplerSound.semanticCandidates prime duplexBase
            (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
            (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
            PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
            u64Canonical u64Wire u64Satisfied coordinate :=
        honestCandidates_eq_semanticCandidates prime field duplexBase
          (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
          transcriptAssignment u64Below candidateBelow coordinate
          u64Canonical u64Wire u64Satisfied
      _ = _ :=
        PiRlcCanonicalSamplerCheckerRefinement.semanticCandidates_eq_candidatePrefix
          prime duplexBase
          (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount constants
          initialBuilder u64Canonical u64Wire u64Satisfied validBatch
          coordinate
  have decodedInitial :
      SymbolicDuplexSemantics.decodedBuilder u64Assignment initialBuilder =
        SymbolicDuplexSemantics.decodedBuilder initial initialBuilder := by
    have laneValuesEqual :
        SymbolicDuplexSemantics.evalState u64Assignment
            initialBuilder.lanes =
          SymbolicDuplexSemantics.evalState initial initialBuilder.lanes := by
      funext lane
      unfold SymbolicDuplexSemantics.evalState
      apply KMulHonest.lcEval_congr
      intro column mentioned
      have beforeBase := lanesInPrefix lane column mentioned
      have beforeU64 :
          column < PiRlcCanonicalSamplerProgram.u64Base duplexBase :=
        Nat.lt_trans beforeBase (by
          simp [PiRlcCanonicalSamplerProgram.u64Base,
            PiRlcCanonicalSamplerProgram.transcriptCalls,
            SymbolicDuplex.stride])
      change
        PiRlcCanonicalU64Honest.batchWitness field
            duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
            PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
            transcriptAssignment column =
          initial column
      exact
        (PiRlcCanonicalU64Honest.batchWitness_before_u64Base field
          duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
          transcriptAssignment beforeU64).trans
            (transcriptBeforeBase beforeBase)
    change
      Poseidon2Duplex.State.mk
          (SymbolicDuplexSemantics.evalState u64Assignment
            initialBuilder.lanes)
          initialBuilder.absorbed =
        Poseidon2Duplex.State.mk
          (SymbolicDuplexSemantics.evalState initial initialBuilder.lanes)
          initialBuilder.absorbed
    exact congrArg
      (fun values =>
        Poseidon2Duplex.State.mk values initialBuilder.absorbed)
      laneValuesEqual
  have selectedCandidatesEqual :
      PiRlcCanonicalSamplerHonest.honestCandidates field
          duplexBase (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
          transcriptAssignment u64Below candidateBelow coordinate =
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.candidatePrefix
          (PiRlcCanonicalMachine.machine constants)
          (SymbolicDuplexSemantics.decodedBuilder initial initialBuilder)
          coordinate.val := by
    simpa [decodedInitial] using candidatesEqual
  let execution := bound.batch.execution coordinate
  have success :
      FirstAccepted.boundedSample ProductionAlphabet.verifier
          PiRlcCanonicalSelector.outputCount
          (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.candidatePrefix
            (PiRlcCanonicalMachine.machine constants)
            (SymbolicDuplexSemantics.decodedBuilder initial initialBuilder)
            coordinate.val) =
        some execution.output := by
    exact FirstAccepted.boundedSample_eq_some_iff_boundedExecution.mpr
      ⟨execution, rfl⟩
  have enoughPrefix :=
    (FirstAccepted.boundedSample_eq_some_iff.mp success).1
  rw [selectedCandidatesEqual]
  exact enoughPrefix

/-- The named honest sampler assignment writes exactly the challenge
coordinates carried by the bounded selected execution.  The `Enough`
argument used by the selector is constructed from `bound`; neither the
selected output nor a second candidate list is supplied by the caller. -/
theorem honestAssignment_output_eq_bound
    (prime : EuclidPrime goldilocksP)
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (lanes : Poseidon2Core.State)
    (initial : Nat → Nat)
    (positive : 0 < duplexBase)
    (lanesInPrefix :
      ∀ lane : Fin Poseidon2Core.width,
        SymbolicDuplexPlacement.ValueInPrefix duplexBase (lanes lane))
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    {challenges :
      Fin PiRlcCanonicalSamplerProgram.coordinateCount →
        Nightstream.SuperNeo.Concrete.RingF}
    (bound :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound
        (PiRlcCanonicalMachine.machine constants)
        (SymbolicDuplexSemantics.decodedBuilder initial
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
        challenges)
    (coordinate : Fin PiRlcCanonicalSamplerProgram.coordinateCount)
    (position : Fin PiRlcCanonicalSelector.outputCount) :
    PiRlcCanonicalSamplerProgram.honestAssignment field duplexBase constants
          lanes initial
          (honestEnough_of_bound prime field duplexBase constants lanes
            initial positive lanesInPrefix initialCanonical constantWire
            bound)
        (PiRlcCanonicalSelector.outputColumn
          (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
          coordinate position) =
      (challenges coordinate
        (PiRlcCanonicalSamplerCheckerRefinement.outputRingPosition
          position)).val := by
  let enough :=
    honestEnough_of_bound prime field duplexBase constants lanes initial
      positive lanesInPrefix initialCanonical constantWire bound
  let assignment :=
    PiRlcCanonicalSamplerProgram.honestAssignment field duplexBase constants
      lanes initial enough
  let initialBuilder :=
    PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes
  have assignmentCanonical : ∀ column, assignment column < goldilocksP :=
    PiRlcCanonicalSamplerProgram.honestAssignment_canonical field duplexBase
      constants lanes initial initialCanonical enough
  have assignmentWire : assignment 0 = 1 :=
    PiRlcCanonicalSamplerProgram.honestAssignment_constantWire field
      duplexBase constants lanes initial enough positive constantWire
  have assignmentSatisfied :
      Satisfies
        (PiRlcCanonicalSamplerProgram.rows duplexBase constants lanes)
        assignment :=
    PiRlcCanonicalSamplerProgram.honestAssignment_satisfies field duplexBase
      constants lanes initial positive lanesInPrefix initialCanonical
      constantWire enough
  have decodedEqual :
      SymbolicDuplexSemantics.decodedBuilder assignment initialBuilder =
        SymbolicDuplexSemantics.decodedBuilder initial initialBuilder := by
    apply congrArg
      (fun laneValues =>
        Poseidon2Duplex.State.mk laneValues initialBuilder.absorbed)
    funext lane
    apply KMulHonest.lcEval_congr
    intro column mentioned
    exact PiRlcCanonicalSamplerProgram.honestAssignment_before_base field
      duplexBase constants lanes initial enough
      (lanesInPrefix lane column mentioned)
  have sampledAssignment :=
    PiRlcCanonicalSamplerCheckerRefinement.samplerRows_sampleChallenge?_eq_some
      prime duplexBase constants lanes assignmentCanonical assignmentWire
      assignmentSatisfied coordinate
  have sampledBound :=
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?_eq_some_of_execution
      (bound.batch.execution coordinate)
  have semanticEqual :
      PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge
          prime duplexBase
          (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
          assignmentCanonical assignmentWire
          (PiRlcCanonicalSamplerProgram.u64Rows_satisfied duplexBase constants
            lanes assignment assignmentSatisfied)
          coordinate =
        challenges coordinate := by
    apply Option.some.inj
    calc
      some
          (PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge
            prime duplexBase
            (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
            (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
            PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
            assignmentCanonical assignmentWire
            (PiRlcCanonicalSamplerProgram.u64Rows_satisfied
              duplexBase constants lanes assignment assignmentSatisfied)
            coordinate) =
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?
            (PiRlcCanonicalMachine.machine constants)
            (SymbolicDuplexSemantics.decodedBuilder assignment initialBuilder)
            coordinate.val :=
        sampledAssignment.symm
      _ =
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?
            (PiRlcCanonicalMachine.machine constants)
            (SymbolicDuplexSemantics.decodedBuilder initial initialBuilder)
            coordinate.val := by rw [decodedEqual]
      _ = some
          (Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.challenge
            bound.batch coordinate) :=
        sampledBound
      _ = some (challenges coordinate) := by
        rw [bound.challenges_eq coordinate]
  have outputEqual :=
    PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge_coordinate_eq_outputColumn
      prime duplexBase
      (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
      (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
      (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
      PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
      assignmentCanonical assignmentWire
      (PiRlcCanonicalSamplerProgram.u64Rows_satisfied duplexBase constants
        lanes assignment assignmentSatisfied)
      (PiRlcCanonicalSamplerProgram.candidateRows_satisfied duplexBase
        constants lanes assignment assignmentSatisfied)
      (PiRlcCanonicalSamplerProgram.selectorRows_satisfied duplexBase
        constants lanes assignment assignmentSatisfied)
      coordinate position
  change assignment
      (PiRlcCanonicalSelector.outputColumn
        (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
        coordinate position) =
    (challenges coordinate
      (PiRlcCanonicalSamplerCheckerRefinement.outputRingPosition
        position)).val
  calc
    assignment
        (PiRlcCanonicalSelector.outputColumn
          (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
          coordinate position) =
        (PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge
          prime duplexBase
          (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initialBuilder
          assignmentCanonical assignmentWire
          (PiRlcCanonicalSamplerProgram.u64Rows_satisfied duplexBase constants
            lanes assignment assignmentSatisfied)
          coordinate
          (PiRlcCanonicalSamplerCheckerRefinement.outputRingPosition
            position)).val :=
      outputEqual.symm
    _ = _ := congrArg
      (fun challenge =>
        (challenge
          (PiRlcCanonicalSamplerCheckerRefinement.outputRingPosition
            position)).val)
      semanticEqual

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonestRefinement
