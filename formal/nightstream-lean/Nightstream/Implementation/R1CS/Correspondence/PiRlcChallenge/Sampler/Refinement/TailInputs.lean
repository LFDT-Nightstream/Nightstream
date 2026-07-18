import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.OneScalar
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.CandidateOrder
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Initialization

/-!
Exact source-order input correspondence for the production 54-of-64 sampler
tail.

Owns: the recursive-profile placement of the three tail input families
(accept, centered symbol, and cumulative count) at the independently defined
candidate addresses.

Does not own: selector semantics, first-accepted correctness, cumulative-count
meaning, coefficient assembly, Rust trace conformance, row removal, or cost
totals.

Emits constraints: no.

Authority boundary: `CandidateOrder` defines source order independently of the
generated tail map. The finite column-map theorems here prove that the recursive
production artifact uses that order. They do not infer candidate decisions from
column adjacency.

| Protocol | Phase | Constraint family | Mathematical obligation | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/tail inputs | candidate address | `candidate = 16*block + 4*lane + part` | every `Fin 64` index has one exact hierarchical address |
| `Pi_RLC` | sampler/tail inputs | accept source | candidate leaf accept wire | local tail accept column is the corresponding lane leaf wire |
| `Pi_RLC` | sampler/tail inputs | symbol source | candidate leaf centered-symbol wire | local tail symbol column is the corresponding lane leaf wire |
| `Pi_RLC` | sampler/tail inputs | cumulative source | accepted count through candidate | local tail cumulative column is the corresponding lane leaf wire |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailInputs

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.CandidateOrder
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Global accept-wire base belonging to one hierarchical candidate leaf. -/
def productionBase (candidate : Fin SelectionRows.candidateCount) : Nat :=
  let location := address candidate
  ChunkOrder.bitStart location.block location.lane + 66 + 23 * location.part.val

private def tailColumnMap : List Nat :=
  AlphabetSamplingResidualTemplate.tailColumnMap
    OneScalarRows.tailBitStarts OneScalarRows.tailFirstAllocated

/-! These are finite implementation-refinement facts, not semantic sampler
theorems. `decide` checks all 64 source-order positions in the kernel. -/

theorem acceptColumnMap : ∀ candidate : Fin SelectionRows.candidateCount,
    Relabel.column tailColumnMap
        (SelectionRows.acceptCol candidate.val) =
      productionBase candidate := by
  decide

theorem symbolColumnMap : ∀ candidate : Fin SelectionRows.candidateCount,
    Relabel.column tailColumnMap
        (SelectionRows.symbolCol candidate.val) =
      productionBase candidate + 21 := by
  decide

theorem cumulativeColumnMap : ∀ candidate : Fin SelectionRows.candidateCount,
    Relabel.column tailColumnMap
        (SelectionRows.cumulativeCol candidate.val) =
      productionBase candidate + 22 := by
  decide

/-- Apart from candidate zero, the tail prefix wire and the lane leaf's prior
counter are the very same production column. Candidate zero is intentionally
handled by the two independently checked zero-initialization equations. -/
theorem priorColumnMap : ∀ candidate : Fin SelectionRows.candidateCount,
    candidate.val ≠ 0 ->
      Relabel.column tailColumnMap
          (SelectionRows.prefixCol candidate.val) =
        Relabel.column
          (AlphabetSamplingResidualTemplate.laneColumnMap
            (ChunkOrder.bitStart (address candidate).block
              (address candidate).lane)
            (OneScalarRows.cumPrev (address candidate).block
              (address candidate).lane))
          (ChunkRows.priorCumulativeCol (address candidate).part.val) := by
  decide

theorem localAssignment_accept
    (assignment : Nat -> Nat)
    (candidate : Fin SelectionRows.candidateCount) :
    TailRows.localAssignment assignment
        (SelectionRows.acceptCol candidate.val) =
      assignment (productionBase candidate) := by
  unfold TailRows.localAssignment TailRows.localAssignmentAt Relabel.assignment
  exact congrArg assignment (acceptColumnMap candidate)

theorem localAssignment_symbol
    (assignment : Nat -> Nat)
    (candidate : Fin SelectionRows.candidateCount) :
    TailRows.localAssignment assignment
        (SelectionRows.symbolCol candidate.val) =
      assignment (productionBase candidate + 21) := by
  unfold TailRows.localAssignment TailRows.localAssignmentAt Relabel.assignment
  exact congrArg assignment (symbolColumnMap candidate)

theorem localAssignment_cumulative
    (assignment : Nat -> Nat)
    (candidate : Fin SelectionRows.candidateCount) :
    TailRows.localAssignment assignment
        (SelectionRows.cumulativeCol candidate.val) =
      assignment (productionBase candidate + 22) := by
  unfold TailRows.localAssignment TailRows.localAssignmentAt Relabel.assignment
  exact congrArg assignment (cumulativeColumnMap candidate)

theorem localAssignment_prefix_eq_lanePrior
    {assignment : Nat -> Nat}
    (candidate : Fin SelectionRows.candidateCount)
    (nonzero : candidate.val ≠ 0) :
    TailRows.localAssignment assignment
        (SelectionRows.prefixCol candidate.val) =
      LaneRows.localAssignment assignment
        (ChunkOrder.bitStart (address candidate).block
          (address candidate).lane)
        (OneScalarRows.cumPrev (address candidate).block
          (address candidate).lane)
        (ChunkRows.priorCumulativeCol (address candidate).part.val) := by
  unfold TailRows.localAssignment TailRows.localAssignmentAt
    LaneRows.localAssignment Relabel.assignment
  exact congrArg assignment (priorColumnMap candidate nonzero)

theorem localAssignment_accept_eq_lane
    (assignment : Nat -> Nat)
    (candidate : Fin SelectionRows.candidateCount) :
    TailRows.localAssignment assignment
        (SelectionRows.acceptCol candidate.val) =
      LaneRows.localAssignment assignment
        (ChunkOrder.bitStart (address candidate).block
          (address candidate).lane)
        (OneScalarRows.cumPrev (address candidate).block
          (address candidate).lane)
        (ChunkRows.acceptCol (address candidate).part.val) := by
  calc
    _ = assignment (productionBase candidate) :=
      localAssignment_accept assignment candidate
    _ = _ := (LaneRows.localAssignment_accept assignment
      (ChunkOrder.bitStart (address candidate).block
        (address candidate).lane)
      (OneScalarRows.cumPrev (address candidate).block
        (address candidate).lane)
      (address candidate).part (address candidate).part.isLt).symm.trans (by
        rfl)

theorem localAssignment_cumulative_eq_lane
    (assignment : Nat -> Nat)
    (candidate : Fin SelectionRows.candidateCount) :
    TailRows.localAssignment assignment
        (SelectionRows.cumulativeCol candidate.val) =
      LaneRows.localAssignment assignment
        (ChunkOrder.bitStart (address candidate).block
          (address candidate).lane)
        (OneScalarRows.cumPrev (address candidate).block
          (address candidate).lane)
        (ChunkRows.cumulativeCol (address candidate).part.val) := by
  calc
    _ = assignment (productionBase candidate + 22) :=
      localAssignment_cumulative assignment candidate
    _ = _ := (LaneRows.localAssignment_cumulative assignment
      (ChunkOrder.bitStart (address candidate).block
        (address candidate).lane)
      (OneScalarRows.cumPrev (address candidate).block
        (address candidate).lane)
      (address candidate).part (address candidate).part.isLt).symm.trans (by
        rfl)

/-- The independent transcript-machine candidate at one source-order address. -/
def machineCandidate
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (candidate : Fin SelectionRows.candidateCount) :
    ProductionAlphabet.Chunk :=
  let location := address candidate
  (digestBlock
      (ChunkOrder.machineBlockInput assignment canonical location.block)
      location.block.val).2
    (ChunkOrder.chunkPosition location.lane location.part)

/-- One tail candidate input is tied simultaneously to the transcript machine
and the verifier-owned rejection/decoding functions. -/
structure CandidateRefines
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (candidate : Fin SelectionRows.candidateCount) : Prop where
  accept : TailRows.localAssignment assignment
      (SelectionRows.acceptCol candidate.val) =
    if ProductionAlphabet.verifier.accepts
        (machineCandidate assignment canonical candidate) then 1 else 0
  symbol : TailRows.localAssignment assignment
      (SelectionRows.symbolCol candidate.val) =
    CandidateOrder.centeredField (ProductionAlphabet.verifier.symbol
      (machineCandidate assignment canonical candidate))

/-- Exact production rows refine both source-order transcript candidates and
the independent verifier-owned decision for every candidate input of the tail. -/
theorem candidate_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : OneScalar.Refines prime canonical one accepted)
    (candidate : Fin SelectionRows.candidateCount) :
    CandidateRefines assignment canonical candidate := by
  let location := address candidate
  have laneResult := refinement.lane location.block location.lane
  refine {
    accept := ?_
    symbol := ?_
  }
  · calc
      TailRows.localAssignment assignment
          (SelectionRows.acceptCol candidate.val) =
          assignment (productionBase candidate) :=
        localAssignment_accept assignment candidate
      _ = LaneRows.localAssignment assignment
          (ChunkOrder.bitStart location.block location.lane)
          (OneScalarRows.cumPrev location.block location.lane)
          (ChunkRows.acceptCol location.part.val) := by
        exact (LaneRows.localAssignment_accept assignment
          (ChunkOrder.bitStart location.block location.lane)
          (OneScalarRows.cumPrev location.block location.lane)
          location.part location.part.isLt).symm.trans (by
            rfl)
      _ = Lane.acceptedBit
          (LaneRows.localAssignment assignment
            (ChunkOrder.bitStart location.block location.lane)
            (OneScalarRows.cumPrev location.block location.lane))
          (LaneRows.sourceBitsBoolean
            (ChunkOrder.accepted_refines_lane prime canonical one accepted
              location.block location.lane))
          location.part :=
        laneResult.production.semantics.accepted location.part
      _ = if ProductionAlphabet.verifier.accepts
            (machineCandidate assignment canonical candidate) then 1 else 0 := by
        unfold Lane.acceptedBit machineCandidate
        rw [laneResult.machineCandidates location.part]
  · calc
      TailRows.localAssignment assignment
          (SelectionRows.symbolCol candidate.val) =
          assignment (productionBase candidate + 21) :=
        localAssignment_symbol assignment candidate
      _ = LaneRows.localAssignment assignment
          (ChunkOrder.bitStart location.block location.lane)
          (OneScalarRows.cumPrev location.block location.lane)
          (ChunkRows.symbolCol location.part.val) := by
        exact (LaneRows.localAssignment_symbol assignment
          (ChunkOrder.bitStart location.block location.lane)
          (OneScalarRows.cumPrev location.block location.lane)
          location.part location.part.isLt).symm.trans (by
            rfl)
      _ = Lane.expectedSymbol
          (LaneRows.localAssignment assignment
            (ChunkOrder.bitStart location.block location.lane)
            (OneScalarRows.cumPrev location.block location.lane))
          (LaneRows.sourceBitsBoolean
            (ChunkOrder.accepted_refines_lane prime canonical one accepted
              location.block location.lane))
          location.part :=
        laneResult.production.semantics.symbols location.part
      _ = CandidateOrder.centeredField (ProductionAlphabet.verifier.symbol
            (machineCandidate assignment canonical candidate)) := by
        unfold Lane.expectedSymbol CandidateOrder.centeredField machineCandidate
        rw [laneResult.machineCandidates location.part]

/-- The tail's prefix input is exactly the corresponding lane leaf's prior
counter. Candidate zero is the only non-column-identical case; both sides are
proved zero by their independently owned initialization equations. -/
theorem prefix_refines_lanePrior
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : OneScalar.Refines prime canonical one accepted)
    (candidate : Fin SelectionRows.candidateCount) :
    TailRows.localAssignment assignment
        (SelectionRows.prefixCol candidate.val) =
      LaneRows.localAssignment assignment
        (ChunkOrder.bitStart (address candidate).block
          (address candidate).lane)
        (OneScalarRows.cumPrev (address candidate).block
          (address candidate).lane)
        (ChunkRows.priorCumulativeCol (address candidate).part.val) := by
  by_cases nonzero : candidate.val ≠ 0
  · exact localAssignment_prefix_eq_lanePrior candidate nonzero
  · have valueZero : candidate.val = 0 := by
      simpa using nonzero
    have candidateZero : candidate = ⟨0, by decide⟩ := by
      apply Fin.ext
      exact valueZero
    subst candidate
    have tailZero :=
      Selection.Initialization.zeroPrefix_eq_zero
        (TailRows.canonical canonical) (TailRows.constantOne one)
        (TailRows.accepted_satisfies accepted)
    rw [show SelectionRows.prefixCol (⟨0, by decide⟩ :
      Fin SelectionRows.candidateCount).val = SelectionRows.zeroPrefixCol by
        rfl]
    rw [tailZero]
    change 0 = assignment OneScalarRows.initialCountColumn
    exact refinement.initialCountZero.symm

/-- Every cumulative wire is an integer recurrence step from the exact prior
prefix and the verifier-owned accept bit carried by the same candidate leaf. -/
theorem cumulative_step
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : OneScalar.Refines prime canonical one accepted)
    (candidate : Fin SelectionRows.candidateCount) :
    TailRows.localAssignment assignment
        (SelectionRows.cumulativeCol candidate.val) =
      TailRows.localAssignment assignment
          (SelectionRows.prefixCol candidate.val) +
        TailRows.localAssignment assignment
          (SelectionRows.acceptCol candidate.val) := by
  let location := address candidate
  have laneResult := refinement.lane location.block location.lane
  calc
    TailRows.localAssignment assignment
        (SelectionRows.cumulativeCol candidate.val) =
      LaneRows.localAssignment assignment
        (ChunkOrder.bitStart location.block location.lane)
        (OneScalarRows.cumPrev location.block location.lane)
        (ChunkRows.cumulativeCol location.part.val) := by
      simpa [location] using
        localAssignment_cumulative_eq_lane assignment candidate
    _ = LaneRows.localAssignment assignment
          (ChunkOrder.bitStart location.block location.lane)
          (OneScalarRows.cumPrev location.block location.lane)
          (ChunkRows.priorCumulativeCol location.part.val) +
        Lane.acceptedBit
          (LaneRows.localAssignment assignment
            (ChunkOrder.bitStart location.block location.lane)
            (OneScalarRows.cumPrev location.block location.lane))
          (LaneRows.sourceBitsBoolean
            (ChunkOrder.accepted_refines_lane prime canonical one accepted
              location.block location.lane))
          location.part :=
      laneResult.production.semantics.cumulative location.part
    _ = TailRows.localAssignment assignment
          (SelectionRows.prefixCol candidate.val) +
        TailRows.localAssignment assignment
          (SelectionRows.acceptCol candidate.val) := by
      rw [← laneResult.production.semantics.accepted location.part]
      rw [prefix_refines_lanePrior prime canonical one accepted refinement candidate]
      rw [localAssignment_accept_eq_lane assignment candidate]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailInputs
