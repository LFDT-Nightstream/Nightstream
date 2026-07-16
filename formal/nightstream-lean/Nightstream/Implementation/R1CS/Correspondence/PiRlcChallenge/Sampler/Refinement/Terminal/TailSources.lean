import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.TailRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Initialization

/-!
Terminal-profile source bindings between lane leaves and the 54-of-64 tail.

Assurance tier: implementation/R1CS correspondence. This file instantiates the
profile-independent lane-to-tail interface for every terminal scalar and proves
the exact source of each accept, symbol, cumulative, and prior-prefix input.

Owns: the terminal tail layout; finite column-map equalities for all 15 times 64
source positions; the independently initialized candidate-zero prefix; and the
complete `SourceBindings` instance used by generic candidate semantics.

Does not own: Poseidon2 transcript provenance, 54-of-64 first-accepted
selection, scalar-to-scalar state chaining, coefficient assembly, Rust trace
conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: nonzero prefix inputs are exact column identities. The
candidate-zero tail prefix and first lane counter are different columns and are
proved equal only because separate readable equations force both to integer
zero. Neither adjacency nor a generated count is treated as authority.

| Protocol | Phase | Constraint family | Terminal input | Proven result |
|---|---|---|---|---|
| `Pi_RLC` | scalar `rho` | tail layout | affine lane and tail columns | exact generic layout for each `rho : Fin 15` |
| `Pi_RLC` | lane/tail boundary | accept/symbol/cumulative | 64 source positions | exact tail-to-lane column identity |
| `Pi_RLC` | accepted-prefix chain | nonzero prior | candidates 1 through 63 | exact prior-column identity |
| `Pi_RLC` | accepted-prefix chain | candidate-zero prior | two independent zero equations | tail prefix zero equals first lane counter zero |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

/-- Exact terminal columns projected into the generic lane-to-tail interface. -/
def layout (rho : Fin ScalarRows.scalarCount) :
    TailCandidateSemantics.Layout :=
  { lanes := ScalarSemantics.layout rho
    tailBitStarts := ScalarRows.tailBitStarts rho
    tailFirstAllocated := ScalarRows.tailFirstAllocated rho }

private def tailColumnMap (rho : Fin ScalarRows.scalarCount) : List Nat :=
  AlphabetSamplingResidualTemplate.tailColumnMap
    (ScalarRows.tailBitStarts rho) (ScalarRows.tailFirstAllocated rho)

private def laneColumnMap
    (rho : Fin ScalarRows.scalarCount)
    (candidate : Fin SelectionRows.candidateCount) : List Nat :=
  AlphabetSamplingResidualTemplate.laneColumnMap
    ((layout rho).lanes.bitStart
      (ScalarLanes.blockAt (TailCandidateSemantics.laneIndex candidate))
      (ScalarLanes.laneAt (TailCandidateSemantics.laneIndex candidate)))
    ((layout rho).lanes.predecessor
      (ScalarLanes.blockAt (TailCandidateSemantics.laneIndex candidate))
      (ScalarLanes.laneAt (TailCandidateSemantics.laneIndex candidate)))

/-! These are closed placement theorems. Kernel evaluation checks every
terminal scalar and candidate address against the independently named maps. -/

theorem acceptColumnMap :
    forall (rho : Fin ScalarRows.scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
      Relabel.column (tailColumnMap rho)
          (SelectionRows.acceptCol candidate.val) =
        Relabel.column (laneColumnMap rho candidate)
          (ChunkRows.acceptCol
            (CandidateOrder.address candidate).part.val) := by
  decide

theorem symbolColumnMap :
    forall (rho : Fin ScalarRows.scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
      Relabel.column (tailColumnMap rho)
          (SelectionRows.symbolCol candidate.val) =
        Relabel.column (laneColumnMap rho candidate)
          (ChunkRows.symbolCol
            (CandidateOrder.address candidate).part.val) := by
  decide

theorem cumulativeColumnMap :
    forall (rho : Fin ScalarRows.scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
      Relabel.column (tailColumnMap rho)
          (SelectionRows.cumulativeCol candidate.val) =
        Relabel.column (laneColumnMap rho candidate)
          (ChunkRows.cumulativeCol
            (CandidateOrder.address candidate).part.val) := by
  decide

theorem nonzeroPriorColumnMap :
    forall (rho : Fin ScalarRows.scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
      candidate.val != 0 ->
      Relabel.column (tailColumnMap rho)
          (SelectionRows.prefixCol candidate.val) =
        Relabel.column (laneColumnMap rho candidate)
          (ChunkRows.priorCumulativeCol
            (CandidateOrder.address candidate).part.val) := by
  decide

/-- Accepted terminal rows establish the complete explicit lane-to-tail source
contract for one scalar. -/
theorem accepted_sourceBindings
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    TailCandidateSemantics.SourceBindings (layout rho) assignment := by
  let lanes := ScalarSemantics.accepted_refines_lanes prime canonical one
    accepted rho
  refine {
    accept := ?_
    symbol := ?_
    cumulative := ?_
    prior := ?_
  }
  · intro candidate
    unfold TailCandidateSemantics.localAssignment
      TailCandidateSemantics.laneAssignment
      PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
      LaneRows.localAssignment Relabel.assignment
    exact congrArg assignment (acceptColumnMap rho candidate)
  · intro candidate
    unfold TailCandidateSemantics.localAssignment
      TailCandidateSemantics.laneAssignment
      PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
      LaneRows.localAssignment Relabel.assignment
    exact congrArg assignment (symbolColumnMap rho candidate)
  · intro candidate
    unfold TailCandidateSemantics.localAssignment
      TailCandidateSemantics.laneAssignment
      PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
      LaneRows.localAssignment Relabel.assignment
    exact congrArg assignment (cumulativeColumnMap rho candidate)
  · intro candidate
    by_cases nonzero : candidate.val != 0
    · unfold TailCandidateSemantics.localAssignment
        TailCandidateSemantics.laneAssignment
        PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
        LaneRows.localAssignment Relabel.assignment
      exact congrArg assignment (nonzeroPriorColumnMap rho candidate nonzero)
    · have valueZero : candidate.val = 0 := by
        simpa using nonzero
      have candidateZero : candidate = ⟨0, by decide⟩ := by
        apply Fin.ext
        exact valueZero
      subst candidate
      have tailZero :=
        Selection.Initialization.zeroPrefix_eq_zero
          (TailRows.canonical canonical rho) (TailRows.constantOne one rho)
          (TailRows.accepted_satisfies accepted rho)
      have laneZero :
          TailCandidateSemantics.laneAssignment (layout rho) assignment
              ⟨0, by decide⟩
              (ChunkRows.priorCumulativeCol
                (CandidateOrder.address ⟨0, by decide⟩).part.val) = 0 := by
        change assignment (ScalarRows.initialCountColumn rho) = 0
        exact lanes.initialCountZero
      change TailCandidateSemantics.localAssignment (layout rho) assignment
          SelectionRows.zeroPrefixCol =
        TailCandidateSemantics.laneAssignment (layout rho) assignment
          ⟨0, by decide⟩
          (ChunkRows.priorCumulativeCol
            (CandidateOrder.address ⟨0, by decide⟩).part.val)
      rw [laneZero]
      simpa [layout, TailCandidateSemantics.localAssignment,
        TailRows.localAssignment] using tailZero

/-- All 64 terminal tail candidates inherit verifier-owned field-chunk
decision semantics. Poseidon2 provenance remains an explicit later theorem. -/
theorem accepted_candidate_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount)
    (candidate : Fin SelectionRows.candidateCount) :
    TailCandidateSemantics.CandidateRefines
      (layout rho) assignment canonical candidate := by
  exact TailCandidateSemantics.candidate_refines
    (ScalarSemantics.accepted_refines_lanes prime canonical one accepted rho)
    (accepted_sourceBindings prime canonical one accepted rho) candidate

/-- Every terminal candidate advances the accepted-prefix count by its own
verifier-derived accept bit. -/
theorem accepted_cumulative_step
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount)
    (candidate : Fin SelectionRows.candidateCount) :
    TailCandidateSemantics.localAssignment (layout rho) assignment
        (SelectionRows.cumulativeCol candidate.val) =
      TailCandidateSemantics.localAssignment (layout rho) assignment
          (SelectionRows.prefixCol candidate.val) +
        TailCandidateSemantics.localAssignment (layout rho) assignment
          (SelectionRows.acceptCol candidate.val) := by
  exact TailCandidateSemantics.cumulative_step
    (ScalarSemantics.accepted_refines_lanes prime canonical one accepted rho)
    (accepted_sourceBindings prime canonical one accepted rho) candidate

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources
