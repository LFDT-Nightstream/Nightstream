import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler.ScalarSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Initialization

/-!
Active lane-to-tail source bindings for the 54-of-64 PiRLC sampler.

Owns: the active tail layout; finite source-column equalities for all
`15 x 64` candidates; the separate candidate-zero proof; and the physical
output-column map for all `15 x 54` selected coefficients.

Does not own: first-accepted selection soundness, Poseidon2 provenance,
ring assembly, source artifact generation, costs, or row removal.

Emits constraints: no.

Authority boundary: candidates 1 through 63 use direct column identity.
Candidate zero does not: its selection-prefix wire and lane-chain initializer
are distinct columns, each proved zero by its own row. Collapsing them into an
all-candidate alias theorem would be false.

| Branch | Multiplicity | Binding |
|---|---:|---|
| accept / symbol / cumulative sources | `15 x 64` each | exact physical column identity |
| nonzero prior source | `15 x 63` | exact physical column identity |
| candidate-zero prior | `15` | two independently checked zero equations |
| selected output | `15 x 54` | local readable output aliases active projection rho column |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.TailSources

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout

/-- Active columns projected into the profile-independent lane-to-tail
interface. -/
def layout (rho : Fin scalarCount) :
    PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.Layout :=
  { lanes := ScalarSemantics.layout rho
    tailBitStarts := tailBitStarts rho
    tailFirstAllocated := tailFirstAllocated rho }

private def tailColumnMap (rho : Fin scalarCount) : List Nat :=
  AlphabetSamplingResidualTemplate.tailColumnMap
    (tailBitStarts rho) (tailFirstAllocated rho)

private def laneColumnMap
    (rho : Fin scalarCount)
    (candidate : Fin SelectionRows.candidateCount) : List Nat :=
  AlphabetSamplingResidualTemplate.laneColumnMap
    ((layout rho).lanes.bitStart
      (PiRlcChallenge.Sampler.Refinement.ScalarLanes.blockAt
        (PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex
          candidate))
      (PiRlcChallenge.Sampler.Refinement.ScalarLanes.laneAt
        (PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex
          candidate)))
    ((layout rho).lanes.predecessor
      (PiRlcChallenge.Sampler.Refinement.ScalarLanes.blockAt
        (PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex
          candidate))
      (PiRlcChallenge.Sampler.Refinement.ScalarLanes.laneAt
        (PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneIndex
          candidate)))

/-! Closed placement theorems. Kernel evaluation checks every active scalar
and finite candidate/output address against the independently named maps. -/

set_option maxHeartbeats 4000000 in
theorem acceptColumnMap :
    ∀ (rho : Fin scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
      Relabel.column (tailColumnMap rho)
          (SelectionRows.acceptCol candidate.val) =
        Relabel.column (laneColumnMap rho candidate)
          (ChunkRows.acceptCol
            (CandidateOrder.address candidate).part.val) := by
  set_option maxRecDepth 1000000 in
    decide

set_option maxHeartbeats 4000000 in
theorem symbolColumnMap :
    ∀ (rho : Fin scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
      Relabel.column (tailColumnMap rho)
          (SelectionRows.symbolCol candidate.val) =
        Relabel.column (laneColumnMap rho candidate)
          (ChunkRows.symbolCol
            (CandidateOrder.address candidate).part.val) := by
  set_option maxRecDepth 1000000 in
    decide

set_option maxHeartbeats 4000000 in
theorem cumulativeColumnMap :
    ∀ (rho : Fin scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
      Relabel.column (tailColumnMap rho)
          (SelectionRows.cumulativeCol candidate.val) =
        Relabel.column (laneColumnMap rho candidate)
          (ChunkRows.cumulativeCol
            (CandidateOrder.address candidate).part.val) := by
  set_option maxRecDepth 1000000 in
    decide

set_option maxHeartbeats 4000000 in
theorem nonzeroPriorColumnMap :
    ∀ (rho : Fin scalarCount)
      (candidate : Fin SelectionRows.candidateCount),
      candidate.val ≠ 0 →
      Relabel.column (tailColumnMap rho)
          (SelectionRows.prefixCol candidate.val) =
        Relabel.column (laneColumnMap rho candidate)
          (ChunkRows.priorCumulativeCol
            (CandidateOrder.address candidate).part.val) := by
  set_option maxRecDepth 1000000 in
    decide

set_option maxHeartbeats 4000000 in
theorem outputColumnMap :
    ∀ (rho : Fin scalarCount) (position : Fin outputCount),
      Relabel.column (tailColumnMap rho)
          (SelectionRows.outputCol position.val) =
        outputColumn rho position := by
  set_option maxRecDepth 1000000 in
    decide

/-- The readable local output is exactly the active physical output column. -/
theorem local_output_eq_physical
    (assignment : Nat → Nat)
    (rho : Fin scalarCount) (position : Fin outputCount) :
    PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
        (layout rho) assignment (SelectionRows.outputCol position.val) =
      assignment (outputColumn rho position) := by
  unfold PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
    PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
    Relabel.assignment
  exact congrArg assignment (outputColumnMap rho position)

/-- Accepted active rows establish all lane-to-tail source bindings for one
scalar. -/
theorem accepted_sourceBindings
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.SourceBindings
      (layout rho) assignment := by
  let lanes := ScalarSemantics.accepted_refines_lanes
    prime canonical one accepted rho
  refine {
    accept := ?_
    symbol := ?_
    cumulative := ?_
    prior := ?_
  }
  · intro candidate
    unfold PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
      PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneAssignment
      PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
      PiRlcChallenge.Sampler.Refinement.LaneRows.localAssignment
      Relabel.assignment
    exact congrArg assignment (acceptColumnMap rho candidate)
  · intro candidate
    unfold PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
      PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneAssignment
      PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
      PiRlcChallenge.Sampler.Refinement.LaneRows.localAssignment
      Relabel.assignment
    exact congrArg assignment (symbolColumnMap rho candidate)
  · intro candidate
    unfold PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
      PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneAssignment
      PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
      PiRlcChallenge.Sampler.Refinement.LaneRows.localAssignment
      Relabel.assignment
    exact congrArg assignment (cumulativeColumnMap rho candidate)
  · intro candidate
    by_cases nonzero : candidate.val ≠ 0
    · unfold PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
        PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneAssignment
        PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
        PiRlcChallenge.Sampler.Refinement.LaneRows.localAssignment
        Relabel.assignment
      exact congrArg assignment
        (nonzeroPriorColumnMap rho candidate nonzero)
    · have valueZero : candidate.val = 0 := by
        simpa using nonzero
      have candidateZero : candidate = ⟨0, by decide⟩ := by
        apply Fin.ext
        exact valueZero
      subst candidate
      have tailZero :=
        PiRlcChallenge.Sampler.Selection.Initialization.zeroPrefix_eq_zero
          (PiRlcChallenge.Sampler.Refinement.TailRows.canonicalAt
            (tailBitStarts rho) (tailFirstAllocated rho) canonical)
          (PiRlcChallenge.Sampler.Refinement.TailRows.constantOneAt
            (tailBitStarts rho) (tailFirstAllocated rho) one)
          (Rows.accepted_readableTail accepted rho)
      have laneZero :
          PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneAssignment
              (layout rho) assignment ⟨0, by decide⟩
              (ChunkRows.priorCumulativeCol
                (CandidateOrder.address ⟨0, by decide⟩).part.val) = 0 := by
        change assignment (initialCountColumn rho) = 0
        exact lanes.initialCountZero
      change
        PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
            (layout rho) assignment SelectionRows.zeroPrefixCol =
          PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.laneAssignment
            (layout rho) assignment ⟨0, by decide⟩
            (ChunkRows.priorCumulativeCol
              (CandidateOrder.address ⟨0, by decide⟩).part.val)
      rw [laneZero]
      simpa [layout,
        PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment]
        using tailZero

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.TailSources
