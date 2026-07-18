import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler.Rows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.ScalarLanes

/-!
Three-matrix diagnostic lane semantics for each PiRLC sampler scalar.

Owns: projection of the artifact layout into the profile-independent
`ScalarLanes` interface, the exact block-major accepted-count chain, and
composition of canonical-lane and residual-sampler row refinement.

Does not own: Poseidon2 provenance of field columns, the 54-of-64 selection
tail, source-row generation, ring assembly, costs, or row removal.

Emits constraints: no.

Authority boundary: canonical-u64 rows prove only a unique field/chunk view.
They become transcript candidates only after a separate Poseidon2 schedule
theorem. The sampler residual rows determine acceptance and symbols for those
already-derived chunks.

| Branch | Mathematical obligation | Lean owner |
|---|---|---|
| `challenge.transcript.lane_bit_decomposition` | canonical field lane gives four ordered 16-bit chunks | `Rows.accepted_canonicalLane_refines` |
| `challenge.sampler.chunk` | four verifier-owned accept/symbol/count transitions per lane | `Rows.accepted_laneRows` |
| `challenge.sampler.count_chain` | 16 lane leaves form one zero-rooted chain | `counterChain` |
| `challenge.sampler.scalar` | all 64 field-derived candidates refine | `accepted_refines_lanes` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.ScalarSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout

/-- One active scalar projected into the reusable sixteen-lane interface. -/
def layout (rho : Fin scalarCount) :
    PiRlcChallenge.Sampler.Refinement.ScalarLanes.Layout :=
  { fieldColumn := fieldColumn rho
    bitStart := bitStart rho
    initialCountColumn := initialCountColumn rho
    predecessor := predecessorColumn rho }

/-- Exact active predecessor links, proved from named layout formulas rather
than inferred from row adjacency. -/
private theorem firstCounterColumn : ∀ rho : Fin scalarCount,
    (layout rho).predecessor
        (PiRlcChallenge.Sampler.Refinement.ScalarLanes.blockAt
          PiRlcChallenge.Sampler.Refinement.ScalarLanes.zeroIndex)
        (PiRlcChallenge.Sampler.Refinement.ScalarLanes.laneAt
          PiRlcChallenge.Sampler.Refinement.ScalarLanes.zeroIndex) =
      (layout rho).initialCountColumn := by
  decide

private theorem successorCounterColumn :
    ∀ (rho : Fin scalarCount)
      (index : Fin PiRlcChallenge.Sampler.Refinement.ScalarLanes.laneCount),
      ∀ nonzero : index.val ≠ 0,
        (layout rho).predecessor
            (PiRlcChallenge.Sampler.Refinement.ScalarLanes.blockAt index)
            (PiRlcChallenge.Sampler.Refinement.ScalarLanes.laneAt index) =
          PiRlcChallenge.Sampler.Refinement.LaneRows.finalCountColumn
            ((layout rho).bitStart
              (PiRlcChallenge.Sampler.Refinement.ScalarLanes.blockAt
                (PiRlcChallenge.Sampler.Refinement.ScalarLanes.previous index
                  nonzero))
              (PiRlcChallenge.Sampler.Refinement.ScalarLanes.laneAt
                (PiRlcChallenge.Sampler.Refinement.ScalarLanes.previous index
                  nonzero))) := by
  decide

theorem counterChain (rho : Fin scalarCount) :
    PiRlcChallenge.Sampler.Refinement.ScalarLanes.CounterChain
      (layout rho) := by
  exact {
    first := firstCounterColumn rho
    successor := successorCounterColumn rho
  }

/-- Accepted active source leaves prove the complete field-derived
sixteen-lane sampler result. Transcript provenance remains open. -/
theorem accepted_refines_lanes
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    PiRlcChallenge.Sampler.Refinement.ScalarLanes.Refines
      assignment canonical (layout rho) := by
  exact
    PiRlcChallenge.Sampler.Refinement.ScalarLanes.satisfyingSamplerRows_refine
      prime canonical one (layout rho) (counterChain rho)
      (fun block lane =>
        Rows.accepted_canonicalLane_refines
          prime canonical one accepted rho block lane)
      (fun block lane => Rows.accepted_laneRows accepted rho block lane)
      (Rows.accepted_initialCount_zero canonical one accepted rho)

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.ScalarSemantics
