import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.ScalarLanes
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows

/-!
Terminal-profile instantiation of the profile-independent sixteen-lane
`Pi_RLC` sampler theorem.

Assurance tier: implementation/R1CS correspondence. For every one of the exact
fifteen terminal scalar coordinates, accepted owner rows imply canonical field
decomposition and verifier-owned accept/symbol/count semantics across all 64
candidates.

Owns: the terminal affine column layout; its exact predecessor-chain theorem;
and one `rho : Fin 15` instantiation of `ScalarLanes.satisfyingRows_refine`.

Does not own: Poseidon2 transcript provenance for the field columns, chaining
between scalar coordinates, 54-of-64 tail selection, coefficient assembly,
Rust trace conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: the terminal generated owner contributes only satisfying
rows. The generic semantic theorem, not owner adjacency or a legacy count,
determines what those rows prove. Until transcript refinement closes, these
results must not be described as sampled `rho` challenges.

| Protocol | Phase | Constraint family | Terminal owner input | Proven result |
|---|---|---|---|---|
| `Pi_RLC` | scalar `rho` | layout | affine field/bit/count columns | exact generic `Layout` instance |
| `Pi_RLC` | count chain | predecessor binding | sixteen lane predecessors | leaf zero starts at zero; every later leaf consumes the prior final count |
| `Pi_RLC` | digest decomposition | canonical-u64 | sixteen accepted canonical pieces | every field has a unique four-chunk representation |
| `Pi_RLC` | rejection sampler | four-candidate lane | sixteen accepted 104-row pieces | verifier-owned accept, symbol, and cumulative semantics |
| `Pi_RLC` | scalar `rho` | all 64 candidates | one accepted terminal owner | final accepted count is at most 64 |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement

/-- Exact terminal columns for one scalar coordinate, projected into the
profile-independent lane interface. -/
def layout (rho : Fin ScalarRows.scalarCount) : ScalarLanes.Layout :=
  { fieldColumn := ScalarRows.fieldColumn rho
    bitStart := ScalarRows.bitStart rho
    initialCountColumn := ScalarRows.initialCountColumn rho
    predecessor := ScalarRows.cumulativePredecessor rho }

private theorem nonzero_index_cases
    (index : Fin ScalarLanes.laneCount)
    (nonzero : index.val ≠ 0) :
    index.val = 1 ∨ index.val = 2 ∨ index.val = 3 ∨
    index.val = 4 ∨ index.val = 5 ∨ index.val = 6 ∨
    index.val = 7 ∨ index.val = 8 ∨ index.val = 9 ∨
    index.val = 10 ∨ index.val = 11 ∨ index.val = 12 ∨
    index.val = 13 ∨ index.val = 14 ∨ index.val = 15 := by
  have indexLt := index.isLt
  simp only [ScalarLanes.laneCount] at indexLt
  omega

/-- Exact terminal predecessor links, proved from the independent affine
layout formulas rather than inferred from row adjacency. -/
theorem counterChain (rho : Fin ScalarRows.scalarCount) :
    ScalarLanes.CounterChain (layout rho) := by
  refine {
    first := ?_
    successor := ?_
  }
  · simp [layout, ScalarLanes.blockAt, ScalarLanes.laneAt,
      ScalarLanes.zeroIndex, ScalarRows.cumulativePredecessor]
  · intro index nonzero
    rcases nonzero_index_cases index nonzero with
      h | h | h | h | h | h | h | h | h | h | h | h | h | h | h
    all_goals
      simp [layout, ScalarLanes.blockAt, ScalarLanes.laneAt,
        ScalarLanes.previous, ScalarRows.cumulativePredecessor,
        ScalarRows.bitStart, LaneRows.finalCountColumn, h]
    all_goals omega

/-- Every terminal scalar's sixteen lane leaves satisfy the independent
canonical decomposition and rejection-sampler semantics. Transcript provenance
and first-accepted tail semantics remain separate, explicit obligations. -/
theorem accepted_refines_lanes
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    ScalarLanes.Refines assignment canonical (layout rho) := by
  exact ScalarLanes.satisfyingRows_refine prime canonical one
    (layout rho) (counterChain rho)
    (fun block lane =>
      ScalarRows.accepted_canonicalRows accepted rho block lane)
    (fun block lane => ScalarRows.accepted_laneRows accepted rho block lane)
    (by
      simpa [layout] using
        ScalarRows.accepted_initialCount_zero canonical one accepted rho)

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarSemantics
