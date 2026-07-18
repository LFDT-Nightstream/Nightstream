import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.LaneRows

/-!
Profile-independent semantic refinement of one 64-candidate `Pi_RLC` scalar
sampler through its sixteen four-candidate lane leaves.

Assurance tier: implementation/R1CS correspondence. The theorem in this file
starts from readable row satisfaction and proves the verifier-owned rejection,
symbol, and accepted-count semantics of every lane.

Owns: the block-major `Fin 16` address; the zero-to-64 accepted-count chain;
and a single strong-induction theorem reusable by recursive and terminal
column layouts.

Does not own: generated-owner membership, Poseidon2 transcript provenance,
54-of-64 tail selection, scalar assembly, Rust trace conformance, constraint
necessity, row removal, or cost totals.

Emits constraints: no.

Authority boundary: `Layout` contains only column addresses. It acquires
meaning from independently supplied canonical-u64 and readable lane-row
satisfaction plus explicit counter-chain equalities. No row count, adjacency,
or generated owner is semantic authority.

| Protocol | Phase | Constraint family | Input obligation | Proven result |
|---|---|---|---|---|
| `Pi_RLC` | scalar sampler | address | `blockAt` / `laneAt` | exactly sixteen block-major leaves |
| `Pi_RLC` | digest decomposition | canonical-u64 | readable rows at each layout leaf | unique field value and four ordered chunks |
| `Pi_RLC` | rejection sampler | four-candidate lane | readable 104-row schema | verifier-owned accept, symbol, and cumulative semantics |
| `Pi_RLC` | count chain | predecessor links | first predecessor is zero; every successor names the prior final count | every lane starts within the 64-candidate bound |
| `Pi_RLC` | scalar sampler | final count | all sixteen leaves | final accepted count is at most 64 |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.ScalarLanes

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Four digest blocks times four field lanes. -/
def laneCount : Nat := 16

/-- Profile-specific columns needed by the profile-independent lane theorem. -/
structure Layout where
  fieldColumn : Fin 4 -> Fin 4 -> Nat
  bitStart : Fin 4 -> Fin 4 -> Nat
  initialCountColumn : Nat
  predecessor : Fin 4 -> Fin 4 -> Nat

/-- Block-major block address of one scalar-local lane index. -/
def blockAt (index : Fin laneCount) : Fin 4 :=
  ⟨index.val / 4, by
    have indexLt := index.isLt
    simp only [laneCount] at indexLt ⊢
    omega⟩

/-- Block-major lane address of one scalar-local lane index. -/
def laneAt (index : Fin laneCount) : Fin 4 :=
  ⟨index.val % 4, Nat.mod_lt _ (by decide)⟩

def zeroIndex : Fin laneCount := ⟨0, by decide⟩
def lastIndex : Fin laneCount := ⟨15, by decide⟩

/-- Preceding scalar-local lane. The nonzero premise makes subtraction exact. -/
def previous (index : Fin laneCount) (_nonzero : index.val ≠ 0) : Fin laneCount :=
  ⟨index.val - 1, by
    have indexLt := index.isLt
    simp only [laneCount] at indexLt ⊢
    omega⟩

/-- The only layout facts used to propagate the accepted-count bound. -/
structure CounterChain (layout : Layout) : Prop where
  first :
    layout.predecessor (blockAt zeroIndex) (laneAt zeroIndex) =
      layout.initialCountColumn
  successor : ∀ (index : Fin laneCount) (nonzero : index.val ≠ 0),
    layout.predecessor (blockAt index) (laneAt index) =
      LaneRows.finalCountColumn
        (layout.bitStart
          (blockAt (previous index nonzero))
          (laneAt (previous index nonzero)))

/-- Mathematical interpretation of one layout leaf. `canonicalLane` gives the
field/chunk meaning without claiming transcript provenance; `samplerLane`
proves the verifier-owned decision rows for those chunks. -/
structure LaneSemantics
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (layout : Layout)
    (index : Fin laneCount) : Prop where
  canonicalLane : ChunkOrder.LaneRefines assignment canonical
    (layout.fieldColumn (blockAt index) (laneAt index))
    (layout.bitStart (blockAt index) (laneAt index))
  samplerLane : LaneRows.Refines assignment canonical
    (layout.fieldColumn (blockAt index) (laneAt index))
    (layout.bitStart (blockAt index) (laneAt index))
    (layout.predecessor (blockAt index) (laneAt index))
    canonicalLane

/-- Complete sixteen-lane result. It intentionally stops before assigning a
Poseidon2 transcript meaning to the profile's field columns. -/
structure Refines
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (layout : Layout) : Prop where
  initialCountZero : assignment layout.initialCountColumn = 0
  lane : ∀ index : Fin laneCount,
    LaneSemantics assignment canonical layout index
  finalCountLe :
    assignment
        (LaneRows.finalCountColumn
          (layout.bitStart (blockAt lastIndex) (laneAt lastIndex))) <=
      ProductionAlphabet.candidateBound

/-- One proof, by strong induction over the sixteen block-major leaves, that
already-derived canonical candidate inputs, readable sampler rows, and explicit
layout links force all lane semantics and keep the accepted-count chain within
the fixed 64-candidate bound. This isolates sampler ownership from the upstream
canonical-u64 and transcript obligations. -/
theorem satisfyingSamplerRows_refine
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (layout : Layout)
    (chain : CounterChain layout)
    (canonicalLanes : ∀ block lane : Fin 4,
      ChunkOrder.LaneRefines assignment canonical
        (layout.fieldColumn block lane) (layout.bitStart block lane))
    (laneRows : ∀ block lane : Fin 4,
      Satisfies
        (AlphabetSamplingResidualTemplate.laneRows
          (layout.bitStart block lane) (layout.predecessor block lane))
        assignment)
    (initialZero : assignment layout.initialCountColumn = 0) :
    Refines assignment canonical layout := by
  have build : ∀ (value : Nat) (valueLt : value < laneCount),
      let index : Fin laneCount := ⟨value, valueLt⟩
      LaneSemantics assignment canonical layout index ∧
        assignment
            (LaneRows.finalCountColumn
              (layout.bitStart (blockAt index) (laneAt index))) <=
          4 * (value + 1) := by
    intro value
    induction value using Nat.strongRecOn with
    | ind value inductionHypothesis =>
        intro valueLt
        let index : Fin laneCount := ⟨value, valueLt⟩
        let canonicalLane := canonicalLanes (blockAt index) (laneAt index)
        have initialWithin :
            assignment (layout.predecessor (blockAt index) (laneAt index)) + 4 <=
              ProductionAlphabet.candidateBound := by
          by_cases valueZero : value = 0
          · have indexZero : index = zeroIndex := by
              apply Fin.ext
              exact valueZero
            rw [indexZero, chain.first, initialZero]
            decide
          · let prior := previous index (by
              simpa [index] using valueZero)
            have priorLtValue : prior.val < value := by
              simp only [prior, previous, index]
              omega
            have priorLtCount : prior.val < laneCount := prior.isLt
            have priorResult :=
              inductionHypothesis prior.val priorLtValue priorLtCount
            have priorBound :
                assignment
                    (LaneRows.finalCountColumn
                      (layout.bitStart (blockAt prior) (laneAt prior))) <=
                  4 * (prior.val + 1) := by
              simpa using priorResult.2
            rw [chain.successor index (by simpa [index] using valueZero)]
            calc
              assignment
                    (LaneRows.finalCountColumn
                      (layout.bitStart
                        (blockAt (previous index (by simpa [index] using valueZero)))
                        (laneAt (previous index (by simpa [index] using valueZero))))) + 4 <=
                  4 * (prior.val + 1) + 4 := by
                simpa [prior] using Nat.add_le_add_right priorBound 4
              _ <= ProductionAlphabet.candidateBound := by
                have valueLt16 : value < 16 := by
                  simpa [laneCount] using valueLt
                simp only [ProductionAlphabet.candidateBound, prior, previous,
                  index, laneCount]
                omega
        have samplerLane := LaneRows.refines prime canonical one
          (layout.fieldColumn (blockAt index) (laneAt index))
          (layout.bitStart (blockAt index) (laneAt index))
          (layout.predecessor (blockAt index) (laneAt index))
          canonicalLane initialWithin
          (laneRows (blockAt index) (laneAt index))
        have finalBoundRaw := samplerLane.finalCount_le_add_four
        have finalBound :
            assignment
                (LaneRows.finalCountColumn
                  (layout.bitStart (blockAt index) (laneAt index))) <=
              4 * (value + 1) := by
          by_cases valueZero : value = 0
          · have indexZero : index = zeroIndex := by
              apply Fin.ext
              exact valueZero
            calc
              assignment
                    (LaneRows.finalCountColumn
                      (layout.bitStart (blockAt index) (laneAt index))) <=
                  assignment
                      (layout.predecessor (blockAt index) (laneAt index)) + 4 :=
                finalBoundRaw
              _ = 4 * (value + 1) := by
                rw [indexZero, chain.first, initialZero, valueZero]
          · let prior := previous index (by
              simpa [index] using valueZero)
            have priorLtValue : prior.val < value := by
              simp only [prior, previous, index]
              omega
            have priorResult := inductionHypothesis prior.val priorLtValue prior.isLt
            have priorBound :
                assignment
                    (LaneRows.finalCountColumn
                      (layout.bitStart (blockAt prior) (laneAt prior))) <=
                  4 * (prior.val + 1) := by
              simpa using priorResult.2
            calc
              assignment
                    (LaneRows.finalCountColumn
                      (layout.bitStart (blockAt index) (laneAt index))) <=
                  assignment
                      (layout.predecessor (blockAt index) (laneAt index)) + 4 :=
                finalBoundRaw
              _ = assignment
                    (LaneRows.finalCountColumn
                      (layout.bitStart
                        (blockAt (previous index (by simpa [index] using valueZero)))
                        (laneAt (previous index (by simpa [index] using valueZero))))) + 4 := by
                rw [chain.successor index (by simpa [index] using valueZero)]
              _ <= 4 * (prior.val + 1) + 4 := by
                simpa [prior] using Nat.add_le_add_right priorBound 4
              _ = 4 * (value + 1) := by
                simp only [prior, previous, index]
                omega
        exact ⟨{
          canonicalLane := canonicalLane
          samplerLane := samplerLane
        }, finalBound⟩
  refine {
    initialCountZero := initialZero
    lane := fun index => (build index.val index.isLt).1
    finalCountLe := ?_
  }
  have bound := (build lastIndex.val lastIndex.isLt).2
  simpa [lastIndex, ProductionAlphabet.candidateBound] using bound

/-- Readable canonical-u64 and sampler rows imply the complete sixteen-lane
result. This wrapper retains the original source-R1CS interface while the core
theorem above lets active profiles keep transcript decomposition and sampler
row ownership separate. -/
theorem satisfyingRows_refine
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (layout : Layout)
    (chain : CounterChain layout)
    (canonicalRows : ∀ block lane : Fin 4,
      Satisfies CanonicalU64.rows
        (ChunkOrder.laneSource assignment
          (layout.fieldColumn block lane) (layout.bitStart block lane)))
    (laneRows : ∀ block lane : Fin 4,
      Satisfies
        (AlphabetSamplingResidualTemplate.laneRows
          (layout.bitStart block lane) (layout.predecessor block lane))
        assignment)
    (initialZero : assignment layout.initialCountColumn = 0) :
    Refines assignment canonical layout := by
  exact satisfyingSamplerRows_refine prime canonical one layout chain
    (fun block lane =>
      ChunkOrder.satisfyingLane_refines prime canonical one
        (layout.fieldColumn block lane) (layout.bitStart block lane)
        (canonicalRows block lane))
    laneRows initialZero

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.ScalarLanes
