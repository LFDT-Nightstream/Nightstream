import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.CandidateOrder
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.ScalarLanes
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailRows

/-!
Profile-independent candidate semantics at the lane-to-selection-tail boundary.

Assurance tier: implementation/R1CS correspondence. Given independently
proved sixteen-lane sampler semantics and explicit source-wire bindings, this
file proves that every tail candidate carries the verifier-owned accept bit,
centered symbol, and accepted-count recurrence for its canonical field chunk.

Owns: the 64-candidate-to-16-lane address map; a profile-independent tail
layout; the explicit lane/tail source-binding contract; field-derived
candidate meaning; and per-candidate accept/symbol/count refinement.

Does not own: generated-owner placement, Poseidon2 transcript provenance for
the field columns, 54-of-64 selection, scalar-to-scalar state chaining,
coefficient assembly, Rust trace conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: `fieldCandidate` is derived from a canonical field column,
not yet from a verifier transcript. Therefore this file deliberately does not
call it a challenge candidate. A later transcript theorem must prove that each
field column is a Poseidon2 output before the final challenge claim closes.

| Protocol | Phase | Constraint family | Input obligation | Proven result |
|---|---|---|---|---|
| `Pi_RLC` | bounded sampler | source order | independent `block/lane/part` address | 64 candidates map exactly onto 16 four-part lane leaves |
| `Pi_RLC` | lane/tail boundary | source wires | explicit accept/symbol/count bindings | tail inputs equal the matching readable lane outputs |
| `Pi_RLC` | rejection sampler | candidate decision | generic lane semantics | verifier-owned accept and centered symbol for each field chunk |
| `Pi_RLC` | accepted-prefix chain | one recurrence leaf | generic lane cumulative equation | `cumulative = prior + accept` for every tail candidate |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Profile-specific lane columns plus the mapped selection-tail allocation. -/
structure Layout where
  lanes : ScalarLanes.Layout
  tailBitStarts : List Nat
  tailFirstAllocated : Nat

/-- Readable selection-tail assignment induced by one profile layout. -/
def localAssignment
    (layout : Layout) (assignment : Nat -> Nat) : Nat -> Nat :=
  TailRows.localAssignmentAt layout.tailBitStarts layout.tailFirstAllocated
    assignment

/-- The sixteen-lane leaf containing one of the 64 source-order candidates. -/
def laneIndex
    (candidate : Fin SelectionRows.candidateCount) : Fin ScalarLanes.laneCount :=
  let location := CandidateOrder.address candidate
  ⟨4 * location.block.val + location.lane.val, by
    have blockLt := location.block.isLt
    have laneLt := location.lane.isLt
    simp only [ScalarLanes.laneCount]
    omega⟩

/-- The lane index preserves the independently defined block address. -/
theorem laneIndex_block
    (candidate : Fin SelectionRows.candidateCount) :
    ScalarLanes.blockAt (laneIndex candidate) =
      (CandidateOrder.address candidate).block := by
  apply Fin.ext
  have blockLt := (CandidateOrder.address candidate).block.isLt
  have laneLt := (CandidateOrder.address candidate).lane.isLt
  simp only [ScalarLanes.blockAt, laneIndex]
  omega

/-- The lane index preserves the independently defined lane address. -/
theorem laneIndex_lane
    (candidate : Fin SelectionRows.candidateCount) :
    ScalarLanes.laneAt (laneIndex candidate) =
      (CandidateOrder.address candidate).lane := by
  apply Fin.ext
  have blockLt := (CandidateOrder.address candidate).block.isLt
  have laneLt := (CandidateOrder.address candidate).lane.isLt
  simp only [ScalarLanes.laneAt, laneIndex]
  omega

/-- Local readable four-candidate lane assignment containing `candidate`. -/
def laneAssignment
    (layout : Layout) (assignment : Nat -> Nat)
    (candidate : Fin SelectionRows.candidateCount) : Nat -> Nat :=
  LaneRows.localAssignment assignment
    (layout.lanes.bitStart
      (ScalarLanes.blockAt (laneIndex candidate))
      (ScalarLanes.laneAt (laneIndex candidate)))
    (layout.lanes.predecessor
      (ScalarLanes.blockAt (laneIndex candidate))
      (ScalarLanes.laneAt (laneIndex candidate)))

/-- The canonical 16-bit field chunk at one source-order position. This has no
Poseidon2 transcript meaning until a separate provenance theorem is supplied. -/
def fieldCandidate
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (candidate : Fin SelectionRows.candidateCount) :
    ProductionAlphabet.Chunk :=
  laneChunk
    (DigestRounds.fieldAt assignment canonical
      (layout.lanes.fieldColumn
        (ScalarLanes.blockAt (laneIndex candidate))
        (ScalarLanes.laneAt (laneIndex candidate))))
    (CandidateOrder.address candidate).part

/-- Explicit interface between the mapped selection tail and the sixteen
readable lane leaves. No equality is inferred from adjacency. -/
structure SourceBindings
    (layout : Layout) (assignment : Nat -> Nat) : Prop where
  accept : forall candidate : Fin SelectionRows.candidateCount,
    localAssignment layout assignment
        (SelectionRows.acceptCol candidate.val) =
      laneAssignment layout assignment candidate
        (ChunkRows.acceptCol (CandidateOrder.address candidate).part.val)
  symbol : forall candidate : Fin SelectionRows.candidateCount,
    localAssignment layout assignment
        (SelectionRows.symbolCol candidate.val) =
      laneAssignment layout assignment candidate
        (ChunkRows.symbolCol (CandidateOrder.address candidate).part.val)
  cumulative : forall candidate : Fin SelectionRows.candidateCount,
    localAssignment layout assignment
        (SelectionRows.cumulativeCol candidate.val) =
      laneAssignment layout assignment candidate
        (ChunkRows.cumulativeCol (CandidateOrder.address candidate).part.val)
  prior : forall candidate : Fin SelectionRows.candidateCount,
    localAssignment layout assignment
        (SelectionRows.prefixCol candidate.val) =
      laneAssignment layout assignment candidate
        (ChunkRows.priorCumulativeCol
          (CandidateOrder.address candidate).part.val)

/-- One selection-tail candidate has the verifier-owned decision and centered
symbol of the matching canonical field chunk. -/
structure CandidateRefines
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (candidate : Fin SelectionRows.candidateCount) : Prop where
  accept : localAssignment layout assignment
      (SelectionRows.acceptCol candidate.val) =
    if ProductionAlphabet.verifier.accepts
        (fieldCandidate layout assignment canonical candidate) then 1 else 0
  symbol : localAssignment layout assignment
      (SelectionRows.symbolCol candidate.val) =
    CandidateOrder.centeredField
      (ProductionAlphabet.verifier.symbol
        (fieldCandidate layout assignment canonical candidate))

/-- Generic lane semantics plus explicit source bindings refine every tail
candidate without assuming transcript provenance for the field columns. -/
theorem candidate_refines
    {assignment : Nat -> Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    {layout : Layout}
    (lanes : ScalarLanes.Refines assignment canonical layout.lanes)
    (bindings : SourceBindings layout assignment)
    (candidate : Fin SelectionRows.candidateCount) :
    CandidateRefines layout assignment canonical candidate := by
  let leaf := lanes.lane (laneIndex candidate)
  let part := (CandidateOrder.address candidate).part
  refine {
    accept := ?_
    symbol := ?_
  }
  · calc
      localAssignment layout assignment
          (SelectionRows.acceptCol candidate.val) =
        laneAssignment layout assignment candidate
          (ChunkRows.acceptCol part.val) := bindings.accept candidate
      _ = Lane.acceptedBit
          (laneAssignment layout assignment candidate)
          (LaneRows.sourceBitsBoolean leaf.canonicalLane) part :=
        leaf.samplerLane.semantics.accepted part
      _ = if ProductionAlphabet.verifier.accepts
            (fieldCandidate layout assignment canonical candidate)
          then 1 else 0 := by
        unfold Lane.acceptedBit fieldCandidate laneAssignment
        rw [leaf.samplerLane.transcriptCandidates part]
  · calc
      localAssignment layout assignment
          (SelectionRows.symbolCol candidate.val) =
        laneAssignment layout assignment candidate
          (ChunkRows.symbolCol part.val) := bindings.symbol candidate
      _ = Lane.expectedSymbol
          (laneAssignment layout assignment candidate)
          (LaneRows.sourceBitsBoolean leaf.canonicalLane) part :=
        leaf.samplerLane.semantics.symbols part
      _ = CandidateOrder.centeredField
          (ProductionAlphabet.verifier.symbol
            (fieldCandidate layout assignment canonical candidate)) := by
        unfold Lane.expectedSymbol CandidateOrder.centeredField fieldCandidate
          laneAssignment
        rw [leaf.samplerLane.transcriptCandidates part]

/-- Every tail cumulative wire is the exact integer recurrence for the same
candidate's prior prefix and verifier-owned accept bit. -/
theorem cumulative_step
    {assignment : Nat -> Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    {layout : Layout}
    (lanes : ScalarLanes.Refines assignment canonical layout.lanes)
    (bindings : SourceBindings layout assignment)
    (candidate : Fin SelectionRows.candidateCount) :
    localAssignment layout assignment
        (SelectionRows.cumulativeCol candidate.val) =
      localAssignment layout assignment
          (SelectionRows.prefixCol candidate.val) +
        localAssignment layout assignment
          (SelectionRows.acceptCol candidate.val) := by
  let leaf := lanes.lane (laneIndex candidate)
  let part := (CandidateOrder.address candidate).part
  calc
    localAssignment layout assignment
        (SelectionRows.cumulativeCol candidate.val) =
      laneAssignment layout assignment candidate
        (ChunkRows.cumulativeCol part.val) := bindings.cumulative candidate
    _ = laneAssignment layout assignment candidate
          (ChunkRows.priorCumulativeCol part.val) +
        laneAssignment layout assignment candidate
          (ChunkRows.acceptCol part.val) := by
      unfold laneAssignment
      rw [leaf.samplerLane.semantics.cumulative part]
      rw [leaf.samplerLane.semantics.accepted part]
    _ = localAssignment layout assignment
          (SelectionRows.prefixCol candidate.val) +
        localAssignment layout assignment
          (SelectionRows.acceptCol candidate.val) := by
      rw [← bindings.prior candidate, ← bindings.accept candidate]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics
