import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Lane
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ChunkOrder

/-!
Exact production-column refinement for one four-candidate `Pi_RLC` sampler
lane.

Owns: transport of the readable 104-row lane schema through the production
column map; the source-bit correspondence to one canonical Poseidon output
lane; and identification of all four local candidates with the independent
transcript machine.

Does not own: Poseidon permutation soundness, the other fifteen lanes,
first-accepted selection, coefficient assembly, Rust trace conformance, row
removal, or aggregate cost totals.

Emits constraints: no.

Authority boundary: the transcript candidate comes from `ChunkOrder` and the
sampler decision comes from `ProductionAlphabet.verifier`. Generated rows and
column maps are used only as implementation objects that must refine both
independent meanings.

| Protocol | Phase | Constraint family | Production object | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/lane | column placement | `laneColumnMap` | every local source/count/auxiliary column has its production interpretation |
| `Pi_RLC` | sampler/lane | four candidate leaves | `laneRows` | exact mapped rows imply the readable 104-row schema |
| `Pi_RLC` | transcript -> sampler | candidate binding | canonical-u64 bit windows | local candidates equal the independent transcript candidates |
| `Pi_RLC` | sampler/lane | semantic decision | local `Lane.refines` | production rows force verifier-owned accept/symbol/count results |
| `Pi_RLC` | sampler/lane | count boundary | mapped final count | production final count advances by at most four verifier decisions |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.LaneRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

def localAssignment
    (assignment : Nat -> Nat) (bitStart cumPrev : Nat) : Nat -> Nat :=
  Relabel.assignment
    (AlphabetSamplingResidualTemplate.laneColumnMap bitStart cumPrev)
    assignment

@[simp] theorem localAssignment_zero
    (assignment : Nat -> Nat) (bitStart cumPrev : Nat) :
    localAssignment assignment bitStart cumPrev 0 = assignment 0 := by
  simp [localAssignment, AlphabetSamplingResidualTemplate.laneColumnMap,
    Relabel.assignment, Relabel.column]

theorem localAssignment_sourceBit
    (assignment : Nat -> Nat) (bitStart cumPrev part offset : Nat)
    (partLt : part < 4) (offsetLt : offset < 16) :
    localAssignment assignment bitStart cumPrev
        (ChunkRows.sourceBitCol part offset) =
      assignment (bitStart + 16 * part + offset) := by
  unfold localAssignment Relabel.assignment Relabel.column
  unfold AlphabetSamplingResidualTemplate.laneColumnMap
  unfold ChunkRows.sourceBitCol
  rw [show 1 + 16 * part + offset = (16 * part + offset) + 1 by omega]
  simp only [List.getD_eq_getElem?_getD, List.cons_append]
  rw [List.getElem?_cons_succ]
  rw [List.getElem?_append_left (by simp; omega)]
  rw [List.getElem?_append_left (by simp; omega)]
  simp only [List.nil_append, List.getElem?_map]
  rw [List.getElem?_range (by omega : 16 * part + offset < 64)]
  change assignment (bitStart + (16 * part + offset)) =
    assignment (bitStart + 16 * part + offset)
  simpa only [Nat.add_assoc]

@[simp] theorem localAssignment_initialCount
    (assignment : Nat -> Nat) (bitStart cumPrev : Nat) :
    localAssignment assignment bitStart cumPrev Lane.initialCountCol =
      assignment cumPrev := by
  simp [localAssignment, AlphabetSamplingResidualTemplate.laneColumnMap,
    Relabel.assignment, Relabel.column, Lane.initialCountCol]

theorem localAssignment_auxiliary
    (assignment : Nat -> Nat) (bitStart cumPrev offset : Nat)
    (offsetLt : offset < 92) :
    localAssignment assignment bitStart cumPrev (66 + offset) =
      assignment (bitStart + 66 + offset) := by
  unfold localAssignment Relabel.assignment Relabel.column
  unfold AlphabetSamplingResidualTemplate.laneColumnMap
  rw [show 66 + offset = (65 + offset) + 1 by omega]
  simp only [List.getD_eq_getElem?_getD, List.cons_append]
  rw [List.getElem?_cons_succ]
  rw [List.getElem?_append_right (by simp)]
  simp only [List.nil_append, List.length_append, List.length_map,
    List.length_range, List.length_singleton, List.getElem?_map]
  rw [show 65 + offset - (64 + 1) = offset by omega]
  rw [List.getElem?_range (by omega : offset < 92)]
  simp only [Option.map_some, Option.getD_some, Nat.add_assoc]

theorem localAssignment_accept
    (assignment : Nat -> Nat) (bitStart cumPrev part : Nat)
    (partLt : part < 4) :
    localAssignment assignment bitStart cumPrev
        (ChunkRows.acceptCol part) =
      assignment (bitStart + 66 + 23 * part) := by
  simpa [ChunkRows.acceptCol, ChunkRows.base] using
    localAssignment_auxiliary assignment bitStart cumPrev (23 * part) (by omega)

theorem localAssignment_symbol
    (assignment : Nat -> Nat) (bitStart cumPrev part : Nat)
    (partLt : part < 4) :
    localAssignment assignment bitStart cumPrev
        (ChunkRows.symbolCol part) =
      assignment (bitStart + 66 + 23 * part + 21) := by
  simpa [ChunkRows.symbolCol, ChunkRows.base, Nat.add_assoc] using
    localAssignment_auxiliary assignment bitStart cumPrev
      (23 * part + 21) (by omega)

theorem localAssignment_cumulative
    (assignment : Nat -> Nat) (bitStart cumPrev part : Nat)
    (partLt : part < 4) :
    localAssignment assignment bitStart cumPrev
        (ChunkRows.cumulativeCol part) =
      assignment (bitStart + 66 + 23 * part + 22) := by
  simpa [ChunkRows.cumulativeCol, ChunkRows.base, Nat.add_assoc] using
    localAssignment_auxiliary assignment bitStart cumPrev
      (23 * part + 22) (by omega)

def finalCountColumn (bitStart : Nat) : Nat := bitStart + 157

@[simp] theorem localAssignment_finalCount
    (assignment : Nat -> Nat) (bitStart cumPrev : Nat) :
    localAssignment assignment bitStart cumPrev Lane.finalCountCol =
      assignment (finalCountColumn bitStart) := by
  simp [localAssignment, AlphabetSamplingResidualTemplate.laneColumnMap,
    Relabel.assignment, Relabel.column, Lane.finalCountCol,
    finalCountColumn, ChunkRows.cumulativeCol, ChunkRows.base]

/-- Exact mapped production rows imply the readable local row hierarchy. The
generated list is used only through the kernel-checked schema equality in
`ChunkRows.rows_eq_generated`. -/
theorem satisfies_local
    {assignment : Nat -> Nat} {bitStart cumPrev : Nat}
    (satisfies : Satisfies
      (AlphabetSamplingResidualTemplate.laneRows bitStart cumPrev)
      assignment) :
    Satisfies ChunkRows.rows
      (localAssignment assignment bitStart cumPrev) := by
  apply (Relabel.satisfies_mapped_iff ChunkRows.rows
    (AlphabetSamplingResidualTemplate.laneColumnMap bitStart cumPrev)
    assignment).mp
  simpa [AlphabetSamplingResidualTemplate.laneRows,
    ChunkRows.rows_eq_generated] using satisfies

/-- Canonical-u64 bitness becomes the exact bitness assumption consumed by
the four local candidate leaves. -/
theorem sourceBitsBoolean
    {assignment : Nat -> Nat} {canonical : ChunkOrder.CanonicalAssignment assignment}
    {fieldColumn bitStart cumPrev : Nat}
    (transcript :
      ChunkOrder.LaneRefines assignment canonical fieldColumn bitStart) :
    Lane.AllSourceBitsBoolean
      (localAssignment assignment bitStart cumPrev) := by
  intro part offset offsetLt
  rw [localAssignment_sourceBit assignment bitStart cumPrev
    part.val offset part.isLt offsetLt]
  have bit := transcript.bitsBoolean (16 * part.val + offset) (by omega)
  rw [ChunkOrder.laneSource_bit assignment fieldColumn bitStart
    (16 * part.val + offset) (by omega)] at bit
  simpa only [Nat.add_assoc] using bit

private theorem foldl_sourceBits
    (assignment : Nat -> Nat) (fieldColumn bitStart cumPrev : Nat)
    (part : Fin 4) (offsets : List Nat)
    (offsetsLt : ∀ offset, offset ∈ offsets -> offset < 16)
    (initial : Nat) :
    offsets.foldl
        (fun value offset =>
          value + 2 ^ offset *
            localAssignment assignment bitStart cumPrev
              (ChunkRows.sourceBitCol part.val offset))
        initial =
      offsets.foldl
        (fun value offset =>
          value + 2 ^ offset *
            ChunkOrder.laneSource assignment fieldColumn bitStart
              (CanonicalU64.bitCol (16 * part.val + offset)))
        initial := by
  induction offsets generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl]
      rw [localAssignment_sourceBit assignment bitStart cumPrev
        part.val head part.isLt (offsetsLt head (by simp))]
      rw [ChunkOrder.laneSource_bit assignment fieldColumn bitStart
        (16 * part.val + head) (by
          have headLt := offsetsLt head (by simp)
          omega)]
      simp only [Nat.add_assoc]
      apply inductionHypothesis
      intro offset member
      exact offsetsLt offset (by simp [member])

/-- The local little-endian candidate is the same integer as the corresponding
canonical-u64 window. This is a structural column-map fact, independent of
any accept/reject witness. -/
theorem chunkValue_eq_bitWindow
    (assignment : Nat -> Nat) (fieldColumn bitStart cumPrev : Nat)
    (part : Fin 4) :
    Chunk.chunkValue (localAssignment assignment bitStart cumPrev) part.val =
      ChunkOrder.bitWindowValue
        (ChunkOrder.laneSource assignment fieldColumn bitStart)
        (16 * part.val) 16 := by
  unfold Chunk.chunkValue ChunkOrder.bitWindowValue
  apply foldl_sourceBits assignment fieldColumn bitStart cumPrev part
  intro offset member
  exact List.mem_range.mp member

/-- Production-lane refinement at the transcript/sampler boundary. -/
structure Refines
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (fieldColumn bitStart cumPrev : Nat)
    (transcript :
      ChunkOrder.LaneRefines assignment canonical fieldColumn bitStart) : Prop where
  semantics : Lane.Refines
    (localAssignment assignment bitStart cumPrev)
    (sourceBitsBoolean transcript)
  transcriptCandidates : ∀ part : Fin 4,
    Chunk.candidate
        (localAssignment assignment bitStart cumPrev) part.val
        ((sourceBitsBoolean transcript) part) =
      laneChunk (DigestRounds.fieldAt assignment canonical fieldColumn) part

theorem refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (fieldColumn bitStart cumPrev : Nat)
    (transcript :
      ChunkOrder.LaneRefines assignment canonical fieldColumn bitStart)
    (initialWithin : assignment cumPrev + 4 <=
      ProductionAlphabet.candidateBound)
    (satisfies : Satisfies
      (AlphabetSamplingResidualTemplate.laneRows bitStart cumPrev)
      assignment) :
    Refines assignment canonical fieldColumn bitStart cumPrev transcript := by
  let source := localAssignment assignment bitStart cumPrev
  have sourceCanonical : ∀ column, source column < goldilocksP :=
    Relabel.canonical canonical
  have sourceOne : source 0 = 1 := by
    simpa [source] using one
  have bits := sourceBitsBoolean (cumPrev := cumPrev) transcript
  have sourceRows : Satisfies ChunkRows.rows source := by
    simpa [source] using satisfies_local satisfies
  have sourceInitialWithin : source Lane.initialCountCol + 4 <=
      ProductionAlphabet.candidateBound := by
    simpa [source] using initialWithin
  refine {
    semantics := Lane.refines prime sourceCanonical sourceOne bits
      sourceInitialWithin sourceRows
    transcriptCandidates := ?_
  }
  intro part
  apply Fin.ext
  exact (chunkValue_eq_bitWindow assignment fieldColumn bitStart cumPrev part).trans
    (transcript.chunks part)

theorem Refines.finalCount_eq
    {assignment : Nat -> Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    {fieldColumn bitStart cumPrev : Nat}
    {transcript :
      ChunkOrder.LaneRefines assignment canonical fieldColumn bitStart}
    (refinement :
      Refines assignment canonical fieldColumn bitStart cumPrev transcript) :
    assignment (finalCountColumn bitStart) =
      assignment cumPrev +
        Lane.acceptedDelta
          (localAssignment assignment bitStart cumPrev)
          (sourceBitsBoolean transcript) := by
  simpa using refinement.semantics.finalCount

theorem Refines.delta_le_four
    {assignment : Nat -> Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    {fieldColumn bitStart cumPrev : Nat}
    {transcript :
      ChunkOrder.LaneRefines assignment canonical fieldColumn bitStart}
    (_refinement :
      Refines assignment canonical fieldColumn bitStart cumPrev transcript) :
    Lane.acceptedDelta
        (localAssignment assignment bitStart cumPrev)
      (sourceBitsBoolean transcript) <= 4 :=
  Lane.acceptedDelta_le_four _ _

theorem Refines.finalCount_le_add_four
    {assignment : Nat -> Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    {fieldColumn bitStart cumPrev : Nat}
    {transcript :
      ChunkOrder.LaneRefines assignment canonical fieldColumn bitStart}
    (refinement :
      Refines assignment canonical fieldColumn bitStart cumPrev transcript) :
    assignment (finalCountColumn bitStart) <= assignment cumPrev + 4 := by
  rw [refinement.finalCount_eq]
  exact Nat.add_le_add_left refinement.delta_le_four _

/-- Full one-lane composition with the pure four-block transcript machine.
The only implementation assumptions are acceptance of the exact transcript
owners and satisfaction of the exact mapped lane rows. -/
structure RefinesMachineLane
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (block laneIndex : Fin 4) (cumPrev : Nat)
    (transcript : ChunkOrder.LaneRefines assignment canonical
      (ChunkOrder.fieldColumn block laneIndex)
      (ChunkOrder.bitStart block laneIndex)) : Prop where
  production : Refines assignment canonical
    (ChunkOrder.fieldColumn block laneIndex)
    (ChunkOrder.bitStart block laneIndex) cumPrev transcript
  machineCandidates : ∀ part : Fin 4,
    Chunk.candidate
        (localAssignment assignment
          (ChunkOrder.bitStart block laneIndex) cumPrev)
        part.val ((sourceBitsBoolean (cumPrev := cumPrev) transcript) part) =
      (digestBlock
        (ChunkOrder.machineBlockInput assignment canonical block)
        block.val).2 (ChunkOrder.chunkPosition laneIndex part)

theorem accepted_refines_machineLane
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (block laneIndex : Fin 4) (cumPrev : Nat)
    (initialWithin : assignment cumPrev + 4 <=
      ProductionAlphabet.candidateBound)
    (satisfies : Satisfies
      (AlphabetSamplingResidualTemplate.laneRows
        (ChunkOrder.bitStart block laneIndex) cumPrev)
      assignment) :
    RefinesMachineLane assignment canonical block laneIndex cumPrev
      (ChunkOrder.accepted_refines_lane prime canonical one accepted
        block laneIndex) := by
  let transcript :=
    ChunkOrder.accepted_refines_lane prime canonical one accepted block laneIndex
  have production := refines prime canonical one
    (ChunkOrder.fieldColumn block laneIndex)
    (ChunkOrder.bitStart block laneIndex) cumPrev transcript
    initialWithin satisfies
  refine {
    production := production
    machineCandidates := ?_
  }
  intro part
  apply Fin.ext
  exact (chunkValue_eq_bitWindow assignment
      (ChunkOrder.fieldColumn block laneIndex)
      (ChunkOrder.bitStart block laneIndex) cumPrev part).trans
    (ChunkOrder.accepted_refines_machineCandidates
      prime canonical one accepted block laneIndex part)

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.LaneRows
