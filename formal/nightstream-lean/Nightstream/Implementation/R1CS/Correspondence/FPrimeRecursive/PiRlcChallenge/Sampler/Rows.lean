import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.SamplerLayout
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.IndexedRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ChunkOrder

/-!
Three-matrix diagnostic row boundary for the fifteen PiRLC challenge samplers.

Owns: readable row constructors, permutation-normalized slice embeddings, and
conditional satisfaction projections. Does not own a full Rust row artifact,
Poseidon2 transcript provenance, sampler semantics, or constraint necessity.

Assurance tier: model-level row transport over caller-supplied embeddings and
whole-row satisfaction. The closed counts below are artifact-derived source-row
extent totals; they do not prove disjoint ownership, gap-free coverage,
production-row identity, encoded cost, or semantic necessity.

| Source-row family | Formula | Rows |
|---|---:|---:|
| canonical-u64 transcript leaves | `15 * 16 * 69` | `16,560` |
| sampler initialization | `15` | `15` |
| chunk acceptance | `15 * 64 * 4` | `3,840` |
| chunk mod-5 | `15 * 64 * 20` | `19,200` |
| chunk symbol + prefix | `15 * 64 * 2` | `1,920` |
| acceptance bound | `15 * 6` | `90` |
| selection initialization | `15` | `15` |
| selection one-hot | `15 * 54 * 12` | `9,720` |
| selection products | `15 * 54 * 33` | `26,730` |
| selection accept/prefix/symbol bindings | `15 * 54 * 3` | `2,430` |
| sampler total (canonical leaves excluded) | sum of sampler families | `63,960` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.Rows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout

/-! ## Readable global row leaves -/

/-- The one physical equation used to initialize a scalar's accepted-prefix
count. Its integer-zero meaning additionally requires canonical residues and
the constant-one wire. -/
def initializationEquation (rho : Fin scalarCount) : Row :=
  ⟨[(initialCountColumn rho, 1)], [(0, 1)], []⟩

/-- Generic canonical-u64 columns renamed into one active digest lane. -/
def canonicalColumnMap
    (rho : Fin scalarCount)
    (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : List Nat :=
  OwnerCertificate.canonicalU64ColumnMap
    (fieldColumn rho block lane) (bitStart rho block lane)

/-- One globally relabeled canonical-u64 transcript leaf. These rows are
transcript ownership, not sampler ownership. -/
def canonicalRows
    (rho : Fin scalarCount)
    (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : List Row :=
  CanonicalU64.rows.map (Relabel.row (canonicalColumnMap rho block lane))

/-- The 104 sampler-residual rows following one canonical transcript leaf. -/
def laneRows
    (rho : Fin scalarCount)
    (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : List Row :=
  AlphabetSamplingResidualTemplate.laneRows
    (bitStart rho block lane)
    (predecessorColumn rho block lane)

/-- The complete mapped 54-of-64 selection tail for one scalar. -/
def tailRows (rho : Fin scalarCount) : List Row :=
  AlphabetSamplingResidualTemplate.tailRows
    (tailBitStarts rho) (tailFirstAllocated rho)

/-! ## Physical slice premises -/

/-- Every canonical transcript leaf occurs at its named normalized source-row
slice, modulo sparse-term order within each A/B/C linear combination. -/
structure CanonicalRowsEmbedded (fullRows : List Row) : Prop where
  lane : ∀ (rho : Fin scalarCount)
      (block : Fin digestBlockCount)
      (lane : Fin lanesPerBlock),
    ActiveIndexedRows.RowsEmbeddedAt fullRows
      (canonicalRow rho block lane) (canonicalRows rho block lane)

/-- Every sampler-owned initialization, residual-lane, and tail leaf occurs at
its named normalized source-row slice. Canonical transcript leaves are
deliberately absent from this structure. -/
structure SamplerRowsEmbedded (fullRows : List Row) : Prop where
  initialization : ∀ rho : Fin scalarCount,
    ActiveIndexedRows.RowsEmbeddedAt fullRows
      (initializationRow rho) [initializationEquation rho]
  lane : ∀ (rho : Fin scalarCount)
      (block : Fin digestBlockCount)
      (lane : Fin lanesPerBlock),
    ActiveIndexedRows.RowsEmbeddedAt fullRows
      (laneResidualRow rho block lane) (laneRows rho block lane)
  tail : ∀ rho : Fin scalarCount,
    ActiveIndexedRows.RowsEmbeddedAt fullRows
      (tailRow rho) (tailRows rho)

/-- Conditional acceptance boundary for the active row list. This packages
whole-list satisfaction and the two independent physical embedding trees; it
does not assert identity with a Rust circuit or trace. -/
structure EmbeddedRowsSatisfied (fullRows : List Row) (assignment : Nat → Nat) : Prop where
  fullSatisfies : Satisfies fullRows assignment
  canonicalEmbedding : CanonicalRowsEmbedded fullRows
  samplerEmbedding : SamplerRowsEmbedded fullRows

/-! ## Satisfaction projections -/

/-- Model-level projection of one initialization source slice. -/
theorem accepted_initializationRows
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    Satisfies [initializationEquation rho] assignment :=
  ActiveIndexedRows.rows_satisfied_of_embeddedAt
    (accepted.samplerEmbedding.initialization rho) accepted.fullSatisfies

/-- Model-level projection of the single initialization equation. -/
theorem accepted_initializationEquation
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    RowHolds assignment (initializationEquation rho) :=
  accepted_initializationRows accepted rho _ (by simp)

/-- Model-level projection of one globally relabeled canonical source leaf. -/
theorem accepted_canonicalRows
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount)
    (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) :
    Satisfies (canonicalRows rho block lane) assignment :=
  ActiveIndexedRows.rows_satisfied_of_embeddedAt
    (accepted.canonicalEmbedding.lane rho block lane) accepted.fullSatisfies

/-- Model-level projection of one sampler-residual lane source leaf. -/
theorem accepted_laneRows
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount)
    (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) :
    Satisfies (laneRows rho block lane) assignment :=
  ActiveIndexedRows.rows_satisfied_of_embeddedAt
    (accepted.samplerEmbedding.lane rho block lane) accepted.fullSatisfies

/-- Model-level projection of one complete mapped sampler tail source leaf. -/
theorem accepted_tailRows
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    Satisfies (tailRows rho) assignment :=
  ActiveIndexedRows.rows_satisfied_of_embeddedAt
    (accepted.samplerEmbedding.tail rho) accepted.fullSatisfies

/-- Model-level Relabel transport from a global canonical leaf to the generic
canonical-u64 assignment view. -/
theorem accepted_localCanonicalRows
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount)
    (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) :
    Satisfies CanonicalU64.rows
      (Relabel.assignment (canonicalColumnMap rho block lane) assignment) := by
  apply (Relabel.satisfies_mapped_iff CanonicalU64.rows
    (canonicalColumnMap rho block lane) assignment).mp
  simpa [canonicalRows] using accepted_canonicalRows accepted rho block lane

/-- Model-level canonical-u64 semantics for one active lane. Transcript
provenance of the field column remains a separate obligation. -/
theorem accepted_canonicalLane_refines
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount)
    (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) :
    ChunkOrder.LaneRefines assignment canonical
      (fieldColumn rho block lane) (bitStart rho block lane) := by
  apply ChunkOrder.satisfyingLane_refines prime canonical one
  simpa [ChunkOrder.laneSource, canonicalColumnMap] using
    accepted_localCanonicalRows accepted rho block lane

/-- Model-level integer interpretation of the embedded initialization row. -/
theorem accepted_initialCount_zero
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    assignment (initialCountColumn rho) = 0 := by
  have rowHolds := accepted_initializationEquation accepted rho
  have valueCanonical := canonical (initialCountColumn rho)
  simpa [initializationEquation, RowHolds, lcEval, one,
    Nat.mod_eq_of_lt valueCanonical] using rowHolds

/-- Model-level transport of the mapped tail into its readable local
`SelectionRows.rows` hierarchy. -/
theorem accepted_readableTail
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    Satisfies PiRlcChallenge.Sampler.SelectionRows.rows
      (PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
        (tailBitStarts rho) (tailFirstAllocated rho)
        assignment) :=
  PiRlcChallenge.Sampler.Refinement.TailRows.satisfyingRows_refine
    (tailBitStarts rho) (tailFirstAllocated rho)
    (accepted_tailRows accepted rho)

/-! ## Artifact-derived source-row extent totals -/

def canonicalTranscriptSourceRowExtent : Nat :=
  scalarCount * laneCount * canonicalRowCount

def samplerInitializationSourceRowExtent : Nat := scalarCount

def chunkAcceptanceSourceRowExtent : Nat :=
  scalarCount * candidateCount * 4

def chunkMod5SourceRowExtent : Nat :=
  scalarCount * candidateCount * 20

def chunkSymbolPrefixSourceRowExtent : Nat :=
  scalarCount * candidateCount * 2

def samplerResidualLaneSourceRowExtent : Nat :=
  chunkAcceptanceSourceRowExtent + chunkMod5SourceRowExtent +
    chunkSymbolPrefixSourceRowExtent

def acceptanceBoundSourceRowExtent : Nat := scalarCount * 6

def selectionInitializationSourceRowExtent : Nat := scalarCount

def selectionOneHotSourceRowExtent : Nat :=
  scalarCount * outputCount * 12

def selectionProductsSourceRowExtent : Nat :=
  scalarCount * outputCount * 33

def selectionBindingsSourceRowExtent : Nat :=
  scalarCount * outputCount * 3

def samplerTailSourceRowExtent : Nat :=
  acceptanceBoundSourceRowExtent + selectionInitializationSourceRowExtent +
    selectionOneHotSourceRowExtent + selectionProductsSourceRowExtent +
    selectionBindingsSourceRowExtent

/-- Canonical transcript leaves are intentionally excluded. -/
def samplerSourceRowExtent : Nat :=
  samplerInitializationSourceRowExtent +
    samplerResidualLaneSourceRowExtent + samplerTailSourceRowExtent

/-- Arithmetic over artifact-derived source-row extents. These equalities do
not assert stage-label ownership, disjointness, gap-free coverage, encoded
cost, semantic minimality, or constraint necessity. -/
theorem sourceRowExtentTable :
    canonicalTranscriptSourceRowExtent = 16560 ∧
    samplerInitializationSourceRowExtent = 15 ∧
    chunkAcceptanceSourceRowExtent = 3840 ∧
    chunkMod5SourceRowExtent = 19200 ∧
    chunkSymbolPrefixSourceRowExtent = 1920 ∧
    samplerResidualLaneSourceRowExtent = 24960 ∧
    acceptanceBoundSourceRowExtent = 90 ∧
    selectionInitializationSourceRowExtent = 15 ∧
    selectionOneHotSourceRowExtent = 9720 ∧
    selectionProductsSourceRowExtent = 26730 ∧
    selectionBindingsSourceRowExtent = 2430 ∧
    samplerTailSourceRowExtent = 38985 ∧
    samplerSourceRowExtent = 63960 := by
  decide

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.Rows
