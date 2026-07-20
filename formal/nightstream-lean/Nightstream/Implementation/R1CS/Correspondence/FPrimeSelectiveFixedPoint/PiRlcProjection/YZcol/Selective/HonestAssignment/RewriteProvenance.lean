import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Chunk0
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Chunk1
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Chunk2
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Chunk3
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Chunk4

/-!
Composes the memory-bounded source-closure and derived-slot checks over the
exact canonical selective rewrite stream.

Owns: canonical composition of source-closure and derived-slot provenance
across the five checked rewrite chunks.

Does not own: chunk computations, honest source semantics, assignment
execution, selected-row satisfaction, or projection authority.

Emits constraints: no.

| Composition leaf | Mathematical obligation | Authority class |
|---|---|---|
| canonical rewrite stream | every source and derived slot has checked provenance | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

theorem rewriteProvenanceKnown :
    ∀ step ∈ decodedRewriteSteps,
      StepSourcesKnown step ∧ StepSlotsCovered step := by
  intro step member
  rw [← derivedProgramChunksExact] at member
  simp only [List.mem_append] at member
  rcases member with member | member | member | member | member
  · exact rewriteProvenanceChunk0 step member
  · exact rewriteProvenanceChunk1 step member
  · exact rewriteProvenanceChunk2 step member
  · exact rewriteProvenanceChunk3 step member
  · exact rewriteProvenanceChunk4 step member

/-- Artifact fact: every source linear combination read by a compact
recurrence is inside the checked compiler closure reconstructed from retained
words. -/
theorem rewriteSourcesKnown :
    ∀ step ∈ decodedRewriteSteps, StepSourcesKnown step := by
  intro step member
  exact (rewriteProvenanceKnown step member).1

/-- Every derived output and predecessor names a slot in the exact checked
derived-column registry. -/
theorem rewriteDerivedSlotsCovered :
    ∀ step ∈ decodedRewriteSteps,
      (match step.output with
        | .source _ => True
        | .derivedProductSum slot => slot ∈ decodedDerivedSlots) ∧
      (match step.previous with
        | none => True
        | some slot => slot ∈ decodedDerivedSlots) := by
  intro step member
  exact (rewriteProvenanceKnown step member).2

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
