import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Chunk0
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Chunk1
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Chunk2
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Chunk3
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Chunk4

/-!
Composes the five independently checked derived-program partitions into the
exact 1,250-row rewrite stream. The partitions are a memory-bounded checking
boundary only; the exported theorem is over the canonical decoded program.

Owns: composition of the five checked partitions into the canonical
derived-program well-formedness theorem.

Does not own: partition checks, honest source semantics, row satisfaction,
projection authority, or security reduction.

Emits constraints: no.

| Composition leaf | Mathematical obligation | Authority class |
|---|---|---|
| canonical program | all five partitions form one well-founded rewrite stream | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

/-- Artifact-grounded SSA fact for the intermediate fields. Every previous
reference is earlier and every derived output is assigned once. -/
theorem derivedProgramWellFormed :
    DerivedWellFormed []
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.decodedRewriteSteps := by
  have valid34 :
      DerivedWellFormed derivedProgramKnown3
        (derivedProgramChunk3 ++ derivedProgramChunk4) :=
    derivedWellFormed_append derivedProgramChunk3WellFormed
      derivedProgramChunk4WellFormed
  have valid234 :
      DerivedWellFormed derivedProgramKnown2
        (derivedProgramChunk2 ++
          (derivedProgramChunk3 ++ derivedProgramChunk4)) :=
    derivedWellFormed_append derivedProgramChunk2WellFormed valid34
  have valid1234 :
      DerivedWellFormed derivedProgramKnown1
        (derivedProgramChunk1 ++
          (derivedProgramChunk2 ++
            (derivedProgramChunk3 ++ derivedProgramChunk4))) :=
    derivedWellFormed_append derivedProgramChunk1WellFormed valid234
  have valid01234 :
      DerivedWellFormed derivedProgramKnown0
        (derivedProgramChunk0 ++
          (derivedProgramChunk1 ++
            (derivedProgramChunk2 ++
              (derivedProgramChunk3 ++ derivedProgramChunk4)))) :=
    derivedWellFormed_append derivedProgramChunk0WellFormed valid1234
  rw [derivedProgramKnown0, derivedProgramChunksExact] at valid01234
  exact valid01234

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
