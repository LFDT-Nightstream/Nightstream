import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Core

/-!
Artifact check for derived-program rows 0 through 249.

Owns: executable well-formedness evidence for derived-program chunk zero.

Does not own: other chunks, program composition, honest semantics, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| rows 0--249 | predecessors are known and derived outputs are fresh | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk0Check :
    derivedShapeWellFormedCheck derivedProgramKnown0
      derivedProgramShapeChunk0 = true := by
  native_decide

theorem derivedProgramChunk0WellFormed :
    DerivedWellFormed derivedProgramKnown0 derivedProgramChunk0 :=
  derivedWellFormed_of_shape_check_true _ _ chunk0Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
