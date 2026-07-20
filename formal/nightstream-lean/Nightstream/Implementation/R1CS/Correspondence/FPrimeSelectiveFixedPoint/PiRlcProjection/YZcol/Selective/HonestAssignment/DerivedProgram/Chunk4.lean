import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Core

/-!
Artifact check for derived-program rows 1000 through 1249.

Owns: executable well-formedness evidence for derived-program chunk four.

Does not own: other chunks, program composition, honest semantics, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| rows 1000--1249 | predecessors are known and derived outputs are fresh | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk4Check :
    derivedShapeWellFormedCheck derivedProgramKnown4
      derivedProgramShapeChunk4 = true := by
  native_decide

theorem derivedProgramChunk4WellFormed :
    DerivedWellFormed derivedProgramKnown4 derivedProgramChunk4 :=
  derivedWellFormed_of_shape_check_true _ _ chunk4Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
