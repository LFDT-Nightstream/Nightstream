import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Core

/-!
Artifact check for derived-program rows 750 through 999.

Owns: executable well-formedness evidence for derived-program chunk three.

Does not own: other chunks, program composition, honest semantics, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| rows 750--999 | predecessors are known and derived outputs are fresh | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk3Check :
    derivedShapeWellFormedCheck derivedProgramKnown3
      derivedProgramShapeChunk3 = true := by
  native_decide

theorem derivedProgramChunk3WellFormed :
    DerivedWellFormed derivedProgramKnown3 derivedProgramChunk3 :=
  derivedWellFormed_of_shape_check_true _ _ chunk3Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
