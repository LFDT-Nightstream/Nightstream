import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Core

/-!
Artifact check for derived-program rows 250 through 499.

Owns: executable well-formedness evidence for derived-program chunk one.

Does not own: other chunks, program composition, honest semantics, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| rows 250--499 | predecessors are known and derived outputs are fresh | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk1Check :
    derivedShapeWellFormedCheck derivedProgramKnown1
      derivedProgramShapeChunk1 = true := by
  native_decide

theorem derivedProgramChunk1WellFormed :
    DerivedWellFormed derivedProgramKnown1 derivedProgramChunk1 :=
  derivedWellFormed_of_shape_check_true _ _ chunk1Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
