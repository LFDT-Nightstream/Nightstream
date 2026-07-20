import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Core

/-!
Source-closure and slot-coverage check for rewrite rows 750 through 999.

Owns: executable provenance evidence for rewrite chunk three.

Does not own: other chunks, canonical composition, assignment execution, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| rows 750--999 | source terms and derived slots are covered | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk3Check :
    rewriteProvenanceShapeCheck rewriteProvenanceChunk3Data = true := by
  native_decide

theorem rewriteProvenanceChunk3 :
    ∀ step ∈ derivedProgramChunk3,
      StepSourcesKnown step ∧ StepSlotsCovered step :=
  rewriteProvenance_of_shape_check_true chunk3Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
