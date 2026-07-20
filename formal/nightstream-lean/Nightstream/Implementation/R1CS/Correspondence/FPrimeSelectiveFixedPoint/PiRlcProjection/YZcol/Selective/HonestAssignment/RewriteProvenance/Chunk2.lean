import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Core

/-!
Source-closure and slot-coverage check for rewrite rows 500 through 749.

Owns: executable provenance evidence for rewrite chunk two.

Does not own: other chunks, canonical composition, assignment execution, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| rows 500--749 | source terms and derived slots are covered | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk2Check :
    rewriteProvenanceShapeCheck rewriteProvenanceChunk2Data = true := by
  native_decide

theorem rewriteProvenanceChunk2 :
    ∀ step ∈ derivedProgramChunk2,
      StepSourcesKnown step ∧ StepSlotsCovered step :=
  rewriteProvenance_of_shape_check_true chunk2Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
