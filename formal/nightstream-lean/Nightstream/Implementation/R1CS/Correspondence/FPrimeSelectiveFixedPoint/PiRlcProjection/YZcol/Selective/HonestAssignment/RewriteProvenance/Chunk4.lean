import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Core

/-!
Source-closure and slot-coverage check for the residual rewrite rows.

Owns: executable provenance evidence for the final rewrite chunk.

Does not own: other chunks, canonical composition, assignment execution, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| residual rows | source terms and derived slots are covered | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk4Check :
    rewriteProvenanceShapeCheck rewriteProvenanceChunk4Data = true := by
  native_decide

theorem rewriteProvenanceChunk4 :
    ∀ step ∈ derivedProgramChunk4,
      StepSourcesKnown step ∧ StepSlotsCovered step :=
  rewriteProvenance_of_shape_check_true chunk4Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
