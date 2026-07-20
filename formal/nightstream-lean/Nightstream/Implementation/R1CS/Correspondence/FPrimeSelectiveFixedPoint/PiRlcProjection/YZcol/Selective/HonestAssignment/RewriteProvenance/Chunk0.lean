import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Core

/-!
Source-closure and slot-coverage check for rewrite rows 0 through 249.

Owns: executable provenance evidence for rewrite chunk zero.

Does not own: other chunks, canonical composition, assignment execution, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| rows 0--249 | source terms and derived slots are covered | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk0Check :
    rewriteProvenanceShapeCheck rewriteProvenanceChunk0Data = true := by
  native_decide

theorem rewriteProvenanceChunk0 :
    ∀ step ∈ derivedProgramChunk0,
      StepSourcesKnown step ∧ StepSlotsCovered step :=
  rewriteProvenance_of_shape_check_true chunk0Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
