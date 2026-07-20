import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance.Core

/-!
Source-closure and slot-coverage check for rewrite rows 250 through 499.

Owns: executable provenance evidence for rewrite chunk one.

Does not own: other chunks, canonical composition, assignment execution, or row satisfaction.

Emits constraints: no.

| Chunk leaf | Mathematical obligation | Authority class |
|---|---|---|
| rows 250--499 | source terms and derived slots are covered | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

set_option maxRecDepth 100000 in
private theorem chunk1Check :
    rewriteProvenanceShapeCheck rewriteProvenanceChunk1Data = true := by
  native_decide

theorem rewriteProvenanceChunk1 :
    ∀ step ∈ derivedProgramChunk1,
      StepSourcesKnown step ∧ StepSlotsCovered step :=
  rewriteProvenance_of_shape_check_true chunk1Check

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
