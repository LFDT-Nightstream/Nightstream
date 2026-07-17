import Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-!
Public semantic surface for the production-shaped Split-NC statement.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.paper.statement` | Section 7.3 obligations are verifier-independent | semantic statement hidden under executable acceptance |
| `nifs.pi_ccs.semantic.split_exact` | FE/NC truth is exactly the paper-ordered statement | obligation omission or duplication |
| `nifs.pi_ccs.semantic.residuals` | uncompressed residuals are sound and complete | compression treated as semantic authority |
-/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

#check SemanticShape.carrierWidth
#check FlatNcDomain.Covers
#check BlockNcDomain.Covers
#check Sources.Data.assignment_freshIndex
#check Sources.Data.assignment_runningIndex
#check Sources.Data.orderedAssignment_getD
#check Semantics.Fe.residualsZero_iff_truth
#check Semantics.Nc.truth_iff_orderedAssignments_normBounded
#check Semantics.Nc.residualsZero_of_truth
#check Semantics.Nc.truth_of_residualsZero
#check Semantics.Nc.residualsZero_iff_truth
#check Semantics.truth_iff_paperHolds
#check Semantics.residualsZero_iff_truth
#check Semantics.residualsZero_iff_paperHolds
#check Verifier.Polynomial.Nc.Point.decode_coordinates
#check Verifier.Polynomial.Nc.SourceProjection.paddedDiagonal_live
#check Verifier.Polynomial.Nc.SourceProjection.sourceValueAt_toCubePoint_eq_embed_paddedDiagonal
#check Verifier.Polynomial.Nc.SourceProjection.rangeValueAt_toCubePoint_eq_embed_cubicResidual
#check Verifier.Polynomial.Nc.SourceProjection.booleanResidualsZero_of_truth
#check Verifier.Polynomial.Nc.SourceProjection.truth_of_booleanResidualsZero
#check Verifier.Polynomial.Nc.SourceProjection.booleanResidualsZero_iff_truth
