import Nightstream.SuperNeo.Folding.PiCCS.SplitNc
import tests.Axioms.Support

/-! Fail-closed dependency gate for independent Phi81 SplitNc semantics. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.assignment_freshIndex' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.assignment_freshIndex

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.assignment_runningIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.assignment_runningIndex

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.orderedAssignment_getD' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.orderedAssignment_getD

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe.residualsZero_iff_truth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe.residualsZero_iff_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.truth_iff_orderedAssignments_normBounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.truth_iff_orderedAssignments_normBounded

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.residualsZero_iff_truth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.residualsZero_iff_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.residualsZero_iff_truth' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.residualsZero_iff_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.truth_iff_paperHolds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.truth_iff_paperHolds

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.residualsZero_iff_paperHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.residualsZero_iff_paperHolds

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.firstCompletedTail_outside_columnCube' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.firstCompletedTail_outside_columnCube

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.no_paperColumnLayout_for_carrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.no_paperColumnLayout_for_carrier

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.logicalWidthCube_does_not_cover' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.logicalWidthCube_does_not_cover

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.PublicInput.ofSources_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.PublicInput.ofSources_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain.tensorWeight_eq_equalityWeight' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain.tensorWeight_eq_equalityWeight

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.rowSumcheckDegreeBound_eq_of_terms_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.rowSumcheckDegreeBound_eq_of_terms_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.polynomial_coordinates_eq_qAtPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.polynomial_coordinates_eq_qAtPoint

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.terminalFromMessage_eq_qAtPoint_of_yRingBoundToSources' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.terminalFromMessage_eq_qAtPoint_of_yRingBoundToSources

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.freshConstantTable_eq_completedMatrixImageTable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.freshConstantTable_eq_completedMatrixImageTable

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.sourceYRingAt_fresh_constant_eq_completedMatrixImage' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.sourceYRingAt_fresh_constant_eq_completedMatrixImage

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.sourceYRingAt_running_eq_computedCoefficient' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.sourceYRingAt_running_eq_computedCoefficient

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.claimedYRing_eq_sourceYRingAt_of_carriedTruth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.claimedYRing_eq_sourceYRingAt_of_carriedTruth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Point.decode_coordinates' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Point.decode_coordinates

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.SourceProjection.sourceValueAt_toCubePoint_eq_embed_paddedDiagonal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.SourceProjection.sourceValueAt_toCubePoint_eq_embed_paddedDiagonal

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.SourceProjection.rangeValueAt_toCubePoint_eq_embed_cubicResidual' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.SourceProjection.rangeValueAt_toCubePoint_eq_embed_cubicResidual
