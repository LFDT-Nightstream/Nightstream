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

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.firstCompletedTail_outside_columnCube' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.firstCompletedTail_outside_columnCube

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.logicalWidthCube_does_not_cover' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage.logicalWidthCube_does_not_cover
