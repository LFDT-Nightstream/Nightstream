import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical block×lane NC semantics. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.decode_carrierColumn' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.decode_carrierColumn

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.carrierColumn_decode' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.carrierColumn_decode

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.residualsZero_of_truth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.residualsZero_of_truth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.truth_of_residualsZero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.truth_of_residualsZero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.residualsZero_iff_truth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.residualsZero_iff_truth
