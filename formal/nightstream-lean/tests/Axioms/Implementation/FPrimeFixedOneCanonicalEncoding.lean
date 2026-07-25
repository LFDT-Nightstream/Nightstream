import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncodingRealization
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.HonestAssignment
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSoundness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepPhysicalCompleteness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalSoundness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalPhysicalCompleteness
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guards for the obligation-10 canonical fixed-one
step and terminal encoding theorems.
-/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncoding.Step.obligation10_of_certificate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncoding.Step.obligation10_of_certificate

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncoding.Terminal.obligation10_of_certificate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncoding.Terminal.obligation10_of_certificate

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncodingRealization.stepObligation10' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncodingRealization.stepObligation10

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncodingRealization.terminalObligation10' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncodingRealization.terminalObligation10

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan.activeSound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan.activeSound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan.activeComplete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan.activeComplete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan.inactiveComplete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitivePlan.inactiveComplete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.HonestAssignment.exists_encodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.HonestAssignment.exists_encodes

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSoundness.physicalSound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSoundness.physicalSound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSoundness.physicalSoundAligned' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSoundness.physicalSoundAligned

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepCompleteness.physicalComplete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepCompleteness.physicalComplete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalSoundness.physicalSound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalSoundness.physicalSound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalCompleteness.physicalComplete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalCompleteness.physicalComplete
