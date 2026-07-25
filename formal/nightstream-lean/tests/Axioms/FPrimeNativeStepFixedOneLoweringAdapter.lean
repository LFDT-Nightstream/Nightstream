import tests.FPrimeNativeStepFixedOneLoweringAdapter
import tests.Axioms.Support

/-!
Fail-closed guards for the fixed-one native-to-lowering semantic adapter.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.step' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.step

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.hashPrior' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.hashPrior

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.hashNext' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.hashNext

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.nifsVerify' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.nifsVerify

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.runningCheck' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.runningCheck

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.freshCheck' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.CallAlignment.freshCheck

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.stepAccepts_iff_directHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.stepAccepts_iff_directHolds

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.terminalAccepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter.terminalAccepts_iff_transition
