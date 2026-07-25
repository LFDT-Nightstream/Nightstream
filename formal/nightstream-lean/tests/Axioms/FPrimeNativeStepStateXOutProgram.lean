import tests.FPrimeNativeStepStateXOutProgram
import tests.Axioms.Support

/-!
Fail-closed guards for the Rust-emitted XOut preimage programs.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.execute_forPreimage' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.execute_forPreimage

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.statelessPlain_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.statelessPlain_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.statelessNebula_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.statelessNebula_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.statefulPlain_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.statefulPlain_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.statefulNebula_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram.statefulNebula_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_eq_canonical' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_eq_canonical

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_starts_with_exact_domain' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_starts_with_exact_domain

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_statelessPlain_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_statelessPlain_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_statelessNebula_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_statelessNebula_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_statefulPlain_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_statefulPlain_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_statefulNebula_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_statefulNebula_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_execute_eq_encodeStateXOutPreimage' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_execute_eq_encodeStateXOutPreimage

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_publicLink_accepts_computedXOut' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement.generated_publicLink_accepts_computedXOut
