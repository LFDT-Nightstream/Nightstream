import tests.FPrimeNativeStepCanonicalPublicInputLinkProgram
import tests.Axioms.Support

/-!
Fail-closed guards for the Rust-emitted native public-link program refinement.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgram.plain_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgram.plain_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgram.run_plain_eq_sourceCheck' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgram.run_plain_eq_sourceCheck

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement.generated_plain_eq_canonical' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement.generated_plain_eq_canonical

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement.generated_plain_cost' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement.generated_plain_cost

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement.generated_run_eq_sourceCheck' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement.generated_run_eq_sourceCheck

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement.generated_run_reduces_to_logicalPaperLink' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement.generated_run_reduces_to_logicalPaperLink
