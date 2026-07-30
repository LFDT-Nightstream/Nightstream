import Nightstream.Implementation.R1CS.Canonical.StepSelectionBoundary
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the step selection boundary.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalStepSelectionBoundary

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.StepSelectionBoundary.holdingMachine_step' does not depend on any axioms -/
#guard_msgs in
#audit_axioms StepSelectionBoundary.holdingMachine_step

/-- info: 'Nightstream.Implementation.R1CS.Canonical.StepSelectionBoundary.flippingMachine_step' does not depend on any axioms -/
#guard_msgs in
#audit_axioms StepSelectionBoundary.flippingMachine_step

/-- info: 'Nightstream.Implementation.R1CS.Canonical.StepSelectionBoundary.step_is_a_real_choice' does not depend on any axioms -/
#guard_msgs in
#audit_axioms StepSelectionBoundary.step_is_a_real_choice

/-- info: 'Nightstream.Implementation.R1CS.Canonical.StepSelectionBoundary.step_selection_is_kernel_checked' does not depend on any axioms -/
#guard_msgs in
#audit_axioms StepSelectionBoundary.step_selection_is_kernel_checked

end NightstreamTests.Axioms.CanonicalStepSelectionBoundary
