import tests.FPrimeNativeStepPaperFreshLinkBoundary
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guard for the native fresh-link paper-boundary
countermodel.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary.currentInterface_admits_nonFactorizingFreshLink' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary.currentInterface_admits_nonFactorizingFreshLink
