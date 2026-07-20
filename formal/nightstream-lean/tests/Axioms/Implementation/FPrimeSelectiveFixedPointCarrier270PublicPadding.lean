import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270
import tests.Axioms.Support

/-! Fail-closed kernel dependency report for fixed-point public padding. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPaddingRefinement.generated_raw_row_refines' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PublicPaddingRefinement.generated_raw_row_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPaddingRefinement.withPublicPaddingZero_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PublicPaddingRefinement.withPublicPaddingZero_satisfies

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPaddingRefinement.generatedRowsSatisfied_iff_fixedPublicPadding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PublicPaddingRefinement.generatedRowsSatisfied_iff_fixedPublicPadding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPaddingRefinement.generatedRowsSatisfied_of_typedAssignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PublicPaddingRefinement.generatedRowsSatisfied_of_typedAssignment
